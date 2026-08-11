# 底盘导航与语音控制完整实施方案

本文档给出 `tube_grabber` 下一阶段接入底盘导航和语音控制的完整方案。它基于本机器人上
已经安装的 Woosh ROS2 Humble 接口、旧 `Grabber` 项目的语音代码以及当前试管抓放主线
整理而成。

本文档是设计、实施和验收说明，不代表导航和语音已经在当前分支中实现。当前代码仍只允许
同一个试管架内部搬运；跨架命令会被安全拒绝。不要仅根据本文中的 ROS2 示例就带管运行，
必须按后文阶段逐项完成空载、无管和带管验收。

## 1. 最终目标

系统有三个底盘站点：

- `HOME`：机器人待机位置；
- `RACK_1`：底盘停靠后，右臂能够观察和操作 `rack_1`；
- `RACK_2`：底盘停靠后，右臂能够观察和操作 `rack_2`。

最终接收一条标准搬运命令：

```text
rack_1.r1c1 -> rack_2.r2c6
```

完整行为应为：

1. 自动判断当前底盘所在站点，不要求操作者手动把机械臂移到观察位；
2. 如果当前不在源架站点，先收拢机械臂，再自动导航到源架；
3. 到站并确认底盘稳定后，机械臂自动进入该架的观察位；
4. 多帧识别架面、试管盖和槽位状态，验证源槽有管、目标槽为空；
5. 抓起试管并确认抓取成功；
6. 同架搬运时不移动底盘，直接完成目标高位复检、放置和最终复扫；
7. 跨架搬运时先将机械臂收至带管运输位，再导航到目标架；
8. 在目标架重新识别并基于最新三维坐标放置，不复用源架处的旧坐标；
9. 最终复扫确认源槽为空、目标槽有管、其他槽位没有异常变化；
10. 默认留在任务结束站点，减少无意义移动；只有任务明确要求或配置开启时才返回
    `HOME`。

推荐默认策略是：

```yaml
return_home_after_success: false
```

这样两个架面站点本身就是下一条任务的有效起点。若下一条任务的源架与当前站点相同，底盘
仍然不移动；若不同，则空载前往新的源架。

## 2. 当前已经具备什么

当前分支已经具备：

- `rack_1.r1c1` 这类标准地址和确定性的命令校验；
- 同架任务的视觉预览、执行前复扫、抓取、带管高位复检、放置和最终复扫；
- 正式抓放时自动把右臂移动到旧 AprilTag 流程验证过的全局观察位；
- 机械臂、两指夹爪、D435、CUDA、模型、标定和运动参数的失败关闭门禁；
- `local` Agent 和可选的 Gemini 文本解析器；
- `navigation` 和 `speech` 的模块边界预留。

当前尚未具备：

- Woosh 底盘适配器；
- `HOME/RACK_1/RACK_2` 三个站点的真实 `mark_no` 配置；
- 空载运输位和带管运输位；
- 跨架任务协调器及其中断恢复；
- 麦克风采集、唤醒、VAD、ASR 和播报的运行时代码；
- 导航、语音和抓放的真机联合验收。

因此当前版本的“自动进入观察位”只指机械臂自动进入观察位，前提仍是底盘已经停在该试管
架的可操作区域。导航接入后，开始阶段应变成“底盘自动去源架站点，再由机械臂自动进观察
位”，全程不要求人工拖动机械臂。

## 3. 本机核对到的底盘接口

本机安装了 ROS2 Humble 和以下 Woosh 接口包：

- `woosh_robot_agent`；
- `woosh_robot_msgs`；
- `woosh_task_msgs`；
- `woosh_ros_msgs`。

旧 ROS2 工作区还包含 `ros2_agv_robot` 示例。最终实现可以参考消息字段和状态订阅方式，
但不应直接把示例当成生产控制器。

### 3.1 启动 Woosh 连接节点

旧示例给出的启动方式是：

```bash
source /opt/ros/humble/setup.bash
ros2 run woosh_robot_agent agent \
  --ros-args -r __ns:=/woosh_robot -p ip:="169.254.128.2"
```

也可以不显式提供 IP，让厂商节点使用其默认配置：

```bash
ros2 run woosh_robot_agent agent --ros-args -r __ns:=/woosh_robot
```

实际部署必须确认底盘 IP、网络接口、ROS Domain、控制模式和厂商节点版本。不要在正式服务中
同时启动两个 `woosh_robot_agent` 实例。

### 3.2 命名站点导航动作

推荐使用：

```text
/woosh_robot/robot/ExecTask
woosh_robot_msgs/action/ExecTask
```

目标中的关键字段是：

```text
arg.task_id       唯一任务 ID
arg.type.value    任务类型；厂商到点示例使用 1，即 K_PICK
arg.direction     方向；到点示例使用 0
arg.task_type_no  组合类型；到点示例使用 0
arg.mark_no       地图中预先创建的目标点编号
```

厂商示例命令是：

```bash
ros2 action send_goal /woosh_robot/robot/ExecTask \
  woosh_robot_msgs/action/ExecTask \
  "{arg: {type: {value: 1}, mark_no: REPLACE_WITH_MARK_NO}}" \
  --feedback
```

这个命令只能在空载、机械臂收拢、现场清空且有人监护时用于单步验证。生产代码应使用
`rclpy.action.ActionClient`，不能通过解析 `ros2` 命令行输出控制任务。

`ExecTask` 的反馈和结果都是 `TaskProc`，需要关注：

- `robot_task_id`：底盘侧任务编号；
- `dest`：实际目的地；
- `state.value`：任务状态；
- `msg`：状态或错误说明；
- `action.type.value`、`action.state.value`：当前动作及状态。

任务状态常量为：

| 值 | 名称 | 处理方式 |
|---:|---|---|
| 0 | `K_STATE_UNDEFINED` | 不可作为成功 |
| 1 | `K_INIT` | 初始化中 |
| 2 | `K_READY` | 已准备 |
| 3 | `K_EXECUTING` | 执行中 |
| 4 | `K_PAUSED` | 暂停，等待恢复或超时处理 |
| 5 | `K_ACTION_WAIT` | 动作等待，不可误判为完成 |
| 6 | `K_TASK_WAIT` | 任务等待，不可误判为完成 |
| 7 | `K_COMPLETED` | 只满足成功条件之一 |
| 8 | `K_CANCELED` | 任务取消 |
| 9 | `K_FAILED` | 任务失败 |

到站成功不能只看其中一个状态。至少同时要求：

1. ROS2 action 的 `WrappedResult.code == SUCCEEDED`；
2. `TaskProc.state.value == K_COMPLETED`；
3. `TaskProc.dest` 与请求的 `mark_no` 一致；
4. 最新底盘状态没有急停、故障或定位丢失；
5. 线速度和角速度连续多帧低于静止阈值；
6. 当前地图 ID/版本仍与配置一致；
7. 到站后的架面视觉验证通过。

### 3.3 必须订阅的状态

以下状态建议使用 `KeepLast(1) + transient_local`，与本机示例保持一致。消息通常没有标准
ROS Header，因此适配器收到每条消息时必须额外记录本机单调时钟；超过新鲜度门限的旧消息
不能通过门禁。

| 接口 | 类型 | 用途 |
|---|---|---|
| `/woosh_robot/robot/RobotState` | `woosh_robot_msgs/msg/RobotState` | 空闲、任务、警告、故障等总状态 |
| `/woosh_robot/robot/Mode` | `woosh_robot_msgs/msg/Mode` | 自动/手动/维护及工作模式 |
| `/woosh_robot/robot/PoseSpeed` | `woosh_robot_msgs/msg/PoseSpeed` | 地图位姿、线速度、角速度、地图 ID |
| `/woosh_robot/robot/Battery` | `woosh_robot_msgs/msg/Battery` | 电量和充电状态 |
| `/woosh_robot/robot/Scene` | `woosh_robot_msgs/msg/Scene` | 场景、地图名、地图 ID 和版本 |
| `/woosh_robot/robot/TaskProc` | `woosh_robot_msgs/msg/TaskProc` | 当前任务目的地和进度 |
| `/woosh_robot/robot/DeviceState` | `woosh_robot_msgs/msg/DeviceState` | 急停和定位等位状态 |
| `/woosh_robot/robot/OperationState` | `woosh_robot_msgs/msg/OperationState` | 可接任务、障碍等运行状态 |

启动时还应调用一次：

```text
/woosh_robot/robot/RobotInfo
woosh_robot_msgs/srv/RobotInfo
```

它提供完整初始快照，后续再由订阅更新。这样既不会在尚未收到状态时误发任务，也不会只依赖
启动时的一份过期快照。

关键位定义如下：

- `OperationState.robot & K_TASKABLE`：底盘可接任务；发目标前必须为真；
- `OperationState.nav & K_IMPEDE`：遇到障碍；应显示“等待障碍清除”，不可直接当成到站；
- `DeviceState.hardware & K_EMG_BTN`：急停触发；立即禁止新任务；
- `DeviceState.software & K_LOCATION`：厂商示例解释为定位准确；导航前后都必须为真。

机器人总状态：

- `K_IDLE = 2`：空闲；
- `K_TASK = 4`：任务中；
- `K_WARNING = 5`：警告；
- `K_FAULT = 6`：故障；
- 其他状态包括未初始化、泊车、跟随、充电和构图。

控制模式：

- `K_AUTO = 1`；
- `K_MANUAL = 2`；
- `K_MAINTAIN = 3`。

工作模式有部署、任务和调度三种。最终允许哪一种要在本机器人空载测试时确认并写入配置，
不能仅凭常量名称猜测。正式运行至少要求控制模式为自动模式。

### 3.4 不应直接复用的旧逻辑

本机存在两类旧底盘逻辑，它们都只能作为参考：

1. `ros2_agv_robot/goto_mark.cpp`：把字符串转发给 `ExecTask`，但收到请求时没有真正使用
   已计算的 `is_taskable` 阻止发送，结果通道也只发布“任务完成”，没有完整表达拒绝、取消、
   失败、超时和错误原因；
2. 旧 `Grabber/controllers/ugv_controller.py`：根据速度和时间估算距离，用内部的 `0.0～1.0`
   变量假装当前位置，没有地图定位、障碍绕行和实际到站闭环。

最终任务不能使用速度×时间开环移动。`/woosh_robot/ros/StepControl` 也只适合现场标定或受控
微调，不应用于 `HOME/RACK_1/RACK_2` 之间的常规导航。

## 4. 三个站点如何建立

### 4.1 在 Woosh 地图中创建点

先用厂商部署界面完成地图、定位和站点创建。需要三个真实命名点，其 `mark_no` 可能是随机
字符串，不一定就是 `HOME`。应用层名称与厂商编号分开保存，例如：

```yaml
stations:
  home:
    mark_no: "REPLACE_WITH_HOME_MARK_NO"
    rack_id: null
  rack_1:
    mark_no: "REPLACE_WITH_RACK_1_MARK_NO"
    rack_id: "rack_1"
  rack_2:
    mark_no: "REPLACE_WITH_RACK_2_MARK_NO"
    rack_id: "rack_2"
```

占位值必须让配置校验失败，不能把示例里的 `A1` 或 `7613B5D` 当成本机器的真实点号。

每个架面站点不仅有平面位置，还必须保存正确的底盘朝向。停靠方向决定右臂工作空间和腕部
相机视野，差 180° 即使“到达同一点”也完全不可用。

### 4.2 物理布置建议

推荐为两个试管架建立相同的机械布置：

- 相同高度和朝向；
- 相同的架体定位治具；
- 架面相对底盘停靠位的横向、纵向距离尽量一致；
- 试管架不能只靠桌面摩擦定位；
- 机器人重复停靠后，完整架面始终位于腕部相机视野和右臂工作空间内。

如果两处物理布局足够一致，可以共用一个 `observation_pose`。如果两处高度、方向或相对位置
不同，必须改为每架独立：

```yaml
racks:
  rack_1:
    observation_pose: ...
  rack_2:
    observation_pose: ...
```

当前代码只有一个全局观察位。因此跨架实现前必须先决定“两处统一治具”还是“每架独立观察
位”，不能在代码里暗中假设两者相同。

### 4.3 站点标定和重复性验收

每个站点按以下顺序建立：

1. 清空机械臂运动范围并把双臂收进底盘轮廓；
2. 在 Woosh 部署界面创建站点和朝向；
3. 记录 `mark_no`、地图 ID、地图名和版本；
4. 空载从不同方向驶入该点；
5. 每次到站记录 `PoseSpeed.pose` 和稳定时间；
6. 到架面站点后让右臂自动进入观察位，执行完整多帧扫描；
7. 记录是否完整看到 8 个关键点、红点、所有槽位以及是否通过三维稳定门禁；
8. 连续重复至少 30 次，统计位置、角度、到站时间和视觉成功率；
9. 以最差可接受到站样本验证机械臂工作空间和碰撞净空；
10. 固定站点后不要随意移动试管架、桌子或治具。

导航精度的最终指标不应只写成一个地图位置误差。真正的验收条件是：重复停靠后，机械臂能
安全到达观察位，架面始终在视野内，视觉可以重新计算准确的三维槽位，而且所有规划点仍在
工作空间和净空内。

## 5. 导航适配器的标准边界

`navigation` 层只负责“站点到站”，不读取相机、不移动机械臂、不操作夹爪。建议暴露如下
概念接口：

```text
status() -> NavigationStatus
resolve_station() -> Station | UNKNOWN
navigate_to(station, timeout_s) -> NavigationResult
cancel_active(reason) -> NavigationResult
wait_until_stationary(timeout_s) -> NavigationStatus
close()
```

`NavigationStatus` 至少包含：

```text
monotonic_timestamp
robot_state
control_mode
work_mode
map_id / map_name / map_version
x_m / y_m / yaw_rad
linear_mps / angular_radps
battery_percent
taskable
impeded
emergency_stop
localization_ok
active_task_id / destination / task_state
```

错误必须是可区分的结构化类型，例如：

- `NavigationUnavailable`：ROS2 action/service 不存在；
- `NavigationStateStale`：状态超过新鲜度门限；
- `NavigationNotTaskable`：底盘不可接任务；
- `NavigationEmergencyStop`：急停；
- `NavigationLocalizationLost`：定位失效；
- `NavigationGoalRejected`：目标被服务器拒绝；
- `NavigationTimeout`：执行超时且已请求取消；
- `NavigationCanceled`：已取消；
- `NavigationFailed`：底盘报告失败；
- `NavigationArrivalMismatch`：完成结果不是请求的站点；
- `NavigationNotSettled`：到站后仍在运动；
- `NavigationVisualDockFailure`：到架位后视觉无法确认可操作条件。

应用层只依赖这个抽象接口。测试使用 `FakeNavigator`，真实运行使用 `WooshNavigator`，不要
让任务协调器直接导入 Woosh 消息类型。

### 5.1 推荐的进程结构

当前 Python 解释器可以找到 `rclpy`，但 CUDA/PyTorch、Ultralytics、RealSense 和 ROS2 的
依赖最终是否能在同一个虚拟环境稳定共存仍需部署测试。推荐先尝试同进程：

```text
TaskSupervisor
 ├─ vision/manipulation（主任务线程）
 └─ WooshNavigator（rclpy executor 专用线程）
```

如果 ROS2 与 CUDA 虚拟环境发生 ABI 或依赖冲突，再拆成两个本机进程：

```text
视觉/机械臂主进程 <-> 严格类型的本机 IPC <-> ROS2 导航桥进程
```

IPC 应使用 Unix domain socket 或仅监听回环地址的服务，携带请求 ID、超时、幂等键和完整
状态，不能用无反馈的 shell 命令，也不能开放一个“任意速度控制”网络接口。

### 5.2 发送一次导航的严格顺序

`navigate_to()` 应按以下顺序执行：

1. 验证站点存在且 `mark_no` 不是占位符；
2. 获取 `RobotInfo` 快照并等待所有必要 topic 有新鲜数据；
3. 确认地图 ID/名称/版本与站点配置一致；
4. 确认自动模式、允许的工作模式、无急停、无故障、定位有效；
5. 确认 `K_TASKABLE`、电量高于门限、当前没有另一个任务；
6. 确认任务协调器已经取得全局运动锁；
7. 确认双臂均处于经真机验收的运输位；
8. 如果持管，额外确认夹爪状态和带管运输状态可信；
9. 若当前位置已在目标站点容差内且视觉/状态确认一致，幂等返回 `ALREADY_THERE`；
10. 生成不重复的 64 位 `task_id`，持久化任务意图；
11. 发送 `ExecTask`，等待 goal response；
12. goal 被拒绝时立即失败，不盲目重发；
13. 持续处理 feedback、`TaskProc` 和状态 topic；
14. 遇到障碍时进入 `WAITING_FOR_OBSTACLE`，播报状态但保持总超时；
15. 急停、定位丢失、故障或通信状态过期时请求取消，并进入安全锁定；
16. 超时时请求 action cancel，等待有限时间确认取消；
17. 收到结果后同时验证 ROS result code、任务状态和目的地；
18. 连续若干采样验证线速度和角速度低于门限；
19. 再次验证地图、定位、急停和故障状态；
20. 到架面站点时，交回协调器做观察位移动和视觉停靠验证。

不要把 `K_IMPEDE` 一出现就自动取消并重发。正常避障可能短时间设置该位；重复发目标容易生成
多个任务。应显示当前状态，在明确超时、故障或人工取消时只取消当前 goal。

### 5.3 建议的初始配置

下面数值只是首轮低速调试起点，必须用实测结果调整：

```yaml
navigation:
  enabled: false
  provider: woosh
  agent_ip: "169.254.128.2"
  namespace: "/woosh_robot"
  expected_map_name: "REPLACE_AFTER_MAPPING"
  expected_map_id: null
  expected_map_version: null

  action_server_timeout_s: 20.0
  state_freshness_s: 1.0
  navigation_timeout_s: 180.0
  cancel_timeout_s: 5.0
  obstacle_notice_after_s: 3.0

  stationary_linear_mps: 0.01
  stationary_angular_radps: 0.02
  stationary_samples: 10
  station_position_tolerance_m: 0.08
  station_yaw_tolerance_deg: 3.0

  minimum_start_battery_percent: 30
  return_home_after_success: false

  stations:
    home:
      mark_no: "REPLACE_WITH_HOME_MARK_NO"
      rack_id: null
    rack_1:
      mark_no: "REPLACE_WITH_RACK_1_MARK_NO"
      rack_id: "rack_1"
    rack_2:
      mark_no: "REPLACE_WITH_RACK_2_MARK_NO"
      rack_id: "rack_2"
```

`enabled` 在所有站点、运输位和失败恢复完成验收前必须保持 `false`。位置容差只用于判断是否
已经在站点附近，不代替 Woosh 任务结果和架面视觉验证。

## 6. 机械臂观察位与底盘运输位

需要明确区分四类机械臂位姿：

1. `observation_pose`：空载时拍摄整个架面的观察位；
2. `target_high_corridor`：同架携管时在目标槽上方复检的动态高位；
3. `transport_pose_empty`：底盘空载导航时双臂收拢位；
4. `transport_pose_loaded`：右夹爪持管时的底盘运输位。

当前已有的全局观察位不是底盘运输位。底盘运动时必须满足：

- 双臂和相机完全位于验证过的底盘扫掠包络内；
- 试管与夹爪均不超出容易撞击桌面、门框或人的区域；
- 持管姿态不让试管滑出，也不让液体因过度倾斜外溢；
- 底盘加减速和转弯时夹爪仍有足够保持力；
- 电缆不会被拉扯；
- 到站并完全静止后机械臂才离开运输位。

带管状态禁止回到当前较低的全局观察位。如果腕部相机在带管运输位看不到目标架，应新增经
验收的 `loaded_observation_pose`，或改用固定/头部相机完成目标架粗确认，然后只在安全高位
让腕部相机做精定位。不能为了看清目标架让试管穿过邻管高度。

两条底盘运动硬门禁应写成不可绕过的条件：

```text
base may move only if arm_motion == IDLE
arm may leave transport pose only if base is stationary and navigation is IDLE
```

任何时候都不允许底盘和机械臂同时运动。

## 7. 完整任务状态机

建议增加唯一的 `TaskSupervisor`，让文字、语音和未来的界面请求都进入同一个入口。它持有
一把全局任务锁，并驱动以下状态：

```text
IDLE
  -> PARSE_AND_VALIDATE
  -> PRECHECK
  -> STOW_EMPTY
  -> NAVIGATE_TO_SOURCE（已在源架则跳过）
  -> SETTLE_AT_SOURCE
  -> MOVE_TO_SOURCE_OBSERVATION
  -> OBSERVE_AND_PREVIEW
  -> CONFIRM_TASK
  -> RESCAN_SOURCE
  -> PICK
  -> VERIFY_PICK
  -> SAME_RACK_TARGET_HIGH（同架）
       或 STOW_LOADED -> NAVIGATE_TO_DESTINATION -> SETTLE_AT_DESTINATION
  -> OBSERVE_DESTINATION_SAFELY
  -> RESCAN_AND_REPLAN_DESTINATION
  -> PLACE
  -> RETREAT
  -> MOVE_TO_FINAL_OBSERVATION
  -> VERIFY_FINAL_SCENE
  -> STOW_EMPTY
  -> OPTIONAL_RETURN_HOME
  -> COMPLETE
```

### 7.1 从 HOME 开始

以 `rack_1.r1c1 -> rack_2.r2c6` 为例：

1. 启动时通过实时地图位姿和站点容差确认当前确实是 `HOME`；
2. 双臂自动进入 `transport_pose_empty`；
3. 自动导航到 `RACK_1`；
4. 底盘静止后，右臂自动进入 `rack_1` 观察位；
5. 识别、确认、复扫并抓取；
6. 右臂自动进入 `transport_pose_loaded`；
7. 导航到 `RACK_2`；
8. 通过带管安全观察位复扫 `rack_2`；
9. 使用新的目标槽三维坐标完成放置；
10. 最终复扫后收臂；
11. 默认停在 `RACK_2`，不返回 `HOME`。

这里没有“人工把机械臂放到观察位”这一步。保留的人工或语音确认只是在系统真正开始抓放前
核对任务和现场，不是让人手动摆机械臂。

### 7.2 同架任务

以 `rack_1.r1c1 -> rack_1.r2c6` 为例：

- 当前在 `RACK_1`：底盘全程不动；
- 当前在 `HOME` 或 `RACK_2`：空载自动导航到 `RACK_1`，之后底盘不动；
- 抓起后沿当前已有的目标高位走廊复检并放置；
- 完成后默认仍停在 `RACK_1`。

### 7.3 跨架任务

当前 `ManipulationWorkflow` 是一次性同架抓放流程，并明确拒绝不同 `rack_id`。跨架接入时
不能简单删除这个检查；应先把内部能力拆成有状态、可验证的阶段：

```text
prepare_pick(source, destination_intent)
execute_pick() -> HeldTubeEvidence
prepare_place(destination, held_tube_evidence)
execute_place() -> PlacementEvidence
verify_transfer() -> TransferEvidence
```

`HeldTubeEvidence` 至少记录源架、源槽、抓取前后观测、夹爪回读和时间。只有协调器持有这份
有效证据时才能发起带管导航。到目标架后必须重新观测，不允许把源架坐标或任务开始时的目标
坐标用于放置。

### 7.4 是否返回 HOME

默认不返回，原因是：

- 减少一次底盘运动和风险；
- 两个架位都可以作为下一条任务起点；
- 减少任务周期；
- 避免带管失败时错误回家。

允许返回 HOME 的条件应同时满足：

- 最终视觉确认任务完成；
- 夹爪为空且状态可信；
- 双臂都在空载运输位；
- 底盘状态健康、可接任务且电量足够；
- 用户命令或配置明确要求返回。

若系统认为仍持有试管、最终复扫失败或夹爪状态不确定，绝对不能自动返回 HOME。

## 8. 站点身份、架面身份和坐标

`rack_id` 的主身份来源应是“当前底盘站点映射”，而不是从两个外观相同的架面中猜编号：

```text
RACK_1 station -> rack_1 calibration
RACK_2 station -> rack_2 calibration
```

红点只定义 `k0` 方向，不用于判断这是 `rack_1` 还是 `rack_2`。如果存在把两个架子对调的
可能，建议为治具增加独立且固定的架号标识，例如不同 AprilTag/二维码，并把身份核对加入到
站点到达验证。不能把红点颜色同时承担方向和架号两个含义。

底盘移动后，机械臂 `base_right` 坐标系随机器人一起移动。目标架所有槽位的三维坐标必须在
目标站重新扫描和计算。禁止把一个站点得到的 `base_right` 三维点跨底盘运动后继续使用。

## 9. 失败、中断和恢复策略

### 9.1 通用原则

- 任一时刻只能有一个搬运任务；
- 每次底盘或机械臂动作都有唯一请求 ID 和有限超时；
- 不确定等同于失败，不能用上一次状态猜测；
- 语音、文字、按钮都只向协调器请求，不直接抢占硬件；
- 物理急停始终高于软件停止和语音停止。

### 9.2 按是否持管分类

未持管失败：

- 停止当前动作；
- 收集底盘、机械臂、夹爪和视觉状态；
- 确认可以安全收臂后才允许回运输位；
- 不自动重新抓取，除非失败类型被明确列入一次性重试白名单。

持管失败：

- 将全局状态锁定为 `HOLDING_UNRESOLVED`；
- 禁止新任务和自动返回 HOME；
- 保持夹爪，不得因异常清理自动张开；
- 如果底盘仍在运动，先请求取消并等待实际静止；
- 播报当前站点、源槽、目标槽和失败原因；
- 由操作人员根据现场选择继续放置、送回源架或人工接管。

### 9.3 障碍、超时和掉线

- `K_IMPEDE`：先等待并提示，不立即重复发送目标；
- action 超时：请求取消一次，等待取消确认，未确认则进入锁定；
- ROS2 状态过期：不得发送新动作；若正在导航，尝试取消并视为位置未知；
- 定位丢失：禁止机械臂离开运输位；
- 急停：停止软件状态机，但不要尝试由软件复位物理急停；
- 到错站：保持收臂，禁止观察和抓放；
- 视觉停靠验证失败：允许在无管状态下按配置重试一次导航；持管时不盲目重试。

## 10. 状态持久化

跨架任务会在较长时间内持有试管，进程崩溃或断电后不能只靠内存恢复。协调器应把最小状态
原子写入本地文件：

```json
{
  "schema_version": 1,
  "task_id": "...",
  "phase": "NAVIGATE_TO_DESTINATION",
  "source": "rack_1.r1c1",
  "destination": "rack_2.r2c6",
  "holding_tube": true,
  "last_confirmed_station": "rack_1",
  "active_navigation_task_id": 123456,
  "updated_at": "..."
}
```

写入应采用同目录临时文件、`fsync` 和原子替换，文件权限限制为当前服务账户。启动时若记录
显示 `holding_tube: true` 或任务阶段不完整，应进入恢复锁定并要求现场核对，不能直接清空
状态或开始下一条语音命令。

“上一次记得在 RACK_1”只能作为提示。真实当前站点仍要用最新地图位姿、任务状态和视觉重新
确认。

## 11. 语音方案：可以复用什么

旧 `Grabber` 中的语音链路是：

```text
PyAudio 输入
  -> sherpa-onnx KeywordSpotter（唤醒词“小浦”）
  -> Silero VAD
  -> sherpa-onnx SenseVoice int8 离线识别
  -> Agent
  -> 科大讯飞 WebSocket TTS
  -> sounddevice 播放
```

其中以下思路可以保留：

- 睡眠、播报、录音、执行四类状态；
- KWS 持续低功耗监听；
- VAD 自动截断一句话；
- SenseVoice 在本地做中文识别；
- 播报期间停止把扬声器声音送入识别，避免自唤醒。

但旧代码不能原样复制，原因包括：

- 音频设备未按稳定名称选择，直接使用默认输入；
- KWS/VAD/ASR 模型路径写死为相对路径，而旧模型目录当前没有可用模型；
- 当前项目运行环境尚未安装 `pyaudio`、`sounddevice` 和 `sherpa_onnx`；
- 录音阶段没有完整的“没有开始说话”超时和一句话总时长限制；
- 状态由多个线程直接写入，没有统一事件队列和严格并发保护；
- 云端 TTS 依赖网络，旧实现还关闭了 TLS 证书校验；
- 旧项目存在硬编码的云端凭据，必须视为已经暴露，立即作废轮换，绝不能复制到新仓库；
- Agent 回调与硬件之间缺少本项目现有的统一任务门禁。

## 12. 本机麦克风现状

本次静态核对在 `/proc/asound` 中看到了两个 USB 摄像头音频采集接口：

- `Generic USB Camera`；
- `WSD usb camera`。

它们都提供 USB Audio capture，其中单声道硬件模式是 `S16_LE / 48 kHz`。这说明相机设备
带有可枚举的音频输入，但不能据此认定机器人有独立、可用于远场识别的内置麦克风。本次运行
会话没有完成 ALSA/PulseAudio 实际录音，最终必须用运行服务的同一个 Linux 用户做录音、
回放、权限和噪声测试。

推荐优先级：

1. 在机器人计算机上接一个固定的 USB 麦克风或麦克风阵列；
2. 近距离原型测试可尝试现有 USB 摄像头麦克风；
3. 外接电脑采音也可以，但应只作为语音前端，把结构化命令发送给机器人任务协调器。

设备选择不能写死为 `card 2` 或 PyAudio 的数字 index，因为重启和插拔会改变编号。应通过
ALSA 长名称、USB VID/PID/序列号或 udev 创建稳定别名。

现有 USB 相机没有原生 16 kHz 单声道模式。若使用它，应以 48 kHz 单声道采集，再用高质量
重采样转换到 16 kHz 供 KWS/VAD/ASR 使用，不能把 48 kHz 样本直接谎报成 16 kHz。

### 12.1 机器人本机麦克风方案

这是推荐方案：

- 音频不离开机器人；
- 唤醒、VAD、ASR 和命令解析可以离线运行；
- 网络中断不影响基本控制；
- 延迟最低，部署关系最简单。

若安装扬声器，应让麦克风与扬声器保持物理距离，并测试回声。播报时暂停普通 ASR；如果需要
播报期间也能说“停止”，应使用独立的高优先级停止词检测器或硬件按钮，而不是重新打开完整
自由文本识别。

### 12.2 外接电脑方案

外接电脑可以录音、运行 ASR，然后向机器人发送标准结构化请求，例如：

```json
{
  "request_id": "uuid",
  "source": "rack_1.r1c1",
  "destination": "rack_2.r2c6",
  "return_home": false
}
```

外接电脑不能直接访问机械臂 SDK、底盘速度接口或夹爪接口。机器人本机仍要执行地址校验、
状态检查、二次确认、视觉复扫和所有运动门禁。

连接应限制在受控局域网，使用身份认证、重放保护和超时。请求 ID 必须幂等：同一个请求因为
网络重发时不能执行两次。外接电脑断线只影响交互，不应使机器人继续一个未确认的新任务。

## 13. 推荐语音流水线

```text
音频采集
  -> 输入设备健康检查
  -> 重采样/单声道归一化
  -> KWS 唤醒
  -> “我在”提示音或短播报
  -> VAD 采集一句话
  -> 本地 ASR
  -> 文本清洗和槽位同音词归一化
  -> 确定性命令解析
  -> 严格 TransferCommand 校验
  -> 语音复述并等待确认
  -> TaskSupervisor.submit()
  -> 状态播报
```

视觉已经要求 CUDA:0。为了避免 ASR 与 YOLO 抢 GPU 导致视觉延迟抖动，首版建议沿用轻量
int8 KWS/VAD/ASR 并在 CPU 上运行，视觉独占本地 CUDA。只有在 Jetson 上分别测完语音延迟、
GPU 显存和完整视觉扫描 P95 后，才考虑把 ASR 切到 GPU。

播报首版推荐使用本地离线 TTS，或者把“我在、请确认、任务开始、遇到障碍、任务完成、任务
失败”等固定安全提示制作成随包音频。云端 TTS 只能作为非关键文本的可选功能，网络和证书
错误不能阻塞硬件状态机。

### 13.1 语音状态机

建议状态如下：

```text
DISABLED
  -> SLEEPING
  -> WAKE_ACK
  -> LISTENING
  -> RECOGNIZING
  -> PARSING
  -> AWAITING_CONFIRMATION
  -> SUBMITTING
  -> MONITORING_TASK
  -> SPEAKING_RESULT
  -> SLEEPING
```

另有可从多数状态进入的：

- `CANCELING`；
- `DEVICE_ERROR`；
- `RECOVERY_LOCKED`。

所有状态变化通过一个线程安全事件队列完成，音频回调只放入音频块，不直接调用机械臂或底盘。

### 13.2 命令格式与中文示例

系统内部永远使用：

```text
rack_1.r1c1 -> rack_2.r2c6
```

可以接受的中文表达包括：

```text
把一号架第一排第一个试管移到二号架第二排第六个
一号架 r1c1 放到二号架 r2c6
将 rack_1 的第一行第一列搬到 rack_2 的第二行第六列
```

解析器要把“一号/1号/rack one/第一架”等有限同义词归一化，但必须拒绝：

- 缺少源架或目标架；
- 只说“第一个”而没有明确行列；
- 行不是 1～2 或列不是 1～6；
- 源和目标相同；
- 一句话解析出多个可能地址；
- ASR 置信度或规则匹配不够；
- 当前已有任务正在执行。

建议首版用确定性语法解析。Gemini 可用于把更自然的句子转换为候选结构化命令，但输出仍要
经过同一套 `TransferCommand` 校验；模型不能获得直接运动工具。

### 13.3 二次确认

识别成功后必须复述完整源和目标：

```text
请确认：把一号架第一排第一列的试管，移动到二号架第二排第六列。说“确认执行”或“取消”。
```

只有在有限时间内识别到明确的“确认执行”才提交任务。“嗯、好、可以吧”等模糊短语不作为
确认。任务确认应绑定 `request_id` 和刚才复述的标准命令，超时后失效。

这一步可以替代键盘输入确认，但不能替代首次真机调试时的现场监护和物理急停。是否在量产
模式取消当前 `EMPTY/MOVE` 键盘门禁，应等语音协调器和所有真机验收完成后再做一个明确配置，
不能让两个确认入口同时抢占标准输入。

### 13.4 停止和取消

“停止任务”是软件取消请求，不等于安全认证急停：

- 导航中：请求取消当前 Woosh action，并等待实际速度归零；
- 机械臂运动中：调用统一运动执行器的停止路径，并保持夹爪当前状态；
- 尚未抓管：终止任务并进入可恢复状态；
- 已抓管：进入 `HOLDING_UNRESOLVED`，不得自动张开夹爪或回 HOME。

在噪声环境中不建议让任意未唤醒语音直接控制运动。可以为“停止”设计独立高阈值关键词，
但物理急停仍是唯一应被操作者依赖的紧急保护。

### 13.5 播报内容

播报应简短、确定、可操作：

- “已识别任务，等待确认”；
- “正在前往一号架”；
- “底盘遇到障碍，正在等待”；
- “正在识别一号架，请勿移动试管架”；
- “已抓取，正在前往二号架”；
- “任务完成，机器人停在二号架”；
- “任务失败，夹爪可能仍持有试管，请检查现场”。

不要朗读底层堆栈或长错误码。屏幕/日志保存完整诊断，语音只报告阶段、是否持管和下一步。

## 14. 推荐代码结构

保持当前模块边界，建议新增：

```text
tube_grabber/
  navigation/
    models.py             # Station、NavigationStatus、NavigationResult
    port.py               # Navigator 协议
    station_registry.py   # rack_id <-> station <-> mark_no
    fake.py               # 离线状态机测试
    woosh_ros2.py         # 唯一导入 rclpy/woosh 消息的实现

  speech/
    models.py             # VoiceEvent、VoiceState、Transcript
    audio.py              # 设备枚举、采集、重采样和健康检查
    kws.py                 # 唤醒词
    vad.py                 # 语音段检测
    asr.py                 # 本地识别
    parser.py              # 有限中文语法 -> TransferCommand
    tts.py                 # 离线播报或固定提示音
    service.py             # 语音状态机和事件队列

  coordinator/
    task_supervisor.py     # 导航与抓放唯一协调器
    state_store.py         # 原子持久化和崩溃恢复锁
```

配置建议拆成：

```text
config/navigation.yaml
config/speech.yaml
config/transport_poses.yaml
```

关键依赖方向：

```text
speech -> TransferCommand -> TaskSupervisor
navigation -> NavigationResult -> TaskSupervisor
workflow -> manipulation evidence -> TaskSupervisor
TaskSupervisor -> ports
hardware/vision/navigation/speech 之间不互相直接调用
```

这样文本 CLI、语音和以后网页界面都只是不同输入适配器，不会产生三套不同的安全逻辑。

## 15. 语音配置建议

```yaml
speech:
  enabled: false
  input_device_match: "REPLACE_WITH_STABLE_ALSA_NAME"
  capture_rate_hz: 48000
  model_rate_hz: 16000
  channels: 1
  chunk_ms: 100

  kws:
    provider: sherpa_onnx
    keyword: "小浦"
    model_dir: "models/speech/kws"
    score: 1.5
    threshold: 0.25

  vad:
    model_path: "models/speech/silero_vad.onnx"
    threshold: 0.5
    wait_for_speech_timeout_s: 5.0
    end_silence_s: 0.6
    maximum_utterance_s: 12.0

  asr:
    provider: sherpa_onnx_sense_voice
    model_dir: "models/speech/sense_voice_int8"
    threads: 2
    device: cpu

  confirmation:
    timeout_s: 8.0
    accepted_phrase: "确认执行"
    cancel_phrases: ["取消", "取消任务"]

  tts:
    provider: offline
    output_device_match: "REPLACE_WITH_STABLE_OUTPUT_NAME"
```

模型权重与视觉权重一样默认不提交到 Git。启动时应检查路径、采样率、设备、依赖和一次短录音，
任何一项失败就保持语音 `DISABLED`，不能影响文字 CLI 和底层安全。

所有云端凭据只从环境变量或受控密钥服务读取。日志不得记录 URL 签名、API key、原始音频或
包含敏感信息的完整请求，除非用户明确开启诊断采集。

## 16. 实施顺序

### 阶段 0：固定当前抓放基线

- 完成 `rack_pose.pt` 训练；
- 完成两个架的双圆心标定；
- 完成相机、手眼、TCP、夹爪和同架抓放验收；
- 记录同架成功率和所有视觉抖动指标。

导航不能掩盖尚未稳定的架面识别。当前基线未通过时，不进入带管底盘测试。

### 阶段 1：只读底盘监控

- 启动 `woosh_robot_agent`；
- 实现 `RobotInfo` 和八类状态订阅；
- 只显示状态，不发送任何运动；
- 验证自动模式、工作模式、急停、定位、障碍、任务和速度位的真实含义；
- 验证消息新鲜度和断线检测。

### 阶段 2：建立和验证站点

- 创建 HOME、RACK_1、RACK_2；
- 写入真实 `mark_no` 和地图信息；
- 机械臂收拢，空载单次导航；
- 验证拒绝、取消、障碍、超时和到错站处理；
- 每站完成至少 30 次重复停靠统计。

### 阶段 3：协调器和 Fake 闭环

- 增加 `FakeNavigator`；
- 拆分 pick/place 阶段；
- 实现同架跳过导航、跨架导航和可选回 HOME；
- 实现任务锁、状态持久化和崩溃恢复；
- 全部使用 fake 硬件验证状态机。

### 阶段 4：空夹爪联合运动

- 验证空载运输位；
- HOME -> RACK_1 -> RACK_2 -> HOME；
- 每次到站自动进入对应观察位并扫描；
- 确认底盘运动和机械臂运动绝不重叠；
- 注入导航失败并验证机械臂保持收拢。

### 阶段 5：同架正式任务

- 从 HOME 自动导航到源架；
- 自动进入观察位；
- 完成同架搬运；
- 默认停在当前架位；
- 证明不需要人工摆放机械臂。

### 阶段 6：带管跨架

- 先用空的、安全替代管验证运输姿态；
- 低速验证抓起、收至带管运输位、跨架、目标复扫和放置；
- 测试刹停、转弯、障碍等待后的夹持可靠性；
- 测试目标视觉失败、导航取消和进程重启时的持管锁定；
- 最后才使用真实试管和真实内容物。

### 阶段 7：语音离线接入

- 固定麦克风和扬声器设备；
- 安装模型和依赖；
- 只做唤醒、识别和命令回显，不执行动作；
- 验证数字、架号、行列和确认词；
- 接入 Fake supervisor；
- 最后接入真实 supervisor，语音层不获得任何硬件对象。

## 17. 测试与验收清单

### 17.1 导航单元测试

- 未配置站点拒绝；
- 状态缺失或过期拒绝；
- 手动/维护模式拒绝；
- 急停、故障、定位失败和低电拒绝；
- 非 `K_TASKABLE` 拒绝；
- goal rejected、aborted、canceled、failed 分别映射；
- `K_COMPLETED` 但目的地不符拒绝；
- action 成功但速度未归零拒绝；
- 已在目标站幂等跳过；
- 取消超时进入锁定；
- 同一个请求 ID 不重复执行。

### 17.2 协调器单元测试

- HOME 到同架源站只导航一次；
- 当前就在源站时不导航；
- 同架抓起后底盘不动；
- 跨架抓起后必须先到带管运输位；
- 未取得抓取证据时不能带管导航；
- 到目标架必须重新扫描和规划；
- 最终复扫失败不能标记成功；
- 持管失败禁止回 HOME；
- `return_home=false` 留在结束站；
- `return_home=true` 只有空夹爪且收臂后执行；
- 进程重启读到未完成持管状态时锁定。

### 17.3 语音单元测试

- 唤醒前普通谈话不提交命令；
- 播报不会再次唤醒自己；
- 无人说话和超长说话都超时；
- 一号/1号、第一行/r1 等归一化正确；
- 模糊地址拒绝；
- ASR 空文本拒绝；
- 只有“确认执行”提交当前请求；
- 确认超时后旧确认无效；
- 重复网络请求只执行一次；
- 任务运行中第二条命令被拒绝并播报忙碌；
- “取消任务”走统一取消，不直接张开夹爪。

### 17.4 真机指标

每次验收保存原始时间戳和结果，至少统计 P50、P95 和最差值：

- 每个站点从多个方向到达至少 30 次；
- action 成功率、超时率、障碍恢复率；
- 到站后的线速度、角速度和稳定等待时间；
- 到站地图位姿离散；
- 自动进入观察位的成功率；
- 到站后架面 8 点和红点检测成功率；
- 槽位三维离散和完整扫描耗时；
- 空载跨站循环至少 30 次；
- 替代管带管跨站至少 30 次；
- 正式试管跨站成功率；
- 安静、风扇、底盘运动背景和多人说话下的 ASR 地址准确率；
- 每小时误唤醒次数；
- 从说完命令到复述确认的延迟；
- 完整视觉推理期间语音服务对 CUDA 帧率和显存的影响。

“流畅”应由这些数据定义，不能只凭单次演示观感。视觉继续固定使用本地 CUDA，语音首版在
CPU 运行，分别记录资源占用后再优化。

## 18. 现场诊断命令

以下命令都是接入前的只读检查：

```bash
source /opt/ros/humble/setup.bash
ros2 service type /woosh_robot/robot/RobotInfo
ros2 action type /woosh_robot/robot/ExecTask
ros2 topic type /woosh_robot/robot/OperationState
ros2 topic type /woosh_robot/robot/DeviceState
ros2 topic type /woosh_robot/robot/PoseSpeed
```

查看接口定义：

```bash
ros2 interface show woosh_robot_msgs/action/ExecTask
ros2 interface show woosh_robot_msgs/msg/TaskProc
ros2 interface show woosh_robot_msgs/msg/OperationState
ros2 interface show woosh_robot_msgs/msg/DeviceState
ros2 interface show woosh_robot_msgs/msg/PoseSpeed
```

查看实时状态：

```bash
ros2 service call /woosh_robot/robot/RobotInfo woosh_robot_msgs/srv/RobotInfo
ros2 topic echo /woosh_robot/robot/RobotState
ros2 topic echo /woosh_robot/robot/Mode
ros2 topic echo /woosh_robot/robot/OperationState
ros2 topic echo /woosh_robot/robot/DeviceState
ros2 topic echo /woosh_robot/robot/PoseSpeed
ros2 topic echo /woosh_robot/robot/TaskProc
```

音频硬件检查应在运行语音服务的同一个用户下执行：

```bash
cat /proc/asound/cards
cat /proc/asound/pcm
arecord -L
arecord -D REPLACE_WITH_DEVICE -f S16_LE -r 48000 -c 1 -d 5 /tmp/mic-test.wav
aplay /tmp/mic-test.wav
```

不要把测试音频或真实语音提交到 Git。测试完成后可删除 `/tmp/mic-test.wav`。

## 19. 正式上线前必须填写的实机参数

- [ ] Woosh agent 的真实 IP 和启动方式；
- [ ] 允许的控制模式与工作模式；
- [ ] HOME、RACK_1、RACK_2 的真实 `mark_no`；
- [ ] 地图名、地图 ID 和版本策略；
- [ ] 三个站点的位姿与容差；
- [ ] 状态新鲜度、导航超时和取消超时；
- [ ] 最低任务电量；
- [ ] 空载运输位；
- [ ] 带管运输位；
- [ ] 两个架是否共用观察位；若不共用，记录各自观察位；
- [ ] 带管目标观察方案；
- [ ] 底盘与双臂互锁；
- [ ] 持管失败恢复流程；
- [ ] 是否默认返回 HOME；推荐否；
- [ ] 麦克风稳定设备名、采样率和权限；
- [ ] 扬声器稳定设备名和回声测试；
- [ ] KWS、VAD、ASR、TTS 模型路径和许可证；
- [ ] 所有旧云端凭据已作废轮换；
- [ ] 语音二次确认和取消词；
- [ ] 导航、视觉、机械臂、夹爪和语音的真机统计均达标。

## 20. 最终推荐结论

底盘使用 Woosh 地图中的命名站点和 `ExecTask` action，任务完成后再通过任务结果、目的地、
实时速度、定位、地图和架面视觉共同确认到站。不要使用旧项目按时间估算距离的 CAN 开环移动，
也不要把 `goto_mark/result` 的一句“任务完成”当作完整闭环。

任务由一个协调器统一管理：开始时自动收臂、导航到源架并自动进入观察位；同架不移动底盘；
跨架抓起后进入带管运输位再导航；到目标架必须重新识别和规划；任务成功后默认留在当前架位。

语音可以做，而且优先使用机器人本机 USB 麦克风、离线 KWS/VAD/SenseVoice 和本地播报。
旧 Grabber 的状态机思路可以参考，但依赖和模型需要重新安装，硬编码云端凭据必须作废，语音
只能生成经过复述确认的标准 `TransferCommand`，绝不能绕过统一协调器直接控制任何硬件。
