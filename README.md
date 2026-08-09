# Tube Grabber

面向 2×6 试管架的视觉引导抓放系统，作为 SURF 项目的正式代码主线维护。系统使用腕部 Intel RealSense D435 获取 RGB-D 图像，以一个两类别 YOLO 模型识别空孔和试管盖，以红/绿 HSV 角点确定机架身份及行列方向，再由右侧 RealMan 七轴机械臂和 RM Plus 两指夹爪完成竖直抓放。

本仓库不是教学 Demo，但“正式主线”也不等于“已经真机验收”。代码结构、假硬件闭环和自动化测试已经完成；最终模型、现场标定、设备依赖和真实机械动作仍需按实验室清单逐项验证。

## 项目状态

截至 2026-08-10，状态明确划分如下：

| 状态 | 内容 |
|---|---|
| 已实现 | 2×6 槽位数据模型；D435、RealMan 右臂和 RM Plus 驱动边界；YOLO + HSV K0 + 深度 + 手眼变换；同架抓放 workflow；纯坐标运动规划；工作空间、姿态、到位和执行前复扫门禁；fake 后端；local/Gemini 文本 Agent；统一 CLI |
| 已离线验证 | 当前 Python 3.13.15 虚拟环境运行 **80 项测试全部通过**，覆盖数据校验、Agent 不可信输出、假硬件端到端流程、模拟 SDK 返回、RealSense Brown–Conrady 畸变处理、视觉几何、网格编号、运动规划、安全拒绝路径和 workflow；默认 fake 模式的 `doctor`、规划和搬运闭环可运行 |
| 待真机验证 | 仓库目前没有 `models/tube_slots.pt`；当前环境未接入 D435、RealMan SDK 和真实机械臂；观测位与运动参数仍锁定；两个机架的架面 Z 尚未填写；TCP 偏移、最终管长和盖顶高度含临时值；手眼矩阵需要在当前安装状态复核；Gemini 真实 API、模型可用性和网络未作为本次离线测试的一部分 |
| 未来功能 | 麦克风语音识别、移动底盘导航、夹管跨站移动、跨架协调与恢复、自动回观测位、动作后自动复扫、动态避障和端到端 VLA 均未实现 |

因此，当前可交付的是一套经过离线测试、具备真机适配和安全门禁的正规工程代码；不是已取得真实抓放成功率数据的最终实验结果。

## 系统范围

当前正式支持：

- 两个固定物理试管架，每个均为 2 行 × 6 列；
- `rack_1` 使用红色 5 mm × 5 mm K0，`rack_2` 使用绿色 K0；
- 一个 YOLO detection 模型，类别严格为 `0: empty_hole`、`1: tube_cap`；
- 相机在机架正上方观察，一次画面只处理一个完整机架；
- 源槽必须为 `OCCUPIED`，目标槽必须为 `EMPTY`；
- 盖顶下方 5 mm 抓取，目标孔位置由像素射线与已标定架面相交得到；
- 右臂单臂工作，左臂全程收回；
- 底盘停止并锁定时的同一机架内搬运；
- 标准槽位地址和可选 Gemini 自然语言输入。

当前不支持：跨架搬运、底盘导航、语音采集、双臂协同、任意规格机架、任意视角、动态障碍物避让或自动失败恢复。跨架命令会在 CLI 和 workflow 两层被拒绝。

### 关于 VLA 的准确表述

本项目不是一个端到端训练的 VLA 模型。视觉由 YOLO、HSV 和几何模块完成，语言由可选的 Gemini 适配器转换为结构化命令，动作由确定性 Python workflow 和运动规划器执行。它可以描述为“视觉—语言—动作模块化系统”或“Agent 辅助的视觉抓放系统”，但不应宣称已训练或部署端到端 VLA。Gemini 不接收机械臂控制函数，也不直接生成三维坐标或运动轨迹。

## 架构

```text
结构化命令 ───────────────────────────────┐
自然语言 ─→ local / Gemini Agent ─→ TransferCommand
                                           │
D435 RGB-D + 右臂拍摄位姿                  │
        ↓                                  │
YOLO(empty_hole / tube_cap) + HSV K0       │
        ↓                                  │
2×6 网格编号 + 深度 + 手眼变换             │
        ↓                                  │
RackObservation（状态、像素、base_right 坐标）
        └──────────────┬───────────────────┘
                       ↓
Workflow（源/目标状态与同架约束）
                       ↓
MotionPlanner（只接收 Point3D / Pose6D）
                       ↓
MotionExecutor（整计划、工作空间和到位校验）
                       ↓
RealMan 右臂 + RM Plus 夹爪
```

模块边界保持单向：

- `agent` 只把文本转换为一个经过确定性校验的 `TransferCommand`；
- `vision` 只把一帧 RGB-D 和拍摄位姿转换为 `RackObservation`；
- `workflow` 决定状态约束和抓放顺序；
- `motion` 只处理坐标、姿态、速度和数值限值，不读取 YOLO、rack 或 SDK；
- `hardware` 只封装厂商接口，不决定任务目标。

项目内部位置统一为 **mm**、角度统一为 **rad**、工作坐标系为 `base_right`。只有 `RealManArm` 驱动边界负责 mm 与 SDK 米制单位之间的转换。

## 目录结构

```text
tube_grabber_final/
├─ config/
│  ├─ app.yaml             # 唯一应用配置
│  ├─ hand_eye.yaml        # camera_rightwrist → end_right 手眼矩阵
│  └─ poses.yaml           # 观测位与竖直工具姿态
├─ models/                 # 正式权重应放为 tube_slots.pt
├─ training/               # 两类别 YOLO 数据集 YAML 模板
├─ artifacts/              # 相机图、毫米深度图、扫描标注图
├─ docs/                   # 架构、模型、Agent 和实验室文档
├─ tests/                  # 离线单元与集成测试
└─ tube_grabber/
   ├─ agent/               # local/Gemini 文本命令适配
   ├─ core/                # 数据对象、接口、解析和异常
   ├─ hardware/            # D435、RealMan、RM Plus 适配
   ├─ vision/              # YOLO、K0、网格、深度和坐标变换
   ├─ motion/              # 纯坐标规划与运动执行校验
   ├─ workflow/            # 扫描、准备、复扫、抓取和放置
   ├─ fakes/               # 无硬件正式接口实现
   ├─ navigation/          # 仅预留，尚无实现
   ├─ speech/              # 仅预留，尚无实现
   ├─ app.py               # fake/real 运行时组装
   └─ cli.py               # 唯一命令入口
```

## 安装

项目要求 Python ≥ 3.10。离线测试已在 Python 3.13.15 通过；真机部署时还必须选择 RealMan 厂商 SDK 实际支持的 Python 版本，不能只根据本项目的最低版本判断兼容性。

Windows PowerShell：

```powershell
cd F:\my_program\surf_ws\tubeGrabber\tube_grabber_final
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .
```

按功能安装可选依赖：

```powershell
# YOLO
.\.venv\Scripts\python.exe -m pip install -e ".[vision]"

# D435
.\.venv\Scripts\python.exe -m pip install -e ".[realsense]"

# Gemini 文本 Agent
.\.venv\Scripts\python.exe -m pip install -e ".[agent]"

# 三项一起安装
.\.venv\Scripts\python.exe -m pip install -e ".[vision,realsense,agent]"
```

RealMan Python SDK 不由 PyPI 依赖自动安装。真机电脑必须按厂商说明提供 `Robotic_Arm.rm_robot_interface`，并用运行本项目的同一个 Python 验证：

```powershell
.\.venv\Scripts\python.exe -c "from Robotic_Arm import rm_robot_interface; print('RealMan SDK OK')"
```

不要从旧仓库复制来源不明的 SDK 或 API key。

## 配置

默认配置为安全的 `fake` 模式。相对路径均从项目根目录解析，也可在子命令前用 `--config` 指定另一份 YAML：

```powershell
.\.venv\Scripts\python.exe -m tube_grabber --config config/app.yaml doctor
```

主要配置及当前状态：

| 配置 | 含义 | 当前状态 |
|---|---|---|
| `runtime.mode` | `fake` 或 `real` | `fake` |
| `agent.provider` | `local` 或 `gemini` | `local`，无网络 |
| `camera.serial` | 腕部 D435 序列号 | 已记录 `405622073249`，待现场连接确认 |
| `arm.ip/port` | 右臂控制器 | 已记录 `169.254.128.19:8080`，待现场确认 |
| `arm.expected_dof` | 连接机械臂轴数门禁 | 必须为 7 |
| `vision.model_path` | 两类别权重 | `models/tube_slots.pt`，文件当前不存在 |
| `vision.class_names` | 模型类别契约 | 必须严格为 `empty_hole/tube_cap` |
| `racks.*.fallback_plane_z_mm` | 各物理机架在 `base_right` 下的架面 Z | 两者均未标定 |
| `geometry.tcp_offset_end_mm` | 法兰到抓取 TCP | 当前为临时值，必须实测 |
| `geometry.cap_top_above_rack_mm` | 盖顶高出架面的高度 | 当前 47 mm，必须用最终管复测 |
| `geometry.tube_total_length_mm` | 携管净空计算所用总长 | 当前 120 mm 为旧管临时值 |
| `config/hand_eye.yaml` | 相机到右臂法兰变换 | 矩阵已存在，待当前安装状态复核 |
| `poses.yaml: observation_pose.confirmed` | 观测位人工确认锁 | `false` |
| `motion.parameters_confirmed` | TCP、工作空间、姿态和高度总确认锁 | `false` |

切换 `real` 前先完成[实验室分阶段检查](docs/LAB_CHECKLIST.md)。不要为通过 `doctor` 而直接把确认项改成 `true`。

Gemini 使用 `agent.model` 和 `agent.api_key_env`。API key 只允许通过部署环境的 `GEMINI_API_KEY` 提供，不能写入 YAML、Python、命令日志或 Git。`--agent-provider gemini` 只覆盖当前命令，不修改配置文件。

## CLI

安装后既可运行 `tube-grabber`，也可使用下文等价且更明确的 `python -m tube_grabber`。查看完整帮助：

```powershell
.\.venv\Scripts\python.exe -m tube_grabber --help
```

| 命令 | 作用 | 真实设备行为 |
|---|---|---|
| `doctor` | 静态检查配置、依赖、模型、手眼和解锁条件 | 不连接硬件；fake 下的 `WAIT` 不代表真机就绪 |
| `arm-status` | 读取右臂法兰位姿、控制器模式和上电状态 | 连接右臂，不发送运动 |
| `camera-check` | 保存一帧对齐彩色图和毫米深度图 | 仅 `real` 可用；启动 D435，不连接机械臂 |
| `scan --rack …` | 输出一个机架的 12 槽状态与架面 Z | 连接右臂和 D435，只读取、不运动；要求右臂已在确认观测位 |
| `plan-transfer` | 扫描、检查并打印 TCP 和全部法兰航点 | 不初始化夹爪，不发送航点运动 |
| `transfer` | 执行一次同架抓放 | **可能运动**；打印计划、人工 `MOVE`、再检查控制器和现场、复扫后才执行 |
| `agent-plan` | 文本解析后复用 `plan-transfer` | Agent 不运动；real 模式仍会连接右臂和相机完成真实规划 |
| `agent-transfer` | 文本解析后复用 `transfer` | **可能运动**；与普通 transfer 完全相同的门禁 |

槽位地址格式为 `rack_1.r1c1`：rack 只能为 `rack_1/rack_2`，行只能为 1–2，列只能为 1–6；编号先行后列，K0 邻近角对应 `r1c1`。

### 无硬件命令

运行前确认 `runtime.mode: fake`：

```powershell
.\.venv\Scripts\python.exe -m tube_grabber doctor
.\.venv\Scripts\python.exe -m tube_grabber scan --rack rack_1
.\.venv\Scripts\python.exe -m tube_grabber plan-transfer --source rack_1.r1c1 --destination rack_1.r2c6
.\.venv\Scripts\python.exe -m tube_grabber transfer --source rack_1.r1c1 --destination rack_1.r2c6
```

每个 fake 机架在进程启动时都固定为 `r1c1` 有管、其他槽为空；状态不会跨 CLI 进程保存。

### local Agent

默认 `local` Agent 不调用网络，只从任意文本中提取恰好两个标准地址。第一个为源，第二个为目标：

```powershell
.\.venv\Scripts\python.exe -m tube_grabber agent-plan --text "rack_1.r1c1 -> rack_1.r1c2"
```

少于或多于两个地址时只请求补充信息，不生成运动计划。local 模式不理解“一号架第一行”这类自由中文表达。

### Gemini `agent-plan`

先安装 `.[agent]`，通过安全的环境变量管理方式设置 `GEMINI_API_KEY`，再运行只读规划：

```powershell
.\.venv\Scripts\python.exe -m tube_grabber agent-plan --agent-provider gemini --text "把一号架第一行第一列的试管移动到一号架第二行第六列"
```

Gemini 只能提出六个结构化槽位字段。自动函数执行被关闭，Python 会再次检查字段完整性、rack、整数行列、范围和源目标是否相同；不完整或非法输出不会被猜测执行。

### Gemini `agent-transfer`

这是运动入口，只有在 `agent-plan`、真实视觉、完整标定和实验室无运动预演均通过后才可使用：

```powershell
.\.venv\Scripts\python.exe -m tube_grabber agent-transfer --agent-provider gemini --text "把一号架第一行第一列的试管移动到一号架第一行第二列"
```

Gemini 只负责生成 `TransferCommand`；之后立即进入与普通 `transfer` 相同的同架检查、状态检查、整计划校验、`MOVE` 人工授权、执行前控制器复核和场景复扫。Gemini API 失败、无函数调用、多函数调用或参数非法时不会运动。跨架自然语言即使被正确解析，也会被导航锁拒绝。

## 安全边界

本项目不是经过安全认证的机器人控制系统。软件检查不能替代控制器限位、物理急停、低速示教和现场监护。

- 默认 `fake`；真机必须同时确认观测位和运动参数。
- 连接时校验右侧七轴设备；真机运动前检查控制器仍为物理模式且已上电。
- 程序不会自动移动到观测位，操作者需用示教器低速到位。
- 视觉必须恰好得到 12 个约定类别；K0 缺失、歧义或位置异常、网格畸变、深度异常、盖顶 Z 离群、架面不一致均拒绝继续。
- 真机搬运要求当前 rack 已填写架面 Z 标定，源有管、目标为空。
- 规划前检查坐标系、工具朝向、管长净空、工作空间和单段距离；机架近场航点使用阻塞式笛卡尔直线运动，每段后回读到位误差。
- 打印计划后右臂发生移动，计划作废。
- 真机输入 `MOVE` 后再次检查控制器、观测位，并复扫 K0、12 槽状态、像素、架面和源/目标坐标；变化超限时拒绝执行。
- 执行异常时软件请求减速停止，但通信失效时仍必须使用物理急停。
- 抓取失败后软件只能保守认为夹爪可能持管；当前没有独立力觉或管存在传感器确认。
- 当前不会自动回观测位或在动作后复扫。终端“运动完成”不等于任务已被视觉验收，必须人工回观测位再运行 `scan`。
- 左臂全程收回，底盘全程停稳锁定；在导航实现前禁止夹着试管移动底盘。
- `agent-transfer` 不会绕过任何安全门，也不因语言模型“很确定”而降低阈值。

首次真机操作必须严格执行[实验室分阶段检查](docs/LAB_CHECKLIST.md)，从 `arm-status`、`camera-check`、重复 `scan` 和 `plan-transfer` 逐级推进。

## 测试

运行全部离线测试：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

本次基线结果：

```text
Python 3.13.15
Ran 80 tests
OK
```

这些测试证明的是软件逻辑和模拟边界，不证明以下事项：

- 新 YOLO 权重的真实 exact-12、中心点精度或抓取成功率；
- D435 在实际反光、照明和安装姿态下的深度质量；
- RealMan SDK 版本、七轴运动学、RM Plus 夹爪和阻塞到位行为在当前 NUC 上一致；
- 手眼、TCP、架面 Z、管长、工作空间和竖直姿态已经物理正确；
- Gemini 线上模型、API key、配额和网络可用；
- 真实机器人完成了任何抓放动作。

代码、模型、相机安装、TCP、手眼矩阵、机架位置或管规格改变后，应重新运行相关离线测试和真机验收阶段。

## 文档

- [架构与维护说明](docs/ARCHITECTURE.md)：模块职责、数据契约、扫描/搬运流程及未来扩展边界。
- [模型与 K0 契约](docs/MODEL_CONTRACT.md)：训练类别、标注、exact-12 指标、K0 和架面标定。
- [Agent 接入说明](docs/AGENT.md)：local/Gemini 配置、手动函数调用和安全约束。
- [实验室分阶段检查](docs/LAB_CHECKLIST.md)：从无运动诊断到首次低速抓放的现场顺序。
- [完整上机与项目路线图](docs/LAB_DEPLOYMENT_AND_ROADMAP.md)：YOLO 训练、真机部署、Gemini/代理、语音硬件检查、分级 TODO 与海报实验设计。
- [模型目录说明](models/README.md)：正式权重路径和类别自检命令。

## 已知限制与后续工作

1. 最终两类别模型尚未训练并放入仓库，当前真实视觉主流程无法启动。
2. 当前配置仍含 TCP、管长、盖顶高度、架面和观测位等待确认参数，真机运动默认锁定。
3. 视觉假设固定俯视、一个完整 2×6 机架、无遮挡且恰好 12 个目标；不会推断缺失槽位，也不支持任意机架。
4. 运动规划是受限工作空间内的竖直抓放航点规划，不是 MoveIt 等完整三维碰撞规划器，不感知动态障碍物。
5. 仅支持同一固定站内搬运；导航、到站静止判断、安全携带位和跨站恢复不存在。
6. Gemini 是不可信命令输入适配器，不是机器人策略模型，也不是端到端 VLA；线上调用尚待单独验证。
7. `speech/` 目前只有接口预留，没有 ASR、唤醒词、麦克风或复读确认。
8. 没有独立抓取成功传感器、自动回观测位、动作后自动复扫和通用故障恢复。

后续实现导航或语音时，应让它们分别输出“已到站且静止”和 `TransferCommand`，继续复用现有视觉、workflow 与纯坐标运动边界，不应把导航或语言模型逻辑写进运动规划器。
