# 实验室分阶段检查清单

这份清单用于第一次把最终主线接到真实设备。每一阶段只增加一种风险；当前阶段失败就停在这里，不跳到后面的运动测试。所有命令都从 `tube_grabber_final` 根目录执行。

## 0. 保留安全锁

第一次到实验室时先保持：

```yaml
runtime:
  mode: fake
motion:
  parameters_confirmed: false
```

并保持 `config/poses.yaml`：

```yaml
observation_pose:
  confirmed: false
```

确认左臂已收回且不会参与、夹爪为空、底盘停稳并锁定、右臂低速、急停可触达、工作空间无人且无松动物体。NUC 上必须停止 `atom/zhixing_ctrl.py` 以及任何会持续发送机械臂命令的旧程序或控制端；本项目不会擅自结束别的进程。不要通过临时扩大 workspace 或关闭检查来“先跑起来”。

## 1. 软件和假硬件闭环

```powershell
python -m unittest discover -s tests -v
python -m tube_grabber doctor
python -m tube_grabber scan --rack rack_1
python -m tube_grabber transfer --source rack_1.r1c1 --destination rack_1.r2c6
```

通过条件：测试无失败；`doctor` 能组装 fake 运行时；假扫描输出两行六列；假搬运完成。`WAIT` 表示真机尚未满足的条件，在 fake 模式中是预期输出，不等于已经具备真机条件。

## 2. 安装真实依赖和模型

```powershell
python -m pip install -e ".[vision,realsense]"
python -c "import pyrealsense2; from ultralytics import YOLO; from Robotic_Arm import rm_robot_interface; print('imports OK')"
python -c "from ultralytics import YOLO; print(YOLO('models/tube_slots.pt').names)"
```

权重必须位于 `models/tube_slots.pt`，类别必须是 `{0: 'empty_hole', 1: 'tube_cap'}`。先按[模型契约](MODEL_CONTRACT.md)完成，不使用旧权重冒充。

## 3. 只连接、不运动

把 `runtime.mode` 改成 `real`，其余两个人工确认锁仍保持 `false`。先检查网络端口：

```powershell
Test-NetConnection 169.254.128.19 -Port 8080
python -m tube_grabber doctor
python -m tube_grabber arm-status
```

`doctor` 此时因未确认位姿/运动参数而返回失败是正常的安全锁。`arm-status` 只能读状态，不能导致右臂运动。记录输出并确认：

- 连接的是右侧七轴机械臂，IP/端口为现场实际值；
- controller mode 显示 `real`，power 状态与控制器一致；
- 位置数量级是 mm、角度数量级是 rad，坐标系是预期的右臂基坐标系；
- RealMan 当前工作坐标系和工具坐标系与手眼标定时一致。
- NUC 进程列表中没有 `zhixing_ctrl.py`，示教器和其他终端也没有正在发送运动。

任一坐标含义不确定时，不开启后续确认锁。

## 4. 只测 D435

```powershell
python -m tube_grabber camera-check
```

确认实际打开序列号 `405622073249` 的腕部 D435，输出 1280×720@30 的对齐彩色/深度图，且有效深度比例合理。检查 `artifacts/` 中：

- 彩色图方向、曝光和完整机架视野正确；
- 深度 PNG 单位是 mm，没有大面积 0、近邻飞点或彩深错位；
- 试管盖、全部空孔和 5 mm × 5 mm K0 都清晰可见。
- 彩色流畸变模型为 `none` 或 D435 常见的 `brown_conrady`；后者会在像素转射线时用 OpenCV 去畸变。若程序报告其他模型或内参与系数自相矛盾，不能绕过检查继续投影。

## 5. 确认观测位和几何量

程序不会自动去观测位。用示教器低速把右臂放到最终垂直俯视位，保持底盘和机架固定，再运行 `arm-status`。把实际读数写入 `config/poses.yaml` 的 `position_mm` 和 `rpy_rad`。

现场逐项核对：

- `hand_eye.yaml` 是当前 D435 到**右臂法兰**的变换，方向为 `camera → flange`，平移单位 mm；
- `tcp_offset_end_mm` 是当前夹爪实际抓取中心相对法兰的偏移；
- `vertical_tool_rpy_rad` 与观察姿态分开，且工具 +Z 相对基坐标 -Z 的倾斜不超过 `maximum_tool_tilt_deg: 1.0`；当前 `[-π, 0, -1.607]` 只是待确认候选；
- `cap_top_above_rack_mm` 是新机架和最终试管完全坐实后盖顶高出架面的实测值；当前 47 mm 只是旧机架记录，必须重测；
- `tube_total_length_mm` 是最终试管盖顶到底端的总长；当前 120 mm 只是旧管记录。`retreat_height_mm` 必须至少等于管长加要求净空（当前 120 + 30 = 150 mm）；
- 抓取深度为 5 mm，夹爪开/合位置 170/135 与当前夹爪一致；
- RM Plus 上电/协议启动等待 1.0 s / 0.8 s 足以稳定初始化；
- approach 50 mm、retreat 150 mm、workspace 和单段最大距离在现场无遮挡且可达。
- 先不迁移旧代码的放置 XY 经验补偿；测量实际放置偏差，只有稳定系统误差存在时才设计显式 `place_xy_correction`。

只有观测位本身已经低速到达并核对无误，才把 `observation_pose.confirmed` 改为 `true`。此时仍保持 `motion.parameters_confirmed: false`。

## 6. 只扫描、不运动

让机械臂停在观测位，分别准备红 K0 的 rack_1 和绿 K0 的 rack_2：

```powershell
python -m tube_grabber scan --rack rack_1
python -m tube_grabber scan --rack rack_2
```

每个机架至少在全空、单管和混合状态下重复多次。逐张检查 `artifacts/scan_*.png` 和终端输出：

- 永远恰好 12 个槽位；
- K0 旁是 `r1c1`，编号先行后列且不会镜像；
- 每个 `EMPTY/OCCUPIED` 与实物一致；
- 标注中心落在真实孔心或盖心；
- 重复扫描的 `plane_z` 和三维位置稳定。
- 多支管的盖顶 Z 彼此一致，不能靠放宽 `maximum_cap_z_deviation_mm` 掩盖飞点；已有标定时，实测架面必须与标定相符。
- 画面中清除其他红/绿物体；若存在多个角外 HSV 候选，程序必须报 K0 歧义而不是猜一个。

使用至少一支完全坐实的试管重复扫描，按[模型契约中的方法](MODEL_CONTRACT.md#架面-z-与-5-mm-抓取深度)给两个 rack 分别写入 `fallback_plane_z_mm`，然后以全空状态复扫。全空仍不能扫描时，不进入运动测试。

## 7. 无运动预演完整计划

确认实际源槽有管、目标槽为空，运行：

```powershell
python -m tube_grabber plan-transfer --source rack_1.r1c1 --destination rack_1.r2c6
```

该命令使用真实视觉并打印源/目标 TCP 与全部法兰航点，但不初始化夹爪、不发送运动。航点应显示 `mode=L`；将打印坐标与现场方向、架面高度和机械臂可达区逐点核对。任何 Z 方向、mm/m 数量级、TCP 偏移、直线轨迹或航点次序不确定，都不要执行 `transfer`。

## 8. 解锁一次低速抓放

所有几何量、workspace、approach/retreat 航点都已在现场确认后，才把：

```yaml
motion:
  parameters_confirmed: true
```

保持 `transit_speed_percent: 10`、`approach_speed_percent: 5` 或更保守的已验证速度。用一支测试管、最近且无遮挡的同架目标执行：

```powershell
python -m tube_grabber transfer --source rack_1.r1c1 --destination rack_1.r1c2
```

程序会再次扫描、打印完整计划，并要求输入精确的 `MOVE`。输入前再次确认左臂收回、底盘锁定、夹爪为空、旧控制进程已停和急停位置。授权后程序会重新读取控制器状态并再次扫描；机械臂、K0、槽位状态、架面或目标坐标发生变化时，旧计划自动作废。每个 LIN 航点完成后还会回读法兰到位误差。全程由一名操作者观察；出现异常立即使用物理急停，不依赖终端。

从这次扫描开始到动作结束，不得再碰机架、试管、底盘或示教移动右臂；打印的计划只对应刚才那一帧和当时的右臂位姿。若现场有任何变化，取消命令并重新扫描。

终端显示“运动完成”不等于任务已经视觉验收；当前版本不会自动回观测位。用示教器低速回到已确认观测位后重新扫描，确认源变为 `EMPTY`、目标变为 `OCCUPIED`，再逐渐测试远列、第二行和另一物理架。建议记录每次源/目标、识别结果、坐标误差和成功/失败原因。

## 9. 当前不能做的测试

不要执行 `rack_1 → rack_2`。CLI 和 workflow 会拒绝跨架，因为导航、到站静止判断、右臂携带位和跨站失败恢复尚未实现。不要删掉这一检查，也不要手工夹住试管后移动底盘来冒充完整流程。

语音同样未接入；当前只接受明确的结构化槽位地址。这两项未来如何接入见[架构说明](ARCHITECTURE.md#未来接入导航和跨架搬运)。

## 建议的真机验收记录

只有完成下面证据链，才把某一版本标记为“真机通过”：

- 完整离线测试输出和当次 Git 提交号；
- `doctor` 在 real 模式全部关键项通过；
- 两个机架、多种占用组合的 exact-12 与槽位状态记录；
- K0 编号、架面 Z、TCP/手眼和航点现场复核记录；
- 同架不同源/目标的多次完整抓放结果；
- 对每次失败保留终端错误与 `artifacts/` 图像。

模型、相机安装、手眼矩阵、TCP、机架位置或夹爪发生变化后，相关阶段必须重做。
