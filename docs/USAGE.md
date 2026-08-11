# Tube Grabber 详细使用说明

本文档说明当前 `agent/publish-final-runtime` 分支从安装、模型准备、标定到真机抓放的
完整使用顺序。除非特别说明，所有命令都在项目根目录执行。

## 1. 当前能力和限制

当前版本支持：

- Intel RealSense D435 彩色图与对齐深度采集；
- YOLO Detection 识别试管盖，并计算盖顶三维坐标；
- YOLO Pose 识别架面 8 个关键点；
- `screw_k0` 旁红点方向校验；
- 多帧异常值剔除、中值融合和稳定性门禁；
- RGB-D 架面 RANSAC 平面拟合；
- 正上方拟合并人工微调 `r1c1`、`r2c6` 两个槽圆；
- 根据两个圆心生成完整 2×6 槽位；
- 同一试管架内的扫描、规划和抓放；
- 正式抓放开始时自动进入已确认的全局观察位；
- 携管高位目标复检、最新坐标重规划和放置后闭环复扫；
- 本地标准地址 Agent，以及可选的 Gemini 自然语言解析。

当前版本不支持：

- 跨试管架底盘导航；
- 语音输入和语音播报；
- 缺少 CUDA 时回退到 CPU；
- 缺少模型或标定时回退到旧槽位 Detection；

跨架命令会被明确拒绝。导航接入前只能执行同一 `rack_id` 内的搬运。

## 2. 安全要求

真机运行必须同时满足：

- 急停按钮处于操作人员随手可触达的位置；
- 左臂收回，右臂周围无人且没有线缆、架体或其他障碍物；
- 底盘停止并锁定；
- 右臂使用真实控制模式且已经上电；
- `atom`、`zhixing_ctrl.py` 等遥操进程没有同时向设备发送指令；
- 相机、模型、手眼、TCP、管长、槽位标定和运动高度均已逐项验证；
- 第一次动作使用低速度，并由一名操作人员全程监护。

不要为了绕过检查而直接把 `confirmed` 或 `parameters_confirmed` 改成 `true`。只有对应
参数在真机上低速测量并确认后，才能解除软件锁。

任何阶段出现通信错误、坐标异常、视野变化、机械臂未到位或夹爪状态不确定时，应停止
任务并使用物理急停处理。软件显示仍持有试管时，不得移动底盘或开始下一次抓取。

## 3. 获取项目

```bash
git clone --branch agent/publish-final-runtime --single-branch \
  https://github.com/surf26/tubeGrabber.git
cd tubeGrabber
```

确认分支：

```bash
git branch --show-current
```

输出应为：

```text
agent/publish-final-runtime
```

## 4. Python 和依赖安装

项目要求 Python 3.10 或更高版本。建议使用独立虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

先根据部署机的 NVIDIA 驱动和 CUDA 版本安装对应的 CUDA 版 PyTorch。安装后确认：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

最后一个值必须为 `True`。然后安装项目及视觉、RealSense 可选依赖：

```bash
python -m pip install -e ".[vision,realsense]"
```

需要 Gemini Agent 时再安装：

```bash
python -m pip install -e ".[agent]"
```

RealMan Python SDK 不由本项目安装，需要按照机器人厂商环境安装，并确认：

```bash
python -c "from Robotic_Arm import rm_robot_interface; print('RealMan SDK OK')"
```

## 5. 项目配置

主配置文件是 `config/app.yaml`。项目内部的长度统一使用毫米，角度统一使用弧度，机械臂
坐标统一使用 `base_right`。

### 5.1 运行模式

初次使用保持：

```yaml
runtime:
  mode: fake
```

只有完成本文档所有模型、标定和只读硬件检查后，才切换为：

```yaml
runtime:
  mode: real
```

### 5.2 相机

核对以下配置与右腕 D435 一致：

- `camera.serial`：相机序列号；
- `width`、`height`、`fps`：运行分辨率和帧率；
- `warmup_frames`：启动后丢弃的预热帧；
- `depth_min_mm`、`depth_max_mm`：允许的深度范围。

相机分辨率、内参或安装姿态改变后，需要重新检查模型效果、手眼标定和槽位标定。

### 5.3 机械臂

核对：

- `arm.ip`、`arm.port`；
- 七自由度 `expected_dof: 7`；
- 工作坐标系 `Base`；
- SDK 工具坐标系 `Arm_Tip`；
- 低速运行所需的关节和直线速度百分比。

项目在自身运动学中应用法兰到夹爪 TCP 偏移，不应在 SDK 中额外启用另一个未知 TCP，
否则会造成重复偏移。

### 5.4 两指夹爪

项目只使用 RealMan 两指夹爪接口，不使用灵巧手接口。需要在安全位置低速实测：

- `open_position`：抓取前张开位置；
- `grip_position`：夹紧试管位置；
- `position_tolerance`：位置回读允许误差；
- `command_timeout_s`：阻塞动作超时。

夹紧时允许夹爪因为接触试管进入力停止状态，但必须有正向力反馈。

### 5.5 手眼和 TCP

`config/hand_eye.yaml` 保存右腕相机到右臂法兰的变换：

```text
camera_rightwrist -> end_right
```

平移单位必须为毫米。相机重新安装、支架松动或相机姿态变化后必须重新做手眼标定。

`geometry.tcp_offset_end_mm` 是法兰到夹爪 TCP 的偏移。还需要实测：

- `cap_top_above_rack_mm`；
- `grasp_depth_below_cap_mm`；
- `seating_adjust_mm`；
- `tube_total_length_mm`；
- `required_carried_clearance_mm`。

这些数值直接影响夹取深度、放置深度和携管平移净空。

### 5.6 观察位和运动参数

`config/poses.yaml` 中：

- `observation_pose` 是扫描时右臂法兰必须到达的位置；
- 正式 `transfer/agent-transfer` 会自动移动到该位置，放置后也会自动返回；
- `scan/plan-transfer` 是诊断或无运动入口，仍要求右臂事先位于该观察位；
- `vertical_tool_rpy_rad` 是抓放时工具竖直向下的姿态；
- `observation_pose.confirmed` 初始必须保持 `false`。

`config/app.yaml` 中 `motion.parameters_confirmed` 初始也必须保持 `false`。完成 TCP、工作
空间、竖直姿态、接近高度和抬升高度的低速真机检查后，才能分别确认这两个锁。
自动进入观察位使用 RealMan `movej_p`，同时受工作空间、单步距离、到位误差和
`maximum_single_orientation_change_deg` 限制；当前 30° 门限沿用旧 AprilTag 流程。
它不是任意起始姿态的避障规划，首次必须在已清空的允许起始区域低速验证完整路径。

## 6. 准备两个模型

最终运行需要：

```text
models/cap.pt
models/rack_pose.pt
```

权重默认不提交到 Git。

### 6.1 试管盖模型

`cap.pt` 必须是 YOLO Detection 模型，并严格只有一个类别：

```text
0: tube_cap
```

每根试管只标注盖子，不标空槽、架面或试管架。框中心应稳定落在能够取得有效盖顶深度的
位置。

### 6.2 架面 Pose 模型

在 `screw_k0` 旁固定贴一个小红圆点。红点应清晰、永久固定且不遮住螺丝孔中心。

采集图片：

```bash
python tools/capture_rack_pose_dataset.py --count 500
```

默认图片保存到 `datasets/rack_pose/images/raw`。采集时应覆盖：

- 空架、满架和不同槽位占用组合；
- 不同颜色的试管盖；
- 正常观察位附近的距离和角度变化；
- 合理范围内的曝光、阴影和反光；
- 轻微遮挡，但不能猜测严重遮挡的关键点。

模型类别必须为：

```text
0: rack_surface
```

每个架面固定标注 8 个关键点，顺序不可改变：

```text
k0, k1, k2, k3, screw_k0, screw_k1, screw_k2, screw_k3
```

`k0～k3` 是物理角点编号，不是图片中的临时左上、右上角。四个角点必须始终沿同一方向
顺时针标注；`screw_kN` 是靠近 `kN` 的螺丝孔中心。所有图片中的 `k0` 都必须对应红点
标识的同一个物理角。

按采集序列划分 train、val、test，不能把连续视频的相邻帧随机分到不同集合。复制并修改
数据配置：

```bash
cp training/rack_pose.yaml.example training/rack_pose.yaml
```

训练：

```bash
python tools/train_rack_pose.py \
  --data training/rack_pose.yaml \
  --model yolo11s-pose.pt \
  --epochs 200 \
  --imgsz 1280 \
  --device 0
```

脚本强制使用 CUDA，并关闭水平翻转，避免红点定义的物理 K0 被错误交换。

将选定权重复制为 `models/rack_pose.pt` 后，在独立测试集验收：

```bash
python tools/validate_rack_pose.py \
  --data training/rack_pose.yaml \
  --weights models/rack_pose.pt \
  --imgsz 1280 \
  --device 0
```

默认离线门限是 pose mAP50 不低于 0.95、mAP50-95 不低于 0.75。最终还必须在真实观察
位连续扫描，验证关键点抖动、三维槽点抖动、盖顶抖动和整链耗时。

## 7. Fake 模式演练

Fake 模式不连接相机、机械臂和夹爪，可用于确认命令、规划和状态机。

运行全部离线测试：

```bash
python -m unittest discover -s tests -v
```

静态检查：

```bash
python -m tube_grabber doctor
```

Fake 模式下模型、CUDA和标定缺失会显示为 `WAIT`，不是假硬件闭环失败。

扫描：

```bash
python -m tube_grabber scan --rack rack_1
```

只规划、不执行：

```bash
python -m tube_grabber plan-transfer \
  --source rack_1.r1c1 \
  --destination rack_1.r1c2
```

Fake 完整执行：

```bash
python -m tube_grabber transfer \
  --source rack_1.r1c1 \
  --destination rack_1.r1c2
```

标准槽位格式为：

```text
rack_1.r1c1 ... rack_1.r2c6
rack_2.r1c1 ... rack_2.r2c6
```

源槽必须是 `OCCUPIED`，目标槽必须是 `EMPTY`，源和目标不能相同。

## 8. 真机只读检查

模型和基础配置准备完成后，把 `runtime.mode` 切换为 `real`，但仍保持两个运动确认锁为
`false`。依次执行：

```bash
python -m tube_grabber doctor
python -m tube_grabber arm-status
python -m tube_grabber camera-check
python -m tube_grabber gripper-status
```

各命令作用：

- `doctor`：检查配置、模型契约、CUDA、依赖、手眼、标定和确认锁；
- `arm-status`：只连接右臂，读取法兰位姿、控制器模式、电源和健康状态；
- `camera-check`：只采集一帧彩色图和对齐深度图，并保存到 `artifacts/`；
- `gripper-status`：只读取两指夹爪在线、使能和错误状态，不发送位置命令。

`camera-check` 后检查彩色图、16 位毫米深度图、有效深度比例和运行时内参。图像尺寸或
对齐异常时不要继续标定。

## 9. 正上方双圆心标定

每个试管架分别标定一次。标定前：

1. 清空 `r1c1` 和 `r2c6`；
2. 确认红点位于 `screw_k0` 旁；
3. 低速手动将腕部相机移动到架面正上方；
4. 保持机械臂和试管架完全静止；
5. 确保整个架面、四角和四个螺丝孔均在画面内。

执行：

```bash
python -m tube_grabber calibrate-rack --rack rack_1
python -m tube_grabber calibrate-rack --rack rack_2
```

界面操作：

1. 在 `r1c1` 槽圆附近点击，程序自动尝试 Hough 圆拟合；
2. 自动拟合失败时会显示默认圆，可继续人工调整；
3. 按住鼠标左键拖动圆心；
4. `[`/`]` 或 `-`/`+` 每次调整半径 1 像素；
5. 方向键或 `I/J/K/L` 每次移动圆心 1 像素；
6. 圆周与真实槽边贴合后按 Enter 或空格确认；
7. 用同样方式确认 `r2c6`；
8. 检查绿色 2×6 网格的所有中心；
9. 按 `s` 保存，Backspace 撤回，`r` 全部重置，`q` 取消。

最终用于槽位计算的是两个人工确认后的圆心，半径只用于帮助判断中心。标定会保存到：

```text
config/racks/rack_1.yaml
config/racks/rack_2.yaml
```

已有标定只有显式添加 `--force` 才会覆盖：

```bash
python -m tube_grabber calibrate-rack --rack rack_1 --force
```

标定视角可以和正常观察位不同。程序会把两个槽圆心和 8 个关键点转换成架面归一化坐标，
正常运行时再通过当前架面四角投影到图像中。

以下情况必须重新标定：

- 架面模型关键点含义或顺序改变；
- 试管架结构、槽距或架面尺寸改变；
- D435 分辨率或内参改变；
- 相机安装姿态改变；
- 标定文件版本不兼容。

相机安装改变时，还必须重新做手眼标定。

## 10. 观察位扫描验收

标定完成后，低速把右臂移回正常观察位。确认真实法兰位姿与 `config/poses.yaml` 一致后，
才把 `observation_pose.confirmed` 设置为 `true`。

分别扫描：

```bash
python -m tube_grabber scan --rack rack_1
python -m tube_grabber scan --rack rack_2
```

输出包含：

- 架面平面 Z；
- 有效稳定帧数；
- 最大关键点像素抖动；
- 最大三维位置抖动；
- 12 个槽位的占用状态和置信度；
- 每个槽位的像素坐标；
- 空槽的架面三维坐标或占用槽的盖顶三维坐标。

开启 `runtime.save_debug_images` 时，标注图保存到 `artifacts/`。

每个架至少重复扫描 20 次，并测试空架、满架、不同占用、不同盖色和轻微遮挡。需要检查：

- 12 个绿色槽位中心没有行列翻转；
- `r1c1` 和 `r2c6` 方向正确；
- 红点移动到错误螺丝附近时识别会被拒绝；
- 架面移动、深度缺失或关键点抖动过大时不会输出可执行坐标；
- 所有三维坐标位于真实机械臂工作区域。

## 11. 无运动规划验收

在夹爪和机械臂运动仍被锁定时，执行：

```bash
python -m tube_grabber plan-transfer \
  --source rack_1.r1c1 \
  --destination rack_1.r2c6
```

该命令会连接真实机械臂和相机进行扫描，但不会初始化夹爪，也不会发送机械臂运动。因此
运行前仍需低速把右臂放在观察位。逐项检查输出中的：

- 源盖顶三维坐标；
- 目标槽架面坐标；
- 抓取和放置 TCP 目标；
- 起始法兰位姿；
- 接近、下降、抬升、平移、放置和撤离航点；
- 每个航点的速度、位置、姿态和运动类型。

用边界槽位重复检查，并确认所有航点满足工作空间、单步距离、工具倾角和携管净空限制。

## 12. 真机分阶段动作

只有完成低速实测后，才设置：

```yaml
motion:
  parameters_confirmed: true
```

第一次完整任务应选择距离短、周围无试管干涉的同架槽位：

```bash
python -m tube_grabber transfer \
  --source rack_1.r1c1 \
  --destination rack_1.r1c2
```

程序执行顺序：

1. 连接右臂和两指夹爪，检查真实模式、上电、控制器和关节错误；
2. 操作人员确认夹爪/TCP 物理空载并输入大写 `EMPTY`；
3. 程序使用旧 AprilTag 流程验证过的 `movej_p` 自动进入全局观察位，并检查到位误差；
4. 启动 D435，执行首轮多帧扫描并打印预览计划；
5. 操作人员确认结果后输入大写 `MOVE`；
6. 在同一观察位执行第二轮多帧扫描，比较现场变化，并用第二轮最新坐标重新生成抓取计划；
7. 执行张开、抓取和 150 mm 垂直上提；
8. 携管移动到目标槽高位安全走廊，不进入较低的全局观察位；
9. 从高位走廊再次多帧识别架面，确认源槽已空、目标槽仍空、其他槽未变化；
10. 只忽略目标槽上方由夹爪持有的一个高位盖子，普通就位盖和其他异常仍会阻断；
11. 使用这次目标复检得到的最新槽位三维坐标重新规划下降并放置；
12. 释放、垂直撤离后，程序自动返回全局观察位；
13. 最终多帧复扫必须确认源槽为空、目标槽占用、其他槽未异常变化，才报告任务完成。

如果目标高位复检失败，程序保持持管状态并停止，不会为了拍摄而携管进入低观察位，也
不会自行释放。如果最终复扫失败，物理动作已经结束但任务结果仍为失败，必须现场检查。

## 13. Agent 使用

### 13.1 本地解析器

本地解析器不联网，文本中必须恰好包含两个标准槽位地址。只生成计划：

```bash
python -m tube_grabber agent-plan \
  --text "rack_1.r1c1 -> rack_1.r1c2"
```

执行入口：

```bash
python -m tube_grabber agent-transfer \
  --text "rack_1.r1c1 -> rack_1.r1c2"
```

### 13.2 Gemini 自然语言解析

先在自己的 Google AI Studio 账号创建密钥，并只保存到环境变量：

```bash
export GEMINI_API_KEY="你的密钥"
```

密钥不得写入 YAML、Python、日志或 Git。先测试只读规划：

```bash
python -m tube_grabber agent-plan \
  --agent-provider gemini \
  --text "把一号架第一行第一列的试管移动到一号架第一行第二列"
```

确认解析后的标准命令、视觉扫描和航点全部正确后，才使用 `agent-transfer`。

Gemini 只提出一个结构化搬运命令，不直接获得机械臂、夹爪或相机函数。Python 会重新检查
架号、行列范围、源目标关系、占用、视觉、规划、执行前现场变化和人工 `MOVE` 授权。

当前语音尚未接入。未来语音识别只应把 ASR 文本送入同一个 Agent 入口，不能绕过现有
安全校验直接控制硬件。

底盘应如何使用 Woosh 命名站点、同架和跨架任务如何调度、开始时如何自动从 HOME 到源架
并自动进入观察位，以及本机/外接电脑语音方案，详见
[`NAVIGATION_AND_VOICE.md`](NAVIGATION_AND_VOICE.md)。该文档是下一阶段实施和验收方案；
当前分支仍未启用导航和语音。

## 14. 常见故障

### `doctor` 报 CUDA 不可用

- 确认安装的是 CUDA 版 PyTorch；
- 确认 NVIDIA 驱动正常；
- 确认 `torch.cuda.is_available()` 为 `True`；
- 确认 `vision.device` 是 GPU 编号字符串，例如 `"0"`。

真机不会回退到 CPU。

### 模型契约错误

- `cap.pt` 必须只有 `{0: tube_cap}`；
- `rack_pose.pt` 必须只有 `{0: rack_surface}`；
- Pose 必须输出 8×3 关键点；
- 关键点顺序必须与配置完全一致。

### 红点检查失败

- 检查红点是否确实位于 `screw_k0` 附近；
- 检查曝光、饱和度和反光；
- 检查 Pose 是否把螺丝孔编号识别错误；
- 不要通过降低门限掩盖错误方向。

### 多帧稳定性失败

- 保持机械臂和试管架静止；
- 检查相机曝光、运动模糊和 USB 带宽；
- 检查关键点标注质量；
- 检查是否存在严重遮挡；
- 通过保存的标注图定位具体异常，不要直接关闭稳定性门禁。

### 架面平面拟合失败

- 检查深度图是否与彩色图对齐；
- 检查架面是否大部分处于有效深度范围；
- 清理镜头和反光干扰；
- 检查关键点是否把 ROI 投影到了错误区域。

### 双圆拟合不准

- 确保相机位于架面正上方；
- 清空 `r1c1` 和 `r2c6`；
- 只需点击槽圆附近，不需要肉眼点中圆心；
- 用圆周贴合槽边，再通过键盘逐像素微调；
- 自动 Hough 拟合失败时使用默认圆继续人工调整；
- 必要时在 `vision.calibration` 中调整搜索半径、允许半径和 Hough 阈值。

### 机械臂状态检查失败

- 确认控制器选择真实模式并上电；
- 确认 IP、端口、Base 和 Arm_Tip 正确；
- 检查控制器和七个关节错误码；
- 停止与 SDK 抢占控制的遥操进程；
- 不要由程序自动清错、自动上电或自动切换真实模式。

### 夹爪状态失败

- 确认使用的是 RealMan 两指夹爪；
- 检查在线、使能、错误码、实际开度和力反馈；
- 在安全位置重新验证张开和夹紧位置；
- 状态不确定时按“可能仍夹着试管”处理。

### 跨架任务被拒绝

这是当前版本的预期安全行为。底盘导航尚未接入，以下命令不会执行：

```text
rack_1.* -> rack_2.*
rack_2.* -> rack_1.*
```

## 15. 推荐验收顺序

严格按照以下顺序推进：

1. 离线单元测试；
2. Fake 扫描、规划和完整任务；
3. CUDA 与两个模型契约；
4. 只读机械臂、相机和夹爪检查；
5. 两个架面的正上方双圆心标定；
6. 正常观察位连续扫描统计；
7. 无运动的全部航点检查；
8. 安全空间中的夹爪开合；
9. 单段低速机械臂动作；
10. 最短距离的同架完整任务；
11. 多槽位重复试验和成功率统计；
12. 视觉、运动和异常恢复稳定后，再开始接入底盘导航和语音。

更精简的现场勾选表见 `docs/LAB_CHECKLIST.md`。
