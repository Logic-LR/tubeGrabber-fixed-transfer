# Tube Grabber 项目总说明与操作手册

本文是本项目的总说明和现场操作手册。它把项目的目标、软件结构、硬件连接、配置
含义、命令入口、当前右臂双试管架任务、机器人端运行方式、故障处理和发布流程放在
一起。除非特别说明，命令都应在项目根目录执行。

> [!CAUTION]
> 本项目可以驱动真实的右臂、夹爪和移动底盘，不是安全认证系统。真机运行前必须保证
> 急停可触达、右臂和底盘运动范围无人无障碍、左臂收回、现场有人低速监护。相机、手眼、
> TCP、观察位、工作空间、架子标定和底盘距离都只适用于当前实验设备，换设备后必须
> 重新测量和低速验证。

## 1. 项目是什么

Tube Grabber 是一个基于 Intel RealSense D435、Ultralytics YOLO、RealMan 七轴机械臂、
RM Plus 两指夹爪和 Woosh 移动底盘的视觉引导试管抓放系统。项目把以下环节连成一个
可测试、可复现的闭环：

1. 用右腕 D435 采集对齐的 RGB-D 图像；
2. 用 `screw.pt` 找到试管架四颗角螺丝，并用 K0 旁的白色 marker 确定架子方向；
3. 用 `cap.pt` 找到试管盖，结合深度计算盖顶三维坐标；
4. 用架面平面、双圆心标定和当前四角投影生成 2×6 槽位；
5. 判断每个槽位是 `OCCUPIED`、`EMPTY` 或 `UNKNOWN`；
6. 在规划阶段检查工作空间、工具方向、单段距离和携管净空；
7. 执行抓取、撤离、移动、放置、释放和最终复扫；
8. 在双架任务中，让底盘移动后重新识别目标架，不复用移动前的旧三维坐标。

当前项目同时保留三类入口：

| 入口 | 适用场景 | 是否可能驱动硬件 |
|---|---|---:|
| `transfer` | 同一个试管架内抓放 | 是 |
| `fixed-transfer` | 已示教的固定点位取管到篮子 | 是 |
| `mobile-transfer` | `rack_1` 取管，经底盘移动后放入 `rack_2` | 是 |

`agent-plan` 和 `agent-transfer` 是自然语言到标准槽位命令的适配入口；Agent 不能绕过
视觉、规划、确认锁或同架/跨架限制。项目没有接入语音输入和语音播报。

## 2. 当前设备和部署信息

下面是当前实验设备的实际连接信息。密码不写入项目、脚本或文档，登录时按现场保存的
凭据输入。

| 项目 | 当前值 |
|---|---|
| 机器人主机 | `192.168.3.68` |
| SSH 用户 | `rm` |
| 当前部署链接 | `/home/rm/tubeGrabber-mobile-current` |
| 实际部署目录 | `/home/rm/deployments/tubeGrabber-mobile-20260909_173721` |
| Conda 环境 | `tube_vision` |
| Conda 根目录 | `/home/rm/miniconda3` |
| 机械臂 | RealMan 七轴，右臂 |
| 机械臂控制器地址 | `169.254.128.19:8080` |
| 相机 | 右腕 Intel RealSense D435 |
| 相机序列号 | `405622073249` |
| 夹爪 | RM Plus ZX 1DOF 两指夹爪 |
| 夹爪波特率 | `9600` |

`192.168.3.68` 是运行项目的机器人主机地址，`169.254.128.19` 是项目从机器人主机
连接右臂控制器使用的地址，两者不是同一个网络端点。

## 3. 当前右臂双架任务做什么

当前推荐的一键入口是机器人上的：

```bash
cd ~/tubeGrabber-mobile-current
./tools/pick_rotate_forward.sh
```

这个脚本会：

1. 检查并停止可能抢占机械臂控制权的 `atom`、`zhixing_ctrl.py`；
2. 用右臂相机在 `rack_1` 扫描，自动选择唯一确认有试管的槽位；
3. 右臂抓取试管并回到带管 `loaded observation/home`；
4. 按当前配置让底盘闭环旋转 `180°`，再沿车体 X 方向前移 `0.30 m`；
5. 到达第二站后重新用右臂相机识别 `rack_2`，连续稳定扫描两次；
6. 从两次均为 `EMPTY` 且有架面坐标的槽位中，按 `r1c1` 到 `r2c6` 顺序选第一个；
7. 以移动后的新三维坐标执行插管、释放和撤离；
8. 右臂回到 `loaded observation/home`，最终复扫 `rack_2`；
9. 放置确认成功后保持右臂姿势不变，让底盘按逆序反向移动回起始位置；
10. 输出完成信息并结束进程。

当前配置中 `verify_pick_after_home: false`，所以抓取回带管 home 后不会再次扫描源架，
而是直接进入底盘移动，目标架会在移动后重新识别。目标架的两次稳定扫描和放置后的
最终复扫仍然执行。

一键脚本没有额外的 `RUN`、`PICK-HOME` 或 Enter 交互确认；执行命令本身就是这条专用
批处理任务的授权。它只适用于已经完成现场检查的当前设备。普通 `transfer`、观察位
工具和部分恢复工具仍可能要求输入确认，不要把它们的交互行为与一键脚本混淆。

## 4. 系统工作原理

```text
右腕 D435 RGB-D + 右臂法兰位姿
             │
             ├─ screw.pt ─→ 四颗角螺丝 ─→ K0 白色 marker 定向
             │                              │
             │                              └─ 多帧稳定性和架面几何检查
             │
             ├─ 架面深度 ROI ─→ RANSAC 平面 ─→ 12 个槽位三维坐标
             │
             └─ cap.pt ─→ 盖中心 + 深度 ─→ 盖顶三维坐标
                                           │
                 r1c1/r2c6 双圆心标定 ─────┘
                                           │
                              OCCUPIED / EMPTY / UNKNOWN
                                           │
                           规划 → 抓放 → 复扫 → 结果确认
```

### 4.1 视觉识别

- `screw.pt` 必须稳定检测到恰好四颗架面角螺丝，类别为 `0: item`；
- 四颗螺丝框中心用于拟合当前架面；
- K0 旁额外的白色 marker 不属于 YOLO 类别，由 HSV 高亮低饱和区域检测；
- marker 用于确定物理 K0，从而避免架子方向翻转；
- `cap.pt` 只检测试管盖，类别同样为 `0: item`；
- 空槽不是另一个 YOLO 类别，而是标定槽位附近没有稳定盖子检测的结果；
- 连续帧会做异常剔除、中心融合、占用投票、槽点抖动和盖顶抖动检查；
- 深度经过相机内参、手眼矩阵和架面平面换算到 `base_right`。

### 4.2 坐标、单位和方向

- 长度：项目内部使用毫米 `mm`；底盘配置的平移距离使用米 `m`；
- 角度：项目内部使用弧度 `rad`，显示和底盘旋转配置可使用度数；
- 机械臂工作坐标：`base_right`；
- 手眼方向：`camera_rightwrist -> end_right`，再组合到 `base_right`；
- 工具坐标：RealMan `Arm_Tip`，项目另外应用自己的 TCP 偏移；
- 架面物理上不一定与控制器 `Base Z` 重合，抓取、放置和撤离沿视觉估计架面法向；
- `rack_1` 和 `rack_2` 必须分别标定，不能复制一个标定文件冒充另一个架子。

### 4.3 底盘闭环

真实底盘不按“速度 × 时间”估算距离，而是调用机器人端的闭环辅助程序：

- `grabber_rotate_relative`：读取实时位姿，完成约 `90°` 的相对旋转并检查平移漂移；
- `grabber_pose_servo`：按实时位姿完成短距离相对平移并检查位置/角度误差；
- `180°` 被拆成两次约 `90°`；
- `0.30 m` 平移被拆成两个 `0.15 m` 段，单段不超过 `0.25 m`；
- 回程顺序是先在当前车体坐标系反向平移，再做两次反向 `90°` 旋转；
- 任一辅助程序超时、非零退出、缺少最终位姿或误差超限，流程立即停止；
- 右臂回到 home 后，底盘回程阶段不再发送右臂指令。

不要使用历史的 `agv_step_rotate*` 二进制替代当前闭环辅助程序。机器人上的
`tools/agv_debug_tools/` 是隔离部署目录，里面的备份、SDK、二进制和调试日志不是
GitHub 项目的 Python 源码组成部分。

## 5. 项目目录

```text
tubeGrabber/
├── config/
│   ├── app.yaml                    # 总配置：硬件、视觉、几何和运动门限
│   ├── hand_eye.yaml               # 右腕相机到右臂法兰的手眼矩阵
│   ├── poses.yaml                  # 右臂观察/home 位姿
│   ├── mobile_transfer.yaml        # rack_1 -> rack_2 双架路线配置
│   ├── fixed_transfer.yaml         # 固定取管到篮子的点位模板
│   └── racks/                      # 每个试管架独立的 2×6 标定
├── models/
│   ├── cap.pt                      # 试管盖 YOLO Detection 权重，不默认提交
│   └── screw.pt                    # 角螺丝 YOLO Detection 权重，不默认提交
├── artifacts/                      # 运行时图片和调试产物
├── tube_grabber/
│   ├── core/                       # 数据模型、错误和硬件端口
│   ├── vision/                     # 检测、深度、平面、标定和槽位匹配
│   ├── motion/                     # TCP 变换、航点、工作空间门禁
│   ├── workflow/                   # 同架抓放和复扫状态机
│   ├── hardware/                   # D435、RealMan、夹爪适配器
│   ├── navigation/                 # Woosh 底盘边界
│   ├── agent/                      # local/Gemini 命令解析
│   ├── app.py                      # 依赖组装和生命周期
│   ├── mobile_transfer.py           # 双架移动协调器
│   └── cli.py                      # 所有命令行入口
├── tools/                          # 观察位、视觉流、TCP 和现场工具
├── tests/                          # 离线单元和状态机测试
├── docs/                           # 项目说明、架构、验收和操作文档
├── pyproject.toml                  # 包信息和可选依赖
└── requirements.txt                # 兼容安装清单
```

代码分层约束是：视觉层不发运动，运动层不理解槽位业务，硬件层只翻译 SDK，业务流程
通过端口和 dataclass 连接各层。详细边界见 `docs/ARCHITECTURE.md`。

## 6. 安装和环境

### 6.1 Windows 或普通 Linux 开发机

```bash
git clone https://github.com/Logic-LR/tubeGrabber-fixed-transfer.git
cd tubeGrabber
python -m venv .venv
```

Linux/macOS 激活：

```bash
source .venv/bin/activate
```

Windows PowerShell 激活：

```powershell
.venv\Scripts\Activate.ps1
```

安装基础包和项目：

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

需要视觉功能时安装：

```bash
python -m pip install -e ".[vision,realsense]"
```

需要 Gemini Agent 时再安装：

```bash
python -m pip install -e ".[agent]"
```

真机环境还必须单独准备与 NVIDIA 驱动匹配的 CUDA 版 PyTorch，以及厂商提供的
RealMan Python SDK。检查：

```bash
python -c "import cv2, yaml, numpy; print('基础依赖 OK')"
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "from Robotic_Arm import rm_robot_interface; print('RealMan SDK OK')"
```

如果第一条报 `No module named cv2`，说明 OpenCV 尚未安装，先执行视觉依赖安装；如果
CUDA 最后一项为 `False`，不要继续真机视觉运行，当前项目不会静默回退 CPU。

### 6.2 机器人端环境

登录机器人：

```bash
ssh rm@192.168.3.68
```

进入当前隔离部署：

```bash
cd ~/tubeGrabber-mobile-current
readlink -f ~/tubeGrabber-mobile-current
conda activate tube_vision
```

也可以不激活环境，使用项目脚本内部固定的 Conda 路径。确认项目和关键模型存在：

```bash
pwd
ls -l tube_grabber config/mobile_transfer.yaml config/racks/rack_1.yaml config/racks/rack_2.yaml
ls -lh models/cap.pt models/screw.pt
```

GitHub 推送不会自动更新机器人目录。机器人运行的是当前部署链接指向的目录，代码更新
后必须经过单独的发布/复制流程，并重新核对链接、模型、辅助程序和关键文件哈希。

## 7. 模型和标定准备

### 7.1 模型文件

必须存在：

```text
models/cap.pt
models/screw.pt
```

两个模型都必须是 YOLO Detection 模型，且元数据只有：

```yaml
names:
  0: item
```

`cap.pt` 只标试管盖；`screw.pt` 只标试管架四颗角螺丝。不要把空孔、架面或白色
marker 标成 Detection 类别。模型权重默认被 `.gitignore` 排除，不要把私有权重或密钥
直接提交 GitHub。

验证类别：

```bash
python -c "from ultralytics import YOLO; print(YOLO('models/cap.pt', task='detect').names)"
python -c "from ultralytics import YOLO; print(YOLO('models/screw.pt', task='detect').names)"
```

右腕相机彩色流的 CUDA 验收：

```bash
python tools/test_yolo_stream.py --rack rack_1
```

此工具只打开 D435 彩色流和 YOLO，不连接机械臂、夹爪，不执行深度定位和运动。按
`q` 或 `Esc` 退出。若测试 `rack_2`，该架必须先完成展示四角标定：

```bash
python tools/test_yolo_stream.py --rack rack_2
```

### 7.2 试管架双圆心标定

每个架子都要独立标定 `r1c1` 和 `r2c6`：

1. 清空两个对角槽；
2. 低速把右腕相机移动到架面正上方；
3. 保持机械臂和架子静止；
4. 执行标定命令；
5. 在两个槽圆附近点击，拖动圆心并调整半径；
6. 检查 2×6 网格方向、槽中心和 K0 方向；
7. 保存标定文件。

命令：

```bash
python -m tube_grabber --real calibrate-rack --rack rack_1
python -m tube_grabber --real calibrate-rack --rack rack_2
```

已有文件默认拒绝覆盖，明确确认后才使用：

```bash
python -m tube_grabber --real calibrate-rack --rack rack_2 --force
```

标定窗口常用按键：

| 按键 | 作用 |
|---|---|
| 鼠标左键拖动 | 移动当前圆心 |
| `[` / `]`、`-` / `+` | 调整圆半径 |
| 方向键或 `I/J/K/L` | 每次微调圆心 1 px |
| Enter / Space | 确认当前圆或角点 |
| `Backspace` | 撤回上一个确认 |
| `r` | 重置当前标定 |
| `s` | 保存 |
| `q` / `Esc` | 取消 |

展示四角点单独标定：

```bash
python -m tube_grabber --real calibrate-corners --rack rack_1
python -m tube_grabber --real calibrate-corners --rack rack_2
```

双圆心标定写入 `config/racks/rack_*.yaml`；展示角点只影响实时预览画面，不替代四螺丝
架面检测。相机分辨率、镜头安装、架子结构或内参改变后必须重新标定。

## 8. 配置说明

### 8.1 总配置 `config/app.yaml`

| 配置段 | 作用 |
|---|---|
| `runtime` | fake/real、交互确认和调试图保存 |
| `camera` | D435 序列号、分辨率、帧率、预热和深度范围 |
| `arm` | 右臂 IP、Base/Arm_Tip、速度和冲突进程 |
| `gripper` | 波特率、电压、开合位置、力和超时 |
| `vision.cap` | 试管盖模型、置信度、IoU 和输入尺寸 |
| `vision.screw` | 螺丝模型、marker 阈值和架面检测参数 |
| `vision.stability` | 采集帧数、内点和抖动门限 |
| `vision.plane` | RANSAC 架面和平面法向门限 |
| `geometry` | 手眼路径、TCP、盖高、抓取深度和管体净空 |
| `motion` | 观察位、速度、工作空间和确认锁 |
| `racks` | 已启用的架子及其标定文件 |

当前实验配置的重要值：

```yaml
runtime:
  mode: fake                         # 一键真机脚本用 --real 覆盖为 real
  require_enter_before_motion: true

arm:
  ip: "169.254.128.19"
  work_frame: "Base"
  tool_frame: "Arm_Tip"

motion:
  parameters_confirmed: true
  confirm_each_step: false
  workspace_min_mm: [-80.0, 250.0, -110.0]
  workspace_max_mm: [250.0, 435.0, 200.0]
```

`runtime.mode: fake` 是配置文件的默认值；专用机器人命令显式加 `--real`，因此不要
因为看到 `mode: fake` 就误以为机器人脚本不会连接真机。相反，普通真机命令也建议
显式写 `--real`，让运行意图清楚可见。

### 8.2 观察位 `config/poses.yaml`

`observation_pose` 是右臂用于扫描架子的全局观察位，也是双架任务的带管运输 home。它
必须同时满足：

- 右腕相机能完整看到四颗螺丝和 K0 marker；
- 夹着的试管不进入架面多边形，不碰架子和周边物体；
- 右臂与底盘路线有足够净空；
- 位置、姿态和工作空间均经过当前设备低速验证。

`observation_pose.confirmed` 只表示观察位已经在当前设备人工确认，不能代替运动参数和
双架路线确认。

### 8.3 双架配置 `config/mobile_transfer.yaml`

当前服务器对齐后的关键值如下：

| 键 | 当前值 | 含义 |
|---|---:|---|
| `pick_home_confirmed` | `true` | 右臂抓取并回带管 home 已确认 |
| `verify_pick_after_home` | `false` | 当前流程跳过抓取后源架复扫 |
| `return_to_start_after_transfer` | `true` | 放置成功后底盘自动反向回程 |
| `confirmed` | `true` | 双架路线已解除顶层确认锁 |
| `loaded_observation_pose.confirmed` | `true` | 带管观察/home 已确认 |
| `chassis.rotation_deg` | `180.0` | 去目标站的总旋转 |
| `chassis.translation_x_m` | `0.30` | 去目标站的总平移 |
| `rotation_speed_radps` | `0.08` | 底盘旋转速度上限 |
| `translation_speed_mps` | `0.03` | 底盘平移速度上限 |
| `maximum_rotation_translation_m` | `0.03` | 旋转期间允许的最大漂移 |
| `timeout_s` | `60.0` | 单个底盘辅助程序超时 |

如果换设备或路线未经重新验收，应先把对应 `confirmed` 锁改回 `false`，同时把
`motion.parameters_confirmed` 和观察位确认锁关闭。修改 TCP 的工具也会自动关闭
`motion.parameters_confirmed`。

### 8.4 关键安全锁

真机执行前至少要理解这些锁：

| 锁 | 为 `false` 时的结果 |
|---|---|
| `config/poses.yaml: observation_pose.confirmed` | 禁止自动进入/确认观察位 |
| `config/app.yaml: motion.parameters_confirmed` | 禁止真实运动 |
| `config/mobile_transfer.yaml: pick_home_confirmed` | 禁止抓取回 home 阶段 |
| `config/mobile_transfer.yaml: loaded_observation_pose.confirmed` | 禁止双架带管运输 |
| `config/mobile_transfer.yaml: confirmed` | 禁止完整双架任务 |
| `config/fixed_transfer.yaml: confirmed` | 禁止固定点位任务 |

不要为了绕过错误直接放宽工作空间或打开确认锁。应先确认错误对应的是配置未完成、
真实位姿不对、坐标系不对还是路线未验收。

## 9. 命令总表

所有全局选项放在子命令前，例如 `python -m tube_grabber --real scan ...`。

### 9.1 离线和只读命令

```bash
# 查看命令帮助
python -m tube_grabber --help
python -m tube_grabber <command> --help

# 静态检查配置、模型契约、依赖、手眼和标定
python -m tube_grabber doctor

# 真机只读检查
python -m tube_grabber --real doctor
python -m tube_grabber --real arm-status
python -m tube_grabber --real camera-check
python -m tube_grabber --real gripper-status

# 在当前观察位扫描；默认打开实时窗口
python -m tube_grabber --real scan --rack rack_1
python -m tube_grabber --real scan --rack rack_2 --no-display

# 只规划，不初始化夹爪、不发送机械臂运动
python -m tube_grabber --real plan-transfer \
  --source rack_1.r1c1 \
  --destination rack_1.r1c2

# 只检查双架配置，不连接任何硬件
python -m tube_grabber plan-mobile-transfer \
  --auto-source --auto-destination
```

命令风险说明：

- `doctor`、`plan-*`：不应发送运动；
- `arm-status`、`camera-check`、`gripper-status`：会连接相应设备，但只做读取/采集；
- `scan`：不自动移动右臂，但要求右臂已经在已确认观察位；
- `calibrate-*`：不会自动移动机械臂，但相机和机械臂会被连接，操作者需要手动摆位；
- `transfer`、`fixed-transfer`、`mobile-pick-home`、`mobile-transfer`：可能发送运动；
- `tools/move_to_observation_pose.py`、恢复脚本：可能直接发送运动，必须单独审查。

### 9.2 同架抓放

先规划：

```bash
python -m tube_grabber plan-transfer \
  --source rack_1.r1c1 \
  --destination rack_1.r1c2
```

再执行：

```bash
python -m tube_grabber --real transfer \
  --source rack_1.r1c1 \
  --destination rack_1.r1c2
```

`transfer` 只允许源和目标在同一个 `rack_id` 内。它会执行观察位、抓取前复扫、抓取、
放置、释放、撤离、回观察位和最终复扫。普通同架任务通常仍会要求人工 Enter 确认；
输入 `q` 可取消。

### 9.3 双架分阶段抓取回 home

用于只验收右臂，不移动底盘、不释放试管：

```bash
python -m tube_grabber plan-mobile-pick-home --auto-source
python -m tube_grabber --real mobile-pick-home --auto-source
```

也可以指定源槽：

```bash
python -m tube_grabber --real mobile-pick-home \
  --source rack_1.r1c1
```

自动源槽模式要求稳定扫描中恰好一个 `OCCUPIED`。流程成功退出后，夹爪仍保持夹紧，
不能直接再次启动抓取命令。现场必须先决定下一步是进入完整双架任务、人工取下试管，
还是使用经过审查的恢复流程。

### 9.4 双架完整任务

规划：

```bash
python -m tube_grabber plan-mobile-transfer \
  --auto-source --auto-destination
```

执行，自动选择源槽和目标空槽：

```bash
python -m tube_grabber --real mobile-transfer \
  --auto-source --auto-destination
```

固定目标槽的写法：

```bash
python -m tube_grabber --real mobile-transfer \
  --source rack_1.r1c1 \
  --destination rack_2.r1c2
```

当前机器人上最推荐的等价入口仍是：

```bash
cd ~/tubeGrabber-mobile-current
./tools/pick_rotate_forward.sh
```

不要在执行这个脚本时再手动输入确认文本，也不要同时启动 `atom` 或
`zhixing_ctrl.py`。脚本会先检查冲突进程，发现无法退出时直接退出，不发送任务运动。

### 9.5 固定取管到篮子

这是独立于相机和 rack 视觉的固定点位流程，当前模板仍未完成确认，不要和双架任务混用：

```bash
python -m tube_grabber plan-fixed-transfer
python -m tube_grabber --real fixed-transfer
```

只有 `config/fixed_transfer.yaml` 中的所有点位不再是 `null`、固定路径已低速验收并且
`confirmed: true` 后，才允许执行。

### 9.6 Agent

本地解析器只接受标准槽位地址，不联网：

```bash
python -m tube_grabber agent-plan \
  --text "rack_1.r1c1 -> rack_1.r1c2"
python -m tube_grabber --real agent-transfer \
  --text "rack_1.r1c1 -> rack_1.r1c2"
```

Gemini 只在本次命令启用，密钥只放环境变量：

```bash
export GEMINI_API_KEY="your-key"
python -m tube_grabber agent-plan \
  --agent-provider gemini \
  --text "把一号架第一行第一列的试管移动到一号架第一行第二列"
```

不要把密钥写入 YAML、Python、日志或 Git。Agent 不能把跨架自然语言命令变成普通
`transfer`，跨架任务仍必须显式使用 `mobile-transfer`。

## 10. 推荐现场操作顺序

### 阶段 A：离线验证

```bash
python -m pytest
python -m unittest discover -s tests -v
python -m compileall -q tube_grabber tools
git diff --check
```

如果本机只安装了最小依赖，`pytest` 可能在导入 `cv2` 时失败；先安装
`python -m pip install -e ".[vision,realsense]"`。离线测试通过不代表真实相机、模型或
机械臂安全。

### 阶段 B：模型和架子

1. 确认 `cap.pt`、`screw.pt` 和 CUDA；
2. 完成 `rack_1`、`rack_2` 独立双圆心标定；
3. 根据需要完成展示角点标定；
4. 在空架、满架、多种占用和轻微遮挡下重复 `scan`；
5. 记录四螺丝、槽中心、盖顶位置的抖动和耗时。

### 阶段 C：只读硬件

```bash
python -m tube_grabber --real doctor
python -m tube_grabber --real arm-status
python -m tube_grabber --real camera-check
python -m tube_grabber --real gripper-status
```

确认右臂为真实模式、已上电、七轴健康、Base/Arm_Tip 正确、D435 序列号正确、夹爪
在线且无错误，且没有外部程序抢占控制。

### 阶段 D：无运动规划

```bash
python -m tube_grabber --real plan-transfer \
  --source rack_1.r1c1 \
  --destination rack_1.r1c2
python -m tube_grabber plan-mobile-transfer \
  --auto-source --auto-destination
```

逐项查看输出的源盖顶、目标孔、TCP、抓取/放置/撤离航点、速度、工具方向和工作空间。

### 阶段 E：低速右臂

先做空载和短距离同架任务，再做 `mobile-pick-home`。右臂带管 home 后，检查试管净空、
相机视野、夹爪力和底盘路径；不要在没有验证的情况下直接运行跨站任务。

### 阶段 F：双架任务

1. 机器人位于起始位置，右臂位于允许起点，夹爪和源架状态符合配置；
2. `rack_1` 只有一个可抓取试管，`rack_2` 至少有一个真实空槽；
3. 左臂收回，底盘路线无人无障碍；
4. 右臂相机视野不会被持管试管遮挡四螺丝和 K0；
5. 直接执行 `./tools/pick_rotate_forward.sh`；
6. 全程观察输出，不要中途启动其他控制程序；
7. 失败时按“可能仍持管”处理，不要直接重复执行。

## 11. 机器人端直接命令

### 11.1 登录和预检

```bash
ssh rm@192.168.3.68
cd ~/tubeGrabber-mobile-current
readlink -f ~/tubeGrabber-mobile-current
conda activate tube_vision
```

机器人上不激活环境时，下面这种写法最稳定：

```bash
env PYTHONPATH="$PWD" /home/rm/miniconda3/bin/conda run \
  -n tube_vision --no-capture-output --cwd "$PWD" \
  python -m tube_grabber --real doctor
```

检查外部控制进程和当前项目进程：

```bash
pgrep -af 'atom|zhixing_ctrl|tube_grabber|grabber_'
```

只读检查当前部署的核心文件：

```bash
sha256sum \
  tube_grabber/mobile_transfer.py \
  tube_grabber/navigation/chassis.py \
  tube_grabber/cli.py \
  tools/pick_rotate_forward.sh
```

### 11.2 机器人上的规划和执行

```bash
env PYTHONPATH="$PWD" /home/rm/miniconda3/bin/conda run \
  -n tube_vision --no-capture-output --cwd "$PWD" \
  python -m tube_grabber plan-mobile-transfer \
  --auto-source --auto-destination
```

专用一键任务：

```bash
./tools/pick_rotate_forward.sh
```

脚本内部等价于：

```bash
env PYTHONPATH="$PWD" /home/rm/miniconda3/bin/conda run \
  -n tube_vision --no-capture-output --cwd "$PWD" \
  python -m tube_grabber --real mobile-transfer \
  --auto-source --auto-destination
```

### 11.3 运行结果判断

成功时应看到类似以下逻辑结果：

```text
移动前自动选择源槽：rack_1.rNcM
移动后自动选择目标槽：rack_2.rNcM
目标架最终闭环复扫：
跨站搬运完成：...；机械臂保持 loaded observation/home，底盘已回机器人起始位置
```

进程退出码为 `0` 才表示软件闭环成功。即使出现“放置动作完成”，只要最终复扫失败，
任务仍应视为失败，现场必须检查目标槽、夹爪和右臂姿态。

## 12. 故障处理

### 12.1 `target Y=... outside [...]`

这是机械臂工作空间门禁，不是视觉程序自动修正。它表示计划目标的 Y 坐标落在
`config/app.yaml` 的 `workspace_min_mm` 和 `workspace_max_mm` 之外。处理顺序：

1. 确认右臂、工具坐标和手眼矩阵没有错；
2. 确认视觉识别的架面和法向正确；
3. 重新检查 TCP、目标深度和观察位；
4. 只有在当前机械臂低速验证过真实可达范围后，才修改工作空间边界；
5. 修改后重新跑 `plan-transfer`，不能只看配置文件不看航点。

不要为了消除一条报错而无限放宽边界。边界放宽后，错误坐标也可能被允许发送到真实
机械臂。

### 12.2 `confirmed is false` 或 `parameters_confirmed is false`

这是故意的配置锁。先完成对应设备的低速测量和验收，再修改对应字段；不要用命令行
参数绕过，也不要复制另一台机器的确认值。

### 12.3 `No module named cv2`、CUDA 或模型错误

```bash
python -m pip install -e ".[vision,realsense]"
python -c "import cv2; print(cv2.__version__)"
python -c "import torch; print(torch.cuda.is_available())"
```

再检查 `models/cap.pt`、`models/screw.pt` 的 Detection 类型和 `{0: item}` 类别契约。
真机不回退 CPU。

### 12.4 `atom` 或 `zhixing_ctrl.py` 冲突

它们是项目外部的控制程序，可能向机械臂发送并发指令。主脚本会尝试停止精确匹配的
进程；如果进程没有退出，脚本应在发送运动前退出。不要在主任务运行中重新启动它们。

### 12.5 视觉扫描不稳定

检查：

- 四颗螺丝是否全部可见且没有反光/遮挡；
- K0 白色 marker 是否唯一且没有与白色底板连成一片；
- 右臂和架子在多帧采集期间是否静止；
- 彩色和深度是否对齐，深度有效率是否足够；
- 架子方向、双圆心标定和相机安装是否改变；
- `artifacts/` 中的标注图是否显示了错误的盖框或架面。

不要直接关闭稳定性门禁来掩盖识别抖动。

### 12.6 底盘辅助程序失败或位姿漂移

发生超时、非零退出、平移漂移、旋转误差或输出没有最终位姿时，代码会停止后续运动。
此时：

1. 保持右臂姿势和夹爪状态不变；
2. 不要直接运行旧的 `agv_step_rotate*`；
3. 记录终端输出、当前底盘位姿和错误码；
4. 由现场人员判断是否物理急停、人工恢复或重新标定；
5. 只有明确确认没有持管且路线安全后，才考虑下一次任务。

如果软件打印“仍为持管”，必须按仍夹着试管处理，即使肉眼暂时看不到试管。

### 12.7 任务中断后的右臂/夹爪状态

主流程在异常时会尽量停止机械臂和底盘；如果抓取命令已经发出，软件会保守地保留
`holding_tube` 状态。不要在这种状态下：

- 重新运行抓取命令；
- 移动底盘；
- 直接松爪；
- 使用不匹配当前设备的恢复脚本。

先由现场确认试管是否还在夹爪、右臂当前姿态、架面和障碍物，再决定人工取管或使用
经过审查的恢复动作。

## 13. 恢复工具边界

以下工具不是正常任务入口，参数与当前实验设备绑定，使用前必须阅读源码并进行现场
确认：

| 工具 | 用途 | 注意 |
|---|---|---|
| `tools/retreat_from_rack.py` | 空载张开夹爪时沿架面法向撤离约 150 mm | 会读取夹爪并要求 `RETREAT`，只适合明确空载情况 |
| `tools/retreat_current_tool_axis.py` | 沿当前工具轴反向直线撤离 | 会直接发送运动，必须先确认方向和障碍 |
| `tools/recover_pick_home.py` | 记录过的右臂抓取中断后回撤并回 home | 会清关节错误并运动，点位只适用于记录设备 |
| `tools/restore_after_mobile_transfer.sh` | 旧版底盘手动反向恢复脚本 | 当前文件仍是旧的 `0.20 m` 路线，不作为当前 `0.30 m` 流程使用 |

当前完整双架任务成功后不需要手动恢复脚本，主流程会在最终放置复扫成功后自动执行
底盘逆向回程。任何恢复工具都不会替代物理急停和现场判断。

## 14. GitHub、版本和机器人部署

### 14.1 本地提交前检查

```bash
git status --short --branch
git diff --check
python -m pytest
```

确认没有把以下内容加入提交：

- `models/*.pt` 私有权重；
- API key、密码、`.env`；
- `artifacts/` 中的临时图片和日志；
- `.pytest_cache`、`__pycache__`、训练输出；
- 机器人上的 SDK、备份、编译二进制和运行日志；
- 未经当前路线验证的旧恢复脚本。

提交和推送：

```bash
git add README.md config docs tests tube_grabber tools/pick_rotate_forward.sh
git diff --cached --stat
git commit -m "describe the change"
git push origin main
```

使用 `git add` 前应先审查未跟踪文件，不要无条件把整个目录的调试产物加入仓库。GitHub
推送完成后，机器人仍然运行原来的部署目录，除非另行执行发布和切换。

### 14.2 机器人更新后的验证

切换到新部署目录后至少核对：

```bash
readlink -f ~/tubeGrabber-mobile-current
cd ~/tubeGrabber-mobile-current
sha256sum \
  tube_grabber/mobile_transfer.py \
  tube_grabber/navigation/chassis.py \
  tube_grabber/cli.py \
  tools/pick_rotate_forward.sh
ls -lh models/cap.pt models/screw.pt
```

然后依次执行 `plan-mobile-transfer`、`--real doctor`、三条只读硬件检查，再考虑真机动作。
不要在任务运行时覆盖当前部署目录；推荐先准备新的隔离目录，完成检查后再切换链接。

## 15. 最短命令速查

### 本地开发机

```bash
cd <project-root>
python -m pip install -e ".[vision,realsense]"
python -m pytest
python -m tube_grabber doctor
python -m tube_grabber plan-mobile-transfer --auto-source --auto-destination
```

### 机器人只读检查

```bash
ssh rm@192.168.3.68
cd ~/tubeGrabber-mobile-current
conda activate tube_vision
python -m tube_grabber --real doctor
python -m tube_grabber --real arm-status
python -m tube_grabber --real camera-check
python -m tube_grabber --real gripper-status
```

### 当前右臂双架一键任务

```bash
cd ~/tubeGrabber-mobile-current
./tools/pick_rotate_forward.sh
```

任务完成条件是：目标架最终复扫确认成功、右臂保持 loaded home、底盘完成反向回程，
并且进程退出码为 `0`。任何失败都先停机检查，不要直接重复执行。

## 16. 相关文档

- `docs/ARCHITECTURE.md`：模块边界、视觉坐标和安全复扫逻辑；
- `docs/MOBILE_TRANSFER.md`：双架移动任务的状态顺序和底盘接口；
- `docs/LAB_CHECKLIST.md`：真机分阶段验收勾选表；
- `docs/USAGE.md`：从安装、模型、标定到同架任务的详细说明；
- `docs/MODEL_CONTRACT.md`：`cap.pt` 和 `screw.pt` 的类别契约；
- `docs/AGENT.md`：local/Gemini Agent 入口和限制；
- `docs/FIXED_TRANSFER.md`：固定点位取管到篮子流程；
- `docs/REPRODUCE_PICK_HOME_2026-09-09_CN.md`：当前右臂抓取回 home 的现场复现记录；
- `docs/LAB_LOG_2026-09-09.md`：实验过程和部署记录。
