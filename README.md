# Tube Grabber

基于 Intel RealSense D435、Ultralytics YOLO 和 RealMan 七轴机械臂的 2×6 试管架视觉引导抓放系统。

项目提供从 RGB-D 感知、试管架标定、槽位状态判断、三维定位、运动规划到抓放后复扫验证的完整同架闭环，同时支持可选的自然语言命令解析。

> [!CAUTION]
> 这是会驱动真实机械臂的实验项目，不是安全认证系统。真机运行必须保证急停可触达、底盘锁定、机械臂周围无人且无障碍，并由操作人员低速监护。不要直接复用仓库中的手眼、TCP、观察位或工作空间参数到另一台设备。

## 当前状态

| 项目 | 状态 |
|---|---|
| D435 对齐 RGB-D 采集 | 已实现 |
| cap / screw 双 YOLO Detection | 已实现，真机固定使用 CUDA:0 |
| 2×6 槽位标定与占用判断 | 已实现 |
| RealMan 七轴机械臂与 RM Plus 两指夹爪 | 已接入 |
| 同一试管架内抓放与最终复扫 | 已实现 |
| 本地 / Gemini 文本命令解析 | 已实现 |
| 单个 rack_1 现场标定 | 已包含；默认配置不要求第二个试管架 |
| 双试管架固定路线搬运 | 已实现独立入口，真机参数默认未确认 |
| 语音输入与播报 | 未实现 |

项目版本：<code>0.1.0</code>。

## 核心能力

- 使用 <code>screw.pt</code> 检测四颗角螺丝，以检测框中心定义架面；
- 使用额外白色 marker 确定物理 K0，避免试管架方向翻转；
- 使用 <code>cap.pt</code> 检测试管盖，并结合对齐深度计算盖顶三维坐标；
- 对连续多帧执行几何校验、异常帧剔除、中值融合与稳定性门禁；
- 在架面 ROI 内使用 RANSAC 拟合真实平面，支持轻微倾斜的试管架；
- 通过 <code>r1c1</code> 与 <code>r2c6</code> 双圆心标定生成完整 2×6 槽位；
- 通过盖子与标定槽位的一对一匹配判断 <code>OCCUPIED</code> / <code>EMPTY</code>；
- 统一使用 mm、rad 和 <code>base_right</code> 坐标系，只在 SDK 边界进行单位转换；
- 规划前检查占用、工作空间、单段距离、工具方向、管体净空和控制器状态；
- 执行前重新扫描并用最新坐标重建抓取计划；
- 放置后自动返回观察位，最终复扫确认源槽为空、目标槽占用。

## 工作原理

~~~text
D435 连续 RGB-D 帧 + 右臂法兰位姿
        │
        ├─ screw.pt ─→ 四颗螺丝中心 ─→ 白色 marker 定向 K0～K3
        │                                  │
        │                                  └─ 多帧过滤与架面几何检查
        │
        ├─ 架面深度 ROI ─→ RANSAC 平面 ─→ 12 个槽位三维坐标
        │
        └─ cap.pt ─→ 盖中心 + 对齐深度 ─→ 盖顶三维坐标
                                                   │
r1c1 / r2c6 双圆心标定 ─→ 2×6 网格 ─→ 槽位占用匹配
                                                   │
                                            RackObservation
                                                   │
                              规划 → 抓取 → 放置 → 最终复扫
~~~

代码保持单向分层：

| 模块 | 职责 |
|---|---|
| <code>core</code> | 坐标、槽位、观测对象、错误类型和硬件接口 |
| <code>vision</code> | YOLO、marker、深度、架面、标定与多帧融合 |
| <code>workflow</code> | 任务状态、占用校验、复扫与抓放顺序 |
| <code>motion</code> | TCP/法兰变换、航点规划和运动门禁 |
| <code>hardware</code> | RealSense、RealMan 和夹爪 SDK 适配 |
| <code>agent</code> | 本地或 Gemini 文本命令解析 |
| <code>app / cli</code> | 依赖组装、生命周期、人工确认和命令入口 |

更详细的设计见 [架构说明](docs/ARCHITECTURE.md)。2026-09-09 的真机抓取参数、
闭环结果和续接事项见 [实验日志](docs/LAB_LOG_2026-09-09.md)。

## 硬件与软件要求

### 硬件

- Intel RealSense D435 腕部相机；
- RealMan 七自由度机械臂；
- RM Plus ZX 1DOF 两指夹爪；
- NVIDIA CUDA GPU；
- 带四颗可见角螺丝和 K0 白色 marker 的 2×6 试管架。

### 软件

- Python 3.10 或更高版本；
- 与本机驱动/CUDA 匹配的 CUDA 版 PyTorch；
- Ultralytics 8.3.x；
- Intel RealSense Python SDK；
- RealMan 厂商 Python SDK，导入名为 <code>Robotic_Arm</code>；
- 有图形界面的桌面会话，用于实时预览和交互标定。

## 安装

### 1. 获取项目并创建环境

~~~bash
git clone <your-repository-url>
cd tubeGrabber

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
~~~

Windows 激活虚拟环境时使用：

~~~powershell
.venv\Scripts\Activate.ps1
~~~

机器人本机已经配置了 Conda 环境 <code>tube_vision</code>，在机器人上运行项目时可以直接激活：

~~~bash
conda activate tube_vision
~~~

激活后应在项目根目录执行后续安装、检查和运行命令。

### 2. 安装项目基础依赖

~~~bash
python -m pip install -e .
~~~

### 3. 真机视觉依赖

先按照部署机的 NVIDIA 驱动与 CUDA 版本安装对应的 CUDA 版 PyTorch，再安装：

~~~bash
python -m pip install -e ".[vision,realsense]"
~~~

检查 CUDA：

~~~bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
~~~

RealMan SDK 由厂商环境提供，不在本项目依赖中：

~~~bash
python -c "from Robotic_Arm import rm_robot_interface; print('RealMan SDK OK')"
~~~

如需 Gemini 自然语言解析：

~~~bash
python -m pip install -e ".[agent]"
~~~

也可以使用 <code>python -m pip install -r requirements.txt</code> 一次安装项目声明的全部第三方依赖；PyTorch 和 RealMan SDK 仍需根据部署机单独安装。

## 模型准备

真实视觉需要两个 Ultralytics YOLO Detection 权重：

~~~text
models/
├── cap.pt      # 试管盖
└── screw.pt    # 四颗架面角螺丝
~~~

两个模型的元数据类别必须都是：

~~~yaml
names:
  0: item
~~~

运行时适配层会分别将 <code>item</code> 转换为 <code>tube_cap</code> 和 <code>screw</code>。验证模型：

~~~bash
python -c "from ultralytics import YOLO; print(YOLO('models/cap.pt', task='detect').names)"
python -c "from ultralytics import YOLO; print(YOLO('models/screw.pt', task='detect').names)"
python tools/test_yolo_stream.py
~~~

<code>test_yolo_stream.py</code> 只打开 D435 彩色流并运行两个模型，不连接机械臂或夹爪。按 <code>q</code> 或 <code>Esc</code> 退出。

权重默认被 <code>.gitignore</code> 排除，不会随普通 <code>git push</code> 上传。需要分发权重时，请确认数据和模型许可证允许公开，再选择 Git LFS 或 GitHub Release；否则在 README/Release 中提供合法下载方式和校验值。完整契约见 [模型契约](docs/MODEL_CONTRACT.md) 和 [模型目录说明](models/README.md)。

## 真机部署

### 1. 先复核配置

主要配置位于：

- <code>config/app.yaml</code>：运行模式、硬件、模型、视觉、几何和运动门限；
- <code>config/hand_eye.yaml</code>：<code>camera_rightwrist → end_right</code> 手眼矩阵；
- <code>config/poses.yaml</code>：固定观察位；
- <code>config/racks/rack_1.yaml</code>：rack_1 槽位与展示角点标定。

新设备部署时，先不要连接或驱动真机，并将以下两个确认锁设为 <code>false</code>：

~~~yaml
# config/poses.yaml
observation_pose:
  confirmed: false

# config/app.yaml
motion:
  parameters_confirmed: false
~~~

仓库中已有的相机序列号、机械臂 IP、手眼矩阵、TCP、观察位、工作空间和确认状态属于当前实验设备。只有重新测量并低速验证后，才可在目标设备上确认。

### 2. 切换 real 并做只读预检

完成配置复核后，将 <code>runtime.mode</code> 显式改为 <code>real</code>。先运行：

~~~bash
python -m tube_grabber doctor
python -m tube_grabber arm-status
python -m tube_grabber camera-check
python -m tube_grabber gripper-status
~~~

- <code>doctor</code> 检查配置、模型契约、CUDA、SDK、手眼、已配置 rack 的标定和运动确认锁；
- <code>arm-status</code> 只读取右臂法兰位姿、真实/仿真模式、电源和健康状态；
- <code>camera-check</code> 采集彩色图和 16 位毫米深度图到 <code>artifacts/</code>；
- <code>gripper-status</code> 只读取 RM Plus 状态，不发送夹爪位置命令。

默认配置只有 <code>rack_1</code>，因此 real 模式只要求
<code>config/racks/rack_1.yaml</code>。普通篮筐不是试管架，不应加入
<code>racks</code> 或伪造槽位标定。

### 3. 标定试管架

每个 rack 都需要单独标定：

1. 清空 <code>r1c1</code> 和 <code>r2c6</code>；
2. 低速把腕部相机移到架面正上方并保持机械臂、试管架静止；
3. 运行标定命令；
4. 点击槽圆附近获得 Hough 初值，拖动圆心并用按键调整；
5. 检查完整 2×6 网格后保存。

~~~bash
python -m tube_grabber calibrate-rack --rack rack_1
~~~

已有标定必须显式覆盖：

~~~bash
python -m tube_grabber calibrate-rack --rack rack_1 --force
~~~

标定窗口操作：

- 鼠标左键：选择或拖动圆心；
- <code>[</code> / <code>]</code> 或 <code>-</code> / <code>+</code>：调整半径；
- 方向键或 <code>I/J/K/L</code>：每次微调 1 px；
- Enter / Space：确认当前圆；
- <code>s</code>：保存；
- Backspace：撤回；
- <code>r</code>：重置；
- <code>q</code>：取消。

展示用 K0～K3 四角可以独立微调，不改变槽锚点：

~~~bash
python -m tube_grabber calibrate-corners --rack rack_1
~~~

详细步骤、重标定条件和现场验收方法见 [完整使用说明](docs/USAGE.md)。

### 4. 扫描与规划验收

先将右臂低速移动到已确认观察位：

~~~bash
python tools/move_to_observation_pose.py
~~~

再进行稳定扫描和零运动规划：

~~~bash
python -m tube_grabber scan --rack rack_1

python -m tube_grabber plan-transfer \
  --source rack_1.r1c1 \
  --destination rack_1.r1c2
~~~

<code>scan</code> 默认打开实时窗口；无桌面环境时添加 <code>--no-display</code>。<code>plan-transfer</code> 不初始化夹爪、不发送机械臂运动，但在 real 模式下会连接机械臂和相机，并要求右臂已经位于观察位。

### 5. 执行抓放

~~~bash
python -m tube_grabber transfer \
  --source rack_1.r1c1 \
  --destination rack_1.r1c2
~~~

当前真机流程：

1. 检查右臂、夹爪、控制器、冲突进程和运动参数；
2. 操作者确认 TCP 空载、路径无障碍，按 Enter 允许自动进入观察位；
3. 执行稳定扫描并打印观测和航点预览；
4. 操作者再次按 Enter 授权抓放；
5. 执行前复扫，并使用最新坐标重建抓取计划；
6. 按配置逐步确认张开、下降、夹紧、抬升、平移、放置和撤离；
7. 自动返回观察位并复扫，只有源空、目标占用且其他槽位无异常才报告成功。

任一人工确认处都可输入 <code>q</code> 取消。当前 <code>destination_recheck_while_carrying: false</code>：腕部相机在持管目标高位无法看到全部四颗螺丝，因此放置复用抓取前完整扫描的目标坐标，释放后再进行完整闭环复扫。

## 命令参考

| 命令 | 作用 | 是否可能运动 |
|---|---|---|
| <code>doctor</code> | 静态检查配置、依赖、模型和标定 | 否 |
| <code>arm-status</code> | 读取机械臂状态与法兰位姿 | 否 |
| <code>camera-check</code> | 保存一帧彩色图和深度图 | 否 |
| <code>gripper-status</code> | 读取夹爪状态 | 否 |
| <code>scan --rack ...</code> | 实时预览并执行稳定扫描 | 否 |
| <code>calibrate-rack --rack ...</code> | 标定两个对角槽位 | 否，但要求人工预先摆位 |
| <code>calibrate-corners --rack ...</code> | 微调展示四角 | 否 |
| <code>plan-transfer</code> | 扫描并打印完整航点 | 否 |
| <code>transfer</code> | 执行同架抓放闭环 | 是 |
| <code>agent-plan</code> | 解析文本并走只读规划流程 | 否 |
| <code>agent-transfer</code> | 解析文本并走完整抓放流程 | 是 |
| <code>plan-fixed-transfer</code> | 校验固定取管到篮子的完整路径，不加载视觉 | 否 |
| <code>fixed-transfer</code> | 执行已确认的固定取管到篮子循环 | 是 |
| <code>plan-mobile-transfer</code> | 校验双架视觉闭环和底盘路线配置 | 否 |
| <code>mobile-transfer</code> | 执行 rack_1 到 rack_2 的跨站视觉闭环 | 是 |

查看全部参数：

~~~bash
python -m tube_grabber --help
python -m tube_grabber <command> --help
~~~

安装为可编辑包后，也可以将 <code>python -m tube_grabber</code> 替换为 <code>tube-grabber</code>。

## 固定取管到篮子

如果取管点和篮子位置始终固定，可以使用不依赖相机、YOLO 和机架标定的独立流程：

~~~bash
python -m tube_grabber plan-fixed-transfer
python -m tube_grabber fixed-transfer
~~~

点位模板位于 <code>config/fixed_transfer.yaml</code>，默认包含空值且
<code>confirmed: false</code>，因此不能直接驱动真机。必须先在当前机械臂与当前
<code>Arm_Tip</code> 工具坐标系下示教五个位姿，完成低速验收后再解除确认锁。完整步骤见
[固定取管到篮子](docs/FIXED_TRANSFER.md)。

## 双试管架移动

同架任务继续使用 `transfer`。固定布置下的 `rack_1 -> rack_2` 任务使用：

~~~bash
python -m tube_grabber plan-mobile-transfer \
  --source rack_1.r1c1 --destination rack_2.r1c2
python -m tube_grabber mobile-transfer \
  --source rack_1.r1c1 --destination rack_2.r1c2
python -m tube_grabber mobile-pick-home --auto-source
~~~

该流程抓取后先回带管 home，确认源槽为空，再让 Woosh 底盘闭环旋转 180 度并
沿车体 X 平移实测距离。到达后重新识别 `rack_2`，不复用移动前的三维坐标；释放后
再次复扫目标架。默认配置缺少距离 X、`rack_2` 标定且确认锁关闭，因此不能直接驱动
真机。完整配置和验收步骤见 [双试管架移动闭环](docs/MOBILE_TRANSFER.md)。

## Agent 命令

默认 <code>local</code> 解析器完全离线，要求文本中恰好包含两个标准槽位地址：

~~~bash
python -m tube_grabber agent-plan \
  --text "rack_1.r1c1 -> rack_1.r1c2"
~~~

Gemini 只负责把自然语言转换为结构化 <code>TransferCommand</code>，不会获得硬件控制函数。API key 必须通过环境变量提供：

~~~bash
export GEMINI_API_KEY="your-key"

python -m tube_grabber agent-plan \
  --agent-provider gemini \
  --text "把一号架第一行第一列的试管移动到一号架第一行第二列"
~~~

不要把密钥写入 YAML、Python、日志或 Git。无论使用哪个 Agent，Python 端都会重新检查槽位范围、源目标关系、占用、视觉、规划、同架限制和人工确认。详见 [Agent 接入说明](docs/AGENT.md)。

## 关键配置

| 配置路径 | 含义 |
|---|---|
| <code>runtime.mode</code> | 运行模式；连接真机时使用 <code>real</code> |
| <code>runtime.require_enter_before_motion</code> | 观察位和抓放前人工确认 |
| <code>runtime.destination_recheck_while_carrying</code> | 是否在持管目标高位重新扫描 |
| <code>camera</code> | 序列号、分辨率、帧率、深度范围 |
| <code>arm</code> | IP、端口、坐标系、速度和冲突进程 |
| <code>gripper</code> | RM Plus 波特率、电压、开合位置、力和超时 |
| <code>vision.cap / screw</code> | 权重、类别、置信度、IoU 和输入尺寸 |
| <code>vision.stability</code> | 多帧数量、内点数量和抖动门限 |
| <code>vision.plane</code> | RANSAC 和架面法向门限 |
| <code>geometry</code> | 手眼、TCP、盖高、抓取深度、管长和净空 |
| <code>motion</code> | 确认锁、速度、航点高度、工作空间和到位误差 |

修改 TCP 时可以使用：

~~~bash
python tools/tune_tcp.py --show
python tools/tune_tcp.py --dry-run --delta 0 0 1
~~~

该工具一旦写入新的 TCP，会自动把 <code>motion.parameters_confirmed</code> 重新锁为 <code>false</code>。<code>tools/retreat_from_rack.py</code> 是仅供现场异常恢复的 150 mm 法向撤离工具，不属于正常任务流程，使用前必须阅读源码并确认当前夹爪空载。

## 项目结构

~~~text
.
├── config/                 # 应用、手眼、观察位和 rack 标定
├── docs/                   # 架构、使用、模型、Agent 和真机清单
├── models/                 # 本地 YOLO 权重（默认不提交）
├── artifacts/              # 运行时调试图（默认不提交）
├── tests/                  # 离线单元与集成测试
├── tools/                  # 真机检查、观察位、TCP 与恢复工具
├── tube_grabber/
│   ├── agent/
│   ├── core/
│   ├── hardware/
│   ├── motion/
│   ├── vision/
│   └── workflow/
├── pyproject.toml
└── requirements.txt
~~~

## 测试

运行全部离线测试：

~~~bash
python -m unittest discover -s tests -v
~~~

测试覆盖：

- 槽位地址、占用状态和命令解析；
- screw 四角排序、marker 定向和几何拒绝；
- 多帧过滤、架面 RANSAC、深度与坐标变换；
- 双圆心标定、2×6 网格和盖子/槽位匹配；
- TCP/法兰转换、倾斜架面运动规划和工作空间门禁；
- RealSense、RealMan 与 RM Plus SDK 返回值适配；
- 执行前现场变化、异常持管状态和最终复扫失败路径。

单元测试不能证明真机识别精度或运动安全。第一次实机运行必须按 [真机分阶段检查清单](docs/LAB_CHECKLIST.md) 从只读检查逐级推进。

## 常见问题

### doctor 报 CUDA 不可用

确认 PyTorch 是 CUDA 构建、NVIDIA 驱动可见，并且 <code>vision.device</code> 指向存在的 GPU。real 模式不会静默回退 CPU。

### 模型契约错误

确认两个权重都是 Detection 模型，且 <code>model.names == {0: 'item'}</code>。旧的 YOLO Pose / <code>rack_pose.pt</code> 不属于当前运行链。

### 命令误用了 rack_2

默认配置只有 <code>rack_1</code>。使用 <code>rack_2</code> 的扫描、标定或搬运命令会被明确拒绝；不要复制 <code>rack_1</code> 的标定文件冒充第二个架子。普通篮筐请使用独立的固定取放流程。

### 扫描不稳定

检查四颗螺丝是否完整可见、白色 marker 是否唯一、反光与遮挡、相机是否静止、深度是否有效，再根据现场统计调整置信度和稳定性门限。

### 无法打开 OpenCV 窗口

从桌面图形会话运行；只做一次稳定扫描时可添加 <code>scan --no-display</code>。

### 机械臂或夹爪预检失败

确认控制器处于真实模式且已上电、七轴和 Base/Arm_Tip 坐标系正确、RM Plus 使用 9600 波特率和工具端 24 V，并停止 <code>atom</code>、<code>zhixing_ctrl.py</code> 等冲突进程。

### 普通 transfer 的跨架任务被拒绝

这是预期行为。`transfer` 只支持同架；经过独立配置和验收的固定双架路线必须使用
`mobile-transfer`。

## GitHub 发布前检查

- 运行完整离线测试并确认通过；
- 检查 <code>git status</code>，不要提交 <code>artifacts/</code>、缓存、数据集或训练输出；
- 确认 <code>GEMINI_API_KEY</code> 等凭据未进入历史、配置或日志；
- 决定模型权重的合法分发方式；普通 Git 提交不会包含 <code>models/*.pt</code>；
- 检查设备 IP、相机序列号、标定和运动参数是否适合公开；
- 如希望他人复制、修改或分发项目，请在发布前添加明确的 <code>LICENSE</code>。

## 文档索引

- [完整使用说明](docs/USAGE.md)
- [架构说明](docs/ARCHITECTURE.md)
- [模型契约](docs/MODEL_CONTRACT.md)
- [Agent 接入说明](docs/AGENT.md)
- [真机分阶段检查清单](docs/LAB_CHECKLIST.md)
- [双试管架移动闭环](docs/MOBILE_TRANSFER.md)

---

连接真实设备前，请先完成整份真机检查清单。
