# Tube Grabber

面向 2×6 试管架的视觉引导抓放主线。最终视觉方案只有一条：

- `cap.pt`：YOLO detection 识别试管盖，结合 D435 对齐深度得到盖顶三维坐标；
- `rack_pose.pt`：YOLO Pose 识别架面 8 点，顺序固定为
  `k0, k1, k2, k3, screw_k0, screw_k1, screw_k2, screw_k3`；
- `screw_k0` 旁的小红圆点独立确认架面方向；
- 连续多帧推理经过几何检查、异常帧剔除和中值聚合；
- RGB-D 架面 ROI 使用 RANSAC 拟合真实平面；
- 正上方标定时自动拟合 `r1c1`、`r2c6` 槽圆，人工微调圆心和半径后生成 12 个槽位；
- 槽位不再使用 YOLO detection，空槽由“没有盖子匹配到该标定槽位”确定。

## 当前状态

代码、fake 闭环和离线测试已经完成。架面 Pose 模型尚未训练，仓库也不提交
`.pt` 权重；真实精度、GPU 帧率、双圆心标定、手眼矩阵、TCP 和运动参数仍需在
部署机与真机上验收。没有模型或标定时 real 模式会明确拒绝运行，不会回退到旧的
孔位 detection 或 CPU 推理。

## 架构

```text
D435 连续 RGB-D 帧 + 右臂观察位姿
        │
        ├─ rack_pose.pt ─→ 8 点 + K0 红点方向检查
        │                  └→ 多帧异常值剔除/中值融合
        │
        ├─ 架面四角 ROI 深度 ─→ RANSAC 平面
        │
        └─ cap.pt ─→ 盖中心深度 ─→ 盖顶 base_right XYZ
                           │
双圆心标定 r1c1/r2c6 ─→ 12 槽投影 ─→ 盖子/槽位匹配
                           │
                    RackObservation
                           │
     自动观察位 → 最新抓取 → 高位目标复检/重规划
                           │
              放置 → 自动回观察位 → 最终复扫
```

边界保持单向：`vision` 只输出观测，`workflow` 只处理任务状态，`motion` 只处理
坐标和轨迹，`hardware` 是唯一调用 RealSense/RealMan SDK 的层。项目内部位置统一
为 mm、角度为 rad、工作坐标系为 `base_right`；只有机械臂驱动边界进行 mm↔m。

详细设计见 [架构说明](docs/ARCHITECTURE.md)，模型与标注见
[模型契约](docs/MODEL_CONTRACT.md)。从安装、训练、标定到真机分阶段执行的完整操作流程见
[详细使用说明](docs/USAGE.md)。底盘站点、跨架状态机、失败恢复、麦克风选择和离线语音方案见
[底盘导航与语音控制完整实施方案](docs/NAVIGATION_AND_VOICE.md)。

## 安装与 GPU

Python 要求 ≥3.10。真实部署应先按本机 CUDA/驱动版本安装 CUDA 版 PyTorch，再安装：

```bash
python -m pip install -e ".[vision,realsense]"
```

RealMan SDK 需要由厂商安装并可导入：

```bash
python -c "from Robotic_Arm import rm_robot_interface; print('RealMan SDK OK')"
```

`config/app.yaml` 的最终推理设备固定为 `"0"`（CUDA:0）。`doctor` 会检查 CUDA
版 PyTorch、GPU、两个模型、两份槽位标定、手眼、观察位和运动参数；real 模式下
任一缺失都会失败，不静默使用 CPU。

## 模型制作

1. 在 `screw_k0` 旁贴好永久小红圆点。
2. 采集架面图片：

```bash
python tools/capture_rack_pose_dataset.py --count 500
```

3. 按 [8 点标注契约](training/RACK_POSE_CONTRACT.md) 标注，并按采集序列划分
   train/val/test。
4. 复制 `training/rack_pose.yaml.example` 为 `training/rack_pose.yaml`，填写数据路径。
5. 用本机 GPU 训练并验收：

```bash
python tools/train_rack_pose.py --data training/rack_pose.yaml
python tools/validate_rack_pose.py --data training/rack_pose.yaml --weights models/rack_pose.pt
```

最终放置：

```text
models/cap.pt
models/rack_pose.pt
```

## 正上方双圆心标定

先清空 `r1c1`、`r2c6`，低速手动将腕部相机调整到架面正上方并保持机械臂静止。
标定视角可以不同于正常运行观察位。真实配置和模型就绪后分别运行：

```bash
python -m tube_grabber calibrate-rack --rack rack_1
python -m tube_grabber calibrate-rack --rack rack_2
```

程序先采集 7 帧并筛出至少 5 个稳定帧，然后显示融合后的 K0～K3、四个螺丝孔和
红点方向结果。标定操作如下：

1. 在 `r1c1` 槽圆附近点击，程序用 Hough 圆拟合给出初始圆；未检测到时使用默认圆。
2. 按住鼠标左键拖动圆心；用 `[`/`]` 或 `-`/`+` 调半径；方向键或 `I/J/K/L`
   每次微调圆心 1 px。
3. 圆周贴合槽边后按 Enter 或空格确认，随后以相同方式确认 `r2c6`。
4. 检查绿色 2×6 网格中心均落在槽中心后按 `s` 保存。Backspace 撤回，`r` 全部
   重置，`q` 取消。

最终使用的是两个调整后圆的圆心；半径仅用于帮助准确贴合槽边。标定写入
`config/racks/rack_1.yaml` 或 `rack_2.yaml`，覆盖必须显式使用 `--force`。

两个圆心和 8 个关键点都会转换到架面归一化坐标，因此正上方标定后可以回到正常
观察位运行。参考检查比较的是跨视角不变的归一化布局，而不是标定图片的绝对像素。
模型关键点语义、机架结构或相机内参改变后应重新标定；相机安装变化还必须重做手眼。
旧的绝对像素标定格式不会被接受，应使用 `--force` 重新生成当前标定。

## 运行命令

```bash
python -m tube_grabber doctor
python -m tube_grabber arm-status
python -m tube_grabber camera-check
python -m tube_grabber gripper-status
python -m tube_grabber scan --rack rack_1
python -m tube_grabber plan-transfer --source rack_1.r1c1 --destination rack_1.r2c6
python -m tube_grabber transfer --source rack_1.r1c1 --destination rack_1.r2c6
```

默认 `runtime.mode: fake`。真实运动还要求观察位与运动参数已确认、控制器为真实模式且
上电、控制器和关节无错误、没有 `atom/zhixing_ctrl.py` 抢占控制。`transfer` 先要求
确认物理空载并输入 `EMPTY`，然后自动进入旧 AprilTag 流程的全局观察位；预览后输入
`MOVE`。执行前复扫会重建抓取计划，携管时在目标高位走廊复检并重规划放置，释放后
自动回观察位，最终确认源空、目标占用才成功。`plan-transfer` 保持零运动承诺，因此
运行前仍需人工将右臂置于观察位。

夹爪按当前 RealMan 两指夹爪接入：只使用两指夹爪专用的
`rm_set_gripper_position` 与 `rm_get_gripper_state`；不会调用六自由度灵巧手的
`rm_set_hand_follow_pos`。阻塞命令返回后还会连续检查在线、使能、错误码、工作模式
和实际开度。

## 测试

```bash
python -m unittest discover -s tests -v
```

离线测试覆盖模型契约、K0 红点、关键点几何、多帧滤波、双点标定、架面 RANSAC、
完整 Pose+cap+槽位流水线、坐标变换、最新坐标重规划、观察位往返、最终状态验证、
运动规划、硬件 SDK 翻译和 fake 抓放闭环。

首次真机必须按 [实验室检查清单](docs/LAB_CHECKLIST.md) 从只读检查逐级推进。
