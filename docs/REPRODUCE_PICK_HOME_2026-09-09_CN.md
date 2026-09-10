# 2026-09-09 右臂视觉抓管实验复现手册

本文记录 2026-09-09 在真实机器人上完成并重复验证的第一阶段任务：机械臂从
home 出发，自动识别试管架中的唯一试管，抓取并夹紧，沿试管架法向回撤，最后
返回带管 home。本文不包含底盘移动、第二个试管架放置或篮子放置。

## 1. 已验证结果

完整闭环为：

```text
空载 home
  -> D435 采集 RGB-D
  -> screw.pt 定位四颗角螺丝并确定试管架坐标
  -> cap.pt 检测唯一试管
  -> 自动选择槽位
  -> 张开夹爪
  -> 到达试管上方并直线下探
  -> 夹紧试管
  -> 沿试管架法向回撤 150 mm
  -> 返回 loaded home
  -> 复扫源架，确认原槽位为空
```

当天完整成功的槽位包括：

| 槽位 | 结果 |
|---|---|
| `rack_1.r1c5` | 抓取、夹紧、返回 home、空槽复扫成功 |
| `rack_1.r1c2` | 连续动作成功，空槽复扫成功 |
| `rack_1.r2c4` | 连续动作成功，空槽复扫自动重试一次后成功 |

`rack_1.r1c1` 曾在夹紧后被旧工作区 X 下限拦截。实测回撤目标约为
`X=-53.14 mm`，随后用恢复脚本完成回撤和回 home。当前下限已扩大为 `-80 mm`，
因此该问题已写入配置修正。

复现代码基线：

```text
仓库：https://github.com/Logic-LR/tubeGrabber-fixed-transfer
分支：main
提交：93c3ca7d9576dec54337d0836162fee4835c0789
```

## 2. 运行环境

### Windows 开发机

```text
项目目录：D:\Docu\Demo\tubeGrabber
```

### 机器人

```text
SSH 地址：rm@192.168.3.68
当前部署链接：/home/rm/tubeGrabber-mobile-current
实际部署目录：/home/rm/deployments/tubeGrabber-mobile-20260909_173721
Conda：/home/rm/miniconda3
运行环境：tube_vision
右臂控制器：169.254.128.19:8080
```

密码不写入仓库、脚本或本文档。

机器人部署目录需要包含两个权重：

```text
models/cap.pt
models/screw.pt
```

`cap.pt` 检测试管盖，`screw.pt` 检测试管架四颗角螺丝。两者不是互相替代的
模型，会由同一个视觉运行时同时加载。

## 3. 当前实机参数

参数来源为 [config/app.yaml](../config/app.yaml)、
[config/poses.yaml](../config/poses.yaml) 和当天实机修正。

| 参数 | 当前值 | 含义 |
|---|---:|---|
| home XYZ | `[89.888, 284.810, 135.422]` mm | 空载观察位和带管返回位 |
| home RPY | `[-2.310, 0.024, -1.554]` rad | home 法兰姿态 |
| pick TCP | `[-3.000, -11.000, 220.000]` mm | 已包含机器人自身右侧 3 mm 修正 |
| 管帽标定高度 | `47.0 mm` | 反光深度异常时的回退高度 |
| 抓取深度 | `28.0 mm` | 从管帽顶向下的夹取深度 |
| approach 高度 | `50.0 mm` | 下探前的架面法向高度 |
| retreat 距离 | `150.0 mm` | 夹紧后的法向回撤距离 |
| 工作区 X | `[-80, 250] mm` | 已覆盖 `r1c1` 回撤轨迹 |
| 工作区 Y | `[270, 435] mm` | 当前右臂实验区 |
| 工作区 Z | `[-110, 200] mm` | 当前右臂实验区 |

视觉运行参数：

- 管帽置信度阈值为 `0.10`，螺丝置信度阈值为 `0.40`。
- 每轮最多采集 12 帧，只要有 1 个完整帧即可继续。
- 自动模式必须识别到恰好一支试管；零支或多支时持续重新采样，不发送抓取动作。
- D435 在反光管帽上返回异常深度时，保留 YOLO 的占用判断，但使用标定的
  `47 mm` 管帽高度计算抓取点。
- K0 白色标记未识别时，当前版本使用图像左上角螺丝作为临时 K0。
- 抓取后的闭环复扫失败会在同一进程中继续轮询。

运动运行参数：

- 执行 `mobile-pick-home` 命令即视为运动授权，不再等待 `PICK-HOME` 输入，
  之后也不逐航点等待人工输入。
- 抓取前采用一次成功视觉结果，不再要求第二轮场景一致性扫描。
- 不做每个航点完成后的毫米级位姿比较，保证动作连续。
- RealMan SDK 的运动调用仍为阻塞调用；控制器、SDK 或通信明确报错仍会停止。
- 第 7 关节在新 SDK 会话中可能报告 `0xF000`，入口会在同一会话自动清除。

## 4. 开机后的复现步骤

### 4.1 放置实验对象

1. 将 `rack_1` 放回标定时的位置和方向，K0 白色标记位于相同角。
2. 保证右臂 D435 能完整看到四颗角螺丝、12 个槽位和 K0 区域。
3. 在 12 个槽位中只放一支试管。
4. 确认夹爪中没有上一次实验遗留的试管。
5. 机械臂应从已示教的 home 附近开始。

### 4.2 从 Windows 登录机器人

```powershell
ssh rm@192.168.3.68
```

如果提示 `Connection timed out`，说明尚未连通机器人 SSH。此时机器人没有收到
任何动作命令，应先检查机器人是否开机、开发机是否接入 `192.168.3.x` 网段，再重试。

### 4.3 确认部署和权重

以下命令在机器人终端执行：

```bash
readlink -f /home/rm/tubeGrabber-mobile-current
cd /home/rm/tubeGrabber-mobile-current
test -f models/cap.pt && echo "cap.pt OK"
test -f models/screw.pt && echo "screw.pt OK"
```

预期部署链接解析到：

```text
/home/rm/deployments/tubeGrabber-mobile-20260909_173721
```

### 4.4 启动完整抓取流程

在机器人终端执行：

```bash
cd /home/rm/tubeGrabber-mobile-current
env PYTHONPATH=$PWD /home/rm/miniconda3/bin/conda run \
  -n tube_vision --no-capture-output \
  python -m tube_grabber --real mobile-pick-home --auto-source
```

命令启动后会直接执行。运行命令前必须由现场人员确认右臂空载、路径无人无障碍且
急停可触达。

从 Windows 直接一条命令启动也可以：

```powershell
ssh -tt rm@192.168.3.68 "env PYTHONPATH=/home/rm/tubeGrabber-mobile-current /home/rm/miniconda3/bin/conda run -n tube_vision --no-capture-output --cwd /home/rm/tubeGrabber-mobile-current python -m tube_grabber --real mobile-pick-home --auto-source"
```

该命令不再等待额外输入。

## 5. 如何判断执行成功

进入动作前应看到类似输出：

```text
已清除第 7 关节错误 0xF000
自动选择源槽：rack_1.r2c4
```

关节没有错误时，不一定出现“已清除”一行。完整成功的结尾是：

```text
抓取回 home 完成：rack_1.r2c4 已确认空；机械臂位于 loaded observation/home，夹爪保持夹紧。
```

成功后的物理状态应同时满足：

- 机械臂位于 loaded home；
- 夹爪保持闭合并持有试管；
- 原槽位为空；
- 进程正常退出；
- `artifacts/` 下生成新的 `scan_rack_1_*.png` 标注图。

2026-09-09 保存的四张标注图位于
[artifacts/2026-09-09](../artifacts/2026-09-09/README.md)，其中最后两张对应
`r1c2` 和 `r2c4` 的连续闭环成功结果。

## 6. 自动重试时的正常现象

以下输出表示视觉正在自动重试，不代表脚本已经停止：

```text
抓取前定位失败，立即重新采样：...
唯一试管判定失败，立即重新采样：...
抓取后闭环复扫失败，立即重新采样：right arm moved across the multi-frame inference window
```

前两类发生在运动前，不会重复发送机械臂动作。第三类发生在机械臂已经回到 loaded
home 之后，只会重新采集图像，不会再次抓取。若画面恢复稳定，流程会自行继续。

## 7. 中断后的恢复

先根据物理状态判断，不要在夹爪已经持管时再次启动完整抓取命令。

### 情况 A：尚未夹紧

如果脚本在识别阶段退出，机械臂通常仍在 home，可以排除网络或画面问题后重新执行
完整命令。如果机械臂已经下探但夹爪未夹紧，先沿当前工具轴直线抬升：

```bash
cd /home/rm/tubeGrabber-mobile-current
env PYTHONPATH=$PWD /home/rm/miniconda3/bin/conda run \
  -n tube_vision --no-capture-output \
  python tools/retreat_current_tool_axis.py --distance-mm 150
```

该脚本不操作夹爪。

### 情况 B：已夹紧但没有返回 home

先执行上述 150 mm 工具轴直线回撤，再运行：

```bash
cd /home/rm/tubeGrabber-mobile-current
env PYTHONPATH=$PWD /home/rm/miniconda3/bin/conda run \
  -n tube_vision --no-capture-output \
  python tools/recover_pick_home.py
```

恢复流程不会松爪，完成后仍是带管 home。两个恢复脚本只适用于本文记录的右臂、
工具坐标系和当前实验布置。

### 情况 C：只是闭环复扫反复失败

此时机械臂已经在 loaded home 且夹爪持管。可以等待视觉自动恢复，或用 `Ctrl+C`
结束复扫；不要再次发送抓取命令。人工取下试管后，下一轮实验才能重新开始。

## 8. 代码对应关系

| 文件 | 作用 |
|---|---|
| `tube_grabber/cli.py` | `mobile-pick-home` 命令入口和一次性确认 |
| `tube_grabber/mobile_transfer.py` | 唯一试管选择、视觉重试、抓取后回 home |
| `tube_grabber/vision/rack_pose.py` | 四螺丝定位、K0 方向处理 |
| `tube_grabber/vision/pose_pipeline.py` | 多帧视觉、深度和平面、管帽高度回退 |
| `tube_grabber/workflow/manipulation.py` | 张爪、下探、夹紧、回撤的状态机 |
| `tube_grabber/motion/executor.py` | 连续航点执行和工作区约束 |
| `tube_grabber/hardware/realman_arm.py` | RealMan SDK、状态读取和关节清错 |
| `config/app.yaml` | 相机、模型、几何与运动参数 |
| `config/mobile_transfer.yaml` | loaded home 和未来底盘路线配置 |

## 9. 尚未完成的部分

昨天只完成了“第一架取管并回 home”。以下内容仍未实机完成：

1. `rack_2` 的独立槽位标定；
2. 底盘旋转 180 度的完整实机闭环；
3. 底盘沿车体 X 移动的实测距离；
4. 到达第二架后的重新识别和放置；
5. 完整的 `rack_1 -> rack_2` 连续任务；
6. 试管架到篮子的版本。

`config/mobile_transfer.yaml` 中的 `translation_x_m` 仍为 `null`，顶层
`confirmed` 仍为 `false`。在完成第二架标定和底盘距离实测前，应保持这两个阻断条件。

## 10. 昨天关机前的最终状态

最后一次成功实验自动选择了 `rack_1.r2c4`，完成抓取、夹紧、回撤、返回 loaded
home，并在一次自动重试后确认源槽为空。记录时机械臂位于 loaded home，夹爪闭合
并持有试管。重新开机后不能假设该状态仍成立，必须以现场实际状态为准。
