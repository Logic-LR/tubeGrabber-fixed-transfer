# 固定取管到篮子

这个流程不使用相机、YOLO、试管架槽位标定或 Agent。机械臂只接受一组经过现场
示教和确认的固定笛卡尔位姿，完成一次固定位置取管和固定篮子投放。

实现借鉴 `dual-arm-shelf-dispenser` 的执行边界：先校验完整路径，再允许夹爪或机械臂
动作；执行前检查真实控制器；每个路点到达后读取 TCP 反馈；任何异常均停止后续动作并
请求机械臂缓停。没有引入该项目的 ROS 2、MoveIt、双臂 worker、底盘或升降柱模块。

## 动作顺序

```text
home
  -> pick_above          movej_p
  -> pick                movel
  -> 夹紧
  -> pick_above          movel
  -> basket_above        movej_p
  -> basket_release      movel
  -> 释放
  -> basket_above        movel
  -> home                movej_p
```

普通转场使用控制器关节规划 `movej_p`；靠近试管、带管抬升、进入篮子和撤离使用
直线运动 `movel`。所有点位都是当前控制器 `Arm_Tip` 在 `base_right` 中的位姿：位置
使用毫米，姿态使用弧度；RealMan 驱动只在 SDK 边界将毫米转换为米。这里没有复用
另一个项目的 `bottleTCP`，也不要将两种工具坐标系下记录的点位混用。

## 1. 示教五个位姿

保持 `config/fixed_transfer.yaml` 的 `confirmed: false`。通过拖动示教或低速点动，
依次将机械臂放到：

1. `home`：固定任务唯一允许的起点和终点；
2. `pick_above`：试管正上方、可以安全水平转场的位置；
3. `pick`：夹爪实际闭合的位置；
4. `basket_above`：篮子中心上方的安全位置；
5. `basket_release`：松开试管的位置，优先保持在篮子边缘上方。

每个位姿使用只读命令记录：

```bash
python -m tube_grabber arm-status
```

将输出的 `position_mm` 和 `rpy_rad` 填入 `config/fixed_transfer.yaml`。不要复制其他
机器人或其他工具坐标系下的数值。

## 2. 离线检查

```bash
python -m tube_grabber plan-fixed-transfer
```

该命令不连接机械臂和夹爪。它会打印每个路点、SDK 运动类型、单位和速度，并按
`config/app.yaml` 中的工作空间与最大单段距离检查整个循环。

如果此时失败，应修正点位或安全范围；不要扩大工作空间参数来绕过失败。

## 3. 分阶段真机验收

第一次验收应保持 `confirmed: false`，使用示教器低速逐段复核。确认以下条件后，才将
`confirmed` 改为 `true`：

- `home` 与机械臂真实起点一致；
- `pick_above -> pick -> pick_above` 是无碰撞直线；
- 持管状态下从 `pick_above` 到 `basket_above` 有足够净空；
- `basket_above -> basket_release -> basket_above` 不进入篮子侧壁；
- `Arm_Tip`、工作坐标系和工具坐标系没有变化；
- `motion.parameters_confirmed` 对当前现场仍然有效。

## 4. 执行

```bash
python -m tube_grabber fixed-transfer
```

真实模式会要求操作者输入精确的 `RUN`，然后才连接并初始化夹爪。连接后还会检查：

- 控制器处于真实模式且已经上电；
- 控制器和七个关节无错误；
- 当前 TCP 与 `home` 的位置和姿态误差在配置容差内；
- 所有路点均位于工作空间内，且没有超长单段运动。

这个固定流程没有环境感知或 MoveIt 碰撞模型。只要机器人、试管架、篮子、工具或
周边障碍发生变化，就应立即把 `confirmed` 改回 `false` 并重新示教验收。
