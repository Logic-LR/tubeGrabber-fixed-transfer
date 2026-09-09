# 双试管架移动闭环

`mobile-transfer` 用于以下固定布置：机器人从 `rack_1` 取管，右臂带管回到已
确认的观察/运输 home，底盘旋转约 180 度并沿新的车体 X 方向平移实测距离，随后
重新识别 `rack_2`、放置并复扫。

这条流程不复用底盘移动前得到的三维目标。`base_right` 会随底盘移动，所有
`rack_2` 放置坐标均由移动后的 D435、cap/screw YOLO、深度和平面拟合重新计算。

## 执行顺序

```text
空载 observation/home
  -> rack_1 首次稳定扫描
  -> rack_1 执行前复扫并比较现场
  -> 抓取并沿架面法向撤离
  -> 带管 observation/transport home
  -> 复扫 rack_1，确认源槽为空且其他槽未变化
  -> 两次约 90 度 Woosh 闭环旋转
  -> 沿车体 X 分段闭环平移总距离 X
  -> 两次扫描 rack_2，确认目标为空且现场稳定
  -> 使用 rack_2 最新三维坐标放置
  -> 回到 observation/home
  -> 最终复扫 rack_2，确认目标占用且其他槽未变化
```

任何阶段失败都会停止后续阶段。夹紧命令发出后，软件保守地视为仍持管；底盘失败、
目标视觉失败或放置前失败均不会自动松爪。

## 两种执行入口

分阶段验收时，先只执行“抓取并回带管 home”：

```bash
python -m tube_grabber plan-mobile-pick-home \
  --source rack_1.r1c1

python -m tube_grabber mobile-pick-home \
  --source rack_1.r1c1
```

源架上只有一支试管时，也可以在稳定扫描后自动选择其槽位：

```bash
python -m tube_grabber mobile-pick-home --auto-source
```

自动模式要求恰好一个槽位为 `OCCUPIED`；识别到零支或多支时会在运动前停止。

该阶段不读取尚未测量的底盘平移距离，不调用 Woosh helper，也不会释放试管。执行
成功后，进程退出时夹爪仍保持夹紧。继续调试前必须由现场人员明确决定是人工取管、
执行专用恢复流程，还是进入下一阶段；不能直接再次运行抓取命令。

同架搬运保持原入口：

```bash
python -m tube_grabber transfer \
  --source rack_1.r1c1 \
  --destination rack_1.r1c2
```

跨站双架搬运使用独立入口：

```bash
python -m tube_grabber plan-mobile-transfer \
  --source rack_1.r1c1 \
  --destination rack_2.r1c2

python -m tube_grabber mobile-transfer \
  --source rack_1.r1c1 \
  --destination rack_2.r1c2
```

`plan-mobile-transfer` 只校验配置，不连接相机、机械臂、夹爪或底盘。

## 必须完成的配置

1. 使用 `calibrate-rack --rack rack_2` 生成独立的
   `config/racks/rack_2.yaml`，禁止复制 `rack_1` 标定。
2. 在 `config/mobile_transfer.yaml` 填入实测 `translation_x_m`。
3. 验证 `loaded_observation_pose` 同时满足带管运输净空和腕部相机视野：试管必须
   位于架面多边形之外，四颗螺丝和 K0 marker 必须完整可见。
4. 低速分段验证两次旋转和所有平移段后，才设置
   `loaded_observation_pose.confirmed: true` 和顶层 `confirmed: true`。
5. 保持 `motion.parameters_confirmed: false`，直到机械臂、两架、底盘路径和周边
   障碍均在最终现场验收完毕。

只验证抓取回 home 时，完成第 3 项以及右臂完整低速路径检查后，才设置
`pick_home_confirmed: true`。该锁不代表底盘路线已确认，不能替代顶层
`confirmed: true`。

## Woosh 边界

代码参考 `dual-arm-shelf-dispenser` 的失败关闭方式，通过机器人端两个辅助程序调用
Woosh 位姿/速度接口：

- `grabber_rotate_relative`：约 90 度相对旋转、实时位姿反馈、平移漂移门禁；
- `grabber_pose_servo`：短距离相对位姿闭环。

180 度会拆为两次约 90 度旋转。总平移 X 会拆成不超过 0.25 m 的小段，以满足参考
位姿伺服器的局部运动范围。每个辅助程序必须在返回前重复发送零速度、验证最终误差
并确认线速度和角速度归零；非零退出、超时或输出缺少最终位姿均视为失败。

辅助程序路径在 `config/mobile_transfer.yaml` 中配置。默认路径只是机器人部署约定，
仓库不会在 Windows 开发机上尝试连接底盘。
