# 架面 YOLO Pose 标注契约

模型只有一个类别：`0: rack_surface`，每个实例固定输出 8 个关键点：

1. `k0`
2. `k1`
3. `k2`
4. `k3`
5. `screw_k0`
6. `screw_k1`
7. `screw_k2`
8. `screw_k3`

`k0～k3` 是架面四角的物理编号，必须沿同一方向顺时针标注；`screw_kN`
是靠近 `kN` 的螺丝孔中心。`screw_k0` 旁固定贴一个小红圆点，作为 K0
方向的永久视觉标志。它们不是槽位名称，槽位始终单独使用
`r1c1～r2c6`。所有图片必须选择同一个物理角作为 `k0`，不能按图片里的
左上角临时决定 K0。

红点必须在采集数据前贴好，并在所有图片中保持同一位置。运行时会独立检查
`screw_k0` 邻域的红色占比明显高于其他三个螺丝孔。双圆心标定会将 8 点参考保存为
跨视角的单位架面坐标，发现关键点语义错误或异常形变时拒绝输出机械臂坐标。
训练脚本关闭水平翻转增强，因为 K0 是带红点的物理身份，不能在增强时被交换。

## 数据要求

- 训练、验证、测试必须按采集序列划分，不能把同一连续视频的相邻帧拆到不同集合。
- 覆盖空架、不同占用组合、不同颜色盖子、曝光变化、反光、轻微遮挡和允许范围内的摆放偏差。
- 四角和螺丝孔即使被短暂遮挡，也要按 YOLO Pose 可见性规则一致标注；严重遮挡样本不要猜点。
- 先用 `tools/capture_rack_pose_dataset.py` 采集，再用任意支持 YOLO Pose 的标注工具输出 Ultralytics 标签。
- 验收不仅看 mAP，还必须在真实观察位连续扫描，满足配置中的关键点像素抖动和槽位三维抖动门限。

训练命令示例：

```bash
python tools/train_rack_pose.py --data training/rack_pose.yaml --model yolo11s-pose.pt
```

验证命令示例：

```bash
python tools/validate_rack_pose.py --data training/rack_pose.yaml --weights models/rack_pose.pt
```
