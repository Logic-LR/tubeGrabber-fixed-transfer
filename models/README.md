# 模型目录

正式运行需要两个独立权重：

```text
models/cap.pt
models/screw.pt
```

`cap.pt` 和 `screw.pt` 都是 Ultralytics YOLO detection 模型。当前两个权重的
内部类别名都严格为：

```text
0: item
```

适配层分别把 `item` 转成 `tube_cap` 和 `screw`。四个 screw 检测框的中心定义
架面，额外白色 marker 指示 K0，K0 的长边邻点为 K1。

权重文件默认不提交到 Git。放入模型后，在项目根目录验证：

```bash
python -c "from ultralytics import YOLO; print(YOLO('models/cap.pt').names)"
python -c "from ultralytics import YOLO; print(YOLO('models/screw.pt', task='detect').names)"
python -m tube_grabber doctor
```

运行时配置为本地 `CUDA:0`，真机不会静默回退 CPU。
