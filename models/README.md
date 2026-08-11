# 模型目录

正式运行需要两个独立权重：

```text
models/cap.pt
models/rack_pose.pt
```

`cap.pt` 是 Ultralytics YOLO detection 模型，类别严格为：

```text
0: tube_cap
```

`rack_pose.pt` 是 Ultralytics YOLO Pose 模型，类别严格为
`0: rack_surface`，关键点数量必须为 8，顺序为：

```text
k0, k1, k2, k3, screw_k0, screw_k1, screw_k2, screw_k3
```

权重文件默认不提交到 Git。放入模型后，在项目根目录验证：

```bash
python -c "from ultralytics import YOLO; print(YOLO('models/cap.pt').names)"
python -c "from ultralytics import YOLO; print(YOLO('models/rack_pose.pt', task='pose').names)"
python -m tube_grabber doctor
```

运行时配置为本地 `CUDA:0`，真机不会静默回退 CPU。架面模型的采集、
标注、训练和验收见 `training/RACK_POSE_CONTRACT.md`。
