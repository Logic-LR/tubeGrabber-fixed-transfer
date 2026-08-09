# 模型目录

正式运行权重必须命名为：

```text
models/tube_slots.pt
```

权重必须是 Ultralytics YOLO 目标检测模型，并且类别严格为：

```text
0: empty_hole
1: tube_cap
```

权重文件默认不提交到 Git。放入模型后，在项目根目录验证：

```powershell
.\.venv\Scripts\python.exe -c "from ultralytics import YOLO; print(YOLO('models/tube_slots.pt').names)"
.\.venv\Scripts\python.exe -m tube_grabber doctor
```

第一条命令必须输出 `{0: 'empty_hole', 1: 'tube_cap'}`。模型的训练、验收和现场测试步骤见 `docs/LAB_DEPLOYMENT_AND_ROADMAP.md` 与 `docs/MODEL_CONTRACT.md`。
