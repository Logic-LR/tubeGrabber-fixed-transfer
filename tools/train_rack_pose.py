#!/usr/bin/env python3
"""Train the one-class eight-keypoint rack model with local CUDA."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="训练架面八关键点 YOLO Pose 模型"
    )
    parser.add_argument("--data", default="training/rack_pose.yaml")
    parser.add_argument("--model", default="yolo11s-pose.pt")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=-1)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/rack_pose")
    parser.add_argument("--name", default="train")
    args = parser.parse_args()
    if not Path(args.data).is_file():
        parser.error(f"dataset YAML does not exist: {args.data}")
    try:
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        parser.error(f"missing CUDA PyTorch/Ultralytics: {exc}")
    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable; training will not silently fall back to CPU")
    try:
        device_index = int(args.device)
        if device_index < 0:
            raise ValueError("device index cannot be negative")
        device_name = torch.cuda.get_device_name(device_index)
    except (TypeError, ValueError, RuntimeError) as exc:
        parser.error(f"--device must select a local CUDA GPU: {exc}")
    print(f"training on CUDA:{device_index} ({device_name})")
    model = YOLO(args.model, task="pose")
    result = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=str(device_index),
        project=args.project,
        name=args.name,
        patience=40,
        plots=True,
        # Mirroring a physically numbered K0 rack can corrupt its orientation.
        fliplr=0.0,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
