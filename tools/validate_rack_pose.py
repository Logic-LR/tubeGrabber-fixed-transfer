#!/usr/bin/env python3
"""Validate pose mAP on held-out sequences and enforce a minimum quality gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="验收架面八关键点模型")
    parser.add_argument("--data", default="training/rack_pose.yaml")
    parser.add_argument("--weights", default="models/rack_pose.pt")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default="0")
    parser.add_argument("--minimum-pose-map50", type=float, default=0.95)
    parser.add_argument("--minimum-pose-map", type=float, default=0.75)
    args = parser.parse_args()
    for path in (args.data, args.weights):
        if not Path(path).is_file():
            parser.error(f"file does not exist: {path}")
    try:
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        parser.error(f"missing CUDA PyTorch/Ultralytics: {exc}")
    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable; validation requires the deployment GPU path")
    try:
        device_index = int(args.device)
        if device_index < 0:
            raise ValueError("device index cannot be negative")
        device_name = torch.cuda.get_device_name(device_index)
    except (TypeError, ValueError, RuntimeError) as exc:
        parser.error(f"--device must select a local CUDA GPU: {exc}")
    print(f"validating on CUDA:{device_index} ({device_name})")
    model = YOLO(args.weights, task="pose")
    names = {int(key): str(value) for key, value in dict(model.names).items()}
    if names != {0: "rack_surface"}:
        parser.error(f"model class contract is invalid: {names}")
    keypoint_shape = tuple(getattr(model.model, "kpt_shape", ()))
    if keypoint_shape != (8, 3):
        parser.error(f"model keypoint shape is {keypoint_shape}, expected (8, 3)")
    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        device=str(device_index),
        plots=True,
    )
    pose_map50 = float(metrics.pose.map50)
    pose_map = float(metrics.pose.map)
    report = {
        "weights": args.weights,
        "pose_map50": pose_map50,
        "pose_map50_95": pose_map,
        "minimum_pose_map50": args.minimum_pose_map50,
        "minimum_pose_map50_95": args.minimum_pose_map,
        "passed": (
            pose_map50 >= args.minimum_pose_map50
            and pose_map >= args.minimum_pose_map
        ),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
