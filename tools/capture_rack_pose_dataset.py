#!/usr/bin/env python3
"""Capture D435 color frames for offline rack-pose annotation; never moves the arm."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tube_grabber.config import load_config, project_path  # noqa: E402
from tube_grabber.hardware.realsense_d435 import D435Camera  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="采集架面 YOLO Pose 标注图片")
    parser.add_argument("--config", default="config/app.yaml")
    parser.add_argument("--output", default="datasets/rack_pose/images/raw")
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--interval", type=float, default=0.15)
    args = parser.parse_args()
    if args.count <= 0 or args.interval < 0.0:
        parser.error("--count must be positive and --interval cannot be negative")

    config = load_config(args.config)
    camera_config = config["camera"]
    camera = D435Camera(
        serial=str(camera_config["serial"]),
        width=int(camera_config["width"]),
        height=int(camera_config["height"]),
        fps=int(camera_config["fps"]),
        warmup_frames=int(camera_config["warmup_frames"]),
        timeout_ms=int(camera_config["timeout_ms"]),
    )
    output = project_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    session = datetime.now().strftime("%Y%m%d_%H%M%S")
    metadata = []
    try:
        camera.start()
        for index in range(args.count):
            frame = camera.capture()
            path = output / f"rack_{session}_{index:05d}.jpg"
            if not cv2.imwrite(str(path), frame.color, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise RuntimeError(f"failed to save {path}")
            metadata.append(
                {
                    "file": path.name,
                    "frame_number": frame.frame_number,
                    "timestamp_ms": frame.timestamp_ms,
                    "intrinsics": {
                        "fx": frame.intrinsics.fx,
                        "fy": frame.intrinsics.fy,
                        "cx": frame.intrinsics.cx,
                        "cy": frame.intrinsics.cy,
                    },
                }
            )
            print(f"[{index + 1}/{args.count}] {path}")
            if args.interval:
                time.sleep(args.interval)
    finally:
        camera.stop()
    metadata_path = output / f"rack_{session}_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
