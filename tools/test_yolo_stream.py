"""CUDA-only live D435 test for cap.pt and screw.pt.

This command opens only the RealSense color stream.  It does not connect to
the arm or gripper and does not run depth, plane fitting, slot matching, or
motion code.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Sequence

import cv2
import numpy as np

# Keep the documented ``python tools/test_yolo_stream.py`` command working
# even when the project has not yet been installed editable.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tube_grabber.config import load_config, project_path
from tube_grabber.core.errors import VisionError
from tube_grabber.core.models import Detection, Pixel
from tube_grabber.vision.detector import YoloCapDetector
from tube_grabber.vision.rack_calibration import load_rack_calibration
from tube_grabber.vision.rack_pose import (
    ScrewMarkerConfig,
    YoloScrewRackDetector,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="D435 彩色流 cap/screw 双 YOLO CUDA 连续推理",
    )
    parser.add_argument("--config", default="config/app.yaml")
    parser.add_argument("--cap-confidence", type=float, default=None)
    parser.add_argument("--screw-confidence", type=float, default=None)
    parser.add_argument("--rack", choices=("rack_1", "rack_2"), default="rack_1")
    args = parser.parse_args()
    config = load_config(args.config)
    vision = config["vision"]
    camera = config["camera"]
    cap_config = vision["cap"]
    screw_config = vision["screw"]
    device = str(vision["device"])
    rack_path = project_path(config["racks"][args.rack]["calibration_path"])
    calibration = load_rack_calibration(rack_path, args.rack)
    if not calibration.display_corners_calibrated:
        raise RuntimeError(
            f"{args.rack} 尚未标定展示角点；先运行 "
            f"python -m tube_grabber calibrate-corners --rack {args.rack}"
        )

    _require_cuda(device)
    cap_detector = YoloCapDetector(
        model_path=project_path(cap_config["model_path"]),
        class_names=cap_config["class_names"],
        confidence=(
            float(args.cap_confidence)
            if args.cap_confidence is not None
            else float(cap_config["confidence"])
        ),
        iou=float(cap_config["iou"]),
        image_size=int(cap_config["image_size"]),
        device=device,
    )
    marker_data = screw_config["marker"]
    screw_detector = YoloScrewRackDetector(
        model_path=project_path(screw_config["model_path"]),
        class_names=screw_config["class_names"],
        confidence=(
            float(args.screw_confidence)
            if args.screw_confidence is not None
            else float(screw_config["confidence"])
        ),
        iou=float(screw_config["iou"]),
        image_size=int(screw_config["image_size"]),
        device=device,
        marker=ScrewMarkerConfig(
            minimum_value=int(marker_data["minimum_value"]),
            maximum_saturation=int(marker_data["maximum_saturation"]),
            search_radius_factor=float(marker_data["search_radius_factor"]),
            screw_exclusion_scale=float(marker_data["screw_exclusion_scale"]),
            minimum_area_px2=int(marker_data["minimum_area_px2"]),
            maximum_area_ratio=float(marker_data["maximum_area_ratio"]),
            minimum_score_ratio=float(marker_data["minimum_score_ratio"]),
            minimum_rack_aspect_ratio=float(
                marker_data["minimum_rack_aspect_ratio"]
            ),
        ),
    )

    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise RuntimeError("未安装 pyrealsense2") from exc

    pipeline = rs.pipeline()
    stream_config = rs.config()
    serial = str(camera.get("serial", "")).strip()
    if serial:
        stream_config.enable_device(serial)
    stream_config.enable_stream(
        rs.stream.color,
        int(camera["width"]),
        int(camera["height"]),
        rs.format.bgr8,
        int(camera["fps"]),
    )
    window = "cap + K0-K3 | CUDA live | q/ESC quit"
    started = False
    try:
        pipeline.start(stream_config)
        started = True
        for _ in range(int(camera["warmup_frames"])):
            pipeline.wait_for_frames(int(camera["timeout_ms"]))
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        while True:
            frame = pipeline.wait_for_frames(
                int(camera["timeout_ms"])
            ).get_color_frame()
            if not frame:
                continue
            image = np.asanyarray(frame.get_data()).copy()
            started_at = time.perf_counter()
            caps = cap_detector.detect(image)
            screws = screw_detector.detect_screws(image)
            rack = None
            try:
                rack = screw_detector.build_rack_detection(image, screws)
            except VisionError:
                pass
            elapsed_ms = (time.perf_counter() - started_at) * 1000.0

            preview = image.copy()
            _draw_caps(preview, caps)
            display_corners = (
                calibration.project_display_corners(rack)
                if rack is not None
                else ()
            )
            _draw_corners(preview, display_corners)
            status = (
                f"CUDA:{device}  {elapsed_ms:.1f}ms  "
                f"caps={len(caps)} corners={len(display_corners)}  "
                + ("rack=OK" if rack is not None else "rack=not found")
            )
            cv2.putText(
                preview,
                status[:150],
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255) if rack is not None else (0, 80, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.imshow(window, preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        if started:
            pipeline.stop()
        try:
            cv2.destroyWindow(window)
        except cv2.error:
            pass
    return 0


def _require_cuda(device: str) -> None:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("未安装 CUDA 版 PyTorch") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用；本测试不回退 CPU")
    index = int(device)
    if index >= torch.cuda.device_count():
        raise RuntimeError(f"CUDA:{index} 不存在")
    print(
        f"使用 CUDA:{index} {torch.cuda.get_device_name(index)}；"
        "按 q 或 ESC 退出"
    )


def _draw_caps(image: np.ndarray, caps: Sequence[Detection]) -> None:
    for cap in caps:
        box = cap.box
        left, top = int(round(box.x1)), int(round(box.y1))
        right, bottom = int(round(box.x2)), int(round(box.y2))
        center = (int(round(box.center.u)), int(round(box.center.v)))
        cv2.rectangle(image, (left, top), (right, bottom), (0, 220, 0), 1)
        cv2.circle(image, center, 2, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(
            image,
            f"{cap.confidence:.2f}",
            (left, max(12, top - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 220, 0),
            1,
            cv2.LINE_AA,
        )


def _draw_corners(
    image: np.ndarray,
    corners: Sequence[Pixel],
) -> None:
    if len(corners) != 4:
        return
    polygon = np.asarray(
        [[round(point.u), round(point.v)] for point in corners],
        dtype=np.int32,
    )
    cv2.polylines(image, [polygon], True, (255, 0, 255), 1, cv2.LINE_AA)
    for index, point in enumerate(corners):
        center = (int(round(point.u)), int(round(point.v)))
        color = (0, 255, 255) if index == 0 else (255, 255, 0)
        cv2.circle(image, center, 4, color, 1, cv2.LINE_AA)
        cv2.putText(
            image,
            f"K{index}",
            (center[0] + 4, center[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            color,
            1,
            cv2.LINE_AA,
        )


if __name__ == "__main__":
    raise SystemExit(main())
