"""Small, explicit scan outputs used during real-system verification."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from tube_grabber.config import project_path
from tube_grabber.core.errors import HardwareError
from tube_grabber.core.models import CameraFrame, Occupancy, RackObservation
from tube_grabber.workflow import PreparedTransfer


def format_observation(observation: RackObservation) -> str:
    lines = [
        f"rack={observation.rack_id}  plane_z={observation.plane_z_mm:.2f} mm",
        (
            f"stable_frames={observation.stability_frame_count}  "
            f"keypoint_spread={observation.maximum_keypoint_spread_px:.2f}px  "
            f"position_spread={observation.maximum_position_spread_mm:.2f}mm"
        ),
    ]
    for row in (1, 2):
        cells = []
        for column in range(1, 7):
            slot = next(
                item
                for item in observation.slots
                if item.address.row == row and item.address.column == column
            )
            state = {
                Occupancy.EMPTY: "EMPTY",
                Occupancy.OCCUPIED: "OCCUPIED",
                Occupancy.UNKNOWN: "UNKNOWN",
            }[slot.occupancy]
            cells.append(f"c{column}:{state}({slot.confidence:.2f})")
        lines.append(f"row {row}  " + "  ".join(cells))

    lines.extend(
        [
            "slot coordinate table (base_right, mm):",
            "  slot           state      conf   pixel[u,v]        reference       xyz[x,y,z]",
        ]
    )
    for slot in sorted(
        observation.slots,
        key=lambda item: (item.address.row, item.address.column),
    ):
        if slot.occupancy is Occupancy.OCCUPIED:
            point = slot.cap_top_base
            reference = "cap_top"
        else:
            point = slot.hole_on_plane_base
            reference = "rack_plane"
        coordinate = (
            f"[{point.x_mm:.2f},{point.y_mm:.2f},{point.z_mm:.2f}]"
            if point is not None
            else "[unavailable]"
        )
        pixel = f"[{slot.pixel.u:.1f},{slot.pixel.v:.1f}]"
        lines.append(
            f"  {slot.address.text:<14} {slot.occupancy.value:<10} "
            f"{slot.confidence:>5.2f}   {pixel:<18} "
            f"{reference:<15} {coordinate}"
        )
    return "\n".join(lines)


def format_prepared_transfer(prepared: PreparedTransfer) -> str:
    source = prepared.source_target
    destination = prepared.destination_target
    lines = [
        "preview TCP targets (base_right, mm):",
        f"  pick  = [{source.x_mm:.2f}, {source.y_mm:.2f}, {source.z_mm:.2f}]",
        (
            "  place = "
            f"[{destination.x_mm:.2f}, {destination.y_mm:.2f}, "
            f"{destination.z_mm:.2f}]"
        ),
        (
            "planned from flange pose: "
            f"[{prepared.start_pose.x_mm:.2f}, "
            f"{prepared.start_pose.y_mm:.2f}, "
            f"{prepared.start_pose.z_mm:.2f}, "
            f"{prepared.start_pose.rx_rad:.4f}, "
            f"{prepared.start_pose.ry_rad:.4f}, "
            f"{prepared.start_pose.rz_rad:.4f}]"
        ),
        "preview flange waypoints (execution refreshes coordinates):",
    ]
    sections = (
        ("pick", prepared.pick_approach),
        ("pick", prepared.pick_retreat),
        ("place", prepared.place_approach),
        ("place", prepared.place_retreat),
    )
    index = 1
    for phase, plan in sections:
        for waypoint in plan.waypoints:
            pose = waypoint.pose
            lines.append(
                f"  {index:02d} {phase}.{waypoint.name:<12} "
                f"mode={'L' if waypoint.linear else 'J_P':<3}  "
                f"v={waypoint.speed_percent:>2}%  "
                f"[{pose.x_mm:.2f}, {pose.y_mm:.2f}, {pose.z_mm:.2f}, "
                f"{pose.rx_rad:.4f}, {pose.ry_rad:.4f}, {pose.rz_rad:.4f}]"
            )
            index += 1
    return "\n".join(lines)


def save_camera_frame(
    frame: CameraFrame,
    *,
    output_dir: str | Path = "artifacts",
    prefix: str = "camera",
) -> tuple[Path, Path]:
    directory = project_path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    color_path = directory / f"{prefix}_{stamp}_color.png"
    depth_path = directory / f"{prefix}_{stamp}_depth_mm.png"
    color = np.asarray(frame.color)
    depth = np.asarray(frame.depth_mm)
    _write_image(color_path, color)
    depth_u16 = np.clip(np.nan_to_num(depth, nan=0.0), 0, 65535).astype(
        np.uint16
    )
    _write_image(depth_path, depth_u16)
    return color_path, depth_path


def save_observation_image(
    frame: CameraFrame,
    observation: RackObservation,
    *,
    output_dir: str | Path = "artifacts",
) -> Path:
    directory = project_path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = directory / f"scan_{observation.rack_id}_{stamp}.png"
    image = np.asarray(frame.color).copy()
    for slot in observation.slots:
        center = (int(round(slot.pixel.u)), int(round(slot.pixel.v)))
        color = (
            (0, 0, 255)
            if slot.occupancy is Occupancy.OCCUPIED
            else (0, 200, 0)
        )
        cv2.circle(image, center, 8, color, 2)
        cv2.putText(
            image,
            f"r{slot.address.row}c{slot.address.column}",
            (center[0] + 7, center[1] - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
    keypoint_names = (
        "k0",
        "k1",
        "k2",
        "k3",
        "screw_k0",
        "screw_k1",
        "screw_k2",
        "screw_k3",
    )
    keypoints = observation.rack_keypoints or (observation.marker,)
    for name, point in zip(keypoint_names, keypoints):
        center = (int(round(point.u)), int(round(point.v)))
        cv2.drawMarker(
            image,
            center,
            (255, 0, 255),
            cv2.MARKER_CROSS,
            18,
            2,
        )
        cv2.putText(
            image,
            name,
            (center[0] + 7, center[1] - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 0, 255),
            1,
            cv2.LINE_AA,
        )
    _write_image(path, image)
    return path


def _write_image(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise HardwareError(f"failed to save image: {path}")
