"""Robust RGB-D rack-plane fitting inside the keypoint quadrilateral."""

from __future__ import annotations

from dataclasses import dataclass
from math import acos, degrees, isfinite
from typing import Sequence

import cv2
import numpy as np

from tube_grabber.core.errors import VisionError
from tube_grabber.core.models import Box, CameraFrame, Pixel, Point3D
from tube_grabber.vision.geometry import pixel_ray_to_plane, validate_transform
from tube_grabber.vision.rack_pose import RackPoseDetection


@dataclass(frozen=True)
class RackPlane:
    """Plane in ``base_right`` as ``normal·point_mm + offset_mm = 0``."""

    normal_base: tuple[float, float, float]
    offset_mm: float
    rms_error_mm: float
    inlier_ratio: float
    inlier_count: int

    def signed_distance_mm(self, point: Point3D) -> float:
        if point.frame != "base_right":
            raise VisionError(f"plane point must use base_right, got {point.frame}")
        return float(
            np.dot(
                np.asarray(self.normal_base),
                np.asarray([point.x_mm, point.y_mm, point.z_mm]),
            )
            + self.offset_mm
        )

    def intersect(
        self,
        pixel: Pixel,
        frame: CameraFrame,
        base_from_camera: object,
    ) -> Point3D:
        return pixel_ray_to_plane(
            pixel,
            self.normal_base,
            self.offset_mm,
            frame.intrinsics,
            base_from_camera,
        )


@dataclass(frozen=True)
class RackPlaneFitConfig:
    sample_stride_px: int
    roi_margin_px: int
    landmark_exclusion_radius_px: int
    ransac_iterations: int
    inlier_threshold_mm: float
    minimum_inliers: int
    minimum_inlier_ratio: float
    maximum_rms_error_mm: float
    maximum_tilt_deg: float

    def __post_init__(self) -> None:
        numeric_values = (
            self.inlier_threshold_mm,
            self.minimum_inlier_ratio,
            self.maximum_rms_error_mm,
            self.maximum_tilt_deg,
        )
        if not all(isfinite(float(value)) for value in numeric_values):
            raise ValueError("rack plane limits must be finite")
        if self.sample_stride_px <= 0 or self.roi_margin_px < 0:
            raise ValueError("rack plane pixel sampling parameters are invalid")
        if self.landmark_exclusion_radius_px < 0 or self.ransac_iterations <= 0:
            raise ValueError("rack plane exclusion/RANSAC parameters are invalid")
        if self.inlier_threshold_mm <= 0.0 or self.minimum_inliers < 3:
            raise ValueError("rack plane inlier parameters are invalid")
        if not 0.0 < self.minimum_inlier_ratio <= 1.0:
            raise ValueError("minimum_inlier_ratio must be in (0, 1]")
        if self.maximum_rms_error_mm <= 0.0 or self.maximum_tilt_deg <= 0.0:
            raise ValueError("rack plane RMS/tilt limits must be positive")


def fit_rack_plane(
    frame: CameraFrame,
    pose: RackPoseDetection,
    base_from_camera: object,
    *,
    excluded_boxes: Sequence[Box],
    depth_min_mm: float,
    depth_max_mm: float,
    config: RackPlaneFitConfig,
) -> RackPlane:
    """Fit the physical top surface while excluding caps, holes and edges."""
    depth = np.asarray(frame.depth_mm, dtype=np.float64)
    if depth.ndim != 2:
        raise VisionError("aligned depth image must be two-dimensional")
    mask = np.zeros(depth.shape, dtype=np.uint8)
    corners = np.asarray(
        [[round(point.u), round(point.v)] for point in pose.corners],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(mask, corners, 255)
    if config.roi_margin_px:
        size = 2 * int(config.roi_margin_px) + 1
        mask = cv2.erode(mask, np.ones((size, size), dtype=np.uint8))
    for keypoint in pose.keypoints:
        cv2.circle(
            mask,
            (int(round(keypoint.pixel.u)), int(round(keypoint.pixel.v))),
            int(config.landmark_exclusion_radius_px),
            0,
            -1,
        )
    for box in excluded_boxes:
        padding = max(2, int(config.landmark_exclusion_radius_px // 2))
        cv2.rectangle(
            mask,
            (
                max(0, int(np.floor(box.x1)) - padding),
                max(0, int(np.floor(box.y1)) - padding),
            ),
            (
                min(depth.shape[1] - 1, int(np.ceil(box.x2)) + padding),
                min(depth.shape[0] - 1, int(np.ceil(box.y2)) + padding),
            ),
            0,
            -1,
        )
    sampling = np.zeros_like(mask)
    sampling[:: config.sample_stride_px, :: config.sample_stride_px] = 255
    valid = (
        (mask > 0)
        & (sampling > 0)
        & np.isfinite(depth)
        & (depth >= float(depth_min_mm))
        & (depth <= float(depth_max_mm))
    )
    rows, columns = np.nonzero(valid)
    if len(rows) < config.minimum_inliers:
        raise VisionError(
            f"rack plane has only {len(rows)} valid depth samples; "
            f"requires {config.minimum_inliers}"
        )
    points_base = _pixels_depth_to_base(
        columns.astype(np.float64),
        rows.astype(np.float64),
        depth[rows, columns],
        frame,
        base_from_camera,
    )
    normal, offset, inliers = _ransac_plane(
        points_base,
        iterations=config.ransac_iterations,
        threshold_mm=float(config.inlier_threshold_mm),
    )
    count = int(inliers.sum())
    ratio = count / len(points_base)
    if count < config.minimum_inliers:
        raise VisionError(f"rack plane inliers {count} < {config.minimum_inliers}")
    if ratio < float(config.minimum_inlier_ratio):
        raise VisionError(
            f"rack plane inlier ratio {ratio:.3f} < {config.minimum_inlier_ratio:.3f}"
        )

    # Orient the normal toward the camera so tube-cap height is positive.
    transform = validate_transform(base_from_camera, "base-from-camera")
    camera_origin = transform[:3, 3]
    centroid = points_base[inliers].mean(axis=0)
    if float(np.dot(normal, camera_origin - centroid)) < 0.0:
        normal = -normal
        offset = -offset
    distances = points_base[inliers] @ normal + offset
    rms = float(np.sqrt(np.mean(distances**2)))
    if rms > float(config.maximum_rms_error_mm):
        raise VisionError(
            f"rack plane RMS {rms:.3f}mm exceeds {config.maximum_rms_error_mm:.3f}mm"
        )
    tilt = degrees(acos(float(np.clip(abs(normal[2]), 0.0, 1.0))))
    if tilt > float(config.maximum_tilt_deg):
        raise VisionError(
            f"rack plane tilt {tilt:.2f}deg exceeds {config.maximum_tilt_deg:.2f}deg"
        )
    return RackPlane(
        normal_base=tuple(float(value) for value in normal),
        offset_mm=float(offset),
        rms_error_mm=rms,
        inlier_ratio=float(ratio),
        inlier_count=count,
    )


def _pixels_depth_to_base(
    columns: np.ndarray,
    rows: np.ndarray,
    depth_mm: np.ndarray,
    frame: CameraFrame,
    base_from_camera: object,
) -> np.ndarray:
    intrinsics = frame.intrinsics
    pixels = np.column_stack((columns, rows)).astype(np.float64)
    if intrinsics.distortion_model == "brown_conrady" and any(
        abs(value) > 1e-12 for value in intrinsics.distortion_coefficients
    ):
        camera_matrix = np.asarray(
            [
                [intrinsics.fx, 0.0, intrinsics.cx],
                [0.0, intrinsics.fy, intrinsics.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        normalized = cv2.undistortPoints(
            pixels.reshape(-1, 1, 2),
            camera_matrix,
            np.asarray(intrinsics.distortion_coefficients, dtype=np.float64),
        ).reshape(-1, 2)
    else:
        normalized = np.column_stack(
            (
                (columns - intrinsics.cx) / intrinsics.fx,
                (rows - intrinsics.cy) / intrinsics.fy,
            )
        )
    camera = np.column_stack(
        (normalized[:, 0] * depth_mm, normalized[:, 1] * depth_mm, depth_mm)
    )
    transform = validate_transform(base_from_camera, "base-from-camera")
    return camera @ transform[:3, :3].T + transform[:3, 3]


def _ransac_plane(
    points: np.ndarray,
    *,
    iterations: int,
    threshold_mm: float,
) -> tuple[np.ndarray, float, np.ndarray]:
    rng = np.random.default_rng(0)
    best: np.ndarray | None = None
    best_error = float("inf")
    for _ in range(iterations):
        sample = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        length = float(np.linalg.norm(normal))
        if length <= 1e-9:
            continue
        normal /= length
        offset = -float(np.dot(normal, sample[0]))
        distances = np.abs(points @ normal + offset)
        candidate = distances <= threshold_mm
        count = int(candidate.sum())
        error = float(np.median(distances[candidate])) if count else float("inf")
        if (
            best is None
            or count > int(best.sum())
            or (count == int(best.sum()) and error < best_error)
        ):
            best = candidate
            best_error = error
    if best is None or int(best.sum()) < 3:
        raise VisionError("rack plane RANSAC could not find a valid plane")
    centered = points[best] - points[best].mean(axis=0)
    _, _, vectors = np.linalg.svd(centered, full_matrices=False)
    normal = vectors[-1]
    normal /= np.linalg.norm(normal)
    offset = -float(np.dot(normal, points[best].mean(axis=0)))
    refined = np.abs(points @ normal + offset) <= threshold_mm
    if int(refined.sum()) >= 3:
        centered = points[refined] - points[refined].mean(axis=0)
        _, _, vectors = np.linalg.svd(centered, full_matrices=False)
        normal = vectors[-1]
        normal /= np.linalg.norm(normal)
        offset = -float(np.dot(normal, points[refined].mean(axis=0)))
    return normal, offset, refined
