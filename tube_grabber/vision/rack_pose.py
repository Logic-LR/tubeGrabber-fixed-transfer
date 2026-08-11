"""YOLO Pose adapter and geometry checks for the eight rack landmarks.

The keypoint order is a model contract.  Names describe physical rack
locations, not image-space top/left, so rack orientation remains stable when
the camera or rack rotates in the image.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Mapping, Protocol, Sequence

import cv2
import numpy as np

from tube_grabber.core.errors import VisionError
from tube_grabber.core.models import Box, Pixel


RACK_CLASS_NAMES = {0: "rack_surface"}
RACK_KEYPOINT_NAMES = (
    "k0",
    "k1",
    "k2",
    "k3",
    "screw_k0",
    "screw_k1",
    "screw_k2",
    "screw_k3",
)


@dataclass(frozen=True)
class RackPoseQualityConfig:
    minimum_rack_area_px2: float
    maximum_opposite_side_ratio: float
    screw_edge_margin: float
    k0_red_patch_radius_px: int
    k0_red_minimum_ratio: float
    k0_red_minimum_ratio_margin: float
    k0_red_minimum_saturation: int
    k0_red_minimum_value: int

    def __post_init__(self) -> None:
        if (
            not isfinite(float(self.minimum_rack_area_px2))
            or self.minimum_rack_area_px2 <= 0
        ):
            raise ValueError("minimum_rack_area_px2 must be positive")
        if (
            not isfinite(float(self.maximum_opposite_side_ratio))
            or self.maximum_opposite_side_ratio < 1.0
        ):
            raise ValueError("maximum_opposite_side_ratio must be at least one")
        if not 0.0 < self.screw_edge_margin < 0.25:
            raise ValueError("screw_edge_margin must be between 0 and 0.25")
        if self.k0_red_patch_radius_px <= 0:
            raise ValueError("k0_red_patch_radius_px must be positive")
        if not 0.0 < self.k0_red_minimum_ratio <= 1.0:
            raise ValueError("k0_red_minimum_ratio must be in (0, 1]")
        if not 0.0 < self.k0_red_minimum_ratio_margin <= 1.0:
            raise ValueError("k0_red_minimum_ratio_margin must be in (0, 1]")
        for name in ("k0_red_minimum_saturation", "k0_red_minimum_value"):
            value = int(getattr(self, name))
            if not 0 <= value <= 255:
                raise ValueError(f"{name} must be in [0, 255]")


@dataclass(frozen=True)
class RackKeypoint:
    name: str
    pixel: Pixel
    confidence: float

    def __post_init__(self) -> None:
        if self.name not in RACK_KEYPOINT_NAMES:
            raise ValueError(f"unsupported rack keypoint name: {self.name}")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("keypoint confidence must be between 0 and 1")


@dataclass(frozen=True)
class RackPoseDetection:
    confidence: float
    box: Box
    keypoints: tuple[RackKeypoint, ...]

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("rack confidence must be between 0 and 1")
        names = tuple(item.name for item in self.keypoints)
        if names != RACK_KEYPOINT_NAMES:
            raise ValueError(
                "rack keypoints must follow the fixed eight-point contract"
            )

    @property
    def corners(self) -> tuple[Pixel, Pixel, Pixel, Pixel]:
        return tuple(  # type: ignore[return-value]
            item.pixel for item in self.keypoints[:4]
        )

    @property
    def screws(self) -> tuple[Pixel, Pixel, Pixel, Pixel]:
        return tuple(  # type: ignore[return-value]
            item.pixel for item in self.keypoints[4:]
        )

    def pixel(self, name: str) -> Pixel:
        return self.keypoints[RACK_KEYPOINT_NAMES.index(name)].pixel


@dataclass(frozen=True)
class RackPoseStability:
    detection: RackPoseDetection
    inlier_indices: tuple[int, ...]
    attempted_frames: int
    maximum_keypoint_spread_px: float
    maximum_frame_residual_px: float


class RackPoseDetectorPort(Protocol):
    def detect(self, color_image: object) -> RackPoseDetection: ...


class YoloRackPoseDetector:
    """Run a one-class Ultralytics pose model with exactly eight keypoints."""

    def __init__(
        self,
        model_path: str | Path,
        class_names: Mapping[int, str],
        keypoint_names: Sequence[str],
        confidence: float,
        keypoint_confidence: float,
        iou: float,
        image_size: int,
        device: str,
    ) -> None:
        names = {int(key): str(value) for key, value in class_names.items()}
        if names != RACK_CLASS_NAMES:
            raise VisionError("rack pose classes must be {0: 'rack_surface'}")
        if tuple(str(value) for value in keypoint_names) != RACK_KEYPOINT_NAMES:
            raise VisionError(
                "rack pose keypoint_names do not match the fixed eight-point contract"
            )
        for name, value in (
            ("confidence", confidence),
            ("keypoint_confidence", keypoint_confidence),
            ("iou", iou),
        ):
            if not 0.0 < float(value) <= 1.0:
                raise VisionError(f"{name} must be between 0 and 1")
        if int(image_size) <= 0:
            raise VisionError("rack pose image_size must be positive")

        self._model_path = str(model_path)
        self._class_names = names
        self._confidence = float(confidence)
        self._keypoint_confidence = float(keypoint_confidence)
        self._iou = float(iou)
        self._image_size = int(image_size)
        self._device = str(device)
        self._model: object | None = None

    def detect(self, color_image: object) -> RackPoseDetection:
        image = np.asarray(color_image)
        if image.ndim != 3 or image.shape[2] != 3:
            raise VisionError("rack pose input must be a BGR image")
        model = self._get_model()
        try:
            results = model.predict(
                source=image,
                conf=self._confidence,
                iou=self._iou,
                imgsz=self._image_size,
                device=self._device,
                verbose=False,
            )
        except Exception as exc:
            raise VisionError(f"rack pose inference failed: {exc}") from exc
        if not results or results[0].boxes is None:
            raise VisionError("rack pose model found no rack surface")

        boxes = results[0].boxes
        if len(boxes) != 1:
            raise VisionError(
                f"expected exactly one rack surface, got {len(boxes)}"
            )
        class_id = int(boxes.cls[0].item())
        if class_id != 0:
            raise VisionError(f"rack pose returned unsupported class id {class_id}")
        keypoints = results[0].keypoints
        if keypoints is None or keypoints.xy is None:
            raise VisionError("rack pose result has no keypoints")
        xy = np.asarray(keypoints.xy[0].cpu(), dtype=np.float64)
        if xy.shape != (len(RACK_KEYPOINT_NAMES), 2):
            raise VisionError(
                f"rack pose returned keypoint shape {xy.shape}, expected (8, 2)"
            )
        if keypoints.conf is None:
            raise VisionError("rack pose model does not provide keypoint confidence")
        scores = np.asarray(keypoints.conf[0].cpu(), dtype=np.float64).reshape(-1)
        if scores.shape != (len(RACK_KEYPOINT_NAMES),):
            raise VisionError("rack pose keypoint confidence shape is invalid")
        if not np.isfinite(xy).all() or not np.isfinite(scores).all():
            raise VisionError("rack pose keypoints contain NaN or infinity")
        if float(scores.min()) < self._keypoint_confidence:
            index = int(np.argmin(scores))
            raise VisionError(
                f"{RACK_KEYPOINT_NAMES[index]} confidence {scores[index]:.3f} "
                f"is below {self._keypoint_confidence:.3f}"
            )

        x1, y1, x2, y2 = (float(value) for value in boxes.xyxy[0].tolist())
        return RackPoseDetection(
            confidence=float(boxes.conf[0].item()),
            box=Box(x1, y1, x2, y2),
            keypoints=tuple(
                RackKeypoint(
                    name,
                    Pixel(float(point[0]), float(point[1])),
                    float(score),
                )
                for name, point, score in zip(RACK_KEYPOINT_NAMES, xy, scores)
            ),
        )

    def _get_model(self) -> object:
        if self._model is not None:
            return self._model
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise VisionError(
                "Ultralytics is not installed; install the vision dependency"
            ) from exc
        try:
            model = YOLO(self._model_path, task="pose")
        except Exception as exc:
            raise VisionError(
                f"cannot load rack pose model {self._model_path}: {exc}"
            ) from exc
        model_names = {int(key): str(value) for key, value in dict(model.names).items()}
        if model_names != self._class_names:
            raise VisionError(
                f"rack pose model classes are {model_names}, "
                f"expected {self._class_names}"
            )
        self._model = model
        return model


def validate_rack_pose_geometry(
    detection: RackPoseDetection,
    *,
    minimum_area_px2: float,
    maximum_opposite_side_ratio: float,
    screw_edge_margin: float,
) -> None:
    """Reject crossed corners, collapsed racks and screw/keypoint permutations."""
    if not isfinite(float(minimum_area_px2)) or minimum_area_px2 <= 0.0:
        raise ValueError("minimum_area_px2 must be positive")
    if maximum_opposite_side_ratio < 1.0:
        raise ValueError("maximum_opposite_side_ratio must be at least one")
    if not 0.0 < screw_edge_margin < 0.25:
        raise ValueError("screw_edge_margin must be between 0 and 0.25")

    corners = _pixels_array(detection.corners).astype(np.float32)
    area = abs(float(cv2.contourArea(corners)))
    if area < float(minimum_area_px2):
        raise VisionError(
            f"rack corner area {area:.1f}px^2 is below {minimum_area_px2:.1f}px^2"
        )
    if not cv2.isContourConvex(corners):
        raise VisionError("rack corners are crossed or non-convex")
    side_lengths = np.linalg.norm(
        np.roll(corners, -1, axis=0) - corners,
        axis=1,
    )
    if float(side_lengths.min()) < 2.0:
        raise VisionError("rack corner side is too short")
    for first, second, name in ((0, 2, "long"), (1, 3, "short")):
        ratio = float(
            max(side_lengths[first], side_lengths[second])
            / min(side_lengths[first], side_lengths[second])
        )
        if ratio > maximum_opposite_side_ratio:
            raise VisionError(
                f"rack {name}-side ratio {ratio:.3f} exceeds "
                f"{maximum_opposite_side_ratio:.3f}"
            )

    image_to_unit = cv2.getPerspectiveTransform(
        corners,
        np.asarray([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32),
    )
    normalized = cv2.perspectiveTransform(
        _pixels_array(detection.screws).astype(np.float32).reshape(1, -1, 2),
        image_to_unit,
    ).reshape(-1, 2)
    low = screw_edge_margin
    high = 1.0 - screw_edge_margin
    expected_quadrants = ((0, 0), (1, 0), (1, 1), (0, 1))
    for index, ((x, y), (right, lower)) in enumerate(
        zip(normalized, expected_quadrants)
    ):
        if not low <= float(x) <= high or not low <= float(y) <= high:
            raise VisionError(
                f"{RACK_KEYPOINT_NAMES[index + 4]} is outside the rack interior"
            )
        if (float(x) >= 0.5) != bool(right) or (float(y) >= 0.5) != bool(lower):
            raise VisionError(
                f"{RACK_KEYPOINT_NAMES[index + 4]} is in the wrong rack quadrant"
            )


def validate_k0_red_marker(
    color_image: object,
    detection: RackPoseDetection,
    *,
    patch_radius_px: int,
    minimum_red_ratio: float,
    minimum_ratio_margin: float,
    minimum_saturation: int,
    minimum_value: int,
) -> tuple[float, float, float, float]:
    """Confirm the red orientation dot is beside screw_k0 and no other screw."""
    image = np.asarray(color_image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise VisionError("K0 red-marker input must be a BGR image")
    if patch_radius_px <= 0:
        raise VisionError("K0 red-marker patch radius must be positive")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red = (
        ((hsv[:, :, 0] <= 10) | (hsv[:, :, 0] >= 170))
        & (hsv[:, :, 1] >= int(minimum_saturation))
        & (hsv[:, :, 2] >= int(minimum_value))
    )
    scores = []
    radius = int(patch_radius_px)
    height, width = red.shape
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    circle = xx * xx + yy * yy <= radius * radius
    for screw in detection.screws:
        u = int(round(screw.u))
        v = int(round(screw.v))
        left, right = max(0, u - radius), min(width, u + radius + 1)
        top, bottom = max(0, v - radius), min(height, v + radius + 1)
        if left >= right or top >= bottom:
            raise VisionError("screw keypoint is outside the image")
        circle_crop = circle[
            top - (v - radius) : bottom - (v - radius),
            left - (u - radius) : right - (u - radius),
        ]
        values = red[top:bottom, left:right][circle_crop]
        scores.append(float(values.mean()) if values.size else 0.0)
    if scores[0] < float(minimum_red_ratio):
        raise VisionError(
            f"screw_k0 red-dot ratio {scores[0]:.3f} is below "
            f"{minimum_red_ratio:.3f}"
        )
    strongest_other = max(scores[1:])
    if scores[0] - strongest_other < float(minimum_ratio_margin):
        raise VisionError(
            "K0 red orientation is ambiguous: "
            f"screw_k0={scores[0]:.3f}, strongest other={strongest_other:.3f}"
        )
    return tuple(scores)  # type: ignore[return-value]


def fuse_rack_pose_detections(
    detections: Sequence[RackPoseDetection],
    *,
    attempted_frames: int,
    minimum_inlier_frames: int,
    maximum_frame_residual_px: float,
    maximum_keypoint_spread_px: float,
    minimum_area_px2: float,
    maximum_opposite_side_ratio: float,
    screw_edge_margin: float,
) -> RackPoseStability:
    """Median-fuse a static rack and reject whole-frame and point jitter."""
    if len(detections) < minimum_inlier_frames:
        raise VisionError(
            f"valid rack pose frames {len(detections)} < {minimum_inlier_frames}"
        )
    arrays = np.asarray(
        [
            [[item.pixel.u, item.pixel.v] for item in detection.keypoints]
            for detection in detections
        ],
        dtype=np.float64,
    )
    center = np.median(arrays, axis=0)
    residuals = np.sqrt(
        np.mean(
            np.sum((arrays - center[None, :, :]) ** 2, axis=2),
            axis=1,
        )
    )
    inliers = np.flatnonzero(residuals <= float(maximum_frame_residual_px))
    if len(inliers) < minimum_inlier_frames:
        raise VisionError(
            f"rack pose inlier frames {len(inliers)} < {minimum_inlier_frames}; "
            f"maximum frame residual is {float(residuals.max()):.2f}px"
        )
    inlier_arrays = arrays[inliers]
    fused_xy = np.median(inlier_arrays, axis=0)
    spreads = np.max(
        np.linalg.norm(inlier_arrays - fused_xy[None, :, :], axis=2),
        axis=0,
    )
    maximum_spread = float(spreads.max())
    if maximum_spread > float(maximum_keypoint_spread_px):
        worst = int(np.argmax(spreads))
        raise VisionError(
            f"{RACK_KEYPOINT_NAMES[worst]} spread {maximum_spread:.2f}px exceeds "
            f"{maximum_keypoint_spread_px:.2f}px"
        )
    confidences = np.asarray(
        [[item.confidence for item in detection.keypoints] for detection in detections],
        dtype=np.float64,
    )[inliers]
    boxes = np.asarray(
        [[d.box.x1, d.box.y1, d.box.x2, d.box.y2] for d in detections],
        dtype=np.float64,
    )[inliers]
    fused_box = np.median(boxes, axis=0)
    fused = RackPoseDetection(
        confidence=float(
            np.median([detections[index].confidence for index in inliers])
        ),
        box=Box(*(float(value) for value in fused_box)),
        keypoints=tuple(
            RackKeypoint(
                name,
                Pixel(float(point[0]), float(point[1])),
                float(np.median(confidences[:, index])),
            )
            for index, (name, point) in enumerate(zip(RACK_KEYPOINT_NAMES, fused_xy))
        ),
    )
    validate_rack_pose_geometry(
        fused,
        minimum_area_px2=minimum_area_px2,
        maximum_opposite_side_ratio=maximum_opposite_side_ratio,
        screw_edge_margin=screw_edge_margin,
    )
    return RackPoseStability(
        detection=fused,
        inlier_indices=tuple(int(value) for value in inliers),
        attempted_frames=int(attempted_frames),
        maximum_keypoint_spread_px=maximum_spread,
        maximum_frame_residual_px=float(residuals[inliers].max()),
    )


def corner_homography(detection: RackPoseDetection) -> tuple[np.ndarray, np.ndarray]:
    """Return image<-unit-rack and unit-rack<-image homographies."""
    corners = _pixels_array(detection.corners).astype(np.float32)
    unit = np.asarray([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    image_from_unit = cv2.getPerspectiveTransform(unit, corners)
    unit_from_image = cv2.getPerspectiveTransform(corners, unit)
    return image_from_unit, unit_from_image


def transform_pixels(points: Sequence[Pixel], homography: object) -> tuple[Pixel, ...]:
    matrix = np.asarray(homography, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise VisionError("homography must be a finite 3x3 matrix")
    array = _pixels_array(points).astype(np.float64).reshape(1, -1, 2)
    transformed = cv2.perspectiveTransform(array, matrix).reshape(-1, 2)
    if not np.isfinite(transformed).all():
        raise VisionError("homography produced invalid pixels")
    return tuple(Pixel(float(x), float(y)) for x, y in transformed)


def _pixels_array(points: Sequence[Pixel]) -> np.ndarray:
    return np.asarray([[point.u, point.v] for point in points], dtype=np.float64)
