"""Four-screw rack-surface detection, orientation and geometry checks.

The detection model returns four interchangeable screw boxes.  Their centers
define the rack surface.  A separate white marker beside physical corner K0
removes the rotational ambiguity; the long edge from K0 defines K0 -> K1.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Mapping, Protocol, Sequence

import cv2
import numpy as np

from tube_grabber.core.errors import VisionError
from tube_grabber.core.models import Box, Detection, Pixel


SCREW_MODEL_CLASSES = {0: "item"}
RACK_KEYPOINT_NAMES = (
    "k0",
    "k1",
    "k2",
    "k3",
)


@dataclass(frozen=True)
class RackPoseQualityConfig:
    minimum_rack_area_px2: float
    maximum_opposite_side_ratio: float

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


@dataclass(frozen=True)
class ScrewMarkerConfig:
    """Pixel-domain limits for finding the extra white K0 marker."""

    minimum_value: int
    maximum_saturation: int
    search_radius_factor: float
    screw_exclusion_scale: float
    minimum_area_px2: int
    maximum_area_ratio: float
    minimum_score_ratio: float
    minimum_rack_aspect_ratio: float

    def __post_init__(self) -> None:
        if not 0 <= int(self.minimum_value) <= 255:
            raise ValueError("marker minimum_value must be in [0, 255]")
        if not 0 <= int(self.maximum_saturation) <= 255:
            raise ValueError("marker maximum_saturation must be in [0, 255]")
        if not 0.1 <= float(self.search_radius_factor) <= 1.0:
            raise ValueError("marker search_radius_factor must be in [0.1, 1.0]")
        if float(self.screw_exclusion_scale) < 1.0:
            raise ValueError("marker screw_exclusion_scale must be at least one")
        if int(self.minimum_area_px2) <= 0:
            raise ValueError("marker minimum_area_px2 must be positive")
        if float(self.maximum_area_ratio) <= 0.0:
            raise ValueError("marker maximum_area_ratio must be positive")
        if float(self.minimum_score_ratio) <= 1.0:
            raise ValueError("marker minimum_score_ratio must exceed one")
        if float(self.minimum_rack_aspect_ratio) <= 1.0:
            raise ValueError("minimum_rack_aspect_ratio must exceed one")


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
                "rack keypoints must follow the fixed four-corner contract"
            )

    @property
    def corners(self) -> tuple[Pixel, Pixel, Pixel, Pixel]:
        return tuple(  # type: ignore[return-value]
            item.pixel for item in self.keypoints[:4]
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


class YoloScrewRackDetector:
    """Use four screw-box centers and the extra white marker as rack corners."""

    def __init__(
        self,
        model_path: str | Path,
        class_names: Mapping[int, str],
        confidence: float,
        iou: float,
        image_size: int,
        device: str,
        marker: ScrewMarkerConfig,
    ) -> None:
        names = {int(key): str(value) for key, value in class_names.items()}
        if names != SCREW_MODEL_CLASSES:
            raise VisionError(f"screw YOLO classes must be {SCREW_MODEL_CLASSES}")
        for name, value in (("confidence", confidence), ("iou", iou)):
            if not 0.0 < float(value) <= 1.0:
                raise VisionError(f"{name} must be between 0 and 1")
        if int(image_size) <= 0:
            raise VisionError("screw image_size must be positive")
        self._model_path = str(model_path)
        self._class_names = names
        self._confidence = float(confidence)
        self._iou = float(iou)
        self._image_size = int(image_size)
        self._device = str(device)
        self._marker = marker
        self._model: object | None = None

    def detect_screws(self, color_image: object) -> list[Detection]:
        """Return raw screw centers so the visual test can always draw them."""
        image = np.asarray(color_image)
        if image.ndim != 3 or image.shape[2] != 3:
            raise VisionError("screw detector input must be a BGR image")
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
            raise VisionError(f"screw YOLO inference failed: {exc}") from exc
        if not results or results[0].boxes is None:
            return []
        boxes = results[0].boxes
        detections: list[Detection] = []
        for index in range(len(boxes)):
            class_id = int(boxes.cls[index].item())
            if class_id not in self._class_names:
                raise VisionError(
                    f"screw YOLO returned unsupported class id {class_id}"
                )
            x1, y1, x2, y2 = (
                float(value) for value in boxes.xyxy[index].tolist()
            )
            detections.append(
                Detection(
                    label="screw",
                    confidence=float(boxes.conf[index].item()),
                    box=Box(x1, y1, x2, y2),
                )
            )
        return detections

    def build_rack_detection(
        self,
        color_image: object,
        screws: Sequence[Detection],
    ) -> RackPoseDetection:
        return rack_detection_from_screws(
            color_image,
            screws,
            marker=self._marker,
        )

    def detect(self, color_image: object) -> RackPoseDetection:
        screws = self.detect_screws(color_image)
        return self.build_rack_detection(color_image, screws)

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
            model = YOLO(self._model_path, task="detect")
        except Exception as exc:
            raise VisionError(
                f"cannot load screw model {self._model_path}: {exc}"
            ) from exc
        model_names = {
            int(key): str(value) for key, value in dict(model.names).items()
        }
        if model_names != self._class_names:
            raise VisionError(
                f"screw model classes are {model_names}, expected {self._class_names}"
            )
        self._model = model
        return model


def rack_detection_from_screws(
    color_image: object,
    screws: Sequence[Detection],
    *,
    marker: ScrewMarkerConfig,
) -> RackPoseDetection:
    """Order four unordered screw centers as K0..K3 using the white marker."""
    image = np.asarray(color_image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise VisionError("screw detector input must be a BGR image")
    if len(screws) != 4:
        raise VisionError(f"expected exactly four rack screws, got {len(screws)}")
    if any(item.label != "screw" for item in screws):
        raise VisionError("rack screw detections contain an unsupported label")

    centers = np.asarray(
        [[item.box.center.u, item.box.center.v] for item in screws],
        dtype=np.float64,
    )
    centroid = centers.mean(axis=0)
    # In image coordinates (+v points down), increasing atan2 angle is the
    # visual clockwise direction.
    indices = np.argsort(
        np.arctan2(
            centers[:, 1] - centroid[1],
            centers[:, 0] - centroid[0],
        )
    )
    ordered = [screws[int(index)] for index in indices]
    polygon = np.asarray(
        [[item.box.center.u, item.box.center.v] for item in ordered],
        dtype=np.float32,
    )
    if not cv2.isContourConvex(polygon):
        raise VisionError("four screw centers do not form a convex rack surface")

    try:
        k0_index = _find_marker_corner(image, ordered, marker)
    except VisionError as error:
        if "white K0 marker was not found" not in str(error):
            raise
        # Auto-pick only needs a stable grid, not a semantic slot label.  Keep
        # the grid usable when the marker is temporarily hidden by selecting
        # the image-space top-left screw as the deterministic origin.
        k0_index = min(
            range(len(ordered)),
            key=lambda index: (
                ordered[index].box.center.u + ordered[index].box.center.v
            ),
        )
    clockwise = ordered[k0_index:] + ordered[:k0_index]
    next_length = _center_distance(clockwise[0], clockwise[1])
    previous_length = _center_distance(clockwise[0], clockwise[-1])
    long_length = max(next_length, previous_length)
    short_length = min(next_length, previous_length)
    if short_length <= 1.0 or (
        long_length / short_length < marker.minimum_rack_aspect_ratio
    ):
        raise VisionError(
            f"screw rack aspect ratio {long_length / max(short_length, 1e-9):.2f} "
            f"is below {marker.minimum_rack_aspect_ratio:.2f}"
        )
    # Always make the six-column direction K0 -> K1.  Reverse the winding if
    # the clockwise neighbour is the short (two-row) edge.
    if previous_length > next_length:
        clockwise = [clockwise[0], clockwise[3], clockwise[2], clockwise[1]]

    x1 = min(item.box.x1 for item in clockwise)
    y1 = min(item.box.y1 for item in clockwise)
    x2 = max(item.box.x2 for item in clockwise)
    y2 = max(item.box.y2 for item in clockwise)
    return RackPoseDetection(
        confidence=float(min(item.confidence for item in clockwise)),
        box=Box(x1, y1, x2, y2),
        keypoints=tuple(
            RackKeypoint(name, item.box.center, item.confidence)
            for name, item in zip(RACK_KEYPOINT_NAMES, clockwise)
        ),
    )


def _find_marker_corner(
    image: np.ndarray,
    screws: Sequence[Detection],
    config: ScrewMarkerConfig,
) -> int:
    """Find the one nearby white component on the green rack surface."""
    blurred = cv2.GaussianBlur(image, (3, 3), 0.0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    white = np.where(
        (hsv[:, :, 1] <= int(config.maximum_saturation))
        & (hsv[:, :, 2] >= int(config.minimum_value)),
        255,
        0,
    ).astype(np.uint8)
    white = cv2.morphologyEx(
        white,
        cv2.MORPH_OPEN,
        np.ones((3, 3), dtype=np.uint8),
    )
    for item in screws:
        box = item.box
        center = box.center
        half_width = 0.5 * (box.x2 - box.x1) * config.screw_exclusion_scale
        half_height = 0.5 * (box.y2 - box.y1) * config.screw_exclusion_scale
        cv2.rectangle(
            white,
            (
                max(0, int(np.floor(center.u - half_width))),
                max(0, int(np.floor(center.v - half_height))),
            ),
            (
                min(image.shape[1] - 1, int(np.ceil(center.u + half_width))),
                min(image.shape[0] - 1, int(np.ceil(center.v + half_height))),
            ),
            0,
            -1,
        )

    count, _, stats, component_centers = cv2.connectedComponentsWithStats(
        white,
        connectivity=8,
    )
    screw_area = float(
        np.median(
            [
                (item.box.x2 - item.box.x1) * (item.box.y2 - item.box.y1)
                for item in screws
            ]
        )
    )
    maximum_area = max(
        float(config.minimum_area_px2),
        screw_area * float(config.maximum_area_ratio),
    )
    centers = np.asarray(
        [[item.box.center.u, item.box.center.v] for item in screws],
        dtype=np.float64,
    )
    rack_polygon = centers.astype(np.float32)
    side_lengths = np.linalg.norm(np.roll(centers, -1, axis=0) - centers, axis=1)
    search_radius = float(np.sort(side_lengths)[:2].mean()) * float(
        config.search_radius_factor
    )
    rack_center = centers.mean(axis=0)
    scores = np.zeros(4, dtype=np.float64)
    for component in range(1, count):
        area = float(stats[component, cv2.CC_STAT_AREA])
        if not float(config.minimum_area_px2) <= area <= maximum_area:
            continue
        point = np.asarray(component_centers[component], dtype=np.float64)
        # The physical K0 tape is on the green rack surface.  Reject labels,
        # reflections and other white objects outside the screw quadrilateral
        # before scoring them as possible orientation markers.
        if cv2.pointPolygonTest(
            rack_polygon,
            (float(point[0]), float(point[1])),
            False,
        ) < 0:
            continue
        distances = np.linalg.norm(centers - point[None, :], axis=1)
        corner = int(np.argmin(distances))
        distance = float(distances[corner])
        if distance > search_radius:
            continue
        inward = rack_center - centers[corner]
        marker_vector = point - centers[corner]
        denominator = float(np.linalg.norm(inward) * np.linalg.norm(marker_vector))
        alignment = (
            float(np.dot(inward, marker_vector) / denominator)
            if denominator > 1e-9
            else 0.0
        )
        direction_bonus = 1.0 + 1.5 * max(0.0, alignment)
        distance_weight = 1.0 / (1.0 + distance / max(search_radius, 1.0))
        score = area * direction_bonus * distance_weight
        scores[corner] = max(scores[corner], score)

    best = int(np.argmax(scores))
    best_score = float(scores[best])
    if best_score <= 0.0:
        raise VisionError("white K0 marker was not found beside any rack screw")
    runner_up = float(np.partition(scores, -2)[-2])
    if runner_up > 0.0 and best_score / runner_up < config.minimum_score_ratio:
        raise VisionError(
            "white K0 marker is ambiguous between screw corners "
            f"(score ratio {best_score / runner_up:.2f} < "
            f"{config.minimum_score_ratio:.2f})"
        )
    return best


def _center_distance(first: Detection, second: Detection) -> float:
    return float(
        np.hypot(
            first.box.center.u - second.box.center.u,
            first.box.center.v - second.box.center.v,
        )
    )


def validate_rack_pose_geometry(
    detection: RackPoseDetection,
    *,
    minimum_area_px2: float,
    maximum_opposite_side_ratio: float,
) -> None:
    """Reject crossed, collapsed or implausibly distorted corner layouts."""
    if not isfinite(float(minimum_area_px2)) or minimum_area_px2 <= 0.0:
        raise ValueError("minimum_area_px2 must be positive")
    if maximum_opposite_side_ratio < 1.0:
        raise ValueError("maximum_opposite_side_ratio must be at least one")
    corners = _pixels_array(detection.corners).astype(np.float32)
    if not cv2.isContourConvex(corners):
        raise VisionError("rack corners are crossed or non-convex")
    area = abs(float(cv2.contourArea(corners)))
    if area < float(minimum_area_px2):
        raise VisionError(
            f"rack corner area {area:.1f}px^2 is below {minimum_area_px2:.1f}px^2"
        )
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

def fuse_rack_pose_detections(
    detections: Sequence[RackPoseDetection],
    *,
    attempted_frames: int,
    minimum_inlier_frames: int,
    maximum_frame_residual_px: float,
    maximum_keypoint_spread_px: float,
    minimum_area_px2: float,
    maximum_opposite_side_ratio: float,
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
