"""Final one-rack perception pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite

import numpy as np

from tube_grabber.core.errors import VisionError
from tube_grabber.core.models import (
    CameraFrame,
    Detection,
    Occupancy,
    Point3D,
    Pose6D,
    RackObservation,
    SlotAddress,
    SlotObservation,
)
from tube_grabber.core.ports import DetectorPort
from tube_grabber.vision.depth import sample_depth_mm
from tube_grabber.vision.geometry import (
    base_from_camera,
    pixel_depth_to_base,
    pixel_ray_to_horizontal_plane,
    validate_transform,
)
from tube_grabber.vision.grid import GridMapper
from tube_grabber.vision.marker import K0MarkerDetector


EMPTY_HOLE = "empty_hole"
TUBE_CAP = "tube_cap"
SUPPORTED_LABELS = {EMPTY_HOLE, TUBE_CAP}


class RackVision:
    """Convert one aligned RGB-D frame into twelve safe slot observations."""

    def __init__(
        self,
        detector: DetectorPort,
        marker_detector: K0MarkerDetector,
        grid_mapper: GridMapper,
        hand_eye_end_from_camera: object,
        rack_marker_colors: Mapping[str, str],
        fallback_plane_z_mm: Mapping[str, float | None],
        required_detection_count: int,
        depth_window_px: int,
        depth_min_mm: float,
        depth_max_mm: float,
        cap_top_above_rack_mm: float,
        maximum_cap_z_deviation_mm: float,
        maximum_plane_calibration_error_mm: float,
    ) -> None:
        if required_detection_count != 12:
            raise VisionError("a 2x6 rack requires exactly 12 detections")
        if not isfinite(float(cap_top_above_rack_mm)) or cap_top_above_rack_mm <= 0.0:
            raise VisionError("cap_top_above_rack_mm must be positive")
        if (
            not isfinite(float(maximum_cap_z_deviation_mm))
            or maximum_cap_z_deviation_mm <= 0.0
        ):
            raise VisionError("maximum_cap_z_deviation_mm must be positive")
        if (
            not isfinite(float(maximum_plane_calibration_error_mm))
            or maximum_plane_calibration_error_mm <= 0.0
        ):
            raise VisionError(
                "maximum_plane_calibration_error_mm must be positive"
            )

        colors = {str(key): str(value) for key, value in rack_marker_colors.items()}
        if not colors or any(color not in {"red", "green"} for color in colors.values()):
            raise VisionError("each rack marker color must be red or green")
        fallback = dict(fallback_plane_z_mm)
        if set(fallback) != set(colors):
            raise VisionError("fallback plane keys must match rack marker keys")
        for rack_id, value in fallback.items():
            if value is not None and not isfinite(float(value)):
                raise VisionError(f"fallback plane for {rack_id} must be finite")

        self._detector = detector
        self._marker_detector = marker_detector
        self._grid_mapper = grid_mapper
        self._hand_eye = validate_transform(
            hand_eye_end_from_camera,
            "hand-eye transform",
        )
        self._rack_marker_colors = colors
        self._fallback_plane_z_mm = fallback
        self._required_detection_count = required_detection_count
        self._depth_window_px = int(depth_window_px)
        self._depth_min_mm = float(depth_min_mm)
        self._depth_max_mm = float(depth_max_mm)
        self._cap_top_above_rack_mm = float(cap_top_above_rack_mm)
        self._maximum_cap_z_deviation_mm = float(
            maximum_cap_z_deviation_mm
        )
        self._maximum_plane_calibration_error_mm = float(
            maximum_plane_calibration_error_mm
        )

    def observe(
        self,
        frame: CameraFrame,
        expected_rack_id: str,
        arm_pose: Pose6D,
    ) -> RackObservation:
        if expected_rack_id not in self._rack_marker_colors:
            raise VisionError(f"unknown rack id: {expected_rack_id}")

        detections = self._detector.detect(frame.color)
        self._validate_detections(detections)

        marker_color = self._rack_marker_colors[expected_rack_id]
        markers = self._marker_detector.detect(frame.color, marker_color)
        mapping = self._grid_mapper.map(detections, markers, expected_rack_id)
        transform = base_from_camera(arm_pose, self._hand_eye)

        cap_points: dict[SlotAddress, Point3D] = {}
        for mapped in mapping.slots:
            if mapped.detection.label != TUBE_CAP:
                continue
            depth_mm = sample_depth_mm(
                frame.depth_mm,
                mapped.detection.box.center,
                self._depth_window_px,
                self._depth_min_mm,
                self._depth_max_mm,
            )
            cap_points[mapped.address] = pixel_depth_to_base(
                mapped.detection.box.center,
                depth_mm,
                frame.intrinsics,
                transform,
            )

        plane_z_mm = self._estimate_plane(expected_rack_id, cap_points)
        slots: list[SlotObservation] = []
        for mapped in mapping.slots:
            detection = mapped.detection
            if detection.label == TUBE_CAP:
                slots.append(
                    SlotObservation(
                        address=mapped.address,
                        occupancy=Occupancy.OCCUPIED,
                        confidence=detection.confidence,
                        pixel=detection.box.center,
                        cap_top_base=cap_points[mapped.address],
                    )
                )
            else:
                hole_point = pixel_ray_to_horizontal_plane(
                    detection.box.center,
                    plane_z_mm,
                    frame.intrinsics,
                    transform,
                )
                slots.append(
                    SlotObservation(
                        address=mapped.address,
                        occupancy=Occupancy.EMPTY,
                        confidence=detection.confidence,
                        pixel=detection.box.center,
                        hole_on_plane_base=hole_point,
                    )
                )

        return RackObservation(
            rack_id=expected_rack_id,
            marker=mapping.marker,
            plane_z_mm=plane_z_mm,
            slots=tuple(slots),
            timestamp_ms=frame.timestamp_ms,
        )

    def _validate_detections(self, detections: list[Detection]) -> None:
        if len(detections) != self._required_detection_count:
            raise VisionError(
                f"expected exactly {self._required_detection_count} detections, "
                f"got {len(detections)}"
            )
        unsupported = sorted(
            {detection.label for detection in detections} - SUPPORTED_LABELS
        )
        if unsupported:
            raise VisionError(f"unsupported detection labels: {unsupported}")

    def _estimate_plane(
        self,
        rack_id: str,
        cap_points: Mapping[SlotAddress, Point3D],
    ) -> float:
        if cap_points:
            cap_z_values = np.asarray(
                [point.z_mm for point in cap_points.values()],
                dtype=np.float64,
            )
            median_cap_z_mm = float(np.median(cap_z_values))
            maximum_deviation_mm = float(
                np.max(np.abs(cap_z_values - median_cap_z_mm))
            )
            if maximum_deviation_mm > self._maximum_cap_z_deviation_mm:
                raise VisionError(
                    "tube-cap depths disagree: maximum Z deviation is "
                    f"{maximum_deviation_mm:.2f} mm; limit is "
                    f"{self._maximum_cap_z_deviation_mm:.2f} mm"
                )

            measured_plane_z_mm = (
                median_cap_z_mm - self._cap_top_above_rack_mm
            )
            fallback = self._fallback_plane_z_mm[rack_id]
            if fallback is not None:
                expected_cap_z_mm = (
                    float(fallback) + self._cap_top_above_rack_mm
                )
                maximum_calibrated_cap_error_mm = float(
                    np.max(np.abs(cap_z_values - expected_cap_z_mm))
                )
                if (
                    maximum_calibrated_cap_error_mm
                    > self._maximum_plane_calibration_error_mm
                ):
                    raise VisionError(
                        f"{rack_id} tube-cap Z differs from calibrated rack "
                        f"geometry by {maximum_calibrated_cap_error_mm:.2f} mm; "
                        f"limit is {self._maximum_plane_calibration_error_mm:.2f} mm"
                    )
                calibration_error_mm = abs(
                    measured_plane_z_mm - float(fallback)
                )
                if (
                    calibration_error_mm
                    > self._maximum_plane_calibration_error_mm
                ):
                    raise VisionError(
                        f"{rack_id} measured rack plane differs from calibration "
                        f"by {calibration_error_mm:.2f} mm; limit is "
                        f"{self._maximum_plane_calibration_error_mm:.2f} mm"
                    )
                return float(fallback)
            return measured_plane_z_mm

        fallback = self._fallback_plane_z_mm[rack_id]
        if fallback is None:
            raise VisionError(
                f"{rack_id} has no tube cap depth and no calibrated fallback plane"
            )
        return float(fallback)
