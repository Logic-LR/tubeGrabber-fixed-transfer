"""Multi-frame rack pose, plane, slot-grid and tube-cap perception."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite
from typing import Mapping, Sequence

import cv2
import numpy as np

from tube_grabber.core.errors import VisionError
from tube_grabber.core.models import (
    CameraFrame,
    Detection,
    Occupancy,
    Pixel,
    Point3D,
    Pose6D,
    RackObservation,
    SlotAddress,
    SlotObservation,
)
from tube_grabber.core.ports import DetectorPort
from tube_grabber.vision.depth import sample_depth_mm
from tube_grabber.vision.geometry import base_from_camera, pixel_depth_to_base
from tube_grabber.vision.plane import RackPlane, RackPlaneFitConfig, fit_rack_plane
from tube_grabber.vision.rack_calibration import RackSlotCalibration
from tube_grabber.vision.rack_pose import (
    RackPoseDetection,
    RackPoseDetectorPort,
    RackPoseQualityConfig,
    RackPoseStability,
    fuse_rack_pose_detections,
    validate_k0_red_marker,
    validate_rack_pose_geometry,
)


TUBE_CAP = "tube_cap"


@dataclass(frozen=True)
class CapturedRackFrame:
    frame: CameraFrame
    arm_pose: Pose6D


@dataclass(frozen=True)
class RackStabilityConfig:
    capture_frames: int
    minimum_inlier_frames: int
    maximum_frame_residual_px: float
    maximum_keypoint_spread_px: float
    minimum_occupancy_agreement: float
    maximum_slot_position_spread_mm: float
    maximum_cap_position_spread_mm: float

    def __post_init__(self) -> None:
        if (
            self.minimum_inlier_frames < 2
            or self.capture_frames < self.minimum_inlier_frames
        ):
            raise ValueError(
                "capture_frames must be >= minimum_inlier_frames >= 2"
            )
        for name in (
            "maximum_frame_residual_px",
            "maximum_keypoint_spread_px",
            "maximum_slot_position_spread_mm",
            "maximum_cap_position_spread_mm",
        ):
            value = float(getattr(self, name))
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be a finite positive number")
        if not 0.5 < self.minimum_occupancy_agreement <= 1.0:
            raise ValueError("minimum_occupancy_agreement must be in (0.5, 1]")


@dataclass(frozen=True)
class RackMatchingConfig:
    depth_window_px: int
    depth_min_mm: float
    depth_max_mm: float
    cap_top_above_rack_mm: float
    maximum_cap_height_error_mm: float
    cap_slot_max_distance_factor: float
    maximum_calibration_keypoint_shift_ratio: float

    def __post_init__(self) -> None:
        if self.depth_window_px <= 0 or self.depth_window_px % 2 == 0:
            raise ValueError("depth_window_px must be a positive odd number")
        if (
            not isfinite(float(self.depth_min_mm))
            or not isfinite(float(self.depth_max_mm))
            or self.depth_min_mm <= 0.0
            or self.depth_max_mm <= self.depth_min_mm
        ):
            raise ValueError("rack vision depth range is invalid")
        for name in (
            "cap_top_above_rack_mm",
            "maximum_cap_height_error_mm",
            "cap_slot_max_distance_factor",
            "maximum_calibration_keypoint_shift_ratio",
        ):
            value = float(getattr(self, name))
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be a finite positive number")


@dataclass(frozen=True)
class _PoseFrame:
    source_index: int
    sample: CapturedRackFrame
    pose: RackPoseDetection


@dataclass(frozen=True)
class _FrameObservation:
    source_index: int
    plane: RackPlane
    slots: tuple[SlotObservation, ...]
    timestamp_ms: float


class PoseRackVision:
    """Final vision path: one pose model, one cap detector, no hole detector."""

    def __init__(
        self,
        *,
        rack_pose_detector: RackPoseDetectorPort,
        cap_detector: DetectorPort,
        calibrations: Mapping[str, RackSlotCalibration | None],
        hand_eye_end_from_camera: object,
        pose_quality: RackPoseQualityConfig,
        stability: RackStabilityConfig,
        plane_config: RackPlaneFitConfig,
        matching: RackMatchingConfig,
    ) -> None:
        if not calibrations:
            raise VisionError("at least one rack calibration entry is required")

        self._pose_detector = rack_pose_detector
        self._cap_detector = cap_detector
        self._calibrations = dict(calibrations)
        self._hand_eye = hand_eye_end_from_camera
        self._pose_quality = pose_quality
        self._stability = stability
        self._plane_config = plane_config
        self._matching = matching

    @property
    def capture_frames(self) -> int:
        return self._stability.capture_frames

    def detect_stable_pose(
        self,
        samples: Sequence[CapturedRackFrame],
    ) -> tuple[RackPoseStability, tuple[_PoseFrame, ...]]:
        """Run pose inference on every frame, then median-filter valid frames."""
        valid: list[_PoseFrame] = []
        failures: list[str] = []
        for index, sample in enumerate(samples):
            try:
                detection = self._pose_detector.detect(sample.frame.color)
                validate_rack_pose_geometry(
                    detection,
                    minimum_area_px2=self._pose_quality.minimum_rack_area_px2,
                    maximum_opposite_side_ratio=(
                        self._pose_quality.maximum_opposite_side_ratio
                    ),
                    screw_edge_margin=self._pose_quality.screw_edge_margin,
                )
                validate_k0_red_marker(
                    sample.frame.color,
                    detection,
                    patch_radius_px=self._pose_quality.k0_red_patch_radius_px,
                    minimum_red_ratio=self._pose_quality.k0_red_minimum_ratio,
                    minimum_ratio_margin=(
                        self._pose_quality.k0_red_minimum_ratio_margin
                    ),
                    minimum_saturation=(
                        self._pose_quality.k0_red_minimum_saturation
                    ),
                    minimum_value=self._pose_quality.k0_red_minimum_value,
                )
                valid.append(_PoseFrame(index, sample, detection))
            except VisionError as exc:
                failures.append(f"frame {index}: {exc}")
        if len(valid) < self._stability.minimum_inlier_frames:
            detail = "; ".join(failures[:3])
            raise VisionError(
                "rack pose valid frames "
                f"{len(valid)} < {self._stability.minimum_inlier_frames}"
                + (f"; {detail}" if detail else "")
            )
        relative = fuse_rack_pose_detections(
            [item.pose for item in valid],
            attempted_frames=len(samples),
            minimum_inlier_frames=self._stability.minimum_inlier_frames,
            maximum_frame_residual_px=self._stability.maximum_frame_residual_px,
            maximum_keypoint_spread_px=self._stability.maximum_keypoint_spread_px,
            minimum_area_px2=self._pose_quality.minimum_rack_area_px2,
            maximum_opposite_side_ratio=(
                self._pose_quality.maximum_opposite_side_ratio
            ),
            screw_edge_margin=self._pose_quality.screw_edge_margin,
        )
        inlier_frames = tuple(valid[index] for index in relative.inlier_indices)
        stability = RackPoseStability(
            detection=relative.detection,
            inlier_indices=tuple(item.source_index for item in inlier_frames),
            attempted_frames=len(samples),
            maximum_keypoint_spread_px=relative.maximum_keypoint_spread_px,
            maximum_frame_residual_px=relative.maximum_frame_residual_px,
        )
        return stability, inlier_frames

    def observe(
        self,
        samples: Sequence[CapturedRackFrame],
        expected_rack_id: str,
        *,
        ignored_elevated_slot: SlotAddress | None = None,
    ) -> RackObservation:
        if expected_rack_id not in self._calibrations:
            raise VisionError(f"unknown rack id: {expected_rack_id}")
        if (
            ignored_elevated_slot is not None
            and ignored_elevated_slot.rack_id != expected_rack_id
        ):
            raise VisionError(
                "ignored elevated cap slot must belong to the observed rack"
            )
        calibration = self._calibrations[expected_rack_id]
        if calibration is None:
            raise VisionError(
                f"{expected_rack_id} has no two-circle calibration; "
                "run calibrate-rack"
            )
        if len(samples) < self._stability.minimum_inlier_frames:
            raise VisionError(
                "captured frames "
                f"{len(samples)} < {self._stability.minimum_inlier_frames}"
            )
        stability, inlier_frames = self.detect_stable_pose(samples)
        calibration.validate_reference_pose(
            stability.detection,
            self._matching.maximum_calibration_keypoint_shift_ratio,
        )
        observations: list[_FrameObservation] = []
        failures: list[str] = []
        for item in inlier_frames:
            try:
                observations.append(
                    self._observe_frame(
                        item,
                        calibration,
                        ignored_elevated_slot=ignored_elevated_slot,
                    )
                )
            except VisionError as exc:
                failures.append(f"frame {item.source_index}: {exc}")
        if len(observations) < self._stability.minimum_inlier_frames:
            detail = "; ".join(failures[:3])
            raise VisionError(
                "complete rack frames "
                f"{len(observations)} < {self._stability.minimum_inlier_frames}"
                + (f"; {detail}" if detail else "")
            )
        return self._aggregate(expected_rack_id, observations, stability)

    def _observe_frame(
        self,
        item: _PoseFrame,
        calibration: RackSlotCalibration,
        *,
        ignored_elevated_slot: SlotAddress | None,
    ) -> _FrameObservation:
        sample = item.sample
        caps = self._cap_detector.detect(sample.frame.color)
        unsupported = sorted({cap.label for cap in caps if cap.label != TUBE_CAP})
        if unsupported:
            raise VisionError(
                f"cap detector returned unsupported labels: {unsupported}"
            )
        transform = base_from_camera(sample.arm_pose, self._hand_eye)
        plane = fit_rack_plane(
            sample.frame,
            item.pose,
            transform,
            excluded_boxes=[cap.box for cap in caps],
            depth_min_mm=self._matching.depth_min_mm,
            depth_max_mm=self._matching.depth_max_mm,
            config=self._plane_config,
        )
        projected = calibration.project_slots(item.pose)
        cap_matches = _match_caps_to_slots(
            caps,
            projected,
            item.pose,
            maximum_distance_factor=self._matching.cap_slot_max_distance_factor,
        )
        slots: list[SlotObservation] = []
        for address, slot_pixel in projected:
            hole = plane.intersect(slot_pixel, sample.frame, transform)
            cap = cap_matches.get(address)
            if cap is None:
                slots.append(
                    SlotObservation(
                        address=address,
                        occupancy=Occupancy.EMPTY,
                        confidence=item.pose.confidence,
                        pixel=slot_pixel,
                        hole_on_plane_base=hole,
                    )
                )
                continue
            depth_mm = sample_depth_mm(
                sample.frame.depth_mm,
                cap.box.center,
                self._matching.depth_window_px,
                self._matching.depth_min_mm,
                self._matching.depth_max_mm,
            )
            cap_point = pixel_depth_to_base(
                cap.box.center,
                depth_mm,
                sample.frame.intrinsics,
                transform,
            )
            cap_height = plane.signed_distance_mm(cap_point)
            height_error = abs(
                cap_height - self._matching.cap_top_above_rack_mm
            )
            if height_error > self._matching.maximum_cap_height_error_mm:
                if (
                    address == ignored_elevated_slot
                    and cap_height
                    > self._matching.cap_top_above_rack_mm
                    + self._matching.maximum_cap_height_error_mm
                ):
                    # During destination recheck, the one carried tube is
                    # deliberately parked above the requested empty slot.  Its
                    # elevated cap may project onto that slot, but it is not a
                    # seated occupant.  Every other high/low mismatch remains
                    # a hard failure.
                    slots.append(
                        SlotObservation(
                            address=address,
                            occupancy=Occupancy.EMPTY,
                            confidence=item.pose.confidence,
                            pixel=slot_pixel,
                            hole_on_plane_base=hole,
                        )
                    )
                    continue
                raise VisionError(
                    f"{address.text} cap height {cap_height:.2f}mm differs from "
                    f"{self._matching.cap_top_above_rack_mm:.2f}mm "
                    f"by {height_error:.2f}mm"
                )
            slots.append(
                SlotObservation(
                    address=address,
                    occupancy=Occupancy.OCCUPIED,
                    confidence=cap.confidence,
                    pixel=cap.box.center,
                    cap_top_base=cap_point,
                    hole_on_plane_base=hole,
                )
            )
        return _FrameObservation(
            source_index=item.source_index,
            plane=plane,
            slots=tuple(slots),
            timestamp_ms=sample.frame.timestamp_ms,
        )

    def _aggregate(
        self,
        rack_id: str,
        frames: Sequence[_FrameObservation],
        stability: RackPoseStability,
    ) -> RackObservation:
        slots: list[SlotObservation] = []
        maximum_position_spread = 0.0
        first = frames[0]
        for slot_index, reference in enumerate(first.slots):
            candidates = [frame.slots[slot_index] for frame in frames]
            if any(item.address != reference.address for item in candidates):
                raise VisionError("slot ordering changed across frames")
            occupied = [
                item
                for item in candidates
                if item.occupancy is Occupancy.OCCUPIED
            ]
            ratio = len(occupied) / len(candidates)
            if ratio >= self._stability.minimum_occupancy_agreement:
                state = Occupancy.OCCUPIED
            elif ratio <= 1.0 - self._stability.minimum_occupancy_agreement:
                state = Occupancy.EMPTY
            else:
                raise VisionError(
                    f"{reference.address.text} occupancy agreement is only "
                    f"{max(ratio, 1.0 - ratio):.3f}"
                )

            holes = [item.hole_on_plane_base for item in candidates]
            if any(point is None for point in holes):
                raise VisionError(f"{reference.address.text} has no plane coordinate")
            hole, hole_spread = _median_point(holes)  # type: ignore[arg-type]
            maximum_position_spread = max(maximum_position_spread, hole_spread)
            if hole_spread > self._stability.maximum_slot_position_spread_mm:
                raise VisionError(
                    f"{reference.address.text} slot spread {hole_spread:.2f}mm exceeds "
                    f"{self._stability.maximum_slot_position_spread_mm:.2f}mm"
                )
            if state is Occupancy.OCCUPIED:
                cap_points = [item.cap_top_base for item in occupied]
                if any(point is None for point in cap_points):
                    raise VisionError(f"{reference.address.text} has no cap coordinate")
                cap, cap_spread = _median_point(cap_points)  # type: ignore[arg-type]
                maximum_position_spread = max(maximum_position_spread, cap_spread)
                if cap_spread > self._stability.maximum_cap_position_spread_mm:
                    raise VisionError(
                        f"{reference.address.text} cap spread "
                        f"{cap_spread:.2f}mm exceeds "
                        f"{self._stability.maximum_cap_position_spread_mm:.2f}mm"
                    )
                pixel = _median_pixel([item.pixel for item in occupied])
                confidence = float(np.median([item.confidence for item in occupied]))
                slots.append(
                    SlotObservation(
                        address=reference.address,
                        occupancy=state,
                        confidence=confidence,
                        pixel=pixel,
                        cap_top_base=cap,
                        hole_on_plane_base=hole,
                    )
                )
            else:
                slots.append(
                    SlotObservation(
                        address=reference.address,
                        occupancy=state,
                        confidence=float(
                            np.median([item.confidence for item in candidates])
                        ),
                        pixel=_median_pixel([item.pixel for item in candidates]),
                        hole_on_plane_base=hole,
                    )
                )
        plane_z = float(
            np.median(
                [
                    slot.hole_on_plane_base.z_mm
                    for slot in slots
                    if slot.hole_on_plane_base is not None
                ]
            )
        )
        keypoints = tuple(item.pixel for item in stability.detection.keypoints)
        return RackObservation(
            rack_id=rack_id,
            marker=keypoints[0],
            plane_z_mm=plane_z,
            slots=tuple(slots),
            timestamp_ms=float(np.median([frame.timestamp_ms for frame in frames])),
            rack_keypoints=keypoints,
            stability_frame_count=len(frames),
            maximum_keypoint_spread_px=stability.maximum_keypoint_spread_px,
            maximum_position_spread_mm=maximum_position_spread,
        )


def _match_caps_to_slots(
    detections: Sequence[Detection],
    projected_slots: Sequence[tuple[SlotAddress, Pixel]],
    pose: RackPoseDetection,
    *,
    maximum_distance_factor: float,
) -> dict[SlotAddress, Detection]:
    slot_pixels = np.asarray([[pixel.u, pixel.v] for _, pixel in projected_slots])
    spacings = []
    for index, (address, pixel) in enumerate(projected_slots):
        for other_address, other_pixel in projected_slots[index + 1 :]:
            if (
                address.row == other_address.row
                and other_address.column == address.column + 1
            ) or (
                address.column == other_address.column
                and other_address.row == address.row + 1
            ):
                spacings.append(hypot(pixel.u - other_pixel.u, pixel.v - other_pixel.v))
    if not spacings or min(spacings) <= 1.0:
        raise VisionError("projected slot spacing is invalid")
    limit = float(np.median(spacings)) * float(maximum_distance_factor)
    polygon = np.asarray(
        [[point.u, point.v] for point in pose.corners],
        dtype=np.float32,
    )
    inside = [
        detection
        for detection in detections
        if cv2.pointPolygonTest(
            polygon,
            (float(detection.box.center.u), float(detection.box.center.v)),
            False,
        )
        >= 0
    ]
    pairs = []
    for cap_index, cap in enumerate(inside):
        center = np.asarray([cap.box.center.u, cap.box.center.v])
        distances = np.linalg.norm(slot_pixels - center[None, :], axis=1)
        slot_index = int(np.argmin(distances))
        distance = float(distances[slot_index])
        if distance > limit:
            raise VisionError(
                f"cap at ({center[0]:.1f}, {center[1]:.1f}) is {distance:.1f}px "
                f"from its nearest slot; limit is {limit:.1f}px"
            )
        pairs.append((distance, cap_index, slot_index))
    matched_caps: set[int] = set()
    matched_slots: set[int] = set()
    result: dict[SlotAddress, Detection] = {}
    for _, cap_index, slot_index in sorted(pairs):
        if cap_index in matched_caps:
            continue
        if slot_index in matched_slots:
            raise VisionError("multiple cap detections map to the same slot")
        matched_caps.add(cap_index)
        matched_slots.add(slot_index)
        result[projected_slots[slot_index][0]] = inside[cap_index]
    return result


def _median_point(points: Sequence[Point3D]) -> tuple[Point3D, float]:
    if not points:
        raise VisionError("cannot aggregate an empty point list")
    frames = {point.frame for point in points}
    if len(frames) != 1:
        raise VisionError("point coordinate frames changed across captures")
    array = np.asarray([[point.x_mm, point.y_mm, point.z_mm] for point in points])
    center = np.median(array, axis=0)
    spread = float(np.max(np.linalg.norm(array - center[None, :], axis=1)))
    return Point3D(*(float(value) for value in center), frame=points[0].frame), spread


def _median_pixel(points: Sequence[Pixel]) -> Pixel:
    array = np.asarray([[point.u, point.v] for point in points])
    center = np.median(array, axis=0)
    return Pixel(float(center[0]), float(center[1]))
