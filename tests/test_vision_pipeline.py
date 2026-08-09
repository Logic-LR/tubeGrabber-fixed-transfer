from __future__ import annotations

import unittest

import cv2
import numpy as np

from tube_grabber.core.errors import VisionError
from tube_grabber.core.models import (
    Box,
    CameraFrame,
    CameraIntrinsics,
    Detection,
    Occupancy,
    Pose6D,
    SlotAddress,
)
from tube_grabber.vision.grid import GridMapper
from tube_grabber.vision.marker import K0MarkerDetector
from tube_grabber.vision.pipeline import RackVision


class FakeDetector:
    def __init__(self, detections: list[Detection]) -> None:
        self.detections = detections

    def detect(self, color_image: object) -> list[Detection]:
        return list(self.detections)


def rack_detections(all_empty: bool = False) -> list[Detection]:
    detections = []
    for row in range(2):
        for column in range(6):
            u = 60.0 + 30.0 * column
            v = 60.0 + 40.0 * row
            label = "empty_hole"
            if not all_empty and row == 0 and column == 0:
                label = "tube_cap"
            detections.append(
                Detection(label, 0.9, Box(u - 4.0, v - 4.0, u + 4.0, v + 4.0))
            )
    return detections


def marker_detector() -> K0MarkerDetector:
    return K0MarkerDetector(
        minimum_area_px=20,
        minimum_saturation=100,
        minimum_value=80,
        maximum_aspect_ratio=1.8,
        hue_ranges={"red": ((0, 10), (170, 179)), "green": ((35, 85),)},
    )


def make_frame() -> CameraFrame:
    color = np.zeros((180, 260, 3), dtype=np.uint8)
    cv2.rectangle(color, (40, 40), (46, 46), (0, 0, 255), -1)
    depth = np.full((180, 260), 100.0, dtype=np.float32)
    return CameraFrame(
        color=color,
        depth_mm=depth,
        intrinsics=CameraIntrinsics(100.0, 100.0, 0.0, 0.0),
        timestamp_ms=1234.0,
        frame_number=7,
    )


def make_vision(
    detections: list[Detection],
    fallback_plane: float | None,
) -> RackVision:
    return RackVision(
        detector=FakeDetector(detections),
        marker_detector=marker_detector(),
        grid_mapper=GridMapper(0.25, 1.5),
        hand_eye_end_from_camera=np.eye(4),
        rack_marker_colors={"rack_1": "red"},
        fallback_plane_z_mm={"rack_1": fallback_plane},
        required_detection_count=12,
        depth_window_px=7,
        depth_min_mm=10.0,
        depth_max_mm=1000.0,
        cap_top_above_rack_mm=10.0,
        maximum_cap_z_deviation_mm=5.0,
        maximum_plane_calibration_error_mm=5.0,
    )


class RackVisionTests(unittest.TestCase):
    def test_cap_depth_defines_plane_and_empty_uses_ray_intersection(self) -> None:
        observation = make_vision(rack_detections(), None).observe(
            make_frame(),
            "rack_1",
            Pose6D(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        )

        self.assertEqual(observation.rack_id, "rack_1")
        self.assertEqual(observation.plane_z_mm, 90.0)
        occupied = observation.slot(SlotAddress("rack_1", 1, 1))
        self.assertEqual(occupied.occupancy, Occupancy.OCCUPIED)
        self.assertIsNotNone(occupied.cap_top_base)
        self.assertAlmostEqual(occupied.cap_top_base.z_mm, 100.0)

        empty = observation.slot(SlotAddress("rack_1", 1, 2))
        self.assertEqual(empty.occupancy, Occupancy.EMPTY)
        self.assertIsNotNone(empty.hole_on_plane_base)
        self.assertAlmostEqual(empty.hole_on_plane_base.x_mm, 81.0)
        self.assertAlmostEqual(empty.hole_on_plane_base.y_mm, 54.0)
        self.assertAlmostEqual(empty.hole_on_plane_base.z_mm, 90.0)

    def test_empty_rack_uses_calibrated_fallback_plane(self) -> None:
        observation = make_vision(rack_detections(all_empty=True), 80.0).observe(
            make_frame(),
            "rack_1",
            Pose6D(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        )
        self.assertEqual(observation.plane_z_mm, 80.0)
        self.assertTrue(
            all(slot.occupancy is Occupancy.EMPTY for slot in observation.slots)
        )

    def test_empty_rack_without_calibration_is_rejected(self) -> None:
        with self.assertRaisesRegex(VisionError, "no calibrated fallback"):
            make_vision(rack_detections(all_empty=True), None).observe(
                make_frame(),
                "rack_1",
                Pose6D(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            )

    def test_eleven_detections_are_rejected(self) -> None:
        with self.assertRaisesRegex(VisionError, "exactly 12"):
            make_vision(rack_detections()[:-1], None).observe(
                make_frame(),
                "rack_1",
                Pose6D(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            )

    def test_disagreeing_cap_depths_are_rejected(self) -> None:
        detections = rack_detections()
        second = detections[1]
        detections[1] = Detection("tube_cap", second.confidence, second.box)
        frame = make_frame()
        frame.depth_mm[56:65, 86:95] = 120.0

        with self.assertRaisesRegex(VisionError, "depths disagree"):
            make_vision(detections, None).observe(
                frame,
                "rack_1",
                Pose6D(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            )

    def test_measured_plane_must_match_calibration(self) -> None:
        with self.assertRaisesRegex(VisionError, "calibrated rack"):
            make_vision(rack_detections(), 80.0).observe(
                make_frame(),
                "rack_1",
                Pose6D(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            )


if __name__ == "__main__":
    unittest.main()
