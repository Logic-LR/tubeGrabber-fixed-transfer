from __future__ import annotations

import unittest

import numpy as np

from tube_grabber.core.errors import VisionError
from tube_grabber.core.models import (
    Box,
    CameraFrame,
    CameraIntrinsics,
    Detection,
    Occupancy,
    Pixel,
    Pose6D,
    SlotAddress,
)
from tube_grabber.vision.plane import RackPlaneFitConfig
from tube_grabber.vision.pose_pipeline import (
    CapturedRackFrame,
    PoseRackVision,
    RackMatchingConfig,
    RackStabilityConfig,
    _match_caps_to_slots,
)
from tube_grabber.vision.rack_calibration import (
    calibrate_display_corners,
    calibrate_slot_grid,
)
from tube_grabber.vision.rack_pose import RackPoseQualityConfig
from tests.test_rack_pose import pose


class _PoseDetector:
    def detect(self, _image: object):
        return pose()


class _CapDetector:
    def detect(self, _image: object):
        # Detection is deliberately offset 20 px from the calibrated slot.
        return [Detection("tube_cap", 0.91, Box(164, 144, 176, 156))]


class PosePipelineTests(unittest.TestCase):
    def test_corner_screw_cap_false_positive_is_ignored(self) -> None:
        projected = tuple(
            (
                SlotAddress("rack_1", row + 1, column + 1),
                Pixel(150 + 60 * column, 150 + 100 * row),
            )
            for row in range(2)
            for column in range(6)
        )
        real_cap = Detection("tube_cap", 0.91, Box(144, 144, 156, 156))
        screw_false_positive = Detection(
            "tube_cap",
            0.63,
            Box(92, 92, 108, 108),
        )

        matches = _match_caps_to_slots(
            [real_cap, screw_false_positive],
            projected,
            pose(),
            maximum_distance_factor=0.45,
        )

        self.assertEqual(
            matches,
            {SlotAddress("rack_1", 1, 1): real_cap},
        )

    def test_multi_frame_pose_plane_cap_and_grid_pipeline(self) -> None:
        color = np.zeros((400, 600, 3), dtype=np.uint8)
        depth = np.full((400, 600), 1000.0, dtype=np.float32)
        depth[144:157, 164:177] = 953.0
        samples = []
        for index in range(5):
            frame = CameraFrame(
                color=color.copy(),
                depth_mm=depth.copy(),
                intrinsics=CameraIntrinsics(500, 500, 300, 200),
                timestamp_ms=float(index),
                frame_number=index,
            )
            samples.append(CapturedRackFrame(frame, Pose6D(0, 0, 0, 0, 0, 0)))
        calibration = calibrate_slot_grid(
            "rack_1", pose(), Pixel(150, 150), Pixel(450, 250)
        )
        calibration = calibrate_display_corners(
            calibration,
            pose(),
            pose().corners,
        )
        vision = PoseRackVision(
            rack_pose_detector=_PoseDetector(),
            cap_detector=_CapDetector(),
            calibrations={"rack_1": calibration},
            hand_eye_end_from_camera=np.eye(4),
            pose_quality=RackPoseQualityConfig(
                minimum_rack_area_px2=10_000,
                maximum_opposite_side_ratio=1.2,
            ),
            stability=RackStabilityConfig(
                capture_frames=5,
                minimum_inlier_frames=5,
                maximum_frame_residual_px=2,
                maximum_keypoint_spread_px=2,
                minimum_occupancy_agreement=0.8,
                maximum_slot_position_spread_mm=0.1,
                maximum_cap_position_spread_mm=0.1,
            ),
            plane_config=RackPlaneFitConfig(
                sample_stride_px=5,
                roi_margin_px=5,
                landmark_exclusion_radius_px=5,
                ransac_iterations=100,
                inlier_threshold_mm=1,
                minimum_inliers=500,
                minimum_inlier_ratio=0.8,
                maximum_rms_error_mm=0.1,
                maximum_tilt_deg=1,
            ),
            matching=RackMatchingConfig(
                depth_window_px=7,
                depth_min_mm=100,
                depth_max_mm=1200,
                cap_top_above_rack_mm=47,
                maximum_cap_height_error_mm=1,
                cap_slot_max_distance_factor=0.45,
            ),
        )
        observation = vision.observe(samples, "rack_1")
        self.assertEqual(observation.stability_frame_count, 5)
        self.assertAlmostEqual(observation.plane_z_mm, 1000.0, places=3)
        source = observation.slots[0]
        self.assertIs(source.occupancy, Occupancy.OCCUPIED)
        self.assertAlmostEqual(
            source.cap_top_base.z_mm,  # type: ignore[union-attr]
            953.0,
            places=3,
        )
        self.assertAlmostEqual(
            source.cap_top_base.x_mm,  # type: ignore[union-attr]
            source.hole_on_plane_base.x_mm,  # type: ignore[union-attr]
        )
        self.assertAlmostEqual(
            source.cap_top_base.y_mm,  # type: ignore[union-attr]
            source.hole_on_plane_base.y_mm,  # type: ignore[union-attr]
        )
        self.assertTrue(
            all(
                slot.occupancy is Occupancy.EMPTY
                for slot in observation.slots[1:]
            )
        )

        visual = vision.detect_visual(color, "rack_1")
        self.assertIsNotNone(visual.pose)
        self.assertEqual(len(visual.display_corners), 4)
        self.assertEqual(len(visual.projected_slots), 12)
        self.assertEqual(len(visual.caps), 1)
        self.assertEqual(visual.rack_error, "")

        missing_depth_samples = []
        for index, sample in enumerate(samples):
            missing_depth = np.asarray(sample.frame.depth_mm).copy()
            missing_depth[144:157, 164:177] = 0.0
            missing_depth_samples.append(
                CapturedRackFrame(
                    CameraFrame(
                        color=sample.frame.color,
                        depth_mm=missing_depth,
                        intrinsics=sample.frame.intrinsics,
                        timestamp_ms=float(index),
                        frame_number=index,
                    ),
                    sample.arm_pose,
                )
            )
        fallback = vision.observe(missing_depth_samples, "rack_1")
        fallback_source = fallback.slot(SlotAddress("rack_1", 1, 1))
        self.assertIs(fallback_source.occupancy, Occupancy.OCCUPIED)
        self.assertAlmostEqual(
            fallback_source.cap_top_base.z_mm,  # type: ignore[union-attr]
            953.0,
            places=3,
        )

        elevated_samples = []
        for index, sample in enumerate(samples):
            elevated_depth = np.asarray(sample.frame.depth_mm).copy()
            elevated_depth[144:157, 164:177] = 900.0
            elevated_samples.append(
                CapturedRackFrame(
                    CameraFrame(
                        color=np.asarray(sample.frame.color).copy(),
                        depth_mm=elevated_depth,
                        intrinsics=sample.frame.intrinsics,
                        timestamp_ms=float(index + 10),
                        frame_number=index + 10,
                    ),
                    sample.arm_pose,
                )
            )
        with self.assertRaises(VisionError):
            vision.observe(elevated_samples, "rack_1")

        carried = vision.observe(
            elevated_samples,
            "rack_1",
            ignored_elevated_slot=SlotAddress("rack_1", 1, 1),
        )
        self.assertIs(carried.slots[0].occupancy, Occupancy.EMPTY)


if __name__ == "__main__":
    unittest.main()
