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
)
from tube_grabber.vision.rack_calibration import calibrate_slot_grid
from tube_grabber.vision.rack_pose import RackPoseQualityConfig
from tests.test_rack_pose import pose


class _PoseDetector:
    def detect(self, _image: object):
        return pose()


class _CapDetector:
    def detect(self, _image: object):
        return [Detection("tube_cap", 0.91, Box(144, 144, 156, 156))]


class PosePipelineTests(unittest.TestCase):
    def test_multi_frame_pose_plane_cap_and_grid_pipeline(self) -> None:
        color = np.zeros((400, 600, 3), dtype=np.uint8)
        cv2.circle(color, (140, 140), 6, (0, 0, 255), -1)
        depth = np.full((400, 600), 1000.0, dtype=np.float32)
        depth[144:157, 144:157] = 953.0
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
        vision = PoseRackVision(
            rack_pose_detector=_PoseDetector(),
            cap_detector=_CapDetector(),
            calibrations={"rack_1": calibration},
            hand_eye_end_from_camera=np.eye(4),
            pose_quality=RackPoseQualityConfig(
                minimum_rack_area_px2=10_000,
                maximum_opposite_side_ratio=1.2,
                screw_edge_margin=0.02,
                k0_red_patch_radius_px=12,
                k0_red_minimum_ratio=0.05,
                k0_red_minimum_ratio_margin=0.03,
                k0_red_minimum_saturation=100,
                k0_red_minimum_value=80,
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
                maximum_calibration_keypoint_shift_ratio=0.03,
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
        self.assertTrue(
            all(
                slot.occupancy is Occupancy.EMPTY
                for slot in observation.slots[1:]
            )
        )

        elevated_samples = []
        for index, sample in enumerate(samples):
            elevated_depth = np.asarray(sample.frame.depth_mm).copy()
            elevated_depth[144:157, 144:157] = 900.0
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
