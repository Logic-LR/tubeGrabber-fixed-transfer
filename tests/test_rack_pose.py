from __future__ import annotations

import unittest

import cv2
import numpy as np

from tube_grabber.core.errors import VisionError
from tube_grabber.core.models import Box, Pixel
from tube_grabber.vision.rack_pose import (
    RACK_KEYPOINT_NAMES,
    RackKeypoint,
    RackPoseDetection,
    fuse_rack_pose_detections,
    validate_k0_red_marker,
    validate_rack_pose_geometry,
)


def pose(shift_x: float = 0.0, *, swap_screws: bool = False) -> RackPoseDetection:
    points = [
        (100, 100),
        (500, 100),
        (500, 300),
        (100, 300),
        (140, 140),
        (460, 140),
        (460, 260),
        (140, 260),
    ]
    if swap_screws:
        points[4], points[6] = points[6], points[4]
    return RackPoseDetection(
        confidence=0.95,
        box=Box(90 + shift_x, 90, 510 + shift_x, 310),
        keypoints=tuple(
            RackKeypoint(name, Pixel(x + shift_x, y), 0.9)
            for name, (x, y) in zip(RACK_KEYPOINT_NAMES, points)
        ),
    )


class RackPoseTests(unittest.TestCase):
    def test_geometry_checks_screw_identity(self) -> None:
        validate_rack_pose_geometry(
            pose(),
            minimum_area_px2=10_000,
            maximum_opposite_side_ratio=1.2,
            screw_edge_margin=0.02,
        )
        with self.assertRaisesRegex(VisionError, "wrong rack quadrant"):
            validate_rack_pose_geometry(
                pose(swap_screws=True),
                minimum_area_px2=10_000,
                maximum_opposite_side_ratio=1.2,
                screw_edge_margin=0.02,
            )

    def test_multi_frame_fusion_rejects_one_whole_frame_outlier(self) -> None:
        stable = fuse_rack_pose_detections(
            [pose(0.0), pose(0.5), pose(-0.5), pose(30.0)],
            attempted_frames=4,
            minimum_inlier_frames=3,
            maximum_frame_residual_px=2.0,
            maximum_keypoint_spread_px=2.0,
            minimum_area_px2=10_000,
            maximum_opposite_side_ratio=1.2,
            screw_edge_margin=0.02,
        )
        self.assertEqual(stable.inlier_indices, (0, 1, 2))
        self.assertAlmostEqual(stable.detection.pixel("k0").u, 100.0)

    def test_red_dot_must_be_at_screw_k0(self) -> None:
        image = np.zeros((400, 600, 3), dtype=np.uint8)
        cv2.circle(image, (140, 140), 6, (0, 0, 255), -1)
        scores = validate_k0_red_marker(
            image,
            pose(),
            patch_radius_px=12,
            minimum_red_ratio=0.05,
            minimum_ratio_margin=0.03,
            minimum_saturation=100,
            minimum_value=80,
        )
        self.assertGreater(scores[0], scores[1])

        wrong = np.zeros_like(image)
        cv2.circle(wrong, (460, 260), 6, (0, 0, 255), -1)
        with self.assertRaisesRegex(VisionError, "screw_k0 red-dot"):
            validate_k0_red_marker(
                wrong,
                pose(),
                patch_radius_px=12,
                minimum_red_ratio=0.05,
                minimum_ratio_margin=0.03,
                minimum_saturation=100,
                minimum_value=80,
            )


if __name__ == "__main__":
    unittest.main()
