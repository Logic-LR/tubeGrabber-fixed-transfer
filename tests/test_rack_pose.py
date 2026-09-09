from __future__ import annotations

import unittest

from tube_grabber.core.errors import VisionError
import cv2
import numpy as np

from tube_grabber.core.models import Box, Detection, Pixel
from tube_grabber.vision.rack_pose import (
    RACK_KEYPOINT_NAMES,
    RackKeypoint,
    RackPoseDetection,
    ScrewMarkerConfig,
    fuse_rack_pose_detections,
    rack_detection_from_screws,
    validate_rack_pose_geometry,
)


def pose(shift_x: float = 0.0, *, crossed: bool = False) -> RackPoseDetection:
    points = [
        (100, 100),
        (500, 100),
        (500, 300),
        (100, 300),
    ]
    if crossed:
        points[1], points[2] = points[2], points[1]
    return RackPoseDetection(
        confidence=0.95,
        box=Box(90 + shift_x, 90, 510 + shift_x, 310),
        keypoints=tuple(
            RackKeypoint(name, Pixel(x + shift_x, y), 0.9)
            for name, (x, y) in zip(RACK_KEYPOINT_NAMES, points)
        ),
    )


class RackPoseTests(unittest.TestCase):
    def test_screw_centers_are_oriented_from_white_k0_marker(self) -> None:
        image = np.full((420, 640, 3), (40, 150, 40), dtype=np.uint8)
        points = ((100, 100), (540, 100), (540, 320), (100, 320))
        detections = []
        for x, y in points:
            cv2.circle(image, (x, y), 6, (0, 0, 0), -1)
            detections.append(
                Detection("screw", 0.9, Box(x - 8, y - 8, x + 8, y + 8))
            )
        # The real marker is on the green rack surface.  A larger white label
        # outside the diagonally opposite corner must not steal K0.  Input
        # detections are deliberately shuffled.
        cv2.circle(image, (124, 124), 6, (245, 245, 245), -1)
        cv2.circle(image, (566, 346), 8, (245, 245, 245), -1)
        detection = rack_detection_from_screws(
            image,
            [detections[2], detections[0], detections[3], detections[1]],
            marker=ScrewMarkerConfig(
                minimum_value=180,
                maximum_saturation=70,
                search_radius_factor=0.45,
                screw_exclusion_scale=1.3,
                minimum_area_px2=18,
                maximum_area_ratio=3.0,
                minimum_score_ratio=1.25,
                minimum_rack_aspect_ratio=1.5,
            ),
        )
        self.assertEqual(detection.pixel("k0"), Pixel(100, 100))
        self.assertEqual(detection.pixel("k1"), Pixel(540, 100))
        self.assertEqual(detection.pixel("k2"), Pixel(540, 320))
        self.assertEqual(detection.pixel("k3"), Pixel(100, 320))

    def test_screw_rack_requires_exactly_four_detections(self) -> None:
        with self.assertRaisesRegex(VisionError, "exactly four"):
            rack_detection_from_screws(
                np.zeros((100, 100, 3), dtype=np.uint8),
                [],
                marker=ScrewMarkerConfig(
                    180, 70, 0.45, 1.3, 18, 3.0, 1.25, 1.5
                ),
            )

    def test_geometry_rejects_crossed_corner_order(self) -> None:
        validate_rack_pose_geometry(
            pose(),
            minimum_area_px2=10_000,
            maximum_opposite_side_ratio=1.2,
        )
        with self.assertRaisesRegex(VisionError, "crossed or non-convex"):
            validate_rack_pose_geometry(
                pose(crossed=True),
                minimum_area_px2=10_000,
                maximum_opposite_side_ratio=1.2,
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
        )
        self.assertEqual(stable.inlier_indices, (0, 1, 2))
        self.assertAlmostEqual(stable.detection.pixel("k0").u, 100.0)

if __name__ == "__main__":
    unittest.main()
