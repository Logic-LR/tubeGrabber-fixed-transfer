from __future__ import annotations

import unittest

import numpy as np

from tube_grabber.core.errors import VisionError
from tube_grabber.core.models import CameraIntrinsics, Pixel, Pose6D
from tube_grabber.vision.depth import sample_depth_mm
from tube_grabber.vision.geometry import (
    base_from_camera,
    pixel_depth_to_base,
    pixel_ray_to_horizontal_plane,
)


class DepthAndGeometryTests(unittest.TestCase):
    def test_depth_sampler_returns_valid_median(self) -> None:
        depth = np.zeros((9, 9), dtype=np.float32)
        depth[3:6, 3:6] = [
            [0.0, 480.0, 500.0],
            [510.0, 520.0, 530.0],
            [2000.0, 540.0, 550.0],
        ]
        value = sample_depth_mm(depth, Pixel(4.0, 4.0), 3, 100.0, 1200.0)
        self.assertEqual(value, 520.0)

    def test_depth_sampler_rejects_missing_depth(self) -> None:
        with self.assertRaisesRegex(VisionError, "no valid depth"):
            sample_depth_mm(
                np.zeros((9, 9), dtype=np.uint16),
                Pixel(4.0, 4.0),
                3,
                100.0,
                1200.0,
            )

    def test_pixel_depth_and_plane_intersection(self) -> None:
        intrinsics = CameraIntrinsics(fx=100.0, fy=100.0, cx=50.0, cy=40.0)
        arm_pose = Pose6D(10.0, 20.0, 30.0, 0.0, 0.0, 0.0)
        transform = base_from_camera(arm_pose, np.eye(4))

        measured = pixel_depth_to_base(
            Pixel(60.0, 50.0),
            100.0,
            intrinsics,
            transform,
        )
        self.assertAlmostEqual(measured.x_mm, 20.0)
        self.assertAlmostEqual(measured.y_mm, 30.0)
        self.assertAlmostEqual(measured.z_mm, 130.0)

        hit = pixel_ray_to_horizontal_plane(
            Pixel(60.0, 50.0),
            120.0,
            intrinsics,
            transform,
        )
        self.assertAlmostEqual(hit.x_mm, 19.0)
        self.assertAlmostEqual(hit.y_mm, 29.0)
        self.assertAlmostEqual(hit.z_mm, 120.0)

    def test_brown_conrady_pixel_is_undistorted_before_projection(self) -> None:
        intrinsics = CameraIntrinsics(
            fx=100.0,
            fy=100.0,
            cx=50.0,
            cy=40.0,
            distortion_model="brown_conrady",
            distortion_coefficients=(0.2, 0.0, 0.0, 0.0, 0.0),
        )
        measured = pixel_depth_to_base(
            Pixel(90.0, 40.0),
            100.0,
            intrinsics,
            np.eye(4),
        )

        # Positive radial distortion means the undistorted X is smaller than
        # the pinhole-only value of 40 mm.
        self.assertGreater(measured.x_mm, 0.0)
        self.assertLess(measured.x_mm, 40.0)
        self.assertAlmostEqual(measured.y_mm, 0.0)
        self.assertAlmostEqual(measured.z_mm, 100.0)


if __name__ == "__main__":
    unittest.main()
