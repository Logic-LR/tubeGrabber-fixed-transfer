from __future__ import annotations

import unittest

import numpy as np

from tube_grabber.core.models import CameraFrame, CameraIntrinsics, Pixel
from tube_grabber.vision.plane import RackPlaneFitConfig, fit_rack_plane
from tests.test_rack_pose import pose


class RackPlaneTests(unittest.TestCase):
    def test_ransac_fits_surface_and_rejects_protruding_patch(self) -> None:
        depth = np.full((400, 600), 1000.0, dtype=np.float32)
        depth[170:230, 250:350] = 950.0
        frame = CameraFrame(
            color=np.zeros((400, 600, 3), dtype=np.uint8),
            depth_mm=depth,
            intrinsics=CameraIntrinsics(500, 500, 300, 200),
            timestamp_ms=1.0,
            frame_number=1,
        )
        plane = fit_rack_plane(
            frame,
            pose(),
            np.eye(4),
            excluded_boxes=(),
            depth_min_mm=100,
            depth_max_mm=1200,
            config=RackPlaneFitConfig(
                sample_stride_px=5,
                roi_margin_px=5,
                landmark_exclusion_radius_px=5,
                ransac_iterations=100,
                inlier_threshold_mm=1.0,
                minimum_inliers=500,
                minimum_inlier_ratio=0.5,
                maximum_rms_error_mm=0.1,
                maximum_tilt_deg=1.0,
            ),
        )
        hit = plane.intersect(Pixel(300, 200), frame, np.eye(4))
        self.assertAlmostEqual(hit.z_mm, 1000.0, places=4)
        self.assertGreater(plane.inlier_ratio, 0.8)


if __name__ == "__main__":
    unittest.main()
