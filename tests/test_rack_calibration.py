from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from tube_grabber.core.models import Pixel
from tube_grabber.vision.rack_calibration import (
    RackCircleFitConfig,
    calibrate_display_corners,
    calibrate_slot_grid,
    fit_slot_circle,
    load_rack_calibration,
    save_rack_calibration,
)
from tests.test_rack_pose import pose


class RackCalibrationTests(unittest.TestCase):
    def test_four_display_corners_follow_detected_screw_homography(self) -> None:
        calibration = calibrate_slot_grid(
            "rack_1", pose(), Pixel(150, 150), Pixel(450, 250)
        )
        targets = (
            Pixel(120, 120),
            Pixel(480, 120),
            Pixel(480, 280),
            Pixel(120, 280),
        )
        taught = calibrate_display_corners(calibration, pose(), targets)
        projected = taught.project_display_corners(pose())
        for actual, expected in zip(projected, targets):
            self.assertAlmostEqual(actual.u, expected.u, places=4)
            self.assertAlmostEqual(actual.v, expected.v, places=4)

        shifted = taught.project_display_corners(pose(shift_x=25))
        for actual, expected in zip(shifted, targets):
            self.assertAlmostEqual(actual.u, expected.u + 25, places=4)
            self.assertAlmostEqual(actual.v, expected.v, places=4)

    def test_two_diagonal_slots_generate_complete_grid(self) -> None:
        calibration = calibrate_slot_grid(
            "rack_1",
            pose(),
            Pixel(150, 150),
            Pixel(450, 250),
        )
        slots = calibration.project_slots(pose())
        self.assertEqual(len(slots), 12)
        self.assertEqual(slots[0][0].text, "rack_1.r1c1")
        self.assertAlmostEqual(slots[0][1].u, 150.0, places=4)
        self.assertEqual(slots[-1][0].text, "rack_1.r2c6")
        self.assertAlmostEqual(slots[-1][1].v, 250.0, places=4)

    def test_calibration_round_trip(self) -> None:
        calibration = calibrate_slot_grid(
            "rack_1", pose(), Pixel(150, 150), Pixel(450, 250)
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rack.yaml"
            save_rack_calibration(calibration, path)
            loaded = load_rack_calibration(path, "rack_1")
        self.assertEqual(loaded.slot_unit_points(), calibration.slot_unit_points())
        self.assertEqual(len(loaded.keypoint_names), 4)
        self.assertEqual(
            loaded.display_corners_unit,
            calibration.display_corners_unit,
        )

    def test_circle_fit_finds_slot_near_approximate_click(self) -> None:
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        cv2.circle(image, (170, 120), 24, (255, 255, 255), 3)
        circle = fit_slot_circle(
            image,
            Pixel(162, 126),
            RackCircleFitConfig(
                search_radius_px=60,
                minimum_radius_px=15,
                maximum_radius_px=35,
                default_radius_px=24,
                hough_dp=1.0,
                hough_min_distance_px=12,
                hough_edge_threshold=80,
                hough_accumulator_threshold=12,
            ),
        )
        self.assertLess(abs(circle.center.u - 170), 3)
        self.assertLess(abs(circle.center.v - 120), 3)
        self.assertLess(abs(circle.radius_px - 24), 4)

if __name__ == "__main__":
    unittest.main()
