from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from tube_grabber.core.errors import VisionError
from tube_grabber.core.models import Pixel
from tube_grabber.vision.rack_calibration import (
    RackCircleFitConfig,
    calibrate_slot_grid,
    fit_slot_circle,
    load_rack_calibration,
    save_rack_calibration,
)
from tube_grabber.vision.rack_pose import RackPoseDetection, transform_pixels
from tests.test_rack_pose import pose


class RackCalibrationTests(unittest.TestCase):
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

    def test_calibration_round_trip_and_normalized_reference_gate(self) -> None:
        calibration = calibrate_slot_grid(
            "rack_1", pose(), Pixel(150, 150), Pixel(450, 250)
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rack.yaml"
            save_rack_calibration(calibration, path)
            loaded = load_rack_calibration(path, "rack_1")
        self.assertEqual(loaded.slot_unit_points(), calibration.slot_unit_points())
        loaded.validate_reference_pose(_perspective_pose(), 0.01)

        source = pose()
        keypoints = list(source.keypoints)
        screw = keypoints[4]
        keypoints[4] = replace(
            screw,
            pixel=Pixel(screw.pixel.u + 30.0, screw.pixel.v),
        )
        changed = replace(source, keypoints=tuple(keypoints))
        with self.assertRaisesRegex(VisionError, "normalized layout changed"):
            loaded.validate_reference_pose(changed, 0.04)

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


def _perspective_pose() -> RackPoseDetection:
    source = pose()
    matrix = np.asarray(
        [
            [1.10, 0.08, 35.0],
            [0.04, 0.92, 18.0],
            [0.00015, 0.00008, 1.0],
        ],
        dtype=np.float64,
    )
    transformed = transform_pixels(
        tuple(item.pixel for item in source.keypoints),
        matrix,
    )
    keypoints = tuple(
        replace(item, pixel=point)
        for item, point in zip(source.keypoints, transformed)
    )
    array = np.asarray([[point.u, point.v] for point in transformed])
    return replace(
        source,
        box=replace(
            source.box,
            x1=float(array[:, 0].min()),
            y1=float(array[:, 1].min()),
            x2=float(array[:, 0].max()),
            y2=float(array[:, 1].max()),
        ),
        keypoints=keypoints,
    )


if __name__ == "__main__":
    unittest.main()
