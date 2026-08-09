from __future__ import annotations

import unittest

import cv2
import numpy as np

from tube_grabber.core.errors import VisionError
from tube_grabber.vision.marker import K0MarkerDetector


class K0MarkerDetectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.detector = K0MarkerDetector(
            minimum_area_px=20,
            minimum_saturation=100,
            minimum_value=80,
            maximum_aspect_ratio=1.8,
            hue_ranges={
                "red": ((0, 10), (170, 179)),
                "green": ((35, 85),),
            },
        )

    def test_detects_red_and_green_separately(self) -> None:
        image = np.zeros((100, 120, 3), dtype=np.uint8)
        cv2.rectangle(image, (10, 15), (16, 21), (0, 0, 255), -1)
        cv2.rectangle(image, (80, 70), (86, 76), (0, 255, 0), -1)

        red = self.detector.detect(image, "red")
        green = self.detector.detect(image, "green")

        self.assertEqual(len(red), 1)
        self.assertAlmostEqual(red[0].u, 13.0)
        self.assertAlmostEqual(red[0].v, 18.0)
        self.assertEqual(len(green), 1)
        self.assertAlmostEqual(green[0].u, 83.0)
        self.assertAlmostEqual(green[0].v, 73.0)

    def test_small_color_noise_is_ignored(self) -> None:
        image = np.zeros((50, 50, 3), dtype=np.uint8)
        image[10:12, 10:12] = (0, 0, 255)
        with self.assertRaisesRegex(VisionError, "no red"):
            self.detector.detect(image, "red")

    def test_long_color_blob_is_not_a_square_k0(self) -> None:
        image = np.zeros((50, 80, 3), dtype=np.uint8)
        cv2.rectangle(image, (10, 10), (50, 15), (0, 0, 255), -1)
        with self.assertRaisesRegex(VisionError, "no red"):
            self.detector.detect(image, "red")


if __name__ == "__main__":
    unittest.main()
