from __future__ import annotations

import unittest

import numpy as np

from tube_grabber.core.errors import VisionError
from tube_grabber.core.models import Box, Detection, Pixel
from tube_grabber.vision.grid import GridMapper


def make_detection(u: float, v: float, label: str = "empty_hole") -> Detection:
    return Detection(label, 0.9, Box(u - 3.0, v - 3.0, u + 3.0, v + 3.0))


def make_affine_grid() -> tuple[list[Detection], dict[tuple[int, int], np.ndarray]]:
    origin = np.array([180.0, 120.0])
    column_step = np.array([24.0, 7.0])
    row_step = np.array([-9.0, 34.0])
    points: dict[tuple[int, int], np.ndarray] = {}
    detections: list[Detection] = []
    for row in range(2):
        for column in range(6):
            point = origin + column * column_step + row * row_step
            points[(row, column)] = point
            detections.append(make_detection(*point))
    return detections, points


class GridMapperTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mapper = GridMapper(
            maximum_spacing_cv=0.25,
            maximum_marker_corner_distance_factor=1.5,
        )

    def test_all_four_k0_corners_choose_row_one_column_one(self) -> None:
        detections, points = make_affine_grid()
        column_step = points[(0, 1)] - points[(0, 0)]
        row_step = points[(1, 0)] - points[(0, 0)]

        for raw_row, raw_column in ((0, 0), (0, 5), (1, 0), (1, 5)):
            with self.subTest(k0=(raw_row, raw_column)):
                column_sign = -1.0 if raw_column == 0 else 1.0
                row_sign = -1.0 if raw_row == 0 else 1.0
                marker_xy = (
                    points[(raw_row, raw_column)]
                    + 0.30 * column_sign * column_step
                    + 0.30 * row_sign * row_step
                )
                mapping = self.mapper.map(
                    detections,
                    [Pixel(*marker_xy)],
                    "rack_1",
                )

                r1c1 = mapping.slots[0]
                expected = points[(raw_row, raw_column)]
                actual = np.array(
                    [r1c1.detection.box.center.u, r1c1.detection.box.center.v]
                )
                np.testing.assert_allclose(actual, expected, atol=1e-6)

                addresses = [slot.address.text for slot in mapping.slots]
                self.assertEqual(len(addresses), 12)
                self.assertEqual(len(set(addresses)), 12)
                self.assertEqual(addresses[0], "rack_1.r1c1")
                self.assertEqual(addresses[-1], "rack_1.r2c6")

    def test_rotated_grid_maps_correctly(self) -> None:
        detections, points = make_affine_grid()
        angle = np.deg2rad(83.0)
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        pivot = np.array([220.0, 170.0])

        rotated_detections = []
        rotated_points = {}
        for (row, column), point in points.items():
            rotated = pivot + rotation @ (point - pivot)
            rotated_points[(row, column)] = rotated
            rotated_detections.append(make_detection(*rotated))

        column_step = rotated_points[(0, 1)] - rotated_points[(0, 0)]
        row_step = rotated_points[(1, 0)] - rotated_points[(0, 0)]
        marker = (
            rotated_points[(1, 5)] + 0.25 * column_step + 0.25 * row_step
        )
        mapping = self.mapper.map(
            rotated_detections,
            [Pixel(*marker)],
            "rack_2",
        )
        r1c1 = mapping.slots[0].detection.box.center
        np.testing.assert_allclose(
            [r1c1.u, r1c1.v],
            rotated_points[(1, 5)],
            atol=1e-6,
        )

    def test_missing_detection_is_rejected(self) -> None:
        detections, points = make_affine_grid()
        with self.assertRaisesRegex(VisionError, "exactly 12"):
            self.mapper.map(detections[:-1], [Pixel(*points[(0, 0)])], "rack_1")

    def test_duplicate_detection_is_rejected(self) -> None:
        detections, points = make_affine_grid()
        duplicated = detections[:-1] + [detections[0]]
        with self.assertRaisesRegex(VisionError, "duplicate"):
            self.mapper.map(duplicated, [Pixel(*points[(0, 0)])], "rack_1")

    def test_abnormal_spacing_is_rejected(self) -> None:
        detections, points = make_affine_grid()
        bad = list(detections)
        centre = bad[4].box.center
        bad[4] = make_detection(centre.u + 90.0, centre.v - 30.0)
        with self.assertRaises(VisionError):
            self.mapper.map(bad, [Pixel(*points[(0, 0)])], "rack_1")

    def test_marker_far_from_all_corners_is_rejected(self) -> None:
        detections, _ = make_affine_grid()
        with self.assertRaisesRegex(VisionError, "too far"):
            self.mapper.map(detections, [Pixel(900.0, 900.0)], "rack_1")

    def test_colored_cap_candidate_is_not_mistaken_for_k0(self) -> None:
        detections, points = make_affine_grid()
        column_step = points[(0, 1)] - points[(0, 0)]
        row_step = points[(1, 0)] - points[(0, 0)]
        real_marker = points[(0, 0)] - 0.25 * column_step - 0.25 * row_step

        mapping = self.mapper.map(
            detections,
            [Pixel(*points[(0, 0)]), Pixel(*real_marker)],
            "rack_1",
        )

        self.assertAlmostEqual(mapping.marker.u, real_marker[0])
        self.assertAlmostEqual(mapping.marker.v, real_marker[1])

    def test_multiple_outside_markers_are_rejected_as_ambiguous(self) -> None:
        detections, points = make_affine_grid()
        column_step = points[(0, 1)] - points[(0, 0)]
        row_step = points[(1, 0)] - points[(0, 0)]
        first = points[(0, 0)] - 0.25 * column_step - 0.25 * row_step
        second = points[(1, 5)] + 0.25 * column_step + 0.25 * row_step

        with self.assertRaisesRegex(VisionError, "ambiguous K0"):
            self.mapper.map(
                detections,
                [Pixel(*first), Pixel(*second)],
                "rack_1",
            )


if __name__ == "__main__":
    unittest.main()
