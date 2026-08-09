from __future__ import annotations

import unittest

from tube_grabber.core.models import Occupancy, Point3D, SlotAddress
from tube_grabber.core.parsing import parse_slot, parse_transfer


class CoreModelsTest(unittest.TestCase):
    def test_parse_slot(self) -> None:
        address = parse_slot("RACK_2.R2C6")
        self.assertEqual(address, SlotAddress("rack_2", 2, 6))
        self.assertEqual(address.text, "rack_2.r2c6")

    def test_invalid_slot_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_slot("rack_1.r3c1")

    def test_same_source_and_destination_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_transfer("rack_1.r1c1", "rack_1.r1c1")

    def test_point_shift_keeps_frame(self) -> None:
        point = Point3D(1.0, 2.0, 3.0, "base_right")
        shifted = point.shifted(dz_mm=5.0)
        self.assertEqual(shifted, Point3D(1.0, 2.0, 8.0, "base_right"))

    def test_occupancy_values_are_stable(self) -> None:
        self.assertEqual(Occupancy.EMPTY.value, "empty")
        self.assertEqual(Occupancy.OCCUPIED.value, "occupied")


if __name__ == "__main__":
    unittest.main()
