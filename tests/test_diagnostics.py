from __future__ import annotations

import unittest

from tube_grabber.app import build_runtime
from tube_grabber.config import load_config
from tube_grabber.diagnostics import format_observation


class DiagnosticsTests(unittest.TestCase):
    def test_observation_includes_slot_state_and_coordinates(self) -> None:
        runtime = build_runtime(load_config())
        observation = runtime.workflow.scan("rack_1")

        output = format_observation(observation)

        self.assertIn("slot coordinate table (base_right, mm):", output)
        self.assertIn("rack_1.r1c1", output)
        self.assertIn("cap_top", output)
        self.assertIn("[20.00,290.00,-423.00]", output)
        self.assertIn("rack_1.r1c2", output)
        self.assertIn("rack_plane", output)
        self.assertIn("[42.00,290.00,-470.00]", output)


if __name__ == "__main__":
    unittest.main()
