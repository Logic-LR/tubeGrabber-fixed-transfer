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
        self.assertIn("[266.77,300.00,-216.77]", output)
        self.assertIn("rack_1.r1c2", output)
        self.assertIn("rack_plane", output)
        self.assertIn("[322.00,300.00,-228.00]", output)
        self.assertIn("rack_up_base=[-0.70711, +0.00000, +0.70711]", output)


if __name__ == "__main__":
    unittest.main()
