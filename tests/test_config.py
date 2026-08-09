from __future__ import annotations

import unittest

from tube_grabber.config import load_config


class ConfigTest(unittest.TestCase):
    def test_default_config_loads(self) -> None:
        config = load_config()
        self.assertEqual(config["runtime"]["mode"], "fake")
        self.assertEqual(config["vision"]["class_names"][0], "empty_hole")
        self.assertEqual(config["vision"]["class_names"][1], "tube_cap")
        self.assertEqual(config["geometry"]["grasp_depth_below_cap_mm"], 5.0)
        self.assertEqual(config["agent"]["provider"], "local")


if __name__ == "__main__":
    unittest.main()
