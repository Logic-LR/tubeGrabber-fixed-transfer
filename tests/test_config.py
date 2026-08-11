from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from tube_grabber.config import load_config, load_yaml
from tube_grabber.core.errors import ConfigError


class ConfigTest(unittest.TestCase):
    def test_default_config_loads(self) -> None:
        config = load_config()
        self.assertEqual(config["runtime"]["mode"], "fake")
        self.assertEqual(config["vision"]["cap"]["class_names"][0], "tube_cap")
        self.assertEqual(
            config["vision"]["rack_pose"]["keypoint_names"][:4],
            ["k0", "k1", "k2", "k3"],
        )
        self.assertEqual(config["geometry"]["grasp_depth_below_cap_mm"], 5.0)
        self.assertEqual(config["agent"]["provider"], "local")

    def test_cpu_inference_is_rejected(self) -> None:
        config = load_yaml("config/app.yaml")
        config["vision"]["device"] = "cpu"
        with TemporaryDirectory() as directory:
            path = Path(directory) / "app.yaml"
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "CUDA device index"):
                load_config(path)


if __name__ == "__main__":
    unittest.main()
