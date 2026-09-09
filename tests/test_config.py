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
        self.assertEqual(config["vision"]["cap"]["class_names"][0], "item")
        self.assertEqual(config["vision"]["screw"]["class_names"][0], "item")
        self.assertEqual(
            config["vision"]["screw"]["model_path"], "models/screw.pt"
        )
        self.assertEqual(config["geometry"]["grasp_depth_below_cap_mm"], 28.0)
        self.assertEqual(config["agent"]["provider"], "local")
        self.assertEqual(set(config["racks"]), {"rack_1", "rack_2"})

    def test_single_rack_config_is_supported(self) -> None:
        config = load_yaml("config/app.yaml")
        config["racks"].pop("rack_2")
        with TemporaryDirectory() as directory:
            path = Path(directory) / "app.yaml"
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            config = load_config(path)
        self.assertEqual(list(config["racks"]), ["rack_1"])

    def test_empty_rack_config_is_rejected(self) -> None:
        config = load_yaml("config/app.yaml")
        config["racks"] = {}
        with TemporaryDirectory() as directory:
            path = Path(directory) / "app.yaml"
            path.write_text(yaml.safe_dump(config), encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "at least one"):
                load_config(path)

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
