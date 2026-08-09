"""Load one readable YAML configuration file."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml

from tube_grabber.core.errors import ConfigError


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path = "config/app.yaml") -> dict[str, Any]:
    """Load and minimally validate the application configuration."""
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if not config_path.is_file():
        raise ConfigError(f"config file does not exist: {config_path}")

    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ConfigError(f"cannot read config: {error}") from error

    if not isinstance(data, dict):
        raise ConfigError("config root must be a mapping")

    required_sections = (
        "runtime",
        "agent",
        "camera",
        "arm",
        "gripper",
        "vision",
        "markers",
        "racks",
        "geometry",
        "motion",
    )
    for section in required_sections:
        if not isinstance(data.get(section), dict):
            raise ConfigError(f"missing config section: {section}")

    mode = data["runtime"].get("mode")
    if mode not in ("fake", "real"):
        raise ConfigError("runtime.mode must be 'fake' or 'real'")

    agent = data["agent"]
    provider = str(agent.get("provider", "")).strip().lower()
    if provider not in ("local", "gemini"):
        raise ConfigError("agent.provider must be local or gemini")
    if provider == "gemini":
        if not str(agent.get("model", "")).strip():
            raise ConfigError("agent.model cannot be empty for Gemini")
        if not str(agent.get("api_key_env", "")).strip():
            raise ConfigError("agent.api_key_env cannot be empty for Gemini")

    class_names = data["vision"].get("class_names")
    if class_names not in ({0: "empty_hole", 1: "tube_cap"}, {"0": "empty_hole", "1": "tube_cap"}):
        raise ConfigError("vision.class_names must define empty_hole and tube_cap")

    if data["arm"].get("work_frame") != "Base":
        raise ConfigError("arm.work_frame must be Base")
    if int(data["arm"].get("expected_dof", 0)) != 7:
        raise ConfigError("arm.expected_dof must be 7")
    if data["arm"].get("tool_frame") != "Arm_Tip":
        raise ConfigError(
            "arm.tool_frame must be Arm_Tip because TCP offset is applied in code"
        )
    if int(data["vision"].get("required_detection_count", 0)) != 12:
        raise ConfigError("vision.required_detection_count must be 12")
    maximum_tool_tilt_deg = float(
        data["motion"].get("maximum_tool_tilt_deg", 0.0)
    )
    if not 0.0 < maximum_tool_tilt_deg <= 5.0:
        raise ConfigError(
            "motion.maximum_tool_tilt_deg must be greater than 0 and at most 5"
        )

    if set(data["racks"]) != {"rack_1", "rack_2"}:
        raise ConfigError("racks must contain exactly rack_1 and rack_2")

    marker_colors = []
    for rack_id, rack in data["racks"].items():
        if not isinstance(rack, dict):
            raise ConfigError(f"racks.{rack_id} must be a mapping")
        if rack.get("marker_color") not in ("red", "green"):
            raise ConfigError(f"racks.{rack_id}.marker_color must be red or green")
        marker_colors.append(rack["marker_color"])
        fallback = rack.get("fallback_plane_z_mm")
        if fallback is not None:
            try:
                fallback_value = float(fallback)
            except (TypeError, ValueError) as error:
                raise ConfigError(
                    f"racks.{rack_id}.fallback_plane_z_mm must be a number or null"
                ) from error
            if not math.isfinite(fallback_value):
                raise ConfigError(
                    f"racks.{rack_id}.fallback_plane_z_mm must be finite"
                )
    if set(marker_colors) != {"red", "green"}:
        raise ConfigError("rack_1 and rack_2 must use different red/green markers")

    return data


def project_path(value: str | Path) -> Path:
    """Resolve a configured path relative to this project."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_yaml(path: str | Path) -> dict[str, Any]:
    resolved = project_path(path)
    if not resolved.is_file():
        raise ConfigError(f"YAML file does not exist: {resolved}")
    try:
        data = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ConfigError(f"cannot read YAML {resolved}: {error}") from error
    if not isinstance(data, dict):
        raise ConfigError(f"YAML root must be a mapping: {resolved}")
    return data
