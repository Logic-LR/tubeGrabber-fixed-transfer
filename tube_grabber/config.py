"""Load one readable YAML configuration file."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from tube_grabber.core.errors import ConfigError
from tube_grabber.vision.rack_calibration import RackCircleFitConfig
from tube_grabber.vision.rack_pose import RACK_KEYPOINT_NAMES


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

    vision = data["vision"]
    for subsection in (
        "cap",
        "rack_pose",
        "calibration",
        "stability",
        "plane",
        "matching",
    ):
        if not isinstance(vision.get(subsection), dict):
            raise ConfigError(f"vision.{subsection} must be a mapping")
    device = str(vision.get("device", "")).strip()
    try:
        device_index = int(device)
    except ValueError as error:
        raise ConfigError(
            "vision.device must be a local CUDA device index such as '0'"
        ) from error
    if device_index < 0 or device != str(device_index):
        raise ConfigError(
            "vision.device must be a local CUDA device index such as '0'"
        )
    cap_names = vision["cap"].get("class_names")
    if cap_names not in ({0: "tube_cap"}, {"0": "tube_cap"}):
        raise ConfigError("vision.cap.class_names must be {0: tube_cap}")
    pose_names = vision["rack_pose"].get("class_names")
    if pose_names not in ({0: "rack_surface"}, {"0": "rack_surface"}):
        raise ConfigError(
            "vision.rack_pose.class_names must be {0: rack_surface}"
        )
    keypoint_names = tuple(
        str(value) for value in vision["rack_pose"].get("keypoint_names", ())
    )
    if keypoint_names != RACK_KEYPOINT_NAMES:
        raise ConfigError(
            "vision.rack_pose.keypoint_names must be k0..k3 then screw_k0..screw_k3"
        )
    if not isinstance(vision["rack_pose"].get("k0_red_marker"), dict):
        raise ConfigError("vision.rack_pose.k0_red_marker must be a mapping")
    circle = vision["calibration"]
    try:
        RackCircleFitConfig(
            search_radius_px=int(circle["search_radius_px"]),
            minimum_radius_px=int(circle["minimum_radius_px"]),
            maximum_radius_px=int(circle["maximum_radius_px"]),
            default_radius_px=int(circle["default_radius_px"]),
            hough_dp=float(circle["hough_dp"]),
            hough_min_distance_px=float(circle["hough_min_distance_px"]),
            hough_edge_threshold=float(circle["hough_edge_threshold"]),
            hough_accumulator_threshold=float(
                circle["hough_accumulator_threshold"]
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ConfigError(f"vision.calibration is invalid: {error}") from error
    capture_frames = int(vision["stability"].get("capture_frames", 0))
    minimum_inliers = int(
        vision["stability"].get("minimum_inlier_frames", 0)
    )
    if minimum_inliers < 2 or capture_frames < minimum_inliers:
        raise ConfigError(
            "vision stability requires capture_frames >= minimum_inlier_frames >= 2"
        )

    if data["arm"].get("work_frame") != "Base":
        raise ConfigError("arm.work_frame must be Base")
    if int(data["arm"].get("expected_dof", 0)) != 7:
        raise ConfigError("arm.expected_dof must be 7")
    if data["arm"].get("tool_frame") != "Arm_Tip":
        raise ConfigError(
            "arm.tool_frame must be Arm_Tip because TCP offset is applied in code"
        )
    maximum_tool_tilt_deg = float(
        data["motion"].get("maximum_tool_tilt_deg", 0.0)
    )
    if not 0.0 < maximum_tool_tilt_deg <= 5.0:
        raise ConfigError(
            "motion.maximum_tool_tilt_deg must be greater than 0 and at most 5"
        )

    if set(data["racks"]) != {"rack_1", "rack_2"}:
        raise ConfigError("racks must contain exactly rack_1 and rack_2")

    for rack_id, rack in data["racks"].items():
        if not isinstance(rack, dict):
            raise ConfigError(f"racks.{rack_id} must be a mapping")
        if not str(rack.get("calibration_path", "")).strip():
            raise ConfigError(f"racks.{rack_id}.calibration_path cannot be empty")

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
