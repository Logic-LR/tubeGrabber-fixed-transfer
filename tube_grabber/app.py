"""Build the final runtime from the single YAML configuration."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from tube_grabber.config import load_yaml, project_path
from tube_grabber.core.errors import ConfigError, HardwareError, VisionError
from tube_grabber.core.models import (
    CameraFrame,
    Occupancy,
    Pixel,
    Point3D,
    Pose6D,
    RackObservation,
    SlotAddress,
    SlotObservation,
)
from tube_grabber.core.ports import ArmPort, CameraPort, GripperPort
from tube_grabber.fakes import FakeArm, FakeGripper, FakeRackObserver
from tube_grabber.hardware import D435Camera, RealManArm, RealManGripper
from tube_grabber.motion import MotionExecutor, MotionPlanner
from tube_grabber.motion.planner import rotation_distance_deg, rpy_to_rotation
from tube_grabber.vision.detector import YoloDetector
from tube_grabber.vision.grid import GridMapper
from tube_grabber.vision.marker import K0MarkerDetector
from tube_grabber.vision.pipeline import RackVision
from tube_grabber.workflow import ManipulationWorkflow


class CameraRackObserver:
    """Capture one stationary wrist-camera frame and run rack perception."""

    def __init__(
        self,
        camera: CameraPort,
        arm: ArmPort,
        vision: RackVision,
    ) -> None:
        self.camera = camera
        self.arm = arm
        self.vision = vision
        self.last_frame: CameraFrame | None = None
        self.last_observation: RackObservation | None = None

    def observe_rack(self, rack_id: str) -> RackObservation:
        pose_before = self.arm.get_pose()
        frame = self.camera.capture()
        pose_after = self.arm.get_pose()
        if _pose_distance_mm(pose_before, pose_after) > 0.5:
            raise VisionError("right arm moved while the D435 frame was captured")
        angle = rotation_distance_deg(
            rpy_to_rotation(
                (pose_before.rx_rad, pose_before.ry_rad, pose_before.rz_rad)
            ),
            rpy_to_rotation(
                (pose_after.rx_rad, pose_after.ry_rad, pose_after.rz_rad)
            ),
        )
        if angle > 0.2:
            raise VisionError("right arm rotated while the D435 frame was captured")

        observation = self.vision.observe(frame, rack_id, pose_after)
        self.last_frame = frame
        self.last_observation = observation
        return observation


@dataclass
class TubeGrabberRuntime:
    """The fully assembled application and its hardware lifecycle."""

    mode: str
    arm: ArmPort
    gripper: GripperPort
    observer: object
    workflow: ManipulationWorkflow
    observation_pose: Pose6D
    observation_pose_confirmed: bool
    motion_parameters_confirmed: bool
    observation_position_tolerance_mm: float
    observation_orientation_tolerance_deg: float
    camera: CameraPort | None = None

    def start(
        self,
        *,
        need_arm: bool,
        need_camera: bool,
        need_gripper: bool,
    ) -> None:
        if need_arm:
            self.arm.connect()
        if need_camera:
            if self.camera is None:
                raise HardwareError("configured runtime has no real D435 camera")
            self.camera.start()
        if need_gripper:
            if not need_arm:
                raise HardwareError("gripper setup requires an arm connection")
            self.gripper.setup()

    def close(self) -> None:
        """Release resources in reverse order; cleanup never hides the main error."""
        if self.camera is not None:
            try:
                self.camera.stop()
            except Exception:
                pass
        try:
            self.arm.disconnect()
        except Exception:
            pass

    def require_observation_pose(self) -> Pose6D:
        if self.mode == "real" and not self.observation_pose_confirmed:
            raise ConfigError(
                "config/poses.yaml observation_pose.confirmed is false; "
                "verify the taught pose at low speed before scanning"
            )
        current = self.arm.get_pose()
        distance = _pose_distance_mm(current, self.observation_pose)
        angle = rotation_distance_deg(
            rpy_to_rotation((current.rx_rad, current.ry_rad, current.rz_rad)),
            rpy_to_rotation(
                (
                    self.observation_pose.rx_rad,
                    self.observation_pose.ry_rad,
                    self.observation_pose.rz_rad,
                )
            ),
        )
        if distance > self.observation_position_tolerance_mm:
            raise HardwareError(
                f"right arm is {distance:.1f} mm from the observation pose; "
                f"limit is {self.observation_position_tolerance_mm:.1f} mm"
            )
        if angle > self.observation_orientation_tolerance_deg:
            raise HardwareError(
                f"right arm orientation is {angle:.2f} deg from the observation pose; "
                f"limit is {self.observation_orientation_tolerance_deg:.2f} deg"
            )
        return current

    def require_motion_ready(self) -> None:
        """Read real-controller state; never power on or change run mode here."""
        if not isinstance(self.arm, RealManArm):
            return
        if not self.motion_parameters_confirmed:
            raise ConfigError(
                "motion.parameters_confirmed is false; verify TCP offset, "
                "workspace and vertical heights at low speed first"
            )
        run_mode = self.arm.get_run_mode()
        if run_mode != 1:
            raise HardwareError(
                f"controller run mode is {run_mode}; select physical/real mode first"
            )
        power_state = self.arm.get_power_state()
        if power_state != 1:
            raise HardwareError(
                f"right arm power state is {power_state}; power it on from the controller first"
            )


def build_runtime(config: dict[str, Any]) -> TubeGrabberRuntime:
    """Create fake or real implementations without connecting any hardware."""
    runtime_config = config["runtime"]
    arm_config = config["arm"]
    camera_config = config["camera"]
    gripper_config = config["gripper"]
    vision_config = config["vision"]
    marker_config = config["markers"]
    geometry_config = config["geometry"]
    motion_config = config["motion"]

    poses = load_yaml(motion_config["poses_path"])
    observation_data = _mapping(poses.get("observation_pose"), "observation_pose")
    observation_pose = _pose_from_config(observation_data, "observation_pose")
    vertical_rpy = _vector3(
        poses.get("vertical_tool_rpy_rad"),
        "vertical_tool_rpy_rad",
    )

    mode = str(runtime_config["mode"])
    if mode == "real":
        arm = RealManArm(
            ip=str(arm_config["ip"]),
            port=int(arm_config["port"]),
            work_frame=str(arm_config.get("work_frame", "Base")),
            tool_frame=str(arm_config.get("tool_frame", "Arm_Tip")),
            expected_dof=int(arm_config.get("expected_dof", 7)),
        )
        camera = D435Camera(
            serial=str(camera_config["serial"]),
            width=int(camera_config["width"]),
            height=int(camera_config["height"]),
            fps=int(camera_config["fps"]),
            warmup_frames=int(camera_config["warmup_frames"]),
            timeout_ms=int(camera_config["timeout_ms"]),
        )
        gripper = RealManGripper(
            arm=arm,
            baudrate=int(gripper_config["baudrate"]),
            tool_voltage=int(gripper_config["tool_voltage"]),
            open_position=int(gripper_config["open_position"]),
            grip_position=int(gripper_config["grip_position"]),
            reset_position=int(gripper_config["reset_position"]),
            command_timeout_s=int(gripper_config["command_timeout_s"]),
            power_on_wait_s=float(gripper_config["power_on_wait_s"]),
            protocol_startup_wait_s=float(
                gripper_config["protocol_startup_wait_s"]
            ),
        )
        detector = YoloDetector(
            model_path=project_path(vision_config["model_path"]),
            class_names=vision_config["class_names"],
            confidence=float(vision_config["confidence"]),
            iou=float(vision_config["iou"]),
            image_size=int(vision_config["image_size"]),
            device=str(vision_config["device"]),
        )
        marker_detector = K0MarkerDetector(
            minimum_area_px=int(marker_config["minimum_area_px"]),
            minimum_saturation=int(marker_config["minimum_saturation"]),
            minimum_value=int(marker_config["minimum_value"]),
            maximum_aspect_ratio=float(
                marker_config["maximum_aspect_ratio"]
            ),
            hue_ranges={
                "red": marker_config["red_hue_ranges"],
                "green": marker_config["green_hue_ranges"],
            },
        )
        hand_eye = load_yaml(geometry_config["hand_eye_path"])
        if hand_eye.get("translation_unit") != "mm":
            raise ConfigError("hand-eye translation_unit must be mm")
        if hand_eye.get("source_frame") != "camera_rightwrist":
            raise ConfigError(
                "hand-eye source_frame must be camera_rightwrist"
            )
        if hand_eye.get("target_frame") != "end_right":
            raise ConfigError("hand-eye target_frame must be end_right")
        rack_colors = {
            rack_id: str(rack["marker_color"])
            for rack_id, rack in config["racks"].items()
        }
        fallback_planes = {
            rack_id: rack.get("fallback_plane_z_mm")
            for rack_id, rack in config["racks"].items()
        }
        vision = RackVision(
            detector=detector,
            marker_detector=marker_detector,
            grid_mapper=GridMapper(
                maximum_spacing_cv=float(vision_config["maximum_spacing_cv"]),
                maximum_marker_corner_distance_factor=float(
                    vision_config["maximum_marker_corner_distance_factor"]
                ),
            ),
            hand_eye_end_from_camera=hand_eye["matrix"],
            rack_marker_colors=rack_colors,
            fallback_plane_z_mm=fallback_planes,
            required_detection_count=int(
                vision_config["required_detection_count"]
            ),
            depth_window_px=int(vision_config["depth_window_px"]),
            depth_min_mm=float(camera_config["depth_min_mm"]),
            depth_max_mm=float(camera_config["depth_max_mm"]),
            cap_top_above_rack_mm=float(
                geometry_config["cap_top_above_rack_mm"]
            ),
            maximum_cap_z_deviation_mm=float(
                vision_config["maximum_cap_z_deviation_mm"]
            ),
            maximum_plane_calibration_error_mm=float(
                vision_config["maximum_plane_calibration_error_mm"]
            ),
        )
        observer: object = CameraRackObserver(camera, arm, vision)
    elif mode == "fake":
        arm = FakeArm(observation_pose)
        camera = None
        gripper = FakeGripper()
        observer = FakeRackObserver(
            *(
                _fake_observation(
                    rack_id,
                    float(geometry_config["cap_top_above_rack_mm"]),
                )
                for rack_id in config["racks"]
            )
        )
    else:
        raise ConfigError("runtime.mode must be fake or real")

    planner = MotionPlanner(
        tcp_offset_end_mm=geometry_config["tcp_offset_end_mm"],
        vertical_tool_rpy_rad=vertical_rpy,
        approach_height_mm=float(motion_config["approach_height_mm"]),
        retreat_height_mm=float(motion_config["retreat_height_mm"]),
        transit_speed_percent=int(arm_config["transit_speed_percent"]),
        approach_speed_percent=int(arm_config["approach_speed_percent"]),
        maximum_orientation_error_deg=float(
            motion_config["maximum_orientation_error_deg"]
        ),
        maximum_tool_tilt_deg=float(
            motion_config["maximum_tool_tilt_deg"]
        ),
        tube_total_length_mm=float(
            geometry_config["tube_total_length_mm"]
        ),
        required_carried_clearance_mm=float(
            geometry_config["required_carried_clearance_mm"]
        ),
    )
    executor = MotionExecutor(
        arm,
        workspace_min_mm=motion_config["workspace_min_mm"],
        workspace_max_mm=motion_config["workspace_max_mm"],
        maximum_single_move_mm=float(motion_config["maximum_single_move_mm"]),
        position_reached_tolerance_mm=float(
            motion_config["position_reached_tolerance_mm"]
        ),
        orientation_reached_tolerance_deg=float(
            motion_config["orientation_reached_tolerance_deg"]
        ),
    )
    workflow = ManipulationWorkflow(
        arm=arm,
        gripper=gripper,
        observer=observer,
        planner=planner,
        executor=executor,
        grasp_depth_below_cap_mm=float(
            geometry_config["grasp_depth_below_cap_mm"]
        ),
        cap_top_above_rack_mm=float(
            geometry_config["cap_top_above_rack_mm"]
        ),
        seating_adjust_mm=float(geometry_config["seating_adjust_mm"]),
        scene_recheck_pixel_tolerance_px=float(
            runtime_config["scene_recheck_pixel_tolerance_px"]
        ),
        scene_recheck_position_tolerance_mm=float(
            runtime_config["scene_recheck_position_tolerance_mm"]
        ),
        scene_recheck_plane_tolerance_mm=float(
            runtime_config["scene_recheck_plane_tolerance_mm"]
        ),
    )
    return TubeGrabberRuntime(
        mode=mode,
        arm=arm,
        gripper=gripper,
        observer=observer,
        workflow=workflow,
        observation_pose=observation_pose,
        observation_pose_confirmed=bool(observation_data.get("confirmed", False)),
        motion_parameters_confirmed=bool(
            motion_config.get("parameters_confirmed", False)
        ),
        observation_position_tolerance_mm=float(
            runtime_config["observation_position_tolerance_mm"]
        ),
        observation_orientation_tolerance_deg=float(
            runtime_config["observation_orientation_tolerance_deg"]
        ),
        camera=camera,
    )


def _fake_observation(
    rack_id: str,
    cap_top_above_rack_mm: float,
) -> RackObservation:
    """Deterministic 2x6 rack used by fake mode and the full-chain test."""
    plane_z_mm = -470.0
    slots: list[SlotObservation] = []
    for row in (1, 2):
        for column in range(1, 7):
            address = SlotAddress(rack_id, row, column)
            x_mm = 20.0 + 22.0 * (column - 1)
            y_mm = 290.0 + 32.0 * (row - 1)
            occupied = row == 1 and column == 1
            slots.append(
                SlotObservation(
                    address=address,
                    occupancy=(
                        Occupancy.OCCUPIED if occupied else Occupancy.EMPTY
                    ),
                    confidence=0.99,
                    pixel=Pixel(x_mm, y_mm),
                    cap_top_base=(
                        Point3D(
                            x_mm,
                            y_mm,
                            plane_z_mm + cap_top_above_rack_mm,
                        )
                        if occupied
                        else None
                    ),
                    hole_on_plane_base=(
                        None
                        if occupied
                        else Point3D(x_mm, y_mm, plane_z_mm)
                    ),
                )
            )
    return RackObservation(
        rack_id=rack_id,
        marker=Pixel(10.0, 280.0),
        plane_z_mm=plane_z_mm,
        slots=tuple(slots),
        timestamp_ms=0.0,
    )


def _pose_from_config(data: dict[str, Any], name: str) -> Pose6D:
    position = _vector3(data.get("position_mm"), f"{name}.position_mm")
    rpy = _vector3(data.get("rpy_rad"), f"{name}.rpy_rad")
    return Pose6D(*position, *rpy)


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{name} must be a mapping")
    return value


def _vector3(value: object, name: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ConfigError(f"{name} must contain three numbers")
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as error:
        raise ConfigError(f"{name} must contain three numbers") from error
    if not all(math.isfinite(item) for item in result):
        raise ConfigError(f"{name} must contain finite numbers")
    return result


def _pose_distance_mm(first: Pose6D, second: Pose6D) -> float:
    return math.sqrt(
        (first.x_mm - second.x_mm) ** 2
        + (first.y_mm - second.y_mm) ** 2
        + (first.z_mm - second.z_mm) ** 2
    )
