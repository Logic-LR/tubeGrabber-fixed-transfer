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
from tube_grabber.vision.detector import YoloCapDetector
from tube_grabber.vision.plane import RackPlaneFitConfig
from tube_grabber.vision.pose_pipeline import (
    CapturedRackFrame,
    PoseRackVision,
    RackMatchingConfig,
    RackStabilityConfig,
)
from tube_grabber.vision.rack_calibration import load_rack_calibration
from tube_grabber.vision.rack_pose import (
    RackPoseQualityConfig,
    RackPoseStability,
    YoloRackPoseDetector,
)
from tube_grabber.workflow import ManipulationWorkflow


class CameraRackObserver:
    """Capture one stationary wrist-camera frame and run rack perception."""

    def __init__(
        self,
        camera: CameraPort,
        arm: ArmPort,
        vision: PoseRackVision,
    ) -> None:
        self.camera = camera
        self.arm = arm
        self.vision = vision
        self.last_frame: CameraFrame | None = None
        self.last_observation: RackObservation | None = None
        self.last_pose_stability: RackPoseStability | None = None

    def observe_rack(self, rack_id: str) -> RackObservation:
        return self.observe_rack_for_task(rack_id)

    def observe_rack_for_task(
        self,
        rack_id: str,
        *,
        ignored_elevated_slot: SlotAddress | None = None,
    ) -> RackObservation:
        samples = self._capture_samples()
        observation = self.vision.observe(
            samples,
            rack_id,
            ignored_elevated_slot=ignored_elevated_slot,
        )
        self.last_frame = min(
            samples,
            key=lambda item: abs(item.frame.timestamp_ms - observation.timestamp_ms),
        ).frame
        self.last_observation = observation
        return observation

    def capture_stable_pose(self) -> tuple[CameraFrame, RackPoseStability]:
        """Capture the same filtered rack pose used by interactive calibration."""
        samples = self._capture_samples()
        stability, _ = self.vision.detect_stable_pose(samples)
        frame = samples[stability.inlier_indices[-1]].frame
        self.last_frame = frame
        self.last_pose_stability = stability
        return frame, stability

    def _capture_samples(self) -> tuple[CapturedRackFrame, ...]:
        samples: list[CapturedRackFrame] = []
        reference_pose: Pose6D | None = None
        for _ in range(self.vision.capture_frames):
            pose_before = self.arm.get_pose()
            frame = self.camera.capture()
            pose_after = self.arm.get_pose()
            self._require_stationary(pose_before, pose_after, "during D435 capture")
            if reference_pose is None:
                reference_pose = pose_after
            else:
                self._require_stationary(
                    reference_pose,
                    pose_after,
                    "across the multi-frame inference window",
                )
            samples.append(CapturedRackFrame(frame, pose_after))
        return tuple(samples)

    @staticmethod
    def _require_stationary(first: Pose6D, second: Pose6D, context: str) -> None:
        if _pose_distance_mm(first, second) > 0.5:
            raise VisionError(f"right arm moved {context}")
        angle = rotation_distance_deg(
            rpy_to_rotation((first.rx_rad, first.ry_rad, first.rz_rad)),
            rpy_to_rotation((second.rx_rad, second.ry_rad, second.rz_rad)),
        )
        if angle > 0.2:
            raise VisionError(f"right arm rotated {context}")


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

    def move_to_observation_pose(self) -> Pose6D:
        """Move to and verify the taught global observation pose."""
        if self.mode == "real" and not self.observation_pose_confirmed:
            raise ConfigError(
                "config/poses.yaml observation_pose.confirmed is false; "
                "verify the taught pose at low speed before automatic motion"
            )
        self.require_motion_ready()
        self.workflow.move_to_observation_pose()
        return self.require_observation_pose()

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
                f"right arm power state is {power_state}; "
                "power it on from the controller first"
            )
        self.arm.require_healthy()


def build_runtime(
    config: dict[str, Any],
    *,
    load_calibrations: bool = True,
) -> TubeGrabberRuntime:
    """Create fake or real implementations without connecting any hardware."""
    runtime_config = config["runtime"]
    arm_config = config["arm"]
    camera_config = config["camera"]
    gripper_config = config["gripper"]
    vision_config = config["vision"]
    geometry_config = config["geometry"]
    motion_config = config["motion"]
    cap_config = vision_config["cap"]
    pose_config = vision_config["rack_pose"]
    stability_config = vision_config["stability"]
    plane_data = vision_config["plane"]
    matching_config = vision_config["matching"]
    pose_quality = _rack_pose_quality_config(pose_config)
    rack_stability = _rack_stability_config(stability_config)
    rack_plane = _rack_plane_config(plane_data)
    rack_matching = _rack_matching_config(
        cap_config,
        matching_config,
        camera_config,
        geometry_config,
    )

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
            reject_conflicting_processes=bool(
                arm_config.get("reject_conflicting_processes", True)
            ),
            conflicting_processes=arm_config.get(
                "conflicting_processes", ("atom", "zhixing_ctrl.py")
            ),
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
            open_position=int(gripper_config["open_position"]),
            grip_position=int(gripper_config["grip_position"]),
            reset_position=int(gripper_config["reset_position"]),
            command_timeout_s=int(gripper_config["command_timeout_s"]),
            position_tolerance=int(gripper_config["position_tolerance"]),
            poll_interval_s=float(gripper_config["poll_interval_s"]),
            state_read_attempts=int(gripper_config["state_read_attempts"]),
            state_read_retry_s=float(gripper_config["state_read_retry_s"]),
        )
        cap_detector = YoloCapDetector(
            model_path=project_path(cap_config["model_path"]),
            class_names=cap_config["class_names"],
            confidence=float(cap_config["confidence"]),
            iou=float(cap_config["iou"]),
            image_size=int(cap_config["image_size"]),
            device=str(vision_config["device"]),
        )
        pose_detector = YoloRackPoseDetector(
            model_path=project_path(pose_config["model_path"]),
            class_names=pose_config["class_names"],
            keypoint_names=pose_config["keypoint_names"],
            confidence=float(pose_config["confidence"]),
            keypoint_confidence=float(pose_config["keypoint_confidence"]),
            iou=float(pose_config["iou"]),
            image_size=int(pose_config["image_size"]),
            device=str(vision_config["device"]),
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
        calibrations = {}
        for rack_id, rack in config["racks"].items():
            calibration_path = project_path(rack["calibration_path"])
            calibrations[rack_id] = (
                load_rack_calibration(calibration_path, rack_id)
                if load_calibrations and calibration_path.is_file()
                else None
            )
        vision = PoseRackVision(
            rack_pose_detector=pose_detector,
            cap_detector=cap_detector,
            calibrations=calibrations,
            hand_eye_end_from_camera=hand_eye["matrix"],
            pose_quality=pose_quality,
            stability=rack_stability,
            plane_config=rack_plane,
            matching=rack_matching,
        )
        observer: object = CameraRackObserver(camera, arm, vision)
    elif mode == "fake":
        arm = FakeArm(observation_pose)
        camera = None
        gripper = FakeGripper()
        cap_height = float(geometry_config["cap_top_above_rack_mm"])
        observer = FakeRackObserver(
            *(
                _fake_observation(rack_id, cap_height)
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
        maximum_single_orientation_change_deg=float(
            motion_config["maximum_single_orientation_change_deg"]
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
        observation_pose=observation_pose,
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
    *,
    occupied: set[tuple[int, int]] | None = None,
) -> RackObservation:
    """Deterministic 2x6 rack used by fake mode and the full-chain test."""
    if occupied is None:
        occupied = {(1, 1)}
    plane_z_mm = -470.0
    slots: list[SlotObservation] = []
    for row in (1, 2):
        for column in range(1, 7):
            address = SlotAddress(rack_id, row, column)
            x_mm = 20.0 + 22.0 * (column - 1)
            y_mm = 290.0 + 32.0 * (row - 1)
            is_occupied = (row, column) in occupied
            hole = Point3D(x_mm, y_mm, plane_z_mm)
            slots.append(
                SlotObservation(
                    address=address,
                    occupancy=(
                        Occupancy.OCCUPIED if is_occupied else Occupancy.EMPTY
                    ),
                    confidence=0.99,
                    pixel=Pixel(x_mm, y_mm),
                    cap_top_base=(
                        Point3D(
                            x_mm,
                            y_mm,
                            plane_z_mm + cap_top_above_rack_mm,
                        )
                        if is_occupied
                        else None
                    ),
                    hole_on_plane_base=hole,
                )
            )
    return RackObservation(
        rack_id=rack_id,
        marker=Pixel(10.0, 280.0),
        plane_z_mm=plane_z_mm,
        slots=tuple(slots),
        timestamp_ms=0.0,
    )


def _rack_pose_quality_config(data: dict[str, Any]) -> RackPoseQualityConfig:
    red = _mapping(data.get("k0_red_marker"), "vision.rack_pose.k0_red_marker")
    return RackPoseQualityConfig(
        minimum_rack_area_px2=float(data["minimum_rack_area_px2"]),
        maximum_opposite_side_ratio=float(data["maximum_opposite_side_ratio"]),
        screw_edge_margin=float(data["screw_edge_margin"]),
        k0_red_patch_radius_px=int(red["patch_radius_px"]),
        k0_red_minimum_ratio=float(red["minimum_red_ratio"]),
        k0_red_minimum_ratio_margin=float(red["minimum_ratio_margin"]),
        k0_red_minimum_saturation=int(red["minimum_saturation"]),
        k0_red_minimum_value=int(red["minimum_value"]),
    )


def _rack_stability_config(data: dict[str, Any]) -> RackStabilityConfig:
    return RackStabilityConfig(
        capture_frames=int(data["capture_frames"]),
        minimum_inlier_frames=int(data["minimum_inlier_frames"]),
        maximum_frame_residual_px=float(data["maximum_frame_residual_px"]),
        maximum_keypoint_spread_px=float(data["maximum_keypoint_spread_px"]),
        minimum_occupancy_agreement=float(data["minimum_occupancy_agreement"]),
        maximum_slot_position_spread_mm=float(
            data["maximum_slot_position_spread_mm"]
        ),
        maximum_cap_position_spread_mm=float(
            data["maximum_cap_position_spread_mm"]
        ),
    )


def _rack_plane_config(data: dict[str, Any]) -> RackPlaneFitConfig:
    return RackPlaneFitConfig(
        sample_stride_px=int(data["sample_stride_px"]),
        roi_margin_px=int(data["roi_margin_px"]),
        landmark_exclusion_radius_px=int(
            data["landmark_exclusion_radius_px"]
        ),
        ransac_iterations=int(data["ransac_iterations"]),
        inlier_threshold_mm=float(data["inlier_threshold_mm"]),
        minimum_inliers=int(data["minimum_inliers"]),
        minimum_inlier_ratio=float(data["minimum_inlier_ratio"]),
        maximum_rms_error_mm=float(data["maximum_rms_error_mm"]),
        maximum_tilt_deg=float(data["maximum_tilt_deg"]),
    )


def _rack_matching_config(
    cap: dict[str, Any],
    matching: dict[str, Any],
    camera: dict[str, Any],
    geometry: dict[str, Any],
) -> RackMatchingConfig:
    return RackMatchingConfig(
        depth_window_px=int(cap["depth_window_px"]),
        depth_min_mm=float(camera["depth_min_mm"]),
        depth_max_mm=float(camera["depth_max_mm"]),
        cap_top_above_rack_mm=float(geometry["cap_top_above_rack_mm"]),
        maximum_cap_height_error_mm=float(cap["maximum_height_error_mm"]),
        cap_slot_max_distance_factor=float(
            matching["cap_slot_max_distance_factor"]
        ),
        maximum_calibration_keypoint_shift_ratio=float(
            matching["maximum_calibration_keypoint_shift_ratio"]
        ),
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
