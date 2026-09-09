"""Fixed-pose tube pickup and basket placement without vision dependencies.

The workflow deliberately keeps configuration, validation, and execution
separate.  A complete route is validated before the gripper or arm receives
its first command, and real execution requires two independent confirmation
locks: the application's motion parameters and this fixed program.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from tube_grabber.core.errors import ConfigError, HardwareError, WorkflowError
from tube_grabber.core.models import MotionPlan, Pose6D, Waypoint
from tube_grabber.core.ports import ArmPort, GripperPort
from tube_grabber.fakes.hardware import FakeArm, FakeGripper
from tube_grabber.hardware.realman_arm import RealManArm
from tube_grabber.hardware.realman_gripper import RealManGripper
from tube_grabber.motion.executor import MotionExecutor
from tube_grabber.motion.planner import rotation_distance_deg, rpy_to_rotation


PROGRAM_SCHEMA = "fixed_tube_transfer.v1"
POSE_NAMES = (
    "home",
    "pick_above",
    "pick",
    "basket_above",
    "basket_release",
)


@dataclass(frozen=True)
class FixedTransferProgram:
    confirmed: bool
    poses: Mapping[str, Pose6D]
    transit_speed_percent: int
    approach_speed_percent: int
    start_position_tolerance_mm: float
    start_orientation_tolerance_deg: float

    def __post_init__(self) -> None:
        missing = [name for name in POSE_NAMES if name not in self.poses]
        extra = [name for name in self.poses if name not in POSE_NAMES]
        if missing or extra:
            raise ConfigError(
                f"fixed transfer poses mismatch; missing={missing}, extra={extra}"
            )
        for name, pose in self.poses.items():
            if pose.frame != "base_right":
                raise ConfigError(f"fixed transfer pose {name} must use base_right")
        _speed(self.transit_speed_percent, "transit_speed_percent")
        _speed(self.approach_speed_percent, "approach_speed_percent")
        _positive(self.start_position_tolerance_mm, "start_position_tolerance_mm")
        _positive(
            self.start_orientation_tolerance_deg,
            "start_orientation_tolerance_deg",
        )

    @property
    def home(self) -> Pose6D:
        return self.poses["home"]

    def plans(self) -> tuple[tuple[str, MotionPlan], ...]:
        transit = self.transit_speed_percent
        approach = self.approach_speed_percent
        return (
            (
                "pick_approach",
                MotionPlan(
                    (
                        Waypoint("pick_above", self.poses["pick_above"], transit),
                        Waypoint("pick", self.poses["pick"], approach, linear=True),
                    )
                ),
            ),
            (
                "carry_and_place",
                MotionPlan(
                    (
                        Waypoint(
                            "pick_retreat",
                            self.poses["pick_above"],
                            approach,
                            linear=True,
                        ),
                        Waypoint(
                            "basket_above",
                            self.poses["basket_above"],
                            transit,
                        ),
                        Waypoint(
                            "basket_release",
                            self.poses["basket_release"],
                            approach,
                            linear=True,
                        ),
                    )
                ),
            ),
            (
                "retreat_and_home",
                MotionPlan(
                    (
                        Waypoint(
                            "basket_retreat",
                            self.poses["basket_above"],
                            approach,
                            linear=True,
                        ),
                        Waypoint("home", self.poses["home"], transit),
                    )
                ),
            ),
        )

    def require_start(self, actual: Pose6D) -> None:
        position_error = _position_distance_mm(actual, self.home)
        orientation_error = rotation_distance_deg(
            rpy_to_rotation((actual.rx_rad, actual.ry_rad, actual.rz_rad)),
            rpy_to_rotation(
                (self.home.rx_rad, self.home.ry_rad, self.home.rz_rad)
            ),
        )
        if position_error > self.start_position_tolerance_mm:
            raise WorkflowError(
                "arm is not at fixed-transfer home: "
                f"position error {position_error:.2f} mm exceeds "
                f"{self.start_position_tolerance_mm:.2f} mm"
            )
        if orientation_error > self.start_orientation_tolerance_deg:
            raise WorkflowError(
                "arm is not at fixed-transfer home: "
                f"orientation error {orientation_error:.2f} deg exceeds "
                f"{self.start_orientation_tolerance_deg:.2f} deg"
            )


class FixedTransferWorkflow:
    """Execute one fixed pickup/place cycle using guarded Cartesian waypoints."""

    def __init__(
        self,
        *,
        arm: ArmPort,
        gripper: GripperPort,
        executor: MotionExecutor,
        program: FixedTransferProgram,
    ) -> None:
        self.arm = arm
        self.gripper = gripper
        self.executor = executor
        self.program = program

    def preview(self, start_pose: Pose6D) -> tuple[tuple[str, MotionPlan], ...]:
        """Validate the complete cycle without sending hardware commands."""
        self.program.require_start(start_pose)
        phases = self.program.plans()
        previous = start_pose
        for _, plan in phases:
            previous = self.executor.validate(plan, previous)
        return phases

    def execute(self) -> None:
        if not self.program.confirmed:
            raise ConfigError(
                "fixed transfer program is not confirmed; teach every pose at "
                "low speed, then set confirmed: true"
            )
        try:
            phases = self.preview(self.arm.get_pose())
            self.gripper.open_for_pick()
            self.executor.execute(phases[0][1])
            self.gripper.grip()
            self.executor.execute(phases[1][1])
            self.gripper.release()
            self.executor.execute(phases[2][1])
        except Exception as error:
            try:
                self.arm.stop()
            except Exception:
                pass
            if isinstance(error, (ConfigError, HardwareError, WorkflowError)):
                raise
            raise WorkflowError(f"fixed transfer failed: {error}") from error


@dataclass
class FixedTransferRuntime:
    mode: str
    arm: ArmPort
    gripper: GripperPort
    executor: MotionExecutor
    workflow: FixedTransferWorkflow
    program: FixedTransferProgram
    motion_parameters_confirmed: bool

    def start(self, *, need_gripper: bool) -> None:
        self.arm.connect()
        if need_gripper:
            self.gripper.setup()

    def require_motion_ready(self) -> None:
        if not isinstance(self.arm, RealManArm):
            return
        if not self.motion_parameters_confirmed:
            raise ConfigError(
                "motion.parameters_confirmed is false; verify TCP, workspace, "
                "and fixed route at low speed first"
            )
        if not self.program.confirmed:
            raise ConfigError(
                "fixed transfer program is not confirmed; teach and verify all poses"
            )
        if self.arm.get_run_mode() != 1:
            raise HardwareError("controller is not in physical/real mode")
        if self.arm.get_power_state() != 1:
            raise HardwareError("right arm is not powered on")
        self.arm.require_healthy()

    def close(self) -> None:
        try:
            self.arm.disconnect()
        except Exception:
            pass


def load_fixed_transfer_program(path: str | Path) -> FixedTransferProgram:
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"fixed transfer config does not exist: {config_path}")
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ConfigError(f"cannot read fixed transfer config: {error}") from error
    if not isinstance(data, Mapping):
        raise ConfigError("fixed transfer config root must be a mapping")
    if data.get("schema_version") != PROGRAM_SCHEMA:
        raise ConfigError(f"schema_version must be {PROGRAM_SCHEMA}")
    if data.get("frame") != "base_right":
        raise ConfigError("fixed transfer frame must be base_right")
    poses_data = _mapping(data.get("poses"), "poses")
    poses = {
        name: _pose(_mapping(poses_data.get(name), f"poses.{name}"), name)
        for name in POSE_NAMES
    }
    motion = _mapping(data.get("motion"), "motion")
    start = _mapping(data.get("start_gate"), "start_gate")
    try:
        return FixedTransferProgram(
            confirmed=bool(data.get("confirmed", False)),
            poses=poses,
            transit_speed_percent=int(motion["transit_speed_percent"]),
            approach_speed_percent=int(motion["approach_speed_percent"]),
            start_position_tolerance_mm=float(start["position_tolerance_mm"]),
            start_orientation_tolerance_deg=float(
                start["orientation_tolerance_deg"]
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ConfigError(f"fixed transfer settings are invalid: {error}") from error


def build_fixed_transfer_runtime(
    app_config: Mapping[str, Any],
    program: FixedTransferProgram,
) -> FixedTransferRuntime:
    runtime_config = _mapping(app_config.get("runtime"), "runtime")
    arm_config = _mapping(app_config.get("arm"), "arm")
    gripper_config = _mapping(app_config.get("gripper"), "gripper")
    motion_config = _mapping(app_config.get("motion"), "motion")
    mode = str(runtime_config.get("mode"))
    if mode == "real":
        arm: ArmPort = RealManArm(
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
        gripper: GripperPort = RealManGripper(
            arm=arm,
            baudrate=int(gripper_config["baudrate"]),
            tool_voltage=int(gripper_config["tool_voltage"]),
            speed=int(gripper_config["speed"]),
            force=int(gripper_config["force"]),
            open_position=int(gripper_config["open_position"]),
            grip_position=int(gripper_config["grip_position"]),
            reset_position=int(gripper_config["reset_position"]),
            command_timeout_s=int(gripper_config["command_timeout_s"]),
            position_tolerance=int(gripper_config["position_tolerance"]),
            poll_interval_s=float(gripper_config["poll_interval_s"]),
            state_read_attempts=int(gripper_config["state_read_attempts"]),
            state_read_retry_s=float(gripper_config["state_read_retry_s"]),
            power_on_wait_s=float(gripper_config["power_on_wait_s"]),
            protocol_startup_wait_s=float(
                gripper_config["protocol_startup_wait_s"]
            ),
        )
    elif mode == "fake":
        arm = FakeArm(program.home)
        gripper = FakeGripper()
    else:
        raise ConfigError("runtime.mode must be fake or real")

    executor = MotionExecutor(
        arm,
        workspace_min_mm=motion_config["workspace_min_mm"],
        workspace_max_mm=motion_config["workspace_max_mm"],
        maximum_single_move_mm=float(motion_config["maximum_single_move_mm"]),
        reached_check_settle_s=float(
            motion_config.get("reached_check_settle_s", 1.0)
        ),
        position_reached_tolerance_mm=float(
            motion_config["position_reached_tolerance_mm"]
        ),
        orientation_reached_tolerance_deg=float(
            motion_config["orientation_reached_tolerance_deg"]
        ),
    )
    workflow = FixedTransferWorkflow(
        arm=arm,
        gripper=gripper,
        executor=executor,
        program=program,
    )
    return FixedTransferRuntime(
        mode=mode,
        arm=arm,
        gripper=gripper,
        executor=executor,
        workflow=workflow,
        program=program,
        motion_parameters_confirmed=bool(
            motion_config.get("parameters_confirmed", False)
        ),
    )


def format_fixed_transfer_program(program: FixedTransferProgram) -> str:
    lines = [
        f"固定取放程序：{'已确认' if program.confirmed else '未确认（禁止真机执行）'}",
        "假定起点：home",
    ]
    action_after = {
        "pick_approach": "夹紧试管",
        "carry_and_place": "释放试管",
        "retreat_and_home": "完成",
    }
    for phase_name, plan in program.plans():
        lines.append(f"[{phase_name}]")
        for waypoint in plan.waypoints:
            pose = waypoint.pose
            mode = "movel" if waypoint.linear else "movej_p"
            lines.append(
                f"  {waypoint.name}: {mode} "
                f"xyz=({pose.x_mm:.1f}, {pose.y_mm:.1f}, {pose.z_mm:.1f}) mm "
                f"rpy=({pose.rx_rad:.4f}, {pose.ry_rad:.4f}, "
                f"{pose.rz_rad:.4f}) rad speed={waypoint.speed_percent}%"
            )
        lines.append(f"  -> {action_after[phase_name]}")
    return "\n".join(lines)


def _pose(data: Mapping[str, Any], name: str) -> Pose6D:
    position = _vector(data.get("position_mm"), f"poses.{name}.position_mm", 3)
    rpy = _vector(data.get("rpy_rad"), f"poses.{name}.rpy_rad", 3)
    return Pose6D(*position, *rpy, frame="base_right")


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{name} must be a mapping")
    return value


def _vector(value: object, name: str, length: int) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ConfigError(f"{name} must contain {length} numbers")
    if len(value) != length:
        raise ConfigError(f"{name} must contain {length} numbers")
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError) as error:
        raise ConfigError(f"{name} must contain {length} finite numbers") from error
    if not all(math.isfinite(item) for item in result):
        raise ConfigError(f"{name} must contain {length} finite numbers")
    return result


def _speed(value: int, name: str) -> int:
    result = int(value)
    if not 1 <= result <= 100:
        raise ConfigError(f"{name} must be between 1 and 100")
    return result


def _positive(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ConfigError(f"{name} must be a positive finite number")
    return result


def _position_distance_mm(first: Pose6D, second: Pose6D) -> float:
    return math.sqrt(
        (first.x_mm - second.x_mm) ** 2
        + (first.y_mm - second.y_mm) ** 2
        + (first.z_mm - second.z_mm) ** 2
    )
