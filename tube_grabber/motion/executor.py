"""The only module that sends planned poses to the arm."""

from __future__ import annotations

import math
from collections.abc import Sequence

from tube_grabber.core.errors import MotionError
from tube_grabber.core.models import MotionPlan, Pose6D
from tube_grabber.core.ports import ArmPort
from tube_grabber.motion.planner import rotation_distance_deg, rpy_to_rotation


_SKIP_POSITION_TOLERANCE_MM = 0.5
_SKIP_ORIENTATION_TOLERANCE_DEG = 0.1


class MotionExecutor:
    def __init__(
        self,
        arm: ArmPort,
        *,
        workspace_min_mm: Sequence[float],
        workspace_max_mm: Sequence[float],
        maximum_single_move_mm: float,
        position_reached_tolerance_mm: float,
        orientation_reached_tolerance_deg: float,
        frame: str = "base_right",
    ) -> None:
        self.arm = arm
        self.workspace_min_mm = _vector3(workspace_min_mm, "workspace_min_mm")
        self.workspace_max_mm = _vector3(workspace_max_mm, "workspace_max_mm")
        if any(
            minimum >= maximum
            for minimum, maximum in zip(
                self.workspace_min_mm,
                self.workspace_max_mm,
            )
        ):
            raise MotionError("workspace minimum must be below workspace maximum")
        self.maximum_single_move_mm = float(maximum_single_move_mm)
        if (
            not math.isfinite(self.maximum_single_move_mm)
            or self.maximum_single_move_mm <= 0
        ):
            raise MotionError("maximum_single_move_mm must be positive")
        self.position_reached_tolerance_mm = _positive_finite(
            position_reached_tolerance_mm,
            "position_reached_tolerance_mm",
        )
        self.orientation_reached_tolerance_deg = _positive_finite(
            orientation_reached_tolerance_deg,
            "orientation_reached_tolerance_deg",
        )
        if not frame:
            raise MotionError("executor frame cannot be empty")
        self.frame = frame

    def validate(
        self,
        plan: MotionPlan,
        start_pose: Pose6D | None = None,
    ) -> Pose6D:
        """Validate every waypoint before motion and return the final pose."""
        previous = start_pose if start_pose is not None else self.arm.get_pose()
        self._require_frame(previous)
        for waypoint in plan.waypoints:
            target = waypoint.pose
            self._require_frame(target)
            self._require_workspace(target)
            distance_mm = _distance(previous, target)
            if distance_mm > self.maximum_single_move_mm:
                raise MotionError(
                    f"{waypoint.name} move {distance_mm:.1f} mm exceeds "
                    f"{self.maximum_single_move_mm:.1f} mm"
                )
            previous = target
        return previous

    def execute(self, plan: MotionPlan) -> None:
        try:
            current = self.arm.get_pose()
            self.validate(plan, current)
            for waypoint in plan.waypoints:
                if _same_pose(current, waypoint.pose):
                    continue
                self.arm.move_pose(
                    waypoint.pose,
                    waypoint.speed_percent,
                    linear=waypoint.linear,
                )
                reached = self.arm.get_pose()
                position_error_mm = _distance(reached, waypoint.pose)
                orientation_error_deg = _orientation_distance_deg(
                    reached,
                    waypoint.pose,
                )
                if (
                    position_error_mm > self.position_reached_tolerance_mm
                    or orientation_error_deg
                    > self.orientation_reached_tolerance_deg
                ):
                    raise MotionError(
                        f"{waypoint.name} did not reach its target: "
                        f"position error {position_error_mm:.2f} mm, "
                        f"orientation error {orientation_error_deg:.2f} deg"
                    )
                current = reached
        except Exception as error:
            try:
                self.arm.stop()
            except Exception:
                pass
            if isinstance(error, MotionError):
                raise
            raise MotionError(f"arm motion failed: {error}") from error

    def _require_frame(self, pose: Pose6D) -> None:
        if pose.frame != self.frame:
            raise MotionError(
                f"arm pose frame must be {self.frame}, got {pose.frame}"
            )

    def _require_workspace(self, pose: Pose6D) -> None:
        for axis, value, minimum, maximum in zip(
            "XYZ",
            (pose.x_mm, pose.y_mm, pose.z_mm),
            self.workspace_min_mm,
            self.workspace_max_mm,
        ):
            if not minimum <= value <= maximum:
                raise MotionError(
                    f"target {axis}={value:.1f} mm outside "
                    f"[{minimum:.1f}, {maximum:.1f}] mm"
                )


def _distance(first: Pose6D, second: Pose6D) -> float:
    return math.sqrt(
        (first.x_mm - second.x_mm) ** 2
        + (first.y_mm - second.y_mm) ** 2
        + (first.z_mm - second.z_mm) ** 2
    )


def _same_pose(first: Pose6D, second: Pose6D) -> bool:
    if _distance(first, second) > _SKIP_POSITION_TOLERANCE_MM:
        return False
    angle = _orientation_distance_deg(first, second)
    return angle <= _SKIP_ORIENTATION_TOLERANCE_DEG


def _orientation_distance_deg(first: Pose6D, second: Pose6D) -> float:
    return rotation_distance_deg(
        rpy_to_rotation((first.rx_rad, first.ry_rad, first.rz_rad)),
        rpy_to_rotation((second.rx_rad, second.ry_rad, second.rz_rad)),
    )


def _vector3(values: Sequence[float], name: str) -> tuple[float, float, float]:
    if len(values) != 3:
        raise MotionError(f"{name} must contain three values")
    result = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in result):
        raise MotionError(f"{name} must contain finite values")
    return result


def _positive_finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise MotionError(f"{name} must be positive")
    return result
