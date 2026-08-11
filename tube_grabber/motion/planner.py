"""Pure coordinate planning for vertical pick-and-place motion."""

from __future__ import annotations

import math
from collections.abc import Sequence

from tube_grabber.core.errors import MotionError
from tube_grabber.core.models import MotionPlan, Point3D, Pose6D, Waypoint


class MotionPlanner:
    """Convert TCP targets into short, vertical flange-motion plans."""

    def __init__(
        self,
        *,
        tcp_offset_end_mm: Sequence[float],
        vertical_tool_rpy_rad: Sequence[float],
        approach_height_mm: float,
        retreat_height_mm: float,
        transit_speed_percent: int,
        approach_speed_percent: int,
        maximum_orientation_error_deg: float,
        maximum_single_orientation_change_deg: float,
        maximum_tool_tilt_deg: float,
        tube_total_length_mm: float,
        required_carried_clearance_mm: float,
    ) -> None:
        self.tcp_offset_end_mm = _vector3(tcp_offset_end_mm, "tcp_offset_end_mm")
        self.vertical_tool_rpy_rad = _vector3(
            vertical_tool_rpy_rad,
            "vertical_tool_rpy_rad",
        )
        self.approach_height_mm = _positive(
            approach_height_mm,
            "approach_height_mm",
        )
        self.retreat_height_mm = _positive(
            retreat_height_mm,
            "retreat_height_mm",
        )
        self.transit_speed_percent = _speed(
            transit_speed_percent,
            "transit_speed_percent",
        )
        self.approach_speed_percent = _speed(
            approach_speed_percent,
            "approach_speed_percent",
        )
        self.maximum_orientation_error_deg = _range(
            maximum_orientation_error_deg,
            "maximum_orientation_error_deg",
            0.0,
            180.0,
        )
        self.maximum_single_orientation_change_deg = _range(
            maximum_single_orientation_change_deg,
            "maximum_single_orientation_change_deg",
            0.0,
            180.0,
        )
        self.maximum_tool_tilt_deg = _range(
            maximum_tool_tilt_deg,
            "maximum_tool_tilt_deg",
            0.0,
            180.0,
        )
        self.tube_total_length_mm = _positive(
            tube_total_length_mm,
            "tube_total_length_mm",
        )
        self.required_carried_clearance_mm = _range(
            required_carried_clearance_mm,
            "required_carried_clearance_mm",
            0.0,
            10_000.0,
        )
        minimum_retreat_mm = (
            self.tube_total_length_mm + self.required_carried_clearance_mm
        )
        if self.retreat_height_mm < minimum_retreat_mm:
            raise MotionError(
                f"retreat_height_mm={self.retreat_height_mm:.1f} cannot clear "
                f"a {self.tube_total_length_mm:.1f} mm tube with "
                f"{self.required_carried_clearance_mm:.1f} mm clearance; "
                f"minimum is {minimum_retreat_mm:.1f} mm"
            )
        self._require_configured_tool_points_down()

    def plan_approach(
        self,
        current_flange: Pose6D,
        target_tcp: Point3D,
    ) -> MotionPlan:
        """Lift safely, move above the target, then descend vertically."""
        corridor = self.plan_above_target(current_flange, target_tcp)
        return MotionPlan(
            corridor.waypoints
            + (
                Waypoint(
                    "descend",
                    self._flange_pose(target_tcp),
                    self.approach_speed_percent,
                    linear=True,
                ),
            )
        )

    def plan_above_target(
        self,
        current_flange: Pose6D,
        target_tcp: Point3D,
    ) -> MotionPlan:
        """Reach the safe corridor above a target without descending."""
        _require_same_frame(current_flange.frame, target_tcp.frame)
        orientation_error_deg = self._require_vertical_orientation(
            current_flange
        )
        current_tcp = flange_to_tcp_point(
            current_flange,
            self.tcp_offset_end_mm,
        )
        safe_z_mm = max(
            current_tcp.z_mm,
            target_tcp.z_mm + self.approach_height_mm,
        )
        lift_tcp = Point3D(
            current_tcp.x_mm,
            current_tcp.y_mm,
            safe_z_mm,
            target_tcp.frame,
        )
        above_tcp = Point3D(
            target_tcp.x_mm,
            target_tcp.y_mm,
            safe_z_mm,
            target_tcp.frame,
        )
        waypoints: list[Waypoint] = []
        if abs(safe_z_mm - current_tcp.z_mm) > 0.5:
            waypoints.append(
                Waypoint(
                    "lift",
                    tcp_to_flange_pose(
                        lift_tcp,
                        (
                            current_flange.rx_rad,
                            current_flange.ry_rad,
                            current_flange.rz_rad,
                        ),
                        self.tcp_offset_end_mm,
                    ),
                    self.transit_speed_percent,
                    linear=True,
                )
            )
        if orientation_error_deg > 0.1:
            waypoints.append(
                Waypoint(
                    "align_vertical",
                    self._flange_pose(lift_tcp),
                    self.approach_speed_percent,
                    linear=True,
                )
            )
        waypoints.append(
            Waypoint(
                "above_target",
                self._flange_pose(above_tcp),
                self.transit_speed_percent,
                linear=True,
            )
        )
        return MotionPlan(tuple(waypoints))

    def plan_pose_move(
        self,
        current_flange: Pose6D,
        target_flange: Pose6D,
        *,
        name: str,
    ) -> MotionPlan:
        """Plan one guarded joint-space move to a taught flange pose."""
        _require_same_frame(current_flange.frame, target_flange.frame)
        change_deg = rotation_distance_deg(
            rpy_to_rotation(
                (
                    current_flange.rx_rad,
                    current_flange.ry_rad,
                    current_flange.rz_rad,
                )
            ),
            rpy_to_rotation(
                (
                    target_flange.rx_rad,
                    target_flange.ry_rad,
                    target_flange.rz_rad,
                )
            ),
        )
        if change_deg > self.maximum_single_orientation_change_deg:
            raise MotionError(
                f"{name} orientation change {change_deg:.2f} deg exceeds "
                f"{self.maximum_single_orientation_change_deg:.2f} deg"
            )
        return MotionPlan(
            (
                Waypoint(
                    name,
                    target_flange,
                    self.transit_speed_percent,
                    linear=False,
                ),
            )
        )

    def plan_retreat(self, current_flange: Pose6D) -> MotionPlan:
        """Move the TCP straight upward after gripping or releasing."""
        self._require_vertical_orientation(current_flange)
        current_tcp = flange_to_tcp_point(
            current_flange,
            self.tcp_offset_end_mm,
        )
        retreat_tcp = current_tcp.shifted(dz_mm=self.retreat_height_mm)
        return MotionPlan(
            (
                Waypoint(
                    "retreat",
                    self._flange_pose(retreat_tcp),
                    self.approach_speed_percent,
                    linear=True,
                ),
            )
        )

    def _flange_pose(self, tcp: Point3D) -> Pose6D:
        return tcp_to_flange_pose(
            tcp,
            self.vertical_tool_rpy_rad,
            self.tcp_offset_end_mm,
        )

    def _require_vertical_orientation(self, pose: Pose6D) -> float:
        current = rpy_to_rotation((pose.rx_rad, pose.ry_rad, pose.rz_rad))
        expected = rpy_to_rotation(self.vertical_tool_rpy_rad)
        error_deg = rotation_distance_deg(current, expected)
        if error_deg > self.maximum_orientation_error_deg:
            raise MotionError(
                f"current tool orientation differs from vertical pose by "
                f"{error_deg:.2f} deg; maximum is "
                f"{self.maximum_orientation_error_deg:.2f} deg"
            )
        return error_deg

    def _require_configured_tool_points_down(self) -> None:
        rotation = rpy_to_rotation(self.vertical_tool_rpy_rad)
        tool_z = (rotation[0][2], rotation[1][2], rotation[2][2])
        cosine = max(-1.0, min(1.0, -tool_z[2]))
        tilt_deg = math.degrees(math.acos(cosine))
        if tilt_deg > self.maximum_tool_tilt_deg:
            raise MotionError(
                "configured vertical_tool_rpy_rad does not point tool +Z "
                f"toward base -Z: tilt is {tilt_deg:.2f} deg; maximum is "
                f"{self.maximum_tool_tilt_deg:.2f} deg"
            )


def flange_to_tcp_point(
    flange: Pose6D,
    tcp_offset_end_mm: Sequence[float],
) -> Point3D:
    """Return TCP position from a flange pose and end-frame TCP offset."""
    offset = _rotate(
        rpy_to_rotation((flange.rx_rad, flange.ry_rad, flange.rz_rad)),
        _vector3(tcp_offset_end_mm, "tcp_offset_end_mm"),
    )
    return Point3D(
        flange.x_mm + offset[0],
        flange.y_mm + offset[1],
        flange.z_mm + offset[2],
        flange.frame,
    )


def tcp_to_flange_pose(
    tcp: Point3D,
    flange_rpy_rad: Sequence[float],
    tcp_offset_end_mm: Sequence[float],
) -> Pose6D:
    """Return the flange pose that places the TCP at ``tcp``."""
    rpy = _vector3(flange_rpy_rad, "flange_rpy_rad")
    offset = _rotate(
        rpy_to_rotation(rpy),
        _vector3(tcp_offset_end_mm, "tcp_offset_end_mm"),
    )
    return Pose6D(
        tcp.x_mm - offset[0],
        tcp.y_mm - offset[1],
        tcp.z_mm - offset[2],
        rpy[0],
        rpy[1],
        rpy[2],
        tcp.frame,
    )


def rpy_to_rotation(
    rpy_rad: Sequence[float],
) -> tuple[tuple[float, float, float], ...]:
    """Create an Rz @ Ry @ Rx rotation matrix."""
    rx, ry, rz = _vector3(rpy_rad, "rpy_rad")
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    return (
        (cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx),
        (sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx),
        (-sy, cy * sx, cy * cx),
    )


def rotation_distance_deg(
    first: Sequence[Sequence[float]],
    second: Sequence[Sequence[float]],
) -> float:
    trace = sum(
        float(first[row][column]) * float(second[row][column])
        for row in range(3)
        for column in range(3)
    )
    cosine = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
    return math.degrees(math.acos(cosine))


def _rotate(
    rotation: Sequence[Sequence[float]],
    vector: Sequence[float],
) -> tuple[float, float, float]:
    return tuple(
        sum(float(rotation[row][column]) * float(vector[column]) for column in range(3))
        for row in range(3)
    )


def _vector3(values: Sequence[float], name: str) -> tuple[float, float, float]:
    if len(values) != 3:
        raise MotionError(f"{name} must contain three values")
    result = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in result):
        raise MotionError(f"{name} must contain finite values")
    return result


def _positive(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise MotionError(f"{name} must be positive")
    return result


def _range(value: float, name: str, minimum: float, maximum: float) -> float:
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise MotionError(f"{name} must be between {minimum} and {maximum}")
    return result


def _speed(value: int, name: str) -> int:
    result = int(value)
    if result != value or not 1 <= result <= 100:
        raise MotionError(f"{name} must be an integer between 1 and 100")
    return result


def _require_same_frame(first: str, second: str) -> None:
    if first != second:
        raise MotionError(f"coordinate frame mismatch: {first} != {second}")
