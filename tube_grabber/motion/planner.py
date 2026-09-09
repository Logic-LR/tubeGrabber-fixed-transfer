"""Pure coordinate planning along the measured rack-surface normal."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from tube_grabber.core.errors import MotionError
from tube_grabber.core.models import MotionPlan, Point3D, Pose6D, Waypoint


class MotionPlanner:
    """Convert TCP targets into guarded rack-normal flange-motion plans."""

    def __init__(
        self,
        *,
        tcp_offset_end_mm: Sequence[float],
        approach_height_mm: float,
        retreat_height_mm: float,
        transit_speed_percent: int,
        approach_speed_percent: int,
        maximum_single_orientation_change_deg: float,
        maximum_tool_axis_misalignment_deg: float,
        tube_total_length_mm: float,
        required_carried_clearance_mm: float,
        destination_tcp_offset_end_mm: Sequence[float] | None = None,
    ) -> None:
        self.tcp_offset_end_mm = _vector3(tcp_offset_end_mm, "tcp_offset_end_mm")
        self.destination_tcp_offset_end_mm = _vector3(
            (
                tcp_offset_end_mm
                if destination_tcp_offset_end_mm is None
                else destination_tcp_offset_end_mm
            ),
            "destination_tcp_offset_end_mm",
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
        self.maximum_single_orientation_change_deg = _range(
            maximum_single_orientation_change_deg,
            "maximum_single_orientation_change_deg",
            0.0,
            180.0,
        )
        self.maximum_tool_axis_misalignment_deg = _range(
            maximum_tool_axis_misalignment_deg,
            "maximum_tool_axis_misalignment_deg",
            0.0,
            90.0,
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

    def plan_approach(
        self,
        current_flange: Pose6D,
        target_tcp: Point3D,
        approach_axis_base: Sequence[float],
        *,
        destination: bool = False,
    ) -> MotionPlan:
        """Lift, traverse and descend along the measured rack normal."""
        corridor = self.plan_above_target(
            current_flange,
            target_tcp,
            approach_axis_base,
            destination=destination,
        )
        aligned = corridor.waypoints[-1].pose
        return MotionPlan(
            corridor.waypoints
            + (
                Waypoint(
                    "descend",
                    self._flange_pose(
                        target_tcp,
                        (aligned.rx_rad, aligned.ry_rad, aligned.rz_rad),
                        destination=destination,
                    ),
                    self.approach_speed_percent,
                    linear=True,
                ),
            )
        )

    def plan_above_target(
        self,
        current_flange: Pose6D,
        target_tcp: Point3D,
        approach_axis_base: Sequence[float],
        *,
        destination: bool = False,
    ) -> MotionPlan:
        """Reach the safe rack-normal corridor without descending."""
        _require_same_frame(current_flange.frame, target_tcp.frame)
        axis = np.asarray(
            _unit_vector(approach_axis_base, "approach_axis_base"),
            dtype=np.float64,
        )
        aligned_rpy, orientation_error_deg = self._aligned_tool_rpy(
            current_flange,
            axis,
        )
        tcp_offset = self._tcp_offset(destination)
        current_tcp = flange_to_tcp_point(
            current_flange,
            tcp_offset,
        )
        current_array = _point_array(current_tcp)
        target_array = _point_array(target_tcp)
        safe_coordinate = max(
            float(np.dot(current_array, axis)),
            float(np.dot(target_array, axis)) + self.approach_height_mm,
        )
        lift_tcp = _point_from_array(
            current_array
            + (safe_coordinate - float(np.dot(current_array, axis))) * axis,
            target_tcp.frame,
        )
        above_tcp = _point_from_array(
            target_array
            + (safe_coordinate - float(np.dot(target_array, axis))) * axis,
            target_tcp.frame,
        )
        waypoints: list[Waypoint] = []
        if _point_distance(lift_tcp, current_tcp) > 0.5:
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
                        tcp_offset,
                    ),
                    self.transit_speed_percent,
                    linear=True,
                )
            )
        if orientation_error_deg > 0.1:
            waypoints.append(
                Waypoint(
                    "align_tool_axis",
                    self._flange_pose(
                        lift_tcp,
                        aligned_rpy,
                        destination=destination,
                    ),
                    self.approach_speed_percent,
                    linear=True,
                )
            )
        waypoints.append(
            Waypoint(
                "above_target",
                self._flange_pose(
                    above_tcp,
                    aligned_rpy,
                    destination=destination,
                ),
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

    def plan_retreat(
        self,
        current_flange: Pose6D,
        approach_axis_base: Sequence[float],
        *,
        destination: bool = False,
    ) -> MotionPlan:
        """Move the TCP away from the rack after gripping or releasing."""
        axis = _unit_vector(approach_axis_base, "approach_axis_base")
        self._require_tool_axis_near(current_flange, axis)
        tcp_offset = self._tcp_offset(destination)
        current_tcp = flange_to_tcp_point(
            current_flange,
            tcp_offset,
        )
        retreat_tcp = _shift_along(current_tcp, axis, self.retreat_height_mm)
        return MotionPlan(
            (
                Waypoint(
                    "retreat",
                    self._flange_pose(
                        retreat_tcp,
                        (
                            current_flange.rx_rad,
                            current_flange.ry_rad,
                            current_flange.rz_rad,
                        ),
                        destination=destination,
                    ),
                    self.approach_speed_percent,
                    linear=True,
                ),
            )
        )

    def _flange_pose(
        self,
        tcp: Point3D,
        rpy_rad: Sequence[float],
        *,
        destination: bool = False,
    ) -> Pose6D:
        return tcp_to_flange_pose(
            tcp,
            rpy_rad,
            self._tcp_offset(destination),
        )

    def _tcp_offset(self, destination: bool) -> tuple[float, float, float]:
        return (
            self.destination_tcp_offset_end_mm
            if destination
            else self.tcp_offset_end_mm
        )

    def _require_tool_axis_near(
        self,
        pose: Pose6D,
        approach_axis_base: Sequence[float],
    ) -> float:
        rotation = np.asarray(
            rpy_to_rotation((pose.rx_rad, pose.ry_rad, pose.rz_rad)),
            dtype=np.float64,
        )
        away = -rotation[:, 2]
        axis = np.asarray(
            _unit_vector(approach_axis_base, "approach_axis_base"),
            dtype=np.float64,
        )
        error_deg = _angle_deg(away, axis)
        if error_deg > self.maximum_tool_axis_misalignment_deg:
            raise MotionError(
                "current tool axis differs from the measured rack normal by "
                f"{error_deg:.2f} deg; maximum is "
                f"{self.maximum_tool_axis_misalignment_deg:.2f} deg"
            )
        return error_deg

    def _aligned_tool_rpy(
        self,
        pose: Pose6D,
        approach_axis_base: Sequence[float],
    ) -> tuple[tuple[float, float, float], float]:
        error_deg = self._require_tool_axis_near(pose, approach_axis_base)
        rotation = np.asarray(
            rpy_to_rotation((pose.rx_rad, pose.ry_rad, pose.rz_rad)),
            dtype=np.float64,
        )
        away = -rotation[:, 2]
        correction = _rotation_between_vectors(away, approach_axis_base)
        aligned = _orthonormal_rotation(correction @ rotation)
        return _rotation_matrix_to_rpy(aligned), error_deg


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


def _unit_vector(
    values: Sequence[float],
    name: str,
) -> tuple[float, float, float]:
    vector = np.asarray(_vector3(values, name), dtype=np.float64)
    length = float(np.linalg.norm(vector))
    if length <= 1e-9:
        raise MotionError(f"{name} cannot be zero")
    return tuple(float(value) for value in vector / length)


def _point_array(point: Point3D) -> np.ndarray:
    return np.asarray([point.x_mm, point.y_mm, point.z_mm], dtype=np.float64)


def _point_from_array(values: np.ndarray, frame: str) -> Point3D:
    return Point3D(*(float(value) for value in values), frame=frame)


def _point_distance(first: Point3D, second: Point3D) -> float:
    _require_same_frame(first.frame, second.frame)
    return float(np.linalg.norm(_point_array(first) - _point_array(second)))


def _shift_along(
    point: Point3D,
    axis: Sequence[float],
    distance_mm: float,
) -> Point3D:
    direction = np.asarray(_unit_vector(axis, "axis"), dtype=np.float64)
    return _point_from_array(
        _point_array(point) + float(distance_mm) * direction,
        point.frame,
    )


def _angle_deg(first: Sequence[float], second: Sequence[float]) -> float:
    first_unit = np.asarray(_unit_vector(first, "first_axis"))
    second_unit = np.asarray(_unit_vector(second, "second_axis"))
    cosine = float(np.clip(np.dot(first_unit, second_unit), -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _rotation_between_vectors(
    source: Sequence[float],
    target: Sequence[float],
) -> np.ndarray:
    """Return the shortest 3x3 rotation carrying source onto target."""
    source_unit = np.asarray(_unit_vector(source, "source_axis"))
    target_unit = np.asarray(_unit_vector(target, "target_axis"))
    cross = np.cross(source_unit, target_unit)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(source_unit, target_unit), -1.0, 1.0))
    if sine < 1e-12:
        if cosine > 0.0:
            return np.eye(3, dtype=np.float64)
        basis = np.zeros(3, dtype=np.float64)
        basis[int(np.argmin(np.abs(source_unit)))] = 1.0
        axis = np.asarray(_unit_vector(np.cross(source_unit, basis), "opposite_axis"))
        return 2.0 * np.outer(axis, axis) - np.eye(3, dtype=np.float64)
    axis = cross / sine
    skew = np.asarray(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=np.float64,
    )
    return (
        np.eye(3, dtype=np.float64)
        + sine * skew
        + (1.0 - cosine) * (skew @ skew)
    )


def _orthonormal_rotation(rotation: np.ndarray) -> np.ndarray:
    u_matrix, _, v_transpose = np.linalg.svd(rotation)
    result = u_matrix @ v_transpose
    if float(np.linalg.det(result)) < 0.0:
        u_matrix[:, -1] *= -1.0
        result = u_matrix @ v_transpose
    return result


def _rotation_matrix_to_rpy(
    rotation: Sequence[Sequence[float]],
) -> tuple[float, float, float]:
    """Convert a rotation matrix with the RealMan Rz @ Ry @ Rx convention."""
    matrix = _orthonormal_rotation(np.asarray(rotation, dtype=np.float64))
    ry = math.asin(float(np.clip(-matrix[2, 0], -1.0, 1.0)))
    if abs(math.cos(ry)) > 1e-8:
        rx = math.atan2(float(matrix[2, 1]), float(matrix[2, 2]))
        rz = math.atan2(float(matrix[1, 0]), float(matrix[0, 0]))
    else:
        rz = 0.0
        if matrix[2, 0] < 0.0:
            rx = math.atan2(float(matrix[0, 1]), float(matrix[0, 2]))
        else:
            rx = math.atan2(float(-matrix[0, 1]), float(-matrix[0, 2]))
    return (rx, ry, rz)


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
