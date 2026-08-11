"""Small immutable data objects shared by all modules.

The project uses millimetres for positions and radians for angles.
RealMan SDK conversion to metres happens only inside the arm driver.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite


def _finite(*values: float) -> bool:
    return all(isfinite(float(value)) for value in values)


@dataclass(frozen=True)
class Pixel:
    u: float
    v: float

    def __post_init__(self) -> None:
        if not _finite(self.u, self.v):
            raise ValueError("pixel coordinates must be finite")


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if not _finite(self.x1, self.y1, self.x2, self.y2):
            raise ValueError("box coordinates must be finite")
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("box must have positive width and height")

    @property
    def center(self) -> Pixel:
        return Pixel((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    box: Box

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("detection label cannot be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("detection confidence must be between 0 and 1")


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    distortion_model: str = "none"
    distortion_coefficients: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not _finite(
            self.fx,
            self.fy,
            self.cx,
            self.cy,
            *self.distortion_coefficients,
        ):
            raise ValueError("camera intrinsics must be finite")
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("camera focal lengths must be positive")
        if self.distortion_model not in {"none", "brown_conrady"}:
            raise ValueError(
                "camera distortion model must be none or brown_conrady"
            )
        if (
            self.distortion_model == "brown_conrady"
            and len(self.distortion_coefficients) not in {4, 5, 8, 12, 14}
        ):
            raise ValueError(
                "brown_conrady distortion requires OpenCV-compatible coefficients"
            )


@dataclass(frozen=True)
class CameraFrame:
    color: object
    depth_mm: object
    intrinsics: CameraIntrinsics
    timestamp_ms: float
    frame_number: int


@dataclass(frozen=True)
class Point3D:
    x_mm: float
    y_mm: float
    z_mm: float
    frame: str = "base_right"

    def __post_init__(self) -> None:
        if not _finite(self.x_mm, self.y_mm, self.z_mm):
            raise ValueError("point coordinates must be finite")
        if not self.frame:
            raise ValueError("point frame cannot be empty")

    def shifted(self, *, dx_mm: float = 0.0, dy_mm: float = 0.0, dz_mm: float = 0.0) -> "Point3D":
        return Point3D(
            self.x_mm + dx_mm,
            self.y_mm + dy_mm,
            self.z_mm + dz_mm,
            self.frame,
        )


@dataclass(frozen=True)
class Pose6D:
    x_mm: float
    y_mm: float
    z_mm: float
    rx_rad: float
    ry_rad: float
    rz_rad: float
    frame: str = "base_right"

    def __post_init__(self) -> None:
        if not _finite(
            self.x_mm,
            self.y_mm,
            self.z_mm,
            self.rx_rad,
            self.ry_rad,
            self.rz_rad,
        ):
            raise ValueError("pose values must be finite")
        if not self.frame:
            raise ValueError("pose frame cannot be empty")

    @property
    def point(self) -> Point3D:
        return Point3D(self.x_mm, self.y_mm, self.z_mm, self.frame)


@dataclass(frozen=True)
class SlotAddress:
    rack_id: str
    row: int
    column: int

    def __post_init__(self) -> None:
        if not self.rack_id:
            raise ValueError("rack_id cannot be empty")
        if self.row not in (1, 2):
            raise ValueError("row must be 1 or 2")
        if not 1 <= self.column <= 6:
            raise ValueError("column must be between 1 and 6")

    @property
    def text(self) -> str:
        return f"{self.rack_id}.r{self.row}c{self.column}"


@dataclass(frozen=True)
class TransferCommand:
    source: SlotAddress
    destination: SlotAddress

    def __post_init__(self) -> None:
        if self.source == self.destination:
            raise ValueError("source and destination cannot be the same slot")


class Occupancy(Enum):
    EMPTY = "empty"
    OCCUPIED = "occupied"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class SlotObservation:
    address: SlotAddress
    occupancy: Occupancy
    confidence: float
    pixel: Pixel
    cap_top_base: Point3D | None = None
    hole_on_plane_base: Point3D | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("slot confidence must be between 0 and 1")


@dataclass(frozen=True)
class RackObservation:
    rack_id: str
    marker: Pixel
    plane_z_mm: float
    slots: tuple[SlotObservation, ...]
    timestamp_ms: float
    rack_keypoints: tuple[Pixel, ...] = ()
    stability_frame_count: int = 1
    maximum_keypoint_spread_px: float = 0.0
    maximum_position_spread_mm: float = 0.0

    def __post_init__(self) -> None:
        if not self.rack_id:
            raise ValueError("rack_id cannot be empty")
        if not _finite(self.plane_z_mm, self.timestamp_ms):
            raise ValueError("rack observation values must be finite")
        if self.stability_frame_count <= 0:
            raise ValueError("stability_frame_count must be positive")
        if not _finite(
            self.maximum_keypoint_spread_px,
            self.maximum_position_spread_mm,
        ):
            raise ValueError("rack stability values must be finite")
        if (
            self.maximum_keypoint_spread_px < 0.0
            or self.maximum_position_spread_mm < 0.0
        ):
            raise ValueError("rack stability values cannot be negative")
        if self.rack_keypoints and len(self.rack_keypoints) != 8:
            raise ValueError("rack_keypoints must be empty or contain eight points")
        if len(self.slots) != 12:
            raise ValueError("a 2x6 rack observation must contain 12 slots")
        addresses = [slot.address for slot in self.slots]
        if len(addresses) != len(set(addresses)):
            raise ValueError("rack observation contains duplicate slot addresses")
        expected = {
            SlotAddress(self.rack_id, row, column)
            for row in (1, 2)
            for column in range(1, 7)
        }
        if set(addresses) != expected:
            raise ValueError("rack observation must contain every slot in this rack")

    def slot(self, address: SlotAddress) -> SlotObservation:
        for observation in self.slots:
            if observation.address == address:
                return observation
        raise KeyError(address.text)


@dataclass(frozen=True)
class Waypoint:
    name: str
    pose: Pose6D
    speed_percent: int
    linear: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("waypoint name cannot be empty")
        if not 1 <= self.speed_percent <= 100:
            raise ValueError("speed_percent must be between 1 and 100")


@dataclass(frozen=True)
class MotionPlan:
    waypoints: tuple[Waypoint, ...]

    def __post_init__(self) -> None:
        if not self.waypoints:
            raise ValueError("motion plan cannot be empty")
