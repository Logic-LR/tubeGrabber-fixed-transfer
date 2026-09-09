"""Manual slot-grid and presentation-corner calibration for a rack."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import yaml

from tube_grabber.core.errors import ConfigError, VisionError
from tube_grabber.core.models import Pixel, SlotAddress
from tube_grabber.vision.rack_pose import (
    RACK_KEYPOINT_NAMES,
    RackPoseDetection,
    corner_homography,
    transform_pixels,
)


SCHEMA_VERSION = 4
LANDMARK_SOURCE = "screw_centers_v1"
DEFAULT_DISPLAY_CORNERS_UNIT = (
    Pixel(0.0, 0.0),
    Pixel(1.0, 0.0),
    Pixel(1.0, 1.0),
    Pixel(0.0, 1.0),
)


@dataclass(frozen=True)
class RackCircleFitConfig:
    search_radius_px: int
    minimum_radius_px: int
    maximum_radius_px: int
    default_radius_px: int
    hough_dp: float
    hough_min_distance_px: float
    hough_edge_threshold: float
    hough_accumulator_threshold: float

    def __post_init__(self) -> None:
        if self.search_radius_px <= 0:
            raise ValueError("circle search radius must be positive")
        if not 0 < self.minimum_radius_px < self.maximum_radius_px:
            raise ValueError("circle radius range is invalid")
        if self.search_radius_px <= self.maximum_radius_px:
            raise ValueError(
                "circle search radius must exceed the maximum circle radius"
            )
        if not (
            self.minimum_radius_px
            <= self.default_radius_px
            <= self.maximum_radius_px
        ):
            raise ValueError(
                "default circle radius must lie inside the radius range"
            )
        for name in (
            "hough_dp",
            "hough_min_distance_px",
            "hough_edge_threshold",
            "hough_accumulator_threshold",
        ):
            value = float(getattr(self, name))
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be a finite positive number")


@dataclass(frozen=True)
class SlotCircle:
    center: Pixel
    radius_px: float
    automatically_fitted: bool = True

    def __post_init__(self) -> None:
        if not isfinite(float(self.radius_px)) or self.radius_px <= 0.0:
            raise ValueError("slot circle radius must be finite and positive")


@dataclass(frozen=True)
class RackSlotCalibration:
    rack_id: str
    rows: int
    columns: int
    first_slot_unit: Pixel
    last_slot_unit: Pixel
    display_corners_unit: tuple[Pixel, Pixel, Pixel, Pixel] = (
        DEFAULT_DISPLAY_CORNERS_UNIT
    )
    display_corners_calibrated: bool = False
    keypoint_names: tuple[str, ...] = RACK_KEYPOINT_NAMES
    calibrated_at_utc: str = ""

    def __post_init__(self) -> None:
        if not self.rack_id:
            raise ValueError("rack_id cannot be empty")
        if self.rows != 2 or self.columns != 6:
            raise ValueError("the final runtime requires a 2x6 rack")
        if self.keypoint_names != RACK_KEYPOINT_NAMES:
            raise ValueError("rack calibration keypoint contract is incompatible")
        first = self.first_slot_unit
        last = self.last_slot_unit
        if not (
            0.0 <= first.u <= 1.0
            and 0.0 <= first.v <= 1.0
            and 0.0 <= last.u <= 1.0
            and 0.0 <= last.v <= 1.0
        ):
            raise ValueError("calibrated slot anchors must lie inside the rack surface")
        if last.u <= first.u or last.v <= first.v:
            raise ValueError(
                "r2c6 must be in the rack +column and +row direction from r1c1"
            )
        if len(self.display_corners_unit) != 4:
            raise ValueError("display corner calibration must contain four points")
        display = np.asarray(
            [[point.u, point.v] for point in self.display_corners_unit],
            dtype=np.float32,
        )
        if not np.all(np.isfinite(display)):
            raise ValueError("display corner coordinates must be finite")
        if not cv2.isContourConvex(display):
            raise ValueError(
                "display corners must form an ordered convex quadrilateral"
            )
        if abs(float(cv2.contourArea(display, oriented=True))) < 1e-4:
            raise ValueError("display corner quadrilateral is too small")

    def slot_unit_points(self) -> tuple[tuple[SlotAddress, Pixel], ...]:
        result = []
        for row in range(self.rows):
            row_fraction = row / (self.rows - 1)
            v = self.first_slot_unit.v + row_fraction * (
                self.last_slot_unit.v - self.first_slot_unit.v
            )
            for column in range(self.columns):
                column_fraction = column / (self.columns - 1)
                u = self.first_slot_unit.u + column_fraction * (
                    self.last_slot_unit.u - self.first_slot_unit.u
                )
                result.append(
                    (SlotAddress(self.rack_id, row + 1, column + 1), Pixel(u, v))
                )
        return tuple(result)

    def project_slots(
        self,
        detection: RackPoseDetection,
    ) -> tuple[tuple[SlotAddress, Pixel], ...]:
        image_from_unit, _ = corner_homography(detection)
        indexed = self.slot_unit_points()
        pixels = transform_pixels([item[1] for item in indexed], image_from_unit)
        return tuple((item[0], pixel) for item, pixel in zip(indexed, pixels))

    def project_display_corners(
        self,
        detection: RackPoseDetection,
    ) -> tuple[Pixel, Pixel, Pixel, Pixel]:
        """Project the four manually taught presentation corners."""
        image_from_unit, _ = corner_homography(detection)
        points = transform_pixels(self.display_corners_unit, image_from_unit)
        return tuple(points)  # type: ignore[return-value]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "landmark_source": LANDMARK_SOURCE,
            "rack_id": self.rack_id,
            "grid": {"rows": self.rows, "columns": self.columns},
            "keypoint_names": list(self.keypoint_names),
            "anchors": {
                "first_slot": "r1c1",
                "last_slot": "r2c6",
                "first_slot_unit": [self.first_slot_unit.u, self.first_slot_unit.v],
                "last_slot_unit": [self.last_slot_unit.u, self.last_slot_unit.v],
            },
            "display_corners_unit": [
                [point.u, point.v] for point in self.display_corners_unit
            ],
            "display_corners_calibrated": self.display_corners_calibrated,
            "calibrated_at_utc": self.calibrated_at_utc,
        }


def calibrate_slot_grid(
    rack_id: str,
    detection: RackPoseDetection,
    first_slot_pixel: Pixel,
    last_slot_pixel: Pixel,
    *,
    rows: int = 2,
    columns: int = 6,
) -> RackSlotCalibration:
    """Convert fitted r1c1/r2c6 circle centers to normalized anchors."""
    _, unit_from_image = corner_homography(detection)
    first, last = transform_pixels(
        (first_slot_pixel, last_slot_pixel),
        unit_from_image,
    )
    margin = 0.02
    for name, point in (("r1c1", first), ("r2c6", last)):
        if not -margin <= point.u <= 1.0 + margin or not (
            -margin <= point.v <= 1.0 + margin
        ):
            raise VisionError(f"clicked {name} is outside the recognized rack surface")
    try:
        return RackSlotCalibration(
            rack_id=str(rack_id),
            rows=int(rows),
            columns=int(columns),
            first_slot_unit=Pixel(
                float(np.clip(first.u, 0.0, 1.0)),
                float(np.clip(first.v, 0.0, 1.0)),
            ),
            last_slot_unit=Pixel(
                float(np.clip(last.u, 0.0, 1.0)),
                float(np.clip(last.v, 0.0, 1.0)),
            ),
            calibrated_at_utc=datetime.now(timezone.utc).isoformat(),
        )
    except ValueError as exc:
        raise VisionError(f"invalid two-circle rack calibration: {exc}") from exc


def calibrate_display_corners(
    calibration: RackSlotCalibration,
    detection: RackPoseDetection,
    corner_pixels: Sequence[Pixel],
) -> RackSlotCalibration:
    """Keep slot anchors and teach four display points relative to screws."""
    if len(corner_pixels) != 4:
        raise VisionError("exactly four display corners are required")
    polygon = np.asarray(
        [[point.u, point.v] for point in corner_pixels],
        dtype=np.float32,
    )
    if not cv2.isContourConvex(polygon):
        raise VisionError(
            "K0-K3 display corners must form an ordered convex quadrilateral"
        )
    if abs(float(cv2.contourArea(polygon, oriented=True))) < 100.0:
        raise VisionError("display corner quadrilateral is too small")
    _, unit_from_image = corner_homography(detection)
    unit = transform_pixels(corner_pixels, unit_from_image)
    try:
        return replace(
            calibration,
            display_corners_unit=tuple(unit),  # type: ignore[arg-type]
            display_corners_calibrated=True,
            calibrated_at_utc=datetime.now(timezone.utc).isoformat(),
        )
    except ValueError as exc:
        raise VisionError(f"invalid four-corner calibration: {exc}") from exc


def load_rack_calibration(
    path: str | Path,
    expected_rack_id: str,
) -> RackSlotCalibration:
    calibration_path = Path(path)
    if not calibration_path.is_file():
        raise ConfigError(f"rack calibration does not exist: {calibration_path}")
    try:
        data = yaml.safe_load(calibration_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(
            f"cannot read rack calibration {calibration_path}: {exc}"
        ) from exc
    if not isinstance(data, Mapping):
        raise ConfigError("rack calibration root must be a mapping")
    try:
        if int(data.get("schema_version", 0)) != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
        if str(data.get("landmark_source", "")) != LANDMARK_SOURCE:
            raise ValueError(f"landmark_source must be {LANDMARK_SOURCE}")
        grid = _mapping(data.get("grid"), "grid")
        anchors = _mapping(data.get("anchors"), "anchors")
        first = _pixel(anchors.get("first_slot_unit"), "first_slot_unit")
        last = _pixel(anchors.get("last_slot_unit"), "last_slot_unit")
        display_data = data.get("display_corners_unit")
        display_corners = (
            DEFAULT_DISPLAY_CORNERS_UNIT
            if display_data is None
            else tuple(
                _pixel(value, f"display_corners_unit[{index}]")
                for index, value in enumerate(
                    _sequence(display_data, "display_corners_unit")
                )
            )
        )
        calibration = RackSlotCalibration(
            rack_id=str(data.get("rack_id", "")),
            rows=int(grid.get("rows", 0)),
            columns=int(grid.get("columns", 0)),
            first_slot_unit=first,
            last_slot_unit=last,
            display_corners_unit=display_corners,  # type: ignore[arg-type]
            display_corners_calibrated=bool(
                data.get("display_corners_calibrated", False)
            ),
            keypoint_names=tuple(
                str(value) for value in data.get("keypoint_names", ())
            ),
            calibrated_at_utc=str(data.get("calibrated_at_utc", "")),
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            f"invalid rack calibration {calibration_path}: {exc}"
        ) from exc
    if calibration.rack_id != expected_rack_id:
        raise ConfigError(
            f"rack calibration is for {calibration.rack_id}, "
            f"expected {expected_rack_id}"
        )
    return calibration


def save_rack_calibration(
    calibration: RackSlotCalibration,
    path: str | Path,
    *,
    force: bool = False,
) -> Path:
    """Atomically save a calibration; overwrite only with explicit force."""
    output = Path(path)
    if output.exists() and not force:
        raise ConfigError(f"rack calibration already exists: {output}; use --force")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    try:
        temporary.write_text(
            yaml.safe_dump(calibration.as_dict(), allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        temporary.replace(output)
    except OSError as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise ConfigError(f"cannot save rack calibration {output}: {exc}") from exc
    return output


def fit_slot_circle(
    color_image: object,
    seed: Pixel,
    config: RackCircleFitConfig,
) -> SlotCircle:
    """Fit the circular slot nearest an operator-provided approximate point."""
    image = np.asarray(color_image)
    if image.ndim != 3 or image.shape[2] != 3:
        raise VisionError("slot-circle input must be a BGR image")
    height, width = image.shape[:2]
    center_u = int(round(seed.u))
    center_v = int(round(seed.v))
    search = config.search_radius_px
    left = max(0, center_u - search)
    right = min(width, center_u + search + 1)
    top = max(0, center_v - search)
    bottom = min(height, center_v + search + 1)
    if right - left < 2 * config.minimum_radius_px or (
        bottom - top < 2 * config.minimum_radius_px
    ):
        raise VisionError("circle search region is too close to the image edge")

    gray = cv2.cvtColor(image[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2.0)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=float(config.hough_dp),
        minDist=float(config.hough_min_distance_px),
        param1=float(config.hough_edge_threshold),
        param2=float(config.hough_accumulator_threshold),
        minRadius=int(config.minimum_radius_px),
        maxRadius=int(config.maximum_radius_px),
    )
    if circles is None or not len(circles[0]):
        raise VisionError("no circular slot was found near the selected point")
    candidates = np.asarray(circles[0], dtype=np.float64)
    candidates[:, 0] += left
    candidates[:, 1] += top
    distances = np.hypot(
        candidates[:, 0] - seed.u,
        candidates[:, 1] - seed.v,
    )
    radius_error = np.abs(candidates[:, 2] - config.default_radius_px)
    index = int(np.argmin(distances + 0.15 * radius_error))
    circle = candidates[index]
    if float(distances[index]) > float(search):
        raise VisionError("the nearest fitted circle is outside the search radius")
    return SlotCircle(
        center=Pixel(float(circle[0]), float(circle[1])),
        radius_px=float(circle[2]),
        automatically_fitted=True,
    )


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _pixel(value: object, name: str) -> Pixel:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
    ):
        raise ValueError(f"{name} must contain two numbers")
    return Pixel(float(value[0]), float(value[1]))


def _sequence(value: object, name: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence")
    return value
