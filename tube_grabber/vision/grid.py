"""Map twelve detection centres to a 2-row by 6-column rack."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite
from typing import Sequence

import numpy as np

from tube_grabber.core.errors import VisionError
from tube_grabber.core.models import Detection, Pixel, SlotAddress


@dataclass(frozen=True)
class MappedDetection:
    address: SlotAddress
    detection: Detection


@dataclass(frozen=True)
class GridMapping:
    marker: Pixel
    slots: tuple[MappedDetection, ...]
    column_spacing_px: float
    row_spacing_px: float
    residual_px: float


class GridMapper:
    """Fit a small affine grid, then use K0 to choose row/column direction."""

    def __init__(
        self,
        maximum_spacing_cv: float,
        maximum_marker_corner_distance_factor: float,
    ) -> None:
        if not 0.0 < maximum_spacing_cv < 1.0:
            raise VisionError("maximum_spacing_cv must be between 0 and 1")
        if (
            not isfinite(float(maximum_marker_corner_distance_factor))
            or maximum_marker_corner_distance_factor <= 0.0
        ):
            raise VisionError(
                "maximum_marker_corner_distance_factor must be positive"
            )
        self._maximum_spacing_cv = float(maximum_spacing_cv)
        self._maximum_marker_corner_distance_factor = float(
            maximum_marker_corner_distance_factor
        )

    def map(
        self,
        detections: Sequence[Detection],
        marker_candidates: Sequence[Pixel],
        rack_id: str,
    ) -> GridMapping:
        if len(detections) != 12:
            raise VisionError(
                f"expected exactly 12 rack detections, got {len(detections)}"
            )
        if not marker_candidates:
            raise VisionError("K0 marker candidate list is empty")

        points = np.array(
            [[item.box.center.u, item.box.center.v] for item in detections],
            dtype=np.float64,
        )
        if not np.isfinite(points).all():
            raise VisionError("detection centres must be finite")
        _reject_duplicate_points(points)

        # A red or green tube cap can also pass the HSV threshold.  K0 is on
        # the rack frame, so a candidate inside any YOLO box cannot be K0.
        marker_candidates = tuple(
            marker
            for marker in marker_candidates
            if not _inside_any_detection(marker, detections)
        )
        if not marker_candidates:
            raise VisionError("all K0 marker candidates overlap rack detections")

        indexed_rows, grid_points = _initial_grid_indices(points)
        origin, column_step, row_step, residual_px = _fit_affine_grid(
            points,
            grid_points,
        )

        column_spacing_px = float(np.linalg.norm(column_step))
        row_spacing_px = float(np.linalg.norm(row_step))
        if column_spacing_px < 1.0 or row_spacing_px < 1.0:
            raise VisionError("rack spacing is too small")

        column_gaps, row_gaps = _measured_gaps(points, indexed_rows)
        _check_spacing("column", column_gaps, self._maximum_spacing_cv)
        _check_spacing("row", row_gaps, self._maximum_spacing_cv)

        spacing_scale = min(column_spacing_px, row_spacing_px)
        if residual_px / spacing_scale > self._maximum_spacing_cv:
            raise VisionError(
                "rack grid residual is too large: "
                f"{residual_px / spacing_scale:.3f}"
            )

        corners = {
            (0, 0): origin,
            (0, 5): origin + 5.0 * column_step,
            (1, 0): origin + row_step,
            (1, 5): origin + 5.0 * column_step + row_step,
        }
        maximum_marker_distance = (
            self._maximum_marker_corner_distance_factor
            * max(column_spacing_px, row_spacing_px)
        )
        marker, k0_corner = _select_unique_k0(
            marker_candidates,
            corners,
            origin,
            column_step,
            row_step,
            maximum_marker_distance,
        )

        k0_row, k0_column = k0_corner
        mapped: list[MappedDetection] = []
        for detection_index, raw_row, raw_column in grid_points:
            logical_row = raw_row if k0_row == 0 else 1 - raw_row
            logical_column = raw_column if k0_column == 0 else 5 - raw_column
            mapped.append(
                MappedDetection(
                    SlotAddress(rack_id, logical_row + 1, logical_column + 1),
                    detections[detection_index],
                )
            )

        mapped.sort(key=lambda item: (item.address.row, item.address.column))
        return GridMapping(
            marker=marker,
            slots=tuple(mapped),
            column_spacing_px=column_spacing_px,
            row_spacing_px=row_spacing_px,
            residual_px=residual_px,
        )


def _reject_duplicate_points(points: np.ndarray) -> None:
    differences = points[:, None, :] - points[None, :, :]
    distances = np.linalg.norm(differences, axis=2)
    distances += np.eye(len(points)) * 1e9
    if float(distances.min()) < 1.0:
        raise VisionError("duplicate detection centres found")


def _inside_any_detection(
    marker: Pixel,
    detections: Sequence[Detection],
) -> bool:
    return any(
        detection.box.x1 <= marker.u <= detection.box.x2
        and detection.box.y1 <= marker.v <= detection.box.y2
        for detection in detections
    )


def _initial_grid_indices(
    points: np.ndarray,
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], list[tuple[int, int, int]]]:
    centered = points - points.mean(axis=0)
    covariance = centered.T @ centered / len(points)
    values, vectors = np.linalg.eigh(covariance)
    if float(values[-1]) <= 0.0 or float(values[0]) <= 0.0:
        raise VisionError("rack detections do not form a two-dimensional grid")

    column_axis = vectors[:, -1]
    row_axis = vectors[:, 0]
    column_projection = centered @ column_axis
    row_projection = centered @ row_axis

    ordered_by_row = np.argsort(row_projection)
    first_row = tuple(
        int(index)
        for index in sorted(
            ordered_by_row[:6], key=lambda index: column_projection[index]
        )
    )
    second_row = tuple(
        int(index)
        for index in sorted(
            ordered_by_row[6:], key=lambda index: column_projection[index]
        )
    )

    grid_points: list[tuple[int, int, int]] = []
    for row, indices in enumerate((first_row, second_row)):
        for column, detection_index in enumerate(indices):
            grid_points.append((detection_index, row, column))
    return (first_row, second_row), grid_points


def _fit_affine_grid(
    points: np.ndarray,
    grid_points: Sequence[tuple[int, int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    design = np.array(
        [[1.0, float(column), float(row)] for _, row, column in grid_points],
        dtype=np.float64,
    )
    measured = np.array(
        [points[index] for index, _, _ in grid_points],
        dtype=np.float64,
    )
    coefficients, _, _, _ = np.linalg.lstsq(design, measured, rcond=None)
    predicted = design @ coefficients
    residual_px = float(np.sqrt(np.mean(np.sum((measured - predicted) ** 2, axis=1))))
    return coefficients[0], coefficients[1], coefficients[2], residual_px


def _measured_gaps(
    points: np.ndarray,
    indexed_rows: tuple[tuple[int, ...], tuple[int, ...]],
) -> tuple[np.ndarray, np.ndarray]:
    first, second = indexed_rows
    column_gaps = []
    for row in (first, second):
        column_gaps.extend(
            hypot(*(points[right] - points[left]))
            for left, right in zip(row, row[1:])
        )
    row_gaps = [
        hypot(*(points[lower] - points[upper]))
        for upper, lower in zip(first, second)
    ]
    return np.asarray(column_gaps), np.asarray(row_gaps)


def _check_spacing(name: str, gaps: np.ndarray, maximum_cv: float) -> None:
    mean = float(gaps.mean())
    if mean < 1.0:
        raise VisionError(f"{name} spacing is too small")
    cv = float(gaps.std() / mean)
    if cv > maximum_cv:
        raise VisionError(f"{name} spacing variation is too large: {cv:.3f}")


def _select_unique_k0(
    marker_candidates: Sequence[Pixel],
    corners: dict[tuple[int, int], np.ndarray],
    origin: np.ndarray,
    column_step: np.ndarray,
    row_step: np.ndarray,
    maximum_distance_px: float,
) -> tuple[Pixel, tuple[int, int]]:
    nearby: list[tuple[float, Pixel, tuple[int, int]]] = []
    for marker in marker_candidates:
        marker_point = np.array([marker.u, marker.v], dtype=np.float64)
        for corner, corner_point in corners.items():
            distance = float(np.linalg.norm(marker_point - corner_point))
            if distance <= maximum_distance_px:
                nearby.append((distance, marker, corner))

    if not nearby:
        nearest = min(
            float(
                np.linalg.norm(
                    np.array([marker.u, marker.v], dtype=np.float64)
                    - corner_point
                )
            )
            for marker in marker_candidates
            for corner_point in corners.values()
        )
        raise VisionError(
            "K0 marker is too far from the rack corner: "
            f"{nearest:.1f}px > {maximum_distance_px:.1f}px"
        )

    basis = np.column_stack((column_step, row_step))
    if abs(float(np.linalg.det(basis))) < 1e-9:
        raise VisionError("rack grid axes are singular")

    valid: list[tuple[float, Pixel, tuple[int, int]]] = []
    for candidate in nearby:
        _, marker, corner = candidate
        grid_coordinate = np.linalg.solve(
            basis,
            np.array([marker.u, marker.v], dtype=np.float64) - origin,
        )
        if _is_outside_corner(grid_coordinate, corner):
            valid.append(candidate)

    if not valid:
        raise VisionError(
            "no K0 candidate is on the outside of a rack corner"
        )
    if len(valid) != 1:
        raise VisionError(
            f"ambiguous K0 marker: {len(valid)} outside-corner candidates"
        )
    return valid[0][1], valid[0][2]


def _is_outside_corner(
    grid_coordinate: np.ndarray,
    corner: tuple[int, int],
) -> bool:
    """Allow a small fit tolerance, but require K0 outside both grid axes."""
    column, row = (float(value) for value in grid_coordinate)
    raw_row, raw_column = corner
    tolerance = 0.15
    column_outside = (
        column <= tolerance
        if raw_column == 0
        else column >= 5.0 - tolerance
    )
    row_outside = (
        row <= tolerance if raw_row == 0 else row >= 1.0 - tolerance
    )
    return column_outside and row_outside
