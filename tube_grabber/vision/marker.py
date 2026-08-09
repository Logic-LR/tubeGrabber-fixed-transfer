"""HSV detection of the red or green K0 rack marker."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import cv2
import numpy as np

from tube_grabber.core.errors import VisionError
from tube_grabber.core.models import Pixel


class K0MarkerDetector:
    """Return all plausible marker centres; the grid chooses the K0 corner."""

    def __init__(
        self,
        minimum_area_px: int,
        minimum_saturation: int,
        minimum_value: int,
        maximum_aspect_ratio: float,
        hue_ranges: Mapping[str, Sequence[Sequence[int]]],
    ) -> None:
        if minimum_area_px <= 0:
            raise VisionError("marker minimum_area_px must be positive")
        if not 0 <= minimum_saturation <= 255:
            raise VisionError("marker minimum_saturation must be in [0, 255]")
        if not 0 <= minimum_value <= 255:
            raise VisionError("marker minimum_value must be in [0, 255]")
        if (
            not math.isfinite(float(maximum_aspect_ratio))
            or maximum_aspect_ratio < 1.0
        ):
            raise VisionError("marker maximum_aspect_ratio must be at least 1")

        self._minimum_area_px = int(minimum_area_px)
        self._minimum_saturation = int(minimum_saturation)
        self._minimum_value = int(minimum_value)
        self._maximum_aspect_ratio = float(maximum_aspect_ratio)
        self._hue_ranges = {
            color: _validate_ranges(color, ranges)
            for color, ranges in hue_ranges.items()
        }
        if set(self._hue_ranges) != {"red", "green"}:
            raise VisionError("marker hue ranges must define red and green")

    def detect(self, color_image: object, color: str) -> tuple[Pixel, ...]:
        if color not in self._hue_ranges:
            raise VisionError(f"unsupported marker color: {color}")

        image = np.asarray(color_image)
        if image.ndim != 3 or image.shape[2] != 3:
            raise VisionError("marker input must be a BGR image with three channels")

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for low_hue, high_hue in self._hue_ranges[color]:
            lower = np.array(
                [low_hue, self._minimum_saturation, self._minimum_value],
                dtype=np.uint8,
            )
            upper = np.array([high_hue, 255, 255], dtype=np.uint8)
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))

        component_count, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
        candidates: list[tuple[int, Pixel]] = []
        for component in range(1, component_count):
            area = int(stats[component, cv2.CC_STAT_AREA])
            if area < self._minimum_area_px:
                continue
            width = int(stats[component, cv2.CC_STAT_WIDTH])
            height = int(stats[component, cv2.CC_STAT_HEIGHT])
            aspect_ratio = max(width, height) / max(1, min(width, height))
            if aspect_ratio > self._maximum_aspect_ratio:
                continue
            u, v = centroids[component]
            candidates.append((area, Pixel(float(u), float(v))))

        if not candidates:
            raise VisionError(f"no {color} K0 marker found")

        candidates.sort(key=lambda item: item[0], reverse=True)
        return tuple(pixel for _, pixel in candidates)


def _validate_ranges(
    color: str,
    ranges: Sequence[Sequence[int]],
) -> tuple[tuple[int, int], ...]:
    validated: list[tuple[int, int]] = []
    for hue_range in ranges:
        if len(hue_range) != 2:
            raise VisionError(f"{color} hue range must contain two values")
        low, high = (int(value) for value in hue_range)
        if not 0 <= low <= high <= 179:
            raise VisionError(f"invalid {color} hue range: {hue_range}")
        validated.append((low, high))
    if not validated:
        raise VisionError(f"{color} hue ranges cannot be empty")
    return tuple(validated)
