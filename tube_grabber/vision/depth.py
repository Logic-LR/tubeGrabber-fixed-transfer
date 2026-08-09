"""Aligned-depth sampling helpers."""

from __future__ import annotations

import numpy as np

from tube_grabber.core.errors import VisionError
from tube_grabber.core.models import Pixel


def sample_depth_mm(
    depth_image: object,
    pixel: Pixel,
    window_px: int,
    minimum_mm: float,
    maximum_mm: float,
) -> float:
    """Return the median valid depth around a color-image pixel."""
    depth = np.asarray(depth_image)
    if depth.ndim != 2:
        raise VisionError("aligned depth image must be two-dimensional")
    if window_px <= 0 or window_px % 2 == 0:
        raise VisionError("depth window must be a positive odd number")
    if minimum_mm <= 0.0 or maximum_mm <= minimum_mm:
        raise VisionError("invalid depth range")

    u = int(round(pixel.u))
    v = int(round(pixel.v))
    height, width = depth.shape
    if not 0 <= u < width or not 0 <= v < height:
        raise VisionError(f"depth pixel ({u}, {v}) is outside the image")

    half = window_px // 2
    patch = depth[
        max(0, v - half) : min(height, v + half + 1),
        max(0, u - half) : min(width, u + half + 1),
    ].astype(np.float64)
    valid = patch[
        np.isfinite(patch)
        & (patch >= float(minimum_mm))
        & (patch <= float(maximum_mm))
    ]
    if valid.size == 0:
        raise VisionError(f"no valid depth around pixel ({u}, {v})")
    return float(np.median(valid))
