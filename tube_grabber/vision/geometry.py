"""Coordinate conversion from aligned D435 pixels to the right-arm base."""

from __future__ import annotations

import cv2
import numpy as np

from tube_grabber.core.errors import VisionError
from tube_grabber.core.models import CameraIntrinsics, Pixel, Point3D, Pose6D


def validate_transform(matrix: object, name: str = "transform") -> np.ndarray:
    transform = np.asarray(matrix, dtype=np.float64)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise VisionError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise VisionError(f"{name} has an invalid final row")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3):
        raise VisionError(f"{name} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-3):
        raise VisionError(f"{name} rotation determinant must be +1")
    return transform.copy()


def pose_to_matrix(pose: Pose6D) -> np.ndarray:
    """Convert base-to-end pose using R = Rz @ Ry @ Rx."""
    if pose.frame != "base_right":
        raise VisionError(f"arm pose must use base_right, got {pose.frame}")

    cx, sx = np.cos(pose.rx_rad), np.sin(pose.rx_rad)
    cy, sy = np.cos(pose.ry_rad), np.sin(pose.ry_rad)
    cz, sz = np.cos(pose.rz_rad), np.sin(pose.rz_rad)
    rotation_x = np.array(
        [[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]]
    )
    rotation_y = np.array(
        [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]]
    )
    rotation_z = np.array(
        [[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]]
    )

    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation_z @ rotation_y @ rotation_x
    matrix[:3, 3] = [pose.x_mm, pose.y_mm, pose.z_mm]
    return matrix


def base_from_camera(arm_pose: Pose6D, end_from_camera: object) -> np.ndarray:
    """Compose T_base_camera = T_base_end @ T_end_camera."""
    return pose_to_matrix(arm_pose) @ validate_transform(
        end_from_camera,
        "hand-eye transform",
    )


def pixel_depth_to_base(
    pixel: Pixel,
    depth_mm: float,
    intrinsics: CameraIntrinsics,
    base_from_camera_matrix: object,
) -> Point3D:
    transform = validate_transform(base_from_camera_matrix, "base-from-camera")
    ray = camera_ray(pixel, intrinsics)
    camera_point = np.array(
        [
            ray[0] * depth_mm,
            ray[1] * depth_mm,
            depth_mm,
            1.0,
        ],
        dtype=np.float64,
    )
    base_point = transform @ camera_point
    return Point3D(*base_point[:3], frame="base_right")


def pixel_ray_to_horizontal_plane(
    pixel: Pixel,
    plane_z_mm: float,
    intrinsics: CameraIntrinsics,
    base_from_camera_matrix: object,
) -> Point3D:
    """Intersect a color-pixel ray with Z=plane_z_mm in base_right."""
    transform = validate_transform(base_from_camera_matrix, "base-from-camera")
    origin = transform[:3, 3]
    camera_direction = camera_ray(pixel, intrinsics)
    direction = transform[:3, :3] @ camera_direction
    if abs(float(direction[2])) < 1e-9:
        raise VisionError("pixel ray is parallel to the rack plane")
    scale = (float(plane_z_mm) - float(origin[2])) / float(direction[2])
    if scale <= 0.0:
        raise VisionError("rack plane is behind the camera")
    point = origin + scale * direction
    return Point3D(float(point[0]), float(point[1]), float(plane_z_mm))


def camera_ray(pixel: Pixel, intrinsics: CameraIntrinsics) -> np.ndarray:
    """Return the undistorted camera ray whose optical-Z component is one."""
    if intrinsics.distortion_model == "none" or not any(
        abs(value) > 1e-12 for value in intrinsics.distortion_coefficients
    ):
        return np.array(
            [
                (pixel.u - intrinsics.cx) / intrinsics.fx,
                (pixel.v - intrinsics.cy) / intrinsics.fy,
                1.0,
            ],
            dtype=np.float64,
        )

    camera_matrix = np.array(
        [
            [intrinsics.fx, 0.0, intrinsics.cx],
            [0.0, intrinsics.fy, intrinsics.cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    image_point = np.array([[[pixel.u, pixel.v]]], dtype=np.float64)
    normalized = cv2.undistortPoints(
        image_point,
        camera_matrix,
        np.asarray(intrinsics.distortion_coefficients, dtype=np.float64),
    ).reshape(2)
    if not np.isfinite(normalized).all():
        raise VisionError("undistorted camera ray is not finite")
    return np.array([normalized[0], normalized[1], 1.0], dtype=np.float64)


# Kept for compatibility with callers from the first runtime revision.
_camera_ray = camera_ray


def pixel_ray_to_plane(
    pixel: Pixel,
    plane_normal_base: object,
    plane_offset_mm: float,
    intrinsics: CameraIntrinsics,
    base_from_camera_matrix: object,
) -> Point3D:
    """Intersect a color-pixel ray with ``normal·point + offset = 0``."""
    transform = validate_transform(base_from_camera_matrix, "base-from-camera")
    normal = np.asarray(plane_normal_base, dtype=np.float64).reshape(-1)
    if normal.shape != (3,) or not np.isfinite(normal).all():
        raise VisionError("rack plane normal must contain three finite values")
    length = float(np.linalg.norm(normal))
    if length <= 1e-9:
        raise VisionError("rack plane normal is zero")
    normal = normal / length
    offset = float(plane_offset_mm) / length
    origin = transform[:3, 3]
    direction = transform[:3, :3] @ camera_ray(pixel, intrinsics)
    denominator = float(np.dot(normal, direction))
    if abs(denominator) < 1e-9:
        raise VisionError("pixel ray is parallel to the rack plane")
    scale = -float(np.dot(normal, origin) + offset) / denominator
    if scale <= 0.0:
        raise VisionError("rack plane is behind the camera")
    point = origin + scale * direction
    return Point3D(float(point[0]), float(point[1]), float(point[2]))
