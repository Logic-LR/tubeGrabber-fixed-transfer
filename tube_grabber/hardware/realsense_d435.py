"""Intel RealSense D435 彩色和对齐深度驱动。"""

from __future__ import annotations

from importlib import import_module

import numpy as np

from tube_grabber.core.errors import HardwareError
from tube_grabber.core.models import CameraFrame, CameraIntrinsics


class D435Camera:
    def __init__(
        self,
        serial: str,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        warmup_frames: int = 30,
        timeout_ms: int = 10_000,
    ) -> None:
        if width <= 0 or height <= 0 or fps <= 0:
            raise ValueError("相机分辨率和帧率必须大于 0")
        if warmup_frames < 0 or timeout_ms <= 0:
            raise ValueError("warmup_frames 不能为负，timeout_ms 必须大于 0")

        self.serial = serial
        self.width = width
        self.height = height
        self.fps = fps
        self.warmup_frames = warmup_frames
        self.timeout_ms = timeout_ms

        self._pipeline: object | None = None
        self._align: object | None = None
        self._depth_scale_mm = 0.0

    def start(self) -> None:
        if self._pipeline is not None:
            return

        try:
            rs = import_module("pyrealsense2")
        except ImportError as exc:
            raise HardwareError("未安装 pyrealsense2，无法启动 D435") from exc

        pipeline: object | None = None
        try:
            pipeline = rs.pipeline()
            config = rs.config()
            if self.serial:
                config.enable_device(self.serial)
            config.enable_stream(
                rs.stream.depth,
                self.width,
                self.height,
                rs.format.z16,
                self.fps,
            )
            config.enable_stream(
                rs.stream.color,
                self.width,
                self.height,
                rs.format.bgr8,
                self.fps,
            )

            profile = pipeline.start(config)
            depth_sensor = profile.get_device().first_depth_sensor()
            depth_scale_mm = float(depth_sensor.get_depth_scale()) * 1000.0
            if depth_scale_mm <= 0:
                raise HardwareError(f"D435 深度比例无效: {depth_scale_mm}")

            align = rs.align(rs.stream.color)
            for _ in range(self.warmup_frames):
                pipeline.wait_for_frames(self.timeout_ms)
        except Exception as exc:
            if pipeline is not None:
                try:
                    pipeline.stop()
                except Exception:
                    pass
            if isinstance(exc, HardwareError):
                raise
            raise HardwareError(f"D435 启动失败: {exc}") from exc

        self._pipeline = pipeline
        self._align = align
        self._depth_scale_mm = depth_scale_mm

    def capture(self) -> CameraFrame:
        pipeline, align = self._require_started()
        try:
            frames = pipeline.wait_for_frames(self.timeout_ms)
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                raise HardwareError("D435 没有同时返回彩色帧和深度帧")

            color = np.asanyarray(color_frame.get_data()).copy()
            depth_raw = np.asanyarray(depth_frame.get_data())
            depth_mm = _depth_to_mm(depth_raw, self._depth_scale_mm)
            if color.shape[:2] != depth_mm.shape[:2]:
                raise HardwareError(
                    f"对齐后图像尺寸不同: color={color.shape[:2]}, depth={depth_mm.shape[:2]}"
                )

            intrinsics = _read_color_intrinsics(color_frame)
            return CameraFrame(
                color=color,
                depth_mm=depth_mm,
                intrinsics=intrinsics,
                timestamp_ms=float(color_frame.get_timestamp()),
                frame_number=int(color_frame.get_frame_number()),
            )
        except Exception as exc:
            if isinstance(exc, HardwareError):
                raise
            raise HardwareError(f"D435 采集失败: {exc}") from exc

    def stop(self) -> None:
        pipeline = self._pipeline
        self._pipeline = None
        self._align = None
        self._depth_scale_mm = 0.0
        if pipeline is None:
            return
        try:
            pipeline.stop()
        except Exception as exc:
            raise HardwareError(f"D435 停止失败: {exc}") from exc

    def _require_started(self) -> tuple[object, object]:
        if self._pipeline is None or self._align is None:
            raise HardwareError("D435 尚未启动，请先调用 start()")
        return self._pipeline, self._align


def _read_color_intrinsics(color_frame: object) -> CameraIntrinsics:
    """从实际运行的彩色帧读取内参。"""
    profile = color_frame.profile.as_video_stream_profile()
    value = profile.get_intrinsics()
    coefficients = tuple(float(item) for item in value.coeffs)
    model = _distortion_model_name(getattr(value, "model", None))
    if model == "none" and any(abs(item) > 1e-6 for item in coefficients):
        raise HardwareError(
            "D435 报告无畸变模型但系数非零，内参不一致: "
            f"{coefficients}"
        )
    return CameraIntrinsics(
        fx=float(value.fx),
        fy=float(value.fy),
        cx=float(value.ppx),
        cy=float(value.ppy),
        distortion_model=model,
        distortion_coefficients=coefficients if model != "none" else (),
    )


def _distortion_model_name(value: object) -> str:
    """Accept the two D435 color models that this geometry layer supports."""
    text = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    try:
        numeric = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        numeric = None
    if value is None or numeric == 0 or text in {"0", "none", "distortion.none"}:
        return "none"
    if numeric == 4 or (
        "brown_conrady" in text
        and "modified_brown_conrady" not in text
        and "inverse_brown_conrady" not in text
    ):
        return "brown_conrady"
    raise HardwareError(f"D435 彩色畸变模型暂不支持: {value!r}")


def _depth_to_mm(depth_raw: np.ndarray, scale_mm: float) -> np.ndarray:
    """RealSense 原始深度值转换成项目统一的 float32 毫米。"""
    if scale_mm <= 0:
        raise HardwareError(f"D435 深度比例无效: {scale_mm}")
    return depth_raw.astype(np.float32) * float(scale_mm)
