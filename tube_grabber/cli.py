"""Single command-line entry point for fake runs and staged lab operation."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from typing import Any, Sequence

import cv2
import numpy as np

from tube_grabber.agent import build_command_agent
from tube_grabber.app import CameraRackObserver, TubeGrabberRuntime, build_runtime
from tube_grabber.config import load_config, load_yaml, project_path
from tube_grabber.core.errors import (
    ConfigError,
    TubeGrabberError,
    VisionError,
    WorkflowError,
)
from tube_grabber.core.models import Pixel, TransferCommand
from tube_grabber.core.parsing import parse_transfer
from tube_grabber.diagnostics import (
    format_observation,
    format_prepared_transfer,
    save_camera_frame,
    save_observation_image,
)
from tube_grabber.vision.geometry import validate_transform
from tube_grabber.vision.rack_calibration import (
    RackCircleFitConfig,
    SlotCircle,
    calibrate_slot_grid,
    fit_slot_circle,
    load_rack_calibration,
    save_rack_calibration,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == "doctor":
            return _doctor(config)

        runtime = build_runtime(
            config,
            load_calibrations=args.command != "calibrate-rack",
        )
        if args.command == "arm-status":
            return _arm_status(runtime)
        if args.command == "camera-check":
            return _camera_check(runtime)
        if args.command == "gripper-status":
            return _gripper_status(runtime)
        if args.command == "scan":
            return _scan(runtime, config, args.rack)
        if args.command == "calibrate-rack":
            return _calibrate_rack(runtime, config, args.rack, force=args.force)
        if args.command in ("agent-plan", "agent-transfer"):
            return _agent_command(
                runtime,
                config,
                args.text,
                provider_override=args.agent_provider,
                execute=args.command == "agent-transfer",
            )
        if args.command in ("plan-transfer", "transfer"):
            command = parse_transfer(args.source, args.destination)
            if args.command == "plan-transfer":
                return _plan_transfer(runtime, config, command)
            return _transfer(runtime, config, command)
        parser.error(f"unknown command: {args.command}")
    except KeyboardInterrupt:
        print("\n已取消。", file=sys.stderr)
        return 130
    except (TubeGrabberError, ValueError, KeyError, TypeError) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 2
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tube-grabber",
        description="2x6 试管架视觉抓放主程序",
    )
    parser.add_argument(
        "--config",
        default="config/app.yaml",
        help="配置文件路径（默认 config/app.yaml）",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "doctor",
        help="静态检查配置、依赖、模型和标定门槛",
    )
    subparsers.add_parser(
        "arm-status",
        help="只连接右臂并读取当前法兰位姿",
    )
    subparsers.add_parser(
        "camera-check",
        help="只采集一帧 D435 彩色图和深度图",
    )
    subparsers.add_parser(
        "gripper-status",
        help="只连接并读取两指夹爪状态，不发送夹爪运动",
    )

    scan = subparsers.add_parser("scan", help="识别一个固定站的 2x6 试管架")
    scan.add_argument("--rack", required=True, choices=("rack_1", "rack_2"))
    calibrate = subparsers.add_parser(
        "calibrate-rack",
        help="正上方多帧识别后拟合并微调 r1c1/r2c6 槽圆",
    )
    calibrate.add_argument("--rack", required=True, choices=("rack_1", "rack_2"))
    calibrate.add_argument(
        "--force",
        action="store_true",
        help="明确覆盖该 rack 已存在的双圆心标定",
    )

    plan = subparsers.add_parser(
        "plan-transfer",
        help="视觉定位并打印全部航点，不初始化夹爪、不发送运动",
    )
    _add_transfer_arguments(plan)
    transfer = subparsers.add_parser(
        "transfer",
        help="执行同一机架内的一次抓放",
    )
    _add_transfer_arguments(transfer)
    agent_plan = subparsers.add_parser(
        "agent-plan",
        help="Agent 解析文本并打印计划，永不发送机械臂运动",
    )
    agent_plan.add_argument(
        "--text",
        required=True,
        help="一条自然语言命令，local 模式使用两个标准槽位地址",
    )
    agent_plan.add_argument(
        "--agent-provider",
        choices=("local", "gemini"),
        default=None,
        help="仅覆盖本次命令的 Agent provider，不修改 app.yaml",
    )
    agent_transfer = subparsers.add_parser(
        "agent-transfer",
        help=(
            "Agent 解析文本，经确定性校验和安全门禁后"
            "执行同架抓放"
        ),
    )
    agent_transfer.add_argument(
        "--text",
        required=True,
        help="一条包含源槽位和目标槽位的自然语言命令",
    )
    agent_transfer.add_argument(
        "--agent-provider",
        choices=("local", "gemini"),
        default=None,
        help="仅覆盖本次命令的 Agent provider，不修改 app.yaml",
    )
    return parser


def _add_transfer_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source", required=True, help="例如 rack_1.r1c1")
    parser.add_argument("--destination", required=True, help="例如 rack_1.r2c6")


def _doctor(config: dict[str, Any]) -> int:
    mode = config["runtime"]["mode"]
    failures: list[str] = []
    print(f"运行模式：{mode}")

    agent_config = config["agent"]
    agent_provider = str(agent_config["provider"])
    if agent_provider == "local":
        _status("OK", "Agent 使用离线标准槽位解析器")
    else:
        key_environment = str(agent_config["api_key_env"])
        if _module_exists("google.genai"):
            _status("OK", "Google Gen AI SDK 可导入")
        else:
            _status("FAIL", 'Gemini Agent 缺少依赖：pip install -e ".[agent]"')
            failures.append("google-genai")
        if os.getenv(key_environment):
            _status("OK", f"Agent 密钥环境变量 {key_environment} 已设置")
        else:
            _status("FAIL", f"Agent 密钥环境变量 {key_environment} 未设置")
            failures.append(key_environment)

    try:
        build_runtime(config)
        _status("OK", "配置可完整组装")
    except Exception as error:
        _status("FAIL", f"配置组装失败：{error}")
        failures.append("configuration")

    try:
        hand_eye = load_yaml(config["geometry"]["hand_eye_path"])
        if hand_eye.get("translation_unit") != "mm":
            raise ConfigError("translation_unit must be mm")
        if hand_eye.get("source_frame") != "camera_rightwrist":
            raise ConfigError("source_frame must be camera_rightwrist")
        if hand_eye.get("target_frame") != "end_right":
            raise ConfigError("target_frame must be end_right")
        validate_transform(hand_eye["matrix"], "hand-eye transform")
        _status("OK", "手眼矩阵为有效的 camera -> right flange 变换（mm）")
    except Exception as error:
        _status("FAIL", f"手眼标定无效：{error}")
        failures.append("hand-eye")

    poses = load_yaml(config["motion"]["poses_path"])
    confirmed = bool(poses.get("observation_pose", {}).get("confirmed", False))
    if confirmed:
        _status("OK", "observation_pose 已人工确认")
    else:
        level = "FAIL" if mode == "real" else "WAIT"
        _status(level, "observation_pose 尚未在真机低速确认")
        if mode == "real":
            failures.append("observation pose")

    motion_confirmed = bool(config["motion"].get("parameters_confirmed", False))
    if motion_confirmed:
        _status("OK", "TCP、工作空间和竖直运动高度已真机确认")
    else:
        level = "FAIL" if mode == "real" else "WAIT"
        _status(level, "TCP、工作空间和竖直运动高度尚未真机确认")
        if mode == "real":
            failures.append("motion parameters")

    for model_name, model_config in (
        ("试管盖 detection", config["vision"]["cap"]),
        ("架面八点 pose", config["vision"]["rack_pose"]),
    ):
        model_path = project_path(model_config["model_path"])
        if model_path.is_file():
            _status("OK", f"{model_name} 模型存在：{model_path}")
        else:
            level = "FAIL" if mode == "real" else "WAIT"
            _status(level, f"{model_name} 模型尚不存在：{model_path}")
            if mode == "real":
                failures.append(f"{model_name} model")

    model_paths = (
        project_path(config["vision"]["cap"]["model_path"]),
        project_path(config["vision"]["rack_pose"]["model_path"]),
    )
    if all(path.is_file() for path in model_paths) and _module_exists(
        "ultralytics"
    ):
        try:
            _validate_model_contracts(*model_paths)
            _status("OK", "两个 YOLO 权重的类别和 Pose 8×3 契约正确")
        except Exception as error:
            _status("FAIL", f"YOLO 权重契约错误：{error}")
            failures.append("model contracts")

    inference_device = str(config["vision"]["device"])
    if inference_device.lower() != "cpu":
        try:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("torch.cuda.is_available() is false")
            device_index = int(inference_device)
            device_name = torch.cuda.get_device_name(device_index)
            _status(
                "OK",
                f"YOLO 使用 CUDA:{device_index}（{device_name}，"
                f"torch CUDA {torch.version.cuda}）",
            )
        except Exception as error:
            level = "FAIL" if mode == "real" else "WAIT"
            _status(level, f"本地 CUDA 推理尚不可用：{error}")
            if mode == "real":
                failures.append("CUDA inference")

    for module, label in (
        ("pyrealsense2", "Intel RealSense Python"),
        ("ultralytics", "Ultralytics YOLO"),
        ("Robotic_Arm.rm_robot_interface", "RealMan Python SDK"),
    ):
        installed = _module_exists(module)
        if installed:
            _status("OK", f"{label} 可导入")
        else:
            level = "FAIL" if mode == "real" else "WAIT"
            _status(level, f"{label} 当前环境不可导入")
            if mode == "real":
                failures.append(module)

    for rack_id, rack in config["racks"].items():
        calibration_path = project_path(rack["calibration_path"])
        if not calibration_path.is_file():
            level = "FAIL" if mode == "real" else "WAIT"
            _status(
                level,
                f"{rack_id} 尚未完成 r1c1/r2c6 双圆心标定："
                f"{calibration_path}",
            )
            if mode == "real":
                failures.append(f"{rack_id} slot calibration")
        else:
            try:
                load_rack_calibration(calibration_path, rack_id)
                _status("OK", f"{rack_id} 双圆心标定格式有效")
            except ConfigError as error:
                _status("FAIL", f"{rack_id} 双圆心标定无效：{error}")
                failures.append(f"{rack_id} slot calibration")

    if failures:
        print("预检未通过：" + ", ".join(failures))
        return 2
    if mode == "fake":
        print(
            "假硬件闭环可运行；WAIT 项是切换 real 前必须处理的"
            "真机条件。"
        )
    else:
        print(
            "静态真机预检通过；仍需依次运行 arm-status、"
            "camera-check、gripper-status、scan。"
        )
    return 0


def _arm_status(runtime: TubeGrabberRuntime) -> int:
    try:
        runtime.start(need_arm=True, need_camera=False, need_gripper=False)
        pose = runtime.arm.get_pose()
        print(
            "right flange (base_right, mm + rad): "
            f"[{pose.x_mm:.3f}, {pose.y_mm:.3f}, {pose.z_mm:.3f}, "
            f"{pose.rx_rad:.6f}, {pose.ry_rad:.6f}, {pose.rz_rad:.6f}]"
        )
        if runtime.mode == "real":
            arm = runtime.arm
            if not hasattr(arm, "get_run_mode") or not hasattr(
                arm, "get_power_state"
            ):
                raise ConfigError("real arm driver cannot report controller state")
            run_mode = arm.get_run_mode()  # type: ignore[attr-defined]
            power_state = arm.get_power_state()  # type: ignore[attr-defined]
            if not hasattr(arm, "require_healthy"):
                raise ConfigError("real arm driver cannot report controller health")
            arm.require_healthy()  # type: ignore[attr-defined]
            print(
                f"controller: mode={'real' if run_mode == 1 else 'simulation'} "
                f"power={'on' if power_state == 1 else 'off'} health=OK"
            )
        return 0
    finally:
        runtime.close()


def _camera_check(runtime: TubeGrabberRuntime) -> int:
    if runtime.mode != "real" or runtime.camera is None:
        raise ConfigError("camera-check requires runtime.mode: real")
    try:
        runtime.start(need_arm=False, need_camera=True, need_gripper=False)
        frame = runtime.camera.capture()
        color_path, depth_path = save_camera_frame(frame)
        depth = np.asarray(frame.depth_mm)
        valid = np.isfinite(depth) & (depth > 0)
        ratio = 100.0 * float(valid.mean()) if depth.size else 0.0
        print(f"彩色图：{color_path}")
        print(f"深度图：{depth_path}")
        print(f"有效深度像素：{ratio:.1f}%")
        print(
            "运行时内参："
            f"fx={frame.intrinsics.fx:.3f}, fy={frame.intrinsics.fy:.3f}, "
            f"cx={frame.intrinsics.cx:.3f}, cy={frame.intrinsics.cy:.3f}"
        )
        print(
            "彩色畸变："
            f"model={frame.intrinsics.distortion_model}, "
            f"coefficients={frame.intrinsics.distortion_coefficients}"
        )
        return 0
    finally:
        runtime.close()


def _gripper_status(runtime: TubeGrabberRuntime) -> int:
    """Read the two-finger gripper contract without commanding its position."""
    if runtime.mode != "real":
        raise ConfigError("gripper-status requires runtime.mode: real")
    try:
        runtime.start(
            need_arm=True,
            need_camera=False,
            need_gripper=True,
        )
        print(
            "two-finger gripper: online=1 enabled=1 error=0; "
            "no position command was sent"
        )
        return 0
    finally:
        runtime.close()


def _calibrate_rack(
    runtime: TubeGrabberRuntime,
    config: dict[str, Any],
    rack_id: str,
    *,
    force: bool,
) -> int:
    """Fit and manually refine the two diagonal slot circles."""
    if runtime.mode != "real" or not isinstance(runtime.observer, CameraRackObserver):
        raise ConfigError("calibrate-rack requires runtime.mode: real")
    output = project_path(config["racks"][rack_id]["calibration_path"])
    if output.exists() and not force:
        raise ConfigError(f"rack calibration already exists: {output}; use --force")
    circle_config = _rack_circle_fit_config(config["vision"]["calibration"])
    window = f"calibrate {rack_id}: overhead circle fit"
    try:
        runtime.start(need_arm=True, need_camera=True, need_gripper=False)
        frame, stability = runtime.observer.capture_stable_pose()
        image = np.asarray(frame.color)
        confirmed: list[SlotCircle] = []
        active: SlotCircle | None = None
        dragging = False
        fit_note = "click near r1c1 to fit its circle"

        def on_mouse(
            event: int,
            x: int,
            y: int,
            flags: int,
            _data: object,
        ) -> None:
            nonlocal active, dragging, fit_note
            if len(confirmed) >= 2:
                return
            point = Pixel(float(x), float(y))
            if event == cv2.EVENT_LBUTTONDOWN:
                dragging = True
                if active is None:
                    try:
                        active = fit_slot_circle(image, point, circle_config)
                        fit_note = "auto fit ready; drag center and adjust radius"
                    except VisionError as error:
                        active = SlotCircle(
                            center=point,
                            radius_px=float(circle_config.default_radius_px),
                            automatically_fitted=False,
                        )
                        fit_note = f"auto fit failed ({error}); using manual circle"
                else:
                    active = _move_slot_circle(
                        active,
                        image,
                        center=point,
                    )
            elif event == cv2.EVENT_MOUSEMOVE and (
                dragging or flags & cv2.EVENT_FLAG_LBUTTON
            ):
                if active is not None:
                    active = _move_slot_circle(
                        active,
                        image,
                        center=point,
                    )
            elif event == cv2.EVENT_LBUTTONUP:
                dragging = False

        try:
            # One displayed pixel must equal one camera pixel for calibration.
            cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(window, on_mouse)
            while True:
                preview = image.copy()
                pose = stability.detection
                corners = np.asarray(
                    [[round(point.u), round(point.v)] for point in pose.corners],
                    dtype=np.int32,
                )
                cv2.polylines(preview, [corners], True, (255, 180, 0), 2)
                for keypoint in pose.keypoints:
                    center = (
                        int(round(keypoint.pixel.u)),
                        int(round(keypoint.pixel.v)),
                    )
                    cv2.drawMarker(
                        preview,
                        center,
                        (255, 0, 255),
                        cv2.MARKER_CROSS,
                        16,
                        2,
                    )
                    cv2.putText(
                        preview,
                        keypoint.name,
                        (center[0] + 5, center[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (255, 0, 255),
                        1,
                        cv2.LINE_AA,
                    )
                for index, circle in enumerate(confirmed):
                    _draw_slot_circle(preview, circle, index, confirmed=True)
                if active is not None:
                    _draw_slot_circle(
                        preview,
                        active,
                        len(confirmed),
                        confirmed=False,
                    )
                calibration = None
                grid_error = ""
                if len(confirmed) == 2:
                    try:
                        calibration = calibrate_slot_grid(
                            rack_id,
                            pose,
                            confirmed[0].center,
                            confirmed[1].center,
                        )
                        for address, pixel in calibration.project_slots(pose):
                            center = (int(round(pixel.u)), int(round(pixel.v)))
                            cv2.circle(preview, center, 5, (0, 220, 0), 1)
                            cv2.putText(
                                preview,
                                f"r{address.row}c{address.column}",
                                (center[0] + 5, center[1] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.35,
                                (0, 220, 0),
                                1,
                                cv2.LINE_AA,
                            )
                    except VisionError as error:
                        calibration = None
                        grid_error = str(error)
                status = (
                    f"frames={len(stability.inlier_indices)}/"
                    f"{stability.attempted_frames} "
                    f"keypoint_spread={stability.maximum_keypoint_spread_px:.2f}px"
                )
                cv2.putText(
                    preview,
                    status,
                    (18, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                target = "r1c1" if len(confirmed) == 0 else "r2c6"
                if len(confirmed) == 2:
                    target = "both circles confirmed"
                instruction = (
                    f"target={target} | drag=center | [ ]=radius | "
                    "Enter=confirm | Backspace=undo"
                )
                cv2.putText(
                    preview,
                    instruction,
                    (18, 58),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    preview,
                    fit_note[:110],
                    (18, 86),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    preview,
                    (grid_error or "r=reset all | s=save | q=cancel")[:110],
                    (18, 112),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255) if grid_error else (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.imshow(window, preview)
                key = cv2.waitKeyEx(20)
                low_key = key & 0xFF
                if low_key in (ord("q"), 27):
                    raise ConfigError("rack calibration cancelled")
                if low_key == ord("r"):
                    confirmed.clear()
                    active = None
                    fit_note = "click near r1c1 to fit its circle"
                elif low_key in (8, 127):
                    if active is not None:
                        active = None
                    elif confirmed:
                        active = confirmed.pop()
                    fit_note = "circle removed; click or refine again"
                elif active is not None and low_key in (13, 10, 32):
                    confirmed.append(active)
                    active = None
                    fit_note = (
                        "click near r2c6 to fit its circle"
                        if len(confirmed) == 1
                        else "inspect the green grid, then press s to save"
                    )
                elif active is not None and low_key in (ord("["), ord("-")):
                    active = _resize_slot_circle(
                        active,
                        circle_config,
                        -1.0,
                    )
                elif active is not None and low_key in (ord("]"), ord("+")):
                    active = _resize_slot_circle(
                        active,
                        circle_config,
                        1.0,
                    )
                elif active is not None:
                    movement = _circle_key_movement(key, low_key)
                    if movement is not None:
                        active = _move_slot_circle(
                            active,
                            image,
                            du=movement[0],
                            dv=movement[1],
                        )
                if low_key == ord("s") and calibration is not None:
                    saved = save_rack_calibration(calibration, output, force=force)
                    print(
                        f"已保存 {rack_id} 双圆心标定：{saved}\n"
                        f"稳定帧={len(stability.inlier_indices)}/"
                        f"{stability.attempted_frames}，最大关键点抖动="
                        f"{stability.maximum_keypoint_spread_px:.2f}px"
                    )
                    return 0
        except cv2.error as exc:
            raise ConfigError(
                "cannot open the calibration window; run from a desktop session"
            ) from exc
        finally:
            try:
                cv2.destroyWindow(window)
            except cv2.error:
                pass
    finally:
        runtime.close()


def _rack_circle_fit_config(data: dict[str, Any]) -> RackCircleFitConfig:
    return RackCircleFitConfig(
        search_radius_px=int(data["search_radius_px"]),
        minimum_radius_px=int(data["minimum_radius_px"]),
        maximum_radius_px=int(data["maximum_radius_px"]),
        default_radius_px=int(data["default_radius_px"]),
        hough_dp=float(data["hough_dp"]),
        hough_min_distance_px=float(data["hough_min_distance_px"]),
        hough_edge_threshold=float(data["hough_edge_threshold"]),
        hough_accumulator_threshold=float(
            data["hough_accumulator_threshold"]
        ),
    )


def _draw_slot_circle(
    image: np.ndarray,
    circle: SlotCircle,
    index: int,
    *,
    confirmed: bool,
) -> None:
    center = (
        int(round(circle.center.u)),
        int(round(circle.center.v)),
    )
    color = (0, 220, 0) if confirmed else (0, 165, 255)
    cv2.circle(image, center, int(round(circle.radius_px)), color, 2)
    cv2.drawMarker(image, center, color, cv2.MARKER_CROSS, 16, 2)
    name = "r1c1" if index == 0 else "r2c6"
    cv2.putText(
        image,
        f"{name} r={circle.radius_px:.1f}px",
        (center[0] + 8, center[1] - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def _move_slot_circle(
    circle: SlotCircle,
    image: np.ndarray,
    *,
    center: Pixel | None = None,
    du: float = 0.0,
    dv: float = 0.0,
) -> SlotCircle:
    target = center or Pixel(circle.center.u + du, circle.center.v + dv)
    radius = circle.radius_px
    height, width = image.shape[:2]
    u = float(np.clip(target.u, radius, width - 1 - radius))
    v = float(np.clip(target.v, radius, height - 1 - radius))
    return SlotCircle(Pixel(u, v), radius, automatically_fitted=False)


def _resize_slot_circle(
    circle: SlotCircle,
    config: RackCircleFitConfig,
    change_px: float,
) -> SlotCircle:
    radius = float(
        np.clip(
            circle.radius_px + change_px,
            config.minimum_radius_px,
            config.maximum_radius_px,
        )
    )
    return SlotCircle(circle.center, radius, automatically_fitted=False)


def _circle_key_movement(
    key: int,
    low_key: int,
) -> tuple[float, float] | None:
    # waitKeyEx arrow codes vary between Linux/X11 and OpenCV backends.
    if key in (2424832, 81, 65361) or low_key == ord("j"):
        return -1.0, 0.0
    if key in (2555904, 83, 65363) or low_key == ord("l"):
        return 1.0, 0.0
    if key in (2490368, 82, 65362) or low_key == ord("i"):
        return 0.0, -1.0
    if key in (2621440, 84, 65364) or low_key == ord("k"):
        return 0.0, 1.0
    return None


def _scan(
    runtime: TubeGrabberRuntime,
    config: dict[str, Any],
    rack_id: str,
) -> int:
    try:
        runtime.start(
            need_arm=True,
            need_camera=runtime.mode == "real",
            need_gripper=False,
        )
        runtime.require_observation_pose()
        observation = runtime.workflow.scan(rack_id)
        print(format_observation(observation))
        _save_scan_if_available(runtime, config)
        return 0
    except BaseException:
        _stop_quietly(runtime)
        raise
    finally:
        runtime.close()


def _plan_transfer(
    runtime: TubeGrabberRuntime,
    config: dict[str, Any],
    command: TransferCommand,
) -> int:
    _require_local_transfer(command)
    try:
        runtime.start(
            need_arm=True,
            need_camera=runtime.mode == "real",
            need_gripper=False,
        )
        runtime.require_observation_pose()
        prepared = runtime.workflow.prepare_transfer(command)
        print(format_observation(prepared.observation))
        print(format_prepared_transfer(prepared))
        _save_scan_if_available(runtime, config)
        print("规划完成：未初始化夹爪，未发送任何机械臂运动。")
        return 0
    except BaseException:
        _stop_quietly(runtime)
        raise
    finally:
        runtime.close()


def _agent_command(
    runtime: TubeGrabberRuntime,
    config: dict[str, Any],
    text: str,
    *,
    provider_override: str | None = None,
    execute: bool,
) -> int:
    agent_config = dict(config["agent"])
    if provider_override is not None:
        agent_config["provider"] = provider_override
    agent = build_command_agent(agent_config)
    decision = agent.interpret(text)
    print(f"Agent：{decision.reply}")
    if decision.command is None:
        print("命令信息不完整，未生成运动计划。")
        runtime.close()
        return 2
    print(
        "确定性校验后的命令："
        f"{decision.command.source.text} -> {decision.command.destination.text}"
    )
    if execute:
        return _transfer(runtime, config, decision.command)
    return _plan_transfer(runtime, config, decision.command)


def _transfer(
    runtime: TubeGrabberRuntime,
    config: dict[str, Any],
    command: TransferCommand,
) -> int:
    _require_local_transfer(command)
    if (
        runtime.mode == "real"
        and not project_path(
            config["racks"][command.source.rack_id]["calibration_path"]
        ).is_file()
    ):
        raise ConfigError(
            f"{command.source.rack_id} 尚未完成 r1c1/r2c6 双圆心标定；"
            "真实运动被锁定"
        )
    try:
        # Match the proven AprilTag workflow: verify the empty tool, then let
        # the program move to the taught global observation pose itself.
        runtime.start(
            need_arm=True,
            need_camera=False,
            need_gripper=True,
        )
        runtime.require_motion_ready()
        if runtime.mode == "real" and bool(
            config["runtime"]["require_enter_before_motion"]
        ):
            print(
                "程序将自动移动右臂到全局观察位。\n"
                "确认左臂已收回、底盘锁定、夹爪/TCP 确实空载、"
                "路径无障碍且急停可触达。"
            )
            try:
                empty_confirmation = input(
                    "输入 EMPTY 并回车，允许移动到观察位："
                ).strip()
            except EOFError as error:
                raise WorkflowError(
                    "observation-pose authorization input is unavailable"
                ) from error
            if empty_confirmation != "EMPTY":
                raise WorkflowError(
                    "operator did not confirm an empty tool for observation motion"
                )
        runtime.move_to_observation_pose()
        runtime.start(
            need_arm=True,
            need_camera=runtime.mode == "real",
            need_gripper=False,
        )
        runtime.require_observation_pose()
        prepared = runtime.workflow.prepare_transfer(command)
        print(format_observation(prepared.observation))
        print(format_prepared_transfer(prepared))
        _save_scan_if_available(runtime, config)

        if runtime.mode == "real" and bool(
            config["runtime"]["require_enter_before_motion"]
        ):
            print(
                "即将执行："
                f"{command.source.text} -> {command.destination.text}\n"
                "确认首轮观测和预览航点正确，急停可触达。"
            )
            try:
                authorization = input("输入 MOVE 并回车开始：").strip()
            except EOFError as error:
                raise WorkflowError(
                    "motion authorization input is unavailable"
                ) from error
            if authorization != "MOVE":
                raise WorkflowError("operator did not authorize motion")

        runtime.require_motion_ready()
        runtime.require_observation_pose()
        prepared = runtime.workflow.refresh_prepared_transfer(prepared)
        _save_scan_if_available(runtime, config)
        print(
            "执行前复扫通过，并已用本次最新坐标重建抓取计划。"
        )
        final = runtime.workflow.execute_transfer(prepared)
        print("最终闭环复扫：")
        print(format_observation(final))
        _save_scan_if_available(runtime, config)
        print(
            f"运动完成：{command.source.text} -> {command.destination.text}；"
            "已自动回到观察位，并确认源槽为空、目标槽占用。"
        )
        return 0
    except BaseException:
        _stop_quietly(runtime)
        if runtime.workflow.holding_tube:
            print(
                "警告：软件状态显示夹爪仍持有试管，"
                "请勿直接移动底盘。",
                file=sys.stderr,
            )
        raise
    finally:
        runtime.close()


def _require_local_transfer(command: TransferCommand) -> None:
    if command.source.rack_id != command.destination.rack_id:
        raise WorkflowError(
            "当前版本尚未接入导航，跨架搬运被锁定；"
            "只能执行同一 rack 内搬运"
        )


def _save_scan_if_available(
    runtime: TubeGrabberRuntime,
    config: dict[str, Any],
) -> None:
    if not bool(config["runtime"]["save_debug_images"]):
        return
    if not isinstance(runtime.observer, CameraRackObserver):
        return
    frame = runtime.observer.last_frame
    observation = runtime.observer.last_observation
    if frame is None or observation is None:
        return
    path = save_observation_image(frame, observation)
    print(f"标注图：{path}")


def _module_exists(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _validate_model_contracts(cap_path: object, pose_path: object) -> None:
    """Load weight metadata only; inference remains a separate CUDA check."""
    from ultralytics import YOLO

    cap = YOLO(str(cap_path), task="detect")
    cap_names = {int(key): str(value) for key, value in dict(cap.names).items()}
    if cap_names != {0: "tube_cap"}:
        raise ConfigError(f"cap classes are {cap_names}, expected {{0: 'tube_cap'}}")

    pose = YOLO(str(pose_path), task="pose")
    pose_names = {
        int(key): str(value) for key, value in dict(pose.names).items()
    }
    if pose_names != {0: "rack_surface"}:
        raise ConfigError(
            f"rack pose classes are {pose_names}, "
            "expected {0: 'rack_surface'}"
        )
    keypoint_shape = tuple(getattr(pose.model, "kpt_shape", ()))
    if keypoint_shape != (8, 3):
        raise ConfigError(
            f"rack pose keypoint shape is {keypoint_shape}, expected (8, 3)"
        )


def _status(level: str, message: str) -> None:
    print(f"[{level:<4}] {message}")


def _stop_quietly(runtime: TubeGrabberRuntime) -> None:
    try:
        runtime.arm.stop()
    except Exception:
        pass


if __name__ == "__main__":
    raise SystemExit(main())
