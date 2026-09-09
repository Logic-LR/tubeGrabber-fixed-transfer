"""Single command-line entry point for fake runs and staged lab operation."""

from __future__ import annotations

import argparse
import importlib.util
import multiprocessing
import os
import queue
import sys
import threading
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
    calibrate_display_corners,
    calibrate_slot_grid,
    fit_slot_circle,
    load_rack_calibration,
    save_rack_calibration,
)
from tube_grabber.vision.pose_pipeline import RackVisualDetection


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
            return _scan(
                runtime,
                config,
                args.rack,
                display=not args.no_display,
            )
        if args.command == "calibrate-rack":
            return _calibrate_rack(runtime, config, args.rack, force=args.force)
        if args.command == "calibrate-corners":
            return _calibrate_corners(runtime, config, args.rack)
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
    scan.add_argument(
        "--no-display",
        action="store_true",
        help="不打开实时窗口，只执行一次稳定扫描并退出",
    )
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
    corners = subparsers.add_parser(
        "calibrate-corners",
        help="用四螺丝架面标定演示画面的 K0-K3 四个交点",
    )
    corners.add_argument("--rack", required=True, choices=("rack_1", "rack_2"))

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
        _status("OK", "TCP、工作空间和架面法向运动高度已真机确认")
    else:
        level = "FAIL" if mode == "real" else "WAIT"
        _status(level, "TCP、工作空间和架面法向运动高度尚未真机确认")
        if mode == "real":
            failures.append("motion parameters")

    for model_name, model_config in (
        ("试管盖 detection", config["vision"]["cap"]),
        ("架面四螺丝 detection", config["vision"]["screw"]),
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
        project_path(config["vision"]["screw"]["model_path"]),
    )
    if all(path.is_file() for path in model_paths) and _module_exists(
        "ultralytics"
    ):
        try:
            _validate_model_contracts(*model_paths)
            _status("OK", "cap/screw 两个 YOLO detection 权重契约正确")
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


def _calibrate_corners(
    runtime: TubeGrabberRuntime,
    config: dict[str, Any],
    rack_id: str,
) -> int:
    """Teach four presentation corners relative to the detected screw plane."""
    if runtime.mode != "real" or not isinstance(runtime.observer, CameraRackObserver):
        raise ConfigError("calibrate-corners requires runtime.mode: real")
    output = project_path(config["racks"][rack_id]["calibration_path"])
    calibration = load_rack_calibration(output, rack_id)
    window = f"calibrate {rack_id}: K0-K3 display corners"
    try:
        runtime.start(need_arm=True, need_camera=True, need_gripper=False)
        frame, stability = runtime.observer.capture_stable_pose()
        image = np.asarray(frame.color)
        screw_pose = stability.detection
        taught = list(calibration.project_display_corners(screw_pose))
        active_index = 0
        dragging = False

        def on_mouse(
            event: int,
            x: int,
            y: int,
            flags: int,
            _data: object,
        ) -> None:
            nonlocal dragging
            if active_index >= 4:
                return
            if event == cv2.EVENT_LBUTTONDOWN:
                dragging = True
                taught[active_index] = Pixel(float(x), float(y))
            elif event == cv2.EVENT_MOUSEMOVE and (
                dragging or flags & cv2.EVENT_FLAG_LBUTTON
            ):
                taught[active_index] = Pixel(float(x), float(y))
            elif event == cv2.EVENT_LBUTTONUP:
                dragging = False

        try:
            cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(window, on_mouse)
            while True:
                preview = image.copy()

                # Screw centers are deliberately visible only in calibration.
                for point in screw_pose.corners:
                    center = (int(round(point.u)), int(round(point.v)))
                    cv2.drawMarker(
                        preview,
                        center,
                        (120, 120, 120),
                        cv2.MARKER_CROSS,
                        8,
                        1,
                        cv2.LINE_AA,
                    )

                taught_array = np.asarray(
                    [[round(point.u), round(point.v)] for point in taught],
                    dtype=np.int32,
                )
                cv2.polylines(
                    preview,
                    [taught_array],
                    True,
                    (255, 0, 255),
                    1,
                    cv2.LINE_AA,
                )
                for index, point in enumerate(taught):
                    center = (int(round(point.u)), int(round(point.v)))
                    color = (
                        (0, 165, 255)
                        if index == active_index
                        else (255, 0, 255)
                    )
                    cv2.circle(preview, center, 4, color, 1, cv2.LINE_AA)
                    cv2.putText(
                        preview,
                        f"K{index}",
                        (center[0] + 5, center[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.42,
                        color,
                        1,
                        cv2.LINE_AA,
                    )

                result = None
                corner_error = ""
                if active_index >= 4:
                    try:
                        result = calibrate_display_corners(
                            calibration,
                            screw_pose,
                            taught,
                        )
                    except VisionError as error:
                        corner_error = str(error)

                status = (
                    f"target={'DONE' if active_index >= 4 else f'K{active_index}'} | "
                    "click/drag or I/J/K/L | Enter=confirm | Backspace=previous"
                )
                cv2.putText(
                    preview,
                    status,
                    (18, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                hint = corner_error or "R=reset | S=save after K3 | Q=cancel"
                cv2.putText(
                    preview,
                    hint[:120],
                    (18, 56),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (0, 0, 255) if corner_error else (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.imshow(window, preview)
                key = cv2.waitKeyEx(20)
                low_key = key & 0xFF
                if low_key in (ord("q"), 27):
                    raise ConfigError("corner calibration cancelled")
                if low_key == ord("r"):
                    taught = list(screw_pose.corners)
                    active_index = 0
                elif low_key in (8, 127):
                    active_index = max(0, active_index - 1)
                elif low_key in (13, 10, 32) and active_index < 4:
                    active_index += 1
                elif active_index < 4:
                    movement = _circle_key_movement(key, low_key)
                    if movement is not None:
                        point = taught[active_index]
                        taught[active_index] = Pixel(
                            float(
                                np.clip(
                                    point.u + movement[0], 0, image.shape[1] - 1
                                )
                            ),
                            float(
                                np.clip(
                                    point.v + movement[1], 0, image.shape[0] - 1
                                )
                            ),
                        )
                if low_key == ord("s") and result is not None:
                    saved = save_rack_calibration(result, output, force=True)
                    print(
                        f"已保存 {rack_id} 四角点展示标定：{saved}\n"
                        "底层仍为四螺丝 DET；scan 只绘制标定后的 K0-K3。"
                    )
                    return 0
        except cv2.error as exc:
            raise ConfigError(
                "cannot open the corner calibration window; run from a desktop session"
            ) from exc
        finally:
            try:
                cv2.destroyWindow(window)
            except cv2.error:
                pass
    finally:
        runtime.close()


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
    *,
    display: bool = True,
) -> int:
    try:
        runtime.start(
            need_arm=True,
            need_camera=runtime.mode == "real",
            need_gripper=False,
        )
        runtime.require_observation_pose()
        if display and runtime.mode == "real":
            return _live_scan(runtime, config, rack_id)
        observation = runtime.workflow.scan(rack_id)
        print(format_observation(observation))
        _save_scan_if_available(runtime, config)
        return 0
    except BaseException:
        _stop_quietly(runtime)
        raise
    finally:
        runtime.close()


def _live_scan(
    runtime: TubeGrabberRuntime,
    config: dict[str, Any],
    rack_id: str,
) -> int:
    if not isinstance(runtime.observer, CameraRackObserver):
        raise VisionError("实时扫描需要 D435 相机观察器")
    window = f"{rack_id} live scan | S/SPACE/ENTER scan | Q/ESC quit"
    try:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    except cv2.error as exc:
        raise VisionError(
            "无法创建 OpenCV 窗口；图形环境不可用时请添加 --no-display"
        ) from exc

    print("实时窗口已启动：按 s、空格或 Enter 完成一次稳定扫描；按 q/ESC 退出。")
    scan_count = 0
    try:
        while True:
            frame, visual = runtime.observer.capture_visual(rack_id)
            preview = np.asarray(frame.color).copy()
            _draw_live_scan_overlay(preview, visual)
            status = (
                f"CUDA:{config['vision']['device']}  "
                f"cap@{float(config['vision']['cap']['confidence']):.2f}="
                f"{len(visual.caps)}  "
                f"corners={len(visual.display_corners)}  "
                + ("rack=OK" if visual.pose is not None else "rack=not found")
            )
            cv2.putText(
                preview,
                status[:150],
                (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (0, 220, 0) if visual.pose is not None else (0, 80, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                preview,
                "S/SPACE/ENTER: stable scan   Q/ESC: quit",
                (10, max(45, preview.shape[0] - 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.imshow(window, preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (ord("s"), ord(" "), 10, 13):
                cv2.putText(
                    preview,
                    "Capturing stable scan...",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.imshow(window, preview)
                cv2.waitKey(1)
                observation = runtime.workflow.scan(rack_id)
                scan_count += 1
                print(f"\n稳定扫描 #{scan_count}：")
                print(format_observation(observation))
                _save_scan_if_available(runtime, config)
            try:
                if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
    finally:
        try:
            cv2.destroyWindow(window)
        except cv2.error:
            pass
    print(f"实时扫描已退出；本次完成 {scan_count} 次稳定扫描。")
    return 0


def _draw_live_scan_overlay(
    image: np.ndarray,
    visual: RackVisualDetection,
) -> None:
    # The live/demo view intentionally exposes only the calibrated K0-K3 layer,
    # never the underlying screw boxes or centers.
    if len(visual.display_corners) == 4:
        corners = np.asarray(
            [
                [int(round(point.u)), int(round(point.v))]
                for point in visual.display_corners
            ],
            dtype=np.int32,
        )
        cv2.polylines(image, [corners], True, (255, 0, 255), 1, cv2.LINE_AA)
        for index, point in enumerate(visual.display_corners):
            center = (int(round(point.u)), int(round(point.v)))
            color = (0, 255, 255) if index == 0 else (255, 255, 0)
            cv2.circle(image, center, 5 if index == 0 else 3, color, 1, cv2.LINE_AA)
            cv2.putText(
                image,
                f"K{index}",
                (center[0] + 4, center[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                color,
                1,
                cv2.LINE_AA,
            )

    for address, point in visual.projected_slots:
        center = (int(round(point.u)), int(round(point.v)))
        cv2.drawMarker(
            image,
            center,
            (255, 120, 0),
            cv2.MARKER_CROSS,
            7,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            f"r{address.row}c{address.column}",
            (center[0] + 4, center[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (255, 120, 0),
            1,
            cv2.LINE_AA,
        )

    for cap in visual.caps:
        box = cap.box
        left, top = int(round(box.x1)), int(round(box.y1))
        right, bottom = int(round(box.x2)), int(round(box.y2))
        center = (int(round(box.center.u)), int(round(box.center.v)))
        cv2.rectangle(image, (left, top), (right, bottom), (0, 220, 0), 1)
        cv2.circle(image, center, 2, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(
            image,
            f"{cap.confidence:.2f}",
            (left, max(12, top - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (0, 220, 0),
            1,
            cv2.LINE_AA,
        )


def _transfer_display_process(
    window: str,
    frames: object,
    ready: object,
    errors: object,
) -> None:
    """Own all Qt/OpenCV GUI calls in a dedicated process main thread."""
    frame_queue = frames
    ready_event = ready
    error_queue = errors
    window_created = False
    try:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        window_created = True
        ready_event.set()
        while True:
            encoded = frame_queue.get()
            if encoded is None:
                break
            image = cv2.imdecode(
                np.frombuffer(encoded, dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            if image is None:
                continue
            cv2.imshow(window, image)
            cv2.waitKey(1)
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                raise RuntimeError("录像窗口被关闭")
    except BaseException as exc:
        try:
            error_queue.put_nowait(str(exc))
        except Exception:
            pass
        ready_event.set()
    finally:
        if window_created:
            try:
                cv2.destroyWindow(window)
                cv2.waitKey(1)
            except cv2.error:
                pass


class _TransferLiveDisplay:
    """Capture/infer in a thread; render with Qt in a separate process."""

    def __init__(
        self,
        observer: CameraRackObserver,
        rack_id: str,
        device: str,
    ) -> None:
        self._observer = observer
        self._rack_id = rack_id
        self._device = device
        self._window = f"{rack_id} transfer recording | wrist camera"
        self._state_lock = threading.RLock()
        self._moving = False
        self._stop = threading.Event()
        self._error: BaseException | None = None
        context = multiprocessing.get_context("spawn")
        self._frames = context.Queue(maxsize=2)
        self._process_ready = context.Event()
        self._process_errors = context.Queue(maxsize=1)
        self._process = context.Process(
            target=_transfer_display_process,
            args=(
                self._window,
                self._frames,
                self._process_ready,
                self._process_errors,
            ),
            name="transfer-display-window",
            daemon=True,
        )
        self._thread = threading.Thread(
            target=self._capture_loop,
            name="transfer-display-capture",
            daemon=True,
        )

    def start(self) -> None:
        self._process.start()
        if not self._process_ready.wait(timeout=30.0):
            self._stop.set()
            raise VisionError("录像显示进程启动超时")
        process_error = self._read_process_error()
        if process_error:
            raise VisionError(f"录像窗口启动失败：{process_error}")
        if not self._process.is_alive():
            raise VisionError("录像显示进程意外退出")
        self._thread.start()

    def set_moving(self, moving: bool) -> None:
        # Holding this lock through stationary inference guarantees a motion
        # command waits for the current YOLO call to finish. Once moving=True,
        # no new inference can start until the reached-pose check completes.
        with self._state_lock:
            process_error = self._read_process_error()
            if process_error:
                raise VisionError(f"录像窗口已停止：{process_error}")
            if self._error is not None:
                raise VisionError(f"录像窗口已停止：{self._error}")
            if not self._process.is_alive():
                raise VisionError("录像显示进程已退出")
            self._moving = bool(moving)

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=30.0)
        self._clear_frame_queue()
        try:
            self._frames.put_nowait(None)
        except queue.Full:
            pass
        if self._process.is_alive():
            self._process.join(timeout=10.0)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5.0)
        self._frames.close()
        self._process_errors.close()

    def _capture_loop(self) -> None:
        try:
            while not self._stop.is_set():
                if not self._process.is_alive():
                    detail = self._read_process_error() or "显示进程意外退出"
                    raise VisionError(detail)
                with self._state_lock:
                    moving = self._moving
                    if not moving:
                        frame, visual = self._observer.capture_transfer_display(
                            self._rack_id,
                            infer=True,
                        )
                    else:
                        frame = None
                        visual = None
                if moving:
                    frame, _ = self._observer.capture_transfer_display(
                        self._rack_id,
                        infer=False,
                    )

                if frame is None:
                    continue
                preview = np.asarray(frame.color).copy()
                if visual is not None:
                    _draw_live_scan_overlay(preview, visual)
                    status = (
                        f"CUDA:{self._device} LIVE | corners="
                        f"{len(visual.display_corners)} slots="
                        f"{len(visual.projected_slots)} caps={len(visual.caps)}"
                    )
                    color = (0, 220, 0)
                else:
                    status = "ARM MOVING | YOLO PAUSED | RAW WRIST VIEW"
                    color = (0, 165, 255)
                cv2.putText(
                    preview,
                    status,
                    (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.52,
                    color,
                    1,
                    cv2.LINE_AA,
                )
                success, encoded = cv2.imencode(
                    ".jpg",
                    preview,
                    (cv2.IMWRITE_JPEG_QUALITY, 90),
                )
                if not success:
                    raise VisionError("录像画面 JPEG 编码失败")
                self._send_latest(encoded.tobytes())
        except BaseException as exc:
            self._error = exc

    def _send_latest(self, encoded: bytes) -> None:
        try:
            self._frames.put_nowait(encoded)
            return
        except queue.Full:
            pass
        try:
            self._frames.get_nowait()
        except queue.Empty:
            pass
        try:
            self._frames.put_nowait(encoded)
        except queue.Full:
            pass

    def _clear_frame_queue(self) -> None:
        while True:
            try:
                self._frames.get_nowait()
            except queue.Empty:
                return

    def _read_process_error(self) -> str:
        try:
            return str(self._process_errors.get_nowait())
        except queue.Empty:
            return ""


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
    live_display: _TransferLiveDisplay | None = None
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
            need_camera=runtime.mode == "real",
            need_gripper=True,
        )
        if runtime.mode == "real":
            if not isinstance(runtime.observer, CameraRackObserver):
                raise VisionError("真机 transfer 缺少相机观察器")
            live_display = _TransferLiveDisplay(
                runtime.observer,
                command.source.rack_id,
                str(config["vision"]["device"]),
            )
            live_display.start()
            runtime.workflow.set_motion_state_changed(live_display.set_moving)
        if runtime.mode == "real" and bool(
            config["motion"].get("confirm_each_step", False)
        ):
            runtime.workflow.set_step_confirmation(_confirm_transfer_step)
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
                    "直接按 Enter 允许移动到观察位，输入 q 取消："
                ).strip().lower()
            except EOFError as error:
                raise WorkflowError(
                    "observation-pose authorization input is unavailable"
                ) from error
            if empty_confirmation in {"q", "quit", "stop"}:
                raise WorkflowError(
                    "operator cancelled observation-pose motion"
                )
            if empty_confirmation:
                raise WorkflowError("观察位确认只接受空回车")
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
                authorization = input(
                    "直接按 Enter 开始执行，输入 q 取消："
                ).strip().lower()
            except EOFError as error:
                raise WorkflowError(
                    "motion authorization input is unavailable"
                ) from error
            if authorization in {"q", "quit", "stop"}:
                raise WorkflowError("operator cancelled transfer motion")
            if authorization:
                raise WorkflowError("执行确认只接受空回车")

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
        runtime.workflow.set_motion_state_changed(None)
        if live_display is not None:
            live_display.stop()
        runtime.close()


def _confirm_transfer_step(description: str) -> None:
    print(f"\n下一步：{description}")
    try:
        answer = input(
            "检查路径和现场；直接按 Enter 继续，输入 q 后回车取消："
        ).strip().lower()
    except EOFError as error:
        raise WorkflowError("逐步确认输入不可用，已停止") from error
    if answer in {"q", "quit", "stop"}:
        raise WorkflowError("操作员取消了逐步执行")
    if answer:
        raise WorkflowError("逐步确认只接受空回车；已停止")


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


def _validate_model_contracts(cap_path: object, screw_path: object) -> None:
    """Load weight metadata only; inference remains a separate CUDA check."""
    from ultralytics import YOLO

    cap = YOLO(str(cap_path), task="detect")
    cap_names = {int(key): str(value) for key, value in dict(cap.names).items()}
    if cap_names != {0: "item"}:
        raise ConfigError(f"cap classes are {cap_names}, expected {{0: 'item'}}")

    screw = YOLO(str(screw_path), task="detect")
    screw_names = {
        int(key): str(value) for key, value in dict(screw.names).items()
    }
    if screw_names != {0: "item"}:
        raise ConfigError(
            f"screw classes are {screw_names}, expected {{0: 'item'}}"
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
