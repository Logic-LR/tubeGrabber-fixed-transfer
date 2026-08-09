"""Single command-line entry point for fake runs and staged lab operation."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from typing import Any, Sequence

import numpy as np

from tube_grabber.agent import build_command_agent
from tube_grabber.app import CameraRackObserver, TubeGrabberRuntime, build_runtime
from tube_grabber.config import load_config, load_yaml, project_path
from tube_grabber.core.errors import ConfigError, TubeGrabberError, WorkflowError
from tube_grabber.core.models import TransferCommand
from tube_grabber.core.parsing import parse_transfer
from tube_grabber.diagnostics import (
    format_observation,
    format_prepared_transfer,
    save_camera_frame,
    save_observation_image,
)
from tube_grabber.vision.geometry import validate_transform


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == "doctor":
            return _doctor(config)

        runtime = build_runtime(config)
        if args.command == "arm-status":
            return _arm_status(runtime)
        if args.command == "camera-check":
            return _camera_check(runtime)
        if args.command == "scan":
            return _scan(runtime, config, args.rack)
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
    subparsers.add_parser("doctor", help="静态检查配置、依赖、模型和标定门槛")
    subparsers.add_parser("arm-status", help="只连接右臂并读取当前法兰位姿")
    subparsers.add_parser("camera-check", help="只采集一帧 D435 彩色图和深度图")

    scan = subparsers.add_parser("scan", help="识别一个固定站的 2x6 试管架")
    scan.add_argument("--rack", required=True, choices=("rack_1", "rack_2"))

    plan = subparsers.add_parser(
        "plan-transfer",
        help="视觉定位并打印全部航点，不初始化夹爪、不发送运动",
    )
    _add_transfer_arguments(plan)
    transfer = subparsers.add_parser("transfer", help="执行同一机架内的一次抓放")
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
        help="Agent 解析文本，经确定性校验和安全门禁后执行同架抓放",
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

    model_path = project_path(config["vision"]["model_path"])
    if model_path.is_file():
        _status("OK", f"YOLO 模型存在：{model_path}")
    else:
        level = "FAIL" if mode == "real" else "WAIT"
        _status(level, f"新两类别 YOLO 模型尚不存在：{model_path}")
        if mode == "real":
            failures.append("model")

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
        if rack.get("fallback_plane_z_mm") is None:
            level = "FAIL" if mode == "real" else "WAIT"
            _status(
                level,
                f"{rack_id} 未填写 fallback_plane_z_mm；全空机架将拒绝定位",
            )
            if mode == "real":
                failures.append(f"{rack_id} plane calibration")
        else:
            _status("OK", f"{rack_id} 已填写架面 Z 标定")

    if failures:
        print("预检未通过：" + ", ".join(failures))
        return 2
    if mode == "fake":
        print("假硬件闭环可运行；WAIT 项是切换 real 前必须处理的真机条件。")
    else:
        print("静态真机预检通过；仍需依次运行 arm-status、camera-check、scan。")
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
            print(
                f"controller: mode={'real' if run_mode == 1 else 'simulation'} "
                f"power={'on' if power_state == 1 else 'off'}"
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
        and config["racks"][command.source.rack_id].get(
            "fallback_plane_z_mm"
        )
        is None
    ):
        raise ConfigError(
            f"{command.source.rack_id} 尚未标定 fallback_plane_z_mm；"
            "真实运动被锁定"
        )
    try:
        runtime.start(
            need_arm=True,
            need_camera=runtime.mode == "real",
            need_gripper=False,
        )
        runtime.require_observation_pose()
        runtime.require_motion_ready()
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
                "确认左臂已收回、底盘锁定、夹爪为空、急停可触达。"
            )
            try:
                authorization = input("输入 MOVE 并回车开始：").strip()
            except EOFError as error:
                raise WorkflowError("motion authorization input is unavailable") from error
            if authorization != "MOVE":
                raise WorkflowError("operator did not authorize motion")

        runtime.start(
            need_arm=True,
            need_camera=runtime.mode == "real",
            need_gripper=True,
        )
        runtime.require_motion_ready()
        runtime.require_observation_pose()
        runtime.workflow.verify_prepared_scene(prepared)
        _save_scan_if_available(runtime, config)
        print("执行前复扫通过：K0、12 槽状态、目标坐标和架面均未变化。")
        runtime.workflow.execute_transfer(prepared)
        print(
            f"运动完成：{command.source.text} -> {command.destination.text}；"
            "尚未自动复扫，请回到观测位后运行 scan 验证结果。"
        )
        return 0
    except BaseException:
        _stop_quietly(runtime)
        if runtime.workflow.holding_tube:
            print(
                "警告：软件状态显示夹爪仍持有试管，请勿直接移动底盘。",
                file=sys.stderr,
            )
        raise
    finally:
        runtime.close()


def _require_local_transfer(command: TransferCommand) -> None:
    if command.source.rack_id != command.destination.rack_id:
        raise WorkflowError(
            "当前版本尚未接入导航，跨架搬运被锁定；只能执行同一 rack 内搬运"
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


def _status(level: str, message: str) -> None:
    print(f"[{level:<4}] {message}")


def _stop_quietly(runtime: TubeGrabberRuntime) -> None:
    try:
        runtime.arm.stop()
    except Exception:
        pass


if __name__ == "__main__":
    raise SystemExit(main())
