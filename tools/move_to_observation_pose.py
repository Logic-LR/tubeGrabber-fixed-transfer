"""Guarded, low-speed move to the configured observation pose."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from importlib import import_module
import math
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tube_grabber.app import build_runtime
from tube_grabber.config import load_config
from tube_grabber.core.errors import TubeGrabberError
from tube_grabber.core.models import Pose6D
from tube_grabber.hardware._realman_sdk import call_sdk, require_success
from tube_grabber.motion.planner import rotation_distance_deg, rpy_to_rotation


def main() -> int:
    parser = argparse.ArgumentParser(
        description="低速移动右臂到 config/poses.yaml 的固定观察位",
    )
    parser.add_argument("--config", default="config/app.yaml")
    parser.add_argument(
        "--speed",
        type=int,
        default=5,
        help="运动速度百分比，限制为 1..5（默认 5）",
    )
    parser.add_argument(
        "--allow-orientation-over-limit",
        action="store_true",
        help="明确允许本次姿态变化超过配置门限；仍需输入 MOVE",
    )
    parser.add_argument(
        "--explicit-joint-ik",
        action="store_true",
        help=(
            "先用当前关节角显式求目标逆解并检查关节路径，再调用 rm_movej；"
            "用于控制器拒绝远距离 rm_movej_p 的情况，需输入 MOVE_JOINT"
        ),
    )
    args = parser.parse_args()
    if not 1 <= args.speed <= 5:
        parser.error("--speed 必须在 1..5")

    try:
        config = load_config(args.config)
        if config["runtime"]["mode"] != "real":
            raise RuntimeError("runtime.mode 必须为 real")
        runtime = build_runtime(config)
        if not runtime.observation_pose_confirmed:
            raise RuntimeError(
                "config/poses.yaml observation_pose.confirmed 必须为 true"
            )
        target = runtime.observation_pose
        _require_target_in_workspace(target, config["motion"])

        connected = False
        try:
            runtime.start(
                need_arm=True,
                need_camera=False,
                need_gripper=False,
            )
            connected = True
            arm = runtime.arm
            if not hasattr(arm, "get_run_mode") or arm.get_run_mode() != 1:
                raise RuntimeError("控制器不是真实模式")
            if not hasattr(arm, "get_power_state") or arm.get_power_state() != 1:
                raise RuntimeError("机械臂未上电")
            if not hasattr(arm, "require_healthy"):
                raise RuntimeError("机械臂驱动无法读取健康状态")
            arm.require_healthy()

            current = arm.get_pose()
            distance_mm = _distance_mm(current, target)
            orientation_deg = _orientation_deg(current, target)
            maximum_distance = float(config["motion"]["maximum_single_move_mm"])
            maximum_orientation = float(
                config["motion"]["maximum_single_orientation_change_deg"]
            )
            if distance_mm > maximum_distance:
                raise RuntimeError(
                    f"移动距离 {distance_mm:.1f} mm 超过门限 "
                    f"{maximum_distance:.1f} mm"
                )
            if (
                orientation_deg > maximum_orientation
                and not args.allow_orientation_over_limit
            ):
                raise RuntimeError(
                    f"姿态变化 {orientation_deg:.2f}° 超过门限 "
                    f"{maximum_orientation:.2f}°；确认直达路径安全后，"
                    "显式添加 --allow-orientation-over-limit"
                )

            print("当前位置：" + _format_pose(current))
            print("目标观察位：" + _format_pose(target))
            method_label = "显式 IK + rm_movej" if args.explicit_joint_ik else "movej_p"
            print(
                f"直达 {method_label}：距离={distance_mm:.1f} mm，"
                f"姿态变化={orientation_deg:.2f}°，速度={args.speed}%"
            )
            if orientation_deg > maximum_orientation:
                print(
                    "警告：本次姿态变化超过项目默认门限；"
                    "脚本不提供环境避障或中间航点。"
                )
            target_joints: list[float] | None = None
            if args.explicit_joint_ik:
                target_joints = _preflight_explicit_joint_path(arm, target)
                confirmation = "MOVE_JOINT"
            else:
                confirmation = "MOVE"
            try:
                answer = input(
                    "确认夹爪空载、上述扫掠空间无障碍、急停可触达；"
                    f"输入 {confirmation}："
                ).strip()
            except EOFError:
                print("\n标准输入已关闭，已取消，未发送运动。")
                return 1
            if answer != confirmation:
                print("已取消，未发送运动。")
                return 1

            if target_joints is None:
                arm.move_pose(target, speed_percent=args.speed, linear=False)
            else:
                _move_joints(arm, target_joints, args.speed)
            reached = runtime.require_observation_pose()
            print("已到观察位：" + _format_pose(reached))
            return 0
        except BaseException:
            if connected:
                try:
                    runtime.arm.stop()
                except Exception:
                    pass
            raise
        finally:
            runtime.close()
    except KeyboardInterrupt:
        print("\n已取消。", file=sys.stderr)
        return 130
    except (TubeGrabberError, RuntimeError, ValueError, KeyError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2


def _require_target_in_workspace(target: Pose6D, motion: dict[str, object]) -> None:
    minimum = tuple(float(value) for value in motion["workspace_min_mm"])
    maximum = tuple(float(value) for value in motion["workspace_max_mm"])
    for axis, value, lower, upper in zip(
        "XYZ",
        (target.x_mm, target.y_mm, target.z_mm),
        minimum,
        maximum,
    ):
        if not lower <= value <= upper:
            raise RuntimeError(
                f"观察位 {axis}={value:.1f} mm 超出工作空间 "
                f"[{lower:.1f}, {upper:.1f}] mm"
            )


def _distance_mm(first: Pose6D, second: Pose6D) -> float:
    return math.sqrt(
        (first.x_mm - second.x_mm) ** 2
        + (first.y_mm - second.y_mm) ** 2
        + (first.z_mm - second.z_mm) ** 2
    )


def _orientation_deg(first: Pose6D, second: Pose6D) -> float:
    return rotation_distance_deg(
        rpy_to_rotation((first.rx_rad, first.ry_rad, first.rz_rad)),
        rpy_to_rotation((second.rx_rad, second.ry_rad, second.rz_rad)),
    )


def _format_pose(pose: Pose6D) -> str:
    return (
        f"[{pose.x_mm:.3f}, {pose.y_mm:.3f}, {pose.z_mm:.3f}, "
        f"{pose.rx_rad:.6f}, {pose.ry_rad:.6f}, {pose.rz_rad:.6f}]"
    )


def _preflight_explicit_joint_path(arm: object, target: Pose6D) -> list[float]:
    """Solve and validate a joint-space path without commanding motion."""
    robot = getattr(arm, "sdk_robot", None)
    if robot is None:
        raise RuntimeError("机械臂驱动未开放 SDK，无法执行显式关节逆解")

    joint_result = call_sdk("读取当前关节角", robot.rm_get_joint_degree)
    require_success("读取当前关节角", joint_result)
    if not isinstance(joint_result, tuple) or len(joint_result) < 2:
        raise RuntimeError(f"当前关节角返回格式错误: {joint_result!r}")
    current = _finite_joint_vector(joint_result[1], arm.dof, "当前关节角")

    try:
        ctypes_wrap = import_module("Robotic_Arm.rm_ctypes_wrap")
        parameters_type = ctypes_wrap.rm_inverse_kinematics_params_t
    except (ImportError, AttributeError) as exc:
        raise RuntimeError("当前 RealMan SDK 缺少逆运动学参数类型") from exc

    sdk_pose = [
        target.x_mm / 1000.0,
        target.y_mm / 1000.0,
        target.z_mm / 1000.0,
        target.rx_rad,
        target.ry_rad,
        target.rz_rad,
    ]
    traversal = getattr(robot, "rm_algo_set_redundant_parameter_traversal_mode", None)
    if traversal is not None:
        traversal(True)
    ik_result = robot.rm_algo_inverse_kinematics(
        parameters_type(current, sdk_pose, 1)
    )
    if not isinstance(ik_result, tuple) or len(ik_result) < 2:
        raise RuntimeError(f"逆运动学返回格式错误: {ik_result!r}")
    if int(ik_result[0]) != 0:
        raise RuntimeError(f"目标观察位逆运动学失败，算法错误码 {ik_result[0]}")
    target_joints = _finite_joint_vector(
        ik_result[1], arm.dof, "目标关节角"
    )

    minimum = _finite_joint_vector(
        robot.rm_algo_get_joint_min_limit(), arm.dof, "关节下限"
    )
    maximum = _finite_joint_vector(
        robot.rm_algo_get_joint_max_limit(), arm.dof, "关节上限"
    )
    for index, (value, lower, upper) in enumerate(
        zip(target_joints, minimum, maximum), start=1
    ):
        if not lower <= value <= upper:
            raise RuntimeError(
                f"目标 J{index}={value:.3f}° 超出限位 "
                f"[{lower:.3f}, {upper:.3f}]°"
            )

    collision_check = getattr(
        robot, "rm_algo_safety_robot_self_collision_detection", None
    )
    if collision_check is None:
        raise RuntimeError("当前 SDK 缺少关节路径自碰撞检查")
    poses: list[list[float]] = []
    for sample_index in range(101):
        ratio = sample_index / 100.0
        sample = [
            first + (second - first) * ratio
            for first, second in zip(current, target_joints)
        ]
        if int(collision_check(sample)) != 0:
            raise RuntimeError(
                f"显式关节路径在 {sample_index}% 处触发自碰撞或关节限位"
            )
        pose = robot.rm_algo_forward_kinematics(sample, 1)
        if not isinstance(pose, Sequence) or len(pose) < 6:
            raise RuntimeError(f"路径正运动学返回格式错误: {pose!r}")
        poses.append([float(value) for value in pose[:6]])

    changes = [
        abs(second - first) for first, second in zip(current, target_joints)
    ]
    if max(changes) > 135.0:
        raise RuntimeError(
            f"显式关节路径最大单轴变化 {max(changes):.2f}° 超过 135° 门限"
        )
    print("当前关节角(°)：" + _format_vector(current, 3))
    print("目标关节角(°)：" + _format_vector(target_joints, 3))
    print("各轴变化(°)：" + _format_vector(changes, 3))
    xyz_mm = [[pose[axis] * 1000.0 for pose in poses] for axis in range(3)]
    print(
        "关节路径正解扫掠范围："
        + ", ".join(
            f"{axis}=[{min(values):.1f}, {max(values):.1f}] mm"
            for axis, values in zip("XYZ", xyz_mm)
        )
    )
    print(
        "注意：以上仅检查机器人自身碰撞和关节限位，"
        "无法识别试管架、桌面、相机线缆等现场障碍。"
    )
    return target_joints


def _move_joints(arm: object, joints: Sequence[float], speed: int) -> None:
    robot = getattr(arm, "sdk_robot", None)
    if robot is None:
        raise RuntimeError("机械臂驱动未开放 SDK，无法执行关节运动")
    result = call_sdk(
        "rm_movej 阻塞运动",
        robot.rm_movej,
        list(joints),
        int(speed),
        0,
        0,
        1,
    )
    try:
        require_success("rm_movej 阻塞运动", result)
    except TubeGrabberError as exc:
        raise RuntimeError(
            f"{exc}；目标关节角(°)={_format_vector(joints, 3)}"
        ) from exc


def _finite_joint_vector(
    values: object, expected: int, name: str
) -> list[float]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise RuntimeError(f"{name}格式错误: {values!r}")
    result = [float(value) for value in values]
    if len(result) != expected or not all(math.isfinite(value) for value in result):
        raise RuntimeError(f"{name}必须包含 {expected} 个有限数值: {values!r}")
    return result


def _format_vector(values: Sequence[float], decimals: int) -> str:
    return "[" + ", ".join(f"{float(value):.{decimals}f}" for value in values) + "]"


if __name__ == "__main__":
    raise SystemExit(main())
