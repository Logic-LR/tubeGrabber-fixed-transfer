#!/usr/bin/env python3
"""One-shot guarded retreat from a rack-normal pick pose.

This recovery command never changes the gripper and never moves laterally to
the observation pose.  It only keeps the current flange orientation and moves
the TCP along the configured physical rack-up direction.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tube_grabber.app import build_runtime
from tube_grabber.config import load_config
from tube_grabber.core.errors import TubeGrabberError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="低速沿架面法向安全撤离")
    parser.add_argument("--distance-mm", type=float, default=150.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config()
    if str(config["runtime"]["mode"]) != "real":
        print("错误：该恢复脚本只允许 runtime.mode=real", file=sys.stderr)
        return 2
    if not 50.0 <= float(args.distance_mm) <= 200.0:
        print("错误：distance-mm 必须在 50..200 mm", file=sys.stderr)
        return 2

    runtime = build_runtime(config, load_calibrations=False)
    try:
        runtime.start(need_arm=True, need_camera=False, need_gripper=False)
        arm = runtime.arm
        robot = getattr(arm, "sdk_robot", None)
        if robot is None or not hasattr(robot, "rm_get_rm_plus_state_info"):
            raise RuntimeError("无法只读确认 RM Plus 夹爪状态")
        result = robot.rm_get_rm_plus_state_info()
        if not isinstance(result, tuple) or len(result) < 2 or result[0] != 0:
            raise RuntimeError(f"读取夹爪状态失败：{result!r}")
        state = result[1]
        position = int(state["pos"][0])
        force = int(state["force"][0])
        open_position = int(config["gripper"]["open_position"])
        tolerance = int(config["gripper"]["position_tolerance"])
        if abs(position - open_position) > tolerance or force != 0:
            raise RuntimeError(
                "夹爪不是已确认的空载张开状态："
                f"position={position}, force={force}"
            )

        current = arm.get_pose()
        axis = tuple(
            float(value)
            for value in config["vision"]["plane"]["expected_normal_base"]
        )
        plan = runtime.workflow.planner.plan_retreat(current, axis)
        # The standard retreat is 150 mm.  A non-default recovery distance is
        # implemented by scaling the sole endpoint from the current flange.
        if abs(float(args.distance_mm) - 150.0) > 1e-9:
            raise RuntimeError("当前恢复只允许已验证的 150 mm 撤离")
        target = plan.waypoints[0].pose
        runtime.workflow.executor.validate(plan, current)
        print(
            "当前法兰："
            f"[{current.x_mm:.2f}, {current.y_mm:.2f}, {current.z_mm:.2f}, "
            f"{current.rx_rad:.4f}, {current.ry_rad:.4f}, {current.rz_rad:.4f}]"
        )
        print(f"夹爪：空载张开 position={position}, force={force}")
        print(
            "撤离目标（3% 直线）："
            f"[{target.x_mm:.2f}, {target.y_mm:.2f}, {target.z_mm:.2f}, "
            f"{target.rx_rad:.4f}, {target.ry_rad:.4f}, {target.rz_rad:.4f}]"
        )
        answer = input(
            "确认撤离方向无障碍、急停可触达；输入 RETREAT 并回车："
        ).strip()
        if answer != "RETREAT":
            print("未授权撤离；没有发送运动。")
            return 2
        runtime.workflow.executor.execute(plan)
        reached = arm.get_pose()
        print(
            "撤离完成："
            f"[{reached.x_mm:.2f}, {reached.y_mm:.2f}, {reached.z_mm:.2f}] mm"
        )
        return 0
    except (TubeGrabberError, RuntimeError, ValueError) as error:
        try:
            runtime.arm.stop()
        except Exception:
            pass
        print(f"错误：{error}", file=sys.stderr)
        return 2
    finally:
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
