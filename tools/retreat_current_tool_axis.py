"""Lift straight away from the rack from the arm's current tool pose."""

from __future__ import annotations

import argparse
import time

from tube_grabber.app import build_runtime
from tube_grabber.config import load_config
from tube_grabber.core.models import Pose6D
from tube_grabber.motion.planner import rpy_to_rotation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--distance-mm", type=float, default=150.0)
    args = parser.parse_args()
    if not 20.0 <= args.distance_mm <= 250.0:
        raise ValueError("distance must be in [20, 250] mm")

    config = load_config("config/app.yaml")
    config["runtime"]["mode"] = "real"
    runtime = build_runtime(config)
    arm = runtime.arm
    arm.connect()
    try:
        for joint_num, error_code in arm.clear_joint_errors():
            print(f"cleared joint {joint_num} error 0x{error_code:04X}")
        arm.require_healthy()
        current = arm.get_pose()
        rotation = rpy_to_rotation(
            (current.rx_rad, current.ry_rad, current.rz_rad)
        )
        away = tuple(-rotation[row][2] for row in range(3))
        target = Pose6D(
            current.x_mm + away[0] * args.distance_mm,
            current.y_mm + away[1] * args.distance_mm,
            current.z_mm + away[2] * args.distance_mm,
            current.rx_rad,
            current.ry_rad,
            current.rz_rad,
            current.frame,
        )
        print(f"current: {current}")
        print(f"linear retreat target: {target}")
        arm.move_pose(target, 5, linear=True)
        print("linear retreat complete")
        return 0
    finally:
        arm.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
