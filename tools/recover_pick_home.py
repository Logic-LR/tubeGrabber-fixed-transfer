"""Recover a held tube after the validated retreat waypoint was interrupted."""

from __future__ import annotations

import argparse
import time

from tube_grabber.app import build_runtime
from tube_grabber.config import load_config
from tube_grabber.core.models import Pose6D


RETREAT = Pose6D(
    84.2,
    316.6,
    144.1,
    -2.343,
    0.025,
    -1.550,
    frame="base_right",
)
HOME = Pose6D(
    89.888,
    284.810,
    135.422,
    -2.310,
    0.024,
    -1.554,
    frame="base_right",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clear-only", action="store_true")
    args = parser.parse_args()
    config = load_config("config/app.yaml")
    config["runtime"]["mode"] = "real"
    runtime = build_runtime(config)
    arm = runtime.arm
    arm.connect()
    try:
        robot = arm.sdk_robot
        before = robot.rm_get_joint_err_flag()
        print(f"joint errors before clear: {before}")
        flags = tuple(int(value) for value in before.get("err_flag", ()))
        if len(flags) != arm.dof:
            raise RuntimeError(f"unexpected joint error response: {before!r}")
        for joint_num, error in enumerate(flags, start=1):
            if error:
                result = robot.rm_set_joint_clear_err(joint_num)
                print(f"clear joint {joint_num} error 0x{error:04X}: {result}")
                if result != 0:
                    raise RuntimeError(f"failed to clear joint {joint_num}: {result}")
        time.sleep(1.0)
        arm.require_healthy()
        if args.clear_only:
            print("controller healthy; no motion was sent")
            return 0
        print("controller healthy; moving to retreat at 5%")
        arm.move_pose(RETREAT, 5, linear=True)
        print("retreat complete; moving to held-tube home at 5%")
        arm.move_pose(HOME, 5, linear=True)
        print("recovery complete; gripper remains closed")
        return 0
    finally:
        arm.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
