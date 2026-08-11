from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from tube_grabber.core.errors import HardwareError
from tube_grabber.core.models import Pose6D
from tube_grabber.hardware._realman_sdk import require_success, sdk_code
from tube_grabber.hardware.realman_arm import (
    RealManArm,
    _handle_id,
    _pose_from_sdk,
    _pose_to_sdk,
)
from tube_grabber.hardware.realman_gripper import RealManGripper
from tube_grabber.hardware.realsense_d435 import (
    _depth_to_mm,
    _read_color_intrinsics,
)


class _Handle:
    id = 7


class _FakeArm:
    class Robot:
        def __init__(self) -> None:
            self.calls: list[tuple] = []
            self.position = 170
            self.mode = 3
            self.current_force = 10

        def rm_set_gripper_position(
            self,
            position: int,
            block: bool,
            timeout: int,
        ) -> int:
            self.calls.append(("gripper_position", position, block, timeout))
            if position == 135:
                # A real tube can stop the jaws before the requested position.
                self.position = 150
                self.mode = 6
                self.current_force = 200
            else:
                self.position = position
                self.mode = 3
                self.current_force = 10
            return 0

        def rm_get_gripper_state(self) -> tuple[int, dict]:
            self.calls.append(("state",))
            return 0, {
                "enable_state": 1,
                "status": 1,
                "error": 0,
                "mode": self.mode,
                "current_force": self.current_force,
                "temperature": 30,
                "actpos": self.position,
            }

    def __init__(self) -> None:
        self.sdk_robot = self.Robot()


class _FakeRobot:
    def __init__(self, thread_mode: int) -> None:
        self.thread_mode = thread_mode
        self.calls: list[tuple] = []

    def rm_create_robot_arm(self, ip: str, port: int) -> _Handle:
        self.calls.append(("connect", ip, port))
        return _Handle()

    def rm_change_work_frame(self, name: str) -> tuple[int, str]:
        self.calls.append(("work_frame", name))
        return 0, "ok"

    def rm_change_tool_frame(self, name: str) -> tuple[int, str]:
        self.calls.append(("tool_frame", name))
        return 0, "ok"

    def rm_get_current_work_frame(self) -> tuple[int, dict]:
        self.calls.append(("get_work_frame",))
        return 0, {"frame_name": "Base"}

    def rm_get_current_tool_frame(self) -> tuple[int, dict]:
        self.calls.append(("get_tool_frame",))
        return 0, {"frame_name": "Arm_Tip"}

    def rm_get_current_arm_state(self) -> tuple[int, dict]:
        self.calls.append(("get_pose",))
        return 0, {"pose": [0.1, 0.2, 0.3, 1.0, 2.0, 3.0]}

    def rm_get_robot_info(self) -> tuple[int, dict]:
        self.calls.append(("robot_info",))
        return 0, {"arm_dof": 7, "arm_model": 0}

    def rm_get_arm_run_mode(self) -> tuple[int, int]:
        self.calls.append(("run_mode",))
        return 0, 1

    def rm_get_arm_power_state(self) -> tuple[int, int]:
        self.calls.append(("power_state",))
        return 0, 1

    def rm_get_controller_state(self) -> dict:
        self.calls.append(("controller_state",))
        return {"return_code": 0, "sys_err": 0}

    def rm_get_joint_err_flag(self) -> dict:
        self.calls.append(("joint_errors",))
        return {
            "return_code": 0,
            "err_flag": [0] * 7,
            "brake_state": [1] * 7,
        }

    def rm_movej_p(self, *args: object) -> tuple[int, str]:
        self.calls.append(("movej_p", *args))
        return 0, "ok"

    def rm_movel(self, *args: object) -> tuple[int, str]:
        self.calls.append(("movel", *args))
        return 0, "ok"

    def rm_set_arm_slow_stop(self) -> int:
        self.calls.append(("stop",))
        return 0

    def rm_delete_robot_arm(self) -> int:
        self.calls.append(("disconnect",))
        return 0


class HardwareHelpersTest(unittest.TestCase):
    def test_sdk_code_accepts_int_and_tuple(self) -> None:
        self.assertEqual(sdk_code(0, "test"), 0)
        self.assertEqual(sdk_code((0, {"value": 1}), "test"), 0)

    def test_sdk_error_contains_operation_and_code(self) -> None:
        with self.assertRaisesRegex(HardwareError, "rm_movej_p.*-4"):
            require_success("rm_movej_p", (-4, "timeout"))

    def test_handle_accepts_object_and_int(self) -> None:
        self.assertEqual(_handle_id(_Handle()), 7)
        self.assertEqual(_handle_id(8), 8)

    def test_pose_unit_conversion_is_only_at_sdk_boundary(self) -> None:
        pose = Pose6D(100.0, -200.0, 350.0, 1.0, 2.0, 3.0)
        sdk_pose = _pose_to_sdk(pose)
        self.assertEqual(sdk_pose, [0.1, -0.2, 0.35, 1.0, 2.0, 3.0])
        self.assertEqual(_pose_from_sdk(sdk_pose), pose)

    def test_pose_dict_return_is_supported(self) -> None:
        pose = _pose_from_sdk(
            {
                "position": {"x": 0.1, "y": 0.2, "z": 0.3},
                "euler": {"rx": 1.0, "ry": 2.0, "rz": 3.0},
            }
        )
        self.assertEqual(pose, Pose6D(100.0, 200.0, 300.0, 1.0, 2.0, 3.0))

    def test_realsense_depth_is_float32_millimetres(self) -> None:
        raw = np.array([[0, 250, 1000]], dtype=np.uint16)
        depth_mm = _depth_to_mm(raw, scale_mm=1.0)
        self.assertEqual(depth_mm.dtype, np.float32)
        np.testing.assert_array_equal(depth_mm, [[0.0, 250.0, 1000.0]])

    def test_arm_sends_blocking_joint_and_linear_pose_moves(self) -> None:
        sdk = SimpleNamespace(
            RoboticArm=_FakeRobot,
            rm_thread_mode_e=SimpleNamespace(RM_TRIPLE_MODE_E=3),
        )
        with patch("tube_grabber.hardware.realman_arm.import_module", return_value=sdk):
            arm = RealManArm(
                "169.254.128.19",
                work_frame="Base",
                reject_conflicting_processes=False,
            )
            arm.connect()
            self.assertEqual(arm.dof, 7)
            self.assertEqual(arm.get_run_mode(), 1)
            self.assertEqual(arm.get_power_state(), 1)
            arm.require_healthy()
            arm.move_pose(Pose6D(100.0, 200.0, 300.0, 1.0, 2.0, 3.0), 5)
            arm.move_pose(
                Pose6D(110.0, 200.0, 300.0, 1.0, 2.0, 3.0),
                4,
                linear=True,
            )
            robot = arm.sdk_robot

        self.assertIn(("work_frame", "Base"), robot.calls)
        self.assertIn(("tool_frame", "Arm_Tip"), robot.calls)
        self.assertIn(("controller_state",), robot.calls)
        self.assertIn(("joint_errors",), robot.calls)
        self.assertIn(
            ("movej_p", [0.1, 0.2, 0.3, 1.0, 2.0, 3.0], 5, 0, 0, 1),
            robot.calls,
        )
        self.assertIn(
            ("movel", [0.11, 0.2, 0.3, 1.0, 2.0, 3.0], 4, 0, 0, 1),
            robot.calls,
        )

    def test_brown_conrady_color_intrinsics_are_preserved(self) -> None:
        intrinsics = SimpleNamespace(
            fx=600.0,
            fy=600.0,
            ppx=640.0,
            ppy=360.0,
            model="distortion.brown_conrady",
            coeffs=[0.1, 0.0, 0.0, 0.0, 0.0],
        )
        profile = SimpleNamespace(get_intrinsics=lambda: intrinsics)
        frame = SimpleNamespace(
            profile=SimpleNamespace(as_video_stream_profile=lambda: profile)
        )

        result = _read_color_intrinsics(frame)
        self.assertEqual(result.distortion_model, "brown_conrady")
        self.assertEqual(
            result.distortion_coefficients,
            (0.1, 0.0, 0.0, 0.0, 0.0),
        )

        intrinsics.model = "distortion.inverse_brown_conrady"
        with self.assertRaisesRegex(HardwareError, "暂不支持"):
            _read_color_intrinsics(frame)

    def test_gripper_uses_two_finger_position_and_feedback(self) -> None:
        arm = _FakeArm()
        gripper = RealManGripper(arm)  # type: ignore[arg-type]
        with patch("tube_grabber.hardware.realman_gripper.time.sleep"):
            gripper.setup()
            gripper.open_for_pick()
            gripper.grip()
            gripper.release()
            gripper.reset()
        calls = arm.sdk_robot.calls
        self.assertEqual(calls[0], ("state",))
        commanded = [call for call in calls if call[0] == "gripper_position"]
        self.assertEqual([call[1] for call in commanded], [170, 135, 170, 1])
        self.assertTrue(all(call[2:] == (True, 5) for call in commanded))
        self.assertNotIn("follow_position", [call[0] for call in calls])
        self.assertGreaterEqual(sum(call[0] == "state" for call in calls), 13)


if __name__ == "__main__":
    unittest.main()
