from __future__ import annotations

import unittest
from unittest.mock import patch

from tube_grabber.app import build_runtime
from tube_grabber.cli import main
from tube_grabber.config import load_config
from tube_grabber.core.errors import HardwareError
from tube_grabber.core.models import Pose6D
from tube_grabber.core.parsing import parse_transfer


class ApplicationIntegrationTests(unittest.TestCase):
    def test_default_fake_runtime_completes_the_full_transfer(self) -> None:
        runtime = build_runtime(load_config())
        try:
            runtime.start(
                need_arm=True,
                need_camera=False,
                need_gripper=True,
            )
            runtime.require_observation_pose()
            runtime.workflow.transfer(
                parse_transfer("rack_1.r1c1", "rack_1.r1c2")
            )
            self.assertGreater(len(runtime.arm.moves), 0)  # type: ignore[attr-defined]
            # Pick/place segments are linear; the final return to the taught
            # observation pose deliberately uses RealMan movej_p.
            self.assertFalse(runtime.arm.moves[-1][2])  # type: ignore[attr-defined]
            self.assertEqual(
                runtime.gripper.actions,  # type: ignore[attr-defined]
                ["setup", "open_for_pick", "grip", "release"],
            )
        finally:
            runtime.close()

    def test_scan_refuses_a_wrong_observation_pose(self) -> None:
        runtime = build_runtime(load_config())
        try:
            runtime.start(
                need_arm=True,
                need_camera=False,
                need_gripper=False,
            )
            runtime.arm.pose = Pose6D(  # type: ignore[attr-defined]
                120.0,
                308.970,
                -172.253,
                -3.073,
                -0.031,
                -1.607,
            )
            with self.assertRaisesRegex(HardwareError, "observation pose"):
                runtime.require_observation_pose()
        finally:
            runtime.close()

    def test_runtime_moves_to_taught_observation_pose_automatically(self) -> None:
        runtime = build_runtime(load_config())
        try:
            runtime.start(
                need_arm=True,
                need_camera=False,
                need_gripper=False,
            )
            runtime.arm.pose = Pose6D(  # type: ignore[attr-defined]
                100.0,
                320.0,
                -150.0,
                -3.0,
                0.0,
                -1.6,
            )

            reached = runtime.move_to_observation_pose()

            self.assertEqual(reached, runtime.observation_pose)
            self.assertEqual(  # type: ignore[attr-defined]
                runtime.arm.moves[-1],
                (runtime.observation_pose, 10, False),
            )
        finally:
            runtime.close()

    def test_fake_closed_loop_supports_a_non_default_destination(self) -> None:
        runtime = build_runtime(load_config())
        try:
            runtime.start(
                need_arm=True,
                need_camera=False,
                need_gripper=True,
            )

            command = parse_transfer("rack_1.r1c1", "rack_1.r2c6")
            final = runtime.workflow.transfer(command)

            self.assertEqual(
                final.slot(command.destination).occupancy.value,
                "occupied",
            )
        finally:
            runtime.close()

    def test_doctor_passes_in_default_fake_mode(self) -> None:
        self.assertEqual(main(["doctor"]), 0)

    def test_cli_fake_transfer_runs_without_prompt(self) -> None:
        result = main(
            [
                "transfer",
                "--source",
                "rack_1.r1c1",
                "--destination",
                "rack_1.r1c2",
            ]
        )
        self.assertEqual(result, 0)

    def test_plan_transfer_does_not_initialize_or_move_hardware(self) -> None:
        runtime = build_runtime(load_config())
        try:
            runtime.start(
                need_arm=True,
                need_camera=False,
                need_gripper=False,
            )
            prepared = runtime.workflow.prepare_transfer(
                parse_transfer("rack_1.r1c1", "rack_1.r1c2")
            )
            self.assertEqual(prepared.command.source.text, "rack_1.r1c1")
            self.assertEqual(runtime.arm.moves, [])  # type: ignore[attr-defined]
            self.assertEqual(runtime.gripper.actions, [])  # type: ignore[attr-defined]
        finally:
            runtime.close()

    def test_agent_plan_uses_the_same_read_only_planning_path(self) -> None:
        result = main(
            [
                "agent-plan",
                "--text",
                "请执行 rack_1.r1c1 -> rack_1.r1c2",
            ]
        )
        self.assertEqual(result, 0)

    def test_agent_plan_incomplete_command_does_not_plan(self) -> None:
        result = main(["agent-plan", "--text", "把这个移动过去"])
        self.assertEqual(result, 2)

    def test_agent_plan_keeps_cross_rack_navigation_lock(self) -> None:
        result = main(
            [
                "agent-plan",
                "--text",
                "rack_1.r1c1 -> rack_2.r1c2",
            ]
        )
        self.assertEqual(result, 2)

    def test_agent_transfer_uses_the_existing_transfer_path(self) -> None:
        with patch("tube_grabber.cli._transfer", return_value=0) as transfer:
            result = main(
                [
                    "agent-transfer",
                    "--text",
                    "rack_1.r1c1 -> rack_1.r1c2",
                ]
            )

        self.assertEqual(result, 0)
        transfer.assert_called_once()
        command = transfer.call_args.args[2]
        self.assertEqual(command.source.text, "rack_1.r1c1")
        self.assertEqual(command.destination.text, "rack_1.r1c2")

    def test_agent_transfer_incomplete_command_never_executes(self) -> None:
        with patch("tube_grabber.cli._transfer") as transfer:
            result = main(["agent-transfer", "--text", "move this tube"])

        self.assertEqual(result, 2)
        transfer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
