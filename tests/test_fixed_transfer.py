from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from tube_grabber.core.errors import ConfigError, WorkflowError
from tube_grabber.core.models import Pose6D
from tube_grabber.fakes.hardware import FakeArm, FakeGripper
from tube_grabber.fixed_transfer import (
    FixedTransferProgram,
    FixedTransferWorkflow,
    load_fixed_transfer_program,
)
from tube_grabber.motion.executor import MotionExecutor


def _program(*, confirmed: bool = True) -> FixedTransferProgram:
    poses = {
        "home": Pose6D(0, 0, 100, 0, 0, 0),
        "pick_above": Pose6D(50, 0, 100, 0, 0, 0),
        "pick": Pose6D(50, 0, 50, 0, 0, 0),
        "basket_above": Pose6D(100, 50, 100, 0, 0, 0),
        "basket_release": Pose6D(100, 50, 60, 0, 0, 0),
    }
    return FixedTransferProgram(
        confirmed=confirmed,
        poses=poses,
        transit_speed_percent=10,
        approach_speed_percent=5,
        start_position_tolerance_mm=2,
        start_orientation_tolerance_deg=1,
    )


def _workflow(program: FixedTransferProgram):
    arm = FakeArm(program.home)
    gripper = FakeGripper()
    arm.connect()
    gripper.setup()
    executor = MotionExecutor(
        arm,
        workspace_min_mm=(-10, -10, 0),
        workspace_max_mm=(150, 100, 150),
        maximum_single_move_mm=150,
        position_reached_tolerance_mm=1,
        orientation_reached_tolerance_deg=1,
    )
    return arm, gripper, FixedTransferWorkflow(
        arm=arm,
        gripper=gripper,
        executor=executor,
        program=program,
    )


class FixedTransferTests(unittest.TestCase):
    def test_program_builds_guarded_three_phase_route(self) -> None:
        phases = _program().plans()

        self.assertEqual([name for name, _ in phases], [
            "pick_approach", "carry_and_place", "retreat_and_home"
        ])
        waypoints = [waypoint for _, plan in phases for waypoint in plan.waypoints]
        self.assertEqual(
            [waypoint.name for waypoint in waypoints],
            [
                "pick_above", "pick", "pick_retreat", "basket_above",
                "basket_release", "basket_retreat", "home",
            ],
        )
        self.assertEqual(
            [waypoint.linear for waypoint in waypoints],
            [False, True, True, False, True, True, False],
        )

    def test_execute_orders_gripper_actions_and_returns_home(self) -> None:
        program = _program()
        arm, gripper, workflow = _workflow(program)

        workflow.execute()

        self.assertEqual(
            gripper.actions,
            ["setup", "open_for_pick", "grip", "release"],
        )
        self.assertEqual(arm.pose, program.home)
        self.assertEqual(len(arm.moves), 7)

    def test_unconfirmed_program_never_moves(self) -> None:
        arm, gripper, workflow = _workflow(_program(confirmed=False))

        with self.assertRaises(ConfigError):
            workflow.execute()

        self.assertEqual(arm.moves, [])
        self.assertEqual(gripper.actions, ["setup"])

    def test_wrong_start_never_operates_gripper_or_arm(self) -> None:
        program = _program()
        arm, gripper, workflow = _workflow(program)
        arm.pose = Pose6D(10, 0, 100, 0, 0, 0)

        with self.assertRaisesRegex(WorkflowError, "not at fixed-transfer home"):
            workflow.execute()

        self.assertEqual(arm.moves, [])
        self.assertEqual(gripper.actions, ["setup"])
        self.assertEqual(arm.stop_count, 1)

    def test_complete_route_is_validated_before_gripper_opens(self) -> None:
        program = _program()
        arm, gripper, workflow = _workflow(program)
        workflow.executor.workspace_max_mm = (80, 100, 150)

        with self.assertRaisesRegex(Exception, "outside"):
            workflow.execute()

        self.assertEqual(arm.moves, [])
        self.assertEqual(gripper.actions, ["setup"])

    def test_loader_rejects_placeholder_nulls(self) -> None:
        content = Path("config/fixed_transfer.yaml").read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixed.yaml"
            path.write_text(content, encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "finite numbers"):
                load_fixed_transfer_program(path)


if __name__ == "__main__":
    unittest.main()
