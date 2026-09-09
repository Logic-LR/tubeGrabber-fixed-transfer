from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from tube_grabber.core.errors import HardwareError, VisionError, WorkflowError
from tube_grabber.core.models import Pose6D, SlotAddress, TransferCommand
from tube_grabber.mobile_transfer import (
    MobileTransferCoordinator,
    MobileTransferProgram,
    load_mobile_pick_home_program,
    load_mobile_transfer_program,
)
from tube_grabber.navigation import ChassisMove, FakeChassis, WooshHelperChassis
from tests import test_workflow as workflow_fixtures


def _program() -> MobileTransferProgram:
    return MobileTransferProgram(
        confirmed=True,
        source_rack="rack_1",
        destination_rack="rack_2",
        loaded_observation_pose=Pose6D(0, 0, 90, 3.141592653589793, 0, 0),
        loaded_observation_confirmed=True,
        chassis_move=ChassisMove(
            rotation_deg=180,
            translation_x_m=0.5,
            rotation_speed_radps=0.08,
            translation_speed_mps=0.03,
            yaw_tolerance_deg=1,
            position_tolerance_m=0.012,
            maximum_rotation_translation_m=0.03,
            timeout_s=120,
        ),
        rotate_helper_path="/tmp/rotate",
        pose_servo_path="/tmp/servo",
    )


def _coordinator(*, chassis: FakeChassis | None = None):
    workflow, arm, gripper, observer = (
        workflow_fixtures.ManipulationWorkflowTests().build_workflow(
            workflow_fixtures.rack_observation()
        )
    )
    observer.observations["rack_2"] = workflow_fixtures.rack_observation("rack_2")
    runtime = SimpleNamespace(
        workflow=workflow,
        arm=arm,
        gripper=gripper,
        observer=observer,
    )
    base = chassis or FakeChassis()
    return (
        MobileTransferCoordinator(
            runtime=runtime,
            chassis=base,
            program=_program(),
        ),
        workflow,
        gripper,
        observer,
        base,
    )


class MobileTransferTests(unittest.TestCase):
    def test_pick_to_home_stops_before_chassis_and_keeps_gripper_closed(self) -> None:
        coordinator, workflow, gripper, observer, chassis = _coordinator()
        source = SlotAddress("rack_1", 1, 1)

        final = coordinator.pick_to_home(source)

        self.assertTrue(workflow.holding_tube)
        self.assertEqual(chassis.moves, [])
        self.assertEqual(observer.calls, ["rack_1"] * 3)
        self.assertEqual(gripper.actions, ["setup", "open_for_pick", "grip"])
        self.assertEqual(final.slot(source).occupancy.value, "empty")

    def test_auto_source_selects_the_only_occupied_slot(self) -> None:
        coordinator, workflow, gripper, observer, chassis = _coordinator()

        source, final = coordinator.pick_detected_to_home()

        self.assertEqual(source, SlotAddress("rack_1", 1, 1))
        self.assertTrue(workflow.holding_tube)
        self.assertEqual(chassis.moves, [])
        self.assertEqual(observer.calls, ["rack_1"] * 3)
        self.assertEqual(gripper.actions, ["setup", "open_for_pick", "grip"])
        self.assertEqual(final.slot(source).occupancy.value, "empty")

    def test_auto_source_retries_visual_failures_without_repeating_motion(self) -> None:
        coordinator, workflow, gripper, observer, chassis = _coordinator()
        coordinator.retry_visual_failures = True
        ordinary_observe = observer.observe_rack_for_task
        attempts = 0

        def flaky_observe(rack_id, *, ignored_elevated_slot=None):
            nonlocal attempts
            attempts += 1
            if attempts in (1, 3):
                raise VisionError("transient camera frame")
            return ordinary_observe(
                rack_id,
                ignored_elevated_slot=ignored_elevated_slot,
            )

        observer.observe_rack_for_task = flaky_observe

        source, final = coordinator.pick_detected_to_home()

        self.assertEqual(source, SlotAddress("rack_1", 1, 1))
        self.assertEqual(attempts, 4)
        self.assertTrue(workflow.holding_tube)
        self.assertEqual(gripper.actions, ["setup", "open_for_pick", "grip"])
        self.assertEqual(final.slot(source).occupancy.value, "empty")

    def test_auto_source_rejects_multiple_occupied_slots(self) -> None:
        workflow, arm, gripper, observer = (
            workflow_fixtures.ManipulationWorkflowTests().build_workflow(
                workflow_fixtures.rack_observation(occupied={(1, 1), (1, 2)})
            )
        )
        coordinator = MobileTransferCoordinator(
            runtime=SimpleNamespace(
                workflow=workflow,
                arm=arm,
                gripper=gripper,
                observer=observer,
            ),
            chassis=None,
            program=_program(),
        )

        with self.assertRaisesRegex(WorkflowError, "found 2"):
            coordinator.pick_detected_to_home()

        self.assertFalse(workflow.holding_tube)
        self.assertEqual(gripper.actions, ["setup"])

    def test_complete_mobile_cycle_rescans_both_racks(self) -> None:
        coordinator, workflow, gripper, observer, chassis = _coordinator()
        command = TransferCommand(
            SlotAddress("rack_1", 1, 1),
            SlotAddress("rack_2", 1, 2),
        )

        final = coordinator.execute(command)

        self.assertFalse(workflow.holding_tube)
        self.assertEqual(len(chassis.moves), 1)
        self.assertEqual(observer.calls, ["rack_1"] * 3 + ["rack_2"] * 3)
        self.assertEqual(
            gripper.actions,
            ["setup", "open_for_pick", "grip", "release"],
        )
        self.assertEqual(final.slot(command.destination).occupancy.value, "occupied")

    def test_chassis_failure_preserves_held_tube_and_never_releases(self) -> None:
        coordinator, workflow, gripper, _, chassis = _coordinator(
            chassis=FakeChassis(fail=True)
        )

        with self.assertRaisesRegex(HardwareError, "chassis"):
            coordinator.execute(
                TransferCommand(
                    SlotAddress("rack_1", 1, 1),
                    SlotAddress("rack_2", 1, 2),
                )
            )

        self.assertTrue(workflow.holding_tube)
        self.assertNotIn("release", gripper.actions)
        self.assertGreaterEqual(chassis.stop_count, 1)

    def test_failed_source_empty_check_prevents_chassis_motion(self) -> None:
        coordinator, workflow, gripper, observer, chassis = _coordinator()
        observer.record_pick = lambda _address: None  # type: ignore[method-assign]

        with self.assertRaisesRegex(WorkflowError, "pick verification failed"):
            coordinator.execute(
                TransferCommand(
                    SlotAddress("rack_1", 1, 1),
                    SlotAddress("rack_2", 1, 2),
                )
            )

        self.assertTrue(workflow.holding_tube)
        self.assertEqual(chassis.moves, [])
        self.assertNotIn("release", gripper.actions)

    def test_template_rejects_unmeasured_translation(self) -> None:
        content = Path("config/mobile_transfer.yaml").read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mobile.yaml"
            path.write_text(content, encoding="utf-8")
            with self.assertRaisesRegex(Exception, "invalid|finite|float"):
                load_mobile_transfer_program(path)

    def test_pick_home_loader_allows_unmeasured_chassis_translation(self) -> None:
        content = Path("config/mobile_transfer.yaml").read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mobile.yaml"
            path.write_text(content, encoding="utf-8")

            program = load_mobile_pick_home_program(path)

        self.assertTrue(program.confirmed)
        self.assertEqual(program.source_rack, "rack_1")

    def test_full_loader_uses_deployed_helper_key_names(self) -> None:
        content = Path("config/mobile_transfer.yaml").read_text(encoding="utf-8")
        content = content.replace("translation_x_m: null", "translation_x_m: 0.5")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mobile.yaml"
            path.write_text(content, encoding="utf-8")

            program = load_mobile_transfer_program(path)

        self.assertEqual(
            program.rotate_helper_path,
            "/home/rm/agv_debug_tools/grabber_rotate_relative",
        )
        self.assertEqual(
            program.pose_servo_path,
            "/home/rm/agv_debug_tools/grabber_pose_servo",
        )

    def test_woosh_move_splits_turn_then_translates(self) -> None:
        class RecordingWoosh(WooshHelperChassis):
            def __init__(self) -> None:
                super().__init__(rotate_helper_path="rotate", pose_servo_path="servo")
                self.commands: list[list[str]] = []

            def _require_helpers(self) -> None:
                return

            def _run(self, command, timeout_s, *, require_pose=True):
                self.commands.append(list(command))
                return "pose x=0 y=0 theta=0"

        chassis = RecordingWoosh()
        chassis.move_to_destination(_program().chassis_move)

        self.assertEqual(
            [item[0] for item in chassis.commands],
            ["rotate", "rotate", "servo", "servo"],
        )
        for command in chassis.commands[:2]:
            angle = float(command[command.index("--rotate-relative-rad") + 1])
            self.assertAlmostEqual(angle, 3.141592653589793 / 2.0)
        translations = [
            float(command[command.index("--dx-m") + 1])
            for command in chassis.commands[2:]
        ]
        self.assertEqual(translations, [0.25, 0.25])


if __name__ == "__main__":
    unittest.main()
