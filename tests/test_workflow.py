from __future__ import annotations

import unittest

from tube_grabber.core.errors import HardwareError, MotionError, WorkflowError
from tube_grabber.core.models import (
    Occupancy,
    Pixel,
    Point3D,
    Pose6D,
    RackObservation,
    SlotAddress,
    SlotObservation,
    TransferCommand,
)
from tube_grabber.fakes import FakeArm, FakeGripper, FakeRackObserver
from tube_grabber.motion import MotionExecutor, MotionPlanner, flange_to_tcp_point
from tube_grabber.workflow import ManipulationWorkflow


def rack_observation(
    rack_id: str = "rack_1",
    *,
    occupied: set[tuple[int, int]] | None = None,
    source_x_mm: float | None = None,
    destination_x_mm: float | None = None,
) -> RackObservation:
    if occupied is None:
        occupied = {(1, 1)}
    slots = []
    for row in (1, 2):
        for column in range(1, 7):
            address = SlotAddress(rack_id, row, column)
            x_mm = float((column - 1) * 20)
            if (row, column) == (1, 1) and source_x_mm is not None:
                x_mm = source_x_mm
            if (row, column) == (1, 2) and destination_x_mm is not None:
                x_mm = destination_x_mm
            y_mm = float((row - 1) * 20)
            is_occupied = (row, column) in occupied
            slots.append(
                SlotObservation(
                    address=address,
                    occupancy=(
                        Occupancy.OCCUPIED if is_occupied else Occupancy.EMPTY
                    ),
                    confidence=0.95,
                    pixel=Pixel(x_mm, y_mm),
                    cap_top_base=(
                        Point3D(x_mm, y_mm, 47.0) if is_occupied else None
                    ),
                    hole_on_plane_base=Point3D(x_mm, y_mm, 0.0),
                )
            )
    return RackObservation(
        rack_id=rack_id,
        marker=Pixel(0, 0),
        plane_z_mm=0.0,
        slots=tuple(slots),
        timestamp_ms=1.0,
    )


class ManipulationWorkflowTests(unittest.TestCase):
    def build_workflow(
        self,
        observation: RackObservation,
        *,
        fail_on_move_number: int | None = None,
        workspace_max_x_mm: float = 250.0,
    ) -> tuple[ManipulationWorkflow, FakeArm, FakeGripper, FakeRackObserver]:
        arm = FakeArm(
            Pose6D(0, 0, 90, 0, 0, 0),
            fail_on_move_number=fail_on_move_number,
        )
        arm.connect()
        gripper = FakeGripper()
        gripper.setup()
        observer = FakeRackObserver(observation)
        planner = MotionPlanner(
            tcp_offset_end_mm=(0, 0, 10),
            vertical_tool_rpy_rad=(0, 0, 0),
            approach_height_mm=20,
            retreat_height_mm=30,
            transit_speed_percent=10,
            approach_speed_percent=5,
            maximum_orientation_error_deg=5,
            maximum_single_orientation_change_deg=180,
            maximum_tool_tilt_deg=180,
            tube_total_length_mm=20,
            required_carried_clearance_mm=10,
        )
        executor = MotionExecutor(
            arm,
            workspace_min_mm=(-50, -50, -100),
            workspace_max_mm=(workspace_max_x_mm, 250, 200),
            maximum_single_move_mm=300,
            position_reached_tolerance_mm=1.0,
            orientation_reached_tolerance_deg=0.5,
        )
        workflow = ManipulationWorkflow(
            arm=arm,
            gripper=gripper,
            observer=observer,
            planner=planner,
            executor=executor,
            observation_pose=Pose6D(0, 0, 90, 0, 0, 0),
            grasp_depth_below_cap_mm=5,
            cap_top_above_rack_mm=47,
            seating_adjust_mm=2,
            scene_recheck_pixel_tolerance_px=5,
            scene_recheck_position_tolerance_mm=3,
            scene_recheck_plane_tolerance_mm=3,
        )
        return workflow, arm, gripper, observer

    def test_local_transfer_runs_complete_sequence(self) -> None:
        workflow, arm, gripper, observer = self.build_workflow(
            rack_observation()
        )
        command = TransferCommand(
            SlotAddress("rack_1", 1, 1),
            SlotAddress("rack_1", 1, 2),
        )
        observer.set_sequence(
            "rack_1",
            [
                rack_observation(),
                rack_observation(),
                rack_observation(occupied=set()),
                rack_observation(occupied={(1, 2)}),
            ],
        )

        final = workflow.transfer(command)

        self.assertFalse(workflow.holding_tube)
        self.assertIs(
            final.slot(SlotAddress("rack_1", 1, 1)).occupancy,
            Occupancy.EMPTY,
        )
        self.assertIs(
            final.slot(SlotAddress("rack_1", 1, 2)).occupancy,
            Occupancy.OCCUPIED,
        )
        self.assertEqual(observer.calls, ["rack_1"] * 4)
        self.assertEqual(
            gripper.actions,
            ["setup", "open_for_pick", "grip", "release"],
        )
        tcp_z_values = [
            flange_to_tcp_point(move[0], (0, 0, 10)).z_mm
            for move in arm.moves
        ]
        self.assertTrue(any(abs(value - 42.0) < 1e-6 for value in tcp_z_values))
        self.assertTrue(any(abs(value - 44.0) < 1e-6 for value in tcp_z_values))
        self.assertFalse(arm.moves[-1][2])
        self.assertEqual(arm.moves[-1][0], Pose6D(0, 0, 90, 0, 0, 0))

    def test_cross_rack_transfer_is_explicitly_rejected(self) -> None:
        workflow, arm, gripper, observer = self.build_workflow(
            rack_observation()
        )
        command = TransferCommand(
            SlotAddress("rack_1", 1, 1),
            SlotAddress("rack_2", 1, 2),
        )

        with self.assertRaisesRegex(WorkflowError, "navigation"):
            workflow.transfer(command)

        self.assertEqual(observer.calls, [])
        self.assertEqual(arm.moves, [])
        self.assertEqual(gripper.actions, ["setup"])

    def test_unknown_source_is_not_treated_as_occupied(self) -> None:
        observation = rack_observation(occupied=set())
        slots = list(observation.slots)
        slots[0] = SlotObservation(
            address=slots[0].address,
            occupancy=Occupancy.UNKNOWN,
            confidence=0.2,
            pixel=slots[0].pixel,
        )
        observation = RackObservation(
            observation.rack_id,
            observation.marker,
            observation.plane_z_mm,
            tuple(slots),
            observation.timestamp_ms,
        )
        workflow, arm, gripper, _ = self.build_workflow(observation)

        with self.assertRaisesRegex(WorkflowError, "must be occupied"):
            workflow.transfer(
                TransferCommand(
                    SlotAddress("rack_1", 1, 1),
                    SlotAddress("rack_1", 1, 2),
                )
            )

        self.assertEqual(arm.moves, [])
        self.assertEqual(gripper.actions, ["setup"])

    def test_unreachable_destination_is_rejected_before_gripping(self) -> None:
        observation = rack_observation(destination_x_mm=400.0)
        workflow, arm, gripper, _ = self.build_workflow(
            observation,
            workspace_max_x_mm=250.0,
        )

        with self.assertRaises(MotionError):
            workflow.transfer(
                TransferCommand(
                    SlotAddress("rack_1", 1, 1),
                    SlotAddress("rack_1", 1, 2),
                )
            )

        self.assertEqual(arm.moves, [])
        self.assertEqual(gripper.actions, ["setup"])

    def test_failed_retreat_preserves_payload_state(self) -> None:
        workflow, arm, gripper, _ = self.build_workflow(
            rack_observation(),
            fail_on_move_number=2,
        )
        observation = workflow.scan("rack_1")

        with self.assertRaises(MotionError):
            workflow.pick(SlotAddress("rack_1", 1, 1), observation)

        self.assertTrue(workflow.holding_tube)
        self.assertEqual(gripper.actions[-1], "grip")
        self.assertEqual(arm.stop_count, 1)

    def test_failed_grip_is_treated_as_possibly_holding_payload(self) -> None:
        class FailingGrip(FakeGripper):
            def grip(self) -> None:
                super().grip()
                raise HardwareError("gripper reply lost")

        workflow, _, _, _ = self.build_workflow(rack_observation())
        gripper = FailingGrip()
        gripper.setup()
        workflow.gripper = gripper
        observation = workflow.scan("rack_1")

        with self.assertRaises(HardwareError):
            workflow.pick(SlotAddress("rack_1", 1, 1), observation)

        self.assertTrue(workflow.holding_tube)

    def test_prepared_transfer_is_rejected_if_arm_moved(self) -> None:
        workflow, arm, gripper, _ = self.build_workflow(rack_observation())
        prepared = workflow.prepare_transfer(
            TransferCommand(
                SlotAddress("rack_1", 1, 1),
                SlotAddress("rack_1", 1, 2),
            )
        )
        arm.pose = Pose6D(arm.pose.x_mm + 2.0, arm.pose.y_mm, arm.pose.z_mm, 0, 0, 0)

        with self.assertRaisesRegex(WorkflowError, "moved after"):
            workflow.execute_transfer(prepared)

        self.assertEqual(arm.moves, [])
        self.assertEqual(gripper.actions, ["setup"])

    def test_prepared_transfer_is_rejected_if_scene_changed(self) -> None:
        workflow, arm, gripper, observer = self.build_workflow(
            rack_observation()
        )
        prepared = workflow.prepare_transfer(
            TransferCommand(
                SlotAddress("rack_1", 1, 1),
                SlotAddress("rack_1", 1, 2),
            )
        )
        observer.observations["rack_1"] = rack_observation(
            occupied={(1, 1), (1, 2)}
        )

        with self.assertRaisesRegex(WorkflowError, "changed from"):
            workflow.verify_prepared_scene(prepared)

        self.assertEqual(arm.moves, [])
        self.assertEqual(gripper.actions, ["setup"])

    def test_refresh_rebuilds_pick_plan_from_latest_coordinates(self) -> None:
        workflow, _, _, observer = self.build_workflow(rack_observation())
        command = TransferCommand(
            SlotAddress("rack_1", 1, 1),
            SlotAddress("rack_1", 1, 2),
        )
        preview = workflow.prepare_transfer(command)
        observer.observations["rack_1"] = rack_observation(source_x_mm=2.0)

        refreshed = workflow.refresh_prepared_transfer(preview)

        self.assertEqual(preview.source_target.x_mm, 0.0)
        self.assertEqual(refreshed.source_target.x_mm, 2.0)
        self.assertEqual(
            flange_to_tcp_point(
                refreshed.pick_approach.waypoints[-1].pose,
                (0, 0, 10),
            ).x_mm,
            2.0,
        )

    def test_destination_recheck_replans_place_from_latest_coordinate(self) -> None:
        workflow, arm, _, observer = self.build_workflow(rack_observation())
        observer.set_sequence(
            "rack_1",
            [
                rack_observation(),
                rack_observation(),
                rack_observation(occupied=set(), destination_x_mm=22.0),
                rack_observation(
                    occupied={(1, 2)},
                    destination_x_mm=22.0,
                ),
            ],
        )

        workflow.transfer(
            TransferCommand(
                SlotAddress("rack_1", 1, 1),
                SlotAddress("rack_1", 1, 2),
            )
        )

        tcp_points = [
            flange_to_tcp_point(move[0], (0, 0, 10)) for move in arm.moves
        ]
        self.assertTrue(
            any(
                abs(point.x_mm - 22.0) < 1e-6
                and abs(point.z_mm - 44.0) < 1e-6
                for point in tcp_points
            )
        )

    def test_final_scan_must_prove_source_empty_and_destination_occupied(self) -> None:
        workflow, arm, _, observer = self.build_workflow(rack_observation())
        observer.set_sequence(
            "rack_1",
            [
                rack_observation(),
                rack_observation(),
                rack_observation(occupied=set()),
                rack_observation(occupied=set()),
            ],
        )

        with self.assertRaisesRegex(WorkflowError, "final verification failed"):
            workflow.transfer(
                TransferCommand(
                    SlotAddress("rack_1", 1, 1),
                    SlotAddress("rack_1", 1, 2),
                )
            )

        self.assertFalse(workflow.holding_tube)
        self.assertEqual(arm.pose, Pose6D(0, 0, 90, 0, 0, 0))

    def test_carrying_tube_cannot_enter_low_observation_pose(self) -> None:
        workflow, arm, _, _ = self.build_workflow(rack_observation())
        workflow._holding_tube = True

        with self.assertRaisesRegex(WorkflowError, "low observation pose"):
            workflow.move_to_observation_pose()

        self.assertEqual(arm.moves, [])


if __name__ == "__main__":
    unittest.main()
