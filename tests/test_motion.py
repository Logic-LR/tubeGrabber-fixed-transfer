from __future__ import annotations

import math
import unittest
from unittest.mock import patch

from tube_grabber.core.errors import MotionError
from tube_grabber.core.models import MotionPlan, Point3D, Pose6D, Waypoint
from tube_grabber.fakes import FakeArm
from tube_grabber.motion import (
    MotionExecutor,
    MotionPlanner,
    flange_to_tcp_point,
    tcp_to_flange_pose,
)


class MotionPlannerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.planner = MotionPlanner(
            tcp_offset_end_mm=(0.0, 0.0, 10.0),
            approach_height_mm=20.0,
            retreat_height_mm=30.0,
            transit_speed_percent=10,
            approach_speed_percent=5,
            maximum_single_orientation_change_deg=30,
            maximum_tool_axis_misalignment_deg=15,
            tube_total_length_mm=20,
            required_carried_clearance_mm=10,
        )

    def test_tcp_flange_conversion_round_trip(self) -> None:
        flange = Pose6D(10, 20, 30, 0.2, -0.3, 1.0)
        offset = (-3.0, -8.5, 220.0)

        tcp = flange_to_tcp_point(flange, offset)
        restored = tcp_to_flange_pose(
            tcp,
            (flange.rx_rad, flange.ry_rad, flange.rz_rad),
            offset,
        )

        self.assertAlmostEqual(restored.x_mm, flange.x_mm)
        self.assertAlmostEqual(restored.y_mm, flange.y_mm)
        self.assertAlmostEqual(restored.z_mm, flange.z_mm)

    def test_approach_lifts_then_descends_along_rack_normal(self) -> None:
        current = Pose6D(0, 0, 0, math.pi, 0, 0)
        target = Point3D(100, 80, 0)

        plan = self.planner.plan_approach(current, target, (0, 0, 1))
        tcp_points = [
            flange_to_tcp_point(waypoint.pose, (0, 0, 10))
            for waypoint in plan.waypoints
        ]

        self.assertEqual(
            [waypoint.name for waypoint in plan.waypoints],
            ["lift", "above_target", "descend"],
        )
        for actual, expected in zip(
            tcp_points,
            (Point3D(0, 0, 20), Point3D(100, 80, 20), target),
        ):
            self.assertAlmostEqual(actual.x_mm, expected.x_mm)
            self.assertAlmostEqual(actual.y_mm, expected.y_mm)
            self.assertAlmostEqual(actual.z_mm, expected.z_mm)
        self.assertEqual(tcp_points[1].x_mm, tcp_points[2].x_mm)
        self.assertEqual(tcp_points[1].y_mm, tcp_points[2].y_mm)
        self.assertTrue(all(waypoint.linear for waypoint in plan.waypoints))

    def test_approach_uses_tilted_rack_normal_instead_of_base_z(self) -> None:
        axis = (-2**-0.5, 0.0, 2**-0.5)
        current = Pose6D(
            -100.0 * axis[0],
            0,
            -100.0 * axis[2],
            math.pi,
            -math.pi / 4,
            0,
        )
        target = Point3D(100, 80, 0)

        plan = self.planner.plan_approach(current, target, axis)
        above = flange_to_tcp_point(plan.waypoints[-2].pose, (0, 0, 10))
        descend = flange_to_tcp_point(plan.waypoints[-1].pose, (0, 0, 10))
        delta = (
            above.x_mm - descend.x_mm,
            above.y_mm - descend.y_mm,
            above.z_mm - descend.z_mm,
        )

        self.assertAlmostEqual(delta[0], axis[0] * 20.0, places=6)
        self.assertAlmostEqual(delta[1], axis[1] * 20.0, places=6)
        self.assertAlmostEqual(delta[2], axis[2] * 20.0, places=6)

    def test_retreat_moves_tcp_up(self) -> None:
        current = Pose6D(100, 80, 10, math.pi, 0, 0)
        plan = self.planner.plan_retreat(current, (0, 0, 1))
        tcp = flange_to_tcp_point(plan.waypoints[0].pose, (0, 0, 10))
        self.assertEqual(tcp, Point3D(100, 80, 30))
        self.assertTrue(plan.waypoints[0].linear)

    def test_approach_rejects_large_tool_axis_error(self) -> None:
        current = Pose6D(0, 0, 40, math.pi - 0.5, 0, 0)

        with self.assertRaisesRegex(MotionError, "tool axis"):
            self.planner.plan_approach(
                current,
                Point3D(100, 80, 0),
                (0, 0, 1),
            )

    def test_taught_pose_move_rejects_a_large_orientation_change(self) -> None:
        with self.assertRaisesRegex(MotionError, "orientation change"):
            self.planner.plan_pose_move(
                Pose6D(0, 0, 0, 0, 0, 0),
                Pose6D(0, 0, 0, 0, 0, 1.0),
                name="observation_pose",
            )

    def test_wrong_tool_axis_is_rejected(self) -> None:
        with self.assertRaisesRegex(MotionError, "tool axis"):
            self.planner.plan_approach(
                Pose6D(0, 0, 0, 0, 0, 0),
                Point3D(0, 0, 0),
                (0, 0, 1),
            )

    def test_retreat_must_clear_the_full_carried_tube(self) -> None:
        with self.assertRaisesRegex(MotionError, "minimum is 30.0 mm"):
            MotionPlanner(
                tcp_offset_end_mm=(0.0, 0.0, 10.0),
                approach_height_mm=20.0,
                retreat_height_mm=29.0,
                transit_speed_percent=10,
                approach_speed_percent=5,
                maximum_single_orientation_change_deg=30,
                maximum_tool_axis_misalignment_deg=15,
                tube_total_length_mm=20,
                required_carried_clearance_mm=10,
            )


class MotionExecutorTests(unittest.TestCase):
    def _executor(self, arm: FakeArm, *, maximum: float = 200.0) -> MotionExecutor:
        return MotionExecutor(
            arm,
            workspace_min_mm=(-100, -100, -100),
            workspace_max_mm=(200, 200, 200),
            maximum_single_move_mm=maximum,
            position_reached_tolerance_mm=1.0,
            orientation_reached_tolerance_deg=0.5,
        )

    def test_executes_valid_plan(self) -> None:
        arm = FakeArm(Pose6D(0, 0, 0, 0, 0, 0))
        arm.connect()
        plan = MotionPlan(
            (
                Waypoint("first", Pose6D(10, 0, 0, 0, 0, 0), 10),
                Waypoint("second", Pose6D(20, 0, 0, 0, 0, 0), 5),
            )
        )

        self._executor(arm).execute(plan)

        self.assertEqual(len(arm.moves), 2)
        self.assertEqual(arm.pose, plan.waypoints[-1].pose)

    def test_motion_state_callback_wraps_each_actual_arm_move(self) -> None:
        arm = FakeArm(Pose6D(0, 0, 0, 0, 0, 0))
        arm.connect()
        states: list[bool] = []
        executor = self._executor(arm)
        executor.set_motion_state_changed(states.append)
        plan = MotionPlan(
            (
                Waypoint("same", arm.pose, 5),
                Waypoint("first", Pose6D(10, 0, 0, 0, 0, 0), 5),
                Waypoint("second", Pose6D(20, 0, 0, 0, 0, 0), 5),
            )
        )

        executor.execute(plan)

        self.assertEqual(states, [True, False, True, False])

    def test_waits_for_settle_before_reached_pose_check(self) -> None:
        arm = FakeArm(Pose6D(0, 0, 0, 0, 0, 0))
        arm.connect()
        executor = MotionExecutor(
            arm,
            workspace_min_mm=(-100, -100, -100),
            workspace_max_mm=(200, 200, 200),
            maximum_single_move_mm=200,
            reached_check_settle_s=1.0,
            position_reached_tolerance_mm=1.0,
            orientation_reached_tolerance_deg=0.5,
        )

        with patch("tube_grabber.motion.executor.time.sleep") as sleep:
            executor.execute(
                MotionPlan(
                    (Waypoint("move", Pose6D(10, 0, 0, 0, 0, 0), 5),)
                )
            )

        sleep.assert_called_once_with(1.0)

    def test_motion_state_is_cleared_when_arm_move_fails(self) -> None:
        class FailingArm(FakeArm):
            def move_pose(
                self,
                pose: Pose6D,
                speed_percent: int,
                *,
                linear: bool = False,
            ) -> None:
                raise RuntimeError("simulated controller failure")

        arm = FailingArm(Pose6D(0, 0, 0, 0, 0, 0))
        arm.connect()
        states: list[bool] = []
        executor = self._executor(arm)
        executor.set_motion_state_changed(states.append)

        with self.assertRaises(MotionError):
            executor.execute(
                MotionPlan(
                    (Waypoint("fail", Pose6D(10, 0, 0, 0, 0, 0), 5),)
                )
            )

        self.assertEqual(states, [True, False])

    def test_rejects_entire_plan_before_first_move(self) -> None:
        arm = FakeArm(Pose6D(0, 0, 0, 0, 0, 0))
        arm.connect()
        plan = MotionPlan(
            (
                Waypoint("valid", Pose6D(10, 0, 0, 0, 0, 0), 10),
                Waypoint("outside", Pose6D(300, 0, 0, 0, 0, 0), 10),
            )
        )

        with self.assertRaises(MotionError):
            self._executor(arm).execute(plan)

        self.assertEqual(arm.moves, [])
        self.assertEqual(arm.stop_count, 1)

    def test_rejects_excessive_single_step(self) -> None:
        arm = FakeArm(Pose6D(0, 0, 0, 0, 0, 0))
        arm.connect()
        plan = MotionPlan(
            (Waypoint("far", Pose6D(150, 0, 0, 0, 0, 0), 10),)
        )

        with self.assertRaisesRegex(MotionError, "exceeds"):
            self._executor(arm, maximum=100).execute(plan)

        self.assertEqual(arm.moves, [])

    def test_skips_same_pose_and_preserves_linear_mode(self) -> None:
        initial = Pose6D(0, 0, 0, 0, 0, 0)
        arm = FakeArm(initial)
        arm.connect()
        target = Pose6D(10, 0, 0, 0, 0, 0)
        plan = MotionPlan(
            (
                Waypoint("same", initial, 10, linear=False),
                Waypoint("linear", target, 5, linear=True),
            )
        )

        self._executor(arm).execute(plan)

        self.assertEqual(arm.moves, [(target, 5, True)])

    def test_rejects_a_waypoint_that_did_not_reach_target(self) -> None:
        class InaccurateArm(FakeArm):
            def move_pose(
                self,
                pose: Pose6D,
                speed_percent: int,
                *,
                linear: bool = False,
            ) -> None:
                super().move_pose(pose, speed_percent, linear=linear)
                self.pose = Pose6D(
                    pose.x_mm + 3.0,
                    pose.y_mm,
                    pose.z_mm,
                    pose.rx_rad,
                    pose.ry_rad,
                    pose.rz_rad,
                    pose.frame,
                )

        arm = InaccurateArm(Pose6D(0, 0, 0, 0, 0, 0))
        arm.connect()
        plan = MotionPlan(
            (Waypoint("missed", Pose6D(10, 0, 0, 0, 0, 0), 5, True),)
        )

        with self.assertRaisesRegex(MotionError, "did not reach"):
            self._executor(arm).execute(plan)

        self.assertEqual(arm.stop_count, 1)


if __name__ == "__main__":
    unittest.main()
