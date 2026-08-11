from __future__ import annotations

import unittest

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
            vertical_tool_rpy_rad=(0.0, 0.0, 0.0),
            approach_height_mm=20.0,
            retreat_height_mm=30.0,
            transit_speed_percent=10,
            approach_speed_percent=5,
            maximum_orientation_error_deg=5,
            maximum_single_orientation_change_deg=30,
            maximum_tool_tilt_deg=180,
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

    def test_approach_lifts_then_descends_vertically(self) -> None:
        current = Pose6D(0, 0, 0, 0, 0, 0)
        target = Point3D(100, 80, 0)

        plan = self.planner.plan_approach(current, target)
        tcp_points = [
            flange_to_tcp_point(waypoint.pose, (0, 0, 10))
            for waypoint in plan.waypoints
        ]

        self.assertEqual(
            [waypoint.name for waypoint in plan.waypoints],
            ["lift", "above_target", "descend"],
        )
        self.assertEqual(tcp_points[0], Point3D(0, 0, 20))
        self.assertEqual(tcp_points[1], Point3D(100, 80, 20))
        self.assertEqual(tcp_points[2], target)
        self.assertEqual(tcp_points[1].x_mm, tcp_points[2].x_mm)
        self.assertEqual(tcp_points[1].y_mm, tcp_points[2].y_mm)
        self.assertTrue(all(waypoint.linear for waypoint in plan.waypoints))

    def test_retreat_moves_tcp_up(self) -> None:
        current = Pose6D(100, 80, -10, 0, 0, 0)
        plan = self.planner.plan_retreat(current)
        tcp = flange_to_tcp_point(plan.waypoints[0].pose, (0, 0, 10))
        self.assertEqual(tcp, Point3D(100, 80, 30))
        self.assertTrue(plan.waypoints[0].linear)

    def test_approach_rejects_non_vertical_start_pose(self) -> None:
        current = Pose6D(0, 0, 40, 0.2, 0, 0)

        with self.assertRaisesRegex(MotionError, "orientation"):
            self.planner.plan_approach(current, Point3D(100, 80, 0))

    def test_taught_pose_move_rejects_a_large_orientation_change(self) -> None:
        with self.assertRaisesRegex(MotionError, "orientation change"):
            self.planner.plan_pose_move(
                Pose6D(0, 0, 0, 0, 0, 0),
                Pose6D(0, 0, 0, 0, 0, 1.0),
                name="observation_pose",
            )

    def test_upward_tool_configuration_is_rejected(self) -> None:
        with self.assertRaisesRegex(MotionError, r"tool \+Z"):
            MotionPlanner(
                tcp_offset_end_mm=(0.0, 0.0, 10.0),
                vertical_tool_rpy_rad=(0.0, 0.0, 0.0),
                approach_height_mm=20.0,
                retreat_height_mm=30.0,
                transit_speed_percent=10,
                approach_speed_percent=5,
                maximum_orientation_error_deg=5,
                maximum_single_orientation_change_deg=30,
                maximum_tool_tilt_deg=1,
                tube_total_length_mm=20,
                required_carried_clearance_mm=10,
            )

    def test_retreat_must_clear_the_full_carried_tube(self) -> None:
        with self.assertRaisesRegex(MotionError, "minimum is 30.0 mm"):
            MotionPlanner(
                tcp_offset_end_mm=(0.0, 0.0, 10.0),
                vertical_tool_rpy_rad=(0.0, 0.0, 0.0),
                approach_height_mm=20.0,
                retreat_height_mm=29.0,
                transit_speed_percent=10,
                approach_speed_percent=5,
                maximum_orientation_error_deg=5,
                maximum_single_orientation_change_deg=30,
                maximum_tool_tilt_deg=180,
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
