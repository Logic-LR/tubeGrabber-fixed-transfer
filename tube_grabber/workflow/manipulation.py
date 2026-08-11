"""Local rack scanning and pick-and-place workflow."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from tube_grabber.core.errors import WorkflowError
from tube_grabber.core.models import (
    Occupancy,
    MotionPlan,
    Pixel,
    Point3D,
    Pose6D,
    RackObservation,
    SlotAddress,
    SlotObservation,
    TransferCommand,
)
from tube_grabber.core.ports import ArmPort, GripperPort
from tube_grabber.motion import MotionExecutor, MotionPlanner
from tube_grabber.motion.planner import rotation_distance_deg, rpy_to_rotation


_PREPARED_POSITION_TOLERANCE_MM = 1.0
_PREPARED_ORIENTATION_TOLERANCE_DEG = 0.5


class RackObserver(Protocol):
    def observe_rack(self, rack_id: str) -> RackObservation: ...


Observer = RackObserver | Callable[[str], RackObservation]


@dataclass(frozen=True)
class PreparedTransfer:
    """A fully checked transfer. Creating it never moves hardware."""

    command: TransferCommand
    observation: RackObservation
    start_pose: Pose6D
    source_target: Point3D
    destination_target: Point3D
    pick_approach: MotionPlan
    pick_retreat: MotionPlan
    place_approach: MotionPlan
    place_retreat: MotionPlan


class ManipulationWorkflow:
    """Execute manipulation at one fixed, already-observed station."""

    def __init__(
        self,
        *,
        arm: ArmPort,
        gripper: GripperPort,
        observer: Observer,
        planner: MotionPlanner,
        executor: MotionExecutor,
        grasp_depth_below_cap_mm: float,
        cap_top_above_rack_mm: float,
        seating_adjust_mm: float,
        scene_recheck_pixel_tolerance_px: float,
        scene_recheck_position_tolerance_mm: float,
        scene_recheck_plane_tolerance_mm: float,
    ) -> None:
        self.arm = arm
        self.gripper = gripper
        self.observer = observer
        self.planner = planner
        self.executor = executor
        self.grasp_depth_below_cap_mm = _positive(
            grasp_depth_below_cap_mm,
            "grasp_depth_below_cap_mm",
        )
        self.cap_top_above_rack_mm = _positive(
            cap_top_above_rack_mm,
            "cap_top_above_rack_mm",
        )
        self.seating_adjust_mm = _finite(
            seating_adjust_mm,
            "seating_adjust_mm",
        )
        self.scene_recheck_pixel_tolerance_px = _positive(
            scene_recheck_pixel_tolerance_px,
            "scene_recheck_pixel_tolerance_px",
        )
        self.scene_recheck_position_tolerance_mm = _positive(
            scene_recheck_position_tolerance_mm,
            "scene_recheck_position_tolerance_mm",
        )
        self.scene_recheck_plane_tolerance_mm = _positive(
            scene_recheck_plane_tolerance_mm,
            "scene_recheck_plane_tolerance_mm",
        )
        if self.cap_top_above_rack_mm <= self.grasp_depth_below_cap_mm:
            raise WorkflowError(
                "cap_top_above_rack_mm must exceed grasp_depth_below_cap_mm"
            )
        self._holding_tube = False

    @property
    def holding_tube(self) -> bool:
        return self._holding_tube

    def scan(self, rack_id: str) -> RackObservation:
        if not rack_id:
            raise WorkflowError("rack_id cannot be empty")
        if hasattr(self.observer, "observe_rack"):
            observation = self.observer.observe_rack(rack_id)  # type: ignore[union-attr]
        else:
            observation = self.observer(rack_id)  # type: ignore[operator]
        if observation.rack_id != rack_id:
            raise WorkflowError(
                f"expected {rack_id}, observed {observation.rack_id}"
            )
        expected = {
            SlotAddress(rack_id, row, column)
            for row in (1, 2)
            for column in range(1, 7)
        }
        actual = {slot.address for slot in observation.slots}
        if actual != expected:
            raise WorkflowError(f"{rack_id} does not contain a complete 2x6 grid")
        return observation

    def pick(
        self,
        address: SlotAddress,
        observation: RackObservation,
    ) -> Point3D:
        if self._holding_tube:
            raise WorkflowError("cannot pick while the gripper already holds a tube")
        target = self._pick_target(address, observation)
        self.gripper.open_for_pick()
        self.executor.execute(
            self.planner.plan_approach(self.arm.get_pose(), target)
        )
        # From the moment a close command is sent, treat the payload as held.
        # If communication fails midway, this conservative state prevents a
        # later command from moving the chassis or starting another pick.
        self._holding_tube = True
        self.gripper.grip()
        self.executor.execute(self.planner.plan_retreat(self.arm.get_pose()))
        return target

    def place(
        self,
        address: SlotAddress,
        observation: RackObservation,
    ) -> Point3D:
        if not self._holding_tube:
            raise WorkflowError("cannot place because the gripper holds no tube")
        target = self._place_target(address, observation)
        self.executor.execute(
            self.planner.plan_approach(self.arm.get_pose(), target)
        )
        self.gripper.release()
        self._holding_tube = False
        self.executor.execute(self.planner.plan_retreat(self.arm.get_pose()))
        return target

    def transfer(self, command: TransferCommand) -> None:
        """Transfer within one rack; cross-rack motion requires navigation."""
        self.execute_transfer(self.prepare_transfer(command))

    def prepare_transfer(self, command: TransferCommand) -> PreparedTransfer:
        """Scan and validate every waypoint without opening the gripper."""
        if command.source.rack_id != command.destination.rack_id:
            raise WorkflowError(
                "cross-rack transfer is disabled until navigation is implemented"
            )
        if self._holding_tube:
            raise WorkflowError("cannot start a transfer while holding a tube")

        observation = self.scan(command.source.rack_id)
        source_target = self._pick_target(command.source, observation)
        destination_target = self._place_target(command.destination, observation)
        start_pose = self.arm.get_pose()
        plans = self._build_transfer_plans(
            start_pose,
            source_target,
            destination_target,
        )
        return PreparedTransfer(
            command=command,
            observation=observation,
            start_pose=start_pose,
            source_target=source_target,
            destination_target=destination_target,
            pick_approach=plans[0],
            pick_retreat=plans[1],
            place_approach=plans[2],
            place_retreat=plans[3],
        )

    def execute_transfer(self, prepared: PreparedTransfer) -> None:
        """Execute only a transfer that already passed perception and planning."""
        if self._holding_tube:
            raise WorkflowError("cannot start a transfer while holding a tube")

        current = self.arm.get_pose()
        position_error_mm = _pose_distance_mm(current, prepared.start_pose)
        orientation_error_deg = rotation_distance_deg(
            rpy_to_rotation(
                (current.rx_rad, current.ry_rad, current.rz_rad)
            ),
            rpy_to_rotation(
                (
                    prepared.start_pose.rx_rad,
                    prepared.start_pose.ry_rad,
                    prepared.start_pose.rz_rad,
                )
            ),
        )
        if (
            position_error_mm > _PREPARED_POSITION_TOLERANCE_MM
            or orientation_error_deg > _PREPARED_ORIENTATION_TOLERANCE_DEG
        ):
            raise WorkflowError(
                "right arm moved after the transfer was prepared; "
                f"position changed {position_error_mm:.2f} mm and orientation "
                f"changed {orientation_error_deg:.2f} deg; scan and plan again"
            )
        for plan in (
            prepared.pick_approach,
            prepared.pick_retreat,
            prepared.place_approach,
            prepared.place_retreat,
        ):
            current = self.executor.validate(plan, current)

        self.gripper.open_for_pick()
        self.executor.execute(prepared.pick_approach)
        self._holding_tube = True
        self.gripper.grip()
        self.executor.execute(prepared.pick_retreat)
        self.executor.execute(prepared.place_approach)
        self.gripper.release()
        self._holding_tube = False
        self.executor.execute(prepared.place_retreat)

    def verify_prepared_scene(
        self,
        prepared: PreparedTransfer,
    ) -> RackObservation:
        """Rescan immediately before motion and reject a stale visual plan."""
        latest = self.scan(prepared.command.source.rack_id)
        original = prepared.observation

        original_keypoints = original.rack_keypoints or (original.marker,)
        latest_keypoints = latest.rack_keypoints or (latest.marker,)
        if len(original_keypoints) != len(latest_keypoints):
            raise WorkflowError("rack keypoint contract changed after planning")
        for index, (first, second) in enumerate(
            zip(original_keypoints, latest_keypoints)
        ):
            shift_px = _pixel_distance(first, second)
            if shift_px > self.scene_recheck_pixel_tolerance_px:
                raise WorkflowError(
                    f"rack keypoint {index} moved {shift_px:.2f} px after planning; "
                    "scan and plan again"
                )
        plane_shift_mm = abs(latest.plane_z_mm - original.plane_z_mm)
        if plane_shift_mm > self.scene_recheck_plane_tolerance_mm:
            raise WorkflowError(
                f"rack plane moved {plane_shift_mm:.2f} mm after planning; "
                "scan and plan again"
            )

        for original_slot in original.slots:
            latest_slot = latest.slot(original_slot.address)
            if latest_slot.occupancy is not original_slot.occupancy:
                raise WorkflowError(
                    f"{original_slot.address.text} changed from "
                    f"{original_slot.occupancy.value} to "
                    f"{latest_slot.occupancy.value} after planning"
                )
            shift_px = _pixel_distance(
                original_slot.pixel,
                latest_slot.pixel,
            )
            if shift_px > self.scene_recheck_pixel_tolerance_px:
                raise WorkflowError(
                    f"{original_slot.address.text} moved {shift_px:.2f} px "
                    "after planning; scan and plan again"
                )

        latest_source = self._pick_target(prepared.command.source, latest)
        latest_destination = self._place_target(
            prepared.command.destination,
            latest,
        )
        for name, original_target, latest_target in (
            ("source", prepared.source_target, latest_source),
            ("destination", prepared.destination_target, latest_destination),
        ):
            shift_mm = _point_distance_mm(original_target, latest_target)
            if shift_mm > self.scene_recheck_position_tolerance_mm:
                raise WorkflowError(
                    f"{name} target moved {shift_mm:.2f} mm after planning; "
                    "scan and plan again"
                )
        return latest

    def _pick_target(
        self,
        address: SlotAddress,
        observation: RackObservation,
    ) -> Point3D:
        slot = self._slot(address, observation)
        if slot.occupancy is not Occupancy.OCCUPIED:
            raise WorkflowError(
                f"source {address.text} must be occupied, got {slot.occupancy.value}"
            )
        if slot.cap_top_base is None:
            raise WorkflowError(f"source {address.text} has no cap coordinate")
        return slot.cap_top_base.shifted(
            dz_mm=-self.grasp_depth_below_cap_mm
        )

    def _place_target(
        self,
        address: SlotAddress,
        observation: RackObservation,
    ) -> Point3D:
        slot = self._slot(address, observation)
        if slot.occupancy is not Occupancy.EMPTY:
            raise WorkflowError(
                f"destination {address.text} must be empty, got {slot.occupancy.value}"
            )
        if slot.hole_on_plane_base is None:
            raise WorkflowError(f"destination {address.text} has no hole coordinate")
        _finite(observation.plane_z_mm, "rack plane_z_mm")
        hole = slot.hole_on_plane_base
        target_z_mm = (
            hole.z_mm
            + self.cap_top_above_rack_mm
            - self.grasp_depth_below_cap_mm
            + self.seating_adjust_mm
        )
        return Point3D(
            hole.x_mm,
            hole.y_mm,
            target_z_mm,
            hole.frame,
        )

    @staticmethod
    def _slot(
        address: SlotAddress,
        observation: RackObservation,
    ) -> SlotObservation:
        if address.rack_id != observation.rack_id:
            raise WorkflowError(
                f"slot {address.text} is not in observed rack {observation.rack_id}"
            )
        try:
            return observation.slot(address)
        except KeyError as error:
            raise WorkflowError(f"slot {address.text} is missing") from error

    def _build_transfer_plans(
        self,
        current: Pose6D,
        source_target: Point3D,
        destination_target: Point3D,
    ) -> tuple[MotionPlan, MotionPlan, MotionPlan, MotionPlan]:
        """Build and validate all four plan segments without moving the arm."""
        source_approach = self.planner.plan_approach(current, source_target)
        current = self.executor.validate(source_approach, current)
        source_retreat = self.planner.plan_retreat(current)
        current = self.executor.validate(source_retreat, current)
        destination_approach = self.planner.plan_approach(
            current,
            destination_target,
        )
        current = self.executor.validate(destination_approach, current)
        destination_retreat = self.planner.plan_retreat(current)
        self.executor.validate(destination_retreat, current)
        return (
            source_approach,
            source_retreat,
            destination_approach,
            destination_retreat,
        )


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise WorkflowError(f"{name} must be finite")
    return result


def _positive(value: float, name: str) -> float:
    result = _finite(value, name)
    if result <= 0:
        raise WorkflowError(f"{name} must be positive")
    return result


def _pose_distance_mm(first: Pose6D, second: Pose6D) -> float:
    if first.frame != second.frame:
        raise WorkflowError(
            f"prepared pose frame changed: {first.frame} != {second.frame}"
        )
    return math.sqrt(
        (first.x_mm - second.x_mm) ** 2
        + (first.y_mm - second.y_mm) ** 2
        + (first.z_mm - second.z_mm) ** 2
    )


def _point_distance_mm(first: Point3D, second: Point3D) -> float:
    if first.frame != second.frame:
        raise WorkflowError(
            f"scene point frame changed: {first.frame} != {second.frame}"
        )
    return math.sqrt(
        (first.x_mm - second.x_mm) ** 2
        + (first.y_mm - second.y_mm) ** 2
        + (first.z_mm - second.z_mm) ** 2
    )


def _pixel_distance(first: Pixel, second: Pixel) -> float:
    return math.hypot(first.u - second.u, first.v - second.v)
