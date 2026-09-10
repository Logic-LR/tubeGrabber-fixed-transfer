"""Closed-loop transfer between two racks separated by a chassis move."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from tube_grabber.core.errors import ConfigError, VisionError, WorkflowError
from tube_grabber.core.models import (
    Occupancy,
    Pose6D,
    RackObservation,
    SlotAddress,
    TransferCommand,
)
from tube_grabber.navigation import (
    ChassisMove,
    ChassisPort,
    FakeChassis,
    WooshHelperChassis,
)


MOBILE_SCHEMA = "mobile_tube_transfer.v1"


@dataclass(frozen=True)
class MobilePickHomeProgram:
    confirmed: bool
    source_rack: str
    loaded_observation_pose: Pose6D
    loaded_observation_confirmed: bool
    verify_pick_after_home: bool = True

    def __post_init__(self) -> None:
        if self.source_rack != "rack_1":
            raise ConfigError("mobile pick-home source must be rack_1")
        if self.loaded_observation_pose.frame != "base_right":
            raise ConfigError("loaded observation pose must use base_right")

    def require_source(self, source: SlotAddress) -> None:
        rack_id = source.rack_id
        if rack_id != self.source_rack:
            raise WorkflowError(
                f"mobile pick-home source must be {self.source_rack}, got {rack_id}"
            )

    def require_real_confirmation(self) -> None:
        if not self.confirmed:
            raise ConfigError(
                "mobile pick-home is not confirmed; validate the arm-only route first"
            )
        if not self.loaded_observation_confirmed:
            raise ConfigError(
                "loaded observation pose is not confirmed with a held tube"
            )


@dataclass(frozen=True)
class MobileTransferRequest:
    """A mobile transfer request with an optional post-move target selector."""

    source: SlotAddress | None = None
    destination: SlotAddress | None = None
    auto_source: bool = False
    auto_destination: bool = False

    def __post_init__(self) -> None:
        if self.source is None and not self.auto_source:
            raise ValueError("mobile transfer requires a source or auto_source")
        if self.source is not None and self.auto_source:
            raise ValueError("source and auto_source are mutually exclusive")
        if self.destination is None and not self.auto_destination:
            raise ValueError(
                "mobile transfer requires a destination or auto_destination"
            )
        if self.destination is not None and self.auto_destination:
            raise ValueError(
                "destination and auto_destination are mutually exclusive"
            )


@dataclass(frozen=True)
class MobileTransferProgram:
    confirmed: bool
    source_rack: str
    destination_rack: str
    loaded_observation_pose: Pose6D
    loaded_observation_confirmed: bool
    chassis_move: ChassisMove
    rotate_helper_path: str
    pose_servo_path: str
    verify_pick_after_home: bool = True
    return_to_start_after_transfer: bool = True

    def __post_init__(self) -> None:
        if self.source_rack == self.destination_rack:
            raise ConfigError("mobile transfer requires two different racks")
        if {self.source_rack, self.destination_rack} != {"rack_1", "rack_2"}:
            raise ConfigError("mobile transfer racks must be rack_1 and rack_2")
        if self.loaded_observation_pose.frame != "base_right":
            raise ConfigError("loaded observation pose must use base_right")
        if not self.rotate_helper_path or not self.pose_servo_path:
            raise ConfigError("Woosh helper paths cannot be empty")

    def require_command(self, command: TransferCommand) -> None:
        self.require_source(command.source)
        if command.destination.rack_id != self.destination_rack:
            raise WorkflowError(
                f"mobile destination must be {self.destination_rack}, got "
                f"{command.destination.rack_id}"
            )

    def require_request(self, request: MobileTransferRequest) -> None:
        if request.source is not None:
            self.require_source(request.source)
        if request.destination is not None:
            if request.destination.rack_id != self.destination_rack:
                raise WorkflowError(
                    f"mobile destination must be {self.destination_rack}, got "
                    f"{request.destination.rack_id}"
                )

    def require_source(self, source: SlotAddress) -> None:
        rack_id = source.rack_id
        if rack_id != self.source_rack:
            raise WorkflowError(
                f"mobile source must be {self.source_rack}, got {rack_id}"
            )

    def require_real_confirmation(self) -> None:
        if not self.confirmed:
            raise ConfigError(
                "mobile transfer is not confirmed; validate the complete route first"
            )
        if not self.loaded_observation_confirmed:
            raise ConfigError(
                "loaded observation pose is not confirmed with a held tube"
            )


class MobileTransferCoordinator:
    """Coordinate vision, arm, gripper, and chassis without stale coordinates."""

    def __init__(
        self,
        *,
        runtime: Any,
        chassis: ChassisPort | None,
        program: MobileTransferProgram | MobilePickHomeProgram,
        before_chassis: Callable[[str], None] | None = None,
        retry_visual_failures: bool = False,
    ) -> None:
        self.runtime = runtime
        self.chassis = chassis
        self.program = program
        self.before_chassis = before_chassis
        self.retry_visual_failures = bool(retry_visual_failures)
        self._selected_source: SlotAddress | None = None
        self._selected_destination: SlotAddress | None = None

    @property
    def selected_source(self) -> SlotAddress | None:
        """The source selected for the most recent successful attempt."""

        return self._selected_source

    @property
    def selected_destination(self) -> SlotAddress | None:
        """The destination selected for the most recent successful attempt."""

        return self._selected_destination

    def execute(
        self,
        request: MobileTransferRequest | TransferCommand,
    ) -> RackObservation:
        if not isinstance(self.program, MobileTransferProgram):
            raise WorkflowError("full mobile transfer requires a complete program")
        if isinstance(request, TransferCommand):
            request = MobileTransferRequest(
                source=request.source,
                destination=request.destination,
            )
        if not isinstance(request, MobileTransferRequest):
            raise WorkflowError("invalid mobile transfer request")
        self.program.require_request(request)
        self._selected_source = None
        self._selected_destination = None
        workflow = self.runtime.workflow
        if request.source is None:
            source, _ = self.pick_detected_to_home(
                on_source_selected=lambda selected: print(
                    f"移动前自动选择源槽：{selected.text}",
                    flush=True,
                )
            )
        else:
            source = request.source
            self.pick_to_home(source)
        self._selected_source = source

        if self.before_chassis is not None:
            move = self.program.chassis_move
            self.before_chassis(
                "底盘带管移动：先旋转 "
                f"{move.rotation_deg:.1f} 度，再沿车体 X 平移 "
                f"{move.translation_x_m:.3f} m"
            )
        if self.chassis is None:
            raise WorkflowError("full mobile transfer requires a chassis adapter")
        try:
            self.chassis.move_to_destination(self.program.chassis_move)
        except Exception:
            try:
                self.chassis.stop()
            except Exception:
                pass
            raise

        # base_right moved with the chassis.  Never reuse a rack_1 point here.
        destination_preview = workflow.scan(self.program.destination_rack)
        destination_latest = workflow.scan(self.program.destination_rack)
        workflow.verify_rack_unchanged(destination_preview, destination_latest)
        destination = request.destination
        if destination is None:
            destination = self.choose_empty_destination(destination_latest)
            print(
                f"移动后自动选择目标槽：{destination.text}",
                flush=True,
            )
        self._selected_destination = destination
        workflow.place(destination, destination_latest)
        self._record_fake("record_place", destination)

        self._move_arm_to_loaded_observation(require_holding=False)
        final = workflow.scan(self.program.destination_rack)
        workflow.verify_place_result(
            destination_latest,
            destination,
            final,
        )
        if self.program.return_to_start_after_transfer:
            # The arm has already returned to loaded observation/home above.
            # Do not issue any more arm commands while undoing the chassis move.
            print(
                "放置验证完成：右臂保持 loaded observation/home，"
                "底盘开始反向返回起始位置。",
                flush=True,
            )
            try:
                self.chassis.return_to_start(self.program.chassis_move)
            except Exception:
                try:
                    self.chassis.stop()
                except Exception:
                    pass
                raise
        return final

    def choose_empty_destination(
        self,
        observation: RackObservation,
    ) -> SlotAddress:
        """Choose a usable EMPTY slot from a stable destination observation."""

        if observation.rack_id != self.program.destination_rack:
            raise WorkflowError(
                f"expected destination rack {self.program.destination_rack}, "
                f"observed {observation.rack_id}"
            )
        candidates = sorted(
            (
                slot
                for slot in observation.slots
                if slot.occupancy is Occupancy.EMPTY
                and slot.hole_on_plane_base is not None
            ),
            key=lambda slot: (slot.address.row, slot.address.column),
        )
        if not candidates:
            raise WorkflowError(
                f"no confirmed empty slot in {observation.rack_id}"
            )
        return candidates[0].address

    def pick_to_home(self, source: SlotAddress) -> RackObservation:
        """Pick one source tube, return home, and prove the source slot emptied."""
        self.program.require_source(source)
        source_latest = self._scan_source(source.rack_id)
        return self._pick_observed_to_home(source, source_latest)

    def pick_detected_to_home(
        self,
        *,
        on_source_selected: Callable[[SlotAddress], None] | None = None,
    ) -> tuple[SlotAddress, RackObservation]:
        """Pick the only occupied source slot found by stable rack scans."""
        while True:
            source_latest = self._scan_source(self.program.source_rack)
            occupied = [
                slot
                for slot in source_latest.slots
                if slot.occupancy is Occupancy.OCCUPIED
            ]
            if len(occupied) == 1:
                break
            addresses = ", ".join(slot.address.text for slot in occupied) or "none"
            error = WorkflowError(
                "auto source requires exactly one occupied slot; "
                f"found {len(occupied)} ({addresses})"
            )
            if not self.retry_visual_failures:
                raise error
            self._report_visual_retry("唯一试管判定", error)
        source = occupied[0].address
        self.program.require_source(source)
        if on_source_selected is not None:
            on_source_selected(source)
        return source, self._pick_observed_to_home(source, source_latest)

    def _scan_source(
        self,
        rack_id: str,
    ) -> RackObservation:
        workflow = self.runtime.workflow
        if workflow.holding_tube:
            raise WorkflowError("cannot start mobile pick-home while holding a tube")
        while True:
            try:
                source_preview = workflow.scan(rack_id)
                if self.retry_visual_failures:
                    return source_preview
                source_latest = workflow.scan(rack_id)
                workflow.verify_rack_unchanged(source_preview, source_latest)
                return source_latest
            except (VisionError, WorkflowError) as error:
                if not self.retry_visual_failures:
                    raise
                self._report_visual_retry("抓取前定位", error)

    def _pick_observed_to_home(
        self,
        source: SlotAddress,
        source_latest: RackObservation,
    ) -> RackObservation:
        workflow = self.runtime.workflow
        workflow.pick(source, source_latest)
        self._record_fake("record_pick", source)

        self._move_arm_to_loaded_observation()
        if not getattr(self.program, "verify_pick_after_home", True):
            print(
                "抓取后闭环复扫：按配置跳过；已回到 loaded observation/home，"
                "后续流程将使用目标架重新识别。",
                flush=True,
            )
            return source_latest
        while True:
            try:
                source_after_pick = workflow.scan(source.rack_id)
                workflow.verify_pick_result(source_latest, source, source_after_pick)
                return source_after_pick
            except (VisionError, WorkflowError) as error:
                if not self.retry_visual_failures:
                    raise
                self._report_visual_retry("抓取后闭环复扫", error)

    @staticmethod
    def _report_visual_retry(stage: str, error: Exception) -> None:
        print(f"{stage}失败，立即重新采样：{error}", flush=True)

    def _move_arm_to_loaded_observation(
        self,
        *,
        require_holding: bool = True,
    ) -> None:
        workflow = self.runtime.workflow
        if require_holding and not workflow.holding_tube:
            raise WorkflowError("loaded observation move requires a held tube")
        current = self.runtime.arm.get_pose()
        plan = workflow.planner.plan_pose_move(
            current,
            self.program.loaded_observation_pose,
            name="loaded_observation_home",
        )
        workflow.executor.execute(plan)

    def _record_fake(self, method_name: str, address: object) -> None:
        method = getattr(self.runtime.observer, method_name, None)
        if method is not None:
            method(address)


def load_mobile_transfer_program(path: str | Path) -> MobileTransferProgram:
    data = _load_mobile_data(path)
    racks = _mapping(data.get("racks"), "racks")
    pose_data = _mapping(data.get("loaded_observation_pose"), "loaded_observation_pose")
    chassis = _mapping(data.get("chassis"), "chassis")
    helpers = _mapping(chassis.get("helpers"), "chassis.helpers")
    try:
        position = _vector(pose_data["position_mm"], "position_mm")
        rpy = _vector(pose_data["rpy_rad"], "rpy_rad")
        move = ChassisMove(
            rotation_deg=float(chassis["rotation_deg"]),
            translation_x_m=float(chassis["translation_x_m"]),
            rotation_speed_radps=float(chassis["rotation_speed_radps"]),
            translation_speed_mps=float(chassis["translation_speed_mps"]),
            yaw_tolerance_deg=float(chassis["yaw_tolerance_deg"]),
            position_tolerance_m=float(chassis["position_tolerance_m"]),
            maximum_rotation_translation_m=float(
                chassis["maximum_rotation_translation_m"]
            ),
            timeout_s=float(chassis["timeout_s"]),
        )
        return MobileTransferProgram(
            confirmed=_strict_bool(data.get("confirmed"), "confirmed"),
            source_rack=str(racks["source"]),
            destination_rack=str(racks["destination"]),
            loaded_observation_pose=Pose6D(*position, *rpy, frame="base_right"),
            loaded_observation_confirmed=_strict_bool(
                pose_data.get("confirmed"),
                "loaded_observation_pose.confirmed",
            ),
            chassis_move=move,
            rotate_helper_path=str(helpers["rotate_helper_path"]),
            pose_servo_path=str(helpers["pose_servo_path"]),
            verify_pick_after_home=_strict_bool(
                data.get("verify_pick_after_home", True),
                "verify_pick_after_home",
            ),
            return_to_start_after_transfer=_strict_bool(
                data.get("return_to_start_after_transfer", True),
                "return_to_start_after_transfer",
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ConfigError(f"mobile transfer settings are invalid: {error}") from error


def load_mobile_pick_home_program(path: str | Path) -> MobilePickHomeProgram:
    data = _load_mobile_data(path)
    racks = _mapping(data.get("racks"), "racks")
    pose_data = _mapping(data.get("loaded_observation_pose"), "loaded_observation_pose")
    try:
        position = _vector(pose_data["position_mm"], "position_mm")
        rpy = _vector(pose_data["rpy_rad"], "rpy_rad")
        return MobilePickHomeProgram(
            confirmed=_strict_bool(
                data.get("pick_home_confirmed"),
                "pick_home_confirmed",
            ),
            source_rack=str(racks["source"]),
            loaded_observation_pose=Pose6D(*position, *rpy, frame="base_right"),
            loaded_observation_confirmed=_strict_bool(
                pose_data.get("confirmed"),
                "loaded_observation_pose.confirmed",
            ),
            verify_pick_after_home=_strict_bool(
                data.get("verify_pick_after_home", True),
                "verify_pick_after_home",
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ConfigError(f"mobile pick-home settings are invalid: {error}") from error


def _load_mobile_data(path: str | Path) -> Mapping[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"mobile transfer config does not exist: {config_path}")
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ConfigError(f"cannot read mobile transfer config: {error}") from error
    if not isinstance(data, Mapping):
        raise ConfigError("mobile transfer config root must be a mapping")
    if data.get("schema_version") != MOBILE_SCHEMA:
        raise ConfigError(f"schema_version must be {MOBILE_SCHEMA}")
    return data


def build_mobile_chassis(
    mode: str,
    program: MobileTransferProgram,
) -> ChassisPort:
    if mode == "fake":
        return FakeChassis()
    if mode == "real":
        return WooshHelperChassis(
            rotate_helper_path=program.rotate_helper_path,
            pose_servo_path=program.pose_servo_path,
        )
    raise ConfigError("runtime.mode must be fake or real")


def format_mobile_transfer_program(program: MobileTransferProgram) -> str:
    move = program.chassis_move
    pose = program.loaded_observation_pose
    return "\n".join(
        (
            "双架移动程序：" + ("已确认" if program.confirmed else "未确认"),
            f"方向：{program.source_rack} -> {program.destination_rack}",
            "带管观察/home："
            f"XYZ=({pose.x_mm:.1f}, {pose.y_mm:.1f}, {pose.z_mm:.1f}) mm "
            f"RPY=({pose.rx_rad:.4f}, {pose.ry_rad:.4f}, {pose.rz_rad:.4f}) rad",
            f"底盘：旋转 {move.rotation_deg:.1f} deg，随后沿自身 X "
            f"平移 {move.translation_x_m:.3f} m",
            "放置后：右臂保持 loaded observation/home，"
            + (
                "底盘反向返回起始位置"
                if program.return_to_start_after_transfer
                else "不执行底盘回程"
            ),
            "定位：目标站重新执行 D435 + screw/cap YOLO + 深度定位；"
            "不复用源站 base_right 坐标",
        )
    )


def format_mobile_pick_home_program(program: MobilePickHomeProgram) -> str:
    pose = program.loaded_observation_pose
    return "\n".join(
        (
            "抓取回 home 阶段：" + ("已确认" if program.confirmed else "未确认"),
            f"源试管架：{program.source_rack}",
            "带管 home："
            f"XYZ=({pose.x_mm:.1f}, {pose.y_mm:.1f}, {pose.z_mm:.1f}) mm "
            f"RPY=({pose.rx_rad:.4f}, {pose.ry_rad:.4f}, {pose.rz_rad:.4f}) rad",
            "范围：识别、抓取、回带管 home、确认源槽为空；不调用底盘、不释放试管",
        )
    )


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{name} must be a mapping")
    return value


def _vector(value: object, name: str) -> tuple[float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ConfigError(f"{name} must contain three finite numbers")
    if len(value) != 3:
        raise ConfigError(f"{name} must contain three finite numbers")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ConfigError(f"{name} must contain three finite numbers")
    return result  # type: ignore[return-value]


def _strict_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{name} must be true or false")
    return value
