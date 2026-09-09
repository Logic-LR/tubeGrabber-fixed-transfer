"""RealMan two-finger gripper driver; no dexterous-hand API is used."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass

from tube_grabber.core.errors import HardwareError
from tube_grabber.hardware._realman_sdk import (
    call_and_require,
    call_sdk,
    require_success,
)
from tube_grabber.hardware.realman_arm import RealManArm


@dataclass(frozen=True)
class _GripperStatus:
    enabled: bool
    online: bool
    error: int
    mode: int
    position: int
    speed: int
    current_force: int


class RealManGripper:
    def __init__(
        self,
        arm: RealManArm,
        baudrate: int = 9600,
        tool_voltage: int = 3,
        speed: int = 100,
        force: int = 30,
        open_position: int = 170,
        grip_position: int = 135,
        reset_position: int = 1,
        command_timeout_s: int = 5,
        release_position: int | None = None,
        position_tolerance: int = 8,
        poll_interval_s: float = 0.1,
        state_read_attempts: int = 6,
        state_read_retry_s: float = 0.25,
        power_on_wait_s: float = 1.0,
        protocol_startup_wait_s: float = 0.8,
    ) -> None:
        if command_timeout_s <= 0:
            raise ValueError("command_timeout_s 必须大于 0")

        self.arm = arm
        self.baudrate = int(baudrate)
        if self.baudrate not in (9600, 115200, 256000, 460800):
            raise ValueError("baudrate 不是 SDK 支持的 RM Plus 波特率")
        self.tool_voltage = int(tool_voltage)
        if self.tool_voltage not in (0, 2, 3):
            raise ValueError("tool_voltage 必须为 0、2 或 3")
        self.speed = _valid_setting(speed, "speed", maximum=5000)
        self.force = _valid_setting(force, "force", maximum=100)
        self.open_position = _valid_position(open_position, "open_position")
        self.grip_position = _valid_position(grip_position, "grip_position")
        self.reset_position = _valid_position(reset_position, "reset_position")
        self.release_position = _valid_position(
            open_position if release_position is None else release_position,
            "release_position",
        )
        self.command_timeout_s = int(command_timeout_s)
        self.position_tolerance = int(position_tolerance)
        if self.position_tolerance < 0:
            raise ValueError("position_tolerance 不能为负")
        self.poll_interval_s = _positive_finite(
            poll_interval_s,
            "poll_interval_s",
        )
        self.state_read_attempts = int(state_read_attempts)
        if self.state_read_attempts <= 0:
            raise ValueError("state_read_attempts 必须大于 0")
        self.state_read_retry_s = _nonnegative_finite(
            state_read_retry_s,
            "state_read_retry_s",
        )
        self.power_on_wait_s = _nonnegative_finite(
            power_on_wait_s,
            "power_on_wait_s",
        )
        self.protocol_startup_wait_s = _nonnegative_finite(
            protocol_startup_wait_s,
            "protocol_startup_wait_s",
        )
        self._ready = False

    def setup(self) -> None:
        robot = self.arm.sdk_robot
        for method_name in (
            "rm_set_tool_voltage",
            "rm_set_rm_plus_mode",
            "rm_get_rm_plus_state_info",
            "rm_set_hand_force",
            "rm_set_hand_speed",
            "rm_set_hand_follow_pos",
        ):
            if not hasattr(robot, method_name):
                raise HardwareError(f"当前 RealMan SDK 不支持 {method_name}")

        self._ready = False
        call_and_require(
            "设置工具端 24V",
            robot.rm_set_tool_voltage,
            self.tool_voltage,
        )
        time.sleep(self.power_on_wait_s)
        call_and_require(
            f"设置 RM Plus {self.baudrate} 波特率",
            robot.rm_set_rm_plus_mode,
            self.baudrate,
        )
        time.sleep(self.protocol_startup_wait_s)
        self._require_ready_status(self._read_status())
        self._ready = True

    def open_for_pick(self) -> None:
        self._move(self.open_position, "抓取前张开夹爪")

    def grip(self) -> None:
        self._move(
            self.grip_position,
            "夹紧试管",
            accept_force_stop=True,
        )

    def release(self) -> None:
        self._move(self.release_position, "释放试管")

    def reset(self) -> None:
        self._move(self.reset_position, "夹爪复位")

    def _move(
        self,
        position: int,
        operation: str,
        *,
        accept_force_stop: bool = False,
    ) -> None:
        if not self._ready:
            raise HardwareError("夹爪尚未初始化，请先调用 setup()")
        before = self._read_status()
        call_and_require(
            "设置 RM Plus 夹爪力限幅",
            self.arm.sdk_robot.rm_set_hand_force,
            self.force,
        )
        call_and_require(
            "设置 RM Plus 夹爪速度",
            self.arm.sdk_robot.rm_set_hand_speed,
            self.speed,
        )
        command = [position, -1, -1, -1, -1, -1]
        call_and_require(
            operation,
            self.arm.sdk_robot.rm_set_hand_follow_pos,
            command,
            False,
        )

        deadline = time.monotonic() + self.command_timeout_s
        settled_samples = 0
        stopped_fault_samples = 0
        movement_seen = abs(before.position - position) <= self.position_tolerance
        last = before
        while time.monotonic() <= deadline:
            last = self._read_status()
            self._require_ready_status(last)
            if abs(last.position - before.position) >= self.position_tolerance:
                movement_seen = True
            if last.mode in {5, 6}:
                settled_samples = 0
                if last.speed == 0:
                    stopped_fault_samples += 1
                    if stopped_fault_samples >= 3:
                        raise HardwareError(
                            f"{operation} 时夹爪连续处于保护或故障状态"
                        )
                else:
                    stopped_fault_samples = 0
                time.sleep(self.poll_interval_s)
                continue

            stopped_fault_samples = 0
            position_reached = (
                abs(last.position - position) <= self.position_tolerance
            )
            force_stopped = (
                accept_force_stop
                and last.mode == 3
                and last.current_force > 0
            )
            terminal = (last.mode == 2 and position_reached) or force_stopped
            if movement_seen and terminal and last.speed == 0:
                settled_samples += 1
                if settled_samples >= 3:
                    return
            else:
                settled_samples = 0
            time.sleep(self.poll_interval_s)
        detail = (
            f"actual={last.position}, mode={last.mode}, "
            f"speed={last.speed}, force={last.current_force}"
        )
        raise HardwareError(
            f"{operation} 后反馈未稳定：target={position}, {detail}, "
            f"tolerance={self.position_tolerance}"
        )

    @staticmethod
    def _require_ready_status(status: _GripperStatus) -> None:
        if not status.online:
            raise HardwareError("两指夹爪离线")
        if not status.enabled:
            raise HardwareError("两指夹爪未使能")
        if status.error:
            raise HardwareError(f"两指夹爪错误码：0x{status.error:02X}")

    def _read_status(self) -> _GripperStatus:
        state = self._read_state_with_retries()
        system_error = _integer(state.get("sys_state", 0), "sys_state")
        dof_error = _first_integer(state.get("dof_err"), "dof_err")
        error = system_error if system_error else dof_error
        return _GripperStatus(
            enabled=error == 0,
            online=True,
            error=error,
            mode=_first_integer(state.get("dof_state"), "dof_state"),
            position=_first_integer(state.get("pos"), "pos"),
            speed=_first_integer(state.get("speed"), "speed"),
            current_force=_first_integer(state.get("force"), "force"),
        )

    def _read_state_with_retries(self) -> Mapping[str, object]:
        last_error: HardwareError | None = None
        for attempt in range(self.state_read_attempts):
            try:
                return self._read_state()
            except HardwareError as exc:
                last_error = exc
                if attempt + 1 < self.state_read_attempts:
                    time.sleep(self.state_read_retry_s)
        raise HardwareError(
            f"连续 {self.state_read_attempts} 次读取两指夹爪状态失败："
            f"{last_error}"
        )

    def _read_state(self) -> Mapping[str, object]:
        result = call_sdk(
            "读取两指夹爪状态",
            self.arm.sdk_robot.rm_get_rm_plus_state_info,
        )
        require_success("读取两指夹爪状态", result)
        if (
            not isinstance(result, tuple)
            or len(result) < 2
            or not isinstance(result[1], Mapping)
        ):
            raise HardwareError(f"两指夹爪状态格式错误: {result!r}")
        return result[1]


def _valid_position(value: int, name: str) -> int:
    position = int(value)
    if not 1 <= position <= 1000:
        raise ValueError(f"{name} 必须在 1..1000")
    return position


def _valid_setting(value: int, name: str, *, maximum: int) -> int:
    result = int(value)
    if not 1 <= result <= maximum:
        raise ValueError(f"{name} 必须在 1..{maximum}")
    return result


def _nonnegative_finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} 必须是有限的非负数")
    return result


def _positive_finite(value: float, name: str) -> float:
    result = _nonnegative_finite(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} 必须大于 0")
    return result


def _integer(value: object, name: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise HardwareError(f"两指夹爪状态 {name} 无效: {value!r}") from exc


def _first_integer(value: object, name: str) -> int:
    if isinstance(value, (list, tuple)):
        if not value:
            raise HardwareError(f"两指夹爪状态 {name} 为空")
        value = value[0]
    return _integer(value, name)
