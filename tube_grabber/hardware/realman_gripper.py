"""RealMan RM Plus 两指夹爪驱动。"""

from __future__ import annotations

import math
import time

from tube_grabber.core.errors import HardwareError
from tube_grabber.hardware._realman_sdk import call_and_require
from tube_grabber.hardware.realman_arm import RealManArm


class RealManGripper:
    def __init__(
        self,
        arm: RealManArm,
        baudrate: int = 9600,
        tool_voltage: int = 3,
        open_position: int = 170,
        grip_position: int = 135,
        reset_position: int = 1,
        command_timeout_s: int = 5,
        power_on_wait_s: float = 1.0,
        protocol_startup_wait_s: float = 0.8,
        release_position: int | None = None,
    ) -> None:
        if baudrate not in (9600, 115200, 256000, 460800):
            raise ValueError("RM Plus 波特率不受支持")
        if tool_voltage not in (2, 3):
            raise ValueError("tool_voltage 只能是 2(12V) 或 3(24V)")
        if command_timeout_s <= 0:
            raise ValueError("command_timeout_s 必须大于 0")

        self.arm = arm
        self.baudrate = baudrate
        self.tool_voltage = tool_voltage
        self.open_position = _valid_position(open_position, "open_position")
        self.grip_position = _valid_position(grip_position, "grip_position")
        self.reset_position = _valid_position(reset_position, "reset_position")
        self.release_position = _valid_position(
            open_position if release_position is None else release_position,
            "release_position",
        )
        self.command_timeout_s = int(command_timeout_s)
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
        if not hasattr(robot, "rm_set_tool_voltage"):
            raise HardwareError("当前 RealMan SDK 不支持 rm_set_tool_voltage")
        if not hasattr(robot, "rm_set_rm_plus_mode"):
            raise HardwareError("当前 RealMan SDK 不支持 rm_set_rm_plus_mode")
        if not hasattr(robot, "rm_set_gripper_position"):
            raise HardwareError("当前 RealMan SDK 不支持 rm_set_gripper_position")

        self._ready = False
        call_and_require(
            "设置夹爪工具电压",
            robot.rm_set_tool_voltage,
            self.tool_voltage,
        )
        time.sleep(self.power_on_wait_s)
        call_and_require(
            "启用 RM Plus 协议",
            robot.rm_set_rm_plus_mode,
            self.baudrate,
        )
        time.sleep(self.protocol_startup_wait_s)
        self._ready = True

    def open_for_pick(self) -> None:
        self._move(self.open_position, "抓取前张开夹爪")

    def grip(self) -> None:
        self._move(self.grip_position, "夹紧试管")

    def release(self) -> None:
        self._move(self.release_position, "释放试管")

    def reset(self) -> None:
        self._move(self.reset_position, "夹爪复位")

    def _move(self, position: int, operation: str) -> None:
        if not self._ready:
            raise HardwareError("夹爪尚未初始化，请先调用 setup()")
        call_and_require(
            operation,
            self.arm.sdk_robot.rm_set_gripper_position,
            position,
            True,
            self.command_timeout_s,
        )


def _valid_position(value: int, name: str) -> int:
    position = int(value)
    if not 1 <= position <= 1000:
        raise ValueError(f"{name} 必须在 1..1000")
    return position


def _nonnegative_finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} 必须是有限的非负数")
    return result
