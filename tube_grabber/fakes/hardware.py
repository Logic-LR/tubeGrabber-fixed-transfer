"""Small deterministic hardware replacements used by tests and fake mode."""

from __future__ import annotations

from tube_grabber.core.errors import HardwareError
from tube_grabber.core.models import Pose6D


class FakeArm:
    def __init__(
        self,
        initial_pose: Pose6D,
        *,
        fail_on_move_number: int | None = None,
    ) -> None:
        self.pose = initial_pose
        self.fail_on_move_number = fail_on_move_number
        self.connected = False
        self.moves: list[tuple[Pose6D, int, bool]] = []
        self.stop_count = 0

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def get_pose(self) -> Pose6D:
        self._require_connected()
        return self.pose

    def move_pose(
        self,
        pose: Pose6D,
        speed_percent: int,
        *,
        linear: bool = False,
    ) -> None:
        self._require_connected()
        move_number = len(self.moves) + 1
        if self.fail_on_move_number == move_number:
            raise HardwareError(f"fake arm failed on move {move_number}")
        self.moves.append((pose, speed_percent, linear))
        self.pose = pose

    def stop(self) -> None:
        self.stop_count += 1

    def _require_connected(self) -> None:
        if not self.connected:
            raise HardwareError("fake arm is not connected")


class FakeGripper:
    def __init__(self) -> None:
        self.ready = False
        self.actions: list[str] = []

    def setup(self) -> None:
        self.ready = True
        self.actions.append("setup")

    def open_for_pick(self) -> None:
        self._record("open_for_pick")

    def grip(self) -> None:
        self._record("grip")

    def release(self) -> None:
        self._record("release")

    def reset(self) -> None:
        self._record("reset")

    def _record(self, action: str) -> None:
        if not self.ready:
            raise HardwareError("fake gripper is not set up")
        self.actions.append(action)
