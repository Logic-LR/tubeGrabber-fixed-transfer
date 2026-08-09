"""Interfaces between the workflow and replaceable implementations."""

from __future__ import annotations

from typing import Protocol

from tube_grabber.core.models import CameraFrame, Detection, Pose6D


class CameraPort(Protocol):
    def start(self) -> None: ...

    def capture(self) -> CameraFrame: ...

    def stop(self) -> None: ...


class DetectorPort(Protocol):
    def detect(self, color_image: object) -> list[Detection]: ...


class ArmPort(Protocol):
    def connect(self) -> None: ...

    def disconnect(self) -> None: ...

    def get_pose(self) -> Pose6D: ...

    def move_pose(
        self,
        pose: Pose6D,
        speed_percent: int,
        *,
        linear: bool = False,
    ) -> None: ...

    def stop(self) -> None: ...


class GripperPort(Protocol):
    def setup(self) -> None: ...

    def open_for_pick(self) -> None: ...

    def grip(self) -> None: ...

    def release(self) -> None: ...

    def reset(self) -> None: ...
