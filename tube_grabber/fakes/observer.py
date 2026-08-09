"""Recorded rack observations for hardware-free workflow runs."""

from __future__ import annotations

from tube_grabber.core.errors import VisionError
from tube_grabber.core.models import RackObservation


class FakeRackObserver:
    def __init__(self, *observations: RackObservation) -> None:
        self.observations = {
            observation.rack_id: observation for observation in observations
        }
        self.calls: list[str] = []

    def observe_rack(self, rack_id: str) -> RackObservation:
        self.calls.append(rack_id)
        try:
            return self.observations[rack_id]
        except KeyError as error:
            raise VisionError(f"no fake observation for {rack_id}") from error
