"""Recorded rack observations for hardware-free workflow runs."""

from __future__ import annotations

from dataclasses import replace

from tube_grabber.core.errors import VisionError
from tube_grabber.core.models import (
    Occupancy,
    RackObservation,
    SlotAddress,
    SlotObservation,
)


class FakeRackObserver:
    def __init__(self, *observations: RackObservation) -> None:
        self.observations = {
            observation.rack_id: observation for observation in observations
        }
        self._sequences: dict[str, list[RackObservation]] = {}
        self._pending_final: dict[str, RackObservation] = {}
        self.calls: list[str] = []

    def observe_rack(self, rack_id: str) -> RackObservation:
        return self.observe_rack_for_task(rack_id)

    def observe_rack_for_task(
        self,
        rack_id: str,
        *,
        ignored_elevated_slot: SlotAddress | None = None,
    ) -> RackObservation:
        self.calls.append(rack_id)
        sequence = self._sequences.get(rack_id)
        if sequence:
            if len(sequence) > 1:
                return sequence.pop(0)
            return sequence[0]
        if ignored_elevated_slot is not None:
            return self._simulate_pick_and_destination_recheck(
                rack_id,
                ignored_elevated_slot,
            )
        pending = self._pending_final.pop(rack_id, None)
        if pending is not None:
            self.observations[rack_id] = pending
            return pending
        try:
            return self.observations[rack_id]
        except KeyError as error:
            raise VisionError(f"no fake observation for {rack_id}") from error

    def set_sequence(
        self,
        rack_id: str,
        observations: list[RackObservation],
    ) -> None:
        if not observations:
            raise ValueError("fake observation sequence cannot be empty")
        if any(item.rack_id != rack_id for item in observations):
            raise ValueError("fake observation sequence rack ids must match")
        self._sequences[rack_id] = list(observations)
        self.observations[rack_id] = observations[-1]

    def _simulate_pick_and_destination_recheck(
        self,
        rack_id: str,
        destination: SlotAddress,
    ) -> RackObservation:
        try:
            before = self.observations[rack_id]
        except KeyError as error:
            raise VisionError(f"no fake observation for {rack_id}") from error
        occupied = [
            slot for slot in before.slots if slot.occupancy is Occupancy.OCCUPIED
        ]
        if len(occupied) != 1:
            raise VisionError(
                "automatic fake transfer requires exactly one occupied source slot"
            )
        source = occupied[0]
        if source.address == destination:
            raise VisionError("fake source and destination cannot be the same slot")
        if source.cap_top_base is None or source.hole_on_plane_base is None:
            raise VisionError("fake occupied source lacks cap or hole coordinates")

        rechecked_slots = tuple(
            replace(
                slot,
                occupancy=Occupancy.EMPTY,
                cap_top_base=None,
            )
            if slot.address == source.address
            else slot
            for slot in before.slots
        )
        rechecked = replace(
            before,
            slots=rechecked_slots,
            timestamp_ms=before.timestamp_ms + 1.0,
        )
        destination_slot = rechecked.slot(destination)
        if (
            destination_slot.occupancy is not Occupancy.EMPTY
            or destination_slot.hole_on_plane_base is None
        ):
            raise VisionError(
                "fake destination must be empty and have a hole coordinate"
            )
        cap_height = (
            source.cap_top_base.z_mm - source.hole_on_plane_base.z_mm
        )
        final_slots: list[SlotObservation] = []
        for slot in rechecked.slots:
            if slot.address == destination:
                final_slots.append(
                    replace(
                        slot,
                        occupancy=Occupancy.OCCUPIED,
                        cap_top_base=slot.hole_on_plane_base.shifted(
                            dz_mm=cap_height
                        ),
                    )
                )
            else:
                final_slots.append(slot)
        self._pending_final[rack_id] = replace(
            rechecked,
            slots=tuple(final_slots),
            timestamp_ms=rechecked.timestamp_ms + 1.0,
        )
        return rechecked
