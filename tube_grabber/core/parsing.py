"""Parse the small command format used by CLI and future speech code."""

from __future__ import annotations

import re

from tube_grabber.core.models import SlotAddress, TransferCommand


_SLOT_PATTERN = re.compile(
    r"^(?P<rack>rack_[12])\.r(?P<row>[12])c(?P<column>[1-6])$",
    re.IGNORECASE,
)


def parse_slot(text: str) -> SlotAddress:
    match = _SLOT_PATTERN.fullmatch(text.strip())
    if match is None:
        raise ValueError(
            f"invalid slot {text!r}; expected rack_1.r1c1 .. rack_2.r2c6"
        )
    return SlotAddress(
        rack_id=match.group("rack").lower(),
        row=int(match.group("row")),
        column=int(match.group("column")),
    )


def parse_transfer(source: str, destination: str) -> TransferCommand:
    return TransferCommand(parse_slot(source), parse_slot(destination))
