"""Offline command adapter for tests and network-independent operation."""

from __future__ import annotations

import re

from tube_grabber.agent.models import AgentDecision
from tube_grabber.core.models import TransferCommand
from tube_grabber.core.parsing import parse_slot


_ADDRESS = re.compile(r"rack_[12]\.r[12]c[1-6]", re.IGNORECASE)


class LocalCommandAgent:
    """Read exactly two canonical slot addresses from arbitrary text."""

    def interpret(self, text: str) -> AgentDecision:
        addresses = _ADDRESS.findall(text)
        if len(addresses) != 2:
            return AgentDecision(
                command=None,
                reply=(
                    "请明确写出一个源槽位和一个目标槽位，例如 "
                    "rack_1.r1c1 -> rack_1.r1c2"
                ),
            )
        command = TransferCommand(
            source=parse_slot(addresses[0]),
            destination=parse_slot(addresses[1]),
        )
        return AgentDecision(
            command=command,
            reply=f"已理解：{command.source.text} -> {command.destination.text}",
        )
