"""Small Agent boundary: language in, one validated command out."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from tube_grabber.core.models import TransferCommand


@dataclass(frozen=True)
class AgentDecision:
    """The Agent either proposes one command or asks for clarification."""

    command: TransferCommand | None
    reply: str

    def __post_init__(self) -> None:
        if not self.reply.strip():
            raise ValueError("agent reply cannot be empty")


class CommandAgent(Protocol):
    def interpret(self, text: str) -> AgentDecision: ...
