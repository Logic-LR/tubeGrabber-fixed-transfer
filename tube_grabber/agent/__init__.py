"""Natural-language adapters. They only produce validated task data."""

from tube_grabber.agent.factory import build_command_agent
from tube_grabber.agent.gemini import GeminiCommandAgent, command_from_arguments
from tube_grabber.agent.local import LocalCommandAgent
from tube_grabber.agent.models import AgentDecision, CommandAgent

__all__ = [
    "AgentDecision",
    "CommandAgent",
    "GeminiCommandAgent",
    "LocalCommandAgent",
    "build_command_agent",
    "command_from_arguments",
]
