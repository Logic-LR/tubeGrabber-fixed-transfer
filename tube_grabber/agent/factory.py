"""Construct the configured command Agent without touching hardware."""

from __future__ import annotations

from collections.abc import Mapping

from tube_grabber.agent.gemini import GeminiCommandAgent
from tube_grabber.agent.local import LocalCommandAgent
from tube_grabber.agent.models import CommandAgent
from tube_grabber.core.errors import ConfigError


def build_command_agent(config: Mapping[str, object]) -> CommandAgent:
    provider = str(config.get("provider", "local")).strip().lower()
    if provider == "local":
        return LocalCommandAgent()
    if provider == "gemini":
        return GeminiCommandAgent(
            model_name=str(config.get("model", "")),
            api_key_environment=str(config.get("api_key_env", "")),
        )
    raise ConfigError("agent.provider must be local or gemini")
