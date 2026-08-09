"""Gemini adapter that proposes data but never executes robot functions."""

from __future__ import annotations

import os
from importlib import import_module
from collections.abc import Mapping
from math import isfinite
from typing import Any

from tube_grabber.agent.models import AgentDecision
from tube_grabber.agent.prompt import SYSTEM_INSTRUCTION
from tube_grabber.core.errors import AgentError, ConfigError
from tube_grabber.core.models import SlotAddress, TransferCommand


_FUNCTION_NAME = "propose_transfer"
_REQUIRED_ARGUMENTS = {
    "source_rack",
    "source_row",
    "source_column",
    "destination_rack",
    "destination_row",
    "destination_column",
}


class GeminiCommandAgent:
    """Use manual function calling so the SDK cannot run robot actions."""

    def __init__(
        self,
        *,
        model_name: str,
        api_key_environment: str,
        client: object | None = None,
        request_config: object | None = None,
    ) -> None:
        if not model_name.strip():
            raise ConfigError("agent.model cannot be empty")
        if not api_key_environment.strip():
            raise ConfigError("agent.api_key_env cannot be empty")

        self._model_name = model_name.strip()
        if client is not None:
            if request_config is None:
                raise ConfigError("an injected Gemini client requires request_config")
            self._client = client
            self._request_config = request_config
            return

        api_key = os.getenv(api_key_environment)
        if not api_key:
            raise ConfigError(
                f"environment variable {api_key_environment} is not set"
            )
        try:
            genai: Any = import_module("google.genai")
            types: Any = import_module("google.genai.types")
        except ImportError as error:
            raise ConfigError(
                'Gemini Agent requires: python -m pip install -e ".[agent]"'
            ) from error

        declaration = types.FunctionDeclaration(
            name=_FUNCTION_NAME,
            description=(
                "Propose one test-tube transfer after all six address fields "
                "are explicit. This function does not move hardware."
            ),
            parameters_json_schema=_function_schema(),
        )
        self._request_config = types.GenerateContentConfig(
            temperature=0.0,
            system_instruction=SYSTEM_INSTRUCTION,
            tools=[types.Tool(function_declarations=[declaration])],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        )
        self._client = genai.Client(api_key=api_key)

    def interpret(self, text: str) -> AgentDecision:
        if not text.strip():
            return AgentDecision(None, "请输入完整的试管搬运命令")
        try:
            models = getattr(self._client, "models")
            response = models.generate_content(
                model=self._model_name,
                contents=text.strip(),
                config=self._request_config,
            )
        except Exception as error:
            raise AgentError(f"Gemini request failed: {error}") from error

        calls = list(getattr(response, "function_calls", None) or ())
        if not calls:
            reply = str(getattr(response, "text", "") or "").strip()
            return AgentDecision(
                None,
                reply or "请明确说明源架行列和目标架行列",
            )
        if len(calls) != 1:
            raise AgentError("Agent must propose exactly one transfer")

        call = calls[0]
        if getattr(call, "name", None) != _FUNCTION_NAME:
            raise AgentError("Agent returned an unsupported function call")
        args = getattr(call, "args", None)
        if not isinstance(args, Mapping):
            raise AgentError("Agent function arguments must be an object")

        command = command_from_arguments(args)
        return AgentDecision(
            command,
            f"已理解：{command.source.text} -> {command.destination.text}",
        )


def command_from_arguments(arguments: Mapping[str, Any]) -> TransferCommand:
    """Deterministically validate untrusted model output."""
    keys = set(arguments)
    missing = sorted(_REQUIRED_ARGUMENTS - keys)
    extra = sorted(keys - _REQUIRED_ARGUMENTS)
    if missing:
        raise AgentError(f"Agent omitted arguments: {missing}")
    if extra:
        raise AgentError(f"Agent returned unexpected arguments: {extra}")

    source = SlotAddress(
        _rack(arguments["source_rack"], "source_rack"),
        _integer(arguments["source_row"], "source_row"),
        _integer(arguments["source_column"], "source_column"),
    )
    destination = SlotAddress(
        _rack(arguments["destination_rack"], "destination_rack"),
        _integer(arguments["destination_row"], "destination_row"),
        _integer(arguments["destination_column"], "destination_column"),
    )
    return TransferCommand(source, destination)


def _rack(value: object, name: str) -> str:
    rack_id = str(value).strip().lower()
    if rack_id not in {"rack_1", "rack_2"}:
        raise AgentError(f"{name} must be rack_1 or rack_2")
    return rack_id


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise AgentError(f"{name} must be an integer")
    if not isinstance(value, (str, int, float)):
        raise AgentError(f"{name} must be an integer")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise AgentError(f"{name} must be an integer") from error
    if not isfinite(number) or not number.is_integer():
        raise AgentError(f"{name} must be an integer")
    return int(number)


def _function_schema() -> dict[str, object]:
    rack = {"type": "string", "enum": ["rack_1", "rack_2"]}
    row = {"type": "integer", "enum": [1, 2]}
    column = {"type": "integer", "minimum": 1, "maximum": 6}
    return {
        "type": "object",
        "properties": {
            "source_rack": rack,
            "source_row": row,
            "source_column": column,
            "destination_rack": rack,
            "destination_row": row,
            "destination_column": column,
        },
        "required": sorted(_REQUIRED_ARGUMENTS),
        "additionalProperties": False,
    }
