from __future__ import annotations

import unittest
from types import SimpleNamespace

from tube_grabber.agent import (
    GeminiCommandAgent,
    LocalCommandAgent,
    command_from_arguments,
)
from tube_grabber.core.errors import AgentError
from tube_grabber.core.models import SlotAddress, TransferCommand


def valid_arguments() -> dict[str, object]:
    return {
        "source_rack": "rack_1",
        "source_row": 1,
        "source_column": 3,
        "destination_rack": "rack_2",
        "destination_row": 2,
        "destination_column": 5,
    }


class LocalCommandAgentTests(unittest.TestCase):
    def test_two_addresses_create_one_command(self) -> None:
        decision = LocalCommandAgent().interpret(
            "请把 RACK_1.R1C1 移动到 rack_1.r2c6"
        )
        self.assertEqual(
            decision.command,
            TransferCommand(
                SlotAddress("rack_1", 1, 1),
                SlotAddress("rack_1", 2, 6),
            ),
        )

    def test_incomplete_text_requests_clarification(self) -> None:
        decision = LocalCommandAgent().interpret("把这个试管移过去")
        self.assertIsNone(decision.command)
        self.assertIn("源槽位", decision.reply)

    def test_more_than_one_task_is_not_guessed(self) -> None:
        decision = LocalCommandAgent().interpret(
            "rack_1.r1c1 -> rack_1.r1c2, then rack_1.r1c3"
        )
        self.assertIsNone(decision.command)

class AgentArgumentValidationTests(unittest.TestCase):
    def test_valid_arguments_create_cross_rack_command(self) -> None:
        command = command_from_arguments(valid_arguments())
        self.assertEqual(command.source, SlotAddress("rack_1", 1, 3))
        self.assertEqual(command.destination, SlotAddress("rack_2", 2, 5))

    def test_missing_argument_is_rejected(self) -> None:
        arguments = valid_arguments()
        del arguments["destination_column"]
        with self.assertRaisesRegex(AgentError, "omitted"):
            command_from_arguments(arguments)

    def test_extra_argument_is_rejected(self) -> None:
        arguments = valid_arguments()
        arguments["speed"] = 100
        with self.assertRaisesRegex(AgentError, "unexpected"):
            command_from_arguments(arguments)

    def test_unknown_rack_is_rejected(self) -> None:
        arguments = valid_arguments()
        arguments["source_rack"] = "rack_3"
        with self.assertRaisesRegex(AgentError, "rack_1 or rack_2"):
            command_from_arguments(arguments)

    def test_fractional_row_is_rejected_instead_of_rounded(self) -> None:
        arguments = valid_arguments()
        arguments["source_row"] = 1.5
        with self.assertRaisesRegex(AgentError, "integer"):
            command_from_arguments(arguments)

    def test_boolean_column_is_rejected(self) -> None:
        arguments = valid_arguments()
        arguments["source_column"] = True
        with self.assertRaisesRegex(AgentError, "integer"):
            command_from_arguments(arguments)


class GeminiCommandAgentTests(unittest.TestCase):
    def _agent_with_response(self, response: object) -> GeminiCommandAgent:
        models = SimpleNamespace(
            generate_content=lambda **_: response,
        )
        client = SimpleNamespace(models=models)
        return GeminiCommandAgent(
            model_name="test-model",
            api_key_environment="UNUSED_IN_TEST",
            client=client,
            request_config=object(),
        )

    def test_manual_function_call_becomes_command(self) -> None:
        call = SimpleNamespace(name="propose_transfer", args=valid_arguments())
        response = SimpleNamespace(function_calls=[call], text=None)
        decision = self._agent_with_response(response).interpret("测试")
        self.assertEqual(decision.command.source.text, "rack_1.r1c3")  # type: ignore[union-attr]
        self.assertEqual(decision.command.destination.text, "rack_2.r2c5")  # type: ignore[union-attr]

    def test_text_response_is_treated_as_clarification(self) -> None:
        response = SimpleNamespace(function_calls=[], text="请问目标是哪个孔？")
        decision = self._agent_with_response(response).interpret("移过去")
        self.assertIsNone(decision.command)
        self.assertEqual(decision.reply, "请问目标是哪个孔？")

    def test_multiple_calls_are_rejected(self) -> None:
        call = SimpleNamespace(name="propose_transfer", args=valid_arguments())
        response = SimpleNamespace(function_calls=[call, call], text=None)
        with self.assertRaisesRegex(AgentError, "exactly one"):
            self._agent_with_response(response).interpret("执行两个任务")


if __name__ == "__main__":
    unittest.main()
