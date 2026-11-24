"""A model that sends arguments a Kedi tool rejects is asked to correct them, not stopped."""

from __future__ import annotations

import asyncio

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from kedi.agent_adapter import PydanticAdapter
from kedi.lang import compile_program, parse_program

SOURCE = """
@double(n: int) -> int:
  ###
  Double a number.
  ###
  > tool:
    risk: read_only
  = `n * 2`

> use: double
[answer] << Double four.
= <answer>
"""


def test_rejected_procedure_arguments_return_a_correction_to_the_model() -> None:
    corrections: list[str] = []

    async def model(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        parts = [
            part
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, (ToolReturnPart, RetryPromptPart))
        ]
        if not parts:
            return ModelResponse(parts=[ToolCallPart("double", {"n": "four"}, tool_call_id="c1")])
        content = str(parts[-1].content)
        if "ToolArgumentError" in content:
            corrections.append(content)
            return ModelResponse(parts=[ToolCallPart("double", {"n": 4}, tool_call_id="c2")])
        return ModelResponse(parts=[TextPart(f"result {content}")])

    runtime = compile_program(parse_program(SOURCE), adapter=PydanticAdapter(FunctionModel(model)))
    try:
        assert runtime.run_main() == "result 8"
    finally:
        asyncio.run(runtime.aclose())

    [correction] = corrections
    assert "Tool double failed: ToolArgumentError" in correction
    assert "Parameter 'n' does not match annotated type" in correction
