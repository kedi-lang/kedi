"""#223: an oversized inline tool result is a fatal author error.

The inline-size safety limit itself does not change. The error message must
be actionable for the program author (enable `> artifacts:` or return less),
not advice for the model, which never sees it. Covered on both adapters,
for buffered and streaming results.
"""

import asyncio
from collections.abc import Iterator

import pytest
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic_ai.messages import (
    ModelResponse,
    RetryPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import FunctionModel

from kedi import ArtifactStream
from kedi.agent_adapter.adapters import LangChainAdapter, PydanticAdapter
from kedi.agent_adapter.tools.execution import with_artifact_result
from kedi.agent_adapter.tools.specs import ToolSpec
from kedi.artifacts import ArtifactManager, ArtifactPolicy, ToolOutputTooLargeError
from kedi.errors import KediExecutionError
from kedi.lang import compile_program, parse_program
from tests.test_langchain_branch_parity import ScriptedModel

PROGRAM = (
    "> artifacts:\n    enabled: false\n"
    "@read_log(day: str) -> str:\n    ###\n    Return the audit log of one day.\n    ###\n"
    "    > tool:\n        risk: read_only\n    = `'x' * 300_000`\n\n"
    "> approval: allow\n> use: read_log\n[reply] << Which actor approved CHG-1?\n= <reply>\n"
)

EXPECTED_MESSAGE = (
    "tool output is 300000 bytes, exceeding the inline safety limit of 100000 bytes; "
    "enable `> artifacts:` for this tool or return less"
)


def test_oversized_output_error_addresses_the_program_author() -> None:
    message = str(ToolOutputTooLargeError(size_bytes=300000, limit_bytes=100000))
    assert message == EXPECTED_MESSAGE
    assert "request a narrower result" not in message


def test_disabled_artifacts_error_is_actionable_for_buffered_results() -> None:
    manager = ArtifactManager(session_id="disabled-buffered-message")
    try:
        wrapped = with_artifact_result(
            ToolSpec(fn=lambda: "x" * 300_000, name="large_tool"),
            manager=manager,
            policy=ArtifactPolicy.patch({"enabled": False}).resolved(),
        )
        with pytest.raises(ToolOutputTooLargeError) as raised:
            wrapped.fn()
        assert str(raised.value) == EXPECTED_MESSAGE
    finally:
        manager.close()


def test_disabled_artifacts_error_is_actionable_for_streamed_results() -> None:
    def chunks() -> Iterator[str]:
        for _ in range(100):
            yield "x" * 10_000

    manager = ArtifactManager(session_id="disabled-stream-message")
    try:
        wrapped = with_artifact_result(
            ToolSpec(fn=lambda: ArtifactStream.text(chunks()), name="stream_tool"),
            manager=manager,
            policy=ArtifactPolicy.patch({"enabled": False}).resolved(),
        )
        with pytest.raises(ToolOutputTooLargeError) as raised:
            wrapped.fn()
        assert str(raised.value).startswith(
            "tool output is 110000 bytes, exceeding the inline safety limit of 100000 bytes; "
            "enable `> artifacts:` for this tool or return less"
        )
    finally:
        manager.close()


def _assert_run_raises_without_model_seeing_result(run_main, seen: list[bool]) -> None:
    with pytest.raises(KediExecutionError) as raised:
        run_main()
    assert "ToolOutputTooLargeError" in str(raised.value)
    assert EXPECTED_MESSAGE in str(raised.value)
    assert not seen, "the model must never see the oversized result"


def test_oversized_inline_tool_result_ends_run_pydantic() -> None:
    seen: list[bool] = []

    def respond(messages, info):
        seen_parts = [
            p
            for m in messages
            for p in getattr(m, "parts", [])
            if isinstance(p, (ToolReturnPart, RetryPromptPart))
        ]
        if not seen_parts:
            return ModelResponse(parts=[ToolCallPart("read_log", {"day": "2026-10-14"})])
        seen.append(True)
        return ModelResponse(parts=[TextPart("narrowed")])

    runtime = compile_program(
        parse_program(PROGRAM), adapter=PydanticAdapter(FunctionModel(respond))
    )
    try:
        _assert_run_raises_without_model_seeing_result(runtime.run_main, seen)
    finally:
        asyncio.run(runtime.aclose())


def test_oversized_inline_tool_result_ends_run_langchain() -> None:
    seen: list[bool] = []

    class Model(ScriptedModel):
        async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
            tools = [m for m in messages if isinstance(m, ToolMessage)]
            if tools:
                seen.append(True)
                return ChatResult(
                    generations=[ChatGeneration(message=AIMessage(content="narrowed"))]
                )
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "name": "read_log",
                                    "args": {"day": "2026-10-14"},
                                    "id": "c1",
                                }
                            ],
                        )
                    )
                ]
            )

    runtime = compile_program(parse_program(PROGRAM), adapter=LangChainAdapter(Model(responses=[])))
    try:
        _assert_run_raises_without_model_seeing_result(runtime.run_main, seen)
    finally:
        asyncio.run(runtime.aclose())
