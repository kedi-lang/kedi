"""Run a native tool loop with bounded history, without credentials or network I/O."""

from __future__ import annotations

from typing import TypeVar

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from kedi.agent_adapter import HistoryProcessorContext, PydanticAdapter
from kedi.agent_adapter.tools.specs import ToolSpec

MessageT = TypeVar("MessageT")


def keep_recent_cycles(ctx: HistoryProcessorContext[MessageT]) -> list[MessageT]:
    editable = [group for group in ctx.groups if group.closed and not group.protected]
    retained = {group.group_id for group in editable[-2:]}
    retained.update(group.group_id for group in ctx.groups if group.protected)
    return [
        message
        for group in ctx.groups
        if group.group_id in retained
        for message in ctx.messages[group.entry_start : group.entry_end]
    ]


async def respond(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
    assert "obsolete request" not in str(messages)
    for message in reversed(messages):
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, ToolReturnPart):
                    assert part.content == 42
                    return ModelResponse(parts=[TextPart("The current total is 42.")])
    return ModelResponse(parts=[ToolCallPart("get_total", {}, tool_call_id="total-1")])


def main() -> None:
    history: list[ModelMessage] = []
    for prompt in ("obsolete request", "recent request one", "recent request two"):
        history.extend(
            [ModelRequest(parts=[UserPromptPart(prompt)]), ModelResponse(parts=[TextPart("done")])]
        )
    adapter = PydanticAdapter(FunctionModel(respond), history_processor=keep_recent_cycles)
    tool = ToolSpec(
        fn=lambda: 42,
        name="get_total",
        description="Read the current total",
        json_schema={"type": "object", "properties": {}},
        risk="read_only",
        artifact_result="never",
    )
    with adapter.tool_scope([tool]):
        result = adapter.run_sync("What is the current total?", message_history=history)
    assert result.usage.requests == 2
    assert "obsolete request" not in str(result.all_messages())
    assert "obsolete request" in str(history)
    print(result.output)


if __name__ == "__main__":
    main()
