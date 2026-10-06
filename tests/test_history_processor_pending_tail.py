import asyncio
import pytest


def select_history(history, policy):
    if policy == "recent" and len(history.groups) > 8:
        return history.keep_recent_groups(2)
    return history.messages


@pytest.mark.parametrize("backend", ["pydantic", "langchain"])
@pytest.mark.parametrize("policy", ["preserve", "recent"])
def test_history_policy_keeps_previous_exchange_and_current_unfinished_tool(backend, policy):
    from kedi.agent_adapter.history_processing import HistoryProcessorRunner

    if backend == "pydantic":
        from pydantic_ai.messages import (
            ModelRequest,
            ModelResponse,
            TextPart,
            ToolCallPart,
            UserPromptPart,
        )
        from kedi.agent_adapter.adapters.pydantic.history_processing import (
            PydanticHistoryMessageCodec,
        )

        codec = PydanticHistoryMessageCodec()
        messages = []
        for i in range(9):
            messages.extend(
                [ModelRequest([UserPromptPart(str(i))]), ModelResponse([TextPart(str(i))])]
            )
        previous = messages[-2:]
        current = [
            ModelRequest([UserPromptPart("current")]),
            ModelResponse([ToolCallPart("read", {}, "pending")]),
        ]
    else:
        from langchain_core.messages import AIMessage, HumanMessage
        from kedi.agent_adapter.adapters.langchain.history_processing import (
            LangChainHistoryMessageCodec,
        )

        codec = LangChainHistoryMessageCodec()
        messages = []
        for i in range(9):
            messages.extend([HumanMessage(str(i)), AIMessage(str(i))])
        previous = messages[-2:]
        current = [
            HumanMessage("current"),
            AIMessage("", tool_calls=[{"name": "read", "args": {}, "id": "pending"}]),
        ]
    messages.extend(current)
    runner = HistoryProcessorRunner(lambda h: select_history(h, policy), codec)

    async def process():
        with runner.run_scope(None):
            return await runner.process(messages, model_id=None)

    result = asyncio.run(process())
    assert result.messages[-4:] == previous + current
    assert result.changed is (policy == "recent")
    assert len(result.messages) == (6 if policy == "recent" else len(messages))
