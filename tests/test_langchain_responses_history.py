import asyncio
from copy import deepcopy
from types import SimpleNamespace

import pytest
from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    messages_to_dict,
)
from langchain_openai import ChatOpenAI

from kedi.agent_adapter.adapters.langchain.runtime import (
    LangChainRunMiddleware,
    langchain_run_scope,
)
from kedi.agent_adapter.compaction import (
    HistoryArchiveSettings,
    LangChainArtifactHistoryProcessor,
    LangChainHistoryCodec,
)
from kedi.agent_adapter.conversation import AdapterConversationContext
from kedi.agent_adapter.subagents import SubagentConversationContext
from kedi.agent_adapter.tools.scope import artifact_admission_scope
from kedi.artifacts import ArtifactManager, ArtifactPolicy


def response(index, *, signed=False):
    content = [
        {
            "type": "function_call",
            "id": f"fc_{index}",
            "call_id": f"call_{index}",
            "name": "next_page",
            "arguments": f'{{"index":{index}}}',
        }
    ]
    if signed:
        content.insert(
            0,
            {"type": "reasoning", "id": f"rs_{index}", "encrypted_content": "synthetic-signature"},
        )
    return AIMessage(
        content=content,
        tool_calls=[{"name": "next_page", "args": {"index": index}, "id": f"call_{index}"}],
    )


@pytest.mark.parametrize("serialized", [False, True])
@pytest.mark.parametrize("signed", [False, True])
def test_responses_call_uses_call_id_and_deduplicates_normalized_tool_call(serialized, signed):
    messages = [response(1, signed=signed), ToolMessage(content="page", tool_call_id="call_1")]
    original = deepcopy(messages)
    raw = messages_to_dict(messages) if serialized else messages
    snapshot = LangChainHistoryCodec().inspect({"langchain_messages": raw})
    assert snapshot.entries[0].kind == "tool-call"
    assert snapshot.entries[0].lifecycle_ids == ("call_1",)
    assert snapshot.entries[0].provider_required is signed
    assert not snapshot.entries[0].unknown
    assert snapshot.entries[1].closes_lifecycle_ids == ("call_1",)
    assert messages == original


@pytest.mark.parametrize("kind", ["function_call", "function_call_output"])
def test_responses_item_id_cannot_replace_missing_call_id(kind):
    message = AIMessage(content=[{"type": kind, "id": "fc_not_a_call_id"}])
    entry = LangChainHistoryCodec().inspect({"langchain_messages": [message]}).entries[0]
    assert entry.unknown
    assert entry.lifecycle_ids == ()
    assert entry.closes_lifecycle_ids == ()


def test_responses_output_closes_call_id_not_item_id():
    message = ToolMessage(
        content=[{"type": "function_call_output", "call_id": "call_1", "output": "ok"}],
        tool_call_id="call_1",
    )
    entry = LangChainHistoryCodec().inspect({"langchain_messages": [message]}).entries[0]
    assert not entry.unknown
    assert entry.closes_lifecycle_ids == ("call_1",)


def test_exact_archival_handles_responses_calls_without_removing_open_tail():
    messages = [HumanMessage(content="Keep the first receipt and collect the remaining pages.")]
    for i in range(8):
        messages.extend(
            [
                response(i, signed=True),
                ToolMessage(content="page evidence " * 300, tool_call_id=f"call_{i}"),
            ]
        )
    messages.append(response(8, signed=True))
    original = deepcopy(messages)
    processor = LangChainArtifactHistoryProcessor(
        HistoryArchiveSettings(
            threshold_tokens=2000,
            preserve_recent_tokens=600,
            minimum_reduction_tokens=400,
        )
    )
    manager = ArtifactManager()
    try:
        with artifact_admission_scope(manager=manager, policy=ArtifactPolicy()):
            result = processor(messages)
        assert len(result) < len(messages)
        assert result[0] is messages[0]
        assert result[-1] is messages[-1]
        assert result[1].additional_kwargs.get("kedi_compaction_checkpoint")
        ref = manager.search()[0].ref_id
        payload = manager.get_value(ref)
        removed_count = len(payload["messages"])
        assert payload["messages"] == messages_to_dict(messages[1 : 1 + removed_count])
        assert messages == original
    finally:
        manager.close()


@pytest.mark.parametrize("subagent", [False, True])
@pytest.mark.parametrize("next_instructions", ["original", "changed", None, "legacy"])
def test_instruction_contract_controls_continuation_without_mutating_model(
    subagent, next_instructions
):
    model = ChatOpenAI(
        model="gpt-5.2",
        api_key="fixture",
        use_responses_api=True,
        use_previous_response_id=True,
        model_kwargs={"previous_response_id": "resp_old"},
    )
    conversation = (
        SubagentConversationContext(conversation_id="test")
        if subagent
        else AdapterConversationContext()
    )
    middleware = LangChainRunMiddleware(archive=None, tool_retries=0, conversation=conversation)
    seen = []

    async def handler(request):
        seen.append(request)
        return SimpleNamespace(result=[AIMessage(content="done")])

    async def run():
        with langchain_run_scope(None) as state:
            await middleware.awrap_model_call(
                ModelRequest(
                    model=model,
                    messages=[HumanMessage(content="first")],
                    system_message=SystemMessage(content="original"),
                ),
                handler,
            )
            if next_instructions != "legacy":
                conversation.resume_state["langchain_instruction_digest"] = state.instruction_digest
        conversation.resume_state["langchain_messages"] = messages_to_dict(
            [HumanMessage(content="prior")]
        )
        previous = deepcopy(conversation.resume_state)
        with langchain_run_scope(None) as state:
            for _ in range(2):
                await middleware.awrap_model_call(
                    ModelRequest(
                        model=model,
                        messages=[HumanMessage(content="followup")],
                        system_message=(
                            SystemMessage(content=next_instructions) if next_instructions else None
                        ),
                        model_settings={"previous_response_id": "resp_old"},
                    ),
                    handler,
                )
            assert state.instructions_changed is (next_instructions != "original")
        assert conversation.resume_state == previous
        assert conversation.result_state == {}
        if not subagent:
            assert not conversation.cache_prefix_changed

    asyncio.run(run())
    assert model.use_previous_response_id is True
    assert model.model_kwargs == {"previous_response_id": "resp_old"}
    assert seen[0].model is model
    if next_instructions == "original":
        assert seen[1].model is model
        assert seen[2].model is model
    else:
        assert seen[1].model.use_previous_response_id is False
        assert seen[2].model.use_previous_response_id is True
        for request in seen[1:]:
            assert "previous_response_id" not in request.model.model_kwargs
            assert "previous_response_id" not in request.model_settings
