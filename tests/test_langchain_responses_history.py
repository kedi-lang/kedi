from copy import deepcopy

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, messages_to_dict

from kedi.agent_adapter.compaction import (
    HistoryArchiveSettings,
    LangChainArtifactHistoryProcessor,
    LangChainHistoryCodec,
)
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
