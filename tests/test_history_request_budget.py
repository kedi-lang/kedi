from __future__ import annotations

from copy import deepcopy

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, message_to_dict
from pydantic_ai.messages import (
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.usage import RequestUsage

from kedi.agent_adapter import LangChainAdapter, PydanticAdapter
from kedi.agent_adapter.compaction.models import HistoryOrigin
from kedi.agent_adapter.history_provenance import (
    LINEAGE_STATE_KEY,
    HistoryLineage,
    HistoryLineageEntry,
)
from kedi.agent_adapter.request_budget import (
    ModelRequestBudget,
    ModelRequestBudgetError,
    ModelRequestBudgetLimits,
    SupportsHistoryTokenEstimate,
    estimate_content_tokens,
    capture_history_usage,
    HISTORY_USAGE_KEY,
)


@pytest.mark.parametrize("adapter", [PydanticAdapter, LangChainAdapter])
def test_history_estimator_is_an_explicit_optional_adapter_contract(adapter):
    assert isinstance(adapter(), SupportsHistoryTokenEstimate)


def test_pydantic_history_counts_current_instructions_not_old_snapshots():
    state = {
        "pydantic_messages": ModelMessagesTypeAdapter.dump_python(
            [
                ModelRequest(parts=[UserPromptPart("update")], instructions="policy " * 2000)
                for _ in range(30)
            ],
            mode="json",
        )
    }
    before = deepcopy(state)
    estimated = PydanticAdapter.estimate_history_tokens(state)
    assert 14_000 < estimated < 25_000
    assert estimated < estimate_content_tokens(state) / 10
    assert state == before
    ModelRequestBudget().reserve("next", resume_state=state, history_tokens=estimated).finish()
    with pytest.raises(ModelRequestBudgetError, match="safe context"):
        ModelRequestBudget().reserve("next", resume_state=state)


@pytest.mark.parametrize("surface", ["pydantic", "langchain"])
@pytest.mark.parametrize("reported", [0, 700])
def test_opaque_reasoning_uses_usage_only_when_available(surface, reported):
    encrypted = "x" * 50_000
    if surface == "pydantic":
        state = {
            "pydantic_messages": ModelMessagesTypeAdapter.dump_python(
                [
                    ModelResponse(
                        parts=[
                            ThinkingPart(content="Visible reasoning.", signature=encrypted),
                            TextPart("done"),
                        ],
                        usage=RequestUsage(output_tokens=reported),
                    )
                ],
                mode="json",
            )
        }
        estimate = PydanticAdapter.estimate_history_tokens
    else:
        state = {
            "langchain_messages": [
                message_to_dict(
                    AIMessage(
                        content=[
                            {"type": "reasoning", "encrypted_content": encrypted},
                            {"type": "text", "text": "done"},
                        ],
                        usage_metadata={
                            "input_tokens": 100,
                            "output_tokens": reported,
                            "total_tokens": 100 + reported,
                        },
                    )
                )
            ]
        }
        estimate = LangChainAdapter.estimate_history_tokens
    before = deepcopy(state)
    tokens = estimate(state)
    assert state == before
    if reported:
        assert reported < tokens < 2_000
    else:
        assert tokens > 50_000


@pytest.mark.parametrize("surface", ["pydantic", "langchain"])
def test_history_visible_tool_content_is_never_filtered_by_metadata_key(surface):
    content = {"metadata": "large fact " * 20_000, "signature": "actual evidence"}
    if surface == "pydantic":
        state = {
            "pydantic_messages": ModelMessagesTypeAdapter.dump_python(
                [ModelRequest(parts=[ToolReturnPart("read", content, tool_call_id="one")])],
                mode="json",
            )
        }
        estimate = PydanticAdapter.estimate_history_tokens
    else:
        state = {
            "langchain_messages": [
                message_to_dict(
                    ToolMessage(
                        content=str(content),
                        tool_call_id="one",
                        artifact="not model content " * 20_000,
                    )
                )
            ]
        }
        estimate = LangChainAdapter.estimate_history_tokens
    tokens = estimate(state)
    assert tokens > 200_000
    with pytest.raises(ModelRequestBudgetError, match="safe context"):
        ModelRequestBudget().reserve("next", resume_state=state, history_tokens=tokens)


def test_langchain_runtime_metadata_and_tool_artifacts_are_not_model_text():
    plain = {"langchain_messages": [message_to_dict(AIMessage(content="ready"))]}
    enriched = deepcopy(plain)
    enriched["langchain_messages"][0]["data"].update(
        response_metadata={"diagnostics": "large " * 1000},
        usage_metadata={"input_tokens": 100, "output_tokens": 10, "total_tokens": 110},
        id="tracking-id",
    )
    assert LangChainAdapter.estimate_history_tokens(
        plain
    ) == LangChainAdapter.estimate_history_tokens(enriched)
    empty_tool = {
        "langchain_messages": [message_to_dict(ToolMessage(content="ready", tool_call_id="a"))]
    }
    full_tool = deepcopy(empty_tool)
    full_tool["langchain_messages"][0]["data"]["artifact"] = "local-only" * 100_000
    assert LangChainAdapter.estimate_history_tokens(
        empty_tool
    ) == LangChainAdapter.estimate_history_tokens(full_tool)


@pytest.mark.parametrize(
    "adapter,key",
    [(PydanticAdapter, "pydantic_messages"), (LangChainAdapter, "langchain_messages")],
)
def test_local_history_lineage_is_not_charged_as_model_context(adapter, key):
    from kedi.lang.ast import Span

    location = Span(0, 100, 1, 1, 2, 1, source_path="examples/review/main.kedi")
    state = {
        key: [],
        LINEAGE_STATE_KEY: HistoryLineage(
            tuple(
                HistoryLineageEntry(
                    str(index),
                    (
                        HistoryOrigin(
                            kind="tool-call",
                            run_id=str(index),
                            source_location=location,
                            tool_name="read_ledger",
                            tool_call_id=str(index),
                        ),
                    ),
                )
                for index in range(600)
            )
        ),
    }
    before = deepcopy(state)
    estimate = adapter.estimate_history_tokens(state)
    assert estimate == adapter.estimate_history_tokens({key: []})
    assert state == before
    ModelRequestBudget().reserve("next", resume_state=state, history_tokens=estimate).finish()
    with pytest.raises(ModelRequestBudgetError, match="safe context"):
        ModelRequestBudget().reserve("next", resume_state=state)


@pytest.mark.parametrize(
    "adapter,key",
    [(PydanticAdapter, "pydantic_messages"), (LangChainAdapter, "langchain_messages")],
)
@pytest.mark.parametrize("extra_key", [LINEAGE_STATE_KEY, "unknown_extension"])
def test_unrecognized_extra_state_is_not_blindly_exempted(adapter, key, extra_key):
    state = {key: [], extra_key: "actual unknown content " * 10_000}
    assert adapter.estimate_history_tokens(state) > 200_000


@pytest.mark.parametrize(
    "adapter,key",
    [(PydanticAdapter, "pydantic_messages"), (LangChainAdapter, "langchain_messages")],
)
@pytest.mark.parametrize("raw", [None, "invalid", [{"unknown": "payload"}]])
def test_malformed_history_keeps_conservative_fallback(adapter, key, raw):
    state = {key: raw}
    assert adapter.estimate_history_tokens(state) == estimate_content_tokens(state)


def test_unicode_history_estimate_counts_utf8_without_dividing_by_four():
    text = "\u4e2d\u6587" * 1000
    state = {"langchain_messages": [message_to_dict(HumanMessage(text))]}
    assert LangChainAdapter.estimate_history_tokens(state) >= len(text.encode("utf-8"))


@pytest.mark.parametrize("invalid", [-1, True, 2.5])
def test_budget_rejects_invalid_adapter_estimate(invalid):
    with pytest.raises(ValueError, match="history_tokens"):
        ModelRequestBudget().reserve("prompt", history_tokens=invalid)


def test_projected_history_still_reserves_cumulative_and_request_limits():
    budget = ModelRequestBudget(
        ModelRequestBudgetLimits(
            context_limit_tokens=10_000,
            safe_context_utilization=1,
            output_reserve_tokens=0,
            cumulative_input_tokens=5_000,
            request_limit=5,
        )
    )
    budget.reserve("prompt", history_tokens=1000).finish()
    with pytest.raises(ModelRequestBudgetError, match="cumulative"):
        budget.reserve("prompt", history_tokens=1000)


@pytest.mark.parametrize("surface", ["pydantic", "langchain"])
def test_successful_capture_anchors_usage_to_unchanged_history(surface):
    from types import SimpleNamespace
    from kedi.agent_adapter.conversation import AdapterConversationContext

    text = "completed known content " * 10_000
    conversation = AdapterConversationContext(resume_state={})
    if surface == "pydantic":
        from kedi.agent_adapter.adapters.pydantic.messages import _capture_conversation_messages

        messages = [
            ModelRequest([UserPromptPart(text)]),
            ModelResponse(
                [TextPart("done")], usage=RequestUsage(input_tokens=50_000, output_tokens=100)
            ),
            ModelRequest([ToolReturnPart("final", "ack", tool_call_id="end")]),
        ]
        _capture_conversation_messages(
            conversation,
            SimpleNamespace(all_messages=lambda: messages, new_messages=lambda: messages),
        )
        adapter, key = PydanticAdapter, "pydantic_messages"
    else:
        messages = [
            HumanMessage(text),
            AIMessage(
                "done",
                usage_metadata={
                    "input_tokens": 50_000,
                    "output_tokens": 100,
                    "total_tokens": 50_100,
                },
            ),
            ToolMessage("ack", tool_call_id="end"),
        ]
        adapter, key = LangChainAdapter, "langchain_messages"
        instance = adapter()
        with instance.conversation_scope(conversation):
            instance._capture_conversation_messages({"messages": messages})
    state = conversation.result_state
    assert HISTORY_USAGE_KEY in state
    before = deepcopy(state)
    measured = adapter.estimate_history_tokens(state)
    assert 50_100 < measured < 51_000
    assert state == before
    ModelRequestBudget().reserve("next", resume_state=state, history_tokens=measured).finish()
    without = {k: v for k, v in state.items() if k != HISTORY_USAGE_KEY}
    assert adapter.estimate_history_tokens(without) > 200_000
    state[key].reverse()
    assert adapter.estimate_history_tokens(state) > 200_000
    state[key].reverse()
    state[key].append(deepcopy(state[key][0]))
    assert adapter.estimate_history_tokens(state) > 400_000
    state[key].pop()
    state["unknown_extension"] = "x" * 200_000
    assert adapter.estimate_history_tokens(state) > 250_000


@pytest.mark.parametrize("input_tokens,output_tokens", [(0, 1), (10, 0), (-1, 10), (True, 10)])
def test_missing_or_invalid_usage_does_not_create_a_checkpoint(input_tokens, output_tokens):
    state = {"messages": []}
    capture_history_usage(state, "messages", input_tokens=input_tokens, output_tokens=output_tokens)
    assert HISTORY_USAGE_KEY not in state


@pytest.mark.parametrize(
    "adapter,key",
    [(PydanticAdapter, "pydantic_messages"), (LangChainAdapter, "langchain_messages")],
)
def test_restored_untyped_checkpoint_cannot_override_conservative_estimate(adapter, key):
    state = {key: [], HISTORY_USAGE_KEY: {"tokens": 1, "digest": "x" * 200_000}}
    assert adapter.estimate_history_tokens(state) > 200_000


def test_measured_checkpoint_is_invalid_for_another_lane_key_or_modified_value():
    from kedi.agent_adapter.request_budget import measured_history_tokens

    state = {"pydantic_messages": [{"content": "known"}]}
    capture_history_usage(state, "pydantic_messages", input_tokens=100, output_tokens=10)
    assert measured_history_tokens(state, "pydantic_messages") == 112
    assert measured_history_tokens(state, "langchain_messages") is None
    state["pydantic_messages"][0]["content"] = "modified"
    assert measured_history_tokens(state, "pydantic_messages") is None


@pytest.mark.parametrize("key", ["pydantic_messages", "langchain_messages"])
def test_usage_checkpoint_roundtrips_storage_without_losing_native_history(key):
    import json
    from kedi.agent_adapter.request_budget import (
        encode_history_usage_state,
        decode_history_usage_state,
    )

    state = {key: [{"content": "known"}]}
    capture_history_usage(state, key, input_tokens=100, output_tokens=10)
    restored = decode_history_usage_state(json.loads(json.dumps(encode_history_usage_state(state))))
    assert restored == state
    assert decode_history_usage_state({key: []}) == {key: []}


@pytest.mark.parametrize(
    "mutation", ["changed-history", "negative-tokens", "unknown-key", "missing-digest"]
)
def test_stored_usage_checkpoint_is_validated_before_trusting_its_estimate(mutation):
    from kedi.agent_adapter.request_budget import (
        encode_history_usage_state,
        decode_history_usage_state,
    )

    state = {"pydantic_messages": [{"content": "known"}]}
    capture_history_usage(state, "pydantic_messages", input_tokens=100, output_tokens=10)
    stored = encode_history_usage_state(state)
    if mutation == "changed-history":
        stored["pydantic_messages"] = [{"content": "different"}]
    elif mutation == "negative-tokens":
        stored[HISTORY_USAGE_KEY]["tokens"] = -1
    elif mutation == "unknown-key":
        stored[HISTORY_USAGE_KEY]["state_key"] = "unknown"
    else:
        del stored[HISTORY_USAGE_KEY]["digest"]
    with pytest.raises(ValueError, match="history usage checkpoint"):
        decode_history_usage_state(stored)
