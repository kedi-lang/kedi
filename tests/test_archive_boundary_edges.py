from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    NativeToolReturnPart,
    RetryPromptPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RequestUsage

from kedi.agent_adapter.compaction import archive
from kedi.agent_adapter.compaction.langchain_archive import LangChainArtifactHistoryProcessor
from kedi.agent_adapter.compaction.models import NativeHistoryEntryRef
from kedi.agent_adapter.compaction.pydantic_archive import PydanticArtifactHistoryCapability
from kedi.agent_adapter.tool_override import artifact_admission_scope
from kedi.artifacts import ArtifactPolicy
from tests.test_pydantic_history_archive import _history, _processor


def test_archive_requires_valid_enabled_admission(monkeypatch):
    processor = _processor()
    messages = _history(cycles=8)
    for policy in (None, ArtifactPolicy.patch({"enabled": False})):
        with artifact_admission_scope(manager=object(), policy=policy):
            assert processor(messages) is messages
    with (
        artifact_admission_scope(
            manager=SimpleNamespace(artifactize=lambda *a, **kw: "not a handle"),
            policy=ArtifactPolicy.patch({"enabled": True}),
        ),
        pytest.raises(RuntimeError, match="did not create an artifact"),
    ):
        processor(messages)
    snapshot = processor.codec.inspect({processor.codec.state_key: messages})
    monkeypatch.setattr(type(processor.codec), "inspect", lambda *a: replace(snapshot, entries=()))
    assert processor(messages) is messages


def test_archive_minimum_reduction_must_fit():
    with pytest.raises(ValueError, match="minimum_reduction_tokens"):
        archive.HistoryArchiveSettings(
            threshold_tokens=100, preserve_recent_tokens=10, minimum_reduction_tokens=100
        )


def test_archive_selection_respects_barriers_and_balances_repeated_calls():
    processor = _processor()
    base = processor.codec.inspect({processor.codec.state_key: _history(cycles=8)})

    def snapshot(specs):
        entries = tuple(
            NativeHistoryEntryRef(
                index=i,
                entry_id=str(i),
                fingerprint=str(i),
                kind=kind,
                token_estimate=tokens,
                lifecycle_ids=opened,
                closes_lifecycle_ids=closed,
            )
            for i, (kind, tokens, opened, closed) in enumerate(specs)
        )
        return replace(base, entries=entries)

    user = ("user", 1, (), ())
    recent = ("assistant", 500, (), ())
    call = ("tool-call", 300, ("a",), ())
    result = ("tool-result", 300, (), ("a",))
    assert processor._select_window(snapshot([call, result, recent])) is None
    assert processor._select_window(snapshot([user, recent])) is None
    assert processor._select_window(snapshot([user, call, ("tool-result", 1, (), ("a",))])) is None
    assert (
        processor._select_window(snapshot([user, ("tool-result", 600, (), ("orphan",)), recent]))
        is None
    )
    selected = processor._select_window(snapshot([user, call, call, result, result, recent]))
    assert selected is not None and selected.source_tokens == 1200
    assert (
        processor._select_window(snapshot([user, call, ("assistant", 1, (), ()), result, recent]))
        is None
    )
    selected = processor._select_window(
        snapshot(
            [
                user,
                call,
                result,
                ("assistant", 1, (), ()),
                ("tool-call", 1, ("b",), ()),
                ("tool-result", 1, (), ("b",)),
                recent,
            ]
        )
    )
    assert selected is not None and selected.range.end == 3
    assert (
        processor._select_window(snapshot([user, call, result, ("tool-result", 0, (), ()), recent]))
        is not None
    )


def test_checkpoint_token_estimation_is_bounded(monkeypatch):
    processor = _processor()
    messages = _history(cycles=8)
    snapshot = processor.codec.inspect({processor.codec.state_key: messages})
    window = processor._select_window(snapshot)
    assert window is not None
    estimates = iter((10, 11, 12))
    monkeypatch.setattr(archive, "conservative_token_estimate", lambda _: next(estimates))
    checkpoint = processor._checkpoint(
        snapshot,
        window=window,
        selected=messages[window.range.start : window.range.end],
        checkpoint_id="test",
        source_digest="test",
        archive_ref="artifact_1",
    )
    assert checkpoint.checkpoint_tokens == 12


def test_native_archive_outcomes_and_invalid_usage():
    processor = _processor()
    messages = [
        ModelResponse(parts=[ToolCallPart("search", {}, "a")]),
        ModelRequest(
            parts=[
                NativeToolReturnPart("search", "ok", "a"),
                RetryPromptPart("retry", tool_name="search"),
                UserPromptPart("context"),
            ]
        ),
    ]
    assert processor.tool_outcomes(messages) == ("search: calls=1; completed=1, retry=1",)
    assert processor.tool_outcomes([ModelRequest(parts=[])]) == ()
    assert processor.tool_outcomes([object()]) == ()
    for count in (0, True):
        assert (
            processor.provider_input_pressure(
                [
                    ModelResponse(
                        parts=[], provider_name="openai", usage=RequestUsage(input_tokens=count)
                    )
                ]
            )
            is None
        )
    native = LangChainArtifactHistoryProcessor()
    assert (
        native.provider_input_pressure(
            [AIMessage(content="").model_copy(update={"usage_metadata": {"input_tokens": True}})]
        )
        is None
    )
    assert native.tool_outcomes(
        [
            HumanMessage(content="task"),
            AIMessage(content="", tool_calls=[{"name": "search", "args": {}, "id": None}]),
        ]
    ) == ("search: calls=1; ",)


def test_history_capability_allows_absent_model_settings():
    capability = PydanticArtifactHistoryCapability()
    request = ModelRequestContext(
        model=TestModel(),
        messages=[],
        model_settings=None,
        model_request_parameters=ModelRequestParameters(),
    )
    assert asyncio.run(capability.before_model_request(None, request)) is request
    assert capability.get_serialization_name() is None
