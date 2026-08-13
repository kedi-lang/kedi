from __future__ import annotations

import asyncio
import copy
import json
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    ChatMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    message_to_dict,
    messages_from_dict,
)
from pydantic_ai import BinaryContent
from pydantic_ai.messages import (
    CompactionPart,
    FilePart,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    NativeToolCallPart,
    NativeToolReturnPart,
    RetryPromptPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolAvailabilityDeltaPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from kedi.agent_adapter.compaction import (
    CheckpointAcceptancePolicy,
    CheckpointAnchor,
    CompactionCapabilities,
    CompactionCheckpoint,
    CompactionOutcome,
    CompactionPreflightError,
    CompactionResult,
    CompactionUsage,
    CompactionValidationError,
    ConversationCompactionTransaction,
    HistoryCodecRegistry,
    HistoryRange,
    HistoryReductionPolicy,
    HistorySnapshot,
    HistoryUnit,
    HistoryUnitGroup,
    KediCompactionTransaction,
    LangChainHistoryCodec,
    NativeCompactionBackend,
    NativeHistoryEntryRef,
    PydanticHistoryCodec,
    resolve_compaction_backend,
)
from kedi.agent_adapter.compaction.langchain_codec import (
    _langchain_content_is_known,
    _langchain_content_kinds,
    _langchain_content_lifecycle,
    _langchain_entry_kind,
    _langchain_has_interrupt_metadata,
    _langchain_has_reasoning,
    _langchain_has_signed_reasoning,
    _langchain_has_text,
)
from kedi.agent_adapter.compaction.langchain_codec import (
    _storage_format as _langchain_storage_format,
)
from kedi.agent_adapter.compaction.pydantic_codec import (
    _pydantic_entry_kind,
)
from kedi.agent_adapter.compaction.pydantic_codec import (
    _storage_format as _pydantic_storage_format,
)
from kedi.agent_adapter.compaction.selection import (
    _eligible_segments,
    _with_reason,
    plan_history_reduction,
)
from kedi.agent_adapter.compaction.validation import (
    _secret_like_values,
    _validate_checkpoint_acceptance,
    validate_compaction_transaction,
)
from kedi.agent_adapter.conversation import ConversationState
from tests.telemetry_helpers import installed_recording_backend


def _pydantic_history() -> list[dict[str, Any]]:
    messages = [
        ModelRequest(parts=[SystemPromptPart("Stable system instructions")]),
        ModelRequest(parts=[UserPromptPart("Inspect the repository and report the decision.")]),
        ModelResponse(
            parts=[
                ThinkingPart("I need the source file.", signature="signed-reasoning-1"),
                ToolCallPart("read_file", {"path": "src/app.py"}, "call-read"),
            ]
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    "read_file",
                    {
                        "artifact_ref": "tool_call_result_1",
                        "preview": "class Application: ...",
                    },
                    "call-read",
                )
            ]
        ),
        ModelResponse(
            parts=[
                TextPart("The file is loaded; delegating focused review."),
                ToolCallPart(
                    "delegate_task",
                    {"subagent": "reviewer", "task": "Review src/app.py"},
                    "call-subagent",
                ),
            ]
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    "delegate_task",
                    {"task_summary": "No blocking defect", "final_result": {"risk": "low"}},
                    "call-subagent",
                )
            ]
        ),
        ModelResponse(parts=[TextPart('{"decision":"ship","risk":"low"}')]),
        ModelRequest(parts=[UserPromptPart("Apply the accepted cleanup workflow.")]),
        ModelResponse(
            parts=[ToolCallPart("apply_patch", {"patch": "first attempt"}, "call-patch-1")]
        ),
        ModelRequest(
            parts=[
                RetryPromptPart(
                    "Patch context did not match; retry with current source.",
                    tool_name="apply_patch",
                    tool_call_id="call-patch-1",
                )
            ]
        ),
        ModelResponse(
            parts=[
                ToolCallPart(
                    "run_dynamic_workflow",
                    {"workflow": "cleanup", "steps": ["inspect", "patch", "test"]},
                    "call-workflow",
                )
            ]
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    "run_dynamic_workflow",
                    {"status": "complete", "changed": ["src/app.py"]},
                    "call-workflow",
                )
            ]
        ),
        ModelResponse(parts=[TextPart("Cleanup completed and tests passed.")]),
        ModelRequest(parts=[UserPromptPart("Delete the production database now.")]),
        ModelResponse(
            parts=[
                ThinkingPart("This mutating action needs approval."),
                ToolCallPart(
                    "delete_database",
                    {"database": "production"},
                    "call-awaiting-approval",
                ),
            ]
        ),
    ]
    return ModelMessagesTypeAdapter.dump_python(messages, mode="json")


def _langchain_history() -> list[dict[str, Any]]:
    messages = [
        SystemMessage(content="Stable system instructions"),
        HumanMessage(content="Inspect the repository and report the decision."),
        AIMessage(
            content=[
                {
                    "type": "reasoning",
                    "reasoning": "I need the source file.",
                    "signature": "signed-reasoning-1",
                }
            ],
            tool_calls=[{"name": "read_file", "args": {"path": "src/app.py"}, "id": "call-read"}],
        ),
        ToolMessage(
            content=json.dumps(
                {
                    "artifact_ref": "tool_call_result_1",
                    "preview": "class Application: ...",
                }
            ),
            tool_call_id="call-read",
        ),
        AIMessage(
            content="The file is loaded; delegating focused review.",
            tool_calls=[
                {
                    "name": "delegate_task",
                    "args": {"subagent": "reviewer", "task": "Review src/app.py"},
                    "id": "call-subagent",
                }
            ],
        ),
        ToolMessage(
            content=json.dumps(
                {"task_summary": "No blocking defect", "final_result": {"risk": "low"}}
            ),
            tool_call_id="call-subagent",
        ),
        AIMessage(content='{"decision":"ship","risk":"low"}'),
        HumanMessage(content="Apply the accepted cleanup workflow."),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "apply_patch",
                    "args": {"patch": "first attempt"},
                    "id": "call-patch-1",
                }
            ],
        ),
        ToolMessage(
            content="Patch context did not match; retry with current source.",
            tool_call_id="call-patch-1",
            status="error",
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "run_dynamic_workflow",
                    "args": {"workflow": "cleanup", "steps": ["inspect", "patch", "test"]},
                    "id": "call-workflow",
                }
            ],
        ),
        ToolMessage(
            content='{"status":"complete","changed":["src/app.py"]}',
            tool_call_id="call-workflow",
        ),
        AIMessage(content="Cleanup completed and tests passed."),
        HumanMessage(content="Delete the production database now."),
        AIMessage(
            content=[{"type": "reasoning", "reasoning": "This action needs approval."}],
            tool_calls=[
                {
                    "name": "delete_database",
                    "args": {"database": "production"},
                    "id": "call-awaiting-approval",
                }
            ],
        ),
    ]
    return [message_to_dict(message) for message in messages]


def _policy(total_tokens: int) -> HistoryReductionPolicy:
    return HistoryReductionPolicy(
        threshold_tokens=max(1, total_tokens - 1),
        target_tokens=max(1, total_tokens // 2),
        preserve_recent_tokens=1,
        checkpoint_budget_tokens=400,
        minimum_reduction_tokens=1,
    )


def _checkpoint(prepared: Any) -> CompactionCheckpoint:
    assert prepared.plan.window is not None
    window = prepared.plan.window
    source = prepared.snapshot.entries[window.native_range.start : window.native_range.end]
    artifact_refs = tuple(sorted({ref_id for entry in source for ref_id in entry.artifact_refs}))
    return CompactionCheckpoint(
        checkpoint_id="checkpoint_fixture_1",
        version=1,
        source_start_id=source[0].entry_id,
        source_end_id=source[-1].entry_id,
        source_digest=prepared.snapshot.source_digest,
        summary=(
            "The repository review completed with a low-risk ship decision, and the accepted "
            "cleanup workflow finished successfully."
        ),
        durable_facts=("src/app.py was reviewed",),
        decisions=("Ship the current implementation",),
        artifact_refs=artifact_refs,
        source_tokens=window.source_tokens,
        checkpoint_tokens=min(
            prepared.plan.policy.checkpoint_budget_tokens,
            window.source_tokens - 1,
        ),
    )


@pytest.mark.parametrize(
    ("codec", "state"),
    [
        (PydanticHistoryCodec(), {"pydantic_messages": _pydantic_history()}),
        (LangChainHistoryCodec(), {"langchain_messages": _langchain_history()}),
    ],
)
def test_native_history_fixtures_freeze_groups_boundaries_and_open_state(
    codec: Any,
    state: dict[str, object],
) -> None:
    transaction = KediCompactionTransaction(codec)
    snapshot = transaction.processor.inspect(state, codec=codec)

    assert [group.entry_start for group in snapshot.groups] == [0, 1, 7, 13]
    assert [group.entry_end for group in snapshot.groups] == [1, 7, 13, 15]
    assert snapshot.groups[0].protected_reasons == ("immutable-prefix",)
    assert snapshot.groups[1].closed is True
    assert snapshot.groups[1].protected_reasons == ("provider-required",)
    assert snapshot.groups[2].closed is True
    assert snapshot.groups[3].closed is False
    assert "open-lifecycle" in snapshot.groups[3].protected_reasons
    assert "call-awaiting-approval" in snapshot.groups[3].lifecycle_ids
    assert snapshot.entries[2].provider_required is True
    assert "tool_call_result_1" in snapshot.entries[3].searchable_text

    prepared = transaction.prepare(state, _policy(snapshot.total_tokens))
    assert prepared.plan.window is not None
    assert prepared.plan.window.native_range.start == 7
    assert prepared.plan.window.native_range.end == 13
    assert prepared.plan.groups[-1].protected is True
    assert "latest-cycle" in prepared.plan.groups[-1].protected_reasons


@pytest.mark.parametrize(
    ("codec", "state_key", "history"),
    [
        (PydanticHistoryCodec(), "pydantic_messages", _pydantic_history()),
        (LangChainHistoryCodec(), "langchain_messages", _langchain_history()),
    ],
)
def test_transactional_splice_preserves_exact_native_prefix_tail_and_source(
    codec: Any,
    state_key: str,
    history: list[object],
) -> None:
    state: dict[str, object] = {state_key: history, "unrelated": {"keep": True}}
    transaction = KediCompactionTransaction(codec)
    initial = transaction.processor.inspect(state, codec=codec)
    prepared = transaction.prepare(state, _policy(initial.total_tokens))
    checkpoint = _checkpoint(prepared)
    assert prepared.plan.window is not None
    native_range = prepared.plan.window.native_range
    prefix = history[: native_range.start]
    tail = history[native_range.end :]

    result = transaction.commit(state, prepared=prepared, checkpoint=checkpoint)

    assert result.processed is not None
    updated = result.processed.state[state_key]
    assert isinstance(updated, list)
    assert all(value is original for value, original in zip(updated[: len(prefix)], prefix))
    updated_tail = updated[len(prefix) + 1 :]
    assert all(value is original for value, original in zip(updated_tail, tail))
    assert state[state_key] is history
    assert state["unrelated"] == {"keep": True}
    assert result.outcome.applied is True
    assert result.outcome.backend == "kedi"
    assert result.outcome.result_tokens is not None
    assert result.outcome.result_tokens < initial.total_tokens


def test_kedi_checkpoints_are_provider_replayable_context_messages() -> None:
    pydantic_state: dict[str, object] = {"pydantic_messages": _pydantic_history()}
    pydantic_transaction = KediCompactionTransaction(PydanticHistoryCodec())
    pydantic_snapshot = pydantic_transaction.processor.inspect(
        pydantic_state,
        codec=pydantic_transaction.codec,
    )
    pydantic_prepared = pydantic_transaction.prepare(
        pydantic_state,
        _policy(pydantic_snapshot.total_tokens),
    )
    pydantic_result = pydantic_transaction.commit(
        pydantic_state,
        prepared=pydantic_prepared,
        checkpoint=_checkpoint(pydantic_prepared),
    )
    assert pydantic_result.processed is not None
    pydantic_messages = ModelMessagesTypeAdapter.validate_python(
        pydantic_result.processed.state["pydantic_messages"]
    )
    pydantic_checkpoint = next(
        message
        for message in pydantic_messages
        if message.metadata and message.metadata.get("kedi_compaction_checkpoint")
    )
    assert isinstance(pydantic_checkpoint, ModelRequest)
    assert len(pydantic_checkpoint.parts) == 1
    assert isinstance(pydantic_checkpoint.parts[0], UserPromptPart)
    assert "<kedi_compaction_checkpoint>" in pydantic_checkpoint.parts[0].content

    langchain_state: dict[str, object] = {"langchain_messages": _langchain_history()}
    langchain_transaction = KediCompactionTransaction(LangChainHistoryCodec())
    langchain_snapshot = langchain_transaction.processor.inspect(
        langchain_state,
        codec=langchain_transaction.codec,
    )
    langchain_prepared = langchain_transaction.prepare(
        langchain_state,
        _policy(langchain_snapshot.total_tokens),
    )
    langchain_result = langchain_transaction.commit(
        langchain_state,
        prepared=langchain_prepared,
        checkpoint=_checkpoint(langchain_prepared),
    )
    assert langchain_result.processed is not None
    langchain_messages = messages_from_dict(
        cast(list[dict[str, Any]], langchain_result.processed.state["langchain_messages"])
    )
    langchain_checkpoint = next(
        message
        for message in langchain_messages
        if message.additional_kwargs.get("kedi_compaction_checkpoint")
    )
    assert isinstance(langchain_checkpoint, HumanMessage)
    assert "<kedi_compaction_checkpoint>" in str(langchain_checkpoint.content)


@pytest.mark.parametrize(
    ("codec", "state_key", "history"),
    [
        (
            PydanticHistoryCodec(),
            "pydantic_messages",
            ModelMessagesTypeAdapter.validate_python(_pydantic_history()),
        ),
        (
            LangChainHistoryCodec(),
            "langchain_messages",
            messages_from_dict(_langchain_history()),
        ),
    ],
)
def test_transactional_splice_preserves_native_object_storage_format(
    codec: Any,
    state_key: str,
    history: list[object],
) -> None:
    state: dict[str, object] = {state_key: history}
    transaction = KediCompactionTransaction(codec)
    snapshot = transaction.processor.inspect(state, codec=codec)
    assert snapshot.storage_format == "native"
    prepared = transaction.prepare(state, _policy(snapshot.total_tokens))
    assert prepared.plan.window is not None
    native_range = prepared.plan.window.native_range

    result = transaction.commit(
        state,
        prepared=prepared,
        checkpoint=_checkpoint(prepared),
    )

    assert result.processed is not None
    updated = result.processed.state[state_key]
    assert isinstance(updated, list)
    assert all(
        retained is source
        for retained, source in zip(updated[: native_range.start], history[: native_range.start])
    )
    assert all(
        retained is source
        for retained, source in zip(
            updated[native_range.start + 1 :],
            history[native_range.end :],
        )
    )
    assert not isinstance(updated[native_range.start], dict)


@pytest.mark.parametrize(
    ("codec", "state_key", "history"),
    [
        (PydanticHistoryCodec(), "pydantic_messages", _pydantic_history()),
        (LangChainHistoryCodec(), "langchain_messages", _langchain_history()),
    ],
)
def test_stale_or_invalid_checkpoint_is_a_no_mutation_failure(
    codec: Any,
    state_key: str,
    history: list[object],
) -> None:
    state: dict[str, object] = {state_key: history}
    transaction = KediCompactionTransaction(codec)
    snapshot = transaction.processor.inspect(state, codec=codec)
    prepared = transaction.prepare(state, _policy(snapshot.total_tokens))
    checkpoint = _checkpoint(prepared)

    stale = {state_key: [*history, history[-1]]}
    with pytest.raises(CompactionValidationError, match="stale"):
        transaction.commit(stale, prepared=prepared, checkpoint=checkpoint)
    assert stale[state_key][-1] is history[-1]

    oversized = replace(
        checkpoint,
        checkpoint_tokens=prepared.plan.policy.checkpoint_budget_tokens + 1,
    )
    with pytest.raises(CompactionValidationError, match="token budget"):
        transaction.commit(state, prepared=prepared, checkpoint=oversized)
    assert state[state_key] is history


def test_tampered_plan_is_rejected_without_mutation() -> None:
    history = _pydantic_history()
    state: dict[str, object] = {"pydantic_messages": history}
    transaction = KediCompactionTransaction(PydanticHistoryCodec())
    snapshot = transaction.processor.inspect(state, codec=transaction.codec)
    prepared = transaction.prepare(state, _policy(snapshot.total_tokens))
    assert prepared.plan.window is not None
    tampered = replace(
        prepared,
        plan=replace(
            prepared.plan,
            unchanged_tail_fingerprints=(),
        ),
    )

    with pytest.raises(CompactionValidationError, match="deterministic source plan"):
        transaction.commit(
            state,
            prepared=tampered,
            checkpoint=_checkpoint(prepared),
        )

    assert state["pydantic_messages"] is history


@pytest.mark.parametrize(
    ("codec", "state_key", "history"),
    [
        (PydanticHistoryCodec(), "pydantic_messages", _pydantic_history()),
        (LangChainHistoryCodec(), "langchain_messages", _langchain_history()),
    ],
)
def test_every_native_entry_mutation_invalidates_prepared_transaction(
    codec: Any,
    state_key: str,
    history: list[object],
) -> None:
    state: dict[str, object] = {state_key: history}
    transaction = KediCompactionTransaction(codec)
    snapshot = transaction.processor.inspect(state, codec=codec)
    prepared = transaction.prepare(state, _policy(snapshot.total_tokens))
    checkpoint = _checkpoint(prepared)

    for index in range(len(history)):
        mutated_history = copy.deepcopy(history)
        entry = mutated_history[index]
        assert isinstance(entry, dict)
        cast("dict[str, Any]", entry)["kedi_mutation_probe"] = index
        mutated_state: dict[str, object] = {state_key: mutated_history}
        with pytest.raises(CompactionValidationError, match="stale"):
            transaction.commit(
                mutated_state,
                prepared=prepared,
                checkpoint=checkpoint,
            )


@pytest.mark.parametrize(
    ("codec", "state"),
    [
        (PydanticHistoryCodec(), {"pydantic_messages": _pydantic_history()}),
        (LangChainHistoryCodec(), {"langchain_messages": _langchain_history()}),
    ],
)
def test_processor_is_deterministic_across_restart_and_threads(
    codec: Any,
    state: dict[str, object],
) -> None:
    transaction = KediCompactionTransaction(codec)
    first_snapshot = transaction.processor.inspect(state, codec=codec)
    policy = _policy(first_snapshot.total_tokens)
    first = transaction.prepare(state, policy)

    serialized = json.loads(json.dumps(state))
    restarted = transaction.prepare(serialized, policy)
    assert restarted.snapshot.source_digest == first.snapshot.source_digest
    assert restarted.plan == first.plan

    with ThreadPoolExecutor(max_workers=4) as executor:
        prepared = list(executor.map(lambda _: transaction.prepare(state, policy), range(12)))
    assert all(item.snapshot.source_digest == first.snapshot.source_digest for item in prepared)
    assert all(item.plan == first.plan for item in prepared)


@pytest.mark.parametrize(
    ("mode", "capabilities", "expected"),
    [
        ("auto", CompactionCapabilities(native_managed=True), "native"),
        (
            "auto",
            CompactionCapabilities(history_export=True, history_replace=True),
            "kedi",
        ),
        ("native", CompactionCapabilities(native_managed=True), "native"),
        (
            "kedi",
            CompactionCapabilities(history_export=True, history_replace=True),
            "kedi",
        ),
        ("disabled", CompactionCapabilities(can_disable=True), "disabled"),
    ],
)
def test_compaction_backend_resolution(
    mode: Any,
    capabilities: CompactionCapabilities,
    expected: str,
) -> None:
    assert resolve_compaction_backend(mode, capabilities) == expected


def test_compaction_backend_preflight_rejects_unsupported_and_double_owned_modes() -> None:
    with pytest.raises(CompactionPreflightError, match="lossless history"):
        resolve_compaction_backend("kedi", CompactionCapabilities(native_managed=True))
    with pytest.raises(CompactionPreflightError, match="native compaction"):
        resolve_compaction_backend("native", CompactionCapabilities(history_export=True))
    with pytest.raises(CompactionPreflightError, match="cannot be disabled"):
        resolve_compaction_backend("disabled", CompactionCapabilities(native_managed=True))
    with pytest.raises(CompactionPreflightError, match="automatic compaction"):
        resolve_compaction_backend("auto", CompactionCapabilities())


def test_history_codec_registry_is_explicit_and_rejects_duplicate_ownership() -> None:
    registry = HistoryCodecRegistry()
    pydantic = PydanticHistoryCodec()
    registry.register(pydantic)
    assert registry.get("pydantic") is pydantic
    with pytest.raises(ValueError, match="already registered"):
        registry.register(PydanticHistoryCodec())
    with pytest.raises(KeyError, match="no history codec"):
        registry.get("codex")


def test_unknown_entries_and_protected_identifiers_cannot_enter_selected_window() -> None:
    history = _pydantic_history()
    history.insert(7, {"kind": "future-provider-message", "payload": "opaque"})
    state: dict[str, object] = {"pydantic_messages": history}
    transaction = KediCompactionTransaction(PydanticHistoryCodec())
    snapshot = transaction.processor.inspect(state, codec=transaction.codec)
    policy = HistoryReductionPolicy(
        threshold_tokens=max(1, snapshot.total_tokens - 1),
        target_tokens=max(1, snapshot.total_tokens // 2),
        preserve_recent_tokens=1,
        checkpoint_budget_tokens=20,
        protected_identifiers=frozenset({"tool_call_result_1"}),
    )
    plan = transaction.processor.plan(snapshot, policy)

    assert any("unknown-entry" in group.protected_reasons for group in plan.groups)
    assert any("protected-reference" in group.protected_reasons for group in plan.groups)
    if plan.window is not None:
        assert all(
            not group.protected for group in plan.groups if group.group_id in plan.window.group_ids
        )


def test_below_threshold_and_empty_history_return_typed_no_ops() -> None:
    transaction = KediCompactionTransaction(PydanticHistoryCodec())
    empty = transaction.prepare(
        {},
        HistoryReductionPolicy(
            threshold_tokens=10,
            target_tokens=5,
            preserve_recent_tokens=1,
            checkpoint_budget_tokens=1,
        ),
    )
    assert empty.plan.applicable is False
    assert empty.plan.no_op_reason == "empty-history"

    state: dict[str, object] = {"pydantic_messages": _pydantic_history()}
    snapshot = transaction.processor.inspect(state, codec=transaction.codec)
    below = transaction.prepare(
        state,
        HistoryReductionPolicy(
            threshold_tokens=snapshot.total_tokens + 1,
            target_tokens=snapshot.total_tokens,
            preserve_recent_tokens=1,
            checkpoint_budget_tokens=1,
        ),
    )
    assert below.plan.applicable is False
    assert below.plan.no_op_reason == "below-threshold"


def test_native_backend_propagates_cancellation_without_fabricating_outcome() -> None:
    started = asyncio.Event()

    async def compact(_state: object, _policy: object) -> CompactionOutcome:
        started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    async def run() -> None:
        backend = NativeCompactionBackend(
            CompactionCapabilities(native_managed=True),
            compact,
        )
        with installed_recording_backend() as telemetry:
            task = asyncio.create_task(backend.compact({}, {}))
            await started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        assert telemetry.metrics[0].attributes["result"] == "cancelled"
        assert "kedi.history.compaction.failures" not in {
            metric.name for metric in telemetry.metrics
        }

    asyncio.run(run())


def test_track_a_telemetry_contains_structure_but_not_history_content() -> None:
    state: dict[str, object] = {"pydantic_messages": _pydantic_history()}
    transaction = KediCompactionTransaction(PydanticHistoryCodec())
    snapshot = transaction.processor.inspect(state, codec=transaction.codec)

    with installed_recording_backend() as telemetry:
        prepared = transaction.prepare(state, _policy(snapshot.total_tokens))
        transaction.commit(
            state,
            prepared=prepared,
            checkpoint=_checkpoint(prepared),
        )

    assert [span.name for span in telemetry.started_spans] == [
        "process history",
        "compact history",
    ]
    serialized_attributes = json.dumps(
        [span.attributes for span in telemetry.started_spans],
        sort_keys=True,
    )
    assert "Delete the production database" not in serialized_attributes
    assert "signed-reasoning-1" not in serialized_attributes
    assert "tool_call_result_1" not in serialized_attributes
    process = telemetry.started_spans[0]
    compact = telemetry.started_spans[1]
    assert process.attributes["kedi.history.compaction.input_messages"] == len(_pydantic_history())
    assert process.attributes["kedi.history.compaction.applicable"] is True
    assert compact.attributes["kedi.history.compaction.mode"] == "kedi"
    assert compact.attributes["kedi.history.compaction.trigger"] == "threshold"
    assert compact.attributes["kedi.history.compaction.prefix_preserved"] is True
    assert compact.attributes["kedi.history.compaction.checkpoint_valid"] is True
    reduction_ratio = cast(
        "float",
        compact.attributes["kedi.history.compaction.reduction_ratio"],
    )
    assert 0 < reduction_ratio < 1
    assert [event.name for event in compact.events] == [
        "compaction triggered",
        "checkpoint validated",
        "checkpoint committed",
    ]
    metric_names = [metric.name for metric in telemetry.metrics]
    assert metric_names == [
        "kedi.history.compactions",
        "kedi.history.compaction.duration",
        "kedi.history.compaction.input_tokens",
        "kedi.history.compaction.output_tokens",
        "kedi.history.compaction.reduction_ratio",
    ]
    assert all(
        set(metric.attributes) <= {"mode", "trigger", "result"} for metric in telemetry.metrics
    )


def test_below_threshold_history_does_not_emit_compaction_telemetry() -> None:
    state: dict[str, object] = {"pydantic_messages": _pydantic_history()}
    transaction = KediCompactionTransaction(PydanticHistoryCodec())
    snapshot = transaction.processor.inspect(state, codec=transaction.codec)

    with installed_recording_backend() as telemetry:
        prepared = transaction.prepare(
            state,
            HistoryReductionPolicy(
                threshold_tokens=snapshot.total_tokens + 1,
                target_tokens=snapshot.total_tokens,
                preserve_recent_tokens=1,
                checkpoint_budget_tokens=1,
            ),
        )

    assert prepared.plan.no_op_reason == "below-threshold"
    assert telemetry.started_spans == []
    assert telemetry.metrics == []


def test_no_safe_window_records_history_processing_reason() -> None:
    state: dict[str, object] = {"pydantic_messages": _pydantic_history()}
    transaction = KediCompactionTransaction(PydanticHistoryCodec())
    snapshot = transaction.processor.inspect(state, codec=transaction.codec)

    with installed_recording_backend() as telemetry:
        prepared = transaction.prepare(
            state,
            HistoryReductionPolicy(
                threshold_tokens=snapshot.total_tokens,
                target_tokens=max(1, snapshot.total_tokens // 2),
                preserve_recent_tokens=0,
                checkpoint_budget_tokens=1,
                minimum_reduction_tokens=snapshot.total_tokens + 1,
            ),
        )

    assert prepared.plan.no_op_reason == "no-safe-window"
    assert len(telemetry.started_spans) == 1
    assert (
        telemetry.started_spans[0].attributes["kedi.history.compaction.no_op_reason"]
        == "no-safe-window"
    )


def _seed_conversation(
    conversation: ConversationState,
    *,
    adapter_shortname: str,
    state_key: str,
    history: list[Any],
) -> None:
    with conversation.adapter_turn(adapter_shortname) as lane:
        lane.result_state[state_key] = history


def test_conversation_commit_rotates_epoch_once_and_invalidates_stale_lanes() -> None:
    conversation = ConversationState(session_id="track-a-commit")
    _seed_conversation(
        conversation,
        adapter_shortname="langchain",
        state_key="langchain_messages",
        history=_langchain_history(),
    )
    history = _pydantic_history()
    _seed_conversation(
        conversation,
        adapter_shortname="pydantic",
        state_key="pydantic_messages",
        history=history,
    )
    transaction = ConversationCompactionTransaction(PydanticHistoryCodec())
    snapshot = transaction.transaction.processor.inspect(
        {"pydantic_messages": history},
        codec=transaction.codec,
    )
    prepared = transaction.prepare(conversation, _policy(snapshot.total_tokens))
    epoch_before = conversation.cache_epoch

    with installed_recording_backend() as telemetry:
        result = transaction.commit(
            conversation,
            prepared=prepared,
            checkpoint=_checkpoint(prepared.prepared),
        )

    assert result.outcome.applied is True
    assert conversation.cache_epoch == epoch_before + 1
    compacted = conversation.adapter_state_snapshot("pydantic")
    assert compacted.cache_epoch == epoch_before + 1
    assert len(cast("list[object]", compacted.state["pydantic_messages"])) < len(history)
    assert conversation.adapter_state_snapshot("langchain").state == {}
    compact_span = telemetry.started_spans[0]
    assert compact_span.attributes["kedi.history.compaction.cache_epoch_changed"] is True
    assert "cache epoch advanced" in {event.name for event in compact_span.events}
    conversation.close()


def test_conversation_prepare_preserves_checkpoint_acceptance_requirements() -> None:
    history = _pydantic_history()
    conversation = ConversationState(session_id="track-a-acceptance-policy")
    _seed_conversation(
        conversation,
        adapter_shortname="pydantic",
        state_key="pydantic_messages",
        history=history,
    )
    transaction = ConversationCompactionTransaction(PydanticHistoryCodec())
    snapshot = transaction.transaction.processor.inspect(
        {"pydantic_messages": history},
        codec=transaction.codec,
    )
    anchor = CheckpointAnchor(
        anchor_id="approval:production",
        kind="approval",
        content="Production deployment is not approved.",
    )
    acceptance = CheckpointAcceptancePolicy(required_anchors=(anchor,))

    prepared = transaction.prepare(
        conversation,
        _policy(snapshot.total_tokens),
        acceptance_policy=acceptance,
    )

    assert prepared.prepared.acceptance_policy is acceptance
    conversation.close()


def test_conversation_stale_lane_or_epoch_cannot_mutate_state() -> None:
    history = _pydantic_history()
    conversation = ConversationState(session_id="track-a-stale")
    _seed_conversation(
        conversation,
        adapter_shortname="pydantic",
        state_key="pydantic_messages",
        history=history,
    )
    transaction = ConversationCompactionTransaction(PydanticHistoryCodec())
    snapshot = transaction.transaction.processor.inspect(
        {"pydantic_messages": history},
        codec=transaction.codec,
    )
    prepared = transaction.prepare(conversation, _policy(snapshot.total_tokens))
    checkpoint = _checkpoint(prepared.prepared)

    changed = [*history, copy.deepcopy(history[-1])]
    _seed_conversation(
        conversation,
        adapter_shortname="pydantic",
        state_key="pydantic_messages",
        history=changed,
    )
    with pytest.raises(CompactionValidationError, match="stale"):
        transaction.commit(conversation, prepared=prepared, checkpoint=checkpoint)
    after_stale = conversation.adapter_state_snapshot("pydantic")
    assert after_stale.cache_epoch == 0
    assert after_stale.state["pydantic_messages"] is changed

    prepared_at_epoch_zero = transaction.prepare(
        conversation,
        _policy(
            transaction.transaction.processor.inspect(
                after_stale.state,
                codec=transaction.codec,
            ).total_tokens
        ),
    )
    conversation.advance_cache_epoch()
    with pytest.raises(CompactionValidationError, match="cache epoch"):
        transaction.commit(
            conversation,
            prepared=prepared_at_epoch_zero,
            checkpoint=_checkpoint(prepared_at_epoch_zero.prepared),
        )
    assert conversation.cache_epoch == 1
    assert conversation.adapter_state_snapshot("pydantic").state == {}
    conversation.close()


def test_concurrent_prepared_plans_allow_only_one_commit() -> None:
    history = _pydantic_history()
    conversation = ConversationState(session_id="track-a-concurrent-commit")
    _seed_conversation(
        conversation,
        adapter_shortname="pydantic",
        state_key="pydantic_messages",
        history=history,
    )
    transaction = ConversationCompactionTransaction(PydanticHistoryCodec())
    snapshot = transaction.transaction.processor.inspect(
        {"pydantic_messages": history},
        codec=transaction.codec,
    )
    policy = _policy(snapshot.total_tokens)
    first = transaction.prepare(conversation, policy)
    stale = transaction.prepare(conversation, policy)

    transaction.commit(
        conversation,
        prepared=first,
        checkpoint=_checkpoint(first.prepared),
    )
    committed = conversation.adapter_state_snapshot("pydantic")
    with pytest.raises(CompactionValidationError, match="cache epoch"):
        transaction.commit(
            conversation,
            prepared=stale,
            checkpoint=_checkpoint(stale.prepared),
        )

    after_stale = conversation.adapter_state_snapshot("pydantic")
    assert after_stale.cache_epoch == committed.cache_epoch == 1
    assert after_stale.state == committed.state
    conversation.close()


class _RaisingCoordinator:
    def __init__(self, error: BaseException) -> None:
        self.error = error

    def apply_checkpoint(self, *_args: Any, **_kwargs: Any) -> CompactionResult:
        raise self.error


@pytest.mark.parametrize(
    "error",
    [RuntimeError("checkpoint failed"), asyncio.CancelledError()],
    ids=["exception", "cancellation"],
)
def test_conversation_failed_or_cancelled_commit_is_atomic(error: BaseException) -> None:
    history = _pydantic_history()
    conversation = ConversationState(session_id=f"track-a-atomic-{type(error).__name__}")
    _seed_conversation(
        conversation,
        adapter_shortname="pydantic",
        state_key="pydantic_messages",
        history=history,
    )
    codec = PydanticHistoryCodec()
    transaction = ConversationCompactionTransaction(
        codec,
        transaction=KediCompactionTransaction(
            codec,
            coordinator=cast(Any, _RaisingCoordinator(error)),
        ),
    )
    snapshot = transaction.transaction.processor.inspect(
        {"pydantic_messages": history},
        codec=codec,
    )
    prepared = transaction.prepare(conversation, _policy(snapshot.total_tokens))
    before = conversation.adapter_state_snapshot("pydantic")

    with pytest.raises(type(error), match=str(error) or None):
        transaction.commit(
            conversation,
            prepared=prepared,
            checkpoint=_checkpoint(prepared.prepared),
        )

    after = conversation.adapter_state_snapshot("pydantic")
    assert after.cache_epoch == before.cache_epoch == 0
    assert after.state == before.state
    assert after.state["pydantic_messages"] is history
    conversation.close()


@pytest.mark.parametrize(
    ("terminal_event", "expected_reason"),
    [
        (None, "live-artifact"),
        ("artifact_released", None),
        ("artifact_expired", None),
    ],
)
def test_conversation_artifact_lifecycle_protects_only_live_references(
    terminal_event: Any,
    expected_reason: str | None,
) -> None:
    history = _pydantic_history()
    conversation = ConversationState(session_id=f"artifact-{terminal_event}")
    conversation.history.append("artifact_created", ref_id="tool_call_result_1")
    if terminal_event is not None:
        conversation.history.append(terminal_event, ref_id="tool_call_result_1")
    _seed_conversation(
        conversation,
        adapter_shortname="pydantic",
        state_key="pydantic_messages",
        history=history,
    )
    transaction = ConversationCompactionTransaction(PydanticHistoryCodec())
    snapshot = transaction.transaction.processor.inspect(
        {"pydantic_messages": history},
        codec=transaction.codec,
    )
    prepared = transaction.prepare(conversation, _policy(snapshot.total_tokens))
    artifact_group = prepared.prepared.plan.groups[1]

    assert "tool_call_result_1" in artifact_group.artifact_refs
    assert (expected_reason in artifact_group.protected_reasons) is (expected_reason is not None)
    conversation.close()


def test_missing_artifact_reference_is_protected_conservatively() -> None:
    history = _pydantic_history()
    conversation = ConversationState(session_id="artifact-missing")
    _seed_conversation(
        conversation,
        adapter_shortname="pydantic",
        state_key="pydantic_messages",
        history=history,
    )
    transaction = ConversationCompactionTransaction(PydanticHistoryCodec())
    snapshot = transaction.transaction.processor.inspect(
        {"pydantic_messages": history},
        codec=transaction.codec,
    )

    prepared = transaction.prepare(conversation, _policy(snapshot.total_tokens))

    assert "missing-artifact" in prepared.prepared.plan.groups[1].protected_reasons
    conversation.close()


def test_non_lifecycle_artifact_events_do_not_change_reference_status() -> None:
    history = _pydantic_history()
    conversation = ConversationState(session_id="artifact-non-lifecycle-events")
    conversation.history.append("tool_call")
    conversation.history.append("artifact_created", ref_id="tool_call_result_1")
    conversation.history.append("artifact_read", ref_id="tool_call_result_1")
    _seed_conversation(
        conversation,
        adapter_shortname="pydantic",
        state_key="pydantic_messages",
        history=history,
    )
    transaction = ConversationCompactionTransaction(PydanticHistoryCodec())
    snapshot = transaction.transaction.processor.inspect(
        {"pydantic_messages": history},
        codec=transaction.codec,
    )

    prepared = transaction.prepare(conversation, _policy(snapshot.total_tokens))

    assert "live-artifact" in prepared.prepared.plan.groups[1].protected_reasons
    conversation.close()


def test_exact_threshold_is_eligible_for_reduction() -> None:
    state: dict[str, object] = {"pydantic_messages": _pydantic_history()}
    transaction = KediCompactionTransaction(PydanticHistoryCodec())
    snapshot = transaction.processor.inspect(state, codec=transaction.codec)
    policy = HistoryReductionPolicy(
        threshold_tokens=snapshot.total_tokens,
        target_tokens=max(1, snapshot.total_tokens // 2),
        preserve_recent_tokens=1,
        checkpoint_budget_tokens=20,
    )

    assert transaction.prepare(state, policy).plan.applicable is True


def test_parallel_out_of_order_tool_results_remain_one_closed_group() -> None:
    messages = [
        ModelRequest(parts=[SystemPromptPart("System")]),
        ModelRequest(parts=[UserPromptPart("Read both files")]),
        ModelResponse(
            parts=[
                ToolCallPart("read_file", {"path": "a"}, "call-a"),
                ToolCallPart("read_file", {"path": "b"}, "call-b"),
            ]
        ),
        ModelRequest(
            parts=[
                ToolReturnPart("read_file", "b", "call-b"),
                ToolReturnPart("read_file", "a", "call-a"),
            ]
        ),
        ModelResponse(parts=[TextPart("Both files were read")]),
        ModelRequest(parts=[UserPromptPart("Current request")]),
    ]
    state: dict[str, object] = {
        "pydantic_messages": ModelMessagesTypeAdapter.dump_python(messages, mode="json")
    }
    transaction = KediCompactionTransaction(PydanticHistoryCodec())
    snapshot = transaction.processor.inspect(state, codec=transaction.codec)

    assert snapshot.groups[1].entry_start == 1
    assert snapshot.groups[1].entry_end == 5
    assert snapshot.groups[1].closed is True
    assert set(snapshot.groups[1].lifecycle_ids) == {"call-a", "call-b"}


def test_randomized_closed_histories_select_only_contiguous_unprotected_groups() -> None:
    for seed in range(30):
        rng = random.Random(seed)
        messages: list[ModelRequest | ModelResponse] = [
            ModelRequest(parts=[SystemPromptPart("Stable system")])
        ]
        for turn in range(rng.randint(3, 9)):
            messages.append(
                ModelRequest(parts=[UserPromptPart(f"Turn {turn}: " + "context " * 80)])
            )
            if rng.choice((True, False)):
                call_id = f"call-{turn}"
                messages.extend(
                    (
                        ModelResponse(parts=[ToolCallPart("lookup", {"turn": turn}, call_id)]),
                        ModelRequest(parts=[ToolReturnPart("lookup", {"value": turn}, call_id)]),
                    )
                )
            messages.append(ModelResponse(parts=[TextPart(f"Completed turn {turn}")]))
        state: dict[str, object] = {
            "pydantic_messages": ModelMessagesTypeAdapter.dump_python(messages, mode="json")
        }
        transaction = KediCompactionTransaction(PydanticHistoryCodec())
        snapshot = transaction.processor.inspect(state, codec=transaction.codec)
        policy = HistoryReductionPolicy(
            threshold_tokens=max(1, snapshot.total_tokens - 1),
            target_tokens=max(1, snapshot.total_tokens // 2),
            preserve_recent_tokens=1,
            checkpoint_budget_tokens=20,
        )

        first = transaction.prepare(state, policy)
        second = transaction.prepare(copy.deepcopy(state), policy)

        assert first.plan == second.plan
        if first.plan.window is None:
            continue
        selected = [
            group for group in first.plan.groups if group.group_id in first.plan.window.group_ids
        ]
        assert selected
        assert all(group.closed and not group.protected for group in selected)
        assert all(
            left.entry_end == right.entry_start for left, right in zip(selected, selected[1:])
        )


def test_policy_range_and_checkpoint_models_reject_invalid_values() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        HistoryRange(start=-1, end=1)
    with pytest.raises(ValueError, match="non-empty"):
        HistoryRange(start=1, end=1)
    with pytest.raises(TypeError, match="integers"):
        cast(Any, HistoryRange)(start=0.5, end=1)
    with pytest.raises(TypeError, match="integers"):
        cast(Any, HistoryRange)(start=False, end=1)
    for name in (
        "threshold_tokens",
        "target_tokens",
        "checkpoint_budget_tokens",
        "minimum_reduction_tokens",
    ):
        values: dict[str, Any] = {
            "threshold_tokens": 10,
            "target_tokens": 5,
            "preserve_recent_tokens": 0,
            "checkpoint_budget_tokens": 1,
            "minimum_reduction_tokens": 1,
        }
        values[name] = 0
        with pytest.raises(ValueError, match=name):
            HistoryReductionPolicy(**values)
    with pytest.raises(ValueError, match="preserve_recent_tokens"):
        HistoryReductionPolicy(
            threshold_tokens=10,
            target_tokens=5,
            preserve_recent_tokens=-1,
            checkpoint_budget_tokens=1,
        )
    with pytest.raises(ValueError, match="target_tokens"):
        HistoryReductionPolicy(
            threshold_tokens=10,
            target_tokens=10,
            preserve_recent_tokens=0,
            checkpoint_budget_tokens=1,
        )

    checkpoint_args: dict[str, Any] = {
        "checkpoint_id": "checkpoint",
        "version": 1,
        "source_start_id": "start",
        "source_end_id": "end",
        "source_digest": "digest",
        "summary": "summary",
        "source_tokens": 10,
        "checkpoint_tokens": 1,
    }
    for name, value, match in (
        ("checkpoint_id", "", "checkpoint_id"),
        ("source_start_id", "", "source_start_id"),
        ("source_end_id", " ", "source_end_id"),
        ("source_digest", "", "source_digest"),
        ("version", 0, "version"),
        ("summary", " ", "summary"),
        ("source_tokens", 0, "token counts"),
        ("checkpoint_tokens", 0, "token counts"),
    ):
        with pytest.raises(ValueError, match=match):
            cast(Any, CompactionCheckpoint)(**{**checkpoint_args, name: value})
    for name, value, match in (
        ("version", True, "version"),
        ("source_tokens", 1.5, "token counts"),
        ("checkpoint_tokens", False, "token counts"),
    ):
        with pytest.raises(TypeError, match=match):
            cast(Any, CompactionCheckpoint)(**{**checkpoint_args, name: value})

    with pytest.raises(ValueError, match="anchor ID"):
        CheckpointAnchor(anchor_id="", kind="fact", content="state")
    with pytest.raises(ValueError, match="anchor content"):
        CheckpointAnchor(anchor_id="state", kind="fact", content=" ")
    anchor = CheckpointAnchor(anchor_id="approval:staging", kind="approval", content="denied")
    with pytest.raises(ValueError, match="anchor IDs"):
        CheckpointAcceptancePolicy(required_anchors=(anchor, anchor))
    with pytest.raises(ValueError, match="anchor IDs"):
        CompactionCheckpoint(
            **checkpoint_args,
            anchors=(anchor, anchor),
        )


@pytest.mark.parametrize(
    ("codec", "state"),
    [
        (
            PydanticHistoryCodec(),
            {
                "pydantic_messages": ModelMessagesTypeAdapter.dump_python(
                    [
                        ModelRequest(
                            parts=[
                                ToolReturnPart("orphan", "result", "orphan-call"),
                            ]
                        )
                    ],
                    mode="json",
                )
            },
        ),
        (
            LangChainHistoryCodec(),
            {
                "langchain_messages": [
                    message_to_dict(ToolMessage(content="result", tool_call_id="orphan-call"))
                ]
            },
        ),
    ],
)
def test_orphan_lifecycle_results_and_userless_cycles_are_never_compactable(
    codec: Any,
    state: dict[str, object],
) -> None:
    snapshot = KediCompactionTransaction(codec).processor.inspect(state, codec=codec)

    assert len(snapshot.groups) == 1
    group = snapshot.groups[0]
    assert group.closed is False
    assert "orphan-lifecycle-result" in group.protected_reasons
    assert "missing-user-cycle" in group.protected_reasons


def test_pydantic_codec_recognizes_native_tools_files_and_tool_availability() -> None:
    messages = [
        ModelRequest(parts=[ToolAvailabilityDeltaPart(added=["web_search"])]),
        ModelRequest(parts=[UserPromptPart("Inspect the attachment.")]),
        ModelResponse(
            parts=[
                NativeToolCallPart("web_search", {"query": "kedi"}, "native-call"),
            ]
        ),
        ModelResponse(
            parts=[
                NativeToolReturnPart("web_search", "result", "native-call"),
            ]
        ),
        ModelResponse(
            parts=[
                FilePart(BinaryContent(b"report", media_type="text/plain")),
                TextPart("Inspection complete."),
            ]
        ),
    ]
    state = {"pydantic_messages": ModelMessagesTypeAdapter.dump_python(messages, mode="json")}

    snapshot = KediCompactionTransaction(PydanticHistoryCodec()).processor.inspect(
        state,
        codec=PydanticHistoryCodec(),
    )

    assert [entry.kind for entry in snapshot.entries] == [
        "system",
        "user",
        "tool-call",
        "tool-result",
        "assistant",
    ]
    assert not any(entry.unknown for entry in snapshot.entries)
    assert "provider-required" in snapshot.groups[0].protected_reasons
    assert snapshot.groups[1].closed is True
    assert snapshot.groups[1].lifecycle_ids == ("native-call",)


def test_pydantic_codec_protects_mixed_system_prompts_and_provider_thinking() -> None:
    messages = [
        ModelRequest(
            parts=[
                SystemPromptPart("Stable instructions"),
                UserPromptPart("Complete the task"),
            ]
        ),
        ModelResponse(
            parts=[
                ThinkingPart(
                    "Provider continuation state",
                    id="reasoning-item-1",
                    provider_details={"opaque": "state"},
                ),
                TextPart("Done"),
            ]
        ),
    ]
    state = {"pydantic_messages": ModelMessagesTypeAdapter.dump_python(messages, mode="json")}

    snapshot = KediCompactionTransaction(PydanticHistoryCodec()).processor.inspect(
        state,
        codec=PydanticHistoryCodec(),
    )

    assert len(snapshot.groups) == 1
    assert snapshot.groups[0].closed is True
    assert snapshot.groups[0].protected_reasons == ("provider-required",)

    metadata_messages = [
        ModelRequest(parts=[UserPromptPart("Continue")]),
        ModelResponse(
            parts=[TextPart("Done")],
            metadata={"reasoning_signature": "opaque-signature"},
        ),
    ]
    metadata_state = {
        "pydantic_messages": ModelMessagesTypeAdapter.dump_python(
            metadata_messages,
            mode="json",
        )
    }
    metadata_snapshot = PydanticHistoryCodec().inspect(metadata_state)
    assert metadata_snapshot.entries[1].provider_required is True


def test_langchain_codec_tracks_server_tools_and_protects_unknown_content() -> None:
    messages = [
        HumanMessage(content="Search the server index."),
        AIMessage(
            content=[
                {
                    "type": "server_tool_call",
                    "id": "server-call",
                    "name": "web_search",
                    "args": {"query": "kedi"},
                }
            ]
        ),
        AIMessage(
            content=[
                {
                    "type": "server_tool_result",
                    "tool_call_id": "server-call",
                    "status": "success",
                    "output": "result",
                }
            ]
        ),
        AIMessage(content="Search complete."),
        HumanMessage(content="Continue with future provider data."),
        AIMessage(content=[{"type": "future-provider-block", "payload": "opaque"}]),
    ]
    state = {"langchain_messages": [message_to_dict(message) for message in messages]}

    snapshot = KediCompactionTransaction(LangChainHistoryCodec()).processor.inspect(
        state,
        codec=LangChainHistoryCodec(),
    )

    assert snapshot.groups[0].closed is True
    assert snapshot.groups[0].lifecycle_ids == ("server-call",)
    assert snapshot.entries[-1].unknown is True
    assert "unknown-entry" in snapshot.groups[-1].protected_reasons


def test_codec_edge_cases_are_classified_conservatively() -> None:
    pydantic = PydanticHistoryCodec()
    assert pydantic.inspect({"pydantic_messages": "invalid"}).entries == ()
    unknown_pydantic = pydantic.inspect({"pydantic_messages": [object()]})
    assert unknown_pydantic.entries[0].unknown is True
    assert unknown_pydantic.groups == ()

    langchain = LangChainHistoryCodec()
    assert langchain.inspect({"langchain_messages": "invalid"}).entries == ()
    unknown_langchain = langchain.inspect({"langchain_messages": [object()]})
    assert unknown_langchain.entries[0].unknown is True
    assert unknown_langchain.groups == ()

    mixed_pydantic = pydantic.inspect(
        {"pydantic_messages": [ModelRequest(parts=[UserPromptPart("native")]), {}]}
    )
    mixed_langchain = langchain.inspect(
        {"langchain_messages": [HumanMessage(content="native"), {}]}
    )
    assert mixed_pydantic.storage_format == "mixed"
    assert mixed_langchain.storage_format == "mixed"
    checkpoint = CompactionCheckpoint(
        checkpoint_id="mixed",
        version=1,
        source_start_id="start",
        source_end_id="end",
        source_digest="digest",
        summary="summary",
        source_tokens=2,
        checkpoint_tokens=1,
    )
    with pytest.raises(TypeError, match="mixes"):
        pydantic.replace_range(
            {},
            snapshot=mixed_pydantic,
            window=HistoryRange(start=0, end=1),
            checkpoint=checkpoint,
        )
    with pytest.raises(TypeError, match="mixes"):
        langchain.replace_range(
            {},
            snapshot=mixed_langchain,
            window=HistoryRange(start=0, end=1),
            checkpoint=checkpoint,
        )
    for codec, snapshot in (
        (pydantic, replace(mixed_pydantic, storage_format="serialized")),
        (langchain, replace(mixed_langchain, storage_format="serialized")),
    ):
        with pytest.raises(TypeError, match="not a message sequence"):
            codec.replace_range(
                {},
                snapshot=snapshot,
                window=HistoryRange(start=0, end=1),
                checkpoint=checkpoint,
            )


def test_codec_kind_helpers_cover_reasoning_chunks_and_unknown_messages() -> None:
    assert (
        _pydantic_entry_kind(
            ModelResponse(parts=[CompactionPart("checkpoint")]),
            checkpoint=True,
            unknown=False,
        )
        == "checkpoint"
    )
    assert (
        _pydantic_entry_kind(
            ModelResponse(parts=[ThinkingPart("reason")]),
            checkpoint=False,
            unknown=False,
        )
        == "reasoning"
    )
    assert (
        _pydantic_entry_kind(
            ModelResponse(parts=[FilePart(BinaryContent(b"file", media_type="text/plain"))]),
            checkpoint=False,
            unknown=False,
        )
        == "assistant"
    )
    assert (
        _pydantic_entry_kind(
            ModelResponse(parts=[]),
            checkpoint=False,
            unknown=False,
        )
        == "unknown"
    )
    assert (
        _pydantic_entry_kind(
            ModelResponse(parts=[]),
            checkpoint=False,
            unknown=True,
        )
        == "unknown"
    )
    assert (
        _pydantic_entry_kind(
            ModelRequest(parts=[]),
            checkpoint=False,
            unknown=False,
        )
        == "unknown"
    )
    assert (
        _pydantic_storage_format(
            (ModelRequest(parts=[UserPromptPart("native")]),),
            native_types=(ModelRequest, ModelResponse),
        )
        == "native"
    )
    assert (
        _pydantic_storage_format(
            (ModelRequest(parts=[]), {}),
            native_types=(ModelRequest, ModelResponse),
        )
        == "mixed"
    )

    reasoning = AIMessage(content=[{"type": "reasoning", "reasoning": "work", "signature": "sig"}])
    assert _langchain_entry_kind(reasoning, checkpoint=False, unknown=False) == "reasoning"
    assert (
        _langchain_entry_kind(
            ChatMessage(content="chat", role="custom"), checkpoint=False, unknown=False
        )
        == "unknown"
    )
    assert _langchain_content_kinds(42) == ["unknown-content"]
    assert _langchain_content_kinds(["plain", 42]) == ["str", "int"]
    assert _langchain_content_is_known(42) is False
    assert _langchain_content_is_known(["plain"]) is True
    assert _langchain_content_is_known([42]) is False
    assert _langchain_content_lifecycle(
        [
            "plain",
            {"type": "tool_call"},
            {"type": "tool_result"},
        ]
    ) == ((), (), True)
    assert _langchain_has_reasoning(reasoning) is True
    assert _langchain_has_signed_reasoning(reasoning) is True
    assert (
        _langchain_entry_kind(AIMessage(content="ignored"), checkpoint=False, unknown=True)
        == "unknown"
    )
    assert (
        _langchain_has_reasoning(
            SimpleNamespace(content=[42, {"type": "text"}], additional_kwargs={})
        )
        is False
    )
    assert (
        _langchain_has_reasoning(
            SimpleNamespace(content=[], additional_kwargs={"reasoning": "work"})
        )
        is True
    )
    assert (
        _langchain_has_signed_reasoning(
            SimpleNamespace(content=[], additional_kwargs={"signature": "sig"})
        )
        is True
    )
    assert _langchain_has_text(AIMessage(content=["plain"])) is True
    assert _langchain_has_text(AIMessage(content="")) is False
    non_text_message = SimpleNamespace(content=42)
    assert _langchain_has_text(non_text_message) is False
    assert (
        _langchain_storage_format((HumanMessage(content="native"),), native_type=HumanMessage)
        == "native"
    )
    assert (
        _langchain_storage_format((HumanMessage(content="native"), {}), native_type=HumanMessage)
        == "mixed"
    )

    inspected = LangChainHistoryCodec().inspect(
        {"langchain_messages": [AIMessageChunk(content="partial")]}
    )
    assert inspected.entries[0].complete is False
    invalid_call = LangChainHistoryCodec().inspect(
        {
            "langchain_messages": [
                AIMessage(
                    content="",
                    invalid_tool_calls=[
                        {
                            "name": "broken",
                            "args": "not-json",
                            "error": "invalid arguments",
                            "id": "broken-call",
                        }
                    ],
                )
            ]
        }
    )
    assert invalid_call.entries[0].unknown is True

    empty_pydantic = PydanticHistoryCodec().inspect(
        {"pydantic_messages": [ModelResponse(parts=[])]}
    )
    assert empty_pydantic.entries[0].kind == "unknown"
    assert empty_pydantic.entries[0].unknown is True

    custom_langchain = LangChainHistoryCodec().inspect(
        {"langchain_messages": [ChatMessage(content="custom", role="custom")]}
    )
    assert custom_langchain.entries[0].kind == "unknown"
    assert custom_langchain.entries[0].unknown is True

    interrupted = AIMessage(
        content="",
        response_metadata={"__interrupt__": {"tool": "apply_patch"}},
    )
    assert _langchain_has_interrupt_metadata(interrupted) is True
    interrupted_snapshot = LangChainHistoryCodec().inspect({"langchain_messages": [interrupted]})
    assert "interrupt" in interrupted_snapshot.entries[0].part_kinds
    assert interrupted_snapshot.entries[0].unknown is True


def test_existing_checkpoint_is_classified_as_a_protected_group() -> None:
    state: dict[str, object] = {"pydantic_messages": _pydantic_history()}
    transaction = KediCompactionTransaction(PydanticHistoryCodec())
    snapshot = transaction.processor.inspect(state, codec=transaction.codec)
    prepared = transaction.prepare(state, _policy(snapshot.total_tokens))
    compacted = transaction.commit(
        state,
        prepared=prepared,
        checkpoint=_checkpoint(prepared),
    )
    assert compacted.processed is not None

    inspected = transaction.processor.inspect(compacted.processed.state, codec=transaction.codec)

    assert any("existing-checkpoint" in group.protected_reasons for group in inspected.groups)


def test_processor_reclassifies_an_unclassified_snapshot() -> None:
    codec = PydanticHistoryCodec()
    snapshot = codec.inspect({"pydantic_messages": _pydantic_history()})
    assert snapshot.groups == ()

    plan = KediCompactionTransaction(codec).processor.plan(
        snapshot,
        _policy(snapshot.total_tokens),
    )

    assert plan.groups


def test_selection_helpers_cover_duplicate_reasons_and_segment_gaps() -> None:
    protected = HistoryUnitGroup(
        group_id="protected",
        unit_ids=(),
        entry_start=0,
        entry_end=1,
        token_estimate=1,
        closed=True,
        protected_reasons=("reason",),
    )
    closed = HistoryUnitGroup(
        group_id="closed",
        unit_ids=(),
        entry_start=1,
        entry_end=2,
        token_estimate=1,
        closed=True,
    )
    assert _with_reason(protected, "reason") is protected
    assert _eligible_segments((closed, protected, closed)) == ((closed,), (closed,))


def _synthetic_selection_snapshot() -> HistorySnapshot:
    tokens = (5, 1, 100)
    entries = tuple(
        NativeHistoryEntryRef(
            index=index,
            entry_id=f"entry-{index}",
            fingerprint=f"fingerprint-{index}",
            kind="assistant",
            token_estimate=token_count,
        )
        for index, token_count in enumerate(tokens)
    )
    units = tuple(
        HistoryUnit(
            unit_id=f"unit-{index}",
            entry_start=index,
            entry_end=index + 1,
            entry_ids=(entry.entry_id,),
            kinds=("assistant",),
            token_estimate=entry.token_estimate,
        )
        for index, entry in enumerate(entries)
    )
    groups = tuple(
        HistoryUnitGroup(
            group_id=f"group-{index}",
            unit_ids=(unit.unit_id,),
            entry_start=index,
            entry_end=index + 1,
            token_estimate=unit.token_estimate,
            closed=True,
            protected_reasons=("separator",) if index == 1 else (),
        )
        for index, unit in enumerate(units)
    )
    return HistorySnapshot(
        adapter_shortname="synthetic",
        state_key="messages",
        source_version=1,
        source_digest="synthetic-digest",
        native_entries=(object(), object(), object()),
        entries=entries,
        units=units,
        groups=groups,
        total_tokens=sum(tokens),
    )


def test_selection_skips_an_uneconomical_segment_before_selecting_the_next() -> None:
    snapshot = _synthetic_selection_snapshot()
    policy = HistoryReductionPolicy(
        threshold_tokens=100,
        target_tokens=20,
        preserve_recent_tokens=0,
        checkpoint_budget_tokens=10,
        minimum_reduction_tokens=20,
    )

    plan = plan_history_reduction(snapshot, policy)

    assert plan.window is not None
    assert plan.window.group_ids == ("group-2",)


def test_selection_protects_a_multi_group_recent_tail_and_cross_group_lifecycle() -> None:
    snapshot = _synthetic_selection_snapshot()
    recent = plan_history_reduction(
        snapshot,
        HistoryReductionPolicy(
            threshold_tokens=100,
            target_tokens=20,
            preserve_recent_tokens=101,
            checkpoint_budget_tokens=10,
            minimum_reduction_tokens=20,
        ),
    )
    assert "recent-tail" in recent.groups[2].protected_reasons
    assert "recent-tail" in recent.groups[1].protected_reasons

    linked_groups = (
        replace(snapshot.groups[0], lifecycle_ids=("shared-call",)),
        snapshot.groups[1],
        replace(snapshot.groups[2], lifecycle_ids=("shared-call",)),
    )
    linked = plan_history_reduction(
        replace(snapshot, groups=linked_groups),
        HistoryReductionPolicy(
            threshold_tokens=100,
            target_tokens=20,
            preserve_recent_tokens=0,
            checkpoint_budget_tokens=10,
            minimum_reduction_tokens=20,
        ),
    )
    assert "cross-group-lifecycle" in linked.groups[0].protected_reasons
    assert "cross-group-lifecycle" in linked.groups[2].protected_reasons


def test_native_backend_validates_capabilities_and_callback_outcome() -> None:
    async def compact(_state: object, _policy: object) -> CompactionOutcome:
        return CompactionOutcome(backend="kedi", applied=True)

    with pytest.raises(ValueError, match="native-managed"):
        NativeCompactionBackend(CompactionCapabilities(), compact)

    async def run() -> None:
        backend = NativeCompactionBackend(
            CompactionCapabilities(native_managed=True),
            compact,
        )
        with pytest.raises(ValueError, match="non-native"):
            await backend.compact({}, {})

    asyncio.run(run())


def test_native_backend_returns_a_valid_native_outcome_unchanged() -> None:
    expected = CompactionOutcome(
        backend="native",
        applied=True,
        trigger="manual",
        source_tokens=1_000,
        result_tokens=250,
        usage=CompactionUsage(
            input_tokens=120,
            output_tokens=30,
            cache_read_tokens=80,
            cache_write_tokens=10,
        ),
    )

    async def compact(_state: object, _policy: object) -> CompactionOutcome:
        return expected

    async def run() -> None:
        backend = NativeCompactionBackend(
            CompactionCapabilities(native_managed=True),
            compact,
        )
        with installed_recording_backend() as telemetry:
            assert await backend.compact({}, {}) is expected

        compact_span = telemetry.started_spans[0]
        assert compact_span.name == "compact history"
        assert compact_span.attributes["kedi.history.compaction.mode"] == "native"
        assert compact_span.attributes["kedi.history.compaction.trigger"] == "manual"
        assert compact_span.attributes["kedi.history.compaction.reduction_ratio"] == 0.75
        assert compact_span.attributes["gen_ai.usage.input_tokens"] == 120
        assert compact_span.attributes["gen_ai.usage.output_tokens"] == 30
        assert compact_span.attributes["gen_ai.usage.cache_read_tokens"] == 80
        assert compact_span.attributes["gen_ai.usage.cache_write_tokens"] == 10
        assert telemetry.metrics[0].attributes == {
            "mode": "native",
            "trigger": "manual",
            "result": "applied",
        }

    asyncio.run(run())


def test_native_backend_records_failure_without_history_content() -> None:
    async def compact(_state: object, _policy: object) -> CompactionOutcome:
        raise RuntimeError("sensitive history payload")

    async def run() -> None:
        backend = NativeCompactionBackend(
            CompactionCapabilities(native_managed=True),
            compact,
        )
        with installed_recording_backend() as telemetry:
            with pytest.raises(RuntimeError, match="sensitive history payload"):
                await backend.compact({}, {})

        compact_span = telemetry.started_spans[0]
        assert compact_span.attributes["error.type"] == "RuntimeError"
        assert "sensitive history payload" not in json.dumps(compact_span.attributes)
        assert [event.name for event in compact_span.events] == [
            "compaction triggered",
            "compaction rejected",
        ]
        assert telemetry.metrics[-1].name == "kedi.history.compaction.failures"
        assert telemetry.metrics[-1].attributes == {
            "mode": "native",
            "trigger": "provider",
        }

    asyncio.run(run())


class _InspectingCodec:
    adapter_shortname = "pydantic"
    state_key = "pydantic_messages"

    def __init__(self, mutate: Any, *, mutate_on_call: int = 1) -> None:
        self.base = PydanticHistoryCodec()
        self.mutate = mutate
        self.mutate_on_call = mutate_on_call
        self.calls = 0

    def inspect(self, state: Any) -> HistorySnapshot:
        self.calls += 1
        snapshot = self.base.inspect(state)
        return self.mutate(snapshot) if self.calls == self.mutate_on_call else snapshot

    def replace_range(
        self,
        state: Any,
        *,
        snapshot: HistorySnapshot,
        window: HistoryRange,
        checkpoint: CompactionCheckpoint,
    ) -> dict[str, object]:
        return self.base.replace_range(
            state,
            snapshot=snapshot,
            window=window,
            checkpoint=checkpoint,
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: replace(value, adapter_shortname="other"), "no longer owns"),
        (lambda value: replace(value, source_version=2), "codec version"),
        (lambda value: replace(value, estimator_version="other"), "estimator version"),
    ],
)
def test_validation_rejects_codec_contract_changes(mutate: Any, message: str) -> None:
    state: dict[str, object] = {"pydantic_messages": _pydantic_history()}
    base = KediCompactionTransaction(PydanticHistoryCodec())
    snapshot = base.processor.inspect(state, codec=base.codec)
    prepared = base.prepare(state, _policy(snapshot.total_tokens))
    codec = _InspectingCodec(mutate)

    with pytest.raises(CompactionValidationError, match=message):
        validate_compaction_transaction(
            state,
            snapshot=prepared.snapshot,
            plan=prepared.plan,
            checkpoint=_checkpoint(prepared),
            codec=codec,
        )


def test_validation_rejects_mixed_history_and_noop_plan() -> None:
    state: dict[str, object] = {"pydantic_messages": _pydantic_history()}
    transaction = KediCompactionTransaction(PydanticHistoryCodec())
    snapshot = transaction.processor.inspect(state, codec=transaction.codec)
    prepared = transaction.prepare(state, _policy(snapshot.total_tokens))

    with pytest.raises(CompactionValidationError, match="mixed"):
        validate_compaction_transaction(
            state,
            snapshot=replace(prepared.snapshot, storage_format="mixed"),
            plan=prepared.plan,
            checkpoint=_checkpoint(prepared),
            codec=transaction.codec,
        )

    noop = transaction.prepare(
        state,
        HistoryReductionPolicy(
            threshold_tokens=snapshot.total_tokens + 1,
            target_tokens=snapshot.total_tokens,
            preserve_recent_tokens=1,
            checkpoint_budget_tokens=1,
        ),
    )
    with pytest.raises(CompactionValidationError, match="no selected window"):
        validate_compaction_transaction(
            state,
            snapshot=noop.snapshot,
            plan=noop.plan,
            checkpoint=replace(
                _checkpoint(prepared),
                source_digest=noop.snapshot.source_digest,
            ),
            codec=transaction.codec,
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"source_digest": "wrong"}, "source snapshot"),
        ({"source_version": 2}, "source snapshot"),
        ({"estimator_version": "wrong"}, "estimator version"),
    ],
)
def test_validation_rejects_plan_provenance_changes(
    changes: dict[str, object],
    message: str,
) -> None:
    state: dict[str, object] = {"pydantic_messages": _pydantic_history()}
    transaction = KediCompactionTransaction(PydanticHistoryCodec())
    snapshot = transaction.processor.inspect(state, codec=transaction.codec)
    prepared = transaction.prepare(state, _policy(snapshot.total_tokens))

    with pytest.raises(CompactionValidationError, match=message):
        validate_compaction_transaction(
            state,
            snapshot=prepared.snapshot,
            plan=replace(prepared.plan, **changes),
            checkpoint=_checkpoint(prepared),
            codec=transaction.codec,
        )


def test_prepare_snapshot_rejects_a_snapshot_owned_by_another_codec() -> None:
    state: dict[str, object] = {"pydantic_messages": _pydantic_history()}
    transaction = KediCompactionTransaction(PydanticHistoryCodec())
    snapshot = transaction.processor.inspect(state, codec=transaction.codec)

    with pytest.raises(ValueError, match="different adapter codec"):
        transaction.prepare_snapshot(
            replace(snapshot, adapter_shortname="langchain"),
            _policy(snapshot.total_tokens),
        )


class _NoTokenCoordinator:
    def apply_checkpoint(self, *_args: Any, **_kwargs: Any) -> CompactionResult:
        return CompactionResult(outcome=CompactionOutcome(backend="kedi", applied=False))


def test_kedi_transaction_allows_an_explicit_noop_outcome_without_token_metrics() -> None:
    transaction = KediCompactionTransaction(
        PydanticHistoryCodec(),
        coordinator=cast(Any, _NoTokenCoordinator()),
    )
    state: dict[str, object] = {"pydantic_messages": _pydantic_history()}
    snapshot = transaction.processor.inspect(state, codec=transaction.codec)
    prepared = transaction.prepare(state, _policy(snapshot.total_tokens))

    result = transaction.commit(
        state,
        prepared=prepared,
        checkpoint=_checkpoint(prepared),
    )

    assert result.outcome.applied is False


class _NoProcessedTransaction:
    def __init__(self, codec: PydanticHistoryCodec) -> None:
        self.codec = codec

    def commit(self, *_args: Any, **_kwargs: Any) -> CompactionResult:
        return CompactionResult(outcome=CompactionOutcome(backend="kedi", applied=True))


def test_conversation_compaction_rejects_owner_mismatch_and_missing_processed_state() -> None:
    codec = PydanticHistoryCodec()
    with pytest.raises(ValueError, match="same adapter"):
        ConversationCompactionTransaction(
            codec,
            transaction=KediCompactionTransaction(LangChainHistoryCodec()),
        )

    history = _pydantic_history()
    conversation = ConversationState(session_id="track-a-invalid-result")
    _seed_conversation(
        conversation,
        adapter_shortname="pydantic",
        state_key="pydantic_messages",
        history=history,
    )
    normal = ConversationCompactionTransaction(codec)
    snapshot = normal.transaction.processor.inspect({"pydantic_messages": history}, codec=codec)
    prepared = normal.prepare(conversation, _policy(snapshot.total_tokens))
    with pytest.raises(CompactionValidationError, match="different adapter"):
        normal.commit(
            conversation,
            prepared=replace(prepared, adapter_shortname="langchain"),
            checkpoint=_checkpoint(prepared.prepared),
        )

    broken = ConversationCompactionTransaction(
        codec,
        transaction=cast(Any, _NoProcessedTransaction(codec)),
    )
    with pytest.raises(CompactionValidationError, match="no replacement state"):
        broken.commit(
            conversation,
            prepared=prepared,
            checkpoint=_checkpoint(prepared.prepared),
        )
    assert conversation.cache_epoch == prepared.cache_epoch
    conversation.close()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source_start_id", "wrong", "source start"),
        ("source_end_id", "wrong", "source end"),
        ("source_digest", "wrong", "source digest"),
        ("source_tokens", 1, "source token count"),
        ("checkpoint_tokens", 10_000, "token budget"),
    ],
)
def test_validation_rejects_checkpoint_contract_changes(
    field: str,
    value: object,
    message: str,
) -> None:
    state: dict[str, object] = {"pydantic_messages": _pydantic_history()}
    transaction = KediCompactionTransaction(PydanticHistoryCodec())
    snapshot = transaction.processor.inspect(state, codec=transaction.codec)
    prepared = transaction.prepare(state, _policy(snapshot.total_tokens))

    with pytest.raises(CompactionValidationError, match=message):
        validate_compaction_transaction(
            state,
            snapshot=prepared.snapshot,
            plan=prepared.plan,
            checkpoint=replace(_checkpoint(prepared), **{field: value}),
            codec=transaction.codec,
        )


def test_checkpoint_acceptance_rejects_untrusted_model_output_without_leaking_secrets() -> None:
    state: dict[str, object] = {"pydantic_messages": _pydantic_history()}
    transaction = KediCompactionTransaction(PydanticHistoryCodec())
    snapshot = transaction.processor.inspect(state, codec=transaction.codec)
    prepared = transaction.prepare(state, _policy(snapshot.total_tokens))
    assert prepared.plan.window is not None
    source_entry = replace(
        prepared.snapshot.entries[prepared.plan.window.native_range.start],
        searchable_text=(
            '{"artifact_ref":"tool_call_result_42",'
            '"API_TOKEN":"sk_compactor_probe_DO_NOT_REPEAT_7491"}'
        ),
        artifact_refs=("tool_call_result_42",),
    )
    checkpoint = replace(
        _checkpoint(prepared),
        artifact_refs=("tool_call_result_42",),
    )
    policy = CheckpointAcceptancePolicy()

    _validate_checkpoint_acceptance(
        source_entries=(source_entry,),
        checkpoint=checkpoint,
        policy=policy,
    )

    with pytest.raises(CompactionValidationError, match="duplicate artifact"):
        _validate_checkpoint_acceptance(
            source_entries=(source_entry,),
            checkpoint=replace(
                checkpoint,
                artifact_refs=("tool_call_result_42", "tool_call_result_42"),
            ),
            policy=policy,
        )
    with pytest.raises(CompactionValidationError, match="outside its source window"):
        _validate_checkpoint_acceptance(
            source_entries=(source_entry,),
            checkpoint=replace(
                checkpoint,
                artifact_refs=("tool_call_result_42 is the live artifact",),
            ),
            policy=policy,
        )
    with pytest.raises(CompactionValidationError, match="checkpoint text"):
        _validate_checkpoint_acceptance(
            source_entries=(source_entry,),
            checkpoint=replace(checkpoint, summary="Read artifact_99 next."),
            policy=policy,
        )

    secret = "sk_compactor_probe_DO_NOT_REPEAT_7491"
    with pytest.raises(CompactionValidationError, match="secret-like") as captured:
        _validate_checkpoint_acceptance(
            source_entries=(source_entry,),
            checkpoint=replace(checkpoint, summary=f"Credential was {secret}"),
            policy=policy,
        )
    assert secret not in str(captured.value)


def test_secret_scanner_only_returns_high_confidence_credential_values() -> None:
    assert _secret_like_values(
        "API_TOKEN=sk_compactor_probe_DO_NOT_REPEAT_7491 "
        "github_pat_1234567890abcdefghijklmnop "
        "AKIA1234567890ABCDEF"
    ) == frozenset(
        {
            "sk_compactor_probe_DO_NOT_REPEAT_7491",
            "github_pat_1234567890abcdefghijklmnop",
            "AKIA1234567890ABCDEF",
        }
    )
    assert _secret_like_values("attempts=3 artifact=tool_call_result_42") == frozenset()


@pytest.mark.parametrize(
    ("codec", "state_key", "history"),
    [
        (PydanticHistoryCodec(), "pydantic_messages", _pydantic_history()),
        (LangChainHistoryCodec(), "langchain_messages", _langchain_history()),
    ],
)
def test_checkpoint_anchors_fail_closed_and_replay_canonical_runtime_state(
    codec: Any,
    state_key: str,
    history: list[object],
) -> None:
    state: dict[str, object] = {state_key: history}
    transaction = KediCompactionTransaction(codec)
    snapshot = transaction.processor.inspect(state, codec=codec)
    anchor = CheckpointAnchor(
        anchor_id="approval:staging",
        kind="approval",
        content="Staging deployment remains unapproved.",
    )
    acceptance = CheckpointAcceptancePolicy(required_anchors=(anchor,))
    prepared = transaction.prepare(
        state,
        _policy(snapshot.total_tokens),
        acceptance_policy=acceptance,
    )
    checkpoint = _checkpoint(prepared)

    with pytest.raises(CompactionValidationError, match="omits required"):
        transaction.commit(state, prepared=prepared, checkpoint=checkpoint)
    with pytest.raises(CompactionValidationError, match="do not match"):
        transaction.commit(
            state,
            prepared=prepared,
            checkpoint=replace(
                checkpoint,
                anchors=(replace(anchor, content="Staging deployment was approved."),),
            ),
        )

    result = transaction.commit(
        state,
        prepared=prepared,
        checkpoint=replace(checkpoint, anchors=(anchor,)),
    )

    assert result.processed is not None
    assert "Staging deployment remains unapproved." in str(result.processed.state[state_key])


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda value, start: replace(
                value,
                entries=(
                    replace(value.entries[0], fingerprint="changed"),
                    *value.entries[1:],
                ),
            ),
            "retained history prefix",
        ),
        (
            lambda value, start: replace(
                value,
                entries=(
                    *value.entries[:-1],
                    replace(value.entries[-1], fingerprint="changed"),
                ),
            ),
            "retained history tail",
        ),
        (
            lambda value, start: replace(
                value,
                native_entries=(copy.deepcopy(value.native_entries[0]), *value.native_entries[1:]),
            ),
            "native prefix",
        ),
        (
            lambda value, start: replace(
                value,
                native_entries=(
                    *value.native_entries[:-1],
                    copy.deepcopy(value.native_entries[-1]),
                ),
            ),
            "native tail",
        ),
        (
            lambda value, start: replace(
                value,
                entries=(
                    *value.entries[:start],
                    replace(value.entries[start], checkpoint=False),
                    *value.entries[start + 1 :],
                ),
            ),
            "checkpoint marker",
        ),
        (
            lambda value, start: replace(
                value,
                entries=(
                    *value.entries[:start],
                    replace(value.entries[start], token_estimate=10_000),
                    *value.entries[start + 1 :],
                ),
            ),
            "serialized checkpoint",
        ),
    ],
)
def test_coordinator_rejects_codec_output_contract_violations(
    mutate: Any,
    message: str,
) -> None:
    state: dict[str, object] = {"pydantic_messages": _pydantic_history()}
    base = KediCompactionTransaction(PydanticHistoryCodec())
    snapshot = base.processor.inspect(state, codec=base.codec)
    prepared = base.prepare(state, _policy(snapshot.total_tokens))
    assert prepared.plan.window is not None
    start = prepared.plan.window.native_range.start
    codec = _InspectingCodec(lambda value: mutate(value, start), mutate_on_call=3)
    transaction = KediCompactionTransaction(codec)

    with pytest.raises(CompactionValidationError, match=message):
        transaction.commit(
            state,
            prepared=prepared,
            checkpoint=_checkpoint(prepared),
        )
