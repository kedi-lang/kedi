from __future__ import annotations

import gc
import weakref
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from kedi.artifacts import ArtifactHistory, ArtifactManager, ArtifactPolicy
from kedi.agent_adapter.conversation import ConversationState


@pytest.mark.parametrize("capacity", [1, 17, 4096])
def test_recent_history_has_bounded_storage_and_monotonic_ids(capacity: int) -> None:
    history = ArtifactHistory(max_events=capacity)
    for index in range(capacity + 50):
        assert history.append("tool_call").sequence == index + 1
    assert history.retained_events == capacity
    assert history.dropped_events == 50
    assert [event.sequence for event in history.snapshot()] == list(range(51, capacity + 51))
    assert history.as_jsonable()[0]["sequence"] == 51


@pytest.mark.parametrize("capacity", [0, -1, True, 1.5, "10"])
def test_history_rejects_invalid_retention(capacity) -> None:
    with pytest.raises(ValueError, match="max_events"):
        ArtifactHistory(max_events=capacity)


def test_history_can_explicitly_opt_out_of_retention() -> None:
    history = ArtifactHistory(max_events=None)
    for _ in range(5000):
        history.append("tool_result")
    assert history.retained_events == 5000
    assert history.dropped_events == 0


def test_evicted_history_does_not_keep_event_payload_alive() -> None:
    class Payload:
        pass

    history = ArtifactHistory(max_events=1)
    payload = Payload()
    reference = weakref.ref(payload)
    history.append("tool_result", data={"payload": payload})
    del payload
    assert reference() is not None
    history.append("tool_call")
    gc.collect()
    assert reference() is None


def test_concurrent_appends_and_checkpoints_preserve_sequence_order() -> None:
    history = ArtifactHistory(max_events=32)

    def append_batch(_: int) -> None:
        for _ in range(500):
            history.append("tool_call")
            saved = history.export_checkpoint()
            restored = ArtifactHistory.from_checkpoint(saved)
            assert restored.export_checkpoint() == saved

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(append_batch, range(8)))
    assert history.retained_events == 32
    assert history.dropped_events == 3968
    assert history.snapshot()[-1].sequence == 4000


@pytest.mark.parametrize("capacity", [1, 7, None])
def test_history_checkpoint_continues_original_sequence(capacity: int | None) -> None:
    original = ArtifactHistory(max_events=capacity)
    for _ in range(20):
        original.append("tool_result")
    restored = ArtifactHistory.from_checkpoint(original.export_checkpoint())
    assert restored.max_events == capacity
    assert restored.dropped_events == original.dropped_events
    assert restored.snapshot() == original.snapshot()
    assert restored.append("tool_call").sequence == 21


def test_old_history_checkpoint_is_accepted_and_bounded() -> None:
    original = ArtifactHistory(max_events=None)
    for _ in range(5000):
        original.append("tool_result")
    restored = ArtifactHistory.from_checkpoint(original.as_jsonable())
    assert restored.retained_events == ArtifactHistory.DEFAULT_MAX_EVENTS
    assert restored.dropped_events == 5000 - ArtifactHistory.DEFAULT_MAX_EVENTS
    assert restored.append("tool_call").sequence == 5001


@pytest.mark.parametrize(
    "change",
    [
        {"max_events": 0},
        {"first_sequence": 0},
        {"first_sequence": True},
        {"first_sequence": 9},
        {"events": {}},
        {"unexpected": True},
    ],
)
def test_invalid_history_checkpoint_is_rejected(change: dict) -> None:
    history = ArtifactHistory(max_events=2)
    history.append("tool_call")
    with pytest.raises(ValueError):
        ArtifactHistory.from_checkpoint({**history.export_checkpoint(), **change})


@pytest.mark.parametrize("value", [None, "invalid", 7])
def test_history_checkpoint_rejects_non_container(value) -> None:
    with pytest.raises(ValueError, match="checkpoint is invalid"):
        ArtifactHistory.from_checkpoint(value)


def test_artifact_checkpoint_retains_live_refs_when_history_rotates(tmp_path: Path) -> None:
    manager = ArtifactManager(history=ArtifactHistory(max_events=2), durable_root=tmp_path)
    handle = manager.artifactize("preserved payload", policy=ArtifactPolicy(threshold_bytes=1))
    for _ in range(10):
        manager.history.append("tool_call")
    checkpoint = manager.export_checkpoint()
    manager.close()
    restored = ArtifactManager.from_checkpoint(checkpoint, durable_root=tmp_path)
    try:
        assert restored.history.max_events == 2
        assert restored.history.dropped_events == 9
        assert restored.lifecycle_snapshot() == (frozenset({handle.ref.ref_id}),) * 2
        assert restored.read(handle.ref.ref_id).content == "preserved payload"
        restored.release(handle.ref.ref_id)
        assert restored.lifecycle_snapshot() == (frozenset(), frozenset({handle.ref.ref_id}))
        assert restored.history.snapshot()[-1].sequence == 13
    finally:
        restored.close()


def test_lifecycle_snapshot_detects_expiry_without_refreshing_idle_deadline() -> None:
    now = 0.0
    manager = ArtifactManager(monotonic=lambda: now, history=ArtifactHistory(max_events=1))
    try:
        handle = manager.artifactize(
            "payload", policy=ArtifactPolicy(threshold_bytes=1, idle_ttl_seconds=5)
        )
        manager.history.append("tool_call")
        now = 4.0
        assert handle.ref.ref_id in manager.lifecycle_snapshot()[0]
        now = 6.0
        live, known = manager.lifecycle_snapshot()
        assert live == frozenset()
        assert known == frozenset({handle.ref.ref_id})
    finally:
        manager.close()


@pytest.mark.parametrize("adapter", ["pydantic", "langchain"])
def test_diagnostic_rotation_does_not_rewrite_provider_history_or_cache_epoch(adapter: str) -> None:
    if adapter == "pydantic":
        from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

        messages = [
            ModelRequest(parts=[UserPromptPart("question")]),
            ModelResponse(parts=[TextPart("answer")]),
        ]
        state_key = "pydantic_messages"
    else:
        from langchain_core.messages import AIMessage, HumanMessage

        messages = [HumanMessage(content="question"), AIMessage(content="answer")]
        state_key = "langchain_messages"
    manager = ArtifactManager(history=ArtifactHistory(max_events=2))
    conversation = ConversationState(artifacts=manager)
    try:
        with conversation.adapter_turn(adapter) as lane:
            lane.result_state[state_key] = messages
        before = conversation.adapter_state_snapshot(adapter)
        handle = manager.artifactize("payload", policy=ArtifactPolicy(threshold_bytes=1))
        for _ in range(20):
            manager.history.append("tool_call")
        manager.release(handle.ref.ref_id)
        after = conversation.adapter_state_snapshot(adapter)
        assert manager.history.dropped_events == 20
        assert after.cache_epoch == before.cache_epoch
        assert after.state == before.state
        assert after.state[state_key] is messages
    finally:
        conversation.close()
        manager.close()


def test_legacy_artifact_checkpoint_without_retention_metadata_restores(tmp_path: Path) -> None:
    manager = ArtifactManager(durable_root=tmp_path)
    handle = manager.artifactize("legacy value", policy=ArtifactPolicy(threshold_bytes=1))
    checkpoint = manager.export_checkpoint()
    checkpoint["history"] = checkpoint["history"]["events"]
    manager.close()
    restored = ArtifactManager.from_checkpoint(checkpoint, durable_root=tmp_path)
    try:
        assert restored.get_value(handle.ref.ref_id) == "legacy value"
        assert restored.history.snapshot()[0].sequence == 1
    finally:
        restored.close()
