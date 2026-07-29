from __future__ import annotations

import json
import sys
import threading
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import pytest
from pydantic import BaseModel, ConfigDict

from kedi.agent_adapter.conversation import ConversationState
from kedi.artifacts import (
    ArtifactAccessError,
    ArtifactExpiredError,
    ArtifactHandle,
    ArtifactManager,
    ArtifactNotFoundError,
    ArtifactPolicy,
    ArtifactPolicyError,
    ArtifactQuotaExceededError,
    ArtifactSerializationError,
    ArtifactStatus,
    FileArtifactStore,
    deserialize_artifact,
    estimate_object_size,
    normalize_artifact_policy,
    parse_byte_size,
    parse_duration,
    render_artifact_content,
    serialize_artifact,
)
from kedi.artifacts import telemetry as artifact_telemetry
from kedi.artifacts.cleanup import ArtifactCleanupService
from kedi.artifacts.models import ArtifactRecord, ArtifactRef
from kedi.artifacts.store import InMemoryArtifactStore, store_size
from kedi.artifacts.tools import artifact_tool_specs
from kedi.lang.compiler.environment import (
    KediEnv,
    model_lookup_value,
    model_text,
    model_value,
    native_value,
)
from kedi.promise import KediPromise


def policy(**values: object) -> ArtifactPolicy:
    return ArtifactPolicy.patch(
        {
            "enabled": True,
            "threshold": 1,
            "cleanup_interval": "1h",
            **values,
        }
    ).resolved()


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (True, "not bool"),
        (-1, "must not be negative"),
        (-1.0, "whole bytes"),
        (1.5, "whole bytes"),
        (object(), "must be bytes"),
        ("100", "must include"),
        ("1tb", "unsupported"),
        ("0.1b", "whole bytes"),
    ],
)
def test_parse_byte_size_rejects_invalid_values(value: object, message: str) -> None:
    with pytest.raises(ArtifactPolicyError, match=message):
        parse_byte_size(value)


def test_parse_byte_size_accepts_whole_numeric_and_binary_units() -> None:
    assert parse_byte_size(3) == 3
    assert parse_byte_size(3.0) == 3
    assert parse_byte_size("1.5KiB") == 1536


@pytest.mark.parametrize(
    ("value", "allow_none", "message"),
    [
        ("none", False, "cannot be none"),
        (True, False, "not bool"),
        ("1", False, "must be seconds"),
        ("1w", False, "unsupported"),
        (object(), False, "must be seconds"),
        (0, False, "greater than zero"),
    ],
)
def test_parse_duration_rejects_invalid_values(
    value: object,
    allow_none: bool,
    message: str,
) -> None:
    with pytest.raises(ArtifactPolicyError, match=message):
        parse_duration(value, allow_none=allow_none)


def test_parse_duration_accepts_supported_forms() -> None:
    assert parse_duration(None, allow_none=True) is None
    assert parse_duration("null", allow_none=True) is None
    assert parse_duration(timedelta(milliseconds=250)) == 0.25
    assert parse_duration(2) == 2.0
    assert parse_duration("1.5m") == 90.0


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ({"enabled": "yes"}, "boolean"),
        ({"store": "remote"}, "memory"),
        ({"path": 1}, "path"),
        ({"preview_chars": 0}, "positive integer"),
        ({"unknown": 1}, "unknown"),
    ],
)
def test_policy_patch_rejects_invalid_fields(
    values: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ArtifactPolicyError, match=message):
        ArtifactPolicy.patch(values)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"store": "remote"}, "store"),
        ({"threshold_bytes": -1}, "threshold"),
        ({"ttl_seconds": 0}, "ttl"),
        ({"idle_ttl_seconds": 0}, "idle_ttl"),
        ({"preview_chars": 0}, "preview_chars"),
        ({"read_max_chars": 0}, "read_max_chars"),
        ({"max_artifacts": 0}, "max_artifacts"),
        ({"session_quota_bytes": 0}, "session_quota"),
        ({"cleanup_interval_seconds": 0}, "cleanup_interval"),
        ({"_specified_fields": frozenset({"missing"})}, "unknown specified"),
    ],
)
def test_direct_policy_construction_validates_invariants(
    kwargs: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(ArtifactPolicyError, match=message):
        ArtifactPolicy(**kwargs)


def test_policy_normalization_and_cache_key() -> None:
    base = ArtifactPolicy.patch({"enabled": True, "preview_chars": 7})
    assert normalize_artifact_policy(None, base=base) is base
    assert normalize_artifact_policy(base) is base
    assert normalize_artifact_policy(False, base=base).enabled is False  # type: ignore[union-attr]
    assert len(base.resolved().as_cache_key()) == 11
    with pytest.raises(ArtifactPolicyError, match="mapping"):
        normalize_artifact_policy("enabled")  # type: ignore[arg-type]


def test_serialization_rendering_and_invalid_codecs() -> None:
    assert render_artifact_content("text", "hello") == "hello"
    assert render_artifact_content("bytes", b"\x00") == "AA=="
    assert render_artifact_content("opaque", object()).startswith("<opaque ")
    assert deserialize_artifact("json", b'{"ok":true}') == {"ok": True}
    with pytest.raises(UnicodeDecodeError):
        deserialize_artifact("text", b"\xff")
    with pytest.raises(ArtifactSerializationError, match="unsupported"):
        deserialize_artifact("yaml", b"value")


def test_render_json_rejects_value_that_lost_serializable_form() -> None:
    with pytest.raises(ArtifactSerializationError, match="lost"):
        render_artifact_content("json", {1: "not a JSON object key"})


def test_serialization_rejects_broken_models_and_dataclasses() -> None:
    class BrokenModel(BaseModel):
        model_config = ConfigDict(arbitrary_types_allowed=True)
        value: Any

        def model_dump(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
            raise ValueError("broken")

    @dataclass
    class Recursive:
        child: Any = None

    recursive = Recursive()
    recursive.child = recursive

    assert serialize_artifact(BrokenModel(value=1), preview_chars=20).codec == "opaque"
    assert serialize_artifact(recursive, preview_chars=20).codec == "opaque"
    assert serialize_artifact({"bad": object()}, preview_chars=20).codec == "opaque"
    assert serialize_artifact([object()], preview_chars=20).codec == "opaque"


def test_serialization_truncation_and_size_estimation_cover_structured_values() -> None:
    @dataclass
    class Item:
        value: int

    class Model(BaseModel):
        value: int

    assert serialize_artifact("short", preview_chars=20).preview == "short"
    assert serialize_artifact("long", preview_chars=1).preview == "l"
    assert estimate_object_size({"model": Model(value=1), "item": Item(2)}) > 0
    assert estimate_object_size([1], max_nodes=0) == sys.maxsize


def _record(*, ref_id: str = "art_test", session_id: str = "session") -> ArtifactRecord:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return ArtifactRecord(
        ref=ArtifactRef[Any](
            ref_id=ref_id,
            logical_type="builtins.str",
            media_type="text/plain; charset=utf-8",
            summary="test",
            preview="value",
            size_bytes=5,
            char_count=5,
            created_at=now,
            expires_at=now + timedelta(hours=1),
        ),
        session_id=session_id,
        codec="text",
        status=ArtifactStatus.ACTIVE,
        created_monotonic=0,
        last_accessed_monotonic=0,
        expires_monotonic=3600,
        idle_ttl_seconds=None,
    )


def test_memory_store_ownership_duplicates_and_lifecycle() -> None:
    store = InMemoryArtifactStore()
    record = _record()
    serialized = serialize_artifact("value", preview_chars=10)
    store.put(record, serialized)

    assert store.get_record("session", "art_test") is record
    assert store.get_value("session", "art_test") == "value"
    assert store.records("other") == ()
    with pytest.raises(ArtifactSerializationError, match="already exists"):
        store.put(record, serialized)
    with pytest.raises(ArtifactAccessError, match="active session"):
        store.get_value("other", "art_test")
    with pytest.raises(ArtifactNotFoundError, match="not found"):
        store.get_value("session", "missing")
    store.delete_payload("session", "art_test")
    assert store.get_value("session", "art_test") is None
    store.close()
    assert store.records("session") == ()


def test_file_store_rejects_opaque_and_unsafe_components(tmp_path: Path) -> None:
    store = FileArtifactStore(tmp_path / "artifacts")
    with pytest.raises(ArtifactSerializationError, match="cannot be stored"):
        store.put(_record(), serialize_artifact(object(), preview_chars=20))
    with pytest.raises(ArtifactAccessError, match="invalid"):
        store.restore_session("../escape", monotonic_now=0, wall_now=datetime.now(timezone.utc))


def test_file_store_rejects_duplicate_records_and_symlink_sessions(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    store = FileArtifactStore(root)
    record = _record()
    serialized = serialize_artifact("value", preview_chars=10)
    store.put(record, serialized)
    with pytest.raises(ArtifactSerializationError, match="already exists"):
        store.put(record, serialized)
    store.close()

    symlink_root = tmp_path / "symlink-artifacts"
    symlink_root.mkdir()
    target = tmp_path / "outside"
    target.mkdir()
    (symlink_root / "session").symlink_to(target, target_is_directory=True)
    linked_store = FileArtifactStore(symlink_root)
    with pytest.raises(ArtifactAccessError, match="symlink"):
        linked_store.restore_session(
            "session",
            monotonic_now=0,
            wall_now=datetime.now(timezone.utc),
        )


def test_file_store_ignores_invalid_metadata_and_reads_after_cache_clear(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    store = FileArtifactStore(root)
    record = _record()
    store.put(record, serialize_artifact("value", preview_chars=10))
    assert store.get_value("session", "art_test") == "value"
    store._native_cache.clear()
    assert store.get_value("session", "art_test") == "value"
    store.close()

    session_dir = root / "session"
    (session_dir / "art_broken.json").write_text("{", encoding="utf-8")
    restored = FileArtifactStore(root)
    records = restored.restore_session(
        "session",
        monotonic_now=10,
        wall_now=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    assert [item.ref_id for item in records] == ["art_test"]
    restored.delete_payload("session", "art_test")
    restored.close()


def test_file_store_skips_foreign_metadata_and_missing_payload(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    session = root / "session"
    session.mkdir(parents=True)
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    ref = ArtifactRef[Any](
        ref_id="art_missing",
        logical_type="builtins.str",
        media_type="text/plain",
        summary="missing",
        preview="missing",
        size_bytes=7,
        char_count=7,
        created_at=now,
        expires_at=now + timedelta(hours=1),
    )
    base = {
        "ref": ref.model_dump(mode="json"),
        "codec": "text",
        "status": "active",
        "source": None,
        "digest": None,
        "idle_ttl_seconds": None,
    }
    (session / "art_foreign.json").write_text(
        json.dumps({**base, "session_id": "foreign"}),
        encoding="utf-8",
    )
    (session / "art_missing.json").write_text(
        json.dumps({**base, "session_id": "session"}),
        encoding="utf-8",
    )

    store = FileArtifactStore(root)
    assert store.restore_session("session", monotonic_now=0, wall_now=now) == ()
    store.close()


def test_file_store_removes_payload_when_metadata_write_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kedi.artifacts.store as store_module

    root = tmp_path / "artifacts"
    store = FileArtifactStore(root)
    original = store_module._atomic_write
    calls = 0

    def fail_metadata(path: Path, payload: bytes, *, mode: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("metadata write failed")
        original(path, payload, mode=mode)

    monkeypatch.setattr(store_module, "_atomic_write", fail_metadata)
    with pytest.raises(OSError, match="metadata"):
        store.put(_record(), serialize_artifact("value", preview_chars=10))
    assert not (root / "session" / "art_test.payload").exists()
    store.close()


def test_atomic_write_removes_temporary_file_on_replace_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kedi.artifacts.store as store_module

    target = tmp_path / "target"

    def fail_replace(self: Path, target: Path) -> Path:
        raise OSError(f"cannot replace {target}")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="cannot replace"):
        store_module._atomic_write(target, b"value", mode=0o600)
    assert list(tmp_path.iterdir()) == []


def test_store_size_counts_only_active_and_pending_records() -> None:
    active = _record(ref_id="art_active")
    pending = _record(ref_id="art_pending")
    pending.status = ArtifactStatus.PENDING_RELEASE
    released = _record(ref_id="art_released")
    released.status = ArtifactStatus.RELEASED
    assert store_size((active, pending, released)) == (2, 10)


def test_manager_disabled_search_read_quota_and_stats() -> None:
    manager = ArtifactManager(session_id="edges")
    try:
        assert manager.artifactize("raw", policy=ArtifactPolicy()) == "raw"
        with pytest.raises(ValueError, match="offset"):
            manager.read("missing", offset=-1)
        with pytest.raises(ValueError, match="limit"):
            manager.search(limit=0)

        first = manager.artifactize("first", policy=policy(max_artifacts=1), source="alpha")
        assert isinstance(first, ArtifactHandle)
        with pytest.raises(ArtifactQuotaExceededError, match="count"):
            manager.artifactize("second", policy=policy(max_artifacts=1))
        assert manager.search("missing") == []
        assert manager.stats()["active_artifacts"] == 1

        with pytest.raises(ValueError, match="max_chars"):
            manager.read(first.ref.ref_id, max_chars=0)
        empty = manager.read(first.ref.ref_id, offset=99)
        assert empty.content == ""
        assert empty.complete is True
        manager.release(first.ref.ref_id, compact_history=False)
        assert manager.search() == []
        assert manager.stats()["released_artifacts"] == 1
    finally:
        manager.close()
        manager.close()
    assert manager.cleanup_expired() == 0


def test_manager_json_pointer_escapes_lists_and_scalar_failure() -> None:
    manager = ArtifactManager(session_id="pointer")
    try:
        artifact = manager.artifactize(
            {"a/b": {"~key": ["zero", "one"]}},
            policy=policy(),
        )
        assert isinstance(artifact, ArtifactHandle)
        assert manager.read(artifact.ref.ref_id, path="").content.startswith("{")
        assert manager.read(artifact.ref.ref_id, path="/a~1b/~0key/1").content == '"one"'
        with pytest.raises(ValueError, match="cannot descend"):
            manager.read(artifact.ref.ref_id, path="/a~1b/~0key/1/value")
    finally:
        manager.close()


def test_manager_bytes_quota_and_expired_release() -> None:
    now = 0.0
    wall = datetime(2026, 1, 1, tzinfo=timezone.utc)
    manager = ArtifactManager(
        session_id="quota",
        monotonic=lambda: now,
        wall_clock=lambda: wall + timedelta(seconds=now),
    )
    try:
        with pytest.raises(ArtifactQuotaExceededError, match="session quota"):
            manager.artifactize("12345", policy=policy(session_quota=4))
        artifact = manager.artifactize("value", policy=policy(ttl="1s"))
        assert isinstance(artifact, ArtifactHandle)
        now = 2.0
        manager.cleanup_expired()
        with pytest.raises(ArtifactExpiredError):
            manager.release(artifact.ref.ref_id)
        assert manager.stats()["expired_artifacts"] == 1
    finally:
        manager.close()


def test_manager_opaque_values_and_file_store_reuse(tmp_path: Path) -> None:
    manager = ArtifactManager(session_id="opaque")
    try:
        opaque = manager.artifactize(object(), policy=policy())
        assert isinstance(opaque, ArtifactHandle)
        assert manager.release(opaque.ref.ref_id).status is ArtifactStatus.RELEASED

        file_policy = policy(store="file", path=tmp_path)
        first = manager.artifactize("first", policy=file_policy)
        second = manager.artifactize("second", policy=file_policy)
        assert isinstance(first, ArtifactHandle)
        assert isinstance(second, ArtifactHandle)
        assert manager.get_value(first.ref.ref_id) == "first"
        assert manager.get_value(second.ref.ref_id) == "second"
        with pytest.raises(ArtifactNotFoundError):
            manager.get_value("art_missing")
    finally:
        manager.close()


def test_manager_restores_expired_file_records_without_deduplication(
    tmp_path: Path,
) -> None:
    now = 0.0
    wall_base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    file_policy = policy(store="file", path=tmp_path, ttl="1s")
    first = ArtifactManager(
        session_id="expired-restore",
        monotonic=lambda: now,
        wall_clock=lambda: wall_base + timedelta(seconds=now),
    )
    artifact = first.artifactize("value", policy=file_policy)
    assert isinstance(artifact, ArtifactHandle)
    first.close()

    now = 2.0
    second = ArtifactManager(
        session_id="expired-restore",
        monotonic=lambda: now,
        wall_clock=lambda: wall_base + timedelta(seconds=now),
    )
    try:
        second._store_for_policy(file_policy)
        assert second.stats()["expired_artifacts"] == 1
        assert second._digest_refs == {}
        assert not tuple(tmp_path.rglob("art_*.payload"))
        assert not tuple(tmp_path.rglob("art_*.json"))
    finally:
        second.close()


def test_manager_search_limit_and_expiry_with_active_reader() -> None:
    now = 0.0
    manager = ArtifactManager(
        session_id="search",
        monotonic=lambda: now,
        wall_clock=lambda: datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=now),
    )
    try:
        artifacts = [
            manager.artifactize(
                f"value-{index}",
                policy=policy(ttl="1s"),
                source=f"source-{index}",
            )
            for index in range(3)
        ]
        assert all(isinstance(item, ArtifactHandle) for item in artifacts)
        assert len(manager.search(limit=2)) == 2
        first = artifacts[0]
        assert isinstance(first, ArtifactHandle)
        with manager._lease(first.ref.ref_id):
            now = 2.0
            assert manager.cleanup_expired() == 3
            assert first.ref.ref_id in manager._ref_stores
            assert (
                manager._memory_store.get_value(
                    manager.session_id,
                    first.ref.ref_id,
                )
                == "value-0"
            )
        assert (
            manager._memory_store.get_value(
                manager.session_id,
                first.ref.ref_id,
            )
            is None
        )
        assert manager.stats()["expired_artifacts"] == 3
    finally:
        manager.close()


class CleanupTarget:
    cleanup_interval_seconds = 0.01

    def __init__(self, *, fail: bool = False) -> None:
        self.called = threading.Event()
        self.fail = fail

    def cleanup_expired(self) -> int:
        self.called.set()
        if self.fail:
            raise RuntimeError("expected cleanup failure")
        return 1


def test_cleanup_service_runs_targets_survives_failures_and_closes() -> None:
    service = ArtifactCleanupService()
    failing = CleanupTarget(fail=True)
    healthy = CleanupTarget()
    service.register(failing)
    service.register(healthy)
    assert healthy.called.wait(timeout=1)
    assert failing.called.is_set()
    service.unregister(healthy)
    service.close()
    service.register(healthy)
    service._run()
    service.close()


def test_artifact_spans_never_attach_structured_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    @contextmanager
    def start_as_current_span(
        name: str,
        *,
        attributes: dict[str, object],
    ) -> Iterator[SimpleNamespace]:
        calls.append((name, attributes))
        yield SimpleNamespace()

    tracer = SimpleNamespace(start_as_current_span=start_as_current_span)
    monkeypatch.setattr(artifact_telemetry, "_tracer", tracer)

    with artifact_telemetry.artifact_span(
        "read",
        ref_id="art_safe",
        size_bytes=12,
        payload={"secret": "must not be traced"},
        preview=["must not be traced"],
    ):
        pass

    assert calls == [
        (
            "artifact read",
            {
                "ref_id": "art_safe",
                "size_bytes": 12,
            },
        )
    ]


class BlockingCleanupTarget:
    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    @property
    def cleanup_interval_seconds(self) -> float:
        self.entered.set()
        assert self.release.wait(timeout=1)
        return 1.0

    def cleanup_expired(self) -> int:
        raise AssertionError("closing service must not invoke cleanup")


def test_cleanup_service_can_close_between_snapshot_and_wait() -> None:
    service = ArtifactCleanupService()
    target = BlockingCleanupTarget()
    service.register(target)
    assert target.entered.wait(timeout=1)

    closed = threading.Event()

    def close_service() -> None:
        service.close()
        closed.set()

    closer = threading.Thread(target=close_service)
    closer.start()
    target.release.set()
    closer.join(timeout=1)
    assert closed.is_set()


class Holder(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    value: Any


@dataclass
class Box:
    value: Any


def test_nested_environment_values_materialize_for_native_and_model_paths() -> None:
    manager = ArtifactManager(session_id="environment")
    try:
        handle = manager.artifactize("payload", policy=policy())
        assert isinstance(handle, ArtifactHandle)
        future: Future[dict[str, Any]] = Future()
        future.set_result({"value": handle})
        promise = KediPromise(future, "value")
        nested = {
            "list": [promise],
            "tuple": (handle,),
            "set": {handle},
            "frozen": frozenset({handle}),
            "model": Holder(value=handle),
            "dataclass": Box(value=handle),
        }

        native = native_value(nested)
        model = model_value(nested)

        assert native["list"] == ["payload"]
        assert native["tuple"] == ("payload",)
        assert native["set"] == {"payload"}
        assert native["frozen"] == frozenset({"payload"})
        assert native["model"].value == "payload"
        assert native["dataclass"].value == "payload"
        assert model["list"][0]["ref_id"] == handle.ref.ref_id
        assert model["set"][0]["ref_id"] == handle.ref.ref_id
        assert model["frozen"][0]["ref_id"] == handle.ref.ref_id
        assert handle.ref.ref_id in model_text(nested)
        assert handle.ref.ref_id in model_text(Holder(value=handle))
        assert handle.ref.ref_id in model_text(Box(value=handle))
        assert model_text("plain") == "plain"

        cyclic: list[Any] = []
        cyclic.append(cyclic)
        cyclic.append(handle)
        resolved_cycle = native_value(cyclic)
        assert resolved_cycle[0] is resolved_cycle
        assert resolved_cycle[1] == "payload"
        assert handle.ref.ref_id in model_text(cyclic)

        arbitrary = object()
        assert native_value([arbitrary, handle])[0] is arbitrary

        env = KediEnv({"artifact": handle})
        assert isinstance(env.raw_value("artifact"), ArtifactHandle)
        assert env.native_value("artifact") == "payload"
        assert env["artifact"] == "payload"
        assert model_lookup_value(SimpleNamespace(), "artifact", env)["ref_id"] == handle.ref.ref_id
    finally:
        manager.close()


def test_conversation_replay_serialization_compaction_and_close() -> None:
    state = ConversationState(session_id="conversation_edges")
    assert state.replay_prompt("first") == "first"

    class BrokenJSON:
        pass

    state.record_exchange("prompt", {"value": BrokenJSON()})
    first_snapshot = state.snapshot()
    assert "BrokenJSON" in first_snapshot[1].content
    assert "Current request:\nnext" in state.replay_prompt("next")
    assert state.as_jsonable()["session_id"] == "conversation_edges"

    artifact = state.artifacts.artifactize("payload", policy=policy())
    assert isinstance(artifact, ArtifactHandle)
    assert state.as_jsonable()["artifact_history"][0]["created_at"].endswith("+00:00")
    ref_id = artifact.ref.ref_id
    state.record_exchange(
        f"read {ref_id}",
        {"nested": [{"ref_id": ref_id, "preview": "payload"}], "raw": ref_id},
    )
    with state.adapter_turn("mock") as turn:
        turn.result_state = {"thread_id": "one"}
    state.artifacts.release(ref_id)
    with state.adapter_turn("mock") as replay:
        assert replay.replay_required is True
        assert replay.resume_state == {}
        replay.result_state = {"thread_id": "two"}
    compacted = state.snapshot()
    assert all(ref_id not in turn.ref_ids for turn in compacted)
    assert "unavailable artifact" in compacted[-1].content

    state.close()
    state.close()
    with pytest.raises(RuntimeError, match="closed"):
        state.record_exchange("late", "result")


def test_conversation_failed_turn_does_not_store_state() -> None:
    state = ConversationState(session_id="conversation_failure")
    try:
        with pytest.raises(LookupError):
            with state.adapter_turn("mock") as turn:
                turn.result_state = {"thread_id": "discard"}
                raise LookupError("failed")
        with state.adapter_turn("mock") as retry:
            assert retry.resume_state == {}
    finally:
        state.close()


def test_conversation_serializes_cyclic_results_with_string_fallback() -> None:
    state = ConversationState(session_id="conversation_cycle")
    try:
        cyclic: list[Any] = []
        cyclic.append(cyclic)
        state.record_exchange("cycle", cyclic)
        assert state.snapshot()[-1].content == "[[...]]"
    finally:
        state.close()


def test_artifact_management_tool_functions_delegate_to_manager() -> None:
    manager = ArtifactManager(session_id="tools")
    try:
        artifact = manager.artifactize("payload", policy=policy())
        assert isinstance(artifact, ArtifactHandle)
        read, search, invalidate = artifact_tool_specs(manager)

        assert read.fn(ref_id=artifact.ref.ref_id)["content"] == "payload"
        assert search.fn(query="builtins.str")[0]["ref_id"] == artifact.ref.ref_id
        assert invalidate.fn(ref_id=artifact.ref.ref_id)["status"] == "released"
    finally:
        manager.close()
