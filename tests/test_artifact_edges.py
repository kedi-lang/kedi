from __future__ import annotations

import asyncio
import json
import sys
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, AsyncIterator, Iterator, cast

import pytest
from pydantic import BaseModel, ConfigDict

import kedi.artifacts.file_store as artifacts_file_store_module
import kedi.artifacts.projection as artifacts_projection_module
from kedi.agent_adapter.conversation import ConversationState
from kedi.artifacts import (
    ArtifactAccessError,
    ArtifactExpiredError,
    ArtifactHandle,
    ArtifactInputError,
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
from kedi.artifacts.manager import _is_expired
from kedi.artifacts.memory_store import InMemoryArtifactStore
from kedi.artifacts.models import ArtifactRecord, ArtifactRef
from kedi.artifacts.projection import (
    _build_utf8_checkpoints,
    _count_utf8_chars,
    _JsonPointerLocator,
    _load_projection_index,
    _locate_json_pointer_range,
    _read_base64_chars,
    _read_utf8_chars,
    _validated_checkpoints,
    _validated_pointer_ranges,
)
from kedi.artifacts.store import store_size
from kedi.artifacts.stream import ArtifactStream
from kedi.artifacts.stream_writer import _StreamingWrite
from kedi.artifacts.streaming import ArtifactStreamEncoder, StreamedArtifact
from kedi.artifacts.tools import artifact_tool_specs
from kedi.lang.compiler.environment import (
    KediEnv,
    model_lookup_value,
    model_text,
    model_value,
    native_value,
)
from kedi.promise import KediPromise
from tests.telemetry_helpers import installed_recording_backend


def policy(**values: object) -> ArtifactPolicy:
    return ArtifactPolicy.patch(
        {
            "enabled": True,
            "threshold": 1,
            "cleanup_interval": "1h",
            **values,
        }
    ).resolved()


def test_store_public_exports_include_lazy_implementations() -> None:
    namespace: dict[str, Any] = {}
    exec("from kedi.artifacts.store import *", namespace)

    assert namespace["FileArtifactStore"] is FileArtifactStore
    assert namespace["InMemoryArtifactStore"] is InMemoryArtifactStore
    assert namespace["store_size"] is store_size


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
    assert len(base.resolved().as_cache_key()) == 12
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


def test_file_store_ignores_invalid_metadata_and_reads_without_native_cache(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    store = FileArtifactStore(root)
    record = _record()
    store.put(record, serialize_artifact("value", preview_chars=10))
    assert store.get_value("session", "art_test") == "value"
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


def test_file_store_restore_removes_interrupted_files_without_touching_active_writer(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifacts"
    store = FileArtifactStore(root)
    store.put(_record(), serialize_artifact("value", preview_chars=10))
    writer = store.begin_write("session", "art_active")
    writer.write(b"value")
    session = root / "session"
    interrupted = session / ".art_interrupted.token.tmp"
    orphaned_payload = session / "art_orphan.payload"
    orphaned_index = session / "art_orphan.index"
    interrupted.write_bytes(b"partial")
    orphaned_payload.write_bytes(b"orphan")
    orphaned_index.write_bytes(b"orphan")

    records = store.restore_session(
        "session",
        monotonic_now=0,
        wall_now=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )

    assert [record.ref_id for record in records] == ["art_test"]
    assert not interrupted.exists()
    assert not orphaned_payload.exists()
    assert not orphaned_index.exists()
    assert tuple(session.glob(".art_active.*.tmp"))

    active_record = _record(ref_id="art_active")
    writer.commit(active_record, _streamed())
    assert store.get_value("session", "art_active") == "value"
    store.close()


def test_file_store_handles_concurrent_independent_sessions(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"

    def round_trip(index: int) -> tuple[str, str]:
        session_id = f"session_{index}"
        ref_id = f"tool_call_result_{index}"
        value = f"{index:05d}"
        writer = FileArtifactStore(root)
        writer.put(
            _record(ref_id=ref_id, session_id=session_id),
            serialize_artifact(value, preview_chars=10),
        )
        writer.close()

        reader = FileArtifactStore(root)
        try:
            records = reader.restore_session(
                session_id,
                monotonic_now=0,
                wall_now=datetime(2026, 1, 1, tzinfo=timezone.utc),
            )
            assert [record.ref_id for record in records] == [ref_id]
            return ref_id, reader.get_value(session_id, ref_id)
        finally:
            reader.close()

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = tuple(executor.map(round_trip, range(32)))

    assert results == tuple((f"tool_call_result_{index}", f"{index:05d}") for index in range(32))


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

    root = tmp_path / "artifacts"
    store = FileArtifactStore(root)
    original = artifacts_projection_module._atomic_write
    calls = 0

    def fail_metadata(path: Path, payload: bytes, *, mode: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("metadata write failed")
        original(path, payload, mode=mode)

    monkeypatch.setattr(artifacts_file_store_module, "_atomic_write", fail_metadata)
    with pytest.raises(OSError, match="metadata"):
        store.put(_record(), serialize_artifact("value", preview_chars=10))
    assert not (root / "session" / "art_test.payload").exists()
    store.close()


def test_atomic_write_removes_temporary_file_on_replace_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    target = tmp_path / "target"

    def fail_replace(self: Path, target: Path) -> Path:
        raise OSError(f"cannot replace {target}")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="cannot replace"):
        artifacts_projection_module._atomic_write(target, b"value", mode=0o600)
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
        manager.release(first.ref.ref_id)
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


def test_file_json_pointer_index_survives_restore_without_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    file_policy = policy(store="file", path=tmp_path)
    first = ArtifactManager(session_id="pointer-restore")
    artifact = first.artifactize(
        {"şehir": {"a/b": ["zero", "Ağβ🙂Z"]}},
        policy=file_policy,
    )
    assert isinstance(artifact, ArtifactHandle)
    ref_id = artifact.ref.ref_id
    first.close()

    second = ArtifactManager(session_id="pointer-restore")
    try:
        store = second._store_for_policy(file_policy)

        def reject_materialization(*args: object, **kwargs: object) -> object:
            raise AssertionError("restored JSON pointer reads must not call get_value")

        monkeypatch.setattr(store, "get_value", reject_materialization)
        chunk = second.read(ref_id, path="/şehir/a~1b/1", offset=1, max_chars=3)
        assert chunk.content == "Ağβ"
        assert chunk.next_offset == 4
    finally:
        second.close()


def test_file_json_pointer_rejects_corrupt_projection_index(tmp_path: Path) -> None:
    file_policy = policy(store="file", path=tmp_path)
    manager = ArtifactManager(session_id="pointer-corrupt")
    try:
        artifact = manager.artifactize({"value": "ok"}, policy=file_policy)
        assert isinstance(artifact, ArtifactHandle)
        index_path = next(tmp_path.rglob(f"{artifact.ref.ref_id}.index"))
        index_path.write_text('{"version":1}', encoding="utf-8")
        with pytest.raises(ArtifactSerializationError, match="projection index"):
            manager.read(artifact.ref.ref_id, path="/value")
    finally:
        manager.close()


def test_file_json_pointer_legacy_scan_does_not_materialize_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    file_policy = policy(store="file", path=tmp_path)
    manager = ArtifactManager(session_id="pointer-legacy")
    try:
        artifact = manager.artifactize(
            {"outer": [{"a/b": "Ağβ🙂Z"}]},
            policy=file_policy,
        )
        assert isinstance(artifact, ArtifactHandle)
        next(tmp_path.rglob(f"{artifact.ref.ref_id}.index")).unlink()
        store = manager._ref_stores[artifact.ref.ref_id]

        def reject_materialization(*args: object, **kwargs: object) -> object:
            raise AssertionError("legacy JSON pointer scans must not call get_value")

        monkeypatch.setattr(store, "get_value", reject_materialization)
        chunk = manager.read(
            artifact.ref.ref_id,
            path="/outer/0/a~1b",
            offset=1,
            offset_from="end",
            max_chars=3,
        )
        assert chunk.content == "β🙂Z"
        assert chunk.next_offset == 4
    finally:
        manager.close()


def _streamed(codec: str = "text", *, size: int = 5) -> StreamedArtifact:
    return StreamedArtifact(
        codec=codec,
        media_type="text/plain",
        logical_type="str",
        size_bytes=size,
        char_count=size,
        preview="value",
        summary="value",
        digest="digest",
    )


def test_memory_stream_writer_rejects_identity_duplicates_and_closed_state() -> None:
    store = InMemoryArtifactStore()
    writer = store.begin_write("session", "art_test")
    writer.write(b"value")
    with pytest.raises(ArtifactSerializationError, match="identity"):
        writer.commit(_record(ref_id="other"), _streamed())

    writer.commit(_record(), _streamed())
    with pytest.raises(ArtifactSerializationError, match="already closed"):
        writer.write(b"again")

    duplicate = store.begin_write("session", "art_test")
    duplicate.write(b"value")
    with pytest.raises(ArtifactSerializationError, match="already exists"):
        duplicate.commit(_record(), _streamed())
    duplicate.abort()
    store.close()


def test_file_stream_writer_rejects_identity_duplicates_and_closed_state(tmp_path: Path) -> None:
    store = FileArtifactStore(tmp_path / "writer")
    mismatch = store.begin_write("session", "art_wrong")
    mismatch.write(b"value")
    with pytest.raises(ArtifactSerializationError, match="identity"):
        mismatch.commit(_record(), _streamed())
    mismatch.abort()

    writer = store.begin_write("session", "art_test")
    writer.write(b"value")
    writer.commit(_record(), _streamed())
    writer.abort()
    with pytest.raises(ArtifactSerializationError, match="already closed"):
        writer.write(b"again")

    duplicate = store.begin_write("session", "art_test")
    duplicate.write(b"value")
    with pytest.raises(ArtifactSerializationError, match="already exists"):
        duplicate.commit(_record(), _streamed())
    assert store.get_value("session", "art_test") == "value"
    assert not tuple((tmp_path / "writer").rglob("*.tmp"))
    store.close()


def test_store_projection_helpers_cover_binary_unicode_and_invalid_payloads(tmp_path: Path) -> None:
    payload = tmp_path / "payload"
    payload.write_text("Ağβ🙂Z", encoding="utf-8")
    assert _count_utf8_chars(payload) == 5
    assert _read_utf8_chars(payload, start=0, limit=0) == ""
    assert _read_utf8_chars(payload, start=1, limit=3) == "ğβ🙂"
    assert _read_utf8_chars(payload, start=0, limit=20) == "Ağβ🙂Z"

    binary = tmp_path / "binary"
    binary.write_bytes(b"abcdef")
    assert _read_base64_chars(binary, start=2, limit=4) == "JjZG"

    invalid = tmp_path / "invalid"
    invalid.write_bytes(b"\xff")
    with pytest.raises(ArtifactSerializationError, match="not valid UTF-8"):
        _read_utf8_chars(invalid, start=0, limit=1)


def test_file_and_memory_projection_reject_invalid_codec_or_json_path(tmp_path: Path) -> None:
    memory = InMemoryArtifactStore()
    memory.put(_record(), serialize_artifact("value", preview_chars=10))
    with pytest.raises(ArtifactInputError, match="JSON paths"):
        memory.read_json_pointer_projection(
            "session",
            "art_test",
            pointer="/value",
            offset=0,
            offset_from="start",
            max_chars=10,
        )
    memory.close()

    store = FileArtifactStore(tmp_path / "projection")
    record = _record()
    record.codec = "opaque"
    directory = store._session_directory("session")
    payload_path = store._payload_path("session", "art_test")
    payload_path.write_bytes(b"value")
    record.payload = payload_path
    store._records[record.ref_id] = record
    with pytest.raises(ArtifactSerializationError, match="no file-backed text projection"):
        store.read_projection(
            "session",
            "art_test",
            offset=0,
            offset_from="start",
            max_chars=10,
        )
    assert directory.exists()
    store.close()


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        [[0]],
        [[-1, 0]],
        [[1, 1]],
        [[0, 0], [0, 1]],
    ],
)
def test_projection_checkpoint_validation_rejects_malformed_values(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        _validated_checkpoints(value)


@pytest.mark.parametrize(
    "value",
    [
        None,
        {1: [0, 1, 0, 1]},
        {"/a": [0, 1, 0]},
        {"/a": [-1, 1, 0, 1]},
        {"/a": [2, 1, 0, 1]},
    ],
)
def test_projection_pointer_range_validation_rejects_malformed_values(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        _validated_pointer_ranges(value)


@pytest.mark.parametrize(
    "payload",
    [
        b"null trailing",
        b"?",
        b'{"a":}',
        b'{"a" 1}',
        b'{"a":1 x}',
        b"[1 x]",
        b'"unterminated',
        b'{"\\uZZZZ":1}',
    ],
)
def test_legacy_json_pointer_locator_rejects_malformed_json(
    tmp_path: Path,
    payload: bytes,
) -> None:
    path = tmp_path / "malformed.json"
    path.write_bytes(payload)
    with pytest.raises(ArtifactSerializationError):
        _locate_json_pointer_range(path, "/missing")


def test_projection_index_loader_rejects_all_malformed_fields(tmp_path: Path) -> None:
    path = tmp_path / "index"
    invalid_indexes = [
        [],
        {"version": 2},
        {
            "version": 1,
            "utf8_checkpoints": [[0, 0]],
            "json_pointer_ranges": {},
            "json_pointer_index_complete": "yes",
        },
    ]
    for value in invalid_indexes:
        path.write_text(json.dumps(value), encoding="utf-8")
        with pytest.raises(ArtifactSerializationError, match="projection index"):
            _load_projection_index(path)


def test_json_pointer_encoder_rejects_zero_cap_and_reports_partial_index() -> None:
    from kedi.artifacts.serialization import encode_json_with_pointers

    with pytest.raises(ValueError, match="must be positive"):
        encode_json_with_pointers({"value": 1}, max_entries=0)
    encoded = encode_json_with_pointers({"outer": {"inner": 1}}, max_entries=1)
    assert encoded.index_complete is False
    assert set(encoded.pointer_ranges) == {""}


def test_read_artifact_tool_rejects_pattern_with_json_path() -> None:
    manager = ArtifactManager(session_id="pattern-path")
    artifact = manager.artifactize({"value": "alpha"}, policy=policy())
    assert isinstance(artifact, ArtifactHandle)
    read = next(spec for spec in artifact_tool_specs(manager) if spec.name == "read_artifact")
    try:
        with pytest.raises(ValueError, match="does not support JSON Pointer"):
            read.fn(ref_id=artifact.ref.ref_id, pattern="alpha", path="/value")
    finally:
        manager.close()


def test_manager_find_and_read_reject_invalid_arguments_and_cap_results() -> None:
    manager = ArtifactManager(session_id="manager-arguments")
    artifact = manager.artifactize("alpha alpha alpha", policy=policy(read_max_chars=5))
    assert isinstance(artifact, ArtifactHandle)
    try:
        with pytest.raises(ValueError, match="cannot be empty"):
            manager.find(artifact.ref.ref_id, "")
        with pytest.raises(ValueError, match="must be positive"):
            manager.find(artifact.ref.ref_id, "alpha", max_matches=0)
        with pytest.raises(ValueError, match="offset_from"):
            manager.read(artifact.ref.ref_id, offset_from="middle")  # type: ignore[arg-type]
        assert len(manager.find(artifact.ref.ref_id, "alpha", max_matches=1)) == 1
        first = manager.read(artifact.ref.ref_id, max_chars=100)
        assert first.requested_max_chars == 100
        assert first.applied_max_chars == 5
        assert first.returned_chars == 5
        assert first.complete is False
        assert first.continuation == {
            "tool": "read_artifact",
            "arguments": {
                "ref_id": artifact.ref.ref_id,
                "offset": first.next_offset,
                "offset_from": "start",
                "max_chars": 5,
            },
        }
        second = manager.read(**first.continuation["arguments"])
        assert second.offset == first.next_offset
    finally:
        manager.close()


@pytest.mark.parametrize("store", ["memory", "file"])
@pytest.mark.parametrize("structured", [False, True])
def test_artifact_tail_continuation_has_no_overlap(
    tmp_path: Path, store: str, structured: bool
) -> None:
    manager = ArtifactManager(session_id=f"tail-{store}-{structured}")
    value = "abcdefghij"
    try:
        artifact = manager.artifactize(
            {"value": value} if structured else value,
            policy=policy(store=store, path=tmp_path, read_max_chars=4),
        )
        assert isinstance(artifact, ArtifactHandle)
        arguments: dict[str, Any] = {
            "ref_id": artifact.ref.ref_id,
            "offset_from": "end",
            "max_chars": 100,
        }
        if structured:
            arguments["path"] = "/value"
        pages: list[str] = []
        while True:
            chunk = manager.read(**arguments)
            pages.append(chunk.content)
            if chunk.complete:
                assert chunk.continuation is None
                break
            assert chunk.continuation is not None
            arguments = chunk.continuation["arguments"]
        assert "".join(reversed(pages)) == (json.dumps(value) if structured else value)
        assert (
            manager.read(
                artifact.ref.ref_id,
                offset=100,
                offset_from="end",
                path="/value" if structured else None,
            ).content
            == ""
        )
    finally:
        manager.close()


def test_async_stream_failure_aborts_transaction_and_closes_source() -> None:
    async def run() -> None:
        closed = False

        async def chunks() -> AsyncIterator[str]:
            nonlocal closed
            try:
                yield "large"
                raise RuntimeError("stream failed")
            finally:
                closed = True

        manager = ArtifactManager(session_id="async-stream-failure")
        try:
            with pytest.raises(RuntimeError, match="stream failed"):
                await manager.artifactize_stream_async(
                    ArtifactStream.text(chunks()),
                    policy=policy(threshold=1),
                )
            assert closed
            assert manager._reserved_artifacts == 0
            assert manager._reserved_bytes == 0
            assert manager.stats()["active_artifacts"] == 0
        finally:
            manager.close()

    asyncio.run(run())


def test_manager_cleanup_error_is_reported_and_reservations_handle_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = 0.0

    def monotonic() -> float:
        return clock

    manager = ArtifactManager(session_id="cleanup-error", monotonic=monotonic)
    artifact = manager.artifactize("value", policy=policy(ttl="1s"))
    assert isinstance(artifact, ArtifactHandle)
    clock = 2.0
    monkeypatch.setattr(
        manager,
        "_delete_payload",
        lambda _record: (_ for _ in ()).throw(RuntimeError("delete failed")),
    )
    try:
        with pytest.raises(RuntimeError, match="delete failed"):
            manager.cleanup_expired()
        manager._grow_stream_reservation(policy(), 0)
    finally:
        manager.close()


def test_manager_ref_collision_and_stream_state_invariants() -> None:
    manager = ArtifactManager(session_id="stream-state")
    stream = ArtifactStream.text([])
    encoder = ArtifactStreamEncoder(stream, preview_chars=10)
    state = _StreamingWrite(manager, policy(), None, False, stream, encoder)
    streamed = _streamed(size=0)
    try:
        manager._ref_stores["artifact_1"] = manager._memory_store
        assert manager._new_ref_id(None) == "artifact_2"

        state.finished = True
        with pytest.raises(RuntimeError, match="already finished"):
            state.finish(streamed)
        state.abort()

        incomplete = _StreamingWrite(manager, policy(), None, False, stream, encoder)
        incomplete.writer = object()
        with pytest.raises(RuntimeError, match="incomplete state"):
            incomplete.finish(streamed)
    finally:
        manager._ref_stores.pop("artifact_1", None)
        manager.close()


def test_stream_registration_and_discard_without_digest_or_metered_size() -> None:
    manager = ArtifactManager(session_id="stream-no-digest")
    store = InMemoryArtifactStore()
    record = _record(ref_id="artifact_1", session_id=manager.session_id)
    record.digest = None
    store.put(record, serialize_artifact("value", preview_chars=10))
    resolved = policy()
    try:
        handle = manager._register_streamed(record=record, store=store, policy=resolved)
        assert handle.ref.ref_id == "artifact_1"
        manager._metered_active_refs.pop("artifact_1")
        manager._discard_streamed(record=record, store=store)
        assert manager.stats()["active_artifacts"] == 0
    finally:
        manager.close()


def test_expiry_predicate_handles_terminal_idle_and_pending_records() -> None:
    released = _record(ref_id="released")
    released.status = ArtifactStatus.RELEASED
    assert not _is_expired(released, now=10_000)

    pending = _record(ref_id="pending")
    pending.status = ArtifactStatus.PENDING_RELEASE
    pending.expires_monotonic = 100
    pending.idle_ttl_seconds = 2
    pending.last_accessed_monotonic = 1
    assert not _is_expired(pending, now=2)
    assert _is_expired(pending, now=3)


def test_manager_find_handles_absent_and_duplicate_overlap_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = ArtifactManager(session_id="find-overlap")
    artifact = manager.artifactize("alpha", policy=policy(read_max_chars=5))
    assert isinstance(artifact, ArtifactHandle)
    calls = 0
    requested_sizes: list[int] = []

    def repeated_page(*_args: object, **_kwargs: object) -> tuple[str, int | None, bool]:
        nonlocal calls
        calls += 1
        requested_sizes.append(cast(int, _kwargs["max_chars"]))
        return ("alpha", 0, False) if calls == 1 else ("alpha", None, True)

    try:
        assert manager.find(artifact.ref.ref_id, "missing") == []
        monkeypatch.setattr(manager._memory_store, "read_projection", repeated_page)
        matches = manager.find(artifact.ref.ref_id, "alpha", context_chars=0)
        assert matches == [{"offset": 0, "match": "alpha", "context": "alpha"}]
        assert requested_sizes == [5, 5]
    finally:
        manager.close()


@pytest.mark.parametrize("pattern", ["z", "xyz"])
@pytest.mark.parametrize("store_kind", ["memory", "file"])
def test_find_retains_only_pattern_boundary_between_pages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pattern: str, store_kind: str
) -> None:
    manager = ArtifactManager(session_id="bounded-find")
    artifact = manager.artifactize(
        "a" * 8192 + "xyz", policy=policy(store=store_kind, path=tmp_path, read_max_chars=64)
    )
    assert isinstance(artifact, ArtifactHandle)
    overlaps: list[int] = []

    class ObservedPage(str):
        def __radd__(self, previous: str) -> str:
            overlaps.append(len(previous))
            return previous + str(self)

    store = manager._ref_stores[artifact.ref.ref_id]
    read_projection = store.read_projection

    def observed_page(*args: Any, **kwargs: Any) -> tuple[str, int | None, bool]:
        text, next_offset, complete = read_projection(*args, **kwargs)
        return ObservedPage(text), next_offset, complete

    try:
        monkeypatch.setattr(store, "read_projection", observed_page)
        matches = manager.find(artifact.ref.ref_id, pattern)
        assert [match["offset"] for match in matches] == [8195 - len(pattern)]
        assert len(overlaps) > 100
        assert max(overlaps) <= len(pattern) - 1
    finally:
        manager.close()


def test_file_store_reads_missing_char_count_and_binary_projection(tmp_path: Path) -> None:
    text_store = FileArtifactStore(tmp_path / "text")
    text_record = _record()
    text_record.ref = text_record.ref.model_copy(update={"char_count": None})
    text_store.put(text_record, serialize_artifact("Ağβ🙂Z", preview_chars=10))
    assert (
        text_store.read_projection(
            "session",
            "art_test",
            offset=1,
            offset_from="start",
            max_chars=3,
        )[0]
        == "ğβ🙂"
    )
    text_store.close()

    byte_store = FileArtifactStore(tmp_path / "bytes")
    byte_record = _record(ref_id="art_bytes")
    byte_record.codec = "bytes"
    byte_record.ref = byte_record.ref.model_copy(
        update={"ref_id": "art_bytes", "char_count": None, "size_bytes": 6}
    )
    byte_store.put(byte_record, serialize_artifact(b"abcdef", preview_chars=10))
    assert (
        byte_store.read_projection(
            "session",
            "art_bytes",
            offset=2,
            offset_from="start",
            max_chars=4,
        )[0]
        == "JjZG"
    )
    byte_store.close()


def test_file_json_pointer_rejects_invalid_non_json_and_missing_paths(tmp_path: Path) -> None:
    store = FileArtifactStore(tmp_path / "json-paths")
    text_record = _record(ref_id="art_text")
    text_record.ref = text_record.ref.model_copy(update={"ref_id": "art_text"})
    store.put(text_record, serialize_artifact("value", preview_chars=10))

    json_record = _record(ref_id="art_json")
    json_record.codec = "json"
    json_record.ref = json_record.ref.model_copy(update={"ref_id": "art_json"})
    store.put(json_record, serialize_artifact({"value": 1}, preview_chars=10))
    try:
        with pytest.raises(ValueError, match="RFC 6901"):
            store.read_json_pointer_projection(
                "session",
                "art_json",
                pointer="value",
                offset=0,
                offset_from="start",
                max_chars=10,
            )
        with pytest.raises(ArtifactInputError, match="JSON paths"):
            store.read_json_pointer_projection(
                "session",
                "art_text",
                pointer="/value",
                offset=0,
                offset_from="start",
                max_chars=10,
            )
        with pytest.raises(ValueError, match="was not found"):
            store.read_json_pointer_projection(
                "session",
                "art_json",
                pointer="/missing",
                offset=0,
                offset_from="start",
                max_chars=10,
            )

        store._projection_index_path("session", "art_json").unlink()
        store._projection_indexes.pop(("session", "art_json"), None)
        with pytest.raises(ValueError, match="was not found"):
            store.read_json_pointer_projection(
                "session",
                "art_json",
                pointer="/missing",
                offset=0,
                offset_from="start",
                max_chars=10,
            )
    finally:
        store.close()


def test_file_stream_writer_removes_payload_when_metadata_write_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    store = FileArtifactStore(tmp_path / "writer-failure")
    writer = store.begin_write("session", "art_test")
    writer.write(b"value")

    def fail_write(*_args: object, **_kwargs: object) -> None:
        raise OSError("metadata failed")

    monkeypatch.setattr(artifacts_file_store_module, "_atomic_write", fail_write)
    try:
        with pytest.raises(OSError, match="metadata failed"):
            writer.commit(_record(), _streamed())
        assert not store._payload_path("session", "art_test").exists()
        assert not store._projection_index_path("session", "art_test").exists()
    finally:
        writer.abort()
        store.close()


def test_file_stream_writer_commits_binary_without_projection_index(tmp_path: Path) -> None:
    store = FileArtifactStore(tmp_path / "binary-writer")
    writer = store.begin_write("session", "art_bytes")
    writer.write(b"abc")
    record = _record(ref_id="art_bytes")
    record.codec = "bytes"
    record.ref = record.ref.model_copy(
        update={"ref_id": "art_bytes", "size_bytes": 3, "char_count": None}
    )
    streamed = _streamed(codec="bytes", size=3)
    try:
        writer.commit(record, streamed)
        assert store.get_value("session", "art_bytes") == b"abc"
        assert not store._projection_index_path("session", "art_bytes").exists()
    finally:
        store.close()


def test_projection_helpers_cover_large_checkpoints_eof_and_skipped_binary_chunks(
    tmp_path: Path,
) -> None:
    large_text = tmp_path / "large-text"
    large_text.write_text("a" * 4_097, encoding="utf-8")
    checkpoints = _build_utf8_checkpoints(large_text)
    assert checkpoints[-1] == (4_096, 4_096)

    mixed_text = tmp_path / "mixed-large-text"
    mixed_text.write_text("a" * 65_536 + "ğ" + "b" * 4_096, encoding="utf-8")
    mixed_checkpoints = _build_utf8_checkpoints(mixed_text)
    assert mixed_checkpoints[-2:] == ((65_536, 65_536), (69_632, 69_633))

    incomplete = tmp_path / "incomplete-utf8"
    incomplete.write_bytes(b"\xc3")
    with pytest.raises(ArtifactSerializationError, match="not valid UTF-8"):
        _read_utf8_chars(incomplete, start=0, limit=2)

    binary = tmp_path / "large-binary"
    binary.write_bytes(b"a" * 60_000)
    assert _read_base64_chars(binary, start=70_000, limit=8) == "YWFhYWFh"
    assert _read_base64_chars(binary, start=90_000, limit=8) == ""


def test_utf8_reader_accepts_decoder_tail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    class Decoder:
        def decode(self, payload: bytes, final: bool = False) -> str:
            return "tail" if final else payload.decode("ascii")

    path = tmp_path / "tail"
    path.write_bytes(b"a")
    monkeypatch.setattr(
        artifacts_projection_module.codecs, "getincrementaldecoder", lambda _name: Decoder
    )
    assert _read_utf8_chars(path, start=0, limit=10) == "atail"


@pytest.mark.parametrize(
    ("payload", "pointer", "expected"),
    [
        (b"{}", "/missing", None),
        (b'{"a":1}', "/missing", None),
        (b'{"a":1, "b":2}', "/missing", None),
        (b"[]", "/0", None),
        (b"[1]", "/1", None),
        (b"[1, 2]", "/2", None),
    ],
)
def test_legacy_json_locator_handles_valid_missing_paths(
    tmp_path: Path,
    payload: bytes,
    pointer: str,
    expected: object,
) -> None:
    path = tmp_path / "valid.json"
    path.write_bytes(payload)
    assert _locate_json_pointer_range(path, pointer) is expected


def test_legacy_json_locator_defensive_parser_invariants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from io import BytesIO

    missing_key = _JsonPointerLocator(BytesIO(b'{"a":1}'), target=("missing",))
    monkeypatch.setattr(missing_key, "_parse_string", lambda *, decode: None)
    with pytest.raises(ArtifactSerializationError, match="object key"):
        missing_key.locate()

    invalid_key = _JsonPointerLocator(BytesIO(b'"x"'), target=())
    monkeypatch.setattr(artifacts_projection_module.json, "loads", lambda _value: 1)
    with pytest.raises(ArtifactSerializationError, match="key is not a string"):
        invalid_key._parse_string(decode=True)

    empty_literal = _JsonPointerLocator(BytesIO(b""), target=())
    with pytest.raises(ArtifactSerializationError, match="literal"):
        empty_literal._parse_literal()


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


def test_artifact_spans_never_attach_structured_payloads() -> None:
    with installed_recording_backend() as backend:
        with backend.start_span(
            scope="agent",
            name="call read_artifact",
            operation="call_tool",
            kind="internal",
            level="lifecycle",
            attributes=None,
        ) as parent:
            with artifact_telemetry.artifact_span(
                "read",
                ref_id="art_safe",
                size_bytes=12,
                payload={"secret": "must not be traced"},
                preview=["must not be traced"],
            ):
                pass

    assert backend.finished_spans == [parent]
    assert [(event.name, event.attributes) for event in parent.events] == [
        (
            "artifact read",
            {
                "kedi.artifact.ref_id": "art_safe",
                "kedi.artifact.size_bytes": 12,
            },
        )
    ]


def test_noop_cleanup_records_metrics_without_opening_a_span(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spans: list[str] = []
    cleanup_metrics: list[float] = []

    @contextmanager
    def artifact_span(action: str, **_attributes: object) -> Iterator[SimpleNamespace]:
        spans.append(action)
        yield SimpleNamespace(set_attribute=lambda *_args: None, record_exception=lambda _exc: None)

    monkeypatch.setattr(artifact_telemetry, "artifact_span", artifact_span)
    monkeypatch.setattr(
        artifact_telemetry,
        "record_cleanup",
        lambda *, duration_seconds: cleanup_metrics.append(duration_seconds),
    )

    manager = ArtifactManager(session_id="quiet-cleanup")
    try:
        assert manager.cleanup_expired() == 0
    finally:
        manager.close()

    assert spans == []
    assert len(cleanup_metrics) == 1


def test_productive_periodic_cleanup_uses_expiry_events_without_cleanup_span(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 0.0
    spans: list[str] = []

    @contextmanager
    def artifact_span(action: str, **_attributes: object) -> Iterator[SimpleNamespace]:
        spans.append(action)
        yield SimpleNamespace(set_attribute=lambda *_args: None, record_exception=lambda _exc: None)

    monkeypatch.setattr(artifact_telemetry, "artifact_span", artifact_span)
    manager = ArtifactManager(session_id="productive-cleanup", monotonic=lambda: now)
    try:
        manager.artifactize("payload", policy=policy(ttl="1s"))
        spans.clear()
        now = 2.0
        assert manager.cleanup_expired() == 1
    finally:
        manager.close()

    assert spans == ["expire"]


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


class GenerationChangeCleanupTarget:
    cleanup_interval_seconds_value = 0.01

    def __init__(self, service: ArtifactCleanupService) -> None:
        self.service = service
        self.entered = threading.Event()
        self.release = threading.Event()

    @property
    def cleanup_interval_seconds(self) -> float:
        self.entered.set()
        assert self.release.wait(timeout=1)
        return self.cleanup_interval_seconds_value

    def cleanup_expired(self) -> int:
        with self.service._condition:
            self.service._closing = True
            self.service._condition.notify_all()
        return 0


def test_cleanup_service_skips_stale_wait_after_registration_changes() -> None:
    service = ArtifactCleanupService()
    target = GenerationChangeCleanupTarget(service)
    service.register(target)
    assert target.entered.wait(timeout=1)

    service.unregister(target)
    service.register(target)
    target.release.set()

    assert service._thread is not None
    service._thread.join(timeout=1)
    assert not service._thread.is_alive()


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


def test_conversation_serialization_release_and_close() -> None:
    state = ConversationState(session_id="conversation_edges")

    class BrokenJSON:
        pass

    state.record_exchange("prompt", {"value": BrokenJSON()})
    first_snapshot = state.snapshot()
    assert "BrokenJSON" in first_snapshot[1].content
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
    turns_before_release = state.snapshot()
    state.artifacts.release(ref_id)
    with state.adapter_turn("mock") as resumed:
        assert resumed.resume_state == {"thread_id": "one"}
        resumed.result_state = {"thread_id": "two"}
    assert state.snapshot() == turns_before_release
    assert ref_id in state.snapshot()[-1].content
    assert state.history.snapshot()[-1].kind == "artifact_released"

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
        tools = {tool.name: tool for tool in artifact_tool_specs(manager)}
        read = tools["read_artifact"]
        search = tools["search_artifacts"]
        release = tools["release_artifact"]

        assert read.fn(ref_id=artifact.ref.ref_id)["content"] == "payload"
        assert search.fn(query="builtins.str")[0]["ref_id"] == artifact.ref.ref_id
        assert release.fn(ref_id=artifact.ref.ref_id)["status"] == "released"
    finally:
        manager.close()
