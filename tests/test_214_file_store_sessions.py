"""File artifact store session isolation (audit #214, item 4.9).

Ref IDs are per-session counters (text_1, text_2, ...), so two sessions
legitimately share a ref_id. The file store used to key its in-memory
records by ref_id alone: putting the second session's record raised
"already exists", and restoring the second session silently overwrote the
first session's record, so the first session's lookups were then reported
as belonging to the other session. Records are now keyed by
(session_id, ref_id).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from kedi.artifacts import ArtifactManager, ArtifactPolicy
from kedi.artifacts.errors import (
    ArtifactNotFoundError,
    ArtifactSerializationError,
)
from kedi.artifacts.file_store import FileArtifactStore
from kedi.artifacts.models import ArtifactRecord, ArtifactRef, ArtifactStatus
from kedi.artifacts.serialization import serialize_artifact


def _record(*, ref_id: str, session_id: str) -> ArtifactRecord:
    # Expiry is relative to now: a fixed past date would make restore mark
    # the record EXPIRED, which changes prune behavior.
    now = datetime.now(timezone.utc)
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


def test_file_store_keeps_same_ref_id_in_two_sessions(tmp_path: Path) -> None:
    """put() for a second session must not collide with the first session's
    record of the same ref_id, and each session reads back its own value."""
    store = FileArtifactStore(tmp_path / "artifacts")
    first = _record(ref_id="text_1", session_id="session-a")
    second = _record(ref_id="text_1", session_id="session-b")
    store.put(first, serialize_artifact("value-a", preview_chars=10))
    store.put(second, serialize_artifact("value-b", preview_chars=10))
    assert store.get_value("session-a", "text_1") == "value-a"
    assert store.get_value("session-b", "text_1") == "value-b"
    assert {record.ref_id for record in store.records("session-a")} == {"text_1"}
    assert {record.ref_id for record in store.records("session-b")} == {"text_1"}
    # A duplicate within the same session is still rejected.
    with pytest.raises(ArtifactSerializationError, match="already exists"):
        store.put(first, serialize_artifact("value-a", preview_chars=10))


def test_file_store_restore_does_not_overwrite_other_session(
    tmp_path: Path,
) -> None:
    """Restoring session B must not replace session A's record of the same
    ref_id - the exact failure the old ref_id-only key produced."""
    root = tmp_path / "artifacts"
    store = FileArtifactStore(root)
    first = _record(ref_id="text_1", session_id="session-a")
    store.put(first, serialize_artifact("value-a", preview_chars=10))
    second = _record(ref_id="text_1", session_id="session-b")
    other = FileArtifactStore(root)
    other.put(second, serialize_artifact("value-b", preview_chars=10))
    other.close()
    # Trigger restore of session B on the first store, which already holds A.
    assert store.get_value("session-b", "text_1") == "value-b"
    assert store.get_value("session-a", "text_1") == "value-a"
    assert store.get_record("session-a", "text_1").session_id == "session-a"
    assert store.get_record("session-b", "text_1").session_id == "session-b"


def test_file_store_missing_record_is_not_found(tmp_path: Path) -> None:
    store = FileArtifactStore(tmp_path / "artifacts")
    with pytest.raises(ArtifactNotFoundError, match="not found"):
        store.get_record("session-a", "missing")
    with pytest.raises(ArtifactNotFoundError, match="not found"):
        store.get_value("session-a", "missing")


def test_file_store_prune_inactive_payload(tmp_path: Path) -> None:
    store = FileArtifactStore(tmp_path / "artifacts")
    record = _record(ref_id="text_1", session_id="session-a")
    store.put(record, serialize_artifact("value-a", preview_chars=10))
    stored = store.get_record("session-a", "text_1")
    with pytest.raises(ValueError, match="active artifact"):
        store.prune_inactive_payload("session-a", "text_1")
    stored.status = ArtifactStatus.RELEASED
    stored.active_readers = 1
    with pytest.raises(ValueError, match="active readers"):
        store.prune_inactive_payload("session-a", "text_1")
    stored.active_readers = 0
    store.prune_inactive_payload("session-a", "text_1")
    assert store._payload_path("session-a", "text_1").read_bytes() == b""


def test_managers_sharing_a_root_keep_their_ref_ids(tmp_path: Path) -> None:
    """End to end through ArtifactManager: two sessions over one root both
    allocate ref_id artifact_1 and each reads back its own content."""
    root = tmp_path / "artifacts"
    policy = ArtifactPolicy.patch(
        {"enabled": True, "threshold": 1, "store": "file", "path": root}
    )
    first = ArtifactManager(session_id="session-a")
    handle_a = first.artifactize("content-a", policy=policy)
    second = ArtifactManager(session_id="session-b")
    handle_b = second.artifactize("content-b", policy=policy)
    assert handle_a.ref.ref_id == handle_b.ref.ref_id
    assert first.get_value(handle_a.ref.ref_id) == "content-a"
    assert second.get_value(handle_b.ref.ref_id) == "content-b"
    first.close()
    second.close()
