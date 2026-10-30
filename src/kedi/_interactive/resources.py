"""Session-local composition of existing conversation and artifact checkpoints."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable

from ..agent_adapter.conversation import ConversationState
from ..artifacts import ArtifactManager
from ..artifacts.file_store import FileArtifactStore
from ..artifacts.history import ArtifactHistory
from ..artifacts.models import ArtifactStatus
from ..artifacts.serialization import SerializedArtifact, serialize_artifact
from .persistence import SessionDumpError, SessionLoadError


@dataclass
class ResourceCapture:
    metadata: dict[str, Any]
    files: dict[str, bytes]
    identities: dict[int, str]
    limit_bytes: int


class RestoredArtifactManager(ArtifactManager):
    """Own extracted snapshot files without using provider-lane cleanup hooks."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.snapshot_cleanup: Callable[[], None] | None = None

    def close(self) -> None:
        cleanup, self.snapshot_cleanup = self.snapshot_cleanup, None
        cleanups: list[Callable[[], None]] = [super().close]
        if cleanup is not None:
            cleanups.append(cleanup)
        close_resources(cleanups)


def close_resources(
    cleanups: Iterable[Callable[[], None]], *, error: BaseException | None = None
) -> None:
    """Attempt every owned cleanup while preserving an in-flight failure."""
    failure = error
    for cleanup in cleanups:
        try:
            cleanup()
        except BaseException as exc:
            if failure is None:
                failure = exc
            else:
                failure.add_note(f"Snapshot resource cleanup failed: {exc}")
    if error is None and failure is not None:
        raise failure


def session_runtimes(session: Any) -> list[Any]:
    """Include initialized imported runtimes, without compiling/loading modules."""
    # Each initialized module owns one runtime, distinct from the session root.
    loaded = session._compiler.module_loader._loaded.values()
    return [session._compiler.rt, *(module.runtime for module in loaded)]


def capture_resources(session: Any) -> ResourceCapture:
    identities: dict[int, str] = {}
    managers: list[ArtifactManager] = []
    conversations: list[ConversationState] = []

    def add_manager(manager: ArtifactManager) -> str:
        token = identities.get(id(manager))
        if token is None:
            token = f"artifacts:{len(managers)}"
            identities[id(manager)] = token
            managers.append(manager)
        return token

    def add_conversation(conversation: ConversationState | None) -> None:
        if conversation is not None and id(conversation) not in identities:
            identities[id(conversation)] = f"conversation:{len(conversations)}"
            conversations.append(conversation)
            add_manager(conversation.artifacts)

    add_conversation(session._conversation_state)
    for runtime in session_runtimes(session):
        add_manager(runtime.artifacts)
        add_conversation(runtime.conversation_state)
        add_conversation(runtime._implicit_conversation_state)
    metadata: dict[str, Any] = {"managers": [], "conversations": []}
    files: dict[str, bytes] = {}
    limit = 0
    with TemporaryDirectory(prefix="kedi-snapshot-resources-") as directory:
        root = Path(directory)
        for manager in managers:
            token = identities[id(manager)]
            relative = f"resources/artifacts-{len(metadata['managers'])}"
            checkpoint, allowance = _stage_manager(manager, root / relative)
            limit += allowance
            metadata["managers"].append(
                {
                    "token": token,
                    "root": relative,
                    "checkpoint": checkpoint,
                    "durable": manager._durable_root is not None,
                }
            )
        for conversation in conversations:
            with conversation._lock:
                if conversation._next_ticket != conversation._next_turn:
                    raise SessionDumpError(["conversation has unfinished reserved turns"])
                lanes = tuple(conversation._adapter_states)
                if len(lanes) > 1:
                    raise SessionDumpError(["conversation contains multiple native adapter lanes"])
                lane = lanes[0] if lanes else "pydantic"
                try:
                    checkpoint = conversation.export_checkpoint(lane)
                except (RuntimeError, ValueError, TypeError) as exc:
                    raise SessionDumpError([f"conversation cannot be checkpointed: {exc}"]) from exc
                metadata["conversations"].append(
                    {
                        "token": identities[id(conversation)],
                        "artifacts": identities[id(conversation.artifacts)],
                        "adapter": lane,
                        "checkpoint": checkpoint,
                    }
                )
        # Kedi alone populates this private staging directory.
        for path in root.rglob("*"):
            if path.is_file():
                files[path.relative_to(root).as_posix()] = path.read_bytes()
    return ResourceCapture(metadata, files, identities, limit)


def _stage_manager(manager: ArtifactManager, root: Path) -> tuple[dict[str, Any], int]:
    """Copy logical records into a temporary durable store; do not mutate the live one."""
    with manager._lock:
        manager._ensure_open()
        if manager._reserved_artifacts or manager._reserved_bytes:
            raise SessionDumpError(["artifact manager has an unfinished producer"])
        records = manager._all_records()
        if any(record.active_readers for record in records):
            raise SessionDumpError(["artifact manager has active readers"])
        history = ArtifactHistory.from_checkpoint(manager.history.export_checkpoint())
        staged = ArtifactManager(
            session_id=manager.session_id,
            history=history,
            durable_root=root,
            monotonic=manager._monotonic,
            wall_clock=manager._wall_clock,
        )
        allowance = 0
        try:
            store = FileArtifactStore(root)
            staged._file_stores[store.root] = store
            for record in records:
                policy = manager._ref_policies[record.ref_id]
                allowance = max(allowance, policy.session_quota_bytes)
                inactive = record.status in {
                    ArtifactStatus.RELEASED,
                    ArtifactStatus.EXPIRED,
                    ArtifactStatus.PENDING_RELEASE,
                }
                if record.codec not in {"text", "json", "bytes"}:
                    raise SessionDumpError([f"artifact {record.ref_id!r} is not persistable"])
                if inactive:
                    payload = b""
                else:
                    source_store = manager._ref_stores[record.ref_id]
                    if isinstance(source_store, FileArtifactStore):
                        if not isinstance(record.payload, Path) or record.payload.is_symlink():
                            raise SessionDumpError(
                                [f"artifact {record.ref_id!r} has an unsafe payload"]
                            )
                        payload = record.payload.read_bytes()
                    else:
                        serialized = serialize_artifact(
                            source_store.get_value(manager.session_id, record.ref_id),
                            preview_chars=policy.preview_chars,
                            require_persistable=True,
                        )
                        if serialized.codec != record.codec or serialized.payload_bytes is None:
                            raise SessionDumpError([f"artifact {record.ref_id!r} changed codec"])
                        payload = serialized.payload_bytes
                    if sha256(payload).hexdigest() != record.digest:
                        raise SessionDumpError(
                            [f"artifact {record.ref_id!r} failed its digest check"]
                        )
                cloned = replace(record, active_readers=0)
                serialized = SerializedArtifact(
                    codec=record.codec,
                    media_type=record.ref.media_type,
                    logical_type=record.ref.logical_type,
                    size_bytes=len(payload),
                    char_count=record.ref.char_count,
                    preview=record.ref.preview,
                    summary=record.ref.summary,
                    payload_bytes=payload,
                    memory_value=None,
                    persistable=True,
                    digest=record.digest,
                    json_pointer_index_complete=False,
                )
                # Inactive JSON has no content to index; metadata retains the original codec.
                if inactive:
                    store.put(cloned, replace(serialized, codec="bytes"))
                else:
                    store.put(cloned, serialized)
                staged._ref_stores[record.ref_id] = store
                staged._ref_policies[record.ref_id] = policy
            checkpoint = json.loads(json.dumps(staged.export_checkpoint(), allow_nan=False))
            # Metadata and projection indexes are bounded separately from original payload quotas.
            allowance += 64 * 1024 * 1024
        except BaseException as exc:
            close_resources((staged.close,), error=exc)
            raise
        staged.close()
        return checkpoint, allowance


def restore_resources(
    metadata: dict[str, Any], root: Path
) -> tuple[dict[str, Any], list[Callable[[], None]]]:
    objects: dict[str, Any] = {}
    cleanups: list[Callable[[], None]] = []
    try:
        for saved in metadata["managers"]:
            token = saved["token"]
            relative = saved["root"]
            if (
                not isinstance(token, str)
                or token in objects
                or not isinstance(relative, str)
                or not relative.startswith("resources/")
                or Path(relative).is_absolute()
                or ".." in Path(relative).parts
            ):
                raise SessionLoadError("Invalid artifact owner in Python snapshot")
            manager = RestoredArtifactManager.from_checkpoint(
                saved["checkpoint"], durable_root=root / relative
            )
            objects[token] = manager
            cleanups.append(manager.close)
            if not saved["durable"]:
                manager._durable_root = None
        for saved in metadata["conversations"]:
            token = saved["token"]
            if not isinstance(token, str) or token in objects:
                raise SessionLoadError("Duplicate conversation owner in Python snapshot")
            conversation = ConversationState.from_checkpoint(
                saved["checkpoint"],
                adapter_shortname=saved["adapter"],
                artifacts=objects[saved["artifacts"]],
            )
            objects[token] = conversation
            cleanups.append(conversation.close)
        return objects, cleanups
    except BaseException as exc:
        close_resources(reversed(cleanups), error=exc)
        if isinstance(exc, (KeyboardInterrupt, SystemExit, SessionLoadError)):
            raise
        raise SessionLoadError(f"Cannot restore Python snapshot resources: {exc}") from exc


def restore_memory_artifacts(session: Any, resources: dict[str, Any]) -> None:
    """Retain native memory-value types/aliases instead of decoding them as JSON."""
    for token, ref_id, value in session._snapshot_memory_artifacts:
        manager = resources[token]
        store = manager._ref_stores[ref_id]
        # Staging rejected every codec other than text, bytes and JSON.
        record = store.get_record(manager.session_id, ref_id)
        manager._memory_store._records[ref_id] = replace(record, payload=value)
        store._records.pop(ref_id)
        manager._ref_stores[ref_id] = manager._memory_store
    del session._snapshot_memory_artifacts
