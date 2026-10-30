"""Explicit Python snapshots at the interactive-session ownership boundary."""

from __future__ import annotations

import shutil
import weakref
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .bundle import read_python_bundle, write_python_bundle
from .persistence import SessionDumpError, SessionLoadError
from .resources import (
    capture_resources,
    close_resources,
    restore_memory_artifacts,
    restore_resources,
)


def dump_session(session: Any, path: str | Path) -> Path:
    from .python_graph import dumps_session

    # Quiescence and ownership checks are part of the graph inventory.
    try:
        resources = capture_resources(session)
        limits = session._subagent_usage_limits
        metadata = {
            "session": {
                "cwd": str(session._cwd),
                "subagent_max_depth": session._subagent_max_depth,
                "subagent_max_concurrency": session._subagent_max_concurrency,
                "subagent_timeout_seconds": session._subagent_timeout_seconds,
                "subagent_usage_limits": None if limits is None else asdict(limits),
                "loop_iteration_limit": session._loop_iteration_limit,
            },
            "resources": resources.metadata,
            "resource_limit_bytes": resources.limit_bytes,
            "root_conversation": resources.identities.get(id(session._conversation_state)),
        }
        graph, modules = dumps_session(session, resources.identities)
        result = write_python_bundle(path, graph, metadata, resources.files, modules)
        session._snapshot_format = "python"
        return result
    except SessionDumpError:
        raise
    except Exception as exc:
        raise SessionDumpError([f"Cannot capture Python session: {exc}"]) from exc


def load_session(
    session_type: type,
    path: str | Path,
    *,
    adapter: Any = None,
    executor: Any = None,
    engine: Any = None,
    conversation_state: Any = None,
    close_conversation_state: bool | None = None,
    trusted: bool = False,
) -> Any:
    # Never import the executable graph machinery before trust and manifest validation.
    bundle = read_python_bundle(path, trusted=trusted)
    session = None
    resource_cleanups = []
    try:
        from ..agent_adapter.subagents.limits import SubagentUsageLimits
        from .python_graph import loads_session

        metadata = bundle.metadata
        saved = dict(metadata["session"])
        cwd = Path(saved["cwd"])
        if not cwd.is_absolute() or not cwd.is_dir():
            raise SessionLoadError("Python snapshot working directory is unavailable")
        limits = saved.pop("subagent_usage_limits")
        if limits is not None:
            saved["subagent_usage_limits"] = SubagentUsageLimits(**limits)
        if conversation_state is not None:
            _validate_supplied_conversation(conversation_state, metadata)
        objects, resource_cleanups = restore_resources(metadata["resources"], bundle.root)
        session = session_type(adapter=adapter, executor=executor, engine=engine, **saved)
        # Dispose only the empty artifact manager created for this new session.
        session._compiler.rt.artifacts.close()
        original_conversation = None
        if conversation_state is not None:
            token = metadata["root_conversation"]
            original_conversation = objects[token]
            objects[token] = conversation_state
        loads_session(session, bundle.graph, objects)
        restore_memory_artifacts(session, objects)
        if original_conversation is not None:
            assert conversation_state is not None
            previous_manager = (
                conversation_state.artifacts if conversation_state._owns_artifacts else None
            )
            # Graph references already point to the supplied owner. Publish its saved
            # logical state only after all graph hydration has succeeded.
            _install_conversation(conversation_state, original_conversation)
            resource_cleanups = [
                cleanup
                for cleanup in resource_cleanups
                if getattr(cleanup, "__self__", None) is not original_conversation
            ]
            if previous_manager is not None:
                # Keep the replaced empty owner's cleanup with the restored owner,
                # rather than mutating caller resources during a fallible load.
                resource_cleanups.append(previous_manager.close)
        root_conversation = session._conversation_state
        if root_conversation is not None and (
            conversation_state is not None or close_conversation_state is False
        ):
            manager = root_conversation.artifacts
            cleanups = [
                cleanup
                for cleanup in reversed(resource_cleanups)
                if getattr(cleanup, "__self__", None) not in (root_conversation, manager)
            ]

            def release_snapshot() -> None:
                close_resources([*cleanups, bundle.cleanup])

            # This new, session-specific manager owns only restored data. Its close
            # callback is not a provider-lane cleanup and survives cache rotation.
            manager.snapshot_cleanup = release_snapshot
            root_conversation._owns_artifacts = True
            session._close_conversation_state = bool(close_conversation_state)
            extraction_owner: Any = manager
        else:
            if close_conversation_state is not None:
                session._close_conversation_state = close_conversation_state
            session._snapshot_cleanups.extend(reversed(resource_cleanups))
            session._snapshot_cleanups.append(bundle.cleanup)
            extraction_owner = session
        # Extracted payloads must not outlive an owner that is never closed. The
        # callback holds only the path, not the owner or the decoded graph.
        weakref.finalize(extraction_owner, shutil.rmtree, bundle.root, True)
        session._snapshot_format = "python"
        return session
    except BaseException as exc:
        cleanups = []
        if session is not None:
            # Failed load must not close caller-owned infrastructure/conversation.
            session._close_conversation_state = False
            cleanups.append(session.close)
        cleanups.extend(reversed(resource_cleanups))
        cleanups.append(bundle.cleanup)
        close_resources(cleanups, error=exc)
        if isinstance(exc, (KeyboardInterrupt, SystemExit, SessionLoadError)):
            raise
        raise SessionLoadError(f"Cannot restore Python session: {exc}") from exc


def _validate_supplied_conversation(conversation: Any, metadata: dict[str, Any]) -> None:
    token = metadata.get("root_conversation")
    records = metadata["resources"]["conversations"]
    checkpoint = next((item["checkpoint"] for item in records if item["token"] == token), None)
    if checkpoint is None or conversation.session_id != checkpoint["session_id"]:
        raise SessionLoadError("Supplied conversation does not match the saved session")
    with conversation._lock:
        if (
            conversation._closed
            or conversation._active_ticket is not None
            or conversation._next_ticket != conversation._next_turn
            or conversation._turns
            or conversation._adapter_states
            or conversation._adapter_cleanups
            or conversation._cache_epoch
            or conversation.artifacts._closed
            or conversation.artifacts._reserved_artifacts
            or conversation.artifacts._reserved_bytes
            or conversation.artifacts._ref_stores
        ):
            raise SessionLoadError("Supplied conversation must be empty and open")


def _install_conversation(target: Any, source: Any) -> None:
    with target._lock:
        target.artifacts = source.artifacts
        target.history = source.history
        target._turns = source._turns
        target._adapter_states = source._adapter_states
        target._adapter_lane_identities = source._adapter_lane_identities
        target._cache_epoch = source._cache_epoch
        target._owns_artifacts = True
