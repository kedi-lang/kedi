from __future__ import annotations

import asyncio
import base64
import builtins
import copy
import json
import multiprocessing
import os
import sqlite3
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from a2a.types import Task, TaskState, TaskStatus
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage, message_to_dict
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel

from kedi.a2a import persistence as persistence_module
from kedi.a2a import runtime as a2a_runtime
from kedi.a2a._storage_io import finish_storage_operation
from kedi.a2a.auth import A2AServerAuth
from kedi.a2a.persistence import A2APersistence
from kedi.a2a.runtime import _source_fingerprint, prepare_a2a_program
from kedi.a2a.server import create_a2a_app
from kedi.agent_adapter.adapters import LangChainAdapter, PydanticAdapter
from kedi.agent_adapter.conversation import ConversationState, _checkpoint_lane_state
from kedi.agent_adapter.history_provenance import HistoryLineage
from kedi.artifacts.errors import ArtifactExpiredError, ArtifactNotFoundError, ArtifactReleasedError
from kedi.artifacts.history import ArtifactHistory
from kedi.artifacts.manager import ArtifactManager
from kedi.artifacts.models import ArtifactStatus
from kedi.artifacts.policy import ArtifactPolicy
from kedi.lang import parse_program

sys.path.insert(0, str(Path(__file__).parent))
from test_a2a import _ServedAdapter  # noqa: E402


@pytest.mark.parametrize("fail", [False, True])
def test_storage_worker_finishes_before_repeated_cancellation(fail: bool) -> None:
    async def scenario() -> None:
        loop = asyncio.get_running_loop()
        started = asyncio.Event()
        release = threading.Event()
        finished = threading.Event()

        def work() -> None:
            loop.call_soon_threadsafe(started.set)
            assert release.wait(5)
            finished.set()
            if fail:
                raise OSError("write failed")

        operation = asyncio.create_task(finish_storage_operation(asyncio.to_thread(work)))
        await asyncio.wait_for(started.wait(), 5)
        try:
            for _ in range(2):
                operation.cancel()
                await asyncio.sleep(0)
                assert not operation.done()
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(operation, 5)
        assert finished.is_set()

    asyncio.run(scenario())


def test_storage_operation_preserves_its_own_cancellation() -> None:
    async def cancelled() -> None:
        raise asyncio.CancelledError()

    async def scenario() -> None:
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(finish_storage_operation(cancelled()), 5)

    asyncio.run(scenario())


def test_cancelling_state_startup_releases_eventually_acquired_lock(
    tmp_path: Path, monkeypatch: Any
) -> None:
    async def scenario() -> None:
        state = _test_state(tmp_path / "state", identity="program-1", max_tasks_per_principal=2)
        acquire = state._lock.acquire
        loop = asyncio.get_running_loop()
        started = asyncio.Event()
        release = threading.Event()

        def blocked_acquire(**kwargs: Any) -> Any:
            loop.call_soon_threadsafe(started.set)
            assert release.wait(5)
            return acquire(**kwargs)

        monkeypatch.setattr(state._lock, "acquire", blocked_acquire)
        opening = asyncio.create_task(state.open())
        await asyncio.wait_for(started.wait(), 5)
        opening.cancel()
        await asyncio.sleep(0)
        assert not opening.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(opening, 5)
        assert not state._lock.is_locked
        reopened = _test_state(tmp_path / "state", identity="program-1", max_tasks_per_principal=2)
        await reopened.open()
        await reopened.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("action", ["abort", "close"])
def test_completion_cleanup_waits_for_inflight_transaction(
    tmp_path: Path, monkeypatch: Any, action: str
) -> None:
    async def scenario() -> None:
        state = _test_state(tmp_path / "state", identity="program-1", max_tasks_per_principal=2)
        await state.open()
        try:
            owner = _context("alice")
            await state.save(_task("task-1", TaskState.TASK_STATE_SUBMITTED), owner)
            artifacts = ArtifactManager(durable_root=state.artifact_root)
            checkpoint = {
                "version": 1,
                "artifacts": artifacts.export_checkpoint(),
                "conversation": None,
            }
            artifacts.close()
            committed = state.stage_completion(
                task_id="task-1", owner="alice", context_id="context-1", checkpoint=checkpoint
            )
            started = asyncio.Event()
            release = asyncio.Event()
            original = state._save_locked

            async def blocked_save(task: Task, context: Any) -> None:
                started.set()
                await release.wait()
                await original(task, context)

            monkeypatch.setattr(state, "_save_locked", blocked_save)
            saving = asyncio.create_task(
                state.save(_task("task-1", TaskState.TASK_STATE_COMPLETED), owner)
            )
            await started.wait()
            aborting = asyncio.create_task(
                state.abort_completion("task-1") if action == "abort" else state.close()
            )
            await asyncio.sleep(0)
            assert not aborting.done()
            release.set()
            await asyncio.wait_for(asyncio.gather(saving, aborting), 5)
            await committed
            if action == "close":
                await state.open()
            assert await state.load_checkpoint("alice", "context-1") == checkpoint
            assert (await state.get("task-1", owner)).status.state == TaskState.TASK_STATE_COMPLETED
        finally:
            await state.close()

    asyncio.run(scenario())


def test_lower_retention_limit_evicts_enough_terminal_tasks(tmp_path: Path) -> None:
    async def scenario() -> None:
        root = tmp_path / "state"
        state = _test_state(root, identity="program-1", max_tasks_per_principal=3)
        owner = _context("alice")
        await state.open()
        for index in range(3):
            await state.save(_task(f"old-{index}", TaskState.TASK_STATE_FAILED), owner)
        await state.close()
        state = _test_state(root, identity="program-1", max_tasks_per_principal=1)
        await state.open()
        try:
            await state.save(_task("new", TaskState.TASK_STATE_SUBMITTED), owner)
            for index in range(3):
                assert await state.get(f"old-{index}", owner) is None
            assert await state.get("new", owner) is not None
        finally:
            await state.close()

    asyncio.run(scenario())


def test_durable_program_configuration_and_commit_requirements(tmp_path: Path) -> None:
    async def scenario() -> None:
        source = tmp_path / "agent.kedi"
        source.write_text(
            "> profile: remote:\n    > instructions: Answer exactly.\n\n> export:\n    remote\n",
            encoding="utf-8",
        )
        program = prepare_a2a_program(
            source, entry="remote", model="test:model", adapter=_ServedAdapter()
        )
        state = _test_state(tmp_path / "state", identity="program-1", max_tasks_per_principal=2)
        program.configure_persistence(state)
        await state.open()
        try:
            with pytest.raises(RuntimeError, match="already configured"):
                program.configure_persistence(state)
            with pytest.raises(RuntimeError, match="commit callback"):
                await program.invoke(
                    principal="alice", context_id="context-1", prompt="hello", output_schema=None
                )
            assert program._sessions[("alice", "context-1")].poisoned
        finally:
            await program.aclose()
            await state.close()

        source.write_text(
            "> profile: remote:\n    > instructions: Answer exactly.\n"
            "    > history:\n        enabled: true\n\n> export:\n    remote\n",
            encoding="utf-8",
        )

        class UnsupportedHistoryAdapter(PydanticAdapter):
            shortname = "unsupported-history"

        program = prepare_a2a_program(
            source,
            entry="remote",
            model="test:model",
            adapter=UnsupportedHistoryAdapter(
                FunctionModel(lambda *_: ModelResponse(parts=[TextPart("ok")]))
            ),
        )
        try:
            with pytest.raises(ValueError, match="does not support adapter"):
                program.configure_persistence(state)
        finally:
            await program.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("history", [False, True])
@pytest.mark.parametrize("failure", ["version", "hydrate", "close", "prune"])
def test_durable_session_failure_cleanup(
    tmp_path: Path, monkeypatch: Any, caplog: Any, history: bool, failure: str
) -> None:
    async def scenario() -> None:
        source = tmp_path / "agent.kedi"
        source.write_text(
            "> profile: remote:\n    > instructions: Answer exactly.\n"
            + ("    > history:\n        enabled: true\n" if history else "")
            + "\n> export:\n    remote\n",
            encoding="utf-8",
        )
        program = prepare_a2a_program(
            source,
            entry="remote",
            model="test:model",
            adapter=PydanticAdapter(
                FunctionModel(lambda *_: ModelResponse(parts=[TextPart("ok")]))
            ),
        )
        state = _test_state(tmp_path / "state", identity="program-1", max_tasks_per_principal=2)
        program.configure_persistence(state)
        await state.open()
        session = program._session("alice", "context-1")
        session.borrowers -= 1
        session.poisoned = True
        original_artifacts = session.artifacts
        closed: list[ArtifactManager] = []
        original_close = ArtifactManager.close

        def track_close(manager: ArtifactManager) -> None:
            closed.append(manager)
            original_close(manager)

        def fail(*_: Any, **__: Any) -> Any:
            raise OSError("storage test failure")

        async def commit(result: Any, checkpoint: dict[str, Any]) -> None:
            assert result == "ok"
            assert checkpoint["version"] == 1

        try:
            with monkeypatch.context() as patch:
                patch.setattr(ArtifactManager, "close", track_close)
                if failure == "version":

                    async def bad_checkpoint(*_: Any) -> Any:
                        return {"version": -1}

                    patch.setattr(state, "load_checkpoint", bad_checkpoint)
                elif failure == "hydrate":
                    patch.setattr(a2a_runtime, "AgentSurface", fail)
                elif failure == "close":
                    patch.setattr(type(session), "close", fail)
                else:
                    patch.setattr(ArtifactManager, "prune_inactive_payloads", fail)
                if failure == "prune":
                    assert (
                        await program.invoke(
                            principal="alice",
                            context_id="context-1",
                            prompt="hello",
                            output_schema=None,
                            on_complete=commit,
                        )
                        == "ok"
                    )
                    assert "could not prune" in caplog.text
                    assert not session.poisoned
                else:
                    with pytest.raises((RuntimeError, OSError)):
                        await program.invoke(
                            principal="alice",
                            context_id="context-1",
                            prompt="hello",
                            output_schema=None,
                            on_complete=commit,
                        )
                    assert session.artifacts is original_artifacts
                    assert session.poisoned
                    if failure != "version":
                        assert any(manager is not original_artifacts for manager in closed)
        finally:
            await program.aclose()
            await state.close()

    asyncio.run(scenario())


def test_artifact_pruning_rejects_unsafe_state(tmp_path: Path, monkeypatch: Any) -> None:
    manager = ArtifactManager()
    assert manager.prune_inactive_payloads({}) == 0
    manager.close()
    manager = ArtifactManager(durable_root=tmp_path)
    try:
        handle = manager.artifactize("payload" * 20, policy=ArtifactPolicy(threshold_bytes=1))
        ref_id = handle.ref.ref_id
        checkpoint = manager.export_checkpoint()
        with pytest.raises(ValueError, match="another session"):
            manager.prune_inactive_payloads({"session_id": "other"})
        store = manager._ref_stores[ref_id]
        record = store.get_record(manager.session_id, ref_id)
        with pytest.raises(ValueError, match="active artifact"):
            store.prune_inactive_payload(manager.session_id, ref_id)
        checkpoint["records"][0]["status"] = ArtifactStatus.RELEASED.value
        assert manager.prune_inactive_payloads(checkpoint) == 0
        record.status = ArtifactStatus.RELEASED
        record.active_readers = 1
        with pytest.raises(ValueError, match="active readers"):
            store.prune_inactive_payload(manager.session_id, ref_id)
        assert manager.prune_inactive_payloads(checkpoint) == 0
        record.active_readers = 0
        with monkeypatch.context() as patch:
            patch.setitem(manager._ref_stores, ref_id, manager._memory_store)
            with pytest.raises(RuntimeError, match="not file-backed"):
                manager.prune_inactive_payloads(checkpoint)
        assert manager.prune_inactive_payloads(checkpoint) == 1
        assert manager.prune_inactive_payloads(checkpoint) == 1
    finally:
        manager.close()


@pytest.mark.parametrize("durable", [False, True])
def test_executor_queue_failure_aborts_uncommitted_state(tmp_path: Path, durable: bool) -> None:
    from a2a.types import Message, Role, TaskArtifactUpdateEvent, TaskStatusUpdateEvent

    from kedi.a2a.server import _KediAgentExecutor

    async def scenario() -> None:
        state = _test_state(tmp_path / "state", identity="program-1", max_tasks_per_principal=2)
        await state.open()
        owner = _context("alice")
        context = SimpleNamespace(
            task_id="task-1",
            context_id="context-1",
            current_task=None,
            requested_extensions=[],
            metadata={},
            message=Message(role=Role.ROLE_USER),
            call_context=owner,
            get_user_input=lambda: "prompt",
        )

        async def enqueue(event: Any) -> None:
            if not durable or isinstance(event, TaskArtifactUpdateEvent):
                raise OSError("event delivery failed")
            if isinstance(event, Task):
                await state.save(event, owner)
            elif isinstance(event, TaskStatusUpdateEvent):
                task = await state.get("task-1", owner)
                task.status.CopyFrom(event.status)
                await state.save(task, owner)

        async def invoke(**kwargs: Any) -> str:
            await kwargs["on_complete"]("answer", {"version": 1})
            pytest.fail("failed delivery must not finish")

        executor = _KediAgentExecutor(
            SimpleNamespace(invoke=invoke), state=state if durable else None
        )
        try:
            if durable:
                await executor.execute(context, SimpleNamespace(enqueue_event=enqueue))
                task = await state.get("task-1", owner)
                assert task.status.state == TaskState.TASK_STATE_FAILED
                assert await state.load_checkpoint("alice", "context-1") is None
                assert not state._pending
            else:
                with pytest.raises(OSError, match="delivery failed"):
                    await executor.execute(context, SimpleNamespace(enqueue_event=enqueue))
        finally:
            await state.close()

    asyncio.run(scenario())


def test_artifact_checkpoint_accepts_non_counter_reference(tmp_path: Path) -> None:
    manager = ArtifactManager(durable_root=tmp_path)
    handle = manager.artifactize("payload" * 20, policy=ArtifactPolicy(threshold_bytes=1))
    checkpoint = manager.export_checkpoint()
    manager.close()
    directory = tmp_path / manager.session_id
    original = handle.ref.ref_id
    replacement = "external"
    metadata_path = directory / f"{original}.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["ref"]["ref_id"] = replacement
    (directory / f"{replacement}.json").write_text(json.dumps(metadata), encoding="utf-8")
    metadata_path.unlink()
    (directory / f"{original}.payload").rename(directory / f"{replacement}.payload")
    checkpoint["records"][0]["ref_id"] = replacement
    restored = ArtifactManager.from_checkpoint(checkpoint, durable_root=tmp_path)
    try:
        assert restored.get_value(replacement) == "payload" * 20
    finally:
        restored.close()


def test_cancelled_checkpoint_keeps_session_locked_until_worker_finishes(
    tmp_path: Path, monkeypatch: Any
) -> None:
    async def scenario() -> None:
        source = tmp_path / "agent.kedi"
        source.write_text(
            "> profile: remote:\n    > instructions: Answer exactly.\n\n> export:\n    remote\n",
            encoding="utf-8",
        )
        program = prepare_a2a_program(
            source, entry="remote", model="test:model", adapter=_ServedAdapter()
        )
        app = create_a2a_app(
            program,
            public_url="http://127.0.0.1:8000",
            auth=None,
            allow_unauthenticated_loopback=True,
            state_dir=tmp_path / "state",
        )
        loop = asyncio.get_running_loop()
        started = asyncio.Event()
        release = threading.Event()
        original = program._session_checkpoint

        def blocked_checkpoint(session: Any) -> dict[str, Any]:
            loop.call_soon_threadsafe(started.set)
            assert release.wait(5)
            return original(session)

        async def commit(_result: Any, _checkpoint: dict[str, Any]) -> None:
            pytest.fail("cancelled execution must not reach completion")

        monkeypatch.setattr(program, "_session_checkpoint", blocked_checkpoint)
        async with app.router.lifespan_context(app):
            invocation = asyncio.create_task(
                program.invoke(
                    principal="alice",
                    context_id="context-1",
                    prompt="hello",
                    output_schema=None,
                    on_complete=commit,
                )
            )
            await asyncio.wait_for(started.wait(), 5)
            invocation.cancel()
            await asyncio.sleep(0)
            session = program._sessions[("alice", "context-1")]
            assert session.lock.locked()
            assert not invocation.done()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(invocation, 5)
            assert not session.lock.locked()
            assert session.poisoned

    asyncio.run(scenario())


def test_interrupted_tasks_can_be_evicted_after_restart(tmp_path: Path) -> None:
    async def scenario() -> None:
        root = tmp_path / "state"
        owner = _context("alice")
        state = _test_state(root, identity="program-1", max_tasks_per_principal=1)
        await state.open()
        await state.save(_task("old", TaskState.TASK_STATE_WORKING), owner)
        await state.close()
        reopened = _test_state(root, identity="program-1", max_tasks_per_principal=1)
        await reopened.open()
        try:
            assert (await reopened.get("old", owner)).status.state == TaskState.TASK_STATE_FAILED
            await reopened.save(_task("new", TaskState.TASK_STATE_SUBMITTED), owner)
            assert await reopened.get("old", owner) is None
            assert await reopened.get("new", owner) is not None
        finally:
            await reopened.close()

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "late_state",
    [TaskState.TASK_STATE_WORKING, TaskState.TASK_STATE_FAILED, TaskState.TASK_STATE_CANCELED],
)
def test_completed_task_cannot_be_overwritten(tmp_path: Path, late_state: TaskState) -> None:
    async def scenario() -> None:
        state = _test_state(tmp_path / "state", identity="program-1", max_tasks_per_principal=2)
        await state.open()
        try:
            owner = _context("alice")
            await state.save(_task("task-1", TaskState.TASK_STATE_SUBMITTED), owner)
            checkpoint = {"version": 1, "turns": ["committed"]}
            commit = state.stage_completion(
                task_id="task-1", owner="alice", context_id="context-1", checkpoint=checkpoint
            )
            completed = _task("task-1", TaskState.TASK_STATE_COMPLETED)
            await state.save(completed, owner)
            await commit
            late = _task("task-1", late_state)
            with pytest.raises(RuntimeError, match="terminal task"):
                await state.save(late, owner)
            assert late == completed
            assert await state.get("task-1", owner) == completed
            assert await state.load_checkpoint("alice", "context-1") == checkpoint
            await state.save(completed, owner)
        finally:
            await state.close()

    asyncio.run(scenario())


def test_artifact_restore_preserves_elapsed_idle_ttl(tmp_path: Path) -> None:
    manager = ArtifactManager(
        durable_root=tmp_path,
        wall_clock=lambda: datetime.now(timezone.utc) - timedelta(minutes=2),
    )
    handle = manager.artifactize(
        "idle data" * 20,
        policy=ArtifactPolicy(threshold_bytes=1, ttl_seconds=3600, idle_ttl_seconds=60),
    )
    checkpoint = manager.export_checkpoint()
    manager.close()
    restored = ArtifactManager.from_checkpoint(checkpoint, durable_root=tmp_path)
    try:
        with pytest.raises(ArtifactExpiredError):
            restored.get_value(handle.ref.ref_id)
    finally:
        restored.close()


def test_artifact_idle_deadline_does_not_reset_on_restore(tmp_path: Path) -> None:
    manager = ArtifactManager(
        durable_root=tmp_path,
        wall_clock=lambda: datetime.now(timezone.utc) - timedelta(seconds=30),
    )
    manager.artifactize(
        "idle data" * 20,
        policy=ArtifactPolicy(threshold_bytes=1, ttl_seconds=3600, idle_ttl_seconds=60),
    )
    checkpoint = manager.export_checkpoint()
    manager.close()
    initial_deadline = datetime.fromisoformat(checkpoint["records"][0]["idle_expires_at"])
    for _ in range(2):
        restored = ArtifactManager.from_checkpoint(checkpoint, durable_root=tmp_path)
        try:
            checkpoint = restored.export_checkpoint()
            deadline = datetime.fromisoformat(checkpoint["records"][0]["idle_expires_at"])
            assert abs((deadline - initial_deadline).total_seconds()) < 0.1
        finally:
            restored.close()


@pytest.mark.parametrize("deadline", [None, "invalid", "2026-01-01T00:00:00"])
def test_artifact_restore_rejects_invalid_idle_deadline(tmp_path: Path, deadline: Any) -> None:
    manager = ArtifactManager(durable_root=tmp_path)
    manager.artifactize(
        "idle data" * 20, policy=ArtifactPolicy(threshold_bytes=1, idle_ttl_seconds=60)
    )
    checkpoint = manager.export_checkpoint()
    manager.close()
    checkpoint["records"][0]["idle_expires_at"] = deadline
    with pytest.raises(ValueError):
        ArtifactManager.from_checkpoint(checkpoint, durable_root=tmp_path)


def test_restored_artifacts_register_cleanup_and_balance_metrics(
    tmp_path: Path, monkeypatch: Any
) -> None:
    from kedi.artifacts import manager as artifact_module

    manager = ArtifactManager(durable_root=tmp_path)
    handle = manager.artifactize("data" * 20, policy=ArtifactPolicy(threshold_bytes=1))
    checkpoint = manager.export_checkpoint()
    manager.close()
    added: list[int] = []
    removed: list[int] = []
    registered: list[ArtifactManager] = []
    monkeypatch.setattr(
        artifact_module.telemetry, "record_active", lambda *, size_bytes: added.append(size_bytes)
    )
    monkeypatch.setattr(
        artifact_module.telemetry,
        "record_payload_removed",
        lambda *, size_bytes: removed.append(size_bytes),
    )
    monkeypatch.setattr(artifact_module.cleanup_service, "register", registered.append)
    restored = ArtifactManager.from_checkpoint(checkpoint, durable_root=tmp_path)
    assert registered == [restored]
    assert restored.history.export_checkpoint() == checkpoint["history"]
    restored.close()
    assert added == removed == [handle.ref.size_bytes]


def test_cancelled_task_rejects_late_completion(tmp_path: Path) -> None:
    async def scenario() -> None:
        state = _test_state(tmp_path / "state", identity="program-1", max_tasks_per_principal=2)
        await state.open()
        try:
            owner = _context("alice")
            await state.save(_task("task-1", TaskState.TASK_STATE_CANCELED), owner)
            committed = state.stage_completion(
                task_id="task-1", owner="alice", context_id="context-1", checkpoint={}
            )
            late = _task("task-1", TaskState.TASK_STATE_COMPLETED)
            with pytest.raises(RuntimeError, match="terminal task"):
                await state.save(late, owner)
            with pytest.raises(RuntimeError, match="terminal task"):
                await committed
            assert late.status.state == TaskState.TASK_STATE_CANCELED
            assert await state.load_checkpoint("alice", "context-1") is None
        finally:
            await state.close()

    asyncio.run(scenario())


def test_artifact_pruning_uses_committed_status(tmp_path: Path) -> None:
    manager = ArtifactManager(durable_root=tmp_path)
    handle = manager.artifactize("retained" * 20, policy=ArtifactPolicy(threshold_bytes=1))
    checkpoint = manager.export_checkpoint()
    manager.release(handle.ref.ref_id)
    assert manager.prune_inactive_payloads(checkpoint) == 0
    manager.close()
    restored = ArtifactManager.from_checkpoint(checkpoint, durable_root=tmp_path)
    try:
        assert restored.get_value(handle.ref.ref_id) == "retained" * 20
    finally:
        restored.close()


def test_pending_release_finishes_on_restore(tmp_path: Path) -> None:
    manager = ArtifactManager(durable_root=tmp_path)
    handle = manager.artifactize("retained" * 20, policy=ArtifactPolicy(threshold_bytes=1))
    with manager._lease(handle.ref.ref_id):
        assert manager.release(handle.ref.ref_id).status is ArtifactStatus.PENDING_RELEASE
        checkpoint = manager.export_checkpoint()
    manager.close()
    restored = ArtifactManager.from_checkpoint(checkpoint, durable_root=tmp_path)
    try:
        assert restored.release(handle.ref.ref_id).already_released
        assert restored.prune_inactive_payloads(checkpoint) == 1
    finally:
        restored.close()


def _context(owner: str) -> Any:
    return SimpleNamespace(user=SimpleNamespace(user_name=owner))


def _task(task_id: str, state: TaskState, *, context_id: str = "context-1") -> Task:
    return Task(id=task_id, context_id=context_id, status=TaskStatus(state=state))


def _test_state(root: Path, **kwargs: Any) -> A2APersistence:
    return A2APersistence(root, require_admission=False, **kwargs)


def test_rejected_admission_cannot_be_saved_later(tmp_path: Path) -> None:
    async def scenario() -> None:
        state = A2APersistence(
            tmp_path / "state",
            identity="program-1",
            max_tasks_per_principal=2,
            max_sessions=0,
        )
        await state.open()
        try:
            task = _task("task-1", TaskState.TASK_STATE_SUBMITTED)
            owner = _context("alice")
            with pytest.raises(RuntimeError, match="not admitted"):
                await state.save(task, owner)
            admission = state.expect_admission(task.id)
            with pytest.raises(RuntimeError, match="session capacity"):
                await state.save(task, owner)
            with pytest.raises(RuntimeError, match="session capacity"):
                await admission
            state.max_sessions = 1
            with pytest.raises(RuntimeError, match="not admitted"):
                await state.save(task, owner)
            assert await state.get(task.id, owner) is None
        finally:
            await state.close()

    asyncio.run(scenario())


def _hold_admitted_task(root: Path, signal: Any) -> None:
    async def run() -> None:
        state = A2APersistence(root, identity="program-1", max_tasks_per_principal=3)
        await state.open()
        admission = state.expect_admission("task-1")
        await state.save(_task("task-1", TaskState.TASK_STATE_SUBMITTED), _context("alice"))
        await admission
        await state.save(_task("task-1", TaskState.TASK_STATE_WORKING), _context("alice"))
        signal.send("admitted")
        await asyncio.Event().wait()

    asyncio.run(run())


def _hold_served_model(source: Path, state_dir: Path, marker: Path, signal: Any) -> None:
    class RecordingAdapter(_ServedAdapter):
        async def invoke(self, *, prompt: str, **kwargs: Any) -> str:
            marker.write_text("invoked", encoding="utf-8")
            return await super().invoke(prompt=prompt, **kwargs)

    async def run() -> None:
        adapter = RecordingAdapter()
        program = prepare_a2a_program(source, entry="remote", model="test:model", adapter=adapter)
        app = create_a2a_app(
            program,
            public_url="http://127.0.0.1:8000",
            auth=A2AServerAuth(scheme="bearer", secret="test-token"),
            state_dir=state_dir,
        )
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://127.0.0.1:8000",
                headers={"A2A-Version": "1.0", "Authorization": "Bearer test-token"},
                timeout=None,
            ) as client:
                asyncio.create_task(
                    client.post(
                        "/a2a",
                        json={
                            "jsonrpc": "2.0",
                            "id": "send",
                            "method": "SendMessage",
                            "params": {
                                "message": {
                                    "messageId": "message-1",
                                    "role": "ROLE_USER",
                                    "parts": [{"text": "slow:hello"}],
                                }
                            },
                        },
                    )
                )
                assert await asyncio.to_thread(adapter.slow_started.wait, 10)
                signal.send("model-running")
                await asyncio.Event().wait()

    asyncio.run(run())


def test_hard_kill_releases_lock_and_reconciles_admitted_task(tmp_path: Path) -> None:
    async def reopen() -> None:
        state = _test_state(tmp_path / "state", identity="program-1", max_tasks_per_principal=3)
        await state.open()
        try:
            task = await state.get("task-1", _context("alice"))
            assert task.status.state == TaskState.TASK_STATE_FAILED
            assert task.metadata["kedi_interruption"]["reason"] == "server_interrupted"
        finally:
            await state.close()

    context = multiprocessing.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    process = context.Process(target=_hold_admitted_task, args=(tmp_path / "state", send))
    process.start()
    send.close()
    try:
        assert receive.poll(15), f"child exited before admitting task: {process.exitcode}"
        assert receive.recv() == "admitted"

        async def locked() -> None:
            with pytest.raises(RuntimeError, match="already in use"):
                await _test_state(
                    tmp_path / "state", identity="program-1", max_tasks_per_principal=3
                ).open()

        asyncio.run(locked())
    finally:
        process.terminate()
        process.join(timeout=10)
        receive.close()
    assert not process.is_alive()
    asyncio.run(reopen())


def test_hard_kill_during_http_model_call_preserves_task_and_session(tmp_path: Path) -> None:
    source = tmp_path / "agent.kedi"
    source.write_text(
        "> profile: remote:\n    > instructions: Answer exactly.\n\n> export:\n    remote\n",
        encoding="utf-8",
    )
    state_dir = tmp_path / "state"
    marker = tmp_path / "model-call.txt"
    context = multiprocessing.get_context("spawn")
    receive, send = context.Pipe(duplex=False)
    process = context.Process(target=_hold_served_model, args=(source, state_dir, marker, send))
    process.start()
    send.close()
    try:
        assert receive.poll(20), f"child exited before model call: {process.exitcode}"
        assert receive.recv() == "model-running"
        assert marker.read_text(encoding="utf-8") == "invoked"
        with sqlite3.connect(state_dir / "state.sqlite3") as connection:
            rows = connection.execute("SELECT id, context_id FROM tasks").fetchall()
        assert len(rows) == 1
        task_id, context_id = rows[0]
    finally:
        process.kill()
        process.join(timeout=10)
        receive.close()
    assert not process.is_alive()

    async def reopen() -> None:
        adapter = _ServedAdapter()
        program = prepare_a2a_program(source, entry="remote", model="test:model", adapter=adapter)
        app = create_a2a_app(
            program,
            public_url="http://127.0.0.1:8000",
            auth=A2AServerAuth(scheme="bearer", secret="test-token"),
            state_dir=state_dir,
        )
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://127.0.0.1:8000",
                headers={"A2A-Version": "1.0", "Authorization": "Bearer test-token"},
                timeout=5,
            ) as client:
                old = await client.post(
                    "/a2a",
                    json={
                        "jsonrpc": "2.0",
                        "id": "get",
                        "method": "GetTask",
                        "params": {"id": task_id},
                    },
                )
                task = old.json()["result"]
                assert task["id"] == task_id
                assert task["contextId"] == context_id
                assert task["status"]["state"] == "TASK_STATE_FAILED"
                assert task["metadata"]["kedi_interruption"]["reason"] == "server_interrupted"
                assert adapter.calls == []
                sent = await client.post(
                    "/a2a",
                    json={
                        "jsonrpc": "2.0",
                        "id": "new",
                        "method": "SendMessage",
                        "params": {
                            "message": {
                                "messageId": "message-2",
                                "contextId": context_id,
                                "role": "ROLE_USER",
                                "parts": [{"text": "again"}],
                            }
                        },
                    },
                )
                assert sent.json()["result"]["task"]["status"]["state"] == ("TASK_STATE_COMPLETED")
                assert adapter.calls == ["again"]

    asyncio.run(reopen())


def test_program_identity_covers_imports_without_checkout_paths(tmp_path: Path) -> None:
    source = "> import: helper\n"
    fingerprints = []
    for name in ("first", "second"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "helper.kedi").write_text("= `1`\n", encoding="utf-8")
        root = directory / "agent.kedi"
        root.write_text(source, encoding="utf-8")
        fingerprints.append(_source_fingerprint(root, parse_program(source), source))
    assert fingerprints[0] == fingerprints[1]
    (tmp_path / "second" / "helper.kedi").write_text("= `2`\n", encoding="utf-8")
    second_root = tmp_path / "second" / "agent.kedi"
    assert _source_fingerprint(second_root, parse_program(source), source) != fingerprints[0]
    (tmp_path / "first" / "helper.kedi").write_text("> import: agent\n", encoding="utf-8")
    first_root = tmp_path / "first" / "agent.kedi"
    assert len(_source_fingerprint(first_root, parse_program(source), source)) == 64


def test_persistence_preflight_rejects_missing_identity_and_dependencies(
    tmp_path: Path, monkeypatch: Any
) -> None:
    source = tmp_path / "agent.kedi"
    source.write_text(
        "> profile: remote:\n    > instructions: Answer exactly.\n\n> export:\n    remote\n",
        encoding="utf-8",
    )
    program = prepare_a2a_program(
        source, entry="remote", model="test:model", adapter=_ServedAdapter()
    )
    program.source_fingerprint = ""
    with pytest.raises(ValueError, match="source fingerprint"):
        create_a2a_app(
            program,
            public_url="http://127.0.0.1:8000",
            auth=None,
            allow_unauthenticated_loopback=True,
            state_dir=tmp_path / "state",
        )
    program.source_fingerprint = "known"
    original_import = builtins.__import__

    def fail_persistence_import(name: str, *args: Any, **kwargs: Any) -> Any:
        level = args[3] if len(args) > 3 else kwargs.get("level", 0)
        if name == "persistence" and level == 1:
            raise ImportError("missing SQLite extra")
        return original_import(name, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", fail_persistence_import)
        with pytest.raises(RuntimeError, match="a2a-persistence"):
            create_a2a_app(
                program,
                public_url="http://127.0.0.1:8000",
                auth=None,
                allow_unauthenticated_loopback=True,
                state_dir=tmp_path / "state",
            )
    asyncio.run(program.aclose())


@pytest.mark.parametrize(
    "case",
    ["root_symlink", "public_root", "artifact_symlink", "database_symlink", "public_database"],
)
def test_state_directory_rejects_unsafe_paths(tmp_path: Path, case: str) -> None:
    if os.name != "posix" and case in {"public_root", "public_database"}:
        pytest.skip("POSIX mode-bit permissions are not enforced on Windows")
    root = tmp_path / "state"
    root.mkdir(mode=0o700)
    if case == "root_symlink":
        link = tmp_path / "link"
        try:
            link.symlink_to(root, target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("directory symlinks are unavailable")
        with pytest.raises(ValueError, match="symlink"):
            _test_state(link, identity="program-1", max_tasks_per_principal=2)
        return
    if case == "public_root":
        root.chmod(0o755)
    elif case == "artifact_symlink":
        try:
            (root / "artifacts").symlink_to(tmp_path, target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("directory symlinks are unavailable")
    elif case == "database_symlink":
        try:
            (root / "state.sqlite3").symlink_to(tmp_path / "target")
        except (OSError, NotImplementedError):
            pytest.skip("file symlinks are unavailable")
    else:
        (root / "state.sqlite3").write_text("", encoding="utf-8")
        (root / "state.sqlite3").chmod(0o644)
    with pytest.raises((ValueError, PermissionError), match="symlink|private|accessible"):
        asyncio.run(_test_state(root, identity="program-1", max_tasks_per_principal=2).open())


def test_state_store_rejects_soft_lock(tmp_path: Path, monkeypatch: Any) -> None:
    async def scenario() -> None:
        state = _test_state(tmp_path / "state", identity="program-1", max_tasks_per_principal=2)
        with monkeypatch.context() as patch:
            patch.setattr(persistence_module, "SoftFileLock", type(state._lock))
            with pytest.raises(RuntimeError, match="OS-backed file lock"):
                await state.open()
        await state.open()
        await state.close()

    asyncio.run(scenario())


def test_artifact_store_rejects_symlinked_child(tmp_path: Path) -> None:
    root = tmp_path / "state"
    root.mkdir(mode=0o700)
    artifacts = root / "artifacts"
    artifacts.mkdir()
    try:
        (artifacts / "untrusted").symlink_to(tmp_path, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are unavailable")
    with pytest.raises(RuntimeError, match="contains a symlink"):
        asyncio.run(_test_state(root, identity="program-1", max_tasks_per_principal=2).open())


class _ScriptedChatModel(BaseChatModel):
    reply: str
    seen: list[int] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "a2a-persistence-test"

    def bind_tools(self, _tools: Any, **_kwargs: Any) -> _ScriptedChatModel:
        return self

    def _generate(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        self.seen.append(len(messages))
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.reply))])


@pytest.mark.parametrize(
    "initial_state",
    [
        TaskState.TASK_STATE_WORKING,
        TaskState.TASK_STATE_INPUT_REQUIRED,
        TaskState.TASK_STATE_AUTH_REQUIRED,
    ],
)
def test_store_reconciles_interrupted_tasks_once_and_scopes_owner(
    tmp_path: Path, initial_state: TaskState
) -> None:
    async def scenario() -> None:
        root = tmp_path / "state"
        first = _test_state(root, identity="program-1", max_tasks_per_principal=3)
        await first.open()
        try:
            await first.save(_task("task-1", initial_state), _context("alice"))
            assert await first.get("task-1", _context("bob")) is None
            with pytest.raises(RuntimeError, match="already in use"):
                await _test_state(root, identity="program-1", max_tasks_per_principal=3).open()
        finally:
            await first.close()
        assert (root / "state.lock").exists()

        second = _test_state(root, identity="program-1", max_tasks_per_principal=3)
        await second.open()
        try:
            task = await second.get("task-1", _context("alice"))
            assert task.status.state == TaskState.TASK_STATE_FAILED
            assert task.metadata["kedi_interruption"]["reason"] == "server_interrupted"
            assert "External effects may have occurred" in task.status.message.parts[0].text
            timestamp = task.metadata["kedi_interruption"]["detected_at"]
        finally:
            await second.close()

        third = _test_state(root, identity="program-1", max_tasks_per_principal=3)
        await third.open()
        try:
            task = await third.get("task-1", _context("alice"))
            assert task.metadata["kedi_interruption"]["detected_at"] == timestamp
        finally:
            await third.close()

    asyncio.run(scenario())


def test_store_commits_task_and_session_together(tmp_path: Path) -> None:
    async def scenario() -> None:
        state = _test_state(tmp_path / "state", identity="program-1", max_tasks_per_principal=2)
        await state.open()
        try:
            owner = _context("alice")
            await state.save(_task("task-1", TaskState.TASK_STATE_SUBMITTED), owner)
            future = state.stage_completion(
                task_id="task-1",
                owner="alice",
                context_id="context-1",
                checkpoint={"version": 1, "turns": ["first"]},
            )
            await state.save(_task("task-1", TaskState.TASK_STATE_COMPLETED), owner)
            await future
            assert (await state.load_checkpoint("alice", "context-1"))["turns"] == ["first"]

            await state.save(_task("task-2", TaskState.TASK_STATE_SUBMITTED), owner)
            failed = state.stage_completion(
                task_id="task-2",
                owner="alice",
                context_id="context-1",
                checkpoint={"bad": object()},
            )
            with pytest.raises(Exception):
                await state.save(_task("task-2", TaskState.TASK_STATE_COMPLETED), owner)
            with pytest.raises(Exception):
                await failed
            assert (await state.get("task-2", owner)).status.state == TaskState.TASK_STATE_SUBMITTED
            assert (await state.load_checkpoint("alice", "context-1"))["turns"] == ["first"]
            with pytest.raises(RuntimeError, match="another principal"):
                await state.save(_task("task-1", TaskState.TASK_STATE_WORKING), _context("bob"))
        finally:
            await state.close()
        with pytest.raises(RuntimeError, match="incompatible"):
            await _test_state(
                tmp_path / "state", identity="other", max_tasks_per_principal=2
            ).open()

    asyncio.run(scenario())


def test_session_capacity_is_checked_at_admission(tmp_path: Path) -> None:
    async def scenario() -> None:
        state = _test_state(
            tmp_path / "state",
            identity="program-1",
            max_tasks_per_principal=3,
            max_sessions=1,
        )
        await state.open()
        try:
            owner = _context("alice")
            await state.save(_task("task-1", TaskState.TASK_STATE_SUBMITTED), owner)
            with pytest.raises(RuntimeError, match="session capacity"):
                await state.save(
                    _task("task-2", TaskState.TASK_STATE_SUBMITTED, context_id="context-2"),
                    owner,
                )
            assert await state.get("task-2", owner) is None
            await state.save(_task("task-1", TaskState.TASK_STATE_FAILED), owner)
            await state.save(
                _task("task-2", TaskState.TASK_STATE_SUBMITTED, context_id="context-2"),
                owner,
            )
        finally:
            await state.close()

    asyncio.run(scenario())


def test_task_retention_and_commit_guards(tmp_path: Path) -> None:
    async def scenario() -> None:
        state = _test_state(tmp_path / "state", identity="program-1", max_tasks_per_principal=2)
        await state.open()
        try:
            owner = _context("alice")
            with pytest.raises(RuntimeError, match="no prepared session checkpoint"):
                await state.save(_task("missing", TaskState.TASK_STATE_COMPLETED), owner)
            await state.save(_task("task-1", TaskState.TASK_STATE_FAILED), owner)
            await state.save(_task("task-2", TaskState.TASK_STATE_WORKING), owner)
            await state.save(_task("task-3", TaskState.TASK_STATE_SUBMITTED), owner)
            assert await state.get("task-1", owner) is None
            with pytest.raises(RuntimeError, match="retention capacity"):
                await state.save(_task("task-4", TaskState.TASK_STATE_WORKING), owner)
            with pytest.raises(RuntimeError, match="changed context"):
                await state.save(
                    _task("task-2", TaskState.TASK_STATE_WORKING, context_id="other"), owner
                )
            await state.delete("task-2", _context("bob"))
            assert await state.get("task-2", owner) is not None
            await state.delete("task-2", owner)
            await state.save(_task("task-4", TaskState.TASK_STATE_SUBMITTED), owner)

            pending = state.stage_completion(
                task_id="task-4",
                owner="alice",
                context_id="context-1",
                checkpoint={"version": 1},
            )
            with pytest.raises(RuntimeError, match="pending completion"):
                state.stage_completion(
                    task_id="task-4", owner="alice", context_id="context-1", checkpoint={}
                )
            with pytest.raises(RuntimeError, match="does not match"):
                await state.save(
                    _task("task-4", TaskState.TASK_STATE_COMPLETED, context_id="other"),
                    owner,
                )
            with pytest.raises(RuntimeError, match="does not match"):
                await pending
            failed_completion = state.stage_completion(
                task_id="task-4",
                owner="alice",
                context_id="context-1",
                checkpoint={"version": 1},
            )
            await state.save(_task("task-4", TaskState.TASK_STATE_FAILED), owner)
            with pytest.raises(RuntimeError, match="ended before checkpoint commit"):
                await failed_completion
            admission = state.expect_admission("new-task")
            with pytest.raises(RuntimeError, match="already being admitted"):
                state.expect_admission("new-task")
            state.abort_admission("new-task")
            assert admission.cancelled()
            completion = state.stage_completion(
                task_id="new-task", owner="alice", context_id="context-1", checkpoint={}
            )
            await state.abort_completion("new-task")
            assert completion.cancelled()
            abandoned_admission = state.expect_admission("abandoned")
            abandoned_completion = state.stage_completion(
                task_id="abandoned", owner="alice", context_id="context-1", checkpoint={}
            )
            already_finished = state.expect_admission("already-finished")
            already_finished.set_result(None)
            await state.close()
            with pytest.raises(RuntimeError, match="store closed"):
                await abandoned_admission
            with pytest.raises(RuntimeError, match="store closed"):
                await abandoned_completion
        finally:
            await state.close()

    asyncio.run(scenario())


def test_store_rejects_bad_protocol_and_unopened_access(tmp_path: Path) -> None:
    async def scenario() -> None:
        root = tmp_path / "state"
        state = _test_state(root, identity="program-1", max_tasks_per_principal=2)
        with pytest.raises(RuntimeError, match="not open"):
            await state.get("task-1", _context("alice"))
        with pytest.raises(RuntimeError, match="not open"):
            await state.load_checkpoint("alice", "context-1")
        await state.open()
        await state.open()
        try:
            await state.save(_task("task-1", TaskState.TASK_STATE_SUBMITTED), _context("alice"))
        finally:
            await state.close()
        with sqlite3.connect(root / "state.sqlite3") as database:
            database.execute(
                "UPDATE tasks SET protocol_version = ? WHERE id = ?", ("2.0", "task-1")
            )
        reopened = _test_state(root, identity="program-1", max_tasks_per_principal=2)
        with pytest.raises(RuntimeError, match="protocol version"):
            await reopened.open()

    asyncio.run(scenario())


@pytest.mark.parametrize("case", ["terminal_marked_active", "missing_task_row"])
def test_reconciliation_detects_inconsistent_execution_rows(tmp_path: Path, case: str) -> None:
    async def seed() -> None:
        state = _test_state(tmp_path / "state", identity="program-1", max_tasks_per_principal=2)
        await state.open()
        try:
            await state.save(_task("task-1", TaskState.TASK_STATE_FAILED), _context("alice"))
        finally:
            await state.close()

    asyncio.run(seed())
    with sqlite3.connect(tmp_path / "state" / "state.sqlite3") as database:
        if case == "terminal_marked_active":
            database.execute(
                "UPDATE kedi_a2a_executions SET disposition = ? WHERE task_id = ?",
                ("active", "task-1"),
            )
        else:
            database.execute("PRAGMA foreign_keys = OFF")
            database.execute(
                "UPDATE kedi_a2a_executions SET disposition = ? WHERE task_id = ?",
                ("active", "task-1"),
            )
            database.execute("DELETE FROM tasks WHERE id = ?", ("task-1",))

    async def reopen() -> None:
        state = _test_state(tmp_path / "state", identity="program-1", max_tasks_per_principal=2)
        if case == "missing_task_row":
            with pytest.raises(RuntimeError, match="without its task"):
                await state.open()
        else:
            await state.open()
            try:
                task = await state.get("task-1", _context("alice"))
                assert task.status.state == TaskState.TASK_STATE_FAILED
            finally:
                await state.close()

    asyncio.run(reopen())


def test_admission_failure_resolves_waiter(tmp_path: Path) -> None:
    async def scenario() -> None:
        state = _test_state(tmp_path / "state", identity="program-1", max_tasks_per_principal=2)
        await state.open()
        try:
            admission = state.expect_admission("task-1")
            with pytest.raises(RuntimeError, match="no prepared session checkpoint"):
                await state.save(_task("task-1", TaskState.TASK_STATE_COMPLETED), _context("alice"))
            with pytest.raises(RuntimeError, match="no prepared session checkpoint"):
                await admission
            assert await state.get("task-1", _context("alice")) is None
            canceled = state.expect_admission("task-2")
            canceled.cancel()
            with pytest.raises(RuntimeError, match="no prepared session checkpoint"):
                await state.save(_task("task-2", TaskState.TASK_STATE_COMPLETED), _context("alice"))
        finally:
            await state.close()

    asyncio.run(scenario())


def test_terminal_capacity_failure_preserves_submitted_task(tmp_path: Path) -> None:
    async def scenario() -> None:
        state = _test_state(tmp_path / "state", identity="program-1", max_tasks_per_principal=2)
        await state.open()
        try:
            owner = _context("alice")
            await state.save(_task("task-1", TaskState.TASK_STATE_SUBMITTED), owner)
            state.max_sessions = 0
            committed = state.stage_completion(
                task_id="task-1", owner="alice", context_id="context-1", checkpoint={}
            )
            with pytest.raises(RuntimeError, match="session capacity"):
                await state.save(_task("task-1", TaskState.TASK_STATE_COMPLETED), owner)
            with pytest.raises(RuntimeError, match="session capacity"):
                await committed
            assert (await state.get("task-1", owner)).status.state == (
                TaskState.TASK_STATE_SUBMITTED
            )
        finally:
            await state.close()

    asyncio.run(scenario())


def test_canceled_completion_waiters_do_not_break_store(tmp_path: Path) -> None:
    async def scenario() -> None:
        state = _test_state(tmp_path / "state", identity="program-1", max_tasks_per_principal=10)
        await state.open()
        try:
            owner = _context("alice")
            admission = state.expect_admission("task-1")
            admission.cancel()
            await state.save(_task("task-1", TaskState.TASK_STATE_SUBMITTED), owner)
            completed = state.stage_completion(
                task_id="task-1", owner="alice", context_id="context-1", checkpoint={}
            )
            completed.cancel()
            await state.save(_task("task-1", TaskState.TASK_STATE_COMPLETED), owner)
            assert (await state.get("task-1", owner)).status.state == (
                TaskState.TASK_STATE_COMPLETED
            )

            await state.save(_task("task-2", TaskState.TASK_STATE_SUBMITTED), owner)
            failed = state.stage_completion(
                task_id="task-2", owner="alice", context_id="context-1", checkpoint={}
            )
            failed.cancel()
            await state.save(_task("task-2", TaskState.TASK_STATE_FAILED), owner)
            assert (await state.get("task-2", owner)).status.state == TaskState.TASK_STATE_FAILED

            await state.save(_task("task-3", TaskState.TASK_STATE_SUBMITTED), owner)
            wrong = state.stage_completion(
                task_id="task-3", owner="alice", context_id="context-1", checkpoint={}
            )
            wrong.cancel()
            with pytest.raises(RuntimeError, match="does not match"):
                await state.save(
                    _task("task-3", TaskState.TASK_STATE_COMPLETED, context_id="other"),
                    owner,
                )
            already_done = state.expect_admission("task-4")
            already_done.set_result(None)
            state.abort_admission("task-4")
            assert already_done.done()
        finally:
            await state.close()

    asyncio.run(scenario())


def test_sqlite_connection_configures_durability() -> None:
    connection = sqlite3.connect(":memory:")
    try:
        persistence_module._configure_sqlite(connection, None)
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert connection.execute("PRAGMA synchronous").fetchone()[0] == 2
        assert connection.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
    finally:
        connection.close()


def test_artifact_and_native_conversation_checkpoint_restores(tmp_path: Path) -> None:
    artifacts = ArtifactManager(session_id="session-one", durable_root=tmp_path)
    conversation = ConversationState(artifacts=artifacts)
    conversation.record_exchange("question", "answer")
    handle = artifacts.artifactize("payload" * 30, policy=ArtifactPolicy(threshold_bytes=1))
    expected_turns = conversation.snapshot()
    saved_artifacts = artifacts.export_checkpoint()
    saved_conversation = conversation.export_checkpoint("pydantic")
    conversation.close()
    artifacts.close()

    restored_artifacts = ArtifactManager.from_checkpoint(saved_artifacts, durable_root=tmp_path)
    restored = ConversationState.from_checkpoint(
        saved_conversation, adapter_shortname="pydantic", artifacts=restored_artifacts
    )
    try:
        assert restored.snapshot() == expected_turns
        assert restored_artifacts.get_value(handle.ref.ref_id) == "payload" * 30
        assert restored_artifacts.export_checkpoint()["records"][0]["ref_id"] == handle.ref.ref_id
    finally:
        restored.close()
        restored_artifacts.close()


@pytest.mark.parametrize("adapter_kind", ["pydantic", "langchain"])
def test_native_tool_messages_keep_call_ids_and_order(tmp_path: Path, adapter_kind: str) -> None:
    artifacts = ArtifactManager(durable_root=tmp_path)
    conversation = ConversationState(artifacts=artifacts)
    if adapter_kind == "pydantic":
        messages: Any = [
            ModelResponse(
                parts=[ToolCallPart(tool_name="lookup", args={"id": 7}, tool_call_id="call-7")]
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(tool_name="lookup", content={"value": 9}, tool_call_id="call-7")
                ]
            ),
        ]
        key = "pydantic_messages"
    else:
        messages = [
            message_to_dict(
                AIMessage(
                    content="",
                    tool_calls=[{"name": "lookup", "args": {"id": 7}, "id": "call-7"}],
                )
            ),
            message_to_dict(ToolMessage(content="9", tool_call_id="call-7")),
        ]
        key = "langchain_messages"
    with conversation.adapter_turn(adapter_kind) as turn:
        turn.result_state[key] = messages
    saved = conversation.export_checkpoint(adapter_kind)
    artifact_checkpoint = artifacts.export_checkpoint()
    conversation.close()
    artifacts.close()

    restored_artifacts = ArtifactManager.from_checkpoint(artifact_checkpoint, durable_root=tmp_path)
    restored = ConversationState.from_checkpoint(
        saved, adapter_shortname=adapter_kind, artifacts=restored_artifacts
    )
    try:
        native = restored.adapter_state_snapshot(adapter_kind).state[key]
        assert len(native) == 2
        if adapter_kind == "pydantic":
            assert native[0]["parts"][0]["tool_call_id"] == "call-7"
            assert native[1]["parts"][0]["tool_call_id"] == "call-7"
        else:
            assert native[0]["data"]["tool_calls"][0]["id"] == "call-7"
            assert native[1]["data"]["tool_call_id"] == "call-7"
    finally:
        restored.close()
        restored_artifacts.close()


@pytest.mark.parametrize(
    ("value", "error"),
    [
        ({}, "must be a list"),
        ([None], "event is invalid"),
        ([{"sequence": 2}], "sequence is invalid"),
        (
            [{"sequence": 1, "kind": None, "data": {}, "created_at": "2026-01-01T00:00:00Z"}],
            "event is invalid",
        ),
        (
            [
                {
                    "sequence": 1,
                    "kind": "read",
                    "data": {},
                    "ref_id": 2,
                    "created_at": "2026-01-01T00:00:00Z",
                }
            ],
            "reference is invalid",
        ),
        ([{"sequence": 1, "kind": "read", "data": {}, "created_at": None}], "timestamp is invalid"),
        (
            [{"sequence": 1, "kind": "read", "data": {}, "created_at": "2026-01-01T00:00:00"}],
            "timezone",
        ),
    ],
)
def test_artifact_history_rejects_invalid_checkpoint(value: Any, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        ArtifactHistory.from_jsonable(value)


def test_conversation_checkpoint_validation(tmp_path: Path) -> None:
    artifacts = ArtifactManager(durable_root=tmp_path)
    conversation = ConversationState(artifacts=artifacts)
    valid = conversation.export_checkpoint("pydantic")
    mutations: list[tuple[dict[str, Any], str]] = [
        ({"session_id": "different"}, "session ID"),
        ({"cache_epoch": -1}, "cache epoch"),
        ({"lane_identity": ""}, "lane identity"),
        ({"turns": None}, "turns must be a list"),
        ({"turns": [{"role": "other", "content": "text"}]}, "turn is invalid"),
        ({"turns": [{"role": "user", "content": 5}]}, "content is invalid"),
        ({"state": None}, "state must be an object"),
    ]
    for patch, error in mutations:
        with pytest.raises(ValueError, match=error):
            ConversationState.from_checkpoint(
                {**valid, **patch}, adapter_shortname="pydantic", artifacts=artifacts
            )
    with conversation.adapter_turn("pydantic") as turn:
        with pytest.raises(RuntimeError, match="active conversation turn"):
            conversation.export_checkpoint("pydantic")
        turn.result_state["pydantic_messages"] = []
    with conversation.adapter_turn("langchain") as turn:
        turn.result_state["langchain_messages"] = []
    with pytest.raises(ValueError, match="another adapter lane"):
        conversation.export_checkpoint("pydantic")
    with pytest.raises(ValueError, match="does not support"):
        _checkpoint_lane_state("unsupported", {})
    with pytest.raises(ValueError, match="unsupported continuation fields"):
        _checkpoint_lane_state("pydantic", {"provider_handle": "socket"})
    with pytest.raises(ValueError, match="rewrite marker"):
        _checkpoint_lane_state("pydantic", {"history_processor_rewritten": "yes"})
    conversation.close()
    artifacts.close()


def test_conversation_lineage_round_trips(tmp_path: Path) -> None:
    artifacts = ArtifactManager(durable_root=tmp_path)
    conversation = ConversationState(artifacts=artifacts)
    with conversation.adapter_turn("pydantic") as turn:
        turn.result_state["history_processor_origins"] = HistoryLineage(entries=())
        turn.result_state["history_processor_rewritten"] = False
    saved = conversation.export_checkpoint("pydantic")
    assert saved["state"]["history_processor_origins"] == {"entries": []}
    restored = ConversationState.from_checkpoint(
        saved, adapter_shortname="pydantic", artifacts=artifacts
    )
    assert restored.adapter_state_snapshot("pydantic").state[
        "history_processor_origins"
    ] == HistoryLineage(entries=())
    restored.close()
    conversation.close()
    artifacts.close()


def test_uncommitted_artifacts_do_not_replace_committed_payloads(tmp_path: Path) -> None:
    first = ArtifactManager(session_id="session-one", durable_root=tmp_path)
    old = first.artifactize("committed" * 20, policy=ArtifactPolicy(threshold_bytes=1))
    checkpoint = first.export_checkpoint()
    first.close()

    attempted = ArtifactManager.from_checkpoint(checkpoint, durable_root=tmp_path)
    attempted.release(old.ref.ref_id)
    new = attempted.artifactize("uncommitted" * 20, policy=ArtifactPolicy(threshold_bytes=1))
    attempted.close()

    recovered = ArtifactManager.from_checkpoint(checkpoint, durable_root=tmp_path)
    try:
        assert recovered.get_value(old.ref.ref_id) == "committed" * 20
        with pytest.raises(ArtifactNotFoundError):
            recovered.get_value(new.ref.ref_id)
        assert not (tmp_path / "session-one" / f"{new.ref.ref_id}.payload").exists()
        assert recovered.prune_inactive_payloads(checkpoint) == 0
        recovered.release(old.ref.ref_id)
        released = recovered.export_checkpoint()
        payload = tmp_path / "session-one" / f"{old.ref.ref_id}.payload"
        assert payload.stat().st_size > 0
        assert recovered.prune_inactive_payloads(released) == 1
        assert payload.stat().st_size == 0
    finally:
        recovered.close()
    reopened = ArtifactManager.from_checkpoint(released, durable_root=tmp_path)
    try:
        with pytest.raises(ArtifactReleasedError):
            reopened.get_value(old.ref.ref_id)
    finally:
        reopened.close()


def test_corrupt_checkpoint_payload_blocks_restore(tmp_path: Path) -> None:
    manager = ArtifactManager(session_id="session-one", durable_root=tmp_path)
    handle = manager.artifactize("original" * 20, policy=ArtifactPolicy(threshold_bytes=1))
    checkpoint = manager.export_checkpoint()
    manager.close()
    (tmp_path / "session-one" / f"{handle.ref.ref_id}.payload").write_text("corrupted")
    with pytest.raises(ValueError, match="corrupt"):
        ArtifactManager.from_checkpoint(checkpoint, durable_root=tmp_path)


def test_artifact_checkpoint_preserves_expiry_and_multiple_references(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="durable store"):
        ArtifactManager().export_checkpoint()

    def old_clock() -> datetime:
        return datetime.now(timezone.utc) - timedelta(days=1)

    manager = ArtifactManager(durable_root=tmp_path, wall_clock=old_clock)
    first = manager.artifactize(
        "first" * 20, policy=ArtifactPolicy(threshold_bytes=1, ttl_seconds=1)
    )
    second = manager.artifactize(
        "second" * 20, policy=ArtifactPolicy(threshold_bytes=1, ttl_seconds=1)
    )
    checkpoint = manager.export_checkpoint()
    manager.close()
    restored = ArtifactManager.from_checkpoint(checkpoint, durable_root=tmp_path)
    try:
        with pytest.raises(ArtifactExpiredError):
            restored.get_value(first.ref.ref_id)
        with pytest.raises(ArtifactExpiredError):
            restored.get_value(second.ref.ref_id)
        assert restored.prune_inactive_payloads(restored.export_checkpoint()) == 2
        for ref_id in (first.ref.ref_id, second.ref.ref_id):
            assert (tmp_path / restored.session_id / f"{ref_id}.payload").stat().st_size == 0
    finally:
        restored.close()


def test_durable_store_reopens_released_record_without_deleting_payload(tmp_path: Path) -> None:
    first = ArtifactManager(session_id="session-one", durable_root=tmp_path)
    old = first.artifactize("old" * 20, policy=ArtifactPolicy(threshold_bytes=1))
    first.close()
    metadata_path = tmp_path / "session-one" / f"{old.ref.ref_id}.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["status"] = "released"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    second = ArtifactManager(session_id="session-one", durable_root=tmp_path)
    try:
        second.artifactize("new" * 20, policy=ArtifactPolicy(threshold_bytes=1))
        assert (tmp_path / "session-one" / f"{old.ref.ref_id}.payload").exists()
    finally:
        second.close()


@pytest.mark.parametrize(
    "mutation",
    [
        "not_object",
        "bad_session",
        "bad_history",
        "bad_records",
        "bad_reference",
        "missing_reference",
        "duplicate_reference",
        "bad_digest",
        "bad_status",
    ],
)
def test_artifact_checkpoint_rejects_invalid_manifest(tmp_path: Path, mutation: str) -> None:
    manager = ArtifactManager(durable_root=tmp_path)
    manager.artifactize("payload" * 20, policy=ArtifactPolicy(threshold_bytes=1))
    checkpoint: Any = copy.deepcopy(manager.export_checkpoint())
    manager.close()
    if mutation == "not_object":
        checkpoint = []
    elif mutation == "bad_session":
        checkpoint["session_id"] = ""
    elif mutation == "bad_history":
        checkpoint["history"] = {}
    elif mutation == "bad_records":
        checkpoint["records"] = None
    elif mutation == "bad_reference":
        checkpoint["records"] = [None]
    elif mutation == "missing_reference":
        checkpoint["records"][0]["ref_id"] = "missing"
    elif mutation == "duplicate_reference":
        checkpoint["records"].append(dict(checkpoint["records"][0]))
    elif mutation == "bad_digest":
        checkpoint["records"][0]["digest"] = "0" * 64
    elif mutation == "bad_status":
        checkpoint["records"][0]["status"] = "unknown"
    with pytest.raises(ValueError):
        ArtifactManager.from_checkpoint(checkpoint, durable_root=tmp_path)


def test_durable_storage_does_not_reject_inline_opaque_values(tmp_path: Path) -> None:
    manager = ArtifactManager(durable_root=tmp_path)
    value = object()
    try:
        assert (
            manager.artifactize(value, policy=ArtifactPolicy(threshold_bytes=10_000_000)) is value
        )
        assert (
            manager.admit_tool_result(value, policy=ArtifactPolicy(threshold_bytes=10_000_000))
            is value
        )
    finally:
        manager.close()


def test_corrupt_committed_artifact_blocks_service_startup(tmp_path: Path) -> None:
    async def scenario() -> None:
        root = tmp_path / "state"
        root.mkdir(mode=0o700)
        manager = ArtifactManager(session_id="session-one", durable_root=root / "artifacts")
        handle = manager.artifactize("original" * 20, policy=ArtifactPolicy(threshold_bytes=1))
        checkpoint = {
            "version": 1,
            "artifacts": manager.export_checkpoint(),
            "conversation": None,
        }
        manager.close()

        state = _test_state(root, identity="program-1", max_tasks_per_principal=2)
        await state.open()
        try:
            owner = _context("alice")
            await state.save(_task("task-1", TaskState.TASK_STATE_SUBMITTED), owner)
            commit = state.stage_completion(
                task_id="task-1",
                owner="alice",
                context_id="context-1",
                checkpoint=checkpoint,
            )
            await state.save(_task("task-1", TaskState.TASK_STATE_COMPLETED), owner)
            await commit
            await state.save(_task("task-2", TaskState.TASK_STATE_WORKING), owner)
        finally:
            await state.close()
        (root / "artifacts" / "session-one" / f"{handle.ref.ref_id}.payload").write_text(
            "corrupted", encoding="utf-8"
        )
        with pytest.raises(ValueError, match="corrupt"):
            await _test_state(root, identity="program-1", max_tasks_per_principal=2).open()
        with sqlite3.connect(root / "state.sqlite3") as database:
            stored = database.execute(
                "SELECT status FROM tasks WHERE id = ?", ("task-2",)
            ).fetchone()
        assert json.loads(stored[0])["state"] == "TASK_STATE_WORKING"

    asyncio.run(scenario())


def test_corrupt_session_metadata_blocks_service_startup(tmp_path: Path) -> None:
    async def seed(root: Path, checkpoint: dict[str, Any]) -> None:
        state = _test_state(root, identity="program-1", max_tasks_per_principal=2)
        await state.open()
        try:
            owner = _context("alice")
            await state.save(_task("task-1", TaskState.TASK_STATE_SUBMITTED), owner)
            committed = state.stage_completion(
                task_id="task-1", owner="alice", context_id="context-1", checkpoint=checkpoint
            )
            await state.save(_task("task-1", TaskState.TASK_STATE_COMPLETED), owner)
            await committed
        finally:
            await state.close()

    root = tmp_path / "state"
    root.mkdir(mode=0o700)
    artifacts = ArtifactManager(durable_root=root / "artifacts")
    checkpoint = {
        "version": 1,
        "artifacts": artifacts.export_checkpoint(),
        "conversation": None,
    }
    artifacts.close()
    asyncio.run(seed(root, checkpoint))
    mutations: list[tuple[dict[str, Any], str]] = [
        ({"version": 2}, "unsupported session checkpoint"),
        ({"conversation": {}}, "incompatible conversation"),
        ({"artifacts": None}, "invalid artifact checkpoint"),
        ({"artifacts": {"session_id": None}}, "invalid artifact session"),
    ]
    for patch, error in mutations:
        with sqlite3.connect(root / "state.sqlite3") as database:
            database.execute(
                "UPDATE kedi_a2a_sessions SET checkpoint = ?",
                (json.dumps({**checkpoint, **patch}),),
            )
        with pytest.raises(RuntimeError, match=error):
            asyncio.run(_test_state(root, identity="program-1", max_tasks_per_principal=2).open())


def test_restart_removes_only_uncommitted_artifact_files(tmp_path: Path) -> None:
    async def scenario() -> None:
        root = tmp_path / "state"
        state = _test_state(root, identity="program-1", max_tasks_per_principal=2)
        await state.open()
        try:
            manager = ArtifactManager(durable_root=state.artifact_root)
            handle = manager.artifactize("abandoned" * 20, policy=ArtifactPolicy(threshold_bytes=1))
            orphan = state.artifact_root / manager.session_id
            assert (orphan / f"{handle.ref.ref_id}.payload").exists()
            manager.close()
            user_directory = state.artifact_root / "user-notes"
            user_directory.mkdir()
            (user_directory / "note.txt").write_text("retain", encoding="utf-8")
            unfamiliar = state.artifact_root / f"session_{'x' * 24}"
            unfamiliar.mkdir()
            (unfamiliar / "note.txt").write_text("retain", encoding="utf-8")
        finally:
            await state.close()
        reopened = _test_state(root, identity="program-1", max_tasks_per_principal=2)
        await reopened.open()
        try:
            assert not orphan.exists()
            assert (user_directory / "note.txt").read_text(encoding="utf-8") == "retain"
            assert (unfamiliar / "note.txt").read_text(encoding="utf-8") == "retain"
        finally:
            await reopened.close()

    asyncio.run(scenario())


def test_a2a_http_result_survives_restart(tmp_path: Path) -> None:
    async def request(
        client: httpx.AsyncClient, method: str, params: dict[str, Any]
    ) -> dict[str, Any]:
        response = await client.post(
            "/a2a",
            json={"jsonrpc": "2.0", "id": "request-1", "method": method, "params": params},
        )
        assert response.status_code == 200
        body = response.json()
        assert "error" not in body, body
        return body["result"]

    async def serve_once(source: Path, state_dir: Path, prompt: str, context_id: str | None):
        adapter = _ServedAdapter()
        program = prepare_a2a_program(source, entry="remote", model="test:model", adapter=adapter)
        app = create_a2a_app(
            program,
            public_url="http://127.0.0.1:8000",
            auth=None,
            allow_unauthenticated_loopback=True,
            state_dir=state_dir,
        )
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://127.0.0.1:8000",
                headers={"A2A-Version": "1.0"},
                timeout=10,
            ) as client:
                first = await request(
                    client,
                    "SendMessage",
                    {
                        "message": {
                            "messageId": f"message-{prompt}",
                            "role": "ROLE_USER",
                            "parts": [{"text": prompt}],
                            **({"contextId": context_id} if context_id else {}),
                        }
                    },
                )
                task = first["task"]
                assert task["status"]["state"] == "TASK_STATE_COMPLETED"
                return task, adapter.calls

    async def scenario() -> None:
        source = tmp_path / "agent.kedi"
        source.write_text(
            "> profile: remote:\n    > instructions: Answer exactly.\n\n> export:\n    remote\n",
            encoding="utf-8",
        )
        state_dir = tmp_path / "state"
        first, calls = await serve_once(source, state_dir, "hello", None)
        assert calls == ["hello"]

        adapter = _ServedAdapter()
        program = prepare_a2a_program(source, entry="remote", model="test:model", adapter=adapter)
        app = create_a2a_app(
            program,
            public_url="http://127.0.0.1:8000",
            auth=None,
            allow_unauthenticated_loopback=True,
            state_dir=state_dir,
        )
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://127.0.0.1:8000",
                headers={"A2A-Version": "1.0"},
                timeout=10,
            ) as client:
                retained = await request(client, "GetTask", {"id": first["id"]})
                assert retained["artifacts"][0]["parts"][0]["text"] == "remote:hello"
                assert adapter.calls == []
                listing = await request(client, "ListTasks", {})
                assert any(task["id"] == first["id"] for task in listing["tasks"])
                followup = await request(
                    client,
                    "SendMessage",
                    {
                        "message": {
                            "messageId": "message-again",
                            "contextId": first["contextId"],
                            "role": "ROLE_USER",
                            "parts": [{"text": "again"}],
                        }
                    },
                )
                assert followup["task"]["contextId"] == first["contextId"]
                assert adapter.calls == ["again"]

    asyncio.run(scenario())


def test_bearer_rotation_does_not_transfer_stored_tasks(tmp_path: Path) -> None:
    async def scenario() -> None:
        source = tmp_path / "agent.kedi"
        source.write_text(
            "> profile: remote:\n    > instructions: Answer exactly.\n\n> export:\n    remote\n",
            encoding="utf-8",
        )
        state_dir = tmp_path / "state"
        task_id = None
        for token in ("token-a", "token-b", "token-a"):
            program = prepare_a2a_program(
                source, entry="remote", model="test:model", adapter=_ServedAdapter()
            )
            app = create_a2a_app(
                program,
                public_url="http://127.0.0.1:8000",
                auth=A2AServerAuth(scheme="bearer", secret=token),
                state_dir=state_dir,
            )
            async with app.router.lifespan_context(app):
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app),
                    base_url="http://127.0.0.1:8000",
                    headers={"A2A-Version": "1.0", "Authorization": f"Bearer {token}"},
                    timeout=10,
                ) as client:
                    if task_id is None:
                        response = await client.post(
                            "/a2a",
                            json={
                                "jsonrpc": "2.0",
                                "id": "send",
                                "method": "SendMessage",
                                "params": {
                                    "message": {
                                        "messageId": "message-1",
                                        "role": "ROLE_USER",
                                        "parts": [{"text": "hello"}],
                                    }
                                },
                            },
                        )
                        task_id = response.json()["result"]["task"]["id"]
                    else:
                        response = await client.post(
                            "/a2a",
                            json={
                                "jsonrpc": "2.0",
                                "id": "get",
                                "method": "GetTask",
                                "params": {"id": task_id},
                            },
                        )
                        if token == "token-b":
                            assert "error" in response.json()
                        else:
                            assert response.json()["result"]["id"] == task_id

    asyncio.run(scenario())


@pytest.mark.parametrize("adapter_kind", ["pydantic", "langchain"])
@pytest.mark.parametrize(
    ("auth", "auth_headers"),
    [
        pytest.param(None, {}, id="loopback"),
        pytest.param(
            A2AServerAuth(scheme="bearer", secret="test-token"),
            {"Authorization": "Bearer test-token"},
            id="bearer",
        ),
        pytest.param(
            A2AServerAuth(scheme="basic", username="operator", secret="test-password"),
            {"Authorization": "Basic " + base64.b64encode(b"operator:test-password").decode()},
            id="basic",
        ),
        pytest.param(
            A2AServerAuth(scheme="api_key", secret="test-key", header="X-Agent-Key"),
            {"X-Agent-Key": "test-key"},
            id="api-key",
        ),
    ],
)
def test_native_history_survives_restart(
    tmp_path: Path,
    adapter_kind: str,
    auth: A2AServerAuth | None,
    auth_headers: dict[str, str],
) -> None:
    async def scenario() -> None:
        source = tmp_path / "agent.kedi"
        source.write_text(
            "> profile: remote:\n"
            "    > instructions: Answer exactly.\n"
            "    > history:\n"
            "        enabled: true\n\n"
            "> export:\n"
            "    remote\n",
            encoding="utf-8",
        )
        context_id = None
        counts = []
        for turn in range(2):
            if adapter_kind == "pydantic":
                seen: list[int] = []

                def respond(messages: Any, _info: Any) -> ModelResponse:
                    seen.append(len(messages))
                    return ModelResponse(parts=[TextPart(f"answer-{turn}")])

                adapter = PydanticAdapter(FunctionModel(respond))
            else:
                model = _ScriptedChatModel(reply=f"answer-{turn}")
                adapter = LangChainAdapter(model)
                seen = model.seen
            program = prepare_a2a_program(
                source, entry="remote", model="test:model", adapter=adapter
            )
            app = create_a2a_app(
                program,
                public_url="http://127.0.0.1:8000",
                auth=auth,
                allow_unauthenticated_loopback=auth is None,
                state_dir=tmp_path / "state",
            )
            async with app.router.lifespan_context(app):
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app),
                    base_url="http://127.0.0.1:8000",
                    headers={"A2A-Version": "1.0", **auth_headers},
                    timeout=10,
                ) as client:
                    message = {
                        "messageId": f"message-{turn}",
                        "role": "ROLE_USER",
                        "parts": [{"text": f"prompt-{turn}"}],
                        **({"contextId": context_id} if context_id else {}),
                    }
                    if auth is not None:
                        rejected = await client.post(
                            "/a2a",
                            headers={
                                "Authorization": "Bearer invalid-test-credential",
                                "X-Agent-Key": "invalid-test-credential",
                            },
                            json={
                                "jsonrpc": "2.0",
                                "id": "rejected",
                                "method": "SendMessage",
                                "params": {"message": message},
                            },
                        )
                        assert rejected.status_code == 401
                        assert seen == []
                    response = await client.post(
                        "/a2a",
                        json={
                            "jsonrpc": "2.0",
                            "id": f"request-{turn}",
                            "method": "SendMessage",
                            "params": {"message": message},
                        },
                    )
                    body = response.json()
                    assert "error" not in body, body
                    task = body["result"]["task"]
                    assert task["status"]["state"] == "TASK_STATE_COMPLETED"
                    assert task["artifacts"][0]["parts"][0]["text"] == f"answer-{turn}"
                    context_id = task["contextId"]
                    counts.append(seen[0])
        assert counts[1] > counts[0]

    asyncio.run(scenario())


def test_parallel_requests_on_one_context_commit_in_order(tmp_path: Path) -> None:
    async def scenario() -> None:
        source = tmp_path / "agent.kedi"
        source.write_text(
            "> profile: remote:\n"
            "    > instructions: Answer exactly.\n"
            "    > history:\n"
            "        enabled: true\n\n"
            "> export:\n"
            "    remote\n",
            encoding="utf-8",
        )
        seen: list[int] = []

        def respond(messages: Any, _info: Any) -> ModelResponse:
            seen.append(len(messages))
            return ModelResponse(parts=[TextPart(f"count={len(messages)}")])

        program = prepare_a2a_program(
            source,
            entry="remote",
            model="test:model",
            adapter=PydanticAdapter(FunctionModel(respond)),
        )
        app = create_a2a_app(
            program,
            public_url="http://127.0.0.1:8000",
            auth=None,
            allow_unauthenticated_loopback=True,
            state_dir=tmp_path / "state",
        )
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://127.0.0.1:8000",
                headers={"A2A-Version": "1.0"},
                timeout=10,
            ) as client:

                async def send(index: int, context_id: str | None = None) -> dict[str, Any]:
                    message = {
                        "messageId": f"message-{index}",
                        "role": "ROLE_USER",
                        "parts": [{"text": f"prompt-{index}"}],
                        **({"contextId": context_id} if context_id else {}),
                    }
                    response = await client.post(
                        "/a2a",
                        json={
                            "jsonrpc": "2.0",
                            "id": f"request-{index}",
                            "method": "SendMessage",
                            "params": {"message": message},
                        },
                    )
                    body = response.json()
                    assert "error" not in body, body
                    return body["result"]["task"]

                first = await send(0)
                second, third = await asyncio.gather(
                    send(1, first["contextId"]), send(2, first["contextId"])
                )
                assert second["status"]["state"] == "TASK_STATE_COMPLETED"
                assert third["status"]["state"] == "TASK_STATE_COMPLETED"
                assert seen == [1, 3, 5]

    asyncio.run(scenario())


def test_failed_checkpoint_does_not_enter_next_turn(tmp_path: Path, monkeypatch: Any) -> None:
    async def scenario() -> None:
        source = tmp_path / "agent.kedi"
        source.write_text(
            "> profile: remote:\n"
            "    > instructions: Answer exactly.\n"
            "    > history:\n"
            "        enabled: true\n\n"
            "> export:\n"
            "    remote\n",
            encoding="utf-8",
        )
        seen: list[int] = []

        def respond(messages: Any, _info: Any) -> ModelResponse:
            seen.append(len(messages))
            return ModelResponse(parts=[TextPart("answer")])

        program = prepare_a2a_program(
            source,
            entry="remote",
            model="test:model",
            adapter=PydanticAdapter(FunctionModel(respond)),
        )
        app = create_a2a_app(
            program,
            public_url="http://127.0.0.1:8000",
            auth=None,
            allow_unauthenticated_loopback=True,
            state_dir=tmp_path / "state",
        )
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://127.0.0.1:8000",
                headers={"A2A-Version": "1.0"},
                timeout=10,
            ) as client:
                context_id = None
                states = []
                for index in range(3):
                    if index == 1:
                        monkeypatch.setattr(a2a_runtime, "DEFAULT_A2A_MAX_CHECKPOINT_BYTES", 1)
                    if index == 2:
                        monkeypatch.setattr(
                            a2a_runtime,
                            "DEFAULT_A2A_MAX_CHECKPOINT_BYTES",
                            64 * 1024 * 1024,
                        )
                    message = {
                        "messageId": f"message-{index}",
                        "role": "ROLE_USER",
                        "parts": [{"text": f"prompt-{index}"}],
                        **({"contextId": context_id} if context_id else {}),
                    }
                    response = await client.post(
                        "/a2a",
                        json={
                            "jsonrpc": "2.0",
                            "id": f"request-{index}",
                            "method": "SendMessage",
                            "params": {"message": message},
                        },
                    )
                    body = response.json()
                    assert "error" not in body, body
                    task = body["result"]["task"]
                    context_id = task["contextId"]
                    states.append(task["status"]["state"])
                assert states == [
                    "TASK_STATE_COMPLETED",
                    "TASK_STATE_FAILED",
                    "TASK_STATE_COMPLETED",
                ]
                assert seen == [1, 3, 3]

    asyncio.run(scenario())


def test_terminal_store_failure_cannot_publish_completed_task(tmp_path: Path) -> None:
    async def scenario() -> None:
        source = tmp_path / "agent.kedi"
        source.write_text(
            "> profile: remote:\n    > instructions: Answer exactly.\n\n> export:\n    remote\n",
            encoding="utf-8",
        )
        adapter = _ServedAdapter()
        program = prepare_a2a_program(source, entry="remote", model="test:model", adapter=adapter)
        app = create_a2a_app(
            program,
            public_url="http://127.0.0.1:8000",
            auth=None,
            allow_unauthenticated_loopback=True,
            state_dir=tmp_path / "state",
        )
        async with app.router.lifespan_context(app):
            assert program._persistence is not None
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://127.0.0.1:8000",
                headers={"A2A-Version": "1.0"},
                timeout=5,
            ) as client:
                program._persistence.max_sessions = 0
                rejected = await client.post(
                    "/a2a",
                    json={
                        "jsonrpc": "2.0",
                        "id": "reject",
                        "method": "SendMessage",
                        "params": {
                            "message": {
                                "messageId": "message-rejected",
                                "role": "ROLE_USER",
                                "parts": [{"text": "rejected"}],
                            }
                        },
                    },
                )
                assert "session capacity" in str(rejected.json())
                assert adapter.calls == []
                program._persistence.max_sessions = 1
                sent = asyncio.create_task(
                    client.post(
                        "/a2a",
                        json={
                            "jsonrpc": "2.0",
                            "id": "send",
                            "method": "SendMessage",
                            "params": {
                                "message": {
                                    "messageId": "message-1",
                                    "role": "ROLE_USER",
                                    "parts": [{"text": "slow:hello"}],
                                }
                            },
                        },
                    )
                )
                assert await asyncio.to_thread(adapter.slow_started.wait, 2)
                program._persistence.max_sessions = 0
                adapter.release_slow.set()
                response = await asyncio.wait_for(sent, timeout=5)
                body = response.json()
                assert "session capacity" in str(body), body
                assert "error" in body or body["result"]["task"]["status"]["state"] != (
                    "TASK_STATE_COMPLETED"
                )
                assert adapter.calls == ["slow:hello"]

    asyncio.run(scenario())


def test_cancelled_task_stays_cancelled_after_restart(tmp_path: Path) -> None:
    async def scenario() -> None:
        source = tmp_path / "agent.kedi"
        source.write_text(
            "> profile: remote:\n    > instructions: Answer exactly.\n\n> export:\n    remote\n",
            encoding="utf-8",
        )
        adapter = _ServedAdapter()
        program = prepare_a2a_program(source, entry="remote", model="test:model", adapter=adapter)
        app = create_a2a_app(
            program,
            public_url="http://127.0.0.1:8000",
            auth=None,
            allow_unauthenticated_loopback=True,
            state_dir=tmp_path / "state",
        )
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://127.0.0.1:8000",
                headers={"A2A-Version": "1.0"},
                timeout=10,
            ) as client:
                sent = (
                    await client.post(
                        "/a2a",
                        json={
                            "jsonrpc": "2.0",
                            "id": "send",
                            "method": "SendMessage",
                            "params": {
                                "message": {
                                    "messageId": "message-slow",
                                    "role": "ROLE_USER",
                                    "parts": [{"text": "slow:wait"}],
                                },
                                "configuration": {"returnImmediately": True},
                            },
                        },
                    )
                ).json()
                assert "error" not in sent, sent
                task_id = sent["result"]["task"]["id"]
                assert await asyncio.to_thread(adapter.slow_started.wait, 2)
                cancelled = (
                    await client.post(
                        "/a2a",
                        json={
                            "jsonrpc": "2.0",
                            "id": "cancel",
                            "method": "CancelTask",
                            "params": {"id": task_id},
                        },
                    )
                ).json()
                assert "error" not in cancelled, cancelled
                assert cancelled["result"]["status"]["state"] == "TASK_STATE_CANCELED"
                adapter.release_slow.set()

        # Reopen through the public app so its complete service identity is reused.
        second = prepare_a2a_program(
            source, entry="remote", model="test:model", adapter=_ServedAdapter()
        )
        second_app = create_a2a_app(
            second,
            public_url="http://127.0.0.1:8000",
            auth=None,
            allow_unauthenticated_loopback=True,
            state_dir=tmp_path / "state",
        )
        async with second_app.router.lifespan_context(second_app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=second_app),
                base_url="http://127.0.0.1:8000",
                headers={"A2A-Version": "1.0"},
                timeout=10,
            ) as client:
                result = (
                    await client.post(
                        "/a2a",
                        json={
                            "jsonrpc": "2.0",
                            "id": "get",
                            "method": "GetTask",
                            "params": {"id": task_id},
                        },
                    )
                ).json()
                assert "error" not in result, result
                assert result["result"]["status"]["state"] == "TASK_STATE_CANCELED"

    asyncio.run(scenario())
