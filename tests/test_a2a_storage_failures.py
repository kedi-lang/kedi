import asyncio
import sqlite3

import httpx
import pytest
from a2a.types import TaskState
from sqlalchemy import event
from sqlalchemy.exc import OperationalError

from kedi.a2a.server import create_a2a_app
from tests.a2a_test_support import disk_full_hook, served_program
from tests.test_a2a_persistence import _context, _task, _test_state


@pytest.mark.asyncio
async def test_failed_checkpoint_restores_sdk_mutable_task(tmp_path):
    state = _test_state(tmp_path / "state", identity="test", max_tasks_per_principal=2)
    await state.open()
    engine = state._engine.sync_engine
    hook = disk_full_hook(tmp_path / "fault.json")
    owner = _context("operator")
    pending = None
    try:
        task = _task("task-1", TaskState.TASK_STATE_WORKING)
        await state.save(task, owner)
        committed = task.SerializeToString()
        pending = state.stage_completion(
            task_id=task.id, owner="operator", context_id=task.context_id, checkpoint={}
        )
        task.status.state = TaskState.TASK_STATE_COMPLETED
        task.artifacts.add(artifact_id="uncommitted", parts=[])
        event.listen(engine, "before_cursor_execute", hook)
        with pytest.raises(OperationalError, match="database or disk is full"):
            await state.save(task, owner)
        with pytest.raises(OperationalError, match="database or disk is full"):
            await pending
        assert task.SerializeToString() == committed
        assert await state.load_checkpoint("operator", task.context_id) is None
        assert (await state.get(task.id, owner)).status.state == TaskState.TASK_STATE_WORKING
        # The SDK's consumer may now persist a failure, without replaying the agent.
        task.status.state = TaskState.TASK_STATE_FAILED
        await state.save(task, owner)
        assert (await state.get(task.id, owner)).status.state == TaskState.TASK_STATE_FAILED
    finally:
        if pending is not None and pending.done() and not pending.cancelled():
            pending.exception()
        event.remove(engine, "before_cursor_execute", hook)
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["pydantic", "langchain"])
async def test_sdk_marks_checkpoint_fault_failed_without_poll_timeout(tmp_path, backend, caplog):
    program = served_program(tmp_path, backend)
    app = create_a2a_app(
        program,
        public_url="http://127.0.0.1:8000",
        auth=None,
        allow_unauthenticated_loopback=True,
        state_dir=tmp_path / "state",
    )
    async with app.router.lifespan_context(app):
        state = program._persistence
        hook = disk_full_hook(tmp_path / "fault.json")
        event.listen(state._engine.sync_engine, "before_cursor_execute", hook)
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://127.0.0.1:8000",
                headers={"A2A-Version": "1.0"},
            ) as client:
                sent = await client.post(
                    "/a2a",
                    json={
                        "jsonrpc": "2.0",
                        "id": "send",
                        "method": "SendMessage",
                        "params": {
                            "message": {
                                "messageId": "first",
                                "role": "ROLE_USER",
                                "parts": [{"text": "Return a receipt."}],
                            },
                            "configuration": {"returnImmediately": True},
                        },
                    },
                )
                task_id = sent.json()["result"]["task"]["id"]
                async with asyncio.timeout(5):
                    while True:
                        response = await client.post(
                            "/a2a",
                            json={
                                "jsonrpc": "2.0",
                                "id": "poll",
                                "method": "GetTask",
                                "params": {"id": task_id},
                            },
                        )
                        task = response.json()["result"]
                        if task["status"]["state"] == "TASK_STATE_FAILED":
                            break
                        assert task["status"]["state"] != "TASK_STATE_COMPLETED"
                        await asyncio.sleep(0.01)
                assert (tmp_path / "fault.json").exists()
        finally:
            event.remove(state._engine.sync_engine, "before_cursor_execute", hook)
    assert "already in a terminal state" not in caplog.text


@pytest.mark.asyncio
async def test_persistent_write_failure_never_exposes_uncommitted_completion(tmp_path):
    root = tmp_path / "state"
    state = _test_state(root, identity="test", max_tasks_per_principal=2)
    await state.open()
    engine = state._engine.sync_engine
    owner = _context("operator")
    task = _task("task-1", TaskState.TASK_STATE_WORKING)
    await state.save(task, owner)

    def reject_writes(connection, cursor, statement, parameters, context, executemany):
        if statement.startswith(("INSERT", "UPDATE")):
            raise sqlite3.OperationalError("database or disk is full")

    pending = state.stage_completion(
        task_id=task.id, owner="operator", context_id=task.context_id, checkpoint={}
    )
    event.listen(engine, "before_cursor_execute", reject_writes)
    try:
        for terminal in (TaskState.TASK_STATE_COMPLETED, TaskState.TASK_STATE_FAILED):
            task.status.state = terminal
            with pytest.raises(OperationalError, match="database or disk is full"):
                await state.save(task, owner)
            assert task.status.state == TaskState.TASK_STATE_WORKING
        with pytest.raises(OperationalError, match="database or disk is full"):
            await pending
        assert await state.load_checkpoint("operator", task.context_id) is None
    finally:
        if pending.done() and not pending.cancelled():
            pending.exception()
        event.remove(engine, "before_cursor_execute", reject_writes)
        await state.close()
    restarted = _test_state(root, identity="test", max_tasks_per_principal=2)
    await restarted.open()
    try:
        restored = await restarted.get(task.id, owner)
        assert restored.status.state == TaskState.TASK_STATE_FAILED
        assert restored.metadata["kedi_interruption"]["reason"] == "server_interrupted"
    finally:
        await restarted.close()
