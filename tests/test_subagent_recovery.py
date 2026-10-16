"""Recover persisted children across abrupt process death without implicit replay."""

from __future__ import annotations

import asyncio
import json
import multiprocessing
from multiprocessing.connection import Connection
from pathlib import Path

import pytest
from pydantic_ai.models.test import TestModel

from kedi.agent_adapter.adapters.pydantic import PydanticAdapter
from kedi.agent_profile.compile import build_capture_env
from kedi.agent_profile.evaluation import eval_profile_definition
from kedi.errors import KediExecutionError
from kedi.lang import compile_program, parse_program

SOURCE = """
> profile: worker:
    > adapter: pydantic
    > instructions: Complete the supplied task.

> profile: manager:
    > adapter: pydantic
    > subagent: worker
"""


class RecoveryAdapter(PydanticAdapter):
    def __init__(self) -> None:
        super().__init__(TestModel())
        self.allowed = False
        self.calls: list[str] = []
        self.started = asyncio.Event()

    async def invoke(self, *, prompt: str, **kwargs: object) -> str:
        assert self.allowed, "Restoring state must not invoke a model"
        self.calls.append(prompt)
        if prompt == "Interrupted":
            self.started.set()
            await asyncio.Event().wait()
        return f"Completed: {prompt}"


def recovery_process(connection: Connection, state_path: str, mode: str) -> None:
    async def scenario() -> None:
        adapter = RecoveryAdapter()
        runtime = compile_program(
            parse_program(SOURCE), adapter=adapter, subagent_state_path=Path(state_path)
        )
        profile = eval_profile_definition(
            runtime, runtime._profile_registry["manager"], build_capture_env(runtime)
        )
        tools = {tool.name: tool for tool in runtime._resolve_agent_tool_specs(profile, adapter)}
        if mode == "start":
            adapter.allowed = True
            completed = await tools["delegate_task"].fn(subagent="worker", task="Initial")
            interrupted = await tools["delegate_task"].fn(
                subagent="worker", task="Interrupted", background=True
            )
            await adapter.started.wait()
            payload = json.loads(Path(state_path).read_text())
            pending = next(run for run in payload["runs"] if run["run_id"] == interrupted["run_id"])
            assert pending["status"] in {"pending", "running"}
            connection.send((completed["run_id"], interrupted["run_id"]))
            await asyncio.Event().wait()
        else:
            completed_id, interrupted_id = connection.recv()
            completed = await runtime._wait_subagent(completed_id)
            assert completed.task_summary == "Completed: Initial"
            with pytest.raises(KediExecutionError, match="interrupted by a process restart"):
                await runtime._wait_subagent(interrupted_id)
            interrupted_run = runtime._subagent_coordinator.runs.get(interrupted_id)
            assert interrupted_run.status == "failed"
            assert interrupted_run.error_type == "InterruptedError"
            assert not adapter.calls
            adapter.allowed = True
            continued = await tools["continue_subagent"].fn(run_id=completed_id, task="Follow-up")
            assert continued["run_id"] != completed_id
            original = runtime._subagent_coordinator.runs.get(completed_id)
            successor = runtime._subagent_coordinator.runs.get(continued["run_id"])
            assert successor.conversation_id == original.conversation_id
            assert continued["task_summary"] == "Completed: Follow-up"
            assert adapter.calls == ["Follow-up"]
            await runtime.aclose()
            connection.send("restored")

    try:
        asyncio.run(asyncio.wait_for(scenario(), 30))
    finally:
        connection.close()


def test_subagent_recovery_after_process_death(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    state_path = tmp_path / "subagents.json"
    parent, child = context.Pipe()
    process = context.Process(target=recovery_process, args=(child, str(state_path), "start"))
    process.start()
    child.close()
    try:
        assert parent.poll(30), "Child did not persist its running state"
        run_ids = parent.recv()
        process.kill()
        process.join(10)
        assert not process.is_alive()
        assert process.exitcode != 0
    finally:
        if process.is_alive():
            process.kill()
            process.join(10)
        parent.close()
    assert isinstance(json.loads(state_path.read_text()), dict)

    parent, child = context.Pipe()
    process = context.Process(target=recovery_process, args=(child, str(state_path), "restore"))
    process.start()
    child.close()
    try:
        parent.send(run_ids)
        assert parent.poll(30), "Restored runtime did not finish"
        assert parent.recv() == "restored"
        process.join(10)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.kill()
            process.join(10)
        parent.close()
