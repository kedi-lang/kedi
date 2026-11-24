from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import Any

import pytest

from kedi.agent_adapter.subagents import coordinator as coordinator_module
from kedi.agent_adapter.subagents.attempts import delegation_scope
from tests.test_subagents import _manager_tools


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", ["delegate_task", "continue_subagent"])
async def test_foreground_cancellation_drains_child_cleanup_even_when_cancelled_again(
    monkeypatch: pytest.MonkeyPatch, tool_name: str
) -> None:
    adapter, runtime, tools = _manager_tools()
    started, cleaning, release, stopped = (asyncio.Event() for _ in range(4))
    arguments: dict[str, Any] = {"subagent": "worker", "task": "Inspect"}
    if tool_name == "continue_subagent":
        first = await tools["delegate_task"].fn(**arguments)
        arguments = {"run_id": first["run_id"], "task": "Follow up"}

    async def block(**_kwargs: Any) -> str:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await release.wait()
            stopped.set()
        return "unreachable"

    monkeypatch.setattr(adapter, "invoke", block)
    with delegation_scope("recover") as ledger:
        parent = asyncio.create_task(tools[tool_name].fn(**arguments))
        try:
            await asyncio.wait_for(started.wait(), 2)
            parent.cancel()
            await asyncio.wait_for(cleaning.wait(), 0.5)
            parent.cancel()
            await asyncio.sleep(0)
            assert not parent.done()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(parent, 2)
            assert stopped.is_set()
            assert runtime._subagent_coordinator.active_runs == 0
            assert ledger.snapshot()[0].status == "cancelled"
        finally:
            release.set()
            parent.cancel()
            await asyncio.gather(parent, return_exceptions=True)
            await runtime.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", ["delegate_task", "continue_subagent"])
async def test_unconfirmed_foreground_stop_remains_pending_after_the_cleanup_deadline(
    monkeypatch: pytest.MonkeyPatch, tool_name: str
) -> None:
    adapter, runtime, tools = _manager_tools()
    started, release = asyncio.Event(), asyncio.Event()
    arguments: dict[str, Any] = {"subagent": "worker", "task": "Inspect"}
    if tool_name == "continue_subagent":
        first = await tools["delegate_task"].fn(**arguments)
        arguments = {"run_id": first["run_id"], "task": "Follow up"}
    monkeypatch.setattr(coordinator_module, "INTERRUPT_CLEANUP_SECONDS", 0.02)

    async def stubborn(**_kwargs: Any) -> str:
        started.set()
        while not release.is_set():
            with suppress(asyncio.CancelledError):
                await release.wait()
        return "stopped"

    monkeypatch.setattr(adapter, "invoke", stubborn)
    with delegation_scope("recover") as ledger:
        parent = asyncio.create_task(tools[tool_name].fn(**arguments))
        try:
            await asyncio.wait_for(started.wait(), 2)
            parent.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(parent, 0.5)
            attempt = ledger.snapshot()[0]
            assert attempt.status == "pending"
            assert attempt.run_id is not None
            assert runtime._subagent_coordinator.status(attempt.run_id) == "running"
        finally:
            release.set()
            await asyncio.gather(parent, return_exceptions=True)
            await runtime.aclose()
