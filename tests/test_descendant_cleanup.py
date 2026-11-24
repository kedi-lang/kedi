from __future__ import annotations

import asyncio

import pytest

from kedi.agent_adapter.subagents import communication
from kedi.agent_adapter.subagents.control import LiveInput
from kedi.agent_adapter.subagents.coordinator import _OwnedSubagentTask
from kedi.errors import KediExecutionError
from tests.test_subagents import _manager_tools


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, "cancel", "quiescence", "timeout"])
async def test_descendant_cleanup_traverses_terminal_parents_and_isolates_failures(
    monkeypatch: pytest.MonkeyPatch, failure: str | None
) -> None:
    _, runtime, _ = _manager_tools()
    coordinator = runtime._subagent_coordinator
    parents = {"root": None, "done": "root", "leaf": "done", "other": "root", "sibling": None}
    for run_id, parent in parents.items():
        coordinator.runs.create(
            run_id=run_id,
            parent_run_id=parent,
            root_run_id="root" if parent else run_id,
            profile_name="worker",
            depth=1,
        )
        coordinator.runs.transition(run_id, "running")
        if run_id == "done":
            coordinator.runs.transition(run_id, "completed")
        coordinator._controls[run_id] = LiveInput(run_id)

    cancelled, waited, quiescent = set(), set(), set()
    error = KediExecutionError("cleanup failed")

    def cancel(run_id: str) -> bool:
        cancelled.add(run_id)
        if failure == "cancel" and run_id == "done":
            raise error
        return True

    async def wait(run_id: str, **_kwargs: object) -> None:
        waited.add(run_id)
        if failure == "timeout":
            await asyncio.Event().wait()

    async def settle(control: LiveInput) -> None:
        quiescent.add(control.target_id)
        if failure == "quiescence" and control.target_id == "done":
            raise error

    monkeypatch.setattr(coordinator, "cancel", cancel)
    monkeypatch.setattr(coordinator, "wait", wait)
    monkeypatch.setattr(LiveInput, "wait_quiescent", settle)
    monkeypatch.setattr(communication, "INTERRUPT_CLEANUP_SECONDS", 0.02)
    try:
        if failure in {"cancel", "quiescence"}:
            with pytest.raises(KediExecutionError) as caught:
                await coordinator.stop_descendants("root")
            assert caught.value is error
        elif failure == "timeout":
            with pytest.raises(RuntimeError, match="descendants did not stop"):
                await coordinator.stop_descendants("root")
        else:
            await coordinator.stop_descendants("root")
        assert cancelled == waited == {"done", "leaf", "other"}
        assert quiescent == (set() if failure == "timeout" else cancelled)
    finally:
        await runtime.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("retained_middle", [False, True])
async def test_retention_keeps_ancestry_until_active_descendants_stop(
    retained_middle: bool,
) -> None:
    _, runtime, _ = _manager_tools()
    coordinator = runtime._subagent_coordinator
    coordinator.max_retained_runs = 1
    if retained_middle:
        coordinator._retained_native_runs.add("middle")
    release = asyncio.Event()

    async def child() -> None:
        await release.wait()

    tasks = []
    for run_id, parent in (("root", None), ("middle", "root"), ("leaf", "middle")):
        coordinator.runs.create(
            run_id=run_id,
            parent_run_id=parent,
            root_run_id="root",
            profile_name="worker",
            depth=1,
        )
        task = asyncio.create_task(child())
        tasks.append(task)
        coordinator._tasks[run_id] = _OwnedSubagentTask(task, asyncio.get_running_loop())
        if run_id != "leaf":
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    try:
        with coordinator._task_lock:
            assert coordinator._prune_finished_locked() == []
        release.set()
        await tasks[-1]
        with coordinator._task_lock:
            assert coordinator._prune_finished_locked() == [
                "root",
                "leaf" if retained_middle else "middle",
            ]
    finally:
        release.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        coordinator._tasks.clear()
        await runtime.aclose()
