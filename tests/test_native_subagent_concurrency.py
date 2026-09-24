from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from threading import Barrier, BrokenBarrierError, Lock, enumerate as enumerate_threads
from typing import Any

import pytest

import kedi.agent_adapter.subagents.native as native_module
from kedi.agent_adapter.execution import OwnedAsyncLoop
from kedi.agent_adapter import LangChainAdapter
from kedi.agent_adapter.subagents.budget import SubagentBudgetExceeded, SubagentConcurrencyExceeded
from kedi.agent_adapter.subagents.native import NativeSubagentOwner
from kedi.agent_adapter.subagents.workflow import dynamic_workflow_tool_spec
from kedi.lang import compile_program, parse_program
from kedi.run_budget import current_run_budget
from tests.test_native_subagent_tasks import _runtime
from tests.test_subagents import RecordingPydanticAdapter


def test_concurrent_first_sync_starts_share_one_owned_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime, _ = _runtime("")
    owner = NativeSubagentOwner(runtime)
    profile = runtime._evaluate_profile_definition(runtime._profile_registry["manager"])
    constructors = Barrier(2)
    created: list[OwnedAsyncLoop] = []
    lock = Lock()

    def overlapping_loop_constructor() -> OwnedAsyncLoop:
        # Without atomic lazy initialization, both threads enter the constructor.
        try:
            constructors.wait(timeout=0.25)
        except BrokenBarrierError:
            pass
        loop = OwnedAsyncLoop()
        with lock:
            created.append(loop)
        return loop

    monkeypatch.setattr(native_module, "OwnedAsyncLoop", overlapping_loop_constructor)
    token = runtime._native_subagent_owner.set(owner)
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(
                    copy_context().run,
                    lambda: owner.wait_sync(
                        owner.start_sync("worker", task="Review.", parent_profile=profile)
                    ),
                )
                for _ in range(2)
            ]
            outcomes = []
            for future in futures:
                try:
                    outcomes.append(future.result(timeout=5))
                except Exception as exc:
                    outcomes.append(exc)
        assert len(created) == 1
        assert all(not isinstance(result, Exception) for result in outcomes), outcomes
    finally:
        # Even the broken implementation's losing loop must be drained.
        for loop in created:
            loop.run(runtime._subagent_coordinator.aclose())
            loop.close()
        runtime._native_subagent_owner.reset(token)
    assert runtime._subagent_coordinator.active_runs == 0


def test_concurrent_waiters_receive_one_materialized_result() -> None:
    release = asyncio.Event()

    class PausedAdapter(RecordingPydanticAdapter):
        async def invoke(self, *, prompt: str, **kwargs: Any) -> str:
            await release.wait()
            return prompt

    runtime, _ = _runtime("", PausedAdapter())

    async def scenario() -> None:
        async with runtime.subagents(parent="manager") as agents:
            job = await agents.start("worker", task="Review.")
            waiters = [asyncio.create_task(job.wait()) for _ in range(32)]
            await asyncio.sleep(0)
            release.set()
            results = await asyncio.wait_for(asyncio.gather(*waiters), timeout=5)
            assert all(result is results[0] for result in results)
            assert await job.wait() is results[0]

    asyncio.run(scenario())


@pytest.mark.parametrize("limit", [1, 2, 8])
@pytest.mark.parametrize("adapter_name", ["pydantic", "langchain"])
def test_parallel_admission_is_atomic_and_releases_all_slots(limit: int, adapter_name: str) -> None:
    release = asyncio.Event()
    entered: list[str] = []

    async def invoke(*, prompt: str, **kwargs: Any) -> str:
        entered.append(prompt)
        await release.wait()
        return prompt

    class PausedPydantic(RecordingPydanticAdapter):
        async def invoke(self, **kwargs: Any) -> str:
            return await invoke(**kwargs)

    class PausedLangChain(LangChainAdapter):
        def __init__(self) -> None:
            super().__init__(model=object())

        async def invoke(self, **kwargs: Any) -> str:
            return await invoke(**kwargs)

    runtime = compile_program(
        parse_program(
            f"> profile: worker:\n    > adapter: {adapter_name}\n"
            f"> profile: manager:\n    > adapter: {adapter_name}\n"
            "    > subagent: worker\n    > max_agents: 100\n"
        ),
        adapter=PausedPydantic() if adapter_name == "pydantic" else PausedLangChain(),
    )
    runtime._subagent_coordinator.max_concurrency = limit

    async def scenario() -> None:
        async with runtime.subagents(parent="manager") as agents:
            results = await asyncio.gather(
                *(agents.start("worker", task=f"Job {index}") for index in range(24)),
                return_exceptions=True,
            )
            jobs = [result for result in results if not isinstance(result, BaseException)]
            errors = [result for result in results if isinstance(result, BaseException)]
            assert len(jobs) == limit
            assert len(errors) == 24 - limit
            assert all(isinstance(error, SubagentConcurrencyExceeded) for error in errors)
            assert runtime._subagent_coordinator.active_runs == limit
            owner = runtime._native_subagent_owner.get()
            profile = runtime._evaluate_profile_definition(runtime._profile_registry["manager"])
            assert owner.budget_for(profile).started == limit
            release.set()
            await asyncio.wait_for(asyncio.gather(*(job.wait() for job in jobs)), 5)
            assert runtime._subagent_coordinator.active_runs == 0
            assert owner.concurrency.active == 0
            await (await agents.start("worker", task="After release.")).wait()
        assert len(entered) == limit + 1
        assert all(owned.task.done() for owned in runtime._subagent_coordinator._tasks.values())
        assert not runtime._subagent_coordinator._timeout_handles
        assert not runtime._subagent_coordinator._retained_native_runs

    asyncio.run(scenario())


def test_concurrent_roots_on_separate_threads_preserve_context_and_budgets() -> None:
    rendezvous = Barrier(8)
    thread_guard = Lock()
    identities: list[tuple[object, object, object]] = []

    class RendezvousAdapter(RecordingPydanticAdapter):
        async def invoke(self, *, prompt: str, **kwargs: Any) -> str:
            before = (self.active_profile, runtime._invocation_id.get(), current_run_budget())
            with thread_guard:
                identities.append(before)
            await asyncio.to_thread(rendezvous.wait, 5)
            after = (self.active_profile, runtime._invocation_id.get(), current_run_budget())
            assert all(left is right for left, right in zip(before, after))
            return prompt

    runtime, _ = _runtime("", RendezvousAdapter())
    runtime._subagent_coordinator.max_concurrency = 8

    async def scenario(index: int) -> str:
        async with runtime.subagents(parent="manager") as agents:
            job = await agents.start("worker", task=f"Root {index}")
            result = await job.wait()
        assert runtime._native_subagent_owner.get() is None
        assert runtime._invocation_id.get() is None
        assert current_run_budget() is None
        return result.task_summary

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(asyncio.run, scenario(index)) for index in range(8)]
        assert [future.result(timeout=10) for future in futures] == [f"Root {i}" for i in range(8)]
    assert len({id(identity[1]) for identity in identities}) == 8
    assert runtime._subagent_coordinator.active_runs == 0
    assert all(owned.task.done() for owned in runtime._subagent_coordinator._tasks.values())
    assert not runtime._subagent_coordinator._timeout_handles
    assert not runtime._subagent_coordinator._retained_native_runs


def test_cancel_one_root_does_not_cancel_another_root() -> None:
    entered = {name: asyncio.Event() for name in ("cancel", "survive")}
    release = asyncio.Event()

    class PausedAdapter(RecordingPydanticAdapter):
        async def invoke(self, *, prompt: str, **kwargs: Any) -> str:
            entered[prompt].set()
            await release.wait()
            return prompt

    runtime, _ = _runtime("", PausedAdapter())

    async def scenario() -> None:
        async def root(name: str) -> str:
            async with runtime.subagents(parent="manager") as agents:
                return (await (await agents.start("worker", task=name)).wait()).task_summary

        cancelled = asyncio.create_task(root("cancel"))
        survivor = asyncio.create_task(root("survive"))
        await asyncio.wait_for(asyncio.gather(*(event.wait() for event in entered.values())), 5)
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        assert not survivor.done()
        assert runtime._subagent_coordinator.active_runs == 1
        release.set()
        assert await asyncio.wait_for(survivor, 5) == "survive"
        assert runtime._subagent_coordinator.active_runs == 0
        assert not runtime._subagent_coordinator._retained_native_runs

    asyncio.run(scenario())


@pytest.mark.parametrize("dynamic_first", [False, True])
def test_native_and_dynamic_starts_share_the_same_parent_budget(dynamic_first: bool) -> None:
    runtime, _ = _runtime("")

    async def scenario() -> None:
        async with runtime.subagents(parent="manager") as agents:
            profile = runtime._evaluate_profile_definition(runtime._profile_registry["manager"])
            workflow = dynamic_workflow_tool_spec(runtime, profile)

            async def native() -> None:
                await (await agents.start("worker", task="Native.")).wait()

            async def dynamic() -> None:
                await workflow.fn(code="await worker(task='Dynamic.')")

            operations = (dynamic, native) if dynamic_first else (native, dynamic)
            await asyncio.gather(*(operation() for operation in operations))
            with pytest.raises(SubagentBudgetExceeded):
                await agents.start("worker", task="Over budget.")
        assert len(runtime._subagent_coordinator.runs.snapshot()) == 2

    asyncio.run(scenario())


def test_repeated_sync_invocations_leave_no_owned_threads() -> None:
    previous = {thread.ident for thread in enumerate_threads()}
    runtime, _ = _runtime(
        "> task [left]: worker:\n    >> First.\n"
        "> task [right]: worker:\n    >> Second.\n"
        "> await: left\n> await: right\n"
    )
    for _ in range(20):
        runtime.run_main()
        assert runtime._subagent_coordinator.active_runs == 0
        assert not runtime._subagent_coordinator._retained_native_runs
    assert not [
        thread
        for thread in enumerate_threads()
        if thread.name == "kedi-invocation-loop" and thread.ident not in previous
    ]


def test_reusing_scope_under_another_root_preserves_context_tokens() -> None:
    runtime, _ = _runtime("")
    reusable = runtime.subagents(parent="manager")

    async def scenario() -> None:
        async with reusable as agents:
            await (await agents.start("worker", task="First root.")).wait()
        async with runtime.subagents(parent="manager"):
            owner = runtime._native_subagent_owner.get()
            ledger = current_run_budget()
            invocation = runtime._invocation_id.get()
            request_budget = runtime._request_budget.get()
            async with reusable as agents:
                await (await agents.start("worker", task="Nested reuse.")).wait()
            assert runtime._native_subagent_owner.get() is owner
            assert current_run_budget() is ledger
            assert runtime._invocation_id.get() is invocation
            assert runtime._request_budget.get() is request_budget
        assert runtime._native_subagent_owner.get() is None
        assert runtime._invocation_id.get() is None
        assert runtime._request_budget.get() is None
        assert current_run_budget() is None

    asyncio.run(scenario())


def test_parallel_sibling_invocations_have_independent_descendant_budgets() -> None:
    class DelegatingAdapter(RecordingPydanticAdapter):
        async def invoke(self, *, prompt: str, **kwargs: Any) -> str:
            if self.active_profile.name == "worker":
                async with runtime.subagents(parent="worker") as agents:
                    child = await agents.start("leaf", task=prompt)
                    return (await child.wait()).task_summary
            return prompt

    runtime = compile_program(
        parse_program(
            "> profile: leaf:\n    > adapter: pydantic\n"
            "> profile: worker:\n    > adapter: pydantic\n"
            "    > subagent: leaf\n    > max_agents: 1\n"
            "> profile: manager:\n    > adapter: pydantic\n"
            "    > subagent: worker\n    > max_agents: 4\n"
        ),
        adapter=DelegatingAdapter(),
    )

    async def scenario() -> None:
        async with runtime.subagents(parent="manager") as agents:
            jobs = await asyncio.gather(
                agents.start("worker", task="Left."),
                agents.start("worker", task="Right."),
            )
            results = await asyncio.wait_for(asyncio.gather(*(job.wait() for job in jobs)), 5)
            assert [result.task_summary for result in results] == ["Left.", "Right."]
        runs = runtime._subagent_coordinator.runs.snapshot()
        assert len(runs) == 4
        assert all(run.status == "completed" for run in runs)
        assert runtime._subagent_coordinator.active_runs == 0

    asyncio.run(scenario())
