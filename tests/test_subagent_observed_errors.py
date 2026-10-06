import asyncio

import pytest

from kedi.errors import KediExecutionError
from kedi.lang import compile_program, parse_program
from tests.test_native_subagent_tasks import PROFILES
from tests.test_subagents import RecordingPydanticAdapter


def test_caught_child_failure_can_recover_inside_same_scope():
    class Adapter(RecordingPydanticAdapter):
        async def invoke(self, *, prompt, **kwargs):
            if prompt == "fail":
                raise ValueError("Expected child failure")
            return await super().invoke(prompt=prompt, **kwargs)

    runtime = compile_program(parse_program(PROFILES), adapter=Adapter())

    async def scenario():
        async with runtime.subagents(parent="manager") as scope:
            child = await scope.start("worker", task="fail")
            for _ in range(2):
                with pytest.raises(ValueError, match="Expected child failure"):
                    await child.wait()
            next_child = await scope.start("worker", task="recover")
            assert (await next_child.wait()).task_summary == "child:recover"
        assert runtime._subagent_coordinator.active_runs == 0
        await runtime.aclose()

    asyncio.run(scenario())


def test_caught_child_cancellation_releases_slot_without_scope_exit_error():
    entered = asyncio.Event()

    class Adapter(RecordingPydanticAdapter):
        async def invoke(self, *, prompt, **kwargs):
            if prompt == "block":
                entered.set()
                await asyncio.Event().wait()
            return await super().invoke(prompt=prompt, **kwargs)

    runtime = compile_program(
        parse_program(PROFILES), adapter=Adapter(), subagent_max_concurrency=1
    )

    async def scenario():
        async with runtime.subagents(parent="manager") as scope:
            child = await scope.start("worker", task="block")
            await entered.wait()
            assert runtime._cancel_subagent(child.run_id)
            with pytest.raises(asyncio.CancelledError):
                await child.wait()
            recovered = await scope.start("worker", task="recover")
            assert (await recovered.wait()).task_summary == "child:recover"
        assert runtime._subagent_coordinator.active_runs == 0
        await runtime.aclose()

    asyncio.run(scenario())


@pytest.mark.parametrize("wait_failure", ["cancel", "transport"])
def test_failed_waiter_does_not_mark_running_child_observed(monkeypatch, wait_failure):
    entered = asyncio.Event()

    class Adapter(RecordingPydanticAdapter):
        async def invoke(self, *, prompt, **kwargs):
            entered.set()
            await asyncio.Event().wait()

    runtime = compile_program(parse_program(PROFILES), adapter=Adapter())

    async def scenario():
        with pytest.raises(KediExecutionError, match="was not awaited"):
            async with runtime.subagents(parent="manager") as scope:
                child = await scope.start("worker", task="block")
                await entered.wait()
                if wait_failure == "transport":

                    async def fail_wait(run_id):
                        raise ConnectionError("wait transport failed")

                    with monkeypatch.context() as patch:
                        patch.setattr(runtime, "_wait_subagent", fail_wait)
                        with pytest.raises(ConnectionError, match="wait transport failed"):
                            await child.wait()
                else:
                    waiter = asyncio.create_task(child.wait())
                    await asyncio.sleep(0)
                    waiter.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await waiter
                assert child.run_id not in child._owner._observed
        assert runtime._subagent_coordinator.active_runs == 0
        await runtime.aclose()

    asyncio.run(scenario())
