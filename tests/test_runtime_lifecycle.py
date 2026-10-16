from __future__ import annotations

import asyncio
import contextvars
import threading

import pytest

from kedi.agent_adapter.execution import run_in_worker, run_sync
from kedi.core.lifecycle import RuntimeLifecycle
from kedi.engine import SequentialEngine
from kedi.lang import compile_program, parse_program
from tests.mock_adapter import MockAdapter


def test_runtime_cannot_close_itself_mid_invocation() -> None:
    runtime = compile_program(
        parse_program("= `close_self()`\n"), adapter=MockAdapter(), engine=SequentialEngine()
    )

    def close_self() -> None:
        run_sync(runtime.aclose())

    runtime.set_initial_globals({"close_self": close_self})
    with pytest.raises(Exception, match="Cannot close KediRuntime from an active invocation"):
        runtime.run_main()
    asyncio.run(runtime.aclose())


def test_retired_admission_context_cannot_bypass_runtime_close() -> None:
    lifecycle = RuntimeLifecycle()
    with lifecycle.admission():
        captured = contextvars.copy_context()
    asyncio.run(lifecycle.aclose())

    def enter() -> None:
        with lifecycle.admission():
            pytest.fail("retired context bypassed closed admission")

    with pytest.raises(RuntimeError, match="KediRuntime is closed"):
        captured.run(enter)


def test_close_rejects_new_invocations_while_joining_admitted_sync_work() -> None:
    async def scenario() -> None:
        started, release = threading.Event(), threading.Event()

        def hold() -> str:
            started.set()
            assert release.wait(3)
            return "finished"

        runtime = compile_program(
            parse_program("= `hold()`\n"),
            adapter=MockAdapter(),
            runtime_globals={"hold": hold},
            engine=SequentialEngine(),
        )
        active = asyncio.create_task(run_in_worker(runtime.run_main))
        assert await asyncio.to_thread(started.wait, 3)
        closing = asyncio.create_task(runtime.aclose())
        try:
            await asyncio.sleep(0.02)
            assert not closing.done()
            with pytest.raises(RuntimeError, match="KediRuntime is closed"):
                runtime.run_main()
        finally:
            release.set()
        assert await active == "finished"
        await closing
        await runtime.aclose()

    asyncio.run(scenario())
