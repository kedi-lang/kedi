from __future__ import annotations

import asyncio
import contextvars
import linecache
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


def test_root_aclose_closes_imported_module_runtimes_and_their_linecache(tmp_path) -> None:
    # #214 1.13 / 1.6: imported modules own their resources, and each runtime's Python
    # origins are named apart in the process-wide linecache and released on close.
    (tmp_path / "helpers.kedi").write_text(
        "@double(x: int) -> int:\n  = `x * 2`\n\n> export:\n  double\n", encoding="utf-8"
    )
    main = tmp_path / "main.kedi"
    main.write_text("> import: helpers:\n  double\n\n= `double(21)`\n", encoding="utf-8")
    runtime = compile_program(parse_program(main.read_text(), source_path=str(main)))
    assert runtime.run_main() == 42
    module = runtime._module_runtimes["helpers"]
    assert module._python_origin_prefix != runtime._python_origin_prefix
    origins = [*runtime._python_origins, *module._python_origins]
    assert origins and all(name in linecache.cache for name in origins)

    asyncio.run(runtime.aclose())

    assert module.artifacts._closed
    assert not any(name in linecache.cache for name in origins)


def test_root_aclose_closes_every_module_runtime_before_raising() -> None:
    root = compile_program(parse_program("= `1`\n"))
    other = compile_program(parse_program("= `2`\n"))
    other.module_name = "other"
    root._module_runtimes["other"] = other

    async def fail() -> None:
        raise RuntimeError("root close failed")

    root._lifecycle.aclose = fail  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="root close failed"):
        asyncio.run(root.aclose())
    assert other.artifacts._closed
