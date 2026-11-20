"""CodeMode tool calls gathered in generated code run concurrently for synchronous tools (#211)."""

import asyncio
import time
from contextvars import ContextVar
from threading import Lock, current_thread, main_thread

import pytest

from kedi.agent_adapter.codemode import CodeModeLimits
from kedi.agent_adapter.codemode.tool_specs import CodeModeToolSpecBridge
from kedi.agent_adapter.tools.specs import ToolSpec

CODE = """import asyncio
values = await asyncio.gather(
    slow_read(index=0), slow_read(index=1), slow_read(index=2), slow_read(index=3)
)
sum(values)
"""

REQUEST = ContextVar("request", default="unset")


class Recorder:
    def __init__(self) -> None:
        self.lock = Lock()
        self.active = 0
        self.peak = 0
        self.threads: set[str] = set()
        self.requests: set[str] = set()

    def enter(self) -> None:
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.threads.add(current_thread().name)
            self.requests.add(REQUEST.get())

    def leave(self) -> None:
        with self.lock:
            self.active -= 1


async def _run(fn, *, sequential: bool = False) -> int:
    tool = ToolSpec(
        fn=fn,
        name="slow_read",
        description="Read one integer after a short delay.",
        json_schema={
            "type": "object",
            "properties": {"index": {"type": "integer"}},
            "required": ["index"],
        },
        risk="read_only",
        sequential=sequential,
    )
    bridge = CodeModeToolSpecBridge(
        [tool], limits=CodeModeLimits(), policies=(), adapter_shortname="langchain"
    )
    try:
        bridge.get_tool_schema(tool_names=["slow_read"])
        return await bridge.execute_code(code=CODE)
    finally:
        await bridge.close()


def _sync_tool(recorder: Recorder):
    def slow_read(*, index: int) -> int:
        recorder.enter()
        try:
            time.sleep(0.08)
            return index
        finally:
            recorder.leave()

    return slow_read


@pytest.mark.parametrize(("sequential", "peak"), [(False, 4), (True, 1)])
def test_gathered_sync_tools_run_off_the_loop_unless_sequential(sequential, peak):
    recorder = Recorder()

    async def scenario() -> int:
        REQUEST.set("request-1")
        return await _run(_sync_tool(recorder), sequential=sequential)

    assert asyncio.run(scenario()) == 6
    assert recorder.peak == peak
    assert recorder.active == 0
    assert main_thread().name not in recorder.threads
    # The caller's context reaches the worker thread.
    assert recorder.requests == {"request-1"}


def test_gathered_async_tools_still_run_on_the_loop():
    recorder = Recorder()

    async def slow_read(*, index: int) -> int:
        recorder.enter()
        try:
            await asyncio.sleep(0.08)
            return index
        finally:
            recorder.leave()

    assert asyncio.run(_run(slow_read)) == 6
    assert recorder.peak == 4
    assert recorder.threads == {main_thread().name}


def test_sync_tools_that_return_an_awaitable_are_awaited():
    def slow_read(*, index: int):
        return asyncio.sleep(0.01, result=index)

    assert asyncio.run(_run(slow_read)) == 6
