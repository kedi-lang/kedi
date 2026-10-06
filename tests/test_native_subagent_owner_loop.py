import asyncio

import httpx
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from kedi.agent_adapter import LangChainAdapter
from kedi.agent_adapter.execution import use_async_owner_loop
from kedi.agent_adapter.subagents.native import NativeSubagentOwner
from kedi.engine import ThreadEngine
from kedi.errors import KediExecutionError
from kedi.lang import compile_program, parse_program
from tests.test_native_subagent_tasks import PROFILES
from tests.test_subagents import RecordingPydanticAdapter


def test_subagent_rejects_stopped_borrowed_loop_without_starting_work():
    runtime = compile_program(parse_program(PROFILES), adapter=RecordingPydanticAdapter())
    loop = asyncio.new_event_loop()
    owner = NativeSubagentOwner(runtime)
    owner._execution_loop = loop

    def unscheduled():
        pytest.fail("Stopped loop must reject work before creating a coroutine")

    try:
        with pytest.raises(KediExecutionError, match="not running"):
            owner.submit(unscheduled)
    finally:
        loop.close()
        asyncio.run(runtime.aclose())


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["pydantic", "langchain"])
async def test_native_tasks_borrow_host_loop_without_closing_http_client(backend):
    host = asyncio.get_running_loop()
    loops = []
    transports = []
    owners = []

    async def respond(request):
        transports.append(request.transport)
        return web.json_response({"value": 7})

    app = web.Application()
    app.router.add_get("/", respond)
    async with TestServer(app) as server, httpx.AsyncClient() as client:
        base = RecordingPydanticAdapter if backend == "pydantic" else LangChainAdapter

        class Adapter(base):
            async def invoke(self, *, prompt, **kwargs):
                loops.append(asyncio.get_running_loop())
                owners.append(runtime._native_subagent_owner.get())
                assert asyncio.get_running_loop() is host
                response = await client.get(str(server.make_url("/")))
                return str(response.json()["value"])

        source = (
            PROFILES.replace("adapter: pydantic", f"adapter: {backend}")
            + """
> task [job]: worker:
    >> Return the supplied value.
> await [answer]: job
> await [again]: job
= `dict(value=answer.task_summary, same=answer is again)`
"""
        )
        engine = ThreadEngine(2)
        runtime = compile_program(parse_program(source), adapter=Adapter(), engine=engine)
        try:
            with use_async_owner_loop():
                for _ in range(2):
                    result = await asyncio.wait_for(asyncio.to_thread(runtime.run_main), 5)
                    assert result == {"value": "7", "same": True}
                    assert runtime._subagent_coordinator.active_runs == 0
                    assert owners[-1]._closed and not owners[-1]._tasks
                    assert owners[-1]._loop is None
            assert loops == [host, host]
            assert len({id(transport) for transport in transports}) == 1
            assert not client.is_closed and not host.is_closed()
        finally:
            await runtime.aclose()
            engine.shutdown()
    assert client.is_closed and not host.is_closed()


@pytest.mark.asyncio
async def test_borrowed_loop_drains_unawaited_child_without_closing_host():
    entered, cancelled = asyncio.Event(), asyncio.Event()
    owners = []
    host = asyncio.get_running_loop()

    class Adapter(RecordingPydanticAdapter):
        async def invoke(self, *, prompt, **kwargs):
            assert asyncio.get_running_loop() is host
            owners.append(runtime._native_subagent_owner.get())
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

    source = (
        PROFILES
        + """
> task [job]: worker:
    >> Wait until cancelled.
`wait_until_entered()`
= done
"""
    )

    def wait_until_entered():
        asyncio.run_coroutine_threadsafe(entered.wait(), host).result(timeout=3)

    runtime = compile_program(
        parse_program(source),
        adapter=Adapter(),
        runtime_globals={"wait_until_entered": wait_until_entered},
    )
    try:
        with use_async_owner_loop(), pytest.raises(KediExecutionError, match="was not awaited"):
            await asyncio.wait_for(asyncio.to_thread(runtime.run_main), 5)
        assert cancelled.is_set()
        assert owners[0]._closed and not owners[0]._tasks
        assert runtime._subagent_coordinator.active_runs == 0
        assert not host.is_closed()
    finally:
        await runtime.aclose()


@pytest.mark.asyncio
async def test_borrowed_loop_ready_first_group_can_start_nested_child_with_one_worker():
    host = asyncio.get_running_loop()
    released = asyncio.Event()
    calls, processed = [], []

    class Adapter(RecordingPydanticAdapter):
        async def invoke(self, *, prompt, **kwargs):
            assert asyncio.get_running_loop() is host
            calls.append(prompt)
            if prompt == "First.":
                await released.wait()
            return prompt

    source = (
        PROFILES.replace("max_agents: 2", "max_agents: 3")
        + """
> task [first]: worker:
    >> First.
> task [second]: worker:
    >> Second.
> task_group:
    > await [answer]: first
    > process:
        `processed.append(answer.task_summary)`
    > await [answer]: second
    > process:
        > task [nested]: worker:
            >> Nested.
        > await [nested_answer]: nested
        `processed.append(nested_answer.task_summary)`
        `release_first()`
= done
"""
    )
    engine = ThreadEngine(1)
    runtime = compile_program(
        parse_program(source),
        adapter=Adapter(),
        engine=engine,
        runtime_globals={
            "processed": processed,
            "release_first": lambda: host.call_soon_threadsafe(released.set),
        },
    )
    try:
        with use_async_owner_loop():
            assert await asyncio.wait_for(asyncio.to_thread(runtime.run_main), 5) == "done"
        assert sorted(calls) == ["First.", "Nested.", "Second."]
        assert processed == ["Nested.", "First."]
        assert runtime._subagent_coordinator.active_runs == 0
    finally:
        released.set()
        await runtime.aclose()
        engine.shutdown()
