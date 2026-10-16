from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

import pytest
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel

import kedi
from kedi.agent_adapter import ConversationState
from kedi.agent_adapter.adapters.pydantic import PydanticAdapter
from kedi.agent_adapter.execution import run_in_worker, run_sync
from kedi.agent_adapter.hooks import HookDispatcher, HookDispatcherSnapshot
from kedi.agent_adapter.hooks.models import HookTimeoutError, UserPromptSubmitRequest
from kedi.agent_adapter.hooks.models import UserPromptSubmitDecision
from kedi.agent_adapter.tools.execution import admit_artifact_result_async
from kedi.agent_profile import HookSettings
from kedi.artifacts import ArtifactManager, ArtifactPolicy, ArtifactStream
from kedi.engine import JobManagerEngine, ThreadEngine
from kedi.lang import compile_program, parse_program
from tests.mock_adapter import MockAdapter


def test_nested_workers_do_not_starve_a_single_worker_host_pool() -> None:
    async def scenario() -> None:
        loop = asyncio.get_running_loop()
        loop.set_default_executor(ThreadPoolExecutor(max_workers=1))

        def nested() -> str:
            return run_sync(run_in_worker(lambda: "completed"))

        assert await asyncio.wait_for(run_in_worker(nested), 3) == "completed"

    asyncio.run(scenario())


def test_async_query_cancels_and_joins_model_cleanup() -> None:
    async def scenario() -> None:
        started = asyncio.Event()
        finished = asyncio.Event()

        async def respond(messages, info):
            started.set()
            try:
                await asyncio.Future()
            finally:
                finished.set()

        @kedi.query(adapter=PydanticAdapter(FunctionModel(respond)))
        async def code() -> str:
            """kedi
            >> The code is [code: str].
            = <code>
            """

        task = asyncio.create_task(code())
        await asyncio.wait_for(started.wait(), 3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 3)
        assert finished.is_set()

    asyncio.run(scenario())


def test_async_query_cache_waiter_cancels_without_cancelling_producer() -> None:
    async def scenario() -> None:
        started = asyncio.Event()
        release = asyncio.Event()
        calls = 0

        async def respond(messages, info):
            nonlocal calls
            calls += 1
            started.set()
            await release.wait()
            return ModelResponse([ToolCallPart(info.output_tools[0].name, {"code": "ready"})])

        @kedi.query(adapter=PydanticAdapter(FunctionModel(respond)), cache=True)
        async def code() -> str:
            """kedi
            >> The code is [code: str].
            = <code>
            """

        producer = asyncio.create_task(code())
        await asyncio.wait_for(started.wait(), 3)
        waiter = asyncio.create_task(code())
        await asyncio.sleep(0.05)
        waiter.cancel()
        try:
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(waiter, 3)
            assert not producer.done()
        finally:
            release.set()
        assert await producer == "ready"
        assert await code() == "ready"
        assert calls == 1

    kedi.clear_cache()
    try:
        asyncio.run(scenario())
    finally:
        kedi.clear_cache()


def test_sync_hook_timeout_keeps_loop_responsive_and_joins_worker() -> None:
    async def scenario() -> None:
        release = threading.Event()
        finished = threading.Event()

        def hook(event):
            try:
                assert release.wait(3)
            finally:
                finished.set()

        dispatcher = HookDispatcher(
            HookDispatcherSnapshot.compose(
                None,
                HookSettings(user_prompt_submit=(hook,)),
                timeout=0.01,
            )
        )
        task = asyncio.create_task(
            dispatcher.dispatch(
                UserPromptSubmitRequest(
                    run_id="run",
                    sequence=0,
                    adapter_shortname="fake",
                    content="prompt",
                )
            )
        )
        try:
            await asyncio.sleep(0.05)
            assert not task.done()
            assert not finished.is_set()
        finally:
            release.set()
        with pytest.raises(HookTimeoutError):
            await task
        assert finished.is_set()

    asyncio.run(scenario())


def test_sync_artifact_stream_cancellation_aborts_and_closes_iterator() -> None:
    async def scenario() -> None:
        release = threading.Event()
        started = threading.Event()
        closed = threading.Event()

        def chunks() -> Iterator[str]:
            try:
                yield "first"
                started.set()
                assert release.wait(3)
                yield "second"
            finally:
                closed.set()

        manager = ArtifactManager(session_id="cancel-stream")
        task = asyncio.create_task(
            admit_artifact_result_async(
                ArtifactStream.text(chunks()),
                manager=manager,
                policy=ArtifactPolicy(threshold_bytes=1),
                tool_name="stream",
                call_id="call",
            )
        )
        try:
            assert await asyncio.to_thread(started.wait, 3)
            task.cancel()
            await asyncio.sleep(0.02)
            assert not task.done()
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert closed.is_set()
        assert manager.stats()["active_artifacts"] == 0
        assert manager._reserved_artifacts == manager._reserved_bytes == 0
        manager.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("terminal", [False, True])
@pytest.mark.parametrize("raw", [False, True])
def test_retry_preserves_conversation_source_order(terminal: bool, raw: bool) -> None:
    calls: list[str] = []

    class RateLimitError(Exception):
        pass

    class FlakyAdapter(MockAdapter):
        def produce_sync(self, *, template, **kwargs):
            calls.append("first" if "first" in template else "second")
            if calls[-1] == "first" and (terminal or len(calls) == 1):
                raise RateLimitError("busy")
            return super().produce_sync(template=template, **kwargs)

        def invoke_sync(self, *, prompt, **kwargs):
            self.produce_sync(template=prompt)
            return "ready"

    engine = JobManagerEngine(max_workers=2, max_attempts=2, _sleep=lambda _: None)
    state = ConversationState()
    source = (
        "[one] << first\n[two] << second\n= <two>\n"
        if raw
        else ">> The first code is [one: str].\n>> The second code is [two: str].\n= <two>\n"
    )
    runtime = compile_program(
        parse_program(source), adapter=FlakyAdapter(), engine=engine, conversation_state=state
    )
    try:
        if terminal:
            with pytest.raises(Exception, match="busy"):
                runtime.run_main()
        else:
            runtime.run_main()
        assert calls == ["first", "first", "second"]
        assert len(state.snapshot()) == (2 if terminal else 4)
    finally:
        engine.shutdown()
        asyncio.run(runtime.aclose())
        state.close()


def test_retry_does_not_admit_request_after_deadline() -> None:
    now = 0.0
    calls: list[float] = []

    class RateLimitError(Exception):
        retry_after = 10

    def sleep(delay: float) -> None:
        nonlocal now
        now += delay

    def job() -> None:
        calls.append(now)
        raise RateLimitError("busy")

    engine = JobManagerEngine(deadline=1, _clock=lambda: now, _sleep=sleep)
    try:
        with pytest.raises(RateLimitError, match="busy"):
            engine.schedule([], job).result(timeout=3)
        assert calls == [0]
        assert now == 1
        assert engine.limiter._inflight == 0
    finally:
        engine.shutdown()


def test_response_cache_distinguishes_tool_closures() -> None:
    def make_lookup(value: str):
        def lookup() -> str:
            return value

        return lookup

    @kedi.query(cache=True)
    def read_value() -> str:
        """kedi
        = `lookup()`
        """

    first, second = make_lookup("first"), make_lookup("second")
    adapter = MockAdapter()
    kedi.clear_cache()
    try:
        for tool, expected in ((first, "first"), (second, "second"), (first, "first")):
            with kedi.context(adapter=adapter, tools=[tool]):
                assert read_value() == expected
    finally:
        kedi.clear_cache()


def test_response_cache_distinguishes_prompt_hook_closures() -> None:
    def make_hook(value: str):
        def hook(event):
            return UserPromptSubmitDecision.edit(f"The {value} is [code].")

        return hook

    def respond(messages, info):
        code = "first" if "first" in repr(messages) else "second"
        return ModelResponse([ToolCallPart(info.output_tools[0].name, {"code": code})])

    @kedi.query(cache=True)
    def code() -> str:
        """kedi
        >> The code is [code: str].
        = <code>
        """

    adapter = PydanticAdapter(FunctionModel(respond))
    hooks = (make_hook("first"), make_hook("second"))
    kedi.clear_cache()
    try:
        for hook, expected in zip(hooks, ("first", "second"), strict=True):
            with kedi.context(adapter=adapter, hooks={"user_prompt_submit": hook}):
                assert code() == expected
    finally:
        kedi.clear_cache()


def test_job_manager_retries_real_pydantic_conversation_without_partial_history() -> None:
    calls = 0

    class RateLimitError(Exception):
        status_code = 429

    def respond(messages, info):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RateLimitError("retry")
        return ModelResponse([ToolCallPart(info.output_tools[0].name, {"code": "ready"})])

    state = ConversationState()
    engine = JobManagerEngine(max_workers=1, max_attempts=2, _sleep=lambda _: None)
    runtime = compile_program(
        parse_program(">> The code is [code: str].\n= <code>\n"),
        adapter=PydanticAdapter(FunctionModel(respond)),
        engine=engine,
        conversation_state=state,
    )
    try:
        assert runtime.run_main() == "ready"
        assert calls == 2
        assert len(state.snapshot()) == 2
    finally:
        engine.shutdown()
        asyncio.run(runtime.aclose())
        state.close()


def test_closed_runtime_rejects_main_procedure_and_surface() -> None:
    runtime = compile_program(parse_program("= done\n"), adapter=MockAdapter())
    asyncio.run(runtime.aclose())
    for invoke in (
        runtime.run_main,
        lambda: runtime._invoke_procedure(lambda: "bad", []),
        runtime.bind_agent,
    ):
        with pytest.raises(RuntimeError, match="KediRuntime is closed"):
            invoke()


def test_retiring_a_completed_ticket_does_not_advance_its_successor() -> None:
    state = ConversationState()
    first = state.reserve_turn()
    with state.adapter_turn("test", ticket=first):
        pass
    state.cancel_turn(first)
    second = state.reserve_turn()
    with state.adapter_turn("test", ticket=second) as turn:
        assert turn.turn_sequence == 1
        state.cancel_turn(first)
        assert state._active_ticket == second.sequence
    state.close()


def test_runtime_close_cancels_active_model_and_waits_for_cleanup() -> None:
    async def scenario() -> None:
        started = threading.Event()
        finished = threading.Event()

        async def respond(messages, info):
            started.set()
            try:
                await asyncio.Future()
            finally:
                finished.set()

        engine = ThreadEngine(max_workers=1)
        runtime = compile_program(
            parse_program(">> The code is [code: str].\n= <code>\n"),
            adapter=PydanticAdapter(FunctionModel(respond)),
            engine=engine,
        )
        task = asyncio.create_task(run_in_worker(runtime.run_main))
        try:
            assert await asyncio.to_thread(started.wait, 3)
            await asyncio.wait_for(runtime.aclose(), 3)
            assert finished.is_set()
            with pytest.raises(Exception, match="interrupted"):
                await task
        finally:
            engine.shutdown()

    asyncio.run(scenario())
