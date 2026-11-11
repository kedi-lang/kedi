from __future__ import annotations

import asyncio
import threading
from concurrent.futures import Future, ThreadPoolExecutor

import pytest

from kedi.debugging import checkpoint, execution_frame, observe_execution
from kedi.engine import ThreadEngine
from kedi.lang import compile_program, parse_program
from kedi.promise import KediPromise
from tests.mock_adapter import MockAdapter


def test_observation_does_not_change_calls_or_force_unused_captures():
    source = ">> The answer is [answer].\n[value: int] = `7`\n= done\n"
    baseline = MockAdapter()
    observed = MockAdapter()
    first = compile_program(parse_program(source), adapter=baseline)
    second = compile_program(parse_program(source), adapter=observed)
    events = []
    pending = []

    def observer(event):
        events.append((event.kind, event.safe))
        if event.kind == "statement":
            value = dict.get(event.env, "answer")
            if isinstance(value, KediPromise):
                pending.append(value._resolved)

    try:
        expected = first.run_main()
        with observe_execution(observer):
            assert second.run_main() == expected
        assert observed.request_calls == baseline.request_calls
        assert pending and not any(pending)
        assert events.count(("model_input", True)) == 1
        assert events.count(("request", False)) == 1
        assert events.count(("model_result", True)) == 1
    finally:
        asyncio.run(first.aclose())
        asyncio.run(second.aclose())


def test_paused_dispatch_does_not_block_other_lane_or_duplicate_effects():
    paused = threading.Event()
    completed = threading.Event()
    release = threading.Event()
    adapter = MockAdapter()
    engine = ThreadEngine(max_workers=2)
    runtime = compile_program(
        parse_program(">> First answer is [first].\n>> Second answer is [second].\n= done\n"),
        adapter=adapter,
        engine=engine,
    )

    def observer(event):
        if event.kind == "model_input" and "First" in event.payload["rendered_input"]:
            paused.set()
            assert release.wait(5)
        if event.kind == "model_result":
            completed.set()

    def run():
        with observe_execution(observer):
            return runtime.run_main()

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(run)
        try:
            assert paused.wait(5)
            assert completed.wait(5)
            assert len(adapter.request_calls) == 1
        finally:
            release.set()
        assert future.result(timeout=5) == "done"
    assert len(adapter.request_calls) == 2
    asyncio.run(runtime.aclose())
    engine.shutdown()


def test_model_frame_without_active_environment(monkeypatch):
    from kedi.agent_adapter.request import PromptDraft
    from kedi.core import conversation

    runtime = compile_program(parse_program("= done\n"), adapter=MockAdapter())
    events = []
    monkeypatch.setattr(conversation, "_execute_conversation_call", lambda *args, **kwargs: "done")
    try:
        assert runtime._current_env.get() is None
        with observe_execution(events.append):
            result = conversation._run_conversation_call(
                runtime,
                runtime.adapter,
                PromptDraft(content="test", kind="raw"),
                lambda request: None,
            )
        assert result == "done"
        assert events[0].env == {}
        assert [event.kind for event in events] == ["model_input", "model_result", "frame_exit"]
    finally:
        asyncio.run(runtime.aclose())


def test_model_usage_is_observed_without_stopping_transaction(monkeypatch):
    from kedi.agent_adapter.invocation import InvocationUsage, report_invocation

    adapter = MockAdapter()
    original = adapter.produce_sync
    runtime = compile_program(
        parse_program(">> The answer is [answer].\n= <answer>\n"), adapter=adapter
    )
    events = []

    def produce(**kwargs):
        report_invocation(
            InvocationUsage(requests=1, input_tokens=100, cache_read_tokens=80, output_tokens=5)
        )
        return original(**kwargs)

    monkeypatch.setattr(adapter, "produce_sync", produce)
    try:
        with observe_execution(events.append):
            runtime.run_main()
        usage = [event for event in events if event.kind == "usage"]
        assert len(usage) == 1
        assert not usage[0].safe
        assert usage[0].payload == {
            "requests": 1,
            "tool_calls": 0,
            "input_tokens": 100,
            "cache_read_tokens": 80,
            "output_tokens": 5,
        }
    finally:
        asyncio.run(runtime.aclose())


def test_async_safe_point_is_observation_only():
    runtime = compile_program(parse_program("= done\n"), adapter=MockAdapter())
    events = []

    async def run():
        with observe_execution(events.append):
            checkpoint(runtime, "statement")
        await runtime.aclose()

    asyncio.run(run())
    assert len(events) == 1
    assert not events[0].safe


def test_reading_completed_capture_does_not_report_waiting():
    runtime = compile_program(parse_program("= done\n"), adapter=MockAdapter())
    future = Future()
    future.set_result({"answer": 42})
    events = []
    try:
        with observe_execution(events.append), execution_frame(runtime, "test", {}):
            assert KediPromise(future, "answer").resolve() == 42
        assert all(event.kind not in {"waiting", "wait_complete"} for event in events)
    finally:
        asyncio.run(runtime.aclose())


def test_nested_frames_restore_on_exception_and_observer_scope_closes():
    runtime = compile_program(
        parse_program(
            "@outer():\n  @inner():\n    = `1 / 0`\n  = <inner()>\n= <outer()>\n",
            source_path="nested.kedi",
        ),
        adapter=MockAdapter(),
    )
    stops = []

    def observe(event):
        if event.kind == "exception":
            stops.append(([frame.name for frame in event.frames], event.span.start_line))

    try:
        with observe_execution(observe), pytest.raises(Exception):
            runtime.run_main()
        checkpoint(runtime, "exception")  # observer no longer installed
        assert len(stops) == 1
        assert stops[0] == (["<program>", "outer", "inner"], 3)
    finally:
        asyncio.run(runtime.aclose())


@pytest.mark.parametrize("parallel", [False, True])
def test_provider_errors_are_observed_once_outside_model_transaction(monkeypatch, parallel):
    from kedi.engine import SequentialEngine
    from kedi.errors import KediExecutionError

    adapter = MockAdapter()
    engine = ThreadEngine(max_workers=1) if parallel else SequentialEngine()
    runtime = compile_program(
        parse_program(">> The answer is [answer].\n= done\n", source_path="provider.kedi"),
        adapter=adapter,
        engine=engine,
    )
    events = []

    def fail(**kwargs):
        raise RuntimeError("Synthetic provider failure")

    monkeypatch.setattr(adapter, "produce_sync", fail)
    try:
        with observe_execution(events.append), pytest.raises(KediExecutionError):
            runtime.run_main()
        failures = [event for event in events if event.kind == "exception" and event.safe]
        assert len(failures) == 1
        assert failures[0].span.start_line == 1
        assert isinstance(failures[0].payload, RuntimeError)
    finally:
        asyncio.run(runtime.aclose())
        engine.shutdown()


def test_model_transaction_disallows_nested_synchronous_stops(monkeypatch):
    from kedi.errors import KediExecutionError, KediTraceFrame

    adapter = MockAdapter()
    runtime = compile_program(
        parse_program(">> The answer is [answer].\n= done\n"), adapter=adapter
    )
    events = []

    def fail(**kwargs):
        checkpoint(runtime, "inside_model")
        with runtime.trace_frame(KediTraceFrame(kind="tool", label="synthetic tool", span=None)):
            raise ValueError("Synthetic tool failure")

    monkeypatch.setattr(adapter, "produce_sync", fail)
    try:
        with observe_execution(events.append), pytest.raises(KediExecutionError):
            runtime.run_main()
        assert not next(event for event in events if event.kind == "inside_model").safe
        failures = [event for event in events if event.kind == "exception"]
        assert [event.safe for event in failures] == [False, True]
    finally:
        asyncio.run(runtime.aclose())
