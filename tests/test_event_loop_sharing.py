"""Model calls that share one provider client must not run on separate live event loops.

Pydantic AI providers support one running event loop at a time: an HTTP client used on a loop
that has stopped is replaced, but a loop that is still running keeps sharing it, and pooled
connections are bound to the loop that opened them. When synchronous Kedi ran every adapter call
on its own short-lived loop, concurrent calls whose follow-up requests (retries, tool-loop steps,
delegation steps) reused another loop's connection failed with "bound to a different event loop" or
"Event loop is closed". These tests drive real OpenAIResponsesModel calls against a local fake
Responses server; no credentials or network access are needed.

Synchronous Kedi work that overlaps in time now shares one loop: operations on any thread, direct
adapter calls and Codex tool calls (design and invariants: docs/EVENT_LOOPS.md). A recording
FunctionModel asserts that deterministically (overlapping requests share one loop); the HTTP cases
confirm it end to end and run under a hard timeout, because the old failure could also hang.
"""

from __future__ import annotations

import asyncio
import contextvars
import itertools
import json
import sys
import threading
import time
from collections.abc import Iterator
from concurrent.futures import Future as ConcurrentFuture
from contextlib import contextmanager, nullcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.openai import OpenAIProvider

import kedi
from kedi.agent_adapter import PydanticAdapter, execution
from kedi.agent_adapter.approvals.policy import (
    ApprovalDecision,
    ApprovalPolicy,
    ApprovalRequest,
    resolve_tool_approval,
)
from kedi.agent_adapter.execution import _operation_event_loop, run_sync, use_async_owner_loop
from kedi.agent_adapter.hooks import HookDispatcher, HookDispatcherSnapshot
from kedi.agent_adapter.hooks.models import UserPromptSubmitRequest
from kedi.agent_adapter.tools.specs import ToolSpec
from kedi.agent_profile import AgentProfile, HookSettings
from kedi.agent_profile.mcp import McpServerSpec
from kedi.engine import SequentialEngine
from kedi.lang import compile_program, parse_program

LOOKUP = """@lookup(key: str) -> str:
  ###
  Look up a value.

  Args:
    key: The key to look up.
  ###
  > tool:
    risk: read_only
  = `"value-" + key`

"""


def _captures(count: int) -> str:
    return "\n= `[" + ", ".join(f"a{i}" for i in range(count)) + "]`\n"


INDEPENDENT = "\n".join(f">> Item {i}: [a{i}: str]" for i in range(8)) + _captures(8)
CHAIN = "\n".join(
    ">> Item 0: [a0: str]" if i == 0 else f">> After <a{i - 1}>: [a{i}: str]" for i in range(4)
) + _captures(4)
TOOL_LOOPS = (
    LOOKUP
    + "> profile: worker:\n  > instructions: Call lookup first.\n  > use: lookup\n\n> use: worker\n"
    + INDEPENDENT
)
DELEGATIONS = (
    "> profile: child:\n  > instructions: Do the task.\n\n"
    "> profile: coordinator:\n  > subagent: child\n  > max_agents: 8\n"
    "  > instructions: Delegate the task to child.\n\n> use: coordinator\n"
    + "\n".join(f">> Item {i}: [a{i}: str]" for i in range(4))
    + _captures(4)
)
NATIVE_TASKS = (
    LOOKUP
    + "> profile: child:\n  > instructions: Do the task.\n  > use: lookup\n\n"
    + "> profile: coordinator:\n  > subagent: child\n  > max_agents: 4\n\n"
    + "@run() -> list[str]:\n  > use: coordinator\n"
    + "".join(
        f"  > task [j{i}]: child:\n    >> Task {i}. The finding is [finding: str].\n"
        for i in range(4)
    )
    + "".join(f"  > await [r{i}]: j{i}\n" for i in range(4))
    + "  = `["
    + ", ".join(f"r{i}.output.finding" for i in range(4))
    + "]`\n\n= `run()`\n"
)


def _instance(schema: dict[str, Any], defs: dict[str, Any]) -> Any:
    """A minimal valid instance of a JSON schema, enough for Kedi's output tools."""
    if "$ref" in schema:
        return _instance(defs[schema["$ref"].rsplit("/", 1)[-1]], defs)
    for key in ("anyOf", "oneOf"):
        if key in schema:
            options = [o for o in schema[key] if o.get("type") != "null"] or schema[key]
            return _instance(options[0], defs)
    if "enum" in schema:
        return schema["enum"][0]
    kind = str(schema.get("type", "string"))
    if kind == "object" or "properties" in schema:
        properties = schema.get("properties", {})
        required = schema.get("required", list(properties))
        return {k: _instance(v, defs) for k, v in properties.items() if k in required}
    return {"integer": 1, "number": 1.0, "boolean": True, "array": []}.get(kind, "ok")


class _FakeResponses(BaseHTTPRequestHandler):
    """Keep-alive Responses API: calls `lookup` or `delegate_task` once when offered, then answers."""

    protocol_version = "HTTP/1.1"
    fail_first = 0
    requests = itertools.count(1)
    lock = threading.Lock()

    def log_message(self, format: str, *args: Any) -> None:
        pass

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        with self.lock:
            number = next(self.requests)
        if number <= self.fail_first:
            self._send(
                500 if number % 2 else 429, {"error": {"message": "busy", "type": "server_error"}}
            )
            return
        tools = {tool["name"]: tool for tool in body.get("tools", []) if "name" in tool}
        answered = any(
            isinstance(item, dict) and item.get("type") == "function_call_output"
            for item in body.get("input", [])
        )
        name = next((t for t in ("lookup", "delegate_task") if t in tools and not answered), None)
        name = name or next((t for t in tools if t.startswith("final_result")), None)
        time.sleep(0.3)  # keep concurrent calls overlapping
        if name is None:
            output = [
                {
                    "type": "message",
                    "id": f"msg_{number}",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "ok", "annotations": []}],
                }
            ]
        else:
            schema = tools[name]["parameters"]
            output = [
                {
                    "type": "function_call",
                    "id": f"fc_{number}",
                    "call_id": f"call_{number}",
                    "status": "completed",
                    "name": name,
                    "arguments": json.dumps(_instance(schema, schema.get("$defs", {}))),
                }
            ]
        self._send(
            200,
            {
                "id": f"resp_{number}",
                "object": "response",
                "created_at": 0,
                "model": "fake",
                "status": "completed",
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
                "output": output,
                "usage": {
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "total_tokens": 2,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens_details": {"reasoning_tokens": 0},
                },
            },
        )


@contextmanager
def _fake_model(fail_first: int = 0) -> Iterator[OpenAIResponsesModel]:
    handler = type(
        "Handler", (_FakeResponses,), {"fail_first": fail_first, "requests": itertools.count(1)}
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        provider = OpenAIProvider(
            base_url=f"http://127.0.0.1:{server.server_port}/v1", api_key="test"
        )
        yield OpenAIResponsesModel("fake", provider=provider)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def _run_http(
    source: str, model: OpenAIResponsesModel, *, host: bool = False, sequential: bool = False
) -> Any:
    runtime = compile_program(
        parse_program(source, source_path="loops.kedi"),
        adapter=PydanticAdapter(model),
        **({"engine": SequentialEngine()} if sequential else {}),
    )
    if not host:
        return runtime.run_main()

    async def async_host() -> Any:
        with use_async_owner_loop():
            return await asyncio.to_thread(runtime.run_main)

    return asyncio.run(async_host())


def _bounded(run: Any, seconds: float = 60.0) -> Any:
    """Run ``run()`` in a thread and fail instead of hanging the suite."""
    outcome: dict[str, Any] = {}

    def target() -> None:
        try:
            outcome["value"] = run()
        except BaseException as exc:  # noqa: BLE001
            outcome["error"] = exc

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(seconds)
    if worker.is_alive():
        raise AssertionError(f"Kedi run did not finish within {seconds} s")
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


class _Recorder:
    """A FunctionModel that records (loop, start, end) for every request."""

    def __init__(self) -> None:
        self.spans: list[tuple[asyncio.AbstractEventLoop, float, float]] = []
        self._lock = threading.Lock()

    async def respond(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        loop, start = asyncio.get_running_loop(), time.monotonic()
        await asyncio.sleep(0.1)  # concurrent requests overlap
        with self._lock:
            self.spans.append((loop, start, time.monotonic()))
        answered = any(part.part_kind == "tool-return" for m in messages for part in m.parts)
        tools = {tool.name: tool for tool in info.function_tools}
        for name in ("outer", "lookup", "delegate_task"):
            if name in tools and not answered:
                schema = tools[name].parameters_json_schema
                return ModelResponse(
                    parts=[ToolCallPart(name, _instance(schema, schema.get("$defs", {})))]
                )
        if info.output_tools:
            output = info.output_tools[0]
            schema = output.parameters_json_schema
            return ModelResponse(
                parts=[ToolCallPart(output.name, _instance(schema, schema.get("$defs", {})))]
            )
        return ModelResponse(parts=[TextPart("ok")])

    def overlapping_loops(self) -> int:
        """How many pairs of time-overlapping requests ran on different loops."""
        spans = sorted(self.spans, key=lambda span: span[1])
        return sum(
            1
            for i, (loop_a, _, end_a) in enumerate(spans)
            for loop_b, start_b, _ in spans[i + 1 :]
            if start_b < end_a and loop_a is not loop_b
        )


def _run_recorded(source: str, recorder: _Recorder, *, host: bool = False) -> Any:
    runtime = compile_program(
        parse_program(source, source_path="loops.kedi"),
        adapter=PydanticAdapter(FunctionModel(recorder.respond)),
    )
    if not host:
        return runtime.run_main()

    async def async_host() -> Any:
        with use_async_owner_loop():
            return await asyncio.to_thread(runtime.run_main)

    return asyncio.run(async_host())


CONCURRENT = pytest.mark.parametrize(
    ("source", "count"),
    [(INDEPENDENT, 8), (TOOL_LOOPS, 8), (DELEGATIONS, 4)],
    ids=["templates", "tool-loops", "delegations"],
)


# Deterministic: overlapping requests on different loops are counted, not raced.


@CONCURRENT
def test_overlapping_requests_share_one_loop(source: str, count: int) -> None:
    recorder = _Recorder()
    assert _run_recorded(source, recorder) == ["ok"] * count
    assert recorder.overlapping_loops() == 0


@CONCURRENT
def test_async_host_keeps_overlapping_requests_on_its_owner_loop(source: str, count: int) -> None:
    recorder = _Recorder()
    assert _run_recorded(source, recorder, host=True) == ["ok"] * count
    assert recorder.overlapping_loops() == 0


def test_native_tasks_run_their_requests_on_the_coordinator_loop() -> None:
    recorder = _Recorder()
    assert _run_recorded(NATIVE_TASKS, recorder) == ["ok"] * 4
    assert recorder.overlapping_loops() == 0


# Over HTTP: synchronous runs and async hosts, the same model reused across runs.


@pytest.mark.parametrize(
    ("source", "fail_first", "expected"),
    [
        (INDEPENDENT, 12, ["ok"] * 8),
        (TOOL_LOOPS, 0, ["ok"] * 8),
        (DELEGATIONS, 0, ["ok"] * 4),
    ],
    ids=["retries", "tool-loops", "delegations"],
)
@pytest.mark.parametrize("host", [False, True], ids=["sync", "async-host"])
def test_concurrent_calls_over_http(
    source: str, fail_first: int, expected: list[str], host: bool
) -> None:
    with _fake_model(fail_first=fail_first) as model:
        for _ in range(2):  # the same model in a second run, after the first loop closed
            assert _bounded(lambda: _run_http(source, model, host=host)) == expected


# Calls that do not overlap, and native tasks on one coordinator loop.


def test_sequential_calls_survive_retries_across_loops() -> None:
    with _fake_model(fail_first=2) as model:
        assert _run_http(INDEPENDENT, model, sequential=True) == ["ok"] * 8


def test_dependent_chain_survives_retries_across_loops() -> None:
    with _fake_model(fail_first=2) as model:
        assert _run_http(CHAIN, model) == ["ok"] * 4


def test_native_tasks_with_tool_loops_share_the_coordinator_loop() -> None:
    with _fake_model(fail_first=2) as model:
        assert _run_http(NATIVE_TASKS, model) == ["ok"] * 4


# The operation scope.


class _CountingLoop(execution.OwnedAsyncLoop):
    started = 0

    def __init__(self) -> None:
        type(self).started += 1
        super().__init__()


@pytest.fixture
def counting_loops(monkeypatch: pytest.MonkeyPatch) -> type[_CountingLoop]:
    counter = type("Counter", (_CountingLoop,), {"started": 0})
    monkeypatch.setattr(execution, "OwnedAsyncLoop", counter)
    return counter


def test_deterministic_operation_starts_no_event_loop(counting_loops: type[_CountingLoop]) -> None:
    runtime = compile_program(parse_program("= `1 + 1`\n"))
    assert runtime.run_main() == 2
    assert counting_loops.started == 0


def test_operation_loop_closes_with_the_operation(counting_loops: type[_CountingLoop]) -> None:
    loops = []
    for _ in range(2):
        recorder = _Recorder()
        assert _run_recorded(INDEPENDENT, recorder) == ["ok"] * 8
        run_loops = {id(loop): loop for loop, _, _ in recorder.spans}
        assert len(run_loops) == 1
        (loop,) = run_loops.values()
        assert loop.is_closed()
        loops.append(loop)
    assert loops[0] is not loops[1]
    assert counting_loops.started == 2


def test_nested_procedure_calls_reuse_the_operation_loop(tmp_path: Any) -> None:
    (tmp_path / "helper.kedi").write_text(
        "@remote() -> str:\n  >> Remote [r: str]\n  = `r`\n\n> export:\n  remote\n",
        encoding="utf-8",
    )
    main = tmp_path / "main.kedi"
    main.write_text(
        "> import: helper:\n  remote\n\n"
        "@local() -> str:\n  >> Local [l: str]\n  = `l`\n\n"
        "[a: str] = `local()`\n[b: str] = `remote()`\n>> Top [c: str]\n= `[a, b, c]`\n",
        encoding="utf-8",
    )
    recorder = _Recorder()
    runtime = compile_program(
        parse_program(main.read_text(encoding="utf-8"), source_path=str(main)),
        adapter=PydanticAdapter(FunctionModel(recorder.respond)),
    )
    assert runtime.run_main() == ["ok"] * 3
    assert len(recorder.spans) == 3
    assert len({id(loop) for loop, _, _ in recorder.spans}) == 1


def test_async_host_creates_no_operation_loop(counting_loops: type[_CountingLoop]) -> None:
    recorder = _Recorder()
    assert _run_recorded(INDEPENDENT, recorder, host=True) == ["ok"] * 8
    assert counting_loops.started == 0


def test_interrupted_wait_cancels_its_task_and_awaits_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started, cleaned = threading.Event(), threading.Event()

    async def slow() -> None:
        started.set()
        try:
            await asyncio.sleep(30)
        finally:
            await asyncio.sleep(0.05)  # cleanup that still needs the loop
            cleaned.set()

    def interrupted(self: ConcurrentFuture, timeout: float | None = None) -> Any:
        started.wait(5)
        raise KeyboardInterrupt

    with _operation_event_loop():
        with monkeypatch.context() as patch:
            patch.setattr(ConcurrentFuture, "result", interrupted)
            with pytest.raises(KeyboardInterrupt):
                run_sync(slow())
        assert cleaned.is_set()


def test_operation_loop_rejects_use_after_close() -> None:
    scope = execution._OperationLoop()
    scope.loop()
    scope.close()
    with pytest.raises(RuntimeError, match="closed"):
        scope.loop()


@pytest.mark.parametrize("in_operation", [True, False], ids=["operation", "direct"])
def test_sync_code_on_the_shared_loop_keeps_a_nested_loop(in_operation: bool) -> None:
    async def inner() -> asyncio.AbstractEventLoop:
        return asyncio.get_running_loop()

    async def outer() -> tuple[asyncio.AbstractEventLoop, asyncio.AbstractEventLoop]:
        # Synchronous code running inline on the shared loop must not block it.
        return asyncio.get_running_loop(), run_sync(inner())

    with _operation_event_loop() if in_operation else nullcontext():
        shared, nested = run_sync(outer())
    assert shared is not nested


@pytest.mark.parametrize("mode", ["direct", "operation", "nested"])
def test_async_tool_running_a_sync_operation_inline_does_not_deadlock(mode: str) -> None:
    recorder = _Recorder()
    source = "\n".join(f">> Item {i}: [a{i}: str]" for i in range(2)) + _captures(2)
    inner = compile_program(
        parse_program(source), adapter=PydanticAdapter(FunctionModel(recorder.respond))
    )

    async def research() -> str:
        """Research the topic."""
        if mode == "nested":  # sync code, then a nested call's loop runs the operation inline

            async def deeper() -> Any:
                return inner.run_main()

            return str(run_sync(deeper()))
        return str(inner.run_main())  # blocks the loop running this tool

    def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if any(part.part_kind == "tool-return" for m in messages for part in m.parts):
            return ModelResponse(parts=[TextPart("done")])
        return ModelResponse(parts=[ToolCallPart("research", {})])

    adapter = PydanticAdapter(FunctionModel(respond))
    tool = ToolSpec(fn=research, name="research", description="Research", risk="read_only")

    def run() -> str:
        with adapter.tool_scope([tool]):
            with _operation_event_loop() if mode == "operation" else nullcontext():
                return adapter.invoke_sync(prompt="Go")

    assert _bounded(run, 20) == "done"
    assert len({id(loop) for loop, _, _ in recorder.spans}) == 1  # the inline operation's loop


def test_a_last_release_on_the_shared_loop_itself_still_closes_it() -> None:
    shared = execution._SharedLoop()
    loop = shared.hold()

    async def release() -> None:
        shared.release(loop)

    asyncio.run_coroutine_threadsafe(release(), loop).result(5)
    fresh = shared.hold()  # waits for the closing loop
    assert loop.is_closed() and fresh is not loop
    shared.release(fresh)


def test_holds_from_before_a_fork_are_stale_in_the_child() -> None:
    shared = execution._SharedLoop()
    parent = shared.hold()
    owned = shared._owned
    assert owned is not None
    shared._reset()  # runs in a forked child, which has no loop thread
    shared.release(parent)  # ignored there: held before the fork
    child = shared.hold()
    assert child is not parent and child.is_running()
    shared.release(child)
    assert child.is_closed()
    owned.close()  # this process did not fork: stop the parent's loop


# Decision models close their per-loop client after each synchronous call.


class _DecisionModel(FunctionModel):
    """Mimics kedi-typesafe: one client per event loop, closed by ``aclose_current``."""

    kedi_prompt_mode = "decision"

    def __init__(self) -> None:
        self.clients: dict[int, dict[str, bool]] = {}
        self.used_after_close = 0
        super().__init__(self._respond)

    async def _respond(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        client = self.clients.setdefault(id(asyncio.get_running_loop()), {"closed": False})
        await asyncio.sleep(0.1)
        if client["closed"]:
            self.used_after_close += 1
        output = info.output_tools[0]
        schema = output.parameters_json_schema
        return ModelResponse(
            parts=[ToolCallPart(output.name, _instance(schema, schema.get("$defs", {})))]
        )

    async def aclose_current(self) -> None:
        client = self.clients.pop(id(asyncio.get_running_loop()), None)
        if client is not None:
            client["closed"] = True


@pytest.mark.parametrize("host", [False, True], ids=["sync", "async-host"])
def test_concurrent_decisions_close_their_shared_client_once(host: bool) -> None:
    model = _DecisionModel()
    source = "\n".join(f">> Item {i} is a duplicate [a{i}: bool]" for i in range(8)) + _captures(8)
    runtime = compile_program(parse_program(source), adapter=PydanticAdapter(model))
    if host:

        async def async_host() -> Any:
            with use_async_owner_loop():
                return await asyncio.to_thread(runtime.run_main)

        result = asyncio.run(async_host())
    else:
        result = runtime.run_main()
    assert result == [True] * 8
    assert model.used_after_close == 0
    assert model.clients == {}  # closed when the operation ended


@pytest.mark.parametrize("host", [False, True], ids=["sync", "async-host"])
def test_overlapping_operations_close_a_shared_decision_client_once(host: bool) -> None:
    model = _DecisionModel()
    single = ">> Item 0 is a duplicate [a0: bool]" + _captures(1)
    chain = "\n".join(
        ">> Item 0 is a duplicate [a0: bool]" if i == 0 else f">> After <a{i - 1}> [a{i}: bool]"
        for i in range(4)
    ) + _captures(4)
    first, second = (
        compile_program(parse_program(source), adapter=PydanticAdapter(model))
        for source in (single, chain)
    )

    async def operations() -> Any:
        async def later() -> Any:
            await asyncio.sleep(0.05)  # ends while the chain has a call in flight
            return await asyncio.to_thread(first.run_main)

        with use_async_owner_loop() if host else nullcontext():
            return await asyncio.gather(later(), asyncio.to_thread(second.run_main))

    assert asyncio.run(operations()) == [[True], [True] * 4]
    assert model.used_after_close == 0
    assert model.clients == {}  # closed once the last operation on the loop ended


class _Duplicate(BaseModel):
    duplicate: bool


@pytest.mark.parametrize("host", [False, True], ids=["sync", "async-host"])
def test_overlapping_direct_decisions_close_a_shared_client_once(host: bool) -> None:
    model = _DecisionModel()
    adapter = PydanticAdapter(model)

    async def calls() -> Any:
        async def call(delay: float) -> Any:
            await asyncio.sleep(delay)
            return await asyncio.to_thread(
                adapter.produce_sync, "Is the item a duplicate?", output_type=_Duplicate
            )

        with use_async_owner_loop() if host else nullcontext():
            return await asyncio.gather(call(0), call(0.05))

    asyncio.run(calls())
    assert model.used_after_close == 0
    assert model.clients == {}


def test_a_decision_on_a_nested_loop_closes_its_client_itself() -> None:
    model = _DecisionModel()
    adapter = PydanticAdapter(model)

    async def inline() -> Any:  # a sync call inside a coroutine runs on a loop of its own
        return adapter.produce_sync("Is the item a duplicate?", output_type=_Duplicate)

    assert run_sync(inline()).duplicate is True
    assert model.clients == {}


def test_cleanup_on_another_loop_is_not_deferred() -> None:
    async def defer() -> bool:
        return execution._defer_loop_cleanup(1, asyncio.sleep)

    with _operation_event_loop():
        # A call on its own loop (not the operation's) closes its client itself.
        assert asyncio.run(defer()) is False


# Sync tools that make nested model calls inside one operation. Four parallel calls: eight
# would exhaust Kedi's eight engine workers, which hangs on stable too (separate issue).

NESTED_TOOLS = (
    LOOKUP
    + "@inner(key: str) -> str:\n  > use: lookup\n  >> Inner <key> [x: str]\n  = `x`\n\n"
    + "@outer(key: str) -> str:\n  ###\n  Ask the inner model.\n\n  Args:\n    key: The key.\n  ###\n"
    + "  > tool:\n    risk: read_only\n  = `inner(key)`\n\n"
    + "> profile: worker:\n  > instructions: Call outer first.\n  > use: outer\n\n> use: worker\n"
    + "\n".join(f">> Item {i}: [a{i}: str]" for i in range(4))
    + _captures(4)
)


def test_nested_tool_model_calls_share_the_operation_loop() -> None:
    recorder = _Recorder()
    assert _bounded(lambda: _run_recorded(NESTED_TOOLS, recorder)) == ["ok"] * 4
    assert len(recorder.spans) > 8  # the four tool calls made nested model calls
    assert recorder.overlapping_loops() == 0


# Other entry points that must use the operation's loop.


def test_interactive_fragments_each_run_on_one_closed_loop() -> None:
    recorder = _Recorder()
    fragment = "\n".join(f">> Item {i}: [a{i}: str]" for i in range(4)) + "\n"
    loops = []
    with kedi.interactive(adapter=PydanticAdapter(FunctionModel(recorder.respond))) as session:
        for _ in range(2):
            before = len(recorder.spans)
            session.execute(fragment)
            fragment_loops = {id(loop): loop for loop, _, _ in recorder.spans[before:]}
            assert len(fragment_loops) == 1
            (loop,) = fragment_loops.values()
            assert loop.is_closed()
            loops.append(loop)
    assert loops[0] is not loops[1]


def _prompt_event() -> UserPromptSubmitRequest:
    return UserPromptSubmitRequest(
        run_id="run", sequence=0, adapter_shortname="fake", content="prompt"
    )


def test_awaitable_from_a_sync_hook_runs_on_the_operation_loop() -> None:
    seen: list[asyncio.AbstractEventLoop] = []

    async def record(event: Any) -> None:
        seen.append(asyncio.get_running_loop())

    # A synchronous handler that returns an awaitable.
    dispatcher = HookDispatcher(
        HookDispatcherSnapshot.compose(
            None, HookSettings(user_prompt_submit=(lambda event: record(event),))
        )
    )
    with _operation_event_loop():
        expected = execution._shared_execution_loop()
        dispatcher.dispatch_sync(_prompt_event())
    assert seen == [expected]


def test_awaitable_from_a_sync_approval_handler_runs_on_the_operation_loop() -> None:
    seen: list[asyncio.AbstractEventLoop] = []

    async def decide(call: Any) -> ApprovalDecision:
        seen.append(asyncio.get_running_loop())
        return ApprovalDecision.allow()

    request = ApprovalRequest.create(
        tool_name="write_draft",
        arguments={},
        risk="mutating",
        adapter_shortname="test",
        description="Write a scratch draft",
        metadata=None,
    )
    with _operation_event_loop():
        expected = execution._shared_execution_loop()
        decision = resolve_tool_approval(ApprovalPolicy.dynamic(lambda call: decide(call)), request)
    assert decision.decision == "allow"
    assert seen == [expected]


def test_codegen_inside_an_operation_uses_the_operation_loop(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from kedi.codegen import runner as runner_module
    from kedi.codegen.mock_agents import MockPassingAgent

    seen: list[asyncio.AbstractEventLoop] = []

    async def record() -> None:
        seen.append(asyncio.get_running_loop())

    class LoopRecordingAgent(MockPassingAgent):
        def generate_tests(self, *args: Any, **kwargs: Any) -> str:
            run_sync(record())
            return super().generate_tests(*args, **kwargs)

        def generate_impl(self, *args: Any, **kwargs: Any) -> str:
            run_sync(record())
            return super().generate_impl(*args, **kwargs)

    monkeypatch.setattr(runner_module, "_select_agent", lambda kind, model: LoopRecordingAgent())
    source = tmp_path / "auto.kedi"
    source.write_text("@f() -> str:\n  > auto:\n    Return ok.\n", encoding="utf-8")
    with _operation_event_loop():
        expected = execution._shared_execution_loop()
        assert runner_module.run_codegen(source, quiet=True, use_cache=False)
        assert not expected.is_closed()
    assert seen == [expected, expected]
    assert expected.is_closed()


def test_mcp_tool_works_in_consecutive_operations(monkeypatch: pytest.MonkeyPatch) -> None:
    server = FastMCP("reader")
    tool_calls: list[str] = []
    model_loops: list[asyncio.AbstractEventLoop] = []

    @server.tool()
    async def read() -> str:
        tool_calls.append("read")
        return "value"

    monkeypatch.setattr(
        "kedi.agent_adapter.adapters.pydantic.tools._mcp_toolset_from_config",
        lambda name, config: MCPToolset(server),
    )

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        model_loops.append(asyncio.get_running_loop())
        if any(part.part_kind == "tool-return" for m in messages for part in m.parts):
            return ModelResponse(parts=[TextPart("done")])
        return ModelResponse(parts=[ToolCallPart(info.function_tools[0].name, {})])

    adapter = PydanticAdapter(FunctionModel(respond))
    profile = AgentProfile(
        mcp_servers=(McpServerSpec(command="reader"),), approval=ApprovalPolicy.allow()
    )
    for _ in range(2):  # a session opened in the first operation must not leak into the second
        with _operation_event_loop(), adapter.profile_override(profile):
            assert adapter.invoke_sync(prompt="Read the value.") == "done"
    assert tool_calls == ["read", "read"]
    first, second = model_loops[:2], model_loops[2:]
    assert len({id(loop) for loop in first}) == len({id(loop) for loop in second}) == 1
    assert first[0] is not second[0]
    assert first[0].is_closed() and second[0].is_closed()


# Work that starts outside one operation's context still shares one loop while it overlaps.


def _concurrently(fn: Any, items: list[Any], seconds: float = 60.0) -> list[Any]:
    """Call ``fn`` on every item at once, each from its own thread; fail instead of hanging."""
    barrier = threading.Barrier(len(items))
    results: dict[int, Any] = {}
    errors: list[BaseException] = []

    def target(index: int, item: Any) -> None:
        try:
            barrier.wait(5)
            results[index] = fn(item)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [
        threading.Thread(target=target, args=(i, item), daemon=True) for i, item in enumerate(items)
    ]
    for thread in threads:
        thread.start()
    deadline = time.monotonic() + seconds
    for thread in threads:
        thread.join(max(0.0, deadline - time.monotonic()))
    if any(thread.is_alive() for thread in threads):
        raise AssertionError(f"Concurrent calls did not finish within {seconds} s")
    if errors:
        raise errors[0]
    return [results[i] for i in range(len(items))]


TWO_TOOL_LOOPS = (
    LOOKUP
    + "> profile: worker:\n  > instructions: Call lookup first.\n  > use: lookup\n\n> use: worker\n"
    + "\n".join(f">> Item {i}: [a{i}: str]" for i in range(2))
    + _captures(2)
)


def _runtimes(source: str, model: Any, count: int) -> list[Any]:
    return [
        compile_program(
            parse_program(source, source_path="loops.kedi"), adapter=PydanticAdapter(model)
        )
        for _ in range(count)
    ]


def test_concurrent_operations_from_user_threads_share_one_loop() -> None:
    recorder = _Recorder()
    source = "\n".join(f">> Item {i}: [a{i}: str]" for i in range(2)) + _captures(2)
    runtimes = _runtimes(source, FunctionModel(recorder.respond), 4)
    assert _concurrently(lambda runtime: runtime.run_main(), runtimes) == [["ok", "ok"]] * 4
    assert recorder.overlapping_loops() == 0


def test_concurrent_direct_adapter_calls_share_one_loop() -> None:
    recorder = _Recorder()
    adapter = PydanticAdapter(FunctionModel(recorder.respond))
    assert (
        _concurrently(lambda i: adapter.invoke_sync(prompt=f"Item {i}"), [0, 1, 2, 3]) == ["ok"] * 4
    )
    assert recorder.overlapping_loops() == 0


def test_codex_tool_bridge_runs_on_the_operation_loop() -> None:
    from kedi.agent_adapter.adapters.codex.app_server import CodexAppServerClient

    client = CodexAppServerClient((sys.executable,))
    seen: list[asyncio.AbstractEventLoop] = []

    async def record() -> str:
        seen.append(asyncio.get_running_loop())
        return "ok"

    try:
        with _operation_event_loop():
            expected = execution._shared_execution_loop()
            assert client._await_sync_result(contextvars.copy_context(), record()) == "ok"
    finally:
        client.close()
    assert seen == [expected]


def test_codex_tool_bridge_and_direct_calls_share_one_loop() -> None:
    from kedi.agent_adapter.adapters.codex.app_server import CodexAppServerClient

    client = CodexAppServerClient((sys.executable,))
    recorder = _Recorder()
    adapter = PydanticAdapter(FunctionModel(recorder.respond))
    seen: list[asyncio.AbstractEventLoop] = []
    started, finished = threading.Event(), threading.Event()

    async def tool() -> str:
        seen.append(asyncio.get_running_loop())
        started.set()
        while not finished.is_set():
            await asyncio.sleep(0.01)
        return "ok"

    def bridge(_: Any) -> Any:
        return client._await_sync_result(contextvars.copy_context(), tool())

    try:
        bridged = threading.Thread(target=bridge, args=(None,), daemon=True)
        bridged.start()
        assert started.wait(5)
        assert adapter.invoke_sync(prompt="Item") == "ok"  # while the tool is running
    finally:
        finished.set()
        client.close()
    assert {id(loop) for loop, _, _ in recorder.spans} == {id(seen[0])}


# The same over HTTP, where a second loop breaks the shared connection pool.


def test_concurrent_operations_from_user_threads_over_http() -> None:
    with _fake_model() as model:
        runtimes = _runtimes(TWO_TOOL_LOOPS, model, 4)
        assert _concurrently(lambda runtime: runtime.run_main(), runtimes) == [["ok", "ok"]] * 4


def test_concurrent_direct_adapter_calls_over_http() -> None:
    with _fake_model(fail_first=4) as model:
        adapter = PydanticAdapter(model)
        assert (
            _concurrently(lambda i: adapter.invoke_sync(prompt=f"Item {i}"), [0, 1, 2, 3])
            == ["ok"] * 4
        )


def test_codex_tool_calls_and_direct_calls_over_http() -> None:
    from kedi.agent_adapter.adapters.codex.app_server import CodexAppServerClient

    client = CodexAppServerClient((sys.executable,))
    with _fake_model(fail_first=4) as model:
        adapter = PydanticAdapter(model)

        def call(bridged: bool) -> str:
            if not bridged:
                return adapter.invoke_sync(prompt="Direct")
            return client._await_sync_result(
                contextvars.copy_context(), adapter.invoke(prompt="Tool")
            )

        try:
            assert _concurrently(call, [True, True, False, False]) == ["ok"] * 4
        finally:
            client.close()


def test_codex_tool_bridge_under_an_async_host_uses_the_host_loop() -> None:
    from kedi.agent_adapter.adapters.codex.app_server import CodexAppServerClient

    client = CodexAppServerClient((sys.executable,))
    seen: list[asyncio.AbstractEventLoop] = []

    async def record() -> str:
        seen.append(asyncio.get_running_loop())
        return "ok"

    async def host() -> asyncio.AbstractEventLoop:
        with use_async_owner_loop():
            context = contextvars.copy_context()
            assert await asyncio.to_thread(client._await_sync_result, context, record()) == "ok"
        return asyncio.get_running_loop()

    try:
        assert seen == [asyncio.run(host())]
        assert client._tool_loop is None  # no shared loop held for the host's tools
    finally:
        client.close()


def test_codex_sync_turn_inline_on_a_running_loop_keeps_a_tool_loop_of_its_own(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kedi.agent_adapter.adapters.codex import app_server

    client = app_server.CodexAppServerClient((sys.executable,))
    seen: list[asyncio.AbstractEventLoop] = []

    async def tool_call() -> str:
        seen.append(asyncio.get_running_loop())
        return "ok"

    def request(method: str, params: Any, **kwargs: Any) -> dict[str, Any]:
        if method == "turn/start":  # the App Server calls the tool while the turn runs
            bridge = client._thread_tools["thread"]["tool"].await_result
            for _ in range(2):
                assert bridge(contextvars.copy_context(), tool_call()) == "ok"
            return {"turn": {"id": "turn"}}
        return {}

    state = app_server._TurnState(messages=["{}"])
    state.done.set()
    monkeypatch.setattr(client, "_start_thread", lambda params: {"thread": {"id": "thread"}})
    monkeypatch.setattr(client, "request", request)
    monkeypatch.setattr(client, "_turn_state", lambda turn_id: state)

    async def inline() -> asyncio.AbstractEventLoop:
        with pytest.raises(app_server.CodexAppServerProtocolError, match="without turn data"):
            client.run_stateless_turn_sync(  # blocks this loop until the turn ends
                thread_params={},
                turn_params={},
                tools={"tool": lambda: None},
                tool_context=contextvars.copy_context(),
                turn_timeout=1,
            )
        return asyncio.get_running_loop()

    try:
        blocked = asyncio.run(inline())
        assert seen[0] is seen[1] and seen[0] is not blocked
        assert not execution._SHARED_LOOP.runs(seen[0])
    finally:
        client.close()
    assert seen[0].is_closed()
