"""Model calls that share one provider client must not run on separate live event loops.

Pydantic AI providers support one running event loop at a time: an HTTP client used on a loop
that has stopped is replaced, but a loop that is still running keeps sharing it, and pooled
connections are bound to the loop that opened them. Synchronous Kedi runs every adapter call on
its own short-lived loop, so concurrent calls whose follow-up requests (retries, tool-loop steps,
delegation steps) reuse another loop's connection fail with "bound to a different event loop" or
"Event loop is closed". These tests drive real OpenAIResponsesModel calls against a local fake
Responses server; no credentials or network access are needed.

Each synchronous operation now runs its async work on one event loop. A recording FunctionModel
asserts that deterministically (overlapping requests share one loop); the HTTP cases confirm it
end to end and run under a hard timeout, because the old failure could also hang.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import threading
import time
from collections.abc import Iterator
from concurrent.futures import Future as ConcurrentFuture
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.openai import OpenAIProvider

from kedi.agent_adapter import PydanticAdapter
from kedi.agent_adapter import execution
from kedi.agent_adapter.execution import operation_event_loop, run_sync, use_async_owner_loop
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

    with operation_event_loop():
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


def test_sync_code_on_the_operation_loop_keeps_a_nested_loop() -> None:
    async def inner() -> asyncio.AbstractEventLoop:
        return asyncio.get_running_loop()

    async def outer() -> tuple[asyncio.AbstractEventLoop, asyncio.AbstractEventLoop]:
        # Synchronous code running inline on the operation loop must not block it.
        return asyncio.get_running_loop(), run_sync(inner())

    with operation_event_loop():
        operation, nested = run_sync(outer())
    assert operation is not nested


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


def test_cleanup_on_another_loop_is_not_deferred() -> None:
    async def defer() -> bool:
        return execution.defer_to_operation_end(1, asyncio.sleep)

    with operation_event_loop():
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
