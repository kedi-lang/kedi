"""Tests for non-blocking (promise-pipelining) template execution.

Covers the core mechanism: independent templates run concurrently, dependent chains pipeline
as continuations, the loud KediPromise surfaces leaks, and sequential vs parallel produce
identical results.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future
from typing import Any

import pytest

from kedi.engine import SequentialEngine, ThreadEngine
from kedi.lang import compile_program, parse_program
from kedi.promise import KediPromise, KediPromiseLeak, force, when_all
from tests.mock_adapter import MockAdapter


class ConcurrencyProbeAdapter(MockAdapter):
    """A mock adapter that sleeps per call and records the peak number of in-flight calls."""

    def __init__(self, delay: float = 0.1) -> None:
        super().__init__()
        self._delay = delay
        self._inflight = 0
        self.max_inflight = 0
        self._probe_lock = threading.Lock()

    def produce_sync(self, *, template: str, output_schema: Any = None, **kwargs: Any) -> Any:
        with self._probe_lock:
            self._inflight += 1
            self.max_inflight = max(self.max_inflight, self._inflight)
        try:
            time.sleep(self._delay)
            return super().produce_sync(template=template, output_schema=output_schema, **kwargs)
        finally:
            with self._probe_lock:
                self._inflight -= 1


def _run(src: str, adapter: MockAdapter, engine: Any) -> Any:
    return compile_program(parse_program(src), adapter=adapter, engine=engine).run_main()


# --------------------------------------------------------------------------------------------
# Promise primitives
# --------------------------------------------------------------------------------------------


def test_promise_resolve_and_memoize() -> None:
    fut: Future = Future()
    fut.set_result({"x": "hello"})
    p = KediPromise(fut, "x")
    assert p.resolve() == "hello"
    assert p.resolve() == "hello"  # memoized


def test_force_passthrough_and_resolve() -> None:
    fut: Future = Future()
    fut.set_result({"k": 7})
    assert force(KediPromise(fut, "k")) == 7
    assert force("plain") == "plain"


def test_promise_is_loud_on_value_use() -> None:
    fut: Future = Future()
    fut.set_result({"x": "v"})
    p = KediPromise(fut, "x")
    with pytest.raises(KediPromiseLeak):
        str(p)
    with pytest.raises(KediPromiseLeak):
        bool(p)
    with pytest.raises(KediPromiseLeak):
        p.some_attribute  # noqa: B018
    with pytest.raises(KediPromiseLeak):
        len(p)
    with pytest.raises(KediPromiseLeak):
        list(p)
    with pytest.raises(KediPromiseLeak):
        p + "x"
    # repr is safe and never resolves.
    assert "KediPromise" in repr(p)


def test_promise_map_is_lazy_and_chains() -> None:
    fut: Future = Future()
    p = KediPromise(fut, "n")
    derived = p.map(lambda v: v * 2)
    fut.set_result({"n": 21})
    assert derived.resolve() == 42


def test_when_all_fires_once() -> None:
    calls = []
    f1: Future = Future()
    f2: Future = Future()
    when_all([f1, f2], lambda: calls.append(1))
    assert calls == []
    f1.set_result(None)
    assert calls == []
    f2.set_result(None)
    assert calls == [1]


def test_when_all_empty_fires_immediately() -> None:
    calls = []
    when_all([], lambda: calls.append(1))
    assert calls == [1]


# --------------------------------------------------------------------------------------------
# End-to-end concurrency
# --------------------------------------------------------------------------------------------

INDEPENDENT_SRC = "@gen():\n  >> alpha [a]\n  >> beta [b]\n  = <a>-<b>\n\n= <gen()>\n"
CHAINS_SRC = (
    "@gen():\n"
    "  >> seed one [a]\n"
    "  >> from <a> make [b]\n"
    "  >> seed two [c]\n"
    "  >> from <c> make [d]\n"
    "  = <b>+<d>\n\n"
    "= <gen()>\n"
)


def test_independent_templates_run_concurrently() -> None:
    adapter = ConcurrencyProbeAdapter(delay=0.15)
    out = _run(INDEPENDENT_SRC, adapter, ThreadEngine(max_workers=4))
    assert out == "a-value-b-value"
    assert adapter.max_inflight >= 2, "independent templates should overlap"


def test_two_chains_pipeline_in_parallel() -> None:
    # A->B and C->D: the two chain heads (A, C) overlap; each chain serializes internally.
    adapter = ConcurrencyProbeAdapter(delay=0.15)
    out = _run(CHAINS_SRC, adapter, ThreadEngine(max_workers=4))
    assert out == "b-value+d-value"
    assert adapter.max_inflight >= 2, "independent chains should overlap"


def test_sequential_does_not_overlap() -> None:
    adapter = ConcurrencyProbeAdapter(delay=0.02)
    out = _run(INDEPENDENT_SRC, adapter, SequentialEngine())
    assert out == "a-value-b-value"
    assert adapter.max_inflight == 1


@pytest.mark.parametrize(
    "src,expected",
    [(INDEPENDENT_SRC, "a-value-b-value"), (CHAINS_SRC, "b-value+d-value")],
)
def test_sequential_parallel_equivalence(src: str, expected: str) -> None:
    seq = _run(src, MockAdapter(), SequentialEngine())
    par = _run(src, MockAdapter(), ThreadEngine(max_workers=4))
    assert seq == par == expected


def test_unconsumed_template_still_runs_under_drain() -> None:
    # 'a' is never referenced after production; the drain must still execute its job.
    src = "@gen():\n  >> alpha [a]\n  >> beta [b]\n  = <b>\n\n= <gen()>\n"
    adapter = ConcurrencyProbeAdapter(delay=0.01)
    out = _run(src, adapter, ThreadEngine(max_workers=4))
    assert out == "b-value"
    # Both templates ran (the unconsumed [a] block included).
    assert len(adapter.model_calls) == 2


def test_concurrent_tool_using_templates_have_isolated_scopes() -> None:
    # Two independent tool-registering templates run concurrently; ToolScope's ContextVar +
    # copy_context must give each its OWN scope (each call sees exactly one tool), proving tool
    # templates can pipeline without serialization.
    from types import SimpleNamespace

    from kedi.agent_adapter.tools.specs import ToolSpec

    class ScopeProbeAdapter(ConcurrencyProbeAdapter):
        def register_tool(self, tool: ToolSpec) -> None:
            proc = tool.proc
            name = f"{proc.name}:{proc.loc.start_line}" if proc and proc.loc else tool.name
            self._tool_scope.register_tool(SimpleNamespace(name=name))

    src = (
        "@bar():\n    = top\n\n"
        "> use: bar\n\n"
        "@foo():\n"
        "    @bar():\n        = inner\n"
        "    > use: bar\n"
        "    >> inner [x]\n"
        "    = done\n\n"
        "[done] = <foo()>\n"
        ">> outer [y]\n"
    )
    adapter = ScopeProbeAdapter(delay=0.1)
    _run(src, adapter, ThreadEngine(max_workers=4))
    # Each call saw exactly one tool, and the right one — scopes did not leak across workers.
    artifact_tools = (
        "read_artifact",
        "search_artifacts",
        "release_artifact",
        "run_artifact_code",
    )
    assert sorted(adapter.tool_calls) == [
        ("bar:1", *artifact_tools),
        ("bar:7", *artifact_tools),
    ]


def test_call_with_output_evaluates_once() -> None:
    # A CallSegment co-occurring with an OutputSegment in a render context must evaluate the call
    # exactly once (was twice: once to build the prompt, once to render). Regression for the
    # evaluate_call double-eval. inner() runs one template; the return adds [w] => 2 calls total.
    adapter = MockAdapter()
    src = "@inner():\n  >> gen [v]\n  = <v>\n\n@outer():\n  = <inner()> and [w]\n\n= <outer()>\n"
    out = compile_program(parse_program(src), adapter=adapter, engine=SequentialEngine()).run_main()
    assert out == "v-value and w-value"
    assert len(adapter.model_calls) == 2  # inner once + w once (not inner twice)


class FailingAdapter(MockAdapter):
    def produce_sync(self, *, template: str, output_schema: Any = None, **kwargs: Any) -> Any:
        raise RuntimeError("boom")


def test_template_failure_surfaces_under_parallel() -> None:
    from kedi.errors import KediExecutionError

    src = "@gen():\n  >> alpha [a]\n  = <a>\n\n= <gen()>\n"
    with pytest.raises(KediExecutionError):
        _run(src, FailingAdapter(), ThreadEngine(max_workers=2))
