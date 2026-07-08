"""Regression tests for the Tier-A concurrency fixes (PR #50 review cycle).

Each test pins one merge-gate fix so a future change can't silently reintroduce the bug:

- A1  ``KEDI_PARALLEL`` parsing (covered in ``test_nonblocking_coverage.py``).
- A2  JobManager retry classifies on the *original* provider error (unwraps ``__kedi_cause__``).
- A3  the depth-0 drain runs even when a procedure body raises, without masking the body error.
- A4  a re-entrant ``schedule`` honors not-yet-ready deps instead of running inline (no deadlock).
- A5  a cancelled Future settles ``map``/``drain`` loudly instead of hanging forever.
- A6  a failed ``pool.submit`` surfaces on the result Future instead of leaving it pending.
- A7  the value-env is snapshot by value at schedule time (later writes don't leak into a worker).
- A8  the interpreter-routing self-check runs once in ``ThreadEngine.__init__``.
- A9  shared parallel engines are cached by worker count (no sticky first-pool-size).
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import CancelledError, Future

import pytest

import kedi.core as core
import kedi.engine as engine_mod
from kedi.engine import JobManagerEngine, RetryPolicy, ThreadEngine
from kedi.errors import KediExecutionError
from kedi.lang import compile_program, parse_program
from kedi.promise import KediPromise
from tests.mock_adapter import MockAdapter


class RateLimitError(Exception):
    """Type name contains 'ratelimit' → classified as a retryable transient error."""


# --- A2: retry classification unwraps the wrapped provider error ----------------------------


def test_a2_is_retryable_unwraps_original() -> None:
    policy = RetryPolicy()
    # core's _job wraps every provider error in KediExecutionError, keeping the provider error on
    # `.original`. is_retryable must classify on the original, not the (terminal) wrapper.
    wrapped = KediExecutionError("wrapped provider failure", original=RateLimitError())
    assert policy.is_retryable(wrapped)

    plain = KediExecutionError("our own validation error")  # no provider cause → terminal
    assert not policy.is_retryable(plain)


def test_a2_job_manager_retries_wrapped_transient() -> None:
    # End-to-end: a job that raises a KediExecutionError wrapping a transient must be retried.
    eng = JobManagerEngine(max_workers=2, _sleep=lambda _d: None)
    calls = {"n": 0}

    def job() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise KediExecutionError("boom", original=RateLimitError())
        return "ok"

    assert eng.schedule([], job).result() == "ok"
    assert calls["n"] == 3  # retried twice, succeeded on the third attempt


# --- A3: drain runs on exceptional exit without masking the primary error -------------------


class AlwaysFailAdapter(MockAdapter):
    def produce_sync(self, *, template: str, output_schema=None, **kwargs):  # type: ignore[override]
        raise RuntimeError("template boom")


def test_a3_drain_runs_on_exceptional_exit_and_preserves_primary(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The procedure schedules a (failing) template, then its body raises. The depth-0 drain must
    # still run on the exceptional exit, await the outstanding job, and LOG its failure rather than
    # mask the body's primary error.
    src = (
        "@gen():\n"
        "  >> failing [x]\n"
        "  ```\n"
        '  raise RuntimeError("body boom")\n'
        "  ```\n"
        "  = done\n\n"
        "= <gen()>\n"
    )
    rt = compile_program(parse_program(src), adapter=AlwaysFailAdapter(), engine=ThreadEngine(2))
    with caplog.at_level(logging.ERROR, logger="kedi"):
        with pytest.raises(BaseException):
            rt.run_main()
    # The drain-after-primary branch only fires when the body already raised AND the drain found a
    # failure — its log is direct evidence that the primary error was preserved, not masked.
    assert any("during drain after a primary error" in rec.message for rec in caplog.records)


# --- A4: re-entrant schedule honors not-yet-ready deps --------------------------------------


def test_a4_reentrant_schedule_defers_until_deps_done() -> None:
    eng = ThreadEngine(2)
    eng._local.on_worker = True  # simulate a tool issuing a nested template on a worker thread
    dep: Future = Future()  # not done yet
    ran: list[int] = []

    def job() -> str:
        ran.append(1)
        return "ok"

    result = eng.schedule([dep], job)
    assert ran == []  # must NOT run inline while a dep is unresolved
    assert not result.done()

    dep.set_result({"k": 1})  # deps ready → continuation fires
    assert result.result() == "ok"
    assert ran == [1]


def test_a4_reentrant_schedule_runs_inline_when_deps_done() -> None:
    eng = ThreadEngine(2)
    eng._local.on_worker = True
    dep: Future = Future()
    dep.set_result({"k": 1})  # already done
    result = eng.schedule([dep], lambda: "v")
    assert result.done()  # ran inline, no pool round-trip
    assert result.result() == "v"


# --- A5: a cancelled Future settles map/drain loudly instead of hanging ----------------------


def test_a5_map_settles_on_cancelled_source() -> None:
    src: Future = Future()
    assert src.cancel()  # cancellable while never submitted
    promise = KediPromise(src, "k")
    derived = promise.map(lambda v: v + 1)
    with pytest.raises(CancelledError):
        derived.resolve()  # derived was settled with the CancelledError, not left pending


def test_a5_drain_surfaces_cancelled_outstanding_future() -> None:
    rt = compile_program(parse_program("= done\n"), adapter=MockAdapter(), engine=ThreadEngine(2))
    cancelled: Future = Future()
    assert cancelled.cancel()
    rt._register_outstanding(cancelled)
    with pytest.raises(CancelledError):
        rt.drain()  # the cancelled job surfaces as an error rather than blocking forever


# --- A6: a failed pool.submit surfaces on the result Future ----------------------------------


def test_a6_schedule_surfaces_pool_submit_failure() -> None:
    eng = ThreadEngine(2)
    eng.shutdown()  # the pool now rejects new submissions
    dep: Future = Future()
    dep.set_result({"k": 1})  # deps ready → _submit runs immediately
    result = eng.schedule([dep], lambda: "v")
    with pytest.raises(RuntimeError):
        result.result(timeout=2)  # submit failure mirrored onto the Future, not a hang


# --- A7: the value-env is snapshot by value at schedule time --------------------------------


class EnvProbeAdapter(MockAdapter):
    """Records the worker's view of ``_current_env`` after a controlled delay."""

    def __init__(self, runtime_box: dict, started: threading.Event, release: threading.Event):
        super().__init__()
        self._box = runtime_box
        self._started = started
        self._release = release
        self.seen_marker: object = "UNSET"

    def produce_sync(self, *, template: str, output_schema=None, **kwargs):  # type: ignore[override]
        self._started.set()
        self._release.wait(2)  # park the worker until the main thread has mutated the live env
        env = self._box["rt"]._current_env.get()
        self.seen_marker = (env or {}).get("marker", "MISSING")
        return super().produce_sync(template=template, output_schema=output_schema, **kwargs)


def test_a7_env_snapshot_isolates_worker_from_later_writes() -> None:
    # marker="first" is set, the template is scheduled (snapshot taken), THEN the live env's marker
    # is rewritten. A by-value snapshot means the parked worker still sees "first".
    src = (
        "@gen():\n"
        "  [marker] = first\n"
        "  >> tmpl [out]\n"
        "  ```\n"
        '  marker = marker + "-second"\n'
        "  ```\n"
        "  = done\n\n"
        "= <gen()>\n"
    )
    started = threading.Event()
    release = threading.Event()
    box: dict = {}
    adapter = EnvProbeAdapter(box, started, release)
    rt = compile_program(parse_program(src), adapter=adapter, engine=ThreadEngine(2))
    box["rt"] = rt

    outcome: dict = {}

    def go() -> None:
        try:
            outcome["result"] = rt.run_main()
        except BaseException as exc:  # noqa: BLE001 - surfaced via the assertion below
            outcome["error"] = exc

    worker = threading.Thread(target=go)
    worker.start()
    assert started.wait(2), "worker never reached produce_sync"
    release.set()
    worker.join(5)

    assert "error" not in outcome, outcome.get("error")
    assert adapter.seen_marker == "first"  # the later "second" write did not leak into the worker


# --- A8: the routing self-check runs once in ThreadEngine.__init__ --------------------------


def test_a8_thread_engine_runs_routing_check_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import kedi.lang.compiler.environment as env_mod

    monkeypatch.setattr(engine_mod, "_ROUTING_CHECKED", False)
    calls: list[int] = []
    monkeypatch.setattr(env_mod, "assert_kedienv_routing", lambda: calls.append(1))

    ThreadEngine(2)
    assert calls == [1]  # checked on first construction (even a directly-built engine)
    ThreadEngine(2)
    assert calls == [1]  # once-flagged: not re-run on the second construction


# --- A9: shared parallel engines are cached by worker count ---------------------------------


def test_a9_shared_engine_cached_by_worker_count() -> None:
    e4a = core.shared_thread_engine(max_workers=4)
    e4b = core.shared_thread_engine(max_workers=4)
    e7 = core.shared_thread_engine(max_workers=7)
    assert e4a is e4b  # same worker count → reused
    assert e4a is not e7  # different worker count → a distinct pool, not the sticky first one
