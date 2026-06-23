"""Coverage-completing tests for the non-blocking layer.

The default suite runs sequentially, so the parallel/job-manager/loud-promise paths need explicit
exercise. These tests target the remaining branches (loud dunders, derived-promise errors, engine
edge paths, engine selection, and the multi-failure drain).
"""

from __future__ import annotations

import copy
import operator
import pickle
from concurrent.futures import Future

import pytest

import kedi.core as core
from kedi.engine import (
    JobManagerEngine,
    RetryPolicy,
    SequentialEngine,
    ThreadEngine,
)
from kedi.errors import KediExecutionError
from kedi.lang import compile_program, parse_program
from kedi.lang.compiler.environment import assert_kedienv_routing
from kedi.promise import KediPromise, KediPromiseLeak, lazy_join
from tests.mock_adapter import MockAdapter


def _done(value: object) -> Future:
    f: Future = Future()
    f.set_result({"k": value})
    return f


def _promise(value: object = "v") -> KediPromise:
    return KediPromise(_done(value), "k")


# --- KediPromise: every loud value-use raises ----------------------------------------------


def test_loud_dunders_all_raise() -> None:
    p = _promise()
    leaks = [
        lambda: str(p),
        lambda: f"{p}",  # __format__
        lambda: bytes(p),
        lambda: bool(p),
        lambda: len(p),
        lambda: iter(p),
        lambda: p.__next__(),
        lambda: p[0],
        lambda: 1 in p,
        lambda: p(),
        lambda: p == 1,
        lambda: p != 1,
        lambda: hash(p),
        lambda: p < 1,
        lambda: p <= 1,
        lambda: p > 1,
        lambda: p >= 1,
        lambda: p + 1,
        lambda: 1 + p,
        lambda: p - 1,
        lambda: p * 1,
        lambda: 1 * p,
        lambda: int(p),
        lambda: float(p),
        lambda: operator.index(p),
        lambda: p.some_attr,
    ]
    for op in leaks:
        with pytest.raises(KediPromiseLeak):
            op()


def test_getattr_dunder_raises_attributeerror_other_names_leak() -> None:
    p = _promise()
    # Undefined dunders raise AttributeError (so copy/pickle machinery doesn't force/recurse)...
    with pytest.raises(AttributeError):
        getattr(p, "__nonexistent_dunder__")
    # ...but any ordinary attribute access is a loud leak.
    with pytest.raises(KediPromiseLeak):
        getattr(p, "_not_a_slot")


def test_copy_is_identity_and_pickle_is_loud() -> None:
    p = _promise()
    assert copy.copy(p) is p
    assert copy.deepcopy(p) is p
    with pytest.raises(KediPromiseLeak):
        pickle.dumps(p)


def test_map_propagates_source_and_fn_errors() -> None:
    failed: Future = Future()
    failed.set_exception(RuntimeError("boom"))
    with pytest.raises(RuntimeError):
        KediPromise(failed, "k").map(lambda v: v).resolve()

    def bad(_v: object) -> object:
        raise ValueError("fn-error")

    with pytest.raises(ValueError):
        _promise().map(bad).resolve()


def test_lazy_join_deferred_combine_error() -> None:
    fut: Future = Future()
    p = KediPromise(fut, "x")

    def combine(_parts: list[object]) -> str:
        raise ValueError("join-error")

    joined = lazy_join(["a", p], combine)
    assert isinstance(joined, KediPromise)
    fut.set_result({"x": "b"})
    with pytest.raises(ValueError):
        joined.resolve()


# --- Engine edge paths ----------------------------------------------------------------------


def test_sequential_engine_exception_and_shutdown() -> None:
    eng = SequentialEngine()

    def boom() -> None:
        raise RuntimeError("x")

    fut = eng.schedule([], boom)
    assert isinstance(fut.exception(), RuntimeError)
    eng.shutdown()  # no-op


def test_thread_engine_inline_bypass_and_shutdown() -> None:
    eng = ThreadEngine(max_workers=2)
    eng._local.on_worker = True  # simulate running on a worker (re-entrant)
    try:
        assert eng.schedule([], lambda: 5).result() == 5

        def boom() -> None:
            raise RuntimeError("inline")

        assert isinstance(eng.schedule([], boom).exception(), RuntimeError)
    finally:
        eng._local.on_worker = False
    eng.shutdown()


def test_jobmanager_inline_bypass_skips_policy() -> None:
    eng = JobManagerEngine(max_workers=2, _sleep=lambda _d: None)
    eng._local.on_worker = True
    try:
        assert eng.schedule([], lambda: 7).result() == 7
    finally:
        eng._local.on_worker = False
    eng.shutdown()


def test_retry_after_of_variants() -> None:
    p = RetryPolicy()

    class WithBad(Exception):
        retry_after = "not-a-number"

    class WithGood(Exception):
        retry_after = 3

    assert p.retry_after_of(WithBad()) is None
    assert p.retry_after_of(WithGood()) == 3.0
    assert p.retry_after_of(Exception()) is None


# --- Engine selection (core._default_engine / shared_thread_engine) -------------------------


@pytest.mark.parametrize(
    "value,parallel",
    [(None, False), ("0", False), ("-1", False), ("1", True), ("4", True), ("notint", True)],
)
def test_default_engine_selection(monkeypatch: pytest.MonkeyPatch, value, parallel) -> None:
    if value is None:
        monkeypatch.delenv("KEDI_PARALLEL", raising=False)
    else:
        monkeypatch.setenv("KEDI_PARALLEL", value)
    eng = core._default_engine()
    assert bool(getattr(eng, "parallel", False)) is parallel


def test_shared_thread_engine_and_self_check() -> None:
    assert_kedienv_routing()  # must not raise on a supported interpreter
    eng = core.shared_thread_engine()
    assert eng is core.shared_thread_engine()  # singleton
    assert getattr(eng, "parallel", False) is True


# --- Drain: multiple concurrent failures (fail-fast first, log the rest) ---------------------


class AlwaysFailAdapter(MockAdapter):
    def produce_sync(self, *, template: str, output_schema=None, **kwargs):  # type: ignore[override]
        raise RuntimeError("fail")


def test_drain_surfaces_first_and_logs_additional_failures() -> None:
    # Two independent, unconsumed failing templates -> both fail at drain; first is raised, the
    # second is logged (exercises the extra-failure branch).
    src = "@gen():\n  >> a [x]\n  >> b [y]\n  = done\n\n= <gen()>\n"
    rt = compile_program(parse_program(src), adapter=AlwaysFailAdapter(), engine=ThreadEngine(2))
    with pytest.raises(KediExecutionError):
        rt.run_main()


class KediFailAdapter(MockAdapter):
    def produce_sync(self, *, template: str, output_schema=None, **kwargs):  # type: ignore[override]
        raise KediExecutionError("kedi-boom")


def test_parallel_job_reraises_kedi_execution_error() -> None:
    # When the produce job raises a KediExecutionError on a worker, _job re-raises it as-is
    # (the `except KediExecutionError: raise` branch) rather than re-wrapping.
    src = "@gen():\n  >> a [x]\n  = <x>\n\n= <gen()>\n"
    rt = compile_program(parse_program(src), adapter=KediFailAdapter(), engine=ThreadEngine(2))
    with pytest.raises(KediExecutionError):
        rt.run_main()
