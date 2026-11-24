"""Unit tests for the adaptive job-manager controller (engine.py).

Mock-backed integration tests can't exercise real rate limits, so the controller's pure pieces
(retry classification, backoff, AIMD, circuit breaker) and the retry loop are tested directly.
"""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import Future

import pytest

from kedi.engine import (
    AIMDLimiter,
    CircuitBreaker,
    CircuitOpenError,
    JobManagerEngine,
    RetryPolicy,
)


class RateLimitError(Exception):
    """Name contains 'ratelimit' → classified retryable."""


class HttpError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


# --- RetryPolicy ----------------------------------------------------------------------------


def test_retry_policy_classifies_transient_as_retryable() -> None:
    p = RetryPolicy()
    assert p.is_retryable(RateLimitError())
    assert p.is_retryable(TimeoutError())
    assert p.is_retryable(HttpError(429))
    assert p.is_retryable(HttpError(503))


def test_retry_policy_classifies_terminal_as_non_retryable() -> None:
    p = RetryPolicy()
    assert not p.is_retryable(ValueError("bad schema"))
    assert not p.is_retryable(HttpError(400))
    assert not p.is_retryable(HttpError(401))
    from kedi.errors import KediExecutionError

    assert not p.is_retryable(KediExecutionError("nope"))


def test_backoff_is_bounded_and_honors_retry_after() -> None:
    p = RetryPolicy(base_delay=1.0, max_delay=10.0)
    for attempt in range(8):
        d = p.backoff(attempt)
        assert 0.0 <= d <= 10.0  # full jitter, never exceeds the cap
    assert p.backoff(0, retry_after=3.0) == 3.0
    assert p.backoff(0, retry_after=999.0) == 10.0  # clamped to max_delay


# --- AIMDLimiter ----------------------------------------------------------------------------


def test_aimd_additive_increase_multiplicative_decrease() -> None:
    lim = AIMDLimiter(max_concurrency=8, initial=4)
    assert lim.limit == 4
    lim.on_success()
    assert lim.limit == 5
    lim.on_throttle()
    assert lim.limit == 2  # 5 * 0.5 -> 2 (floored via int())
    for _ in range(20):
        lim.on_success()
    assert lim.limit == 8  # clamped to max
    for _ in range(20):
        lim.on_throttle()
    assert lim.limit == 1  # clamped to minimum


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_concurrency": 0}, "max_concurrency must be at least 1"),
        ({"max_concurrency": 4, "minimum": 0}, "minimum must be at least 1"),
        (
            {"max_concurrency": 2, "minimum": 3},
            "minimum must not exceed max_concurrency",
        ),
        (
            {"max_concurrency": 4, "minimum": 2, "initial": 1},
            "initial must be between minimum and max_concurrency",
        ),
        (
            {"max_concurrency": 4, "initial": 5},
            "initial must be between minimum and max_concurrency",
        ),
    ],
)
def test_aimd_rejects_invalid_configured_bounds(
    kwargs: dict[str, int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        AIMDLimiter(**kwargs)


def test_aimd_acquire_release_caps_inflight() -> None:
    lim = AIMDLimiter(max_concurrency=2, initial=2)
    lim.acquire()
    lim.acquire()
    released = []

    import threading

    def third() -> None:
        lim.acquire()
        released.append(True)
        lim.release()

    t = threading.Thread(target=third)
    t.start()
    t.join(timeout=0.2)
    assert released == []  # blocked: in-flight at ceiling
    lim.release()  # frees a slot
    t.join(timeout=1.0)
    assert released == [True]
    lim.release()


# --- CircuitBreaker -------------------------------------------------------------------------


def test_circuit_breaker_opens_then_resets() -> None:
    cb = CircuitBreaker(threshold=3)
    cb.check()  # closed
    cb.record_failure()
    cb.record_failure()
    cb.check()  # 2 < 3, still closed
    cb.record_failure()
    with pytest.raises(CircuitOpenError):
        cb.check()
    cb.record_success()  # resets
    cb.check()  # closed again


# --- JobManagerEngine retry loop ------------------------------------------------------------


def _engine() -> JobManagerEngine:
    # No-op sleep so backoff doesn't slow tests.
    return JobManagerEngine(max_workers=4, _sleep=lambda _d: None)


def test_engine_retries_transient_then_succeeds() -> None:
    eng = _engine()
    calls = {"n": 0}

    def job() -> str:
        calls["n"] += 1
        if calls["n"] < 3:
            raise RateLimitError()
        return "done"

    assert eng.schedule([], job).result() == "done"
    assert calls["n"] == 3


def test_engine_gives_up_after_max_attempts() -> None:
    eng = JobManagerEngine(max_workers=2, max_attempts=4, _sleep=lambda _d: None)
    calls = {"n": 0}

    def job() -> str:
        calls["n"] += 1
        raise RateLimitError()

    with pytest.raises(RateLimitError):
        eng.schedule([], job).result()
    assert calls["n"] == 4


def test_engine_does_not_retry_terminal_error() -> None:
    eng = _engine()
    calls = {"n": 0}

    def job() -> str:
        calls["n"] += 1
        raise ValueError("bad")

    with pytest.raises(ValueError):
        eng.schedule([], job).result()
    assert calls["n"] == 1  # terminal → no retry


def test_engine_success_passes_through() -> None:
    eng = _engine()
    assert eng.schedule([], lambda: 42).result() == 42


@pytest.mark.parametrize("error", [ValueError("invalid"), asyncio.CancelledError(), SystemExit(2)])
def test_terminal_failures_do_not_open_the_backend_breaker(error: BaseException) -> None:
    engine = JobManagerEngine(max_workers=1, breaker_threshold=1)

    def fail() -> None:
        raise error

    try:
        for _ in range(2):
            with pytest.raises(type(error)):
                engine.schedule([], fail).result(timeout=2)
        assert engine.schedule([], lambda: "ready").result(timeout=2) == "ready"
    finally:
        engine.shutdown()


def test_control_exceptions_are_not_retryable_even_with_an_http_status() -> None:
    class StopRun(BaseException):
        status_code = 429

    assert not RetryPolicy().is_retryable(StopRun())


def test_first_retry_uses_the_base_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    delays: list[float] = []
    monkeypatch.setattr("kedi.engine.random.uniform", lambda _low, high: high)
    engine = JobManagerEngine(max_workers=1, base_delay=0.25, _sleep=delays.append)
    calls = 0

    def job() -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise RateLimitError()
        return "ready"

    try:
        assert engine.schedule([], job).result(timeout=2) == "ready"
        assert delays == [0.25, 0.5]
    finally:
        engine.shutdown()


@pytest.mark.parametrize("fails", [False, True])
def test_stolen_work_reuses_the_workers_permit(
    monkeypatch: pytest.MonkeyPatch, fails: bool
) -> None:
    engine = JobManagerEngine(max_workers=1, initial_concurrency=1)
    started = threading.Event()
    queued = threading.Event()
    children: list[Future[str]] = []
    acquire = engine.limiter.acquire
    acquisitions = 0

    def checked_acquire() -> None:
        nonlocal acquisitions
        # Fail deterministically instead of leaving a deadlocked worker on regression.
        assert engine.limiter._inflight == 0, "nested AIMD permit"
        acquisitions += 1
        acquire()

    monkeypatch.setattr(engine.limiter, "acquire", checked_acquire)

    def parent() -> str:
        started.set()
        assert queued.wait(2)
        return engine.wait(children[0])

    def child() -> str:
        if fails:
            raise ValueError("child failed")
        return "child result"

    try:
        result = engine.schedule([], parent)
        assert started.wait(2)
        children.append(engine.schedule([], child))
        queued.set()
        if fails:
            with pytest.raises(ValueError, match="child failed"):
                result.result(timeout=2)
        else:
            assert result.result(timeout=2) == "child result"
        assert acquisitions == 1
        assert engine.limiter._inflight == 0
        assert engine.schedule([], lambda: "next").result(timeout=2) == "next"
        assert acquisitions == 2
    finally:
        queued.set()
        engine.shutdown()
