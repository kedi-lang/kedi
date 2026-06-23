"""Unit tests for the adaptive job-manager controller (engine.py).

Mock-backed integration tests can't exercise real rate limits, so the controller's pure pieces
(retry classification, backoff, AIMD, circuit breaker) and the retry loop are tested directly.
"""

from __future__ import annotations

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
