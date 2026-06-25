"""Execution engines for Kedi template jobs.

An engine decides *when* a produce job (one LLM request + validation) runs and how its
result Future is fulfilled. The promise/dataflow layer (``promise.py``) decides *readiness*
(a job's dependency promises must resolve first); the engine decides *admission*.

- :class:`SequentialEngine` (the default) runs each job inline the moment it is scheduled and
  returns an already-completed Future. With promises disabled this reproduces today's behavior
  byte-for-byte.
- :class:`ThreadEngine` (opt-in) schedules each job as a continuation of its dependency Futures
  (via :func:`when_all`, so no worker ever blocks waiting on a dependency) and runs it on a
  bounded ``ThreadPoolExecutor`` under a snapshot of the submitting context. A job triggered
  re-entrantly from a worker (a tool that issues a nested template) runs inline to avoid
  pool starvation.

- :class:`JobManagerEngine` (opt-in, for real rate-limited LLMs) layers an adaptive controller on
  :class:`ThreadEngine`: AIMD concurrency, retry of transient/throttle failures with exponential
  backoff + full jitter, and a circuit breaker. Mock-backed tests use :class:`ThreadEngine` (no
  rate limits); the controller's pure pieces are unit-tested directly.
"""

from __future__ import annotations

import contextvars
import random
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Sequence

from .promise import when_all

__all__ = [
    "Engine",
    "SequentialEngine",
    "ThreadEngine",
    "JobManagerEngine",
    "RetryPolicy",
    "AIMDLimiter",
    "CircuitBreaker",
    "CircuitOpenError",
]

Job = Callable[[], Any]


class Engine:
    """Engine interface. ``parallel`` flags whether promises should flow through the env."""

    parallel: bool = False

    def schedule(self, deps: Sequence[Future], job: Job) -> Future:  # pragma: no cover - iface
        raise NotImplementedError

    def shutdown(self) -> None:  # pragma: no cover - iface
        pass


class SequentialEngine(Engine):
    """Run each job inline and return a completed Future (today's behavior)."""

    parallel = False

    def schedule(self, deps: Sequence[Future], job: Job) -> Future:
        fut: Future = Future()
        try:
            fut.set_result(job())
        except BaseException as exc:  # noqa: BLE001 - mirror onto the Future for uniform handling
            fut.set_exception(exc)
        return fut

    def shutdown(self) -> None:
        return None


_ROUTING_CHECKED = False


def _settle(fut: Future, job: Job) -> None:
    """Run ``job`` and mirror its result/exception onto ``fut``."""
    try:
        fut.set_result(job())
    except BaseException as exc:  # noqa: BLE001
        fut.set_exception(exc)


class ThreadEngine(Engine):
    """Run produce jobs concurrently on a bounded pool, gated on their dependency Futures."""

    parallel = True

    def __init__(self, max_workers: int = 8) -> None:
        global _ROUTING_CHECKED
        if not _ROUTING_CHECKED:
            # Refuse parallel mode loudly if lazy promise resolution can't work on this interpreter.
            # Runs once for ANY parallel engine — including ones constructed directly or passed via
            # compile_program(engine=...) — not just shared_thread_engine().
            from .lang.compiler.environment import assert_kedienv_routing

            assert_kedienv_routing()
            _ROUTING_CHECKED = True
        self._pool = ThreadPoolExecutor(
            max_workers=max(1, max_workers), thread_name_prefix="kedi-llm"
        )
        self._local = threading.local()

    def _on_worker(self) -> bool:
        return getattr(self._local, "on_worker", False)

    def schedule(self, deps: Sequence[Future], job: Job) -> Future:
        dep_list = list(deps)
        result: Future = Future()

        # Re-entrant (a tool on a worker issued this template). Run inline ONLY when every dep is
        # already resolved; otherwise schedule it as a continuation so we neither run before deps
        # are ready nor block this worker waiting on a pool Future (which would risk deadlock).
        if self._on_worker():
            if all(d.done() for d in dep_list):
                _settle(result, job)
            else:
                when_all(dep_list, lambda: _settle(result, job))
            return result

        # Capture the submitting context NOW (carries _current_env, _trace_stack, agent profile,
        # dspy.context, tool scope) so the worker runs with the right contextvars.
        ctx = contextvars.copy_context()

        def _run() -> None:
            prev = getattr(self._local, "on_worker", False)
            self._local.on_worker = True
            try:
                result.set_result(ctx.run(job))
            except BaseException as exc:  # noqa: BLE001
                result.set_exception(exc)
            finally:
                self._local.on_worker = prev

        def _submit() -> None:
            try:
                self._pool.submit(_run)
            except BaseException as exc:  # noqa: BLE001 - a shut-down pool surfaces, never hangs
                result.set_exception(exc)

        # Submit to the pool only once every dependency has completed — a continuation, never a
        # blocking wait inside a worker.
        when_all(dep_list, _submit)
        return result

    def shutdown(self) -> None:
        self._pool.shutdown(wait=True)


# ---------------------------------------------------------------------------------------------
# Adaptive job-manager controller (for real, rate-limited LLM backends)
# ---------------------------------------------------------------------------------------------

# Substrings identifying retryable (transient / throttle) failures, matched against the
# exception's type name. Provider libraries aren't imported, so we classify structurally.
_RETRYABLE_NAME_HINTS = (
    "ratelimit",
    "timeout",
    "timederror",  # asyncio.TimeoutError / TimeoutError
    "connection",
    "serviceunavailable",
    "apiconnection",
    "internalserver",
    "overloaded",
    "toomanyrequests",
)
_RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


class CircuitOpenError(RuntimeError):
    """Raised when the circuit breaker is open (too many consecutive failures)."""


class RetryPolicy:
    """Classifies failures and computes exponential backoff with full jitter."""

    def __init__(
        self,
        *,
        max_attempts: int = 6,
        base_delay: float = 0.5,
        max_delay: float = 60.0,
        deadline: float = 120.0,
    ) -> None:
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.deadline = deadline

    def is_retryable(self, exc: BaseException) -> bool:
        from .errors import KediExecutionError

        # Core wraps provider errors in KediExecutionError before the engine sees them; classify on
        # the ORIGINAL provider exception (kept on `.original`) so a 429/timeout still retries.
        if isinstance(exc, KediExecutionError) and isinstance(exc.original, BaseException):
            exc = exc.original
        # Our own validation / unsupported-feature errors are terminal — never retry them.
        if isinstance(exc, (KediExecutionError, ValueError, TypeError, KeyError)):
            return False
        # Only the standard HTTP ``status_code`` is treated as a status (not ``.code``, which many
        # non-HTTP exceptions use for unrelated values — e.g. socket/sqlite error numbers).
        status = getattr(exc, "status_code", None)
        if isinstance(status, int) and status in _RETRYABLE_STATUS:
            return True
        if isinstance(status, int):  # an explicit non-retryable HTTP status (e.g. 400/401/403)
            return False
        name = type(exc).__name__.lower()
        return any(hint in name for hint in _RETRYABLE_NAME_HINTS)

    def backoff(self, attempt: int, *, retry_after: float | None = None) -> float:
        """Full-jitter exponential backoff for ``attempt`` (0-based); honor ``Retry-After``."""
        if retry_after is not None and retry_after >= 0:
            return min(self.max_delay, retry_after)
        ceiling = min(self.max_delay, self.base_delay * (2**attempt))
        return random.uniform(0, ceiling)  # full jitter (random is fine in production code)

    @staticmethod
    def retry_after_of(exc: BaseException) -> float | None:
        value = getattr(exc, "retry_after", None)
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None


class AIMDLimiter:
    """Additive-increase / multiplicative-decrease concurrency limiter.

    A dynamic semaphore: the in-flight ceiling rises by 1 on success and halves on a throttle
    signal, clamped to ``[1, max_concurrency]`` — self-tuning to the backend's real limit without
    knowing it. ``acquire`` blocks while in-flight has reached the current ceiling.
    """

    def __init__(
        self, *, max_concurrency: int = 8, initial: int | None = None, minimum: int = 1
    ) -> None:
        self.max_concurrency = max(1, max_concurrency)
        self.minimum = max(1, minimum)
        self._limit = float(
            min(self.max_concurrency, initial or max(self.minimum, min(4, self.max_concurrency)))
        )
        self._inflight = 0
        self._cv = threading.Condition()

    @property
    def limit(self) -> int:
        return int(self._limit)

    def acquire(self) -> None:
        with self._cv:
            while self._inflight >= int(self._limit):
                self._cv.wait()
            self._inflight += 1

    def release(self) -> None:
        with self._cv:
            self._inflight -= 1
            self._cv.notify_all()

    def on_success(self) -> None:
        with self._cv:
            self._limit = min(float(self.max_concurrency), self._limit + 1.0)
            self._cv.notify_all()

    def on_throttle(self) -> None:
        with self._cv:
            self._limit = max(float(self.minimum), self._limit * 0.5)
            # ceiling dropped; no notify needed (fewer permits available)


class CircuitBreaker:
    """Opens after ``threshold`` consecutive failures; any success resets it."""

    def __init__(self, threshold: int = 20) -> None:
        self.threshold = threshold
        self._consecutive = 0
        self._lock = threading.Lock()

    def record_success(self) -> None:
        with self._lock:
            self._consecutive = 0

    def record_failure(self) -> None:
        with self._lock:
            self._consecutive += 1

    def check(self) -> None:
        with self._lock:
            if self._consecutive >= self.threshold:
                raise CircuitOpenError(
                    f"Circuit breaker open after {self._consecutive} consecutive failures; "
                    "backend appears unavailable. Aborting further template calls."
                )


class JobManagerEngine(ThreadEngine):
    """ThreadEngine + adaptive controller for real rate-limited backends.

    Each produce job runs under an AIMD concurrency permit and is retried on transient/throttle
    failures with exponential backoff + full jitter (honoring ``Retry-After``), bounded by
    ``max_attempts`` and a per-job wall-clock ``deadline``; a circuit breaker aborts the run if the
    backend looks dead. (Backoff currently sleeps on the worker; a delayed-retry queue that frees
    the slot during backoff is a future refinement.)
    """

    def __init__(
        self,
        max_workers: int = 8,
        *,
        max_attempts: int = 6,
        base_delay: float = 0.5,
        max_delay: float = 60.0,
        deadline: float = 120.0,
        breaker_threshold: int = 20,
        initial_concurrency: int | None = None,
        _clock: Callable[[], float] = time.monotonic,
        _sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        super().__init__(max_workers=max_workers)
        self.policy = RetryPolicy(
            max_attempts=max_attempts,
            base_delay=base_delay,
            max_delay=max_delay,
            deadline=deadline,
        )
        self.limiter = AIMDLimiter(max_concurrency=max_workers, initial=initial_concurrency)
        self.breaker = CircuitBreaker(breaker_threshold)
        self._clock = _clock
        self._sleep = _sleep

    def schedule(self, deps: Sequence[Future], job: Job) -> Future:
        # A re-entrant job (tool on a worker issuing a nested template) must NOT re-acquire an AIMD
        # permit or re-count against the breaker — that would deadlock the non-reentrant limiter.
        # Run it bare; only top-level jobs go through the controller.
        if self._on_worker():
            return super().schedule(deps, job)
        return super().schedule(deps, lambda: self._run_with_policy(job))

    def _run_with_policy(self, job: Job) -> Any:
        self.breaker.check()
        self.limiter.acquire()
        start = self._clock()
        try:
            attempt = 0
            while True:
                try:
                    result = job()
                    self.limiter.on_success()
                    self.breaker.record_success()
                    return result
                except BaseException as exc:  # noqa: BLE001
                    self.breaker.record_failure()
                    retryable = self.policy.is_retryable(exc)
                    if retryable:
                        self.limiter.on_throttle()
                    attempt += 1
                    elapsed = self._clock() - start
                    if (
                        not retryable
                        or attempt >= self.policy.max_attempts
                        or elapsed >= self.policy.deadline
                    ):
                        raise
                    self._sleep(
                        self.policy.backoff(attempt, retry_after=self.policy.retry_after_of(exc))
                    )
        finally:
            self.limiter.release()
