from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, replace
from hashlib import sha256
from threading import Lock
from typing import Generator, Literal, TypeAlias
from uuid import uuid4

from .runs import SubagentRunStatus, safe_subagent_error_message

DelegationFailurePolicy: TypeAlias = Literal["recover", "fail_closed"]
DelegationAttemptStatus: TypeAlias = Literal[
    "pending",
    "completed",
    "rejected",
    "failed",
    "cancelled",
    "timed_out",
]

_FAILURE_STATUSES = frozenset({"rejected", "failed", "cancelled", "timed_out"})
_UNRESOLVED_STATUSES = frozenset((*_FAILURE_STATUSES, "pending"))


@dataclass(frozen=True, slots=True, kw_only=True)
class DelegationAttempt:
    attempt_id: str
    requested_profile: str
    task_fingerprint: str
    started_order: int
    status: DelegationAttemptStatus = "pending"
    finished_order: int | None = None
    run_id: str | None = None
    failure_reason: str | None = None
    replacement_attempt_id: str | None = None


class UnresolvedDelegationError(RuntimeError):
    def __init__(self, attempts: tuple[DelegationAttempt, ...]) -> None:
        self.attempts = attempts
        details = "; ".join(_attempt_failure_detail(attempt) for attempt in attempts)
        super().__init__(
            "Subagent delegation failed without a successful recovery: "
            f"{details}. Retry the same task successfully before returning a result."
        )


class DelegationAttemptLedger:
    """Task-local evidence for delegation attempts made by one agent invocation."""

    def __init__(self) -> None:
        self._attempts: dict[str, DelegationAttempt] = {}
        self._sequence = 0
        self._lock = Lock()

    def begin(self, *, requested_profile: str, task: str) -> DelegationAttempt:
        with self._lock:
            attempt = DelegationAttempt(
                attempt_id=uuid4().hex,
                requested_profile=requested_profile,
                task_fingerprint=sha256(task.encode()).hexdigest(),
                started_order=self._next_order(),
            )
            self._attempts[attempt.attempt_id] = attempt
            return attempt

    def bind_run(self, attempt_id: str, run_id: str) -> DelegationAttempt:
        with self._lock:
            attempt = self._pending(attempt_id)
            updated = replace(attempt, run_id=run_id)
            self._attempts[attempt_id] = updated
            return updated

    def complete(self, attempt_id: str) -> DelegationAttempt:
        with self._lock:
            attempt = self._pending(attempt_id)
            finished_order = self._next_order()
            completed = replace(
                attempt,
                status="completed",
                finished_order=finished_order,
            )
            self._attempts[attempt_id] = completed
            for candidate_id, candidate in tuple(self._attempts.items()):
                if not _is_recoverable_by(candidate, completed):
                    continue
                self._attempts[candidate_id] = replace(
                    candidate,
                    replacement_attempt_id=attempt_id,
                )
            return completed

    def fail(
        self,
        attempt_id: str,
        *,
        status: DelegationAttemptStatus,
        error: BaseException,
    ) -> DelegationAttempt:
        if status not in _FAILURE_STATUSES:
            raise ValueError(f"Invalid delegation failure status: {status!r}")
        with self._lock:
            attempt = self._pending(attempt_id)
            failed = replace(
                attempt,
                status=status,
                finished_order=self._next_order(),
                failure_reason=safe_subagent_error_message(error),
            )
            self._attempts[attempt_id] = failed
            return failed

    def unresolved(self) -> tuple[DelegationAttempt, ...]:
        with self._lock:
            return tuple(
                attempt
                for attempt in self._attempts.values()
                if attempt.status in _UNRESOLVED_STATUSES and attempt.replacement_attempt_id is None
            )

    def snapshot(self) -> tuple[DelegationAttempt, ...]:
        with self._lock:
            return tuple(self._attempts.values())

    def pending_for_run(self, run_id: str) -> DelegationAttemptHandle | None:
        with self._lock:
            attempt = next(
                (
                    candidate
                    for candidate in self._attempts.values()
                    if candidate.run_id == run_id and candidate.status == "pending"
                ),
                None,
            )
        if attempt is None:
            return None
        return DelegationAttemptHandle(self, attempt.attempt_id)

    def _pending(self, attempt_id: str) -> DelegationAttempt:
        attempt = self._attempts[attempt_id]
        if attempt.status != "pending":
            raise RuntimeError(f"Delegation attempt {attempt_id!r} is already {attempt.status}")
        return attempt

    def _next_order(self) -> int:
        self._sequence += 1
        return self._sequence


@dataclass(frozen=True, slots=True)
class DelegationAttemptHandle:
    ledger: DelegationAttemptLedger
    attempt_id: str

    @property
    def run_id(self) -> str | None:
        return next(
            attempt.run_id
            for attempt in self.ledger.snapshot()
            if attempt.attempt_id == self.attempt_id
        )

    def complete(self) -> DelegationAttempt:
        return self.ledger.complete(self.attempt_id)

    def fail(
        self,
        *,
        status: DelegationAttemptStatus,
        error: BaseException,
    ) -> DelegationAttempt:
        return self.ledger.fail(self.attempt_id, status=status, error=error)


_ACTIVE_LEDGER: ContextVar[DelegationAttemptLedger | None] = ContextVar(
    "kedi_delegation_attempt_ledger",
    default=None,
)
_ACTIVE_ATTEMPT: ContextVar[DelegationAttemptHandle | None] = ContextVar(
    "kedi_delegation_attempt",
    default=None,
)


@contextmanager
def delegation_scope(
    policy: DelegationFailurePolicy,
) -> Generator[DelegationAttemptLedger, None, None]:
    if policy not in {"recover", "fail_closed"}:
        raise ValueError(f"Unknown subagent failure policy: {policy!r}")
    ledger = DelegationAttemptLedger()
    token = _ACTIVE_LEDGER.set(ledger)
    try:
        yield ledger
    except BaseException:
        raise
    else:
        unresolved = ledger.unresolved()
        if policy == "fail_closed" and unresolved:
            raise UnresolvedDelegationError(unresolved)
    finally:
        _ACTIVE_LEDGER.reset(token)


@contextmanager
def track_delegation_attempt(
    *,
    requested_profile: str,
    task: str,
) -> Generator[DelegationAttemptHandle | None, None, None]:
    ledger = _ACTIVE_LEDGER.get()
    if ledger is None:
        yield None
        return

    attempt = ledger.begin(requested_profile=requested_profile, task=task)
    handle = DelegationAttemptHandle(ledger, attempt.attempt_id)
    token: Token[DelegationAttemptHandle | None] = _ACTIVE_ATTEMPT.set(handle)
    try:
        yield handle
    finally:
        _ACTIVE_ATTEMPT.reset(token)


def bind_current_delegation_run(run_id: str) -> None:
    handle = _ACTIVE_ATTEMPT.get()
    if handle is not None:
        handle.ledger.bind_run(handle.attempt_id, run_id)


def active_delegation_attempt_for_run(run_id: str) -> DelegationAttemptHandle | None:
    ledger = _ACTIVE_LEDGER.get()
    return None if ledger is None else ledger.pending_for_run(run_id)


def attempt_status_from_run(status: SubagentRunStatus) -> DelegationAttemptStatus:
    if status == "cancelled":
        return "cancelled"
    if status == "timed_out":
        return "timed_out"
    return "failed"


def _is_recoverable_by(
    failed: DelegationAttempt,
    completed: DelegationAttempt,
) -> bool:
    return (
        failed.status in _FAILURE_STATUSES
        and failed.replacement_attempt_id is None
        and failed.finished_order is not None
        and completed.started_order > failed.finished_order
        and completed.task_fingerprint == failed.task_fingerprint
    )


def _attempt_failure_detail(attempt: DelegationAttempt) -> str:
    run = f", run {attempt.run_id}" if attempt.run_id is not None else ""
    reason = f": {attempt.failure_reason}" if attempt.failure_reason else ""
    return (
        f"attempt {attempt.attempt_id[:8]} for {attempt.requested_profile!r}"
        f"{run} {attempt.status}{reason}"
    )
