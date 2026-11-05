from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from threading import Lock
from typing import Generator, Literal

from kedi.errors import KediExecutionError

BudgetDimension = Literal["request", "tool_attempt"]


@dataclass(frozen=True, slots=True)
class RunBudget:
    request_limit: int | None = None
    tool_attempt_limit: int | None = None

    def __post_init__(self) -> None:
        for name in ("request_limit", "tool_attempt_limit"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{name} must be a non-negative integer or None")

    def overlay(self, other: RunBudget) -> RunBudget:
        return RunBudget(
            request_limit=(
                self.request_limit if other.request_limit is None else other.request_limit
            ),
            tool_attempt_limit=(
                self.tool_attempt_limit
                if other.tool_attempt_limit is None
                else other.tool_attempt_limit
            ),
        )


class RunBudgetExceeded(KediExecutionError):
    def __init__(
        self, dimension: BudgetDimension, limit: int, consumed: int, scope_depth: int
    ) -> None:
        self.dimension = dimension
        self.limit = limit
        self.consumed = consumed
        self.scope_depth = scope_depth
        super().__init__(
            message=f"Run {dimension} budget exhausted (limit {limit}, consumed {consumed})"
        )


class RunBudgetLedger:
    """One scope in a lineage; all admissions lock the root before charging ancestors."""

    def __init__(self, budget: RunBudget, parent: RunBudgetLedger | None = None) -> None:
        self.budget = budget
        self.parent = parent
        self._spent: dict[BudgetDimension, int] = {"request": 0, "tool_attempt": 0}
        self._lock = parent._lock if parent is not None else Lock()

    def spent(self, dimension: BudgetDimension) -> int:
        with self._lock:
            return self._spent[dimension]

    def set_budget(self, budget: RunBudget) -> None:
        with self._lock:
            self.budget = self.budget.overlay(budget)

    def has_limit(self, dimension: BudgetDimension) -> bool:
        with self._lock:
            current: RunBudgetLedger | None = self
            while current is not None:
                limit = (
                    current.budget.request_limit
                    if dimension == "request"
                    else current.budget.tool_attempt_limit
                )
                if limit is not None:
                    return True
                current = current.parent
            return False

    def admit(self, dimension: BudgetDimension) -> None:
        lineage: list[RunBudgetLedger] = []
        current: RunBudgetLedger | None = self
        while current is not None:
            lineage.append(current)
            current = current.parent
        with self._lock:
            for depth, ledger in enumerate(lineage):
                limit = (
                    ledger.budget.request_limit
                    if dimension == "request"
                    else ledger.budget.tool_attempt_limit
                )
                if limit is not None and ledger._spent[dimension] >= limit:
                    raise RunBudgetExceeded(
                        dimension, limit, ledger._spent[dimension], len(lineage) - depth - 1
                    )
            for ledger in lineage:
                ledger._spent[dimension] += 1


_ledger: ContextVar[RunBudgetLedger | None] = ContextVar("kedi_run_budget", default=None)


def current_run_budget() -> RunBudgetLedger | None:
    return _ledger.get()


@contextmanager
def run_budget_scope(budget: RunBudget | None) -> Generator[RunBudgetLedger | None, None, None]:
    parent = _ledger.get()
    if budget is None and parent is not None:
        yield parent
        return
    ledger = RunBudgetLedger(budget or RunBudget(), parent)
    token = _ledger.set(ledger)
    try:
        yield ledger
    finally:
        _ledger.reset(token)


def admit_run_operation(dimension: BudgetDimension) -> None:
    ledger = _ledger.get()
    if ledger is not None:
        ledger.admit(dimension)


def assert_no_hidden_model_retries(model: object) -> None:
    """SDK retries must be disabled when a strict generation ceiling is requested."""
    ledger = current_run_budget()
    if ledger is None or not ledger.has_limit("request"):
        return
    provider = getattr(model, "provider", None)
    provider_client = getattr(provider, "client", None)
    if provider is not None and not hasattr(provider_client, "max_retries"):
        raise KediExecutionError(
            "A request run budget cannot enforce this provider's opaque retry policy; "
            "use a provider exposing disabled SDK retries"
        )
    clients = (
        model,
        provider_client,
        getattr(model, "root_client", None),
        getattr(model, "root_async_client", None),
    )
    for client in clients:
        if client is not None and hasattr(client, "max_retries"):
            retries = getattr(client, "max_retries")
            if retries != 0:
                raise KediExecutionError(
                    "A request run budget requires provider SDK max_retries=0; "
                    "hidden retries cannot be counted at the model boundary"
                )


def assert_run_budget_supported(adapter: object, *, opaque_tools: bool = False) -> None:
    ledger = _ledger.get()
    if ledger is None:
        return
    capabilities = getattr(adapter, "capabilities", None)
    dimensions: tuple[tuple[BudgetDimension, str], ...] = (
        ("request", "request_budget"),
        ("tool_attempt", "tool_attempt_budget"),
    )
    for dimension, feature in dimensions:
        if not ledger.has_limit(dimension):
            continue
        supported = (
            capabilities is not None
            and callable(getattr(capabilities, "supports", None))
            and capabilities.supports(feature)
        )
        if not supported or (dimension == "tool_attempt" and opaque_tools):
            raise KediExecutionError(
                f"Adapter {getattr(adapter, 'shortname', type(adapter).__name__)!r} "
                f"cannot enforce {dimension} run budget for this execution"
            )
