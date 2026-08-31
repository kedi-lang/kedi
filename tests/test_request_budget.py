from __future__ import annotations

import threading
from decimal import Decimal

import pytest

from kedi.agent_adapter.invocation import InvocationUsage
from kedi.agent_adapter.request_budget import (
    ModelRequestBudget,
    ModelRequestBudgetError,
    ModelRequestBudgetLimits,
    estimate_request_tokens,
)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"context_limit_tokens": 0}, "context_limit_tokens"),
        ({"safe_context_utilization": 0}, "safe_context_utilization"),
        ({"safe_context_utilization": 1.1}, "safe_context_utilization"),
        ({"output_reserve_tokens": -1}, "output_reserve_tokens"),
        ({"cumulative_input_tokens": 0}, "cumulative_input_tokens"),
        ({"request_limit": 0}, "request_limit"),
        (
            {
                "context_limit_tokens": 10,
                "safe_context_utilization": 0.5,
                "output_reserve_tokens": 5,
            },
            "no safe input context",
        ),
    ],
)
def test_model_request_budget_limits_validate(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ModelRequestBudgetLimits(**kwargs)  # type: ignore[arg-type]


def _limits(**kwargs: object) -> ModelRequestBudgetLimits:
    values: dict[str, object] = {
        "context_limit_tokens": 10_000,
        "safe_context_utilization": 1,
        "output_reserve_tokens": 0,
        "cumulative_input_tokens": 50_000,
        "request_limit": 10,
    }
    values.update(kwargs)
    return ModelRequestBudgetLimits(**values)  # type: ignore[arg-type]


def test_model_request_budget_tracks_estimates_and_actual_usage() -> None:
    budget = ModelRequestBudget(_limits())
    reservation = budget.reserve(
        "prompt",
        context=({"schema": "value"},),
        resume_state={"messages": ["prior"]},
    )
    estimated = budget.snapshot().cumulative_input_tokens
    assert estimated == estimate_request_tokens(
        "prompt",
        {"schema": "value"},
        {"messages": ["prior"]},
    )

    reservation.observe(InvocationUsage(requests=2, input_tokens=200, output_tokens=5))
    reservation.observe(InvocationUsage(requests=1, input_tokens=100, total_tokens=120))
    snapshot = reservation.finish()

    assert snapshot.requests == 3
    assert snapshot.cumulative_input_tokens == 300
    assert snapshot.estimated_request_tokens == estimated
    with pytest.raises(RuntimeError, match="already closed"):
        reservation.finish()


def test_model_request_budget_reconciles_cached_input_as_uncached_usage() -> None:
    budget = ModelRequestBudget(_limits(cumulative_input_tokens=500_000, request_limit=128))
    reservation = budget.reserve("prompt")

    reservation.observe(
        InvocationUsage(
            requests=58,
            input_tokens=2_029_375,
            cache_read_tokens=1_628_928,
        )
    )

    snapshot = reservation.finish()

    assert snapshot.requests == 58
    assert snapshot.cumulative_input_tokens == 400_447


def test_invocation_usage_aggregates_cache_and_cost_without_negative_uncached_tokens() -> None:
    usage = InvocationUsage(
        requests=1,
        input_tokens=100,
        cache_read_tokens=80,
        cache_write_tokens=10,
        cost_usd=Decimal("0.01"),
    ) + InvocationUsage(
        requests=2,
        input_tokens=20,
        cache_read_tokens=50,
        cache_write_tokens=5,
        cost_usd=Decimal("0.02"),
    )

    assert usage == InvocationUsage(
        requests=3,
        input_tokens=120,
        cache_read_tokens=130,
        cache_write_tokens=15,
        cost_usd=Decimal("0.03"),
    )
    assert usage.uncached_input_tokens == 0
    assert usage + InvocationUsage(requests=1) == InvocationUsage(
        requests=4,
        input_tokens=120,
        cache_read_tokens=130,
        cache_write_tokens=15,
        cost_usd=Decimal("0.03"),
    )


def test_model_request_budget_keeps_estimate_without_reported_usage() -> None:
    budget = ModelRequestBudget(_limits())
    reservation = budget.reserve("prompt")
    estimated = budget.snapshot().cumulative_input_tokens

    assert reservation.finish().cumulative_input_tokens == estimated


def test_model_request_budget_rejects_request_context_before_reserving() -> None:
    budget = ModelRequestBudget(
        _limits(
            context_limit_tokens=2_100,
            cumulative_input_tokens=10_000,
        )
    )

    with pytest.raises(ModelRequestBudgetError, match="safe context ceiling"):
        budget.reserve("x" * 100)
    assert budget.snapshot().requests == 0


def test_model_request_budget_rejects_request_and_cumulative_limits() -> None:
    request_budget = ModelRequestBudget(_limits(request_limit=1))
    request_budget.reserve("").finish()
    with pytest.raises(ModelRequestBudgetError, match="request limit"):
        request_budget.reserve("")

    cumulative_budget = ModelRequestBudget(_limits(cumulative_input_tokens=4_100))
    cumulative_budget.reserve("x" * 1_000).finish()
    with pytest.raises(ModelRequestBudgetError, match="cumulative model input"):
        cumulative_budget.reserve("x" * 1_000)


@pytest.mark.parametrize(
    ("usage", "message"),
    [
        (InvocationUsage(requests=2, input_tokens=10), "used 2 requests"),
        (InvocationUsage(requests=1, input_tokens=4_101), "cumulative uncached input tokens"),
    ],
)
def test_model_request_budget_reports_provider_side_overrun(
    usage: InvocationUsage,
    message: str,
) -> None:
    limits = _limits(request_limit=1, cumulative_input_tokens=4_100)
    reservation = ModelRequestBudget(limits).reserve("")
    reservation.observe(usage)

    with pytest.raises(ModelRequestBudgetError, match=message):
        reservation.finish()


def test_model_request_budget_can_preserve_original_failure_after_overrun() -> None:
    reservation = ModelRequestBudget(_limits(request_limit=1)).reserve("")
    reservation.observe(InvocationUsage(requests=2))

    assert reservation.finish(raise_on_exceeded=False).requests == 2


def test_model_request_budget_reservations_are_atomic() -> None:
    budget = ModelRequestBudget(_limits(request_limit=2))
    barrier = threading.Barrier(3)
    outcomes: list[str] = []

    def reserve() -> None:
        barrier.wait()
        try:
            budget.reserve("")
        except ModelRequestBudgetError:
            outcomes.append("rejected")
        else:
            outcomes.append("accepted")

    threads = [threading.Thread(target=reserve) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(outcomes) == ["accepted", "accepted", "rejected"]


def test_request_token_estimator_handles_bytes_strings_json_and_cycles() -> None:
    cyclic: list[object] = []
    cyclic.append(cyclic)

    assert estimate_request_tokens(b"abc") == 2_051
    assert estimate_request_tokens("ç") == 2_050
    assert estimate_request_tokens({"b": 2, "a": 1}) > 2_048
    assert estimate_request_tokens(cyclic) > 2_048
