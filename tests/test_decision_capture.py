from __future__ import annotations

import asyncio
from contextvars import copy_context
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from kedi import bind, capture_decisions, query
from kedi.decisions import publish_decision_record
from tests.mock_adapter import MockAdapter
from tests.test_decisions import DecisionAdapter, FailingDecisionAdapter, _info_from_metadata


def test_query_returns_plain_value_with_inspectable_evidence() -> None:
    adapter = DecisionAdapter()

    @query(adapter=adapter)
    def review(ticket: str) -> bool:
        """kedi
        >> Does <ticket> need review? [approved: bool]
        = `approved`
        """

    with capture_decisions() as run:
        assert review("charged twice") is True
    assert len(adapter.request_calls) == 1
    (decision,) = run.decisions
    assert decision.field_path == ("approved",)
    assert decision.probability == 0.92
    assert decision.threshold == 0.85
    assert decision.matches_inputs(ticket="charged twice")
    assert not decision.matches_inputs(ticket="changed")
    with pytest.raises(FrozenInstanceError):
        decision.field_path = ("changed",)  # type: ignore[misc]
    review("outside")
    assert len(run.decisions) == 1


def test_bind_captures_each_field_of_one_call(tmp_path: Path) -> None:
    path = tmp_path / "review.kedi"
    path.write_text(
        ">> Is it approved [approved: bool] and ready [ready: bool]?\n= `approved and ready`\n"
    )

    @bind(file=path, adapter=DecisionAdapter())
    def review() -> bool: ...

    with capture_decisions() as run:
        assert review() is True
    assert [item.field_path for item in run.decisions] == [("approved",), ("ready",)]
    assert run.decisions[0].call_id == run.decisions[1].call_id


def test_nested_scopes_and_failures_restore_capture() -> None:
    @query(adapter=DecisionAdapter())
    def review() -> bool:
        """kedi
        >> Approved [approved: bool]
        = `approved`
        """

    @query(adapter=FailingDecisionAdapter())
    def fail() -> bool:
        """kedi
        >> Approved [approved: bool]
        = `approved`
        """

    with capture_decisions() as outer:
        review()
        with pytest.raises(Exception, match="provider failed"):
            with capture_decisions() as inner:
                review()
                fail()
        review()
    assert len(outer.decisions) == 3
    assert len(inner.decisions) == 1
    assert inner.decisions[0].call_id == outer.decisions[1].call_id


def test_async_calls_are_isolated_and_parent_collects_children() -> None:
    @query(adapter=DecisionAdapter())
    async def review(ticket: str) -> bool:
        """kedi
        >> Review <ticket>: [approved: bool]
        = `approved`
        """

    async def one(ticket: str):
        with capture_decisions() as run:
            assert await review(ticket) is True
        return run

    async def exercise():
        with capture_decisions() as outer:
            first, second = await asyncio.gather(one("first"), one("second"))
        assert len(first.decisions) == len(second.decisions) == 1
        assert first.decisions[0].matches_inputs(ticket="first")
        assert second.decisions[0].matches_inputs(ticket="second")
        assert len(outer.decisions) == 2

    asyncio.run(exercise())


def test_non_jev_and_cache_hits_do_not_invent_decisions() -> None:
    @query(adapter=MockAdapter())
    def plain() -> bool:
        """kedi
        >> Approved [approved: bool]
        = `approved`
        """

    adapter = DecisionAdapter()

    @query(adapter=adapter, cache=True)
    def cached() -> bool:
        """kedi
        >> Approved [approved: bool]
        = `approved`
        """

    with capture_decisions() as run:
        assert plain() is True
        assert cached() is True
    assert len(run.decisions) == 1
    with capture_decisions() as hit:
        assert cached() is True
    assert hit.decisions == ()
    assert len(adapter.request_calls) == 1


def test_closed_capture_ignores_late_inherited_work() -> None:
    record = _info_from_metadata({}, "approved")._record
    with capture_decisions() as run:
        inherited = copy_context()
    inherited.run(publish_decision_record, record)
    assert run.decisions == ()
