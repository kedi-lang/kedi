"""Unmeasured usage is unknown, not zero (#197, Q1)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from pydantic_ai.usage import UsageLimits

from kedi.agent_adapter.adapters.claude.messages import _claude_invocation_usage
from kedi.agent_adapter.adapters.codex.adapter import _codex_invocation_usage
from kedi.agent_adapter.adapters.jsonrpc.adapter import _usage
from kedi.agent_adapter.adapters.langchain.messages import _langchain_invocation_usage
from kedi.agent_adapter.adapters.langchain.runtime import _require_no_token_limits
from kedi.agent_adapter.invocation import InvocationUsage
from kedi.agent_adapter.telemetry import _set_token_attributes
from kedi.errors import KediExecutionError

UNKNOWN = InvocationUsage.unmeasured()


def test_an_unmeasured_count_stays_unknown_through_sums() -> None:
    measured = InvocationUsage(requests=1, input_tokens=10, output_tokens=2, total_tokens=12)
    assert (InvocationUsage() + measured) == measured
    total = InvocationUsage() + measured + UNKNOWN
    assert total.requests == 2
    assert (total.input_tokens, total.output_tokens, total.total_tokens) == (None, None, None)
    assert UNKNOWN.uncached_input_tokens is None
    # An unknown cache count never lowers the input that was sent.
    assert InvocationUsage(input_tokens=10, cache_read_tokens=None).uncached_input_tokens == 10
    assert InvocationUsage(input_tokens=10, cache_read_tokens=4).uncached_input_tokens == 6


def test_a_bridge_count_it_left_out_is_unknown() -> None:
    assert _usage(None) == UNKNOWN
    assert _usage({"requests": 2, "inputTokens": 5, "outputTokens": 1}) == InvocationUsage(
        requests=2,
        input_tokens=5,
        cache_read_tokens=None,
        cache_write_tokens=None,
        output_tokens=1,
        total_tokens=6,
    )
    partial = _usage({"inputTokens": 5, "outputTokens": True, "totalTokens": -1})
    assert (partial.requests, partial.input_tokens) == (1, 5)
    assert (partial.output_tokens, partial.total_tokens) == (None, None)


def test_adapters_without_usage_report_unknown_counts() -> None:
    assert _langchain_invocation_usage(None) == UNKNOWN
    assert _langchain_invocation_usage({"messages": "text"}) == UNKNOWN
    no_metadata = SimpleNamespace(usage_metadata=None)
    assert _langchain_invocation_usage({"messages": [no_metadata]}) == UNKNOWN
    assert _claude_invocation_usage(SimpleNamespace(usage=None, num_turns=3)) == (
        InvocationUsage.unmeasured(requests=3)
    )
    turn: dict[str, Any] = {"toolCalls": 2}
    assert _codex_invocation_usage(SimpleNamespace(turn=turn)) == InvocationUsage.unmeasured(  # pyright: ignore[reportArgumentType]
        tool_calls=2
    )


def test_langchain_refuses_a_token_limit_it_cannot_count() -> None:
    _require_no_token_limits(UsageLimits(request_limit=3))
    with pytest.raises(KediExecutionError, match="total_tokens_limit cannot be enforced"):
        _require_no_token_limits(UsageLimits(total_tokens_limit=100))


def test_telemetry_leaves_unmeasured_counts_off_the_span() -> None:
    attributes: dict[str, Any] = {}
    span = SimpleNamespace(set_attribute=attributes.__setitem__)
    _set_token_attributes(span, UNKNOWN)  # pyright: ignore[reportArgumentType]
    assert attributes == {}
    _set_token_attributes(span, InvocationUsage(input_tokens=3, output_tokens=1))  # pyright: ignore[reportArgumentType]
    assert attributes["gen_ai.usage.input_tokens"] == 3
    assert attributes["kedi.usage.uncached_input_tokens"] == 3
