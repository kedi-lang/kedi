"""Adapter-neutral limits for one child-agent execution."""

from __future__ import annotations

from dataclasses import dataclass

MAX_SUBAGENT_REQUESTS = 50
MAX_SUBAGENT_TOOL_CALLS = 100
MAX_SUBAGENT_TOKENS = 1_000_000


@dataclass(frozen=True, slots=True, kw_only=True)
class SubagentUsageLimits:
    """Per-child adapter limits, separate from model generation settings."""

    request_limit: int = 8
    tool_calls_limit: int = 16
    input_tokens_limit: int | None = None
    output_tokens_limit: int | None = None
    total_tokens_limit: int | None = None

    def __post_init__(self) -> None:
        _validate_limit("request_limit", self.request_limit)
        _validate_limit("tool_calls_limit", self.tool_calls_limit)
        _validate_limit("input_tokens_limit", self.input_tokens_limit)
        _validate_limit("output_tokens_limit", self.output_tokens_limit)
        _validate_limit("total_tokens_limit", self.total_tokens_limit)

    def capped(self) -> SubagentUsageLimits:
        return SubagentUsageLimits(
            request_limit=min(self.request_limit, MAX_SUBAGENT_REQUESTS),
            tool_calls_limit=min(self.tool_calls_limit, MAX_SUBAGENT_TOOL_CALLS),
            input_tokens_limit=_capped_optional(self.input_tokens_limit),
            output_tokens_limit=_capped_optional(self.output_tokens_limit),
            total_tokens_limit=_capped_optional(self.total_tokens_limit),
        )

    def to_pydantic(self) -> object:
        from pydantic_ai import UsageLimits

        return UsageLimits(
            request_limit=self.request_limit,
            tool_calls_limit=self.tool_calls_limit,
            input_tokens_limit=self.input_tokens_limit,
            output_tokens_limit=self.output_tokens_limit,
            total_tokens_limit=self.total_tokens_limit,
            count_tokens_before_request=(
                self.input_tokens_limit is not None or self.total_tokens_limit is not None
            ),
        )


def _validate_limit(name: str, value: int | None) -> None:
    if value is not None and value < 1:
        raise ValueError(f"{name} must be at least 1")


def _capped_optional(value: int | None) -> int | None:
    return None if value is None else min(value, MAX_SUBAGENT_TOKENS)
