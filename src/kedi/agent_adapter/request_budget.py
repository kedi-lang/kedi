from __future__ import annotations

import json
from dataclasses import dataclass
from threading import RLock
from typing import Any, Iterable, Mapping

from .invocation import InvocationUsage

_DEFAULT_CONTEXT_LIMIT_TOKENS = 200_000
_DEFAULT_SAFE_CONTEXT_UTILIZATION = 0.85
_DEFAULT_OUTPUT_RESERVE_TOKENS = 16_384
_DEFAULT_CUMULATIVE_INPUT_TOKENS = 1_000_000
_DEFAULT_REQUEST_LIMIT = 128
_REQUEST_OVERHEAD_TOKENS = 2_048


class ModelRequestBudgetError(RuntimeError):
    """Raised before another model request can exceed a host safety ceiling."""


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelRequestBudgetLimits:
    context_limit_tokens: int = _DEFAULT_CONTEXT_LIMIT_TOKENS
    safe_context_utilization: float = _DEFAULT_SAFE_CONTEXT_UTILIZATION
    output_reserve_tokens: int = _DEFAULT_OUTPUT_RESERVE_TOKENS
    cumulative_input_tokens: int = _DEFAULT_CUMULATIVE_INPUT_TOKENS
    request_limit: int = _DEFAULT_REQUEST_LIMIT

    def __post_init__(self) -> None:
        if self.context_limit_tokens <= 0:
            raise ValueError("context_limit_tokens must be positive")
        if not 0 < self.safe_context_utilization <= 1:
            raise ValueError("safe_context_utilization must be between zero and one")
        if self.output_reserve_tokens < 0:
            raise ValueError("output_reserve_tokens must not be negative")
        if self.cumulative_input_tokens <= 0:
            raise ValueError("cumulative_input_tokens must be positive")
        if self.request_limit <= 0:
            raise ValueError("request_limit must be positive")
        if self.safe_input_tokens <= 0:
            raise ValueError("output_reserve_tokens leaves no safe input context")

    @property
    def safe_input_tokens(self) -> int:
        utilized = int(self.context_limit_tokens * self.safe_context_utilization)
        return utilized - self.output_reserve_tokens


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelRequestBudgetSnapshot:
    requests: int
    cumulative_input_tokens: int
    estimated_request_tokens: int


class ModelRequestReservation:
    def __init__(
        self,
        budget: ModelRequestBudget,
        *,
        estimated_tokens: int,
    ) -> None:
        self._budget = budget
        self._estimated_tokens = estimated_tokens
        self._usage = InvocationUsage()
        self._closed = False

    def observe(self, usage: InvocationUsage) -> None:
        self._usage = self._usage + usage

    def finish(self, *, raise_on_exceeded: bool = True) -> ModelRequestBudgetSnapshot:
        if self._closed:
            raise RuntimeError("model request reservation is already closed")
        self._closed = True
        return self._budget._finish(
            estimated_tokens=self._estimated_tokens,
            usage=self._usage,
            raise_on_exceeded=raise_on_exceeded,
        )


class ModelRequestBudget:
    """Thread-safe host ceiling for all model requests in one outer Kedi run."""

    def __init__(self, limits: ModelRequestBudgetLimits | None = None) -> None:
        self.limits = limits or ModelRequestBudgetLimits()
        self._lock = RLock()
        self._requests = 0
        self._cumulative_input_tokens = 0

    def reserve(
        self,
        prompt: str,
        *,
        context: Iterable[Any] = (),
        resume_state: Mapping[str, Any] | None = None,
    ) -> ModelRequestReservation:
        estimated_tokens = estimate_request_tokens(
            prompt,
            *context,
            resume_state,
        )
        from kedi.artifacts.telemetry import model_context_preflight_span

        with model_context_preflight_span(
            estimated_input_tokens=estimated_tokens,
            safe_input_tokens=self.limits.safe_input_tokens,
            cumulative_input_limit=self.limits.cumulative_input_tokens,
            request_limit=self.limits.request_limit,
        ) as span:
            with self._lock:
                if estimated_tokens > self.limits.safe_input_tokens:
                    span.set_attribute("kedi.context.decision", "context_rejected")
                    raise ModelRequestBudgetError(
                        "estimated model request is "
                        f"{estimated_tokens} tokens, exceeding the safe context ceiling of "
                        f"{self.limits.safe_input_tokens} tokens"
                    )
                next_requests = self._requests + 1
                if next_requests > self.limits.request_limit:
                    span.set_attribute("kedi.context.decision", "request_limit_rejected")
                    raise ModelRequestBudgetError(
                        f"model request limit of {self.limits.request_limit} would be exceeded"
                    )
                next_input = self._cumulative_input_tokens + estimated_tokens
                if next_input > self.limits.cumulative_input_tokens:
                    span.set_attribute("kedi.context.decision", "cumulative_input_rejected")
                    raise ModelRequestBudgetError(
                        "cumulative model input would reach "
                        f"{next_input} tokens, exceeding the run limit of "
                        f"{self.limits.cumulative_input_tokens} tokens"
                    )
                self._requests = next_requests
                self._cumulative_input_tokens = next_input
                span.set_attribute("kedi.context.decision", "admitted")
                span.set_attribute("kedi.context.request_number", next_requests)
                span.set_attribute("kedi.context.reserved_cumulative_input_tokens", next_input)
        return ModelRequestReservation(self, estimated_tokens=estimated_tokens)

    def snapshot(self, *, estimated_request_tokens: int = 0) -> ModelRequestBudgetSnapshot:
        with self._lock:
            return ModelRequestBudgetSnapshot(
                requests=self._requests,
                cumulative_input_tokens=self._cumulative_input_tokens,
                estimated_request_tokens=estimated_request_tokens,
            )

    def _finish(
        self,
        *,
        estimated_tokens: int,
        usage: InvocationUsage,
        raise_on_exceeded: bool,
    ) -> ModelRequestBudgetSnapshot:
        actual_requests = max(1, usage.requests)
        actual_input_tokens = usage.input_tokens if usage.input_tokens > 0 else estimated_tokens
        with self._lock:
            self._requests += actual_requests - 1
            self._cumulative_input_tokens += actual_input_tokens - estimated_tokens
            snapshot = self.snapshot(estimated_request_tokens=estimated_tokens)
            exceeded = _exceeded_message(snapshot, self.limits)
        if exceeded is not None and raise_on_exceeded:
            raise ModelRequestBudgetError(exceeded)
        return snapshot


def estimate_request_tokens(*values: Any) -> int:
    """Conservatively estimate tokens without relying on a provider tokenizer."""

    byte_count = sum(len(_portable_bytes(value)) for value in values if value is not None)
    return _REQUEST_OVERHEAD_TOKENS + byte_count


def _portable_bytes(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("utf-8")
    try:
        text = json.dumps(value, ensure_ascii=False, default=str, sort_keys=True)
    except (TypeError, ValueError):
        text = str(value)
    return text.encode("utf-8")


def _exceeded_message(
    snapshot: ModelRequestBudgetSnapshot,
    limits: ModelRequestBudgetLimits,
) -> str | None:
    if snapshot.requests > limits.request_limit:
        return (
            f"model run used {snapshot.requests} requests, exceeding the host limit of "
            f"{limits.request_limit}"
        )
    if snapshot.cumulative_input_tokens > limits.cumulative_input_tokens:
        return (
            "model run used "
            f"{snapshot.cumulative_input_tokens} cumulative input tokens, exceeding the host "
            f"limit of {limits.cumulative_input_tokens}"
        )
    return None
