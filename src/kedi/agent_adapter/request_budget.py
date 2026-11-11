from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from threading import RLock
from typing import Any, Generator, Iterable, Mapping, Protocol, runtime_checkable

from .invocation import InvocationUsage

_DEFAULT_CONTEXT_LIMIT_TOKENS = 200_000
_DEFAULT_SAFE_CONTEXT_UTILIZATION = 0.85
_DEFAULT_OUTPUT_RESERVE_TOKENS = 16_384
_DEFAULT_CUMULATIVE_INPUT_TOKENS = 1_000_000
_DEFAULT_REQUEST_LIMIT = 128
_REQUEST_OVERHEAD_TOKENS = 2_048
_LIMITS_OVERRIDE: ContextVar[ModelRequestBudgetLimits | None]
HISTORY_USAGE_KEY = "history_usage_checkpoint"


@dataclass(frozen=True, slots=True)
class HistoryUsageCheckpoint:
    """Measured replay size bound to one exact, successfully captured history."""

    state_key: str
    digest: str
    tokens: int


def capture_history_usage(
    state: dict[str, Any],
    state_key: str,
    *,
    input_tokens: int,
    output_tokens: int,
    trailing_messages: Any = (),
) -> None:
    state.pop(HISTORY_USAGE_KEY, None)
    if any(type(value) is not int or value <= 0 for value in (input_tokens, output_tokens)):
        return
    state[HISTORY_USAGE_KEY] = HistoryUsageCheckpoint(
        state_key,
        hashlib.sha256(_portable_bytes(state[state_key])).hexdigest(),
        input_tokens + output_tokens + estimate_content_tokens(trailing_messages),
    )


def measured_history_tokens(state: Mapping[str, Any], state_key: str) -> int | None:
    checkpoint = state.get(HISTORY_USAGE_KEY)
    if (
        isinstance(checkpoint, HistoryUsageCheckpoint)
        and checkpoint.state_key == state_key
        and checkpoint.digest == hashlib.sha256(_portable_bytes(state.get(state_key))).hexdigest()
    ):
        return checkpoint.tokens
    return None


def encode_history_usage_state(state: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(state)
    checkpoint = result.get(HISTORY_USAGE_KEY)
    if isinstance(checkpoint, HistoryUsageCheckpoint):
        result[HISTORY_USAGE_KEY] = asdict(checkpoint)
    return result


def decode_history_usage_state(state: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(state)
    if HISTORY_USAGE_KEY not in result:
        return result
    value = result[HISTORY_USAGE_KEY]
    if not isinstance(value, dict) or set(value) != {"state_key", "digest", "tokens"}:
        raise ValueError("Invalid stored history usage checkpoint")
    key, digest, tokens = value["state_key"], value["digest"], value["tokens"]
    if (
        key not in ("pydantic_messages", "langchain_messages")
        or key not in result
        or not isinstance(digest, str)
        or type(tokens) is not int
        or tokens <= 0
        or digest != hashlib.sha256(_portable_bytes(result[key])).hexdigest()
    ):
        raise ValueError("Stored history usage checkpoint does not match native history")
    result[HISTORY_USAGE_KEY] = HistoryUsageCheckpoint(key, digest, tokens)
    return result


class ModelRequestBudgetError(RuntimeError):
    """Raised before another model request can exceed a host safety ceiling."""


@runtime_checkable
class SupportsHistoryTokenEstimate(Protocol):
    def estimate_history_tokens(self, state: Mapping[str, Any]) -> int:
        """Estimate replayed model content, excluding framework-only metadata."""
        ...


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


_LIMITS_OVERRIDE = ContextVar("kedi_model_request_budget_limits", default=None)


@contextmanager
def use_model_request_budget_limits(
    limits: ModelRequestBudgetLimits,
) -> Generator[None, None, None]:
    """Apply host limits to budgets created in the current execution context."""

    token = _LIMITS_OVERRIDE.set(limits)
    try:
        yield
    finally:
        _LIMITS_OVERRIDE.reset(token)


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
        self.limits = limits or _LIMITS_OVERRIDE.get() or ModelRequestBudgetLimits()
        self._lock = RLock()
        self._requests = 0
        self._cumulative_input_tokens = 0

    def reserve(
        self,
        prompt: str,
        *,
        context: Iterable[Any] = (),
        resume_state: Mapping[str, Any] | None = None,
        history_tokens: int | None = None,
        media_tokens: int = 0,
    ) -> ModelRequestReservation:
        if history_tokens is not None and (
            not isinstance(history_tokens, int)
            or isinstance(history_tokens, bool)
            or history_tokens < 0
        ):
            raise ValueError("history_tokens must be a nonnegative integer")
        estimated_tokens = (
            estimate_request_tokens(
                prompt,
                *context,
                resume_state if history_tokens is None else None,
            )
            + (history_tokens or 0)
            + media_tokens
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
        actual_input_tokens = (
            usage.uncached_input_tokens if usage.input_tokens > 0 else estimated_tokens
        )
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

    return _REQUEST_OVERHEAD_TOKENS + estimate_content_tokens(*values)


def estimate_content_tokens(*values: Any) -> int:
    """Byte-based upper estimate for visible content, without request overhead."""
    return sum(len(_portable_bytes(value)) for value in values if value is not None)


_MEDIA_MIN_TOKENS = 258
_IMAGE_TOKENS = 1_600
# Bytes per token, rough upper bounds: PDFs carry text and page images, audio and video are
# sampled by the provider at a fixed token rate per second.
_MEDIA_BYTES_PER_TOKEN = {"document": 50, "audio": 500, "video": 1_000}


def estimate_media_tokens(kind: str, size: int | None) -> int:
    """Conservative token estimate for one binary attachment, without decoding it."""

    if kind == "image" or size is None:
        return _IMAGE_TOKENS
    return max(_MEDIA_MIN_TOKENS, size // _MEDIA_BYTES_PER_TOKEN.get(kind, 50))


def estimate_attachment_tokens(attachments: Iterable[Any]) -> int:
    """Estimate a prompt's attachments: text documents as text, other media by kind."""

    total = 0
    for attachment in attachments:
        if attachment.is_text and attachment.media.data is not None:
            total += len(attachment.inline_text().encode("utf-8"))
        else:
            total += len(attachment.label()) + estimate_media_tokens(
                attachment.media.kind, attachment.media.size
            )
    return total


def project_media_payloads(value: Any) -> tuple[Any, int]:
    """Replace inline media payloads in serialized messages with short placeholders.

    Base64 media would otherwise count as text in byte-based estimates and fill searchable
    history text. Returns the projected value and the media token estimate.
    """

    tokens = 0

    def placeholder(kind: str, encoded: str) -> str:
        nonlocal tokens
        size = len(encoded) * 3 // 4
        tokens += estimate_media_tokens(kind, size)
        return f"[{kind}: {size} bytes]"

    def visit(node: Any) -> Any:
        if isinstance(node, Mapping):
            key = _media_payload_key(node)
            return {
                name: (placeholder(_payload_kind(node), item) if name == key else visit(item))
                for name, item in node.items()
            }
        if isinstance(node, (list, tuple)):
            return [visit(item) for item in node]
        if isinstance(node, str) and node.startswith("data:") and ";base64," in node[:128]:
            media_type, encoded = node[5:].split(";base64,", 1)
            return placeholder(_kind_for_media_type(media_type), encoded)
        return node

    return visit(value), tokens


def _media_payload_key(node: Mapping[Any, Any]) -> str | None:
    # Pydantic AI BinaryContent, LangChain v1 data blocks and v0 base64 blocks.
    if node.get("kind") == "binary" and isinstance(node.get("data"), str):
        return "data"
    if node.get("type") in {"image", "audio", "video", "file"}:
        if isinstance(node.get("base64"), str):
            return "base64"
        if node.get("source_type") == "base64" and isinstance(node.get("data"), str):
            return "data"
    return None


def _payload_kind(node: Mapping[Any, Any]) -> str:
    block_type = node.get("type")
    if block_type in {"image", "audio", "video"}:
        return str(block_type)
    if block_type == "file":
        return "document"
    media_type = node.get("media_type") or node.get("mime_type") or ""
    return _kind_for_media_type(str(media_type))


def _kind_for_media_type(media_type: str) -> str:
    major = media_type.split("/", 1)[0].lower()
    return major if major in {"image", "audio", "video"} else "document"


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
            f"{snapshot.cumulative_input_tokens} cumulative uncached input tokens, "
            "exceeding the host "
            f"limit of {limits.cumulative_input_tokens}"
        )
    return None
