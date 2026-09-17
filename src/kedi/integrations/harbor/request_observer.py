from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields, is_dataclass
from datetime import date, datetime, timezone
from datetime import time as datetime_time
from decimal import Decimal, InvalidOperation
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from pydantic_ai.capabilities import (
    AbstractCapability,
    CapabilityOrdering,
    WrapModelRequestHandler,
)
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter, ModelResponse
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.tools import RunContext
from pydantic_core import to_json

from ...agent_adapter.invocation import InvocationUsage
from ...agent_adapter.request import current_agent_run_request, request_diagnostics

_SCHEMA_VERSION = 1
_EVENT_FILENAME = "model-requests.jsonl"
_MAX_EVENT_BYTES = 128 * 1024
_MAX_EVENTS = 4_096
_MAX_RECORDED_MESSAGE_DIGESTS = 512
_MAX_CHANGED_MESSAGE_DETAILS = 16
_MAX_TOOL_NAMES = 256
_PUBLIC_ENDPOINT_HOSTS = frozenset(
    {
        "api.openai.com",
        "api.openrouter.ai",
        "chatgpt.com",
        "openrouter.ai",
    }
)
_HASHED_SETTING_PARTS = ("cache_key", "previous_response_id", "response_id")
_VISIBLE_SETTING_NAMES = frozenset(
    {
        "max_tokens",
        "parallel_tool_calls",
        "seed",
        "temperature",
        "timeout",
        "top_p",
        "openai_prompt_cache_retention",
        "openai_store",
    }
)


@dataclass(frozen=True, slots=True)
class _Fingerprint:
    digest: str
    byte_count: int


@dataclass(frozen=True, slots=True)
class _PartFingerprint:
    kind: str
    digest: str
    byte_count: int


@dataclass(frozen=True, slots=True)
class _MessageFingerprint:
    kind: str
    digest: str
    byte_count: int
    parts: tuple[_PartFingerprint, ...]


@dataclass(slots=True)
class _MessageCacheEntry:
    message: ModelMessage
    revision: object
    fingerprint: _MessageFingerprint


@dataclass(frozen=True, slots=True, kw_only=True)
class RequestEvidenceSummary:
    enabled: bool
    event_file: str | None
    events: int
    model_requests_started: int
    model_requests_completed: int
    model_requests_failed: int
    observer_failures: int
    dropped_events: int
    byte_count: int
    observer_duration_seconds: float
    transport_attempts_observed: bool = False

    @property
    def usable(self) -> bool:
        return (
            self.enabled
            and self.model_requests_started > 0
            and self.observer_failures == 0
            and self.dropped_events == 0
            and self.model_requests_started
            == self.model_requests_completed + self.model_requests_failed
        )

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "usable": self.usable}


@dataclass(frozen=True, slots=True, kw_only=True)
class RecoveredRequestUsage:
    usage: InvocationUsage
    invalid_events: int
    events_without_cost: int


def recover_request_usage(path: Path) -> RecoveredRequestUsage | None:
    """Recover completed request usage from append-only request evidence."""

    if not path.is_file():
        return None
    requests = 0
    tool_calls = 0
    input_tokens = 0
    cache_read_tokens = 0
    cache_write_tokens = 0
    output_tokens = 0
    total_tokens = 0
    cost_usd = Decimal(0)
    invalid_events = 0
    events_without_cost = 0
    try:
        lines = path.open(encoding="utf-8")
    except OSError:
        return None
    with lines:
        for line in lines:
            try:
                event = json.loads(line)
            except (TypeError, ValueError):
                invalid_events += 1
                continue
            if not isinstance(event, Mapping) or event.get("event") != "model_request_completed":
                continue
            response = event.get("response")
            if not isinstance(response, Mapping):
                invalid_events += 1
                continue
            usage = response.get("usage")
            values = _normalized_usage_values(usage)
            if values is None:
                invalid_events += 1
                continue
            requests += 1
            input_tokens += values["input_tokens"]
            cache_read_tokens += values["cache_read_tokens"]
            cache_write_tokens += values["cache_write_tokens"]
            output_tokens += values["output_tokens"]
            total_tokens += values["total_tokens"]
            parts = response.get("parts")
            if isinstance(parts, Sequence) and not isinstance(parts, (str, bytes, bytearray)):
                tool_calls += sum(
                    isinstance(part, Mapping) and part.get("kind") == "tool-call" for part in parts
                )
            recovered_cost = _nonnegative_decimal(
                usage.get("cost_usd") if isinstance(usage, Mapping) else None
            )
            if recovered_cost is None:
                events_without_cost += 1
            else:
                cost_usd += recovered_cost
    if requests == 0:
        return None
    return RecoveredRequestUsage(
        usage=InvocationUsage(
            requests=requests,
            tool_calls=tool_calls,
            input_tokens=input_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost_usd=None if events_without_cost else cost_usd,
        ),
        invalid_events=invalid_events,
        events_without_cost=events_without_cost,
    )


class PydanticRequestEvidenceCapability(AbstractCapability[Any]):
    """Record bounded request-shape evidence without changing model traffic."""

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position="innermost")

    def __init__(
        self,
        *,
        records_dir: Path,
        trial_id: str,
        manifest_digest: str | None = None,
    ) -> None:
        self._path = records_dir / _EVENT_FILENAME
        self._trial_id = trial_id
        self._manifest_digest = manifest_digest
        self._sequence = 0
        self._event_count = 0
        self._started = 0
        self._completed = 0
        self._failed = 0
        self._observer_failures = 0
        self._dropped_events = 0
        self._observer_duration_ns = 0
        self._last_request_started_ns: int | None = None
        self._previous_messages: tuple[_MessageFingerprint, ...] = ()
        self._message_cache: dict[int, _MessageCacheEntry] = {}

    @property
    def event_path(self) -> Path:
        return self._path

    def summary(self) -> RequestEvidenceSummary:
        try:
            byte_count = self._path.stat().st_size
        except OSError:
            byte_count = 0
        return RequestEvidenceSummary(
            enabled=True,
            event_file=self._path.name if self._path.is_file() else None,
            events=self._event_count,
            model_requests_started=self._started,
            model_requests_completed=self._completed,
            model_requests_failed=self._failed,
            observer_failures=self._observer_failures,
            dropped_events=self._dropped_events,
            byte_count=byte_count,
            observer_duration_seconds=self._observer_duration_ns / 1_000_000_000,
        )

    async def wrap_model_request(
        self,
        ctx: RunContext[Any],
        *,
        request_context: ModelRequestContext,
        handler: WrapModelRequestHandler,
    ) -> ModelResponse:
        self._sequence += 1
        sequence = self._sequence
        started_ns = time.perf_counter_ns()
        previous_started_ns = self._last_request_started_ns
        self._last_request_started_ns = started_ns
        self._started += 1
        request_snapshot = self._capture_request(
            ctx,
            request_context,
            sequence=sequence,
            since_previous_seconds=(
                None
                if previous_started_ns is None
                else max(0.0, (started_ns - previous_started_ns) / 1_000_000_000)
            ),
        )
        self._record(
            {
                "schema_version": _SCHEMA_VERSION,
                "event": "model_request_started",
                "recorded_at": _utc_now(),
                "sequence": sequence,
                **request_snapshot,
            }
        )
        try:
            response = await handler(request_context)
        except BaseException as exc:
            self._failed += 1
            self._record(
                {
                    "schema_version": _SCHEMA_VERSION,
                    "event": "model_request_failed",
                    "recorded_at": _utc_now(),
                    "sequence": sequence,
                    "duration_seconds": max(
                        0.0,
                        (time.perf_counter_ns() - started_ns) / 1_000_000_000,
                    ),
                    "error": {
                        "type": type(exc).__name__,
                        "cancelled": isinstance(exc, asyncio.CancelledError),
                    },
                }
            )
            raise
        self._completed += 1
        self._record(
            {
                "schema_version": _SCHEMA_VERSION,
                "event": "model_request_completed",
                "recorded_at": _utc_now(),
                "sequence": sequence,
                "duration_seconds": max(
                    0.0,
                    (time.perf_counter_ns() - started_ns) / 1_000_000_000,
                ),
                "response": _response_summary(response),
            }
        )
        return response

    def _capture_request(
        self,
        ctx: RunContext[Any],
        request_context: ModelRequestContext,
        *,
        sequence: int,
        since_previous_seconds: float | None,
    ) -> dict[str, Any]:
        try:
            messages = tuple(
                self._message_fingerprint(message) for message in request_context.messages
            )
            history = _history_delta(self._previous_messages, messages)
            self._previous_messages = messages
            parameters = request_context.model_request_parameters
            settings = request_context.model_settings or {}
            semantic_request = current_agent_run_request()
            return {
                "correlation": {
                    "trial_id": self._trial_id,
                    "manifest_digest": self._manifest_digest,
                    "kedi_run_id": None
                    if semantic_request is None
                    else semantic_request.metadata.run_id,
                    "kedi_turn_sequence": (
                        None
                        if semantic_request is None
                        else semantic_request.metadata.turn_sequence
                    ),
                    "pydantic_run_id": ctx.run_id,
                    "conversation_id_digest": _optional_digest(ctx.conversation_id),
                    "run_step": ctx.run_step,
                    "retry": ctx.retry,
                    "model_call_sequence": sequence,
                    "attempt_sequence": None,
                },
                "agent_request": (
                    None
                    if semantic_request is None
                    else asdict(request_diagnostics(semantic_request))
                ),
                "request": {
                    "model": _model_name(request_context),
                    "streaming": request_context.streaming,
                    "signature_digest": _digest_json(
                        {
                            "messages": [message.digest for message in messages],
                            "settings": _fingerprint(settings).digest,
                            "parameters": _fingerprint(parameters).digest,
                        }
                    ),
                    "settings": _settings_summary(settings),
                    "instructions": _sequence_summary(parameters.instruction_parts or ()),
                    "tools": _tools_summary(parameters),
                    "output": _output_summary(parameters),
                    "history": history,
                    "cache": {
                        "time_since_previous_call_seconds": since_previous_seconds,
                        "mode": _cache_mode(settings),
                        "settings": _cache_setting_summary(settings),
                        "kedi_cache_epoch": (
                            None
                            if semantic_request is None or semantic_request.conversation is None
                            else semantic_request.conversation.cache_epoch
                        ),
                    },
                    "transport": {
                        "observed": False,
                        "reason": "not available at the Pydantic model-request boundary",
                    },
                },
            }
        except Exception as exc:
            self._observer_failures += 1
            return {
                "correlation": {
                    "trial_id": self._trial_id,
                    "manifest_digest": self._manifest_digest,
                    "model_call_sequence": sequence,
                    "attempt_sequence": None,
                },
                "request": {
                    "capture_error": type(exc).__name__,
                    "transport": {
                        "observed": False,
                        "reason": "request capture failed before the transport boundary",
                    },
                },
            }

    def _message_fingerprint(self, message: ModelMessage) -> _MessageFingerprint:
        cache_key = id(message)
        revision = _revision_token(message)
        cached = self._message_cache.get(cache_key)
        if cached is not None and cached.message is message and cached.revision == revision:
            return cached.fingerprint
        payload = _canonical_message_json(
            ModelMessagesTypeAdapter.dump_json(
                [message],
                warnings="none",
                fallback=_unknown_value,
            )
        )
        part_fingerprints = tuple(_part_fingerprint(part) for part in message.parts)
        fingerprint = _MessageFingerprint(
            kind=str(message.kind),
            digest=hashlib.sha256(payload).hexdigest(),
            byte_count=len(payload),
            parts=part_fingerprints,
        )
        self._message_cache[cache_key] = _MessageCacheEntry(
            message=message,
            revision=revision,
            fingerprint=fingerprint,
        )
        return fingerprint

    def _record(self, event: Mapping[str, Any]) -> None:
        observed_at = time.perf_counter_ns()
        try:
            if self._event_count >= _MAX_EVENTS:
                self._dropped_events += 1
                return
            payload = (
                json.dumps(
                    event,
                    allow_nan=False,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n"
            ).encode("utf-8")
            if len(payload) > _MAX_EVENT_BYTES:
                self._dropped_events += 1
                payload = (
                    json.dumps(
                        {
                            "schema_version": _SCHEMA_VERSION,
                            "event": "observation_dropped",
                            "recorded_at": _utc_now(),
                            "sequence": event.get("sequence"),
                            "reason": "event_size_limit",
                            "original_byte_count": len(payload),
                            "original_digest": hashlib.sha256(payload).hexdigest(),
                        },
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                    + "\n"
                ).encode("utf-8")
            self._path.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(self._path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
            try:
                view = memoryview(payload)
                while view:
                    view = view[os.write(descriptor, view) :]
            finally:
                os.close(descriptor)
            self._event_count += 1
        except Exception:
            self._observer_failures += 1
        finally:
            self._observer_duration_ns += time.perf_counter_ns() - observed_at


def _history_delta(
    previous: Sequence[_MessageFingerprint],
    current: Sequence[_MessageFingerprint],
) -> dict[str, Any]:
    common_messages = 0
    for before, after in zip(previous, current, strict=False):
        if before.digest != after.digest:
            break
        common_messages += 1
    common_parts = 0
    first_changed_part: int | None = None
    if common_messages < len(previous) and common_messages < len(current):
        before_parts = previous[common_messages].parts
        after_parts = current[common_messages].parts
        for before, after in zip(before_parts, after_parts, strict=False):
            if before.digest != after.digest:
                break
            common_parts += 1
        if common_parts < max(len(before_parts), len(after_parts)):
            first_changed_part = common_parts
    unchanged_prefix_bytes = sum(item.byte_count for item in current[:common_messages])
    current_suffix = current[common_messages:]
    previous_suffix = previous[common_messages:]
    digests = [item.digest for item in current]
    if len(digests) > _MAX_RECORDED_MESSAGE_DIGESTS:
        half = _MAX_RECORDED_MESSAGE_DIGESTS // 2
        recorded_digests = [*digests[:half], *digests[-half:]]
        omitted_digests = len(digests) - len(recorded_digests)
    else:
        recorded_digests = digests
        omitted_digests = 0
    changed_details = [
        {
            "index": common_messages + offset,
            "kind": item.kind,
            "digest": item.digest,
            "byte_count": item.byte_count,
            "parts": [asdict(part) for part in item.parts],
        }
        for offset, item in enumerate(current_suffix[:_MAX_CHANGED_MESSAGE_DETAILS])
    ]
    return {
        "representation": "pydantic-model-message-transport-semantic-json",
        "message_count": len(current),
        "previous_message_count": len(previous),
        "serialized_bytes": sum(item.byte_count for item in current),
        "previous_serialized_bytes": sum(item.byte_count for item in previous),
        "sequence_digest": _digest_json(digests),
        "message_digests": recorded_digests,
        "omitted_message_digests": omitted_digests,
        "common_prefix_messages": common_messages,
        "common_prefix_parts_in_changed_message": common_parts,
        "unchanged_prefix_bytes": unchanged_prefix_bytes,
        "current_suffix_bytes": sum(item.byte_count for item in current_suffix),
        "previous_suffix_bytes": sum(item.byte_count for item in previous_suffix),
        "added_messages": max(0, len(current) - len(previous)),
        "removed_messages": max(0, len(previous) - len(current)),
        "first_changed": (
            None
            if common_messages == len(previous) == len(current)
            else {"message": common_messages, "part": first_changed_part}
        ),
        "changed_messages": changed_details,
        "omitted_changed_messages": max(0, len(current_suffix) - len(changed_details)),
    }


def _response_summary(response: ModelResponse) -> dict[str, Any]:
    usage = response.usage
    details = {
        str(key): value
        for key, value in sorted((usage.details or {}).items())
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    parts = tuple(_fingerprint(part) for part in response.parts)
    return {
        "model": response.model_name,
        "provider": response.provider_name,
        "provider_endpoint": _endpoint_category(response.provider_url),
        "provider_response_id_digest": _optional_digest(response.provider_response_id),
        "finish_reason": response.finish_reason,
        "state": response.state,
        "parts": [
            {
                "kind": str(getattr(part, "part_kind", type(part).__name__)),
                "digest": fingerprint.digest,
                "byte_count": fingerprint.byte_count,
            }
            for part, fingerprint in zip(response.parts, parts, strict=True)
        ],
        "usage": {
            "input_tokens": usage.input_tokens,
            "cache_read_tokens": usage.cache_read_tokens,
            "cache_write_tokens": usage.cache_write_tokens,
            "output_tokens": usage.output_tokens,
            "total_tokens": usage.total_tokens,
            "cost_usd": float(usage.cost) if usage.cost is not None else None,
            "details": details,
            "provenance": "pydantic_ai_normalized_response_usage",
        },
    }


def _normalized_usage_values(value: Any) -> dict[str, int] | None:
    if not isinstance(value, Mapping):
        return None
    result: dict[str, int] = {}
    for name in (
        "input_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "output_tokens",
        "total_tokens",
    ):
        item = value.get(name)
        if not isinstance(item, int) or isinstance(item, bool) or item < 0:
            return None
        result[name] = item
    return result


def _nonnegative_decimal(value: Any) -> Decimal | None:
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        result = Decimal(str(value))
    except InvalidOperation:
        return None
    return result if result.is_finite() and result >= 0 else None


def _part_fingerprint(part: Any) -> _PartFingerprint:
    fingerprint = _canonical_message_value_fingerprint(part)
    return _PartFingerprint(
        kind=str(getattr(part, "part_kind", type(part).__name__)),
        digest=fingerprint.digest,
        byte_count=fingerprint.byte_count,
    )


def _settings_summary(settings: Mapping[str, Any]) -> dict[str, Any]:
    fingerprint = _fingerprint(settings)
    visible: dict[str, Any] = {}
    for key, value in settings.items():
        name = str(key)
        if name in _VISIBLE_SETTING_NAMES and (
            value is None or isinstance(value, (bool, int, float, str))
        ):
            visible[name] = value
        elif any(part in name.lower() for part in _HASHED_SETTING_PARTS):
            visible[name + "_digest"] = _optional_digest(value)
    return {
        "keys": sorted(str(key) for key in settings),
        "digest": fingerprint.digest,
        "byte_count": fingerprint.byte_count,
        "visible": visible,
    }


def _cache_setting_summary(settings: Mapping[str, Any]) -> dict[str, Any]:
    relevant: dict[str, Any] = {}
    for key, value in settings.items():
        name = str(key)
        lowered = name.lower()
        if (
            "cache" not in lowered
            and "previous_response_id" not in lowered
            and name != "openai_store"
        ):
            continue
        if any(part in lowered for part in _HASHED_SETTING_PARTS):
            relevant[name + "_digest"] = _optional_digest(value)
        elif value is None or isinstance(value, (bool, int, float, str)):
            relevant[name] = value
        else:
            relevant[name + "_digest"] = _fingerprint(value).digest
    return relevant


def _cache_mode(settings: Mapping[str, Any]) -> str:
    if any("previous_response_id" in str(key).lower() and value for key, value in settings.items()):
        return "provider_continuation"
    return "full_message_replay"


def _sequence_summary(values: Sequence[Any]) -> dict[str, Any]:
    fingerprints = tuple(_fingerprint(value) for value in values)
    return {
        "count": len(values),
        "sequence_digest": _digest_json([item.digest for item in fingerprints]),
        "items": [asdict(item) for item in fingerprints],
    }


def _tools_summary(parameters: Any) -> dict[str, Any]:
    function_tools = tuple(parameters.function_tools)
    native_tools = tuple(parameters.native_tools)
    output_tools = tuple(parameters.output_tools)
    return {
        "function": _named_contract_summary(function_tools),
        "native": _named_contract_summary(native_tools),
        "output": _named_contract_summary(output_tools),
        "revealed_names": sorted(parameters.revealed_tool_names)[:_MAX_TOOL_NAMES],
        "revealed_names_omitted": max(
            0,
            len(parameters.revealed_tool_names) - _MAX_TOOL_NAMES,
        ),
    }


def _named_contract_summary(values: Sequence[Any]) -> dict[str, Any]:
    names = [str(getattr(value, "name", type(value).__name__)) for value in values]
    fingerprints = [_fingerprint(value).digest for value in values]
    return {
        "count": len(values),
        "names": names[:_MAX_TOOL_NAMES],
        "names_omitted": max(0, len(names) - _MAX_TOOL_NAMES),
        "ordered_contract_digest": _digest_json(fingerprints),
    }


def _output_summary(parameters: Any) -> dict[str, Any]:
    output_object = _fingerprint(parameters.output_object)
    prompted_template = _fingerprint(parameters.prompted_output_template)
    return {
        "mode": parameters.output_mode,
        "allow_text": parameters.allow_text_output,
        "allow_image": parameters.allow_image_output,
        "object_digest": output_object.digest,
        "object_bytes": output_object.byte_count,
        "prompted_template_digest": prompted_template.digest,
        "prompted_template_bytes": prompted_template.byte_count,
    }


def _model_name(request_context: ModelRequestContext) -> str:
    if request_context.model_id:
        return request_context.model_id
    for name in ("model_name", "system"):
        value = getattr(request_context.model, name, None)
        if isinstance(value, str) and value:
            return value
    return type(request_context.model).__name__


def _fingerprint(value: Any) -> _Fingerprint:
    try:
        payload = to_json(
            value,
            ensure_ascii=True,
            fallback=_unknown_value,
            serialize_unknown=True,
        )
    except Exception:
        payload = json.dumps(
            _unknown_value(value),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    return _Fingerprint(hashlib.sha256(payload).hexdigest(), len(payload))


def _canonical_message_value_fingerprint(value: Any) -> _Fingerprint:
    try:
        payload = to_json(
            value,
            ensure_ascii=True,
            fallback=_unknown_value,
            serialize_unknown=True,
        )
    except Exception:
        return _fingerprint(value)
    canonical = _canonical_message_json(payload)
    return _Fingerprint(hashlib.sha256(canonical).hexdigest(), len(canonical))


def _canonical_message_json(payload: bytes) -> bytes:
    value = json.loads(payload)
    canonical = _without_message_runtime_fields(value)
    return json.dumps(
        canonical,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _without_message_runtime_fields(value: Any) -> Any:
    if isinstance(value, list):
        return [_without_message_runtime_fields(item) for item in value]
    if not isinstance(value, Mapping):
        return value
    canonical = {str(key): item for key, item in value.items()}
    if "parts" in canonical and canonical.get("kind") in {"request", "response"}:
        canonical.pop("conversation_id", None)
        canonical.pop("run_id", None)
        canonical.pop("timestamp", None)
        parts = canonical.get("parts")
        if isinstance(parts, list):
            canonical["parts"] = [_without_message_runtime_fields(part) for part in parts]
    elif "part_kind" in canonical:
        canonical.pop("timestamp", None)
    return canonical


def _revision_token(value: Any, *, depth: int = 0) -> object:
    if value is None or isinstance(value, (bool, int, float, complex, Enum)):
        return (type(value), value)
    if isinstance(value, (str, bytes)):
        return (type(value), id(value), len(value))
    if isinstance(value, (datetime, date, datetime_time)):
        return (type(value), value)
    if depth >= 20:
        return (type(value), id(value))
    if is_dataclass(value) and not isinstance(value, type):
        return (
            type(value),
            tuple(
                (field.name, _revision_token(getattr(value, field.name), depth=depth + 1))
                for field in fields(value)
            ),
        )
    if isinstance(value, Mapping):
        return (
            type(value),
            len(value),
            tuple(
                (
                    _revision_token(key, depth=depth + 1),
                    _revision_token(item, depth=depth + 1),
                )
                for key, item in value.items()
            ),
        )
    if isinstance(value, (list, tuple)):
        return (
            type(value),
            len(value),
            tuple(_revision_token(item, depth=depth + 1) for item in value),
        )
    return (type(value), id(value))


def _unknown_value(value: Any) -> dict[str, str]:
    kind = type(value)
    return {"type": f"{kind.__module__}.{kind.__qualname__}"}


def _digest_json(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _optional_digest(value: Any) -> str | None:
    if value is None:
        return None
    return _fingerprint(value).digest


def _endpoint_category(value: str | None) -> str | None:
    if not value:
        return None
    try:
        parsed = urlsplit(value)
        port = f":{parsed.port}" if parsed.port is not None else ""
    except ValueError:
        return "configured"
    if not parsed.scheme or not parsed.hostname:
        return "configured"
    origin = f"{parsed.scheme}://{parsed.hostname}{port}"
    if parsed.hostname in _PUBLIC_ENDPOINT_HOSTS:
        return origin + parsed.path
    return f"custom:{_digest_json(origin + parsed.path)[:16]}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = ["PydanticRequestEvidenceCapability", "RequestEvidenceSummary"]
