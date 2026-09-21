from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

KEDI_STRUCTURED_OUTPUT_EXTENSION_URI = (
    "https://docs.kedi-lang.org/a2a/extensions/structured-output/v1/"
)
KEDI_STRUCTURED_OUTPUT_METADATA_KEY = "kediStructuredOutput"
KEDI_STRUCTURED_OUTPUT_ARTIFACT_NAME = "kedi-structured-output"
KEDI_TEXT_ARTIFACT_NAME = "kedi-result"
KEDI_STRUCTURED_OUTPUT_VERSION = 1
DEFAULT_A2A_MAX_REQUEST_BYTES = 512 * 1024
DEFAULT_A2A_MAX_RESPONSE_BYTES = 4 * 1024 * 1024
# Keep the advertised transport limit aligned with subagent schema validation.
DEFAULT_A2A_MAX_SCHEMA_BYTES = 32 * 1024
DEFAULT_A2A_MAX_TASKS_PER_PRINCIPAL = 2_048


class A2AError(RuntimeError):
    """Base error for Kedi's A2A client and task lifecycle."""

    def __init__(
        self,
        message: str,
        *,
        task_id: str | None = None,
        context_id: str | None = None,
        state: str | None = None,
    ) -> None:
        super().__init__(message)
        self.task_id = task_id
        self.context_id = context_id
        self.state = state


class A2ARemoteTaskError(A2AError):
    """Raised when a remote A2A task reaches a non-success terminal state."""


class A2ATransportError(A2AError):
    """Raised when an A2A task can no longer be observed through the transport."""


class A2AAmbiguousSendError(A2ATransportError):
    """Raised when a send may have reached the peer but yielded no task handle."""


class A2ATaskTimeoutError(A2AError):
    """Raised when local observation reaches its deadline without cancelling the task."""


@dataclass(frozen=True, slots=True, kw_only=True)
class A2ARemoteResult:
    text: str | None
    data: Any | None
    task_id: str | None
    context_id: str | None
    state: str | None


@dataclass(frozen=True, slots=True, kw_only=True)
class A2ARemoteTaskSnapshot:
    """Provider-neutral view returned by A2A status and cancellation operations."""

    task_id: str
    context_id: str | None
    state: str | None
    text: str | None


def structured_output_request(schema: dict[str, Any]) -> dict[str, Any]:
    return {
        KEDI_STRUCTURED_OUTPUT_METADATA_KEY: {
            "version": KEDI_STRUCTURED_OUTPUT_VERSION,
            "schema": schema,
        }
    }


def structured_output_schema(metadata: dict[str, Any]) -> dict[str, Any] | None:
    value = metadata.get(KEDI_STRUCTURED_OUTPUT_METADATA_KEY)
    if not isinstance(value, dict) or value.get("version") != KEDI_STRUCTURED_OUTPUT_VERSION:
        return None
    schema = value.get("schema")
    return schema if isinstance(schema, dict) else None


def response_size(value: Any) -> int:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    except (TypeError, ValueError):
        encoded = repr(value).encode("utf-8", errors="replace")
    return len(encoded)
