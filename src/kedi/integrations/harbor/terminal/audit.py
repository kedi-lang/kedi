from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Sequence

from kedi.agent_adapter.hooks import (
    HookEvent,
    PostToolUseEvent,
    PostToolUseFailureEvent,
    PreToolUseRequest,
)
from kedi.integrations.harbor.records import is_secret_name, redact_text, utc_now
from kedi.integrations.harbor.terminal.output import (
    _TOOL_AUDIT_KEY_CHARS,
    _TOOL_AUDIT_MAX_DEPTH,
    _TOOL_AUDIT_MAX_ITEMS,
    _TOOL_AUDIT_PREVIEW_CHARS,
)

if TYPE_CHECKING:
    from kedi.integrations.harbor.terminal.session import TerminalSession


def _write_event(self: TerminalSession, event: dict[str, Any]) -> None:
    payload = {"timestamp": utc_now(), **event}
    line = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str) + "\n"
    with self._lock:
        with self._events_path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())


def _record_tool_event(self: TerminalSession, event: HookEvent) -> None:
    if not isinstance(event, (PreToolUseRequest, PostToolUseEvent, PostToolUseFailureEvent)):
        return
    try:
        payload: dict[str, Any] = {
            "timestamp": utc_now(),
            "event": event.event,
            "run_id": event.run_id,
            "parent_run_id": event.parent_run_id,
            "sequence": event.sequence,
            "adapter": event.adapter_shortname,
            "agent_name": event.agent_name,
            "profile_name": event.profile_name,
            "tool_call_id": event.tool_call_id,
            "tool_name": event.tool_name,
            "native_tool_name": event.native_tool_name,
            "origin": event.origin,
            "description": _audit_value(event.description),
            "metadata": _audit_value(event.metadata),
            "arguments": _audit_value(event.arguments),
        }
        if isinstance(event, PostToolUseEvent):
            payload["result"] = _audit_value(event.result)
        elif isinstance(event, PostToolUseFailureEvent):
            payload.update(
                {
                    "error_type": event.error_type,
                    "error_message": _audit_value(event.error_message),
                    "duration_seconds": event.duration_seconds,
                    "interrupted": event.interrupted,
                }
            )
        line = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str) + "\n"
        with self._lock:
            next_count = self._tool_event_count + 1
            with self._tool_events_path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
                if isinstance(event, PostToolUseFailureEvent) or next_count % 16 == 0:
                    os.fsync(handle.fileno())
            self._tool_event_count = next_count
            if isinstance(event, PostToolUseFailureEvent):
                self._tool_failure_count += 1
    except Exception:
        with self._lock:
            self._tool_audit_errors += 1


def _flush_tool_events(self: TerminalSession) -> None:
    try:
        if not self._tool_events_path.is_file():
            return
        with self._tool_events_path.open("ab") as handle:
            handle.flush()
            os.fsync(handle.fileno())
    except OSError:
        with self._lock:
            self._tool_audit_errors += 1


def _audit_value(value: Any, *, depth: int = 0) -> Any:
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return {"kind": "float", "value": str(value)}
    if isinstance(value, str):
        return _audit_text(value)
    if isinstance(value, bytes):
        half = _TOOL_AUDIT_PREVIEW_CHARS // 2
        preview = value if len(value) <= _TOOL_AUDIT_PREVIEW_CHARS else value[:half] + value[-half:]
        text = preview.decode("utf-8", errors="replace")
        return {
            "kind": "bytes",
            "size_bytes": len(value),
            "sha256": hashlib.sha256(value).hexdigest(),
            "preview": _audit_text(text),
            "truncated": len(value) > _TOOL_AUDIT_PREVIEW_CHARS,
        }
    if isinstance(value, os.PathLike):
        return _audit_text(os.fspath(value))
    if depth >= _TOOL_AUDIT_MAX_DEPTH:
        return {"kind": "depth_limit", "type": type(value).__name__}
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= _TOOL_AUDIT_MAX_ITEMS:
                break
            name = str(key)
            output[_audit_mapping_key(name)] = (
                "[REDACTED]" if is_secret_name(name) else _audit_value(item, depth=depth + 1)
            )
        if len(value) > _TOOL_AUDIT_MAX_ITEMS:
            output["__audit_truncated_items__"] = len(value) - _TOOL_AUDIT_MAX_ITEMS
        return output
    if isinstance(value, Sequence):
        captured = [
            _audit_value(value[index], depth=depth + 1)
            for index in range(min(len(value), _TOOL_AUDIT_MAX_ITEMS))
        ]
        if len(value) <= _TOOL_AUDIT_MAX_ITEMS:
            return captured
        return {
            "kind": "sequence",
            "total_items": len(value),
            "items": captured,
            "truncated": True,
        }
    return _audit_text(repr(value))


def _audit_mapping_key(value: str) -> str:
    redacted = redact_text(value)
    if len(redacted) <= _TOOL_AUDIT_KEY_CHARS:
        return redacted
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    marker = f"...[sha256:{digest}]..."
    preview_chars = _TOOL_AUDIT_KEY_CHARS - len(marker)
    head_chars = preview_chars // 2
    return redacted[:head_chars] + marker + redacted[-(preview_chars - head_chars) :]


def _audit_text(value: str) -> str | dict[str, Any]:
    if len(value) <= _TOOL_AUDIT_PREVIEW_CHARS:
        return redact_text(value)
    half = _TOOL_AUDIT_PREVIEW_CHARS // 2
    preview = (
        redact_text(value[:half]) + "\n... audit preview omitted ...\n" + redact_text(value[-half:])
    )
    digest = hashlib.sha256()
    size_bytes = 0
    for offset in range(0, len(value), 65_536):
        chunk = value[offset : offset + 65_536].encode("utf-8")
        digest.update(chunk)
        size_bytes += len(chunk)
    return {
        "kind": "text",
        "characters": len(value),
        "size_bytes": size_bytes,
        "sha256": digest.hexdigest(),
        "preview": preview,
        "truncated": True,
    }
