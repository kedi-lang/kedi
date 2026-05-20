from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol


class ExecutorDebugExporter(Protocol):
    def export(self, event: "ExecutorDebugEvent") -> None: ...


@dataclass(slots=True)
class ExecutorDebugEvent:
    event: str
    timestamp: str
    executor: str
    step: str
    payload: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        event: str,
        *,
        executor: str,
        step: str,
        **payload: Any,
    ) -> "ExecutorDebugEvent":
        return cls(
            event=event,
            timestamp=datetime.now(timezone.utc).isoformat(),
            executor=executor,
            step=step,
            payload=_sanitize(payload),
        )


class MarkdownDebugExporter:
    """Append human-readable executor debug events to a Markdown file."""

    def __init__(
        self,
        output_path: str | Path | None = None,
        *,
        default_name: str = "kedi",
    ) -> None:
        self.output_path = _resolve_export_path(output_path, default_name=default_name)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def export(self, event: ExecutorDebugEvent) -> None:
        with self.output_path.open("a", encoding="utf-8") as handle:
            handle.write(_render_markdown_event(event))
            handle.write("\n\n")


def default_debug_export_path(
    default_name: str = "kedi",
    *,
    now: datetime | None = None,
) -> Path:
    timestamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%d_%H%M%S")
    base_name = _normalize_base_name(default_name)
    return Path.cwd() / f"{base_name}_debug_export_{timestamp}.md"


def _resolve_export_path(
    output_path: str | Path | None,
    *,
    default_name: str,
) -> Path:
    if output_path is None:
        return default_debug_export_path(default_name)
    return Path(output_path).expanduser()


def _normalize_base_name(name: str) -> str:
    candidate = Path(name.strip() or "kedi")
    if candidate.suffix == ".kedi":
        raw = f"{candidate.stem}_kedi"
    elif candidate.suffix:
        raw = candidate.stem
    else:
        raw = candidate.name
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", raw).strip("_").lower()
    return normalized or "kedi"


def _render_markdown_event(event: ExecutorDebugEvent) -> str:
    lines = [
        f"## {event.event.replace('_', ' ').title()}",
        f"- Time: `{event.timestamp}`",
        f"- Executor: `{event.executor}`",
        f"- Step: `{event.step}`",
    ]

    inline_fields = [
        ("execution_kind", "Execution kind"),
        ("function_name", "Function"),
        ("fallback_executor", "Fallback executor"),
        ("mode", "Mode"),
        ("error_type", "Error type"),
        ("error", "Error"),
        ("external_functions", "External functions"),
    ]
    for key, label in inline_fields:
        value = event.payload.get(key)
        if value in (None, "", {}, [], ()):
            continue
        lines.append(f"- {label}: `{value}`")

    block_fields = [
        ("code", "Code"),
        ("inputs", "Inputs"),
        ("local_env", "Local env"),
        ("env_map", "Env map"),
        ("args", "Args"),
        ("kwargs", "Kwargs"),
        ("result", "Result"),
        ("output", "Output"),
    ]
    for key, label in block_fields:
        value = event.payload.get(key)
        if value in (None, "", {}, [], ()):
            continue
        lines.append(f"\n### {label}\n```python\n{_format_block(value)}\n```")

    return "\n".join(lines)


def _format_block(value: Any) -> str:
    if isinstance(value, str):
        return value
    return repr(value)


def _sanitize(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_sanitize(item) for item in value]
    return repr(value)
