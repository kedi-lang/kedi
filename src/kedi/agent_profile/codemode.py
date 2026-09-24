from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
from typing import Any, Final, Mapping, TypedDict

CODEMODE_LIMIT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "default_search_limit",
        "max_search_limit",
        "max_hydrated_tools",
        "max_discovery_result_bytes",
        "max_code_chars",
        "max_nested_calls",
        "max_concurrent_calls",
        "max_tool_result_bytes",
        "max_total_tool_result_bytes",
        "max_print_bytes",
        "request_timeout",
    }
)
CODEMODE_SETTING_FIELDS: Final[frozenset[str]] = CODEMODE_LIMIT_FIELDS | {
    "enabled",
    "preload_tools",
}


class CodeModeLimitValues(TypedDict):
    default_search_limit: int
    max_search_limit: int
    max_hydrated_tools: int
    max_discovery_result_bytes: int
    max_code_chars: int
    max_nested_calls: int
    max_concurrent_calls: int
    max_tool_result_bytes: int
    max_total_tool_result_bytes: int
    max_print_bytes: int
    request_timeout: float


@dataclass(frozen=True, slots=True)
class CodeModeSettings:
    """Kedi-owned CodeMode activation and execution limits."""

    enabled: bool = True
    preload_tools: tuple[str, ...] = ()
    default_search_limit: int = 10
    max_search_limit: int = 50
    max_hydrated_tools: int = 64
    max_discovery_result_bytes: int = 256_000
    max_code_chars: int = 20_000
    max_nested_calls: int = 64
    max_concurrent_calls: int = 8
    max_tool_result_bytes: int = 256_000
    max_total_tool_result_bytes: int = 1_000_000
    max_print_bytes: int = 256_000
    request_timeout: float = 60.0

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("codemode enabled must be a boolean")
        object.__setattr__(self, "preload_tools", _normalize_preload_tools(self.preload_tools))
        integer_fields = {
            name: getattr(self, name) for name in CODEMODE_LIMIT_FIELDS if name != "request_timeout"
        }
        invalid_types = [
            name
            for name, value in integer_fields.items()
            if isinstance(value, bool) or not isinstance(value, int)
        ]
        if invalid_types:
            raise TypeError(
                "codemode integer settings must be integers: " + ", ".join(sorted(invalid_types))
            )
        non_positive = [name for name, value in integer_fields.items() if value <= 0]
        if non_positive:
            raise ValueError("codemode limits must be positive: " + ", ".join(sorted(non_positive)))
        if isinstance(self.request_timeout, bool) or not isinstance(
            self.request_timeout, (int, float)
        ):
            raise TypeError("codemode request_timeout must be a number")
        if self.request_timeout <= 0:
            raise ValueError("codemode request_timeout must be positive")
        if self.default_search_limit > self.max_search_limit:
            raise ValueError("codemode default_search_limit cannot exceed max_search_limit")

    def limit_values(self) -> CodeModeLimitValues:
        return {
            "default_search_limit": self.default_search_limit,
            "max_search_limit": self.max_search_limit,
            "max_hydrated_tools": self.max_hydrated_tools,
            "max_discovery_result_bytes": self.max_discovery_result_bytes,
            "max_code_chars": self.max_code_chars,
            "max_nested_calls": self.max_nested_calls,
            "max_concurrent_calls": self.max_concurrent_calls,
            "max_tool_result_bytes": self.max_tool_result_bytes,
            "max_total_tool_result_bytes": self.max_total_tool_result_bytes,
            "max_print_bytes": self.max_print_bytes,
            "request_timeout": float(self.request_timeout),
        }


def normalize_codemode_settings(
    value: CodeModeSettings | Mapping[str, Any] | bool | None,
    *,
    enabled: bool = True,
) -> CodeModeSettings | None:
    if value is None:
        return None
    if isinstance(value, CodeModeSettings):
        return value
    if isinstance(value, bool):
        return CodeModeSettings(enabled=value)
    unknown = set(value) - CODEMODE_SETTING_FIELDS
    if unknown:
        raise ValueError("unknown codemode settings: " + ", ".join(sorted(unknown)))
    values = dict(value)
    values.setdefault("enabled", enabled)
    return CodeModeSettings(**values)


def _normalize_preload_tools(value: str | Sequence[str]) -> tuple[str, ...]:
    values: Sequence[str] = (value,) if isinstance(value, str) else value
    if not isinstance(values, Sequence) or isinstance(values, (bytes, bytearray)):
        raise TypeError("codemode preload_tools must be a tool name or a sequence of tool names")
    normalized: list[str] = []
    seen: set[str] = set()
    for name in values:
        if not isinstance(name, str):
            raise TypeError("codemode preload_tools entries must be strings")
        if not name:
            raise ValueError("codemode preload_tools entries cannot be empty")
        if name not in seen:
            normalized.append(name)
            seen.add(name)
    return tuple(normalized)


__all__ = [
    "CODEMODE_LIMIT_FIELDS",
    "CODEMODE_SETTING_FIELDS",
    "CodeModeLimitValues",
    "CodeModeSettings",
    "normalize_codemode_settings",
]
