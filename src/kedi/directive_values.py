"""Shared value contracts for DSL parsing, runtime evaluation, and editor options."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ValueRule:
    description: str
    options: tuple[str, ...] = ()
    kind: str = "option"
    minimum: float | None = None
    aliases: tuple[str, ...] = ()
    exclusive_minimum: bool = False

    @property
    def completion_values(self) -> tuple[str, ...]:
        return ("true", "false", *self.options) if self.kind == "bool_or_option" else self.options

    def parse(self, raw: str) -> Any:
        if self.kind == "activation":
            if raw not in self.options:
                raise ValueError(
                    f"expected {' or '.join(self.options)} or a Python boolean expression"
                )
            return raw == "enabled"
        if self.kind == "bool":
            if raw not in {"true", "false"}:
                raise ValueError("must be 'true' or 'false' or a Python boolean expression")
            return raw == "true"
        if self.kind == "callable":
            raise ValueError("expected an inline Python callable reference")
        if self.kind == "collection":
            raise ValueError("lists and mappings require an inline Python expression")
        if self.kind == "python":
            raise ValueError("expected an inline Python expression")
        value = parse_scalar(raw)
        if self.kind in {"option", "string", "path"}:
            value = raw
        self.validate(value)
        return value

    def validate(self, value: Any) -> None:
        if self.kind in {"activation", "bool"}:
            if type(value) is not bool:
                raise ValueError("expected a boolean; Python expressions must return True or False")
        elif self.kind == "callable":
            if not callable(value):
                raise ValueError("expected a callable")
        elif self.kind == "python":
            if value is None:
                raise ValueError("expected a Python object, not a reset value")
        elif self.kind == "collection":
            if not isinstance(value, (Mapping, Sequence)) or isinstance(value, (str, bytes)):
                raise ValueError("expected a list or mapping")
        elif self.kind in {"int", "number"}:
            expected = (int,) if self.kind == "int" else (int, float)
            if isinstance(value, bool) or not isinstance(value, expected):
                raise ValueError(f"expected {'an integer' if self.kind == 'int' else 'a number'}")
            if self.minimum is not None:
                valid = value > self.minimum if self.exclusive_minimum else value >= self.minimum
                if not valid:
                    operator = ">" if self.exclusive_minimum else ">="
                    raise ValueError(f"expected a value {operator} {self.minimum:g}")
        elif self.kind == "string":
            if not isinstance(value, str):
                raise ValueError("expected a string")
        elif self.kind == "path":
            if not isinstance(value, (str, Path)):
                raise ValueError("expected a path string or Path")
        elif self.kind == "tool_choice":
            # Provider-specific structured selections are checked by the adapter.
            if value is None or isinstance(value, bool):
                raise ValueError("expected none, auto, required, or a structured tool selection")
            if isinstance(value, str) and value not in self.options:
                raise ValueError(f"expected one of: {', '.join(self.options)}")
        elif self.kind == "bool_or_option":
            if type(value) is not bool and value not in self.options:
                raise ValueError(f"expected true, false, or one of: {', '.join(self.options)}")
        elif value not in self.options and value not in self.aliases:
            raise ValueError(f"expected one of: {', '.join(self.options)}")


def parse_scalar(raw: str) -> Any:
    """Decode only explicit DSL scalars; never infer Python containers or null."""
    value = raw.strip()
    if value == "true":
        return True
    if value == "false":
        return False
    if value.startswith(("'", '"', "[", "{")):
        raise ValueError("quoted values, lists, and mappings require an inline Python expression")
    if re.fullmatch(r"[+-]?[0-9]+(?:_[0-9]+)*", value):
        return int(value)
    if re.fullmatch(
        r"[+-]?(?:[0-9]+(?:_[0-9]+)*(?:\.[0-9_]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?", value
    ):
        return float(value)
    return value


BOOLEAN = ValueRule("Boolean switch", ("true", "false"), "bool")
ACTIVATION = ValueRule("Enable or disable this feature", ("enabled", "disabled"), "activation")
CALLABLE = ValueRule("Application callback", kind="callable")
COLLECTION = ValueRule("Python list or mapping", kind="collection")
PYTHON_OBJECT = ValueRule("Application-provided Python object", kind="python")
POSITIVE_INT = ValueRule("Positive integer", kind="int", minimum=1)
INTEGER = ValueRule("Integer value", kind="int")
NONNEGATIVE_INT = ValueRule("Nonnegative integer", kind="int", minimum=0)
NUMBER = ValueRule("Numeric value", kind="number")
STRING = ValueRule("Text value", kind="string")
PATH = ValueRule("Filesystem path", kind="path")

DIRECTIVE_RULES = {
    **{name: ACTIVATION for name in ("history", "skills", "codemode", "hooks")},
    "defer": BOOLEAN,
    "effort": ValueRule(
        "Model reasoning effort", ("minimal", "low", "medium", "high", "xhigh", "max")
    ),
    "approval": ValueRule("Built-in tool approval policy", ("allow", "deny")),
    "workflow": ValueRule("Subagent orchestration mode", ("delegate", "dynamic")),
}

FIELD_RULES: dict[str, dict[str, ValueRule]] = {
    "hooks": {"enabled": BOOLEAN},
    "history": {
        "enabled": BOOLEAN,
        "compaction_mode": ValueRule("History compaction mode", ("native", "disabled")),
        "compaction_threshold": POSITIVE_INT,
        "processor": CALLABLE,
        "processor_condition": CALLABLE,
    },
    "skills": {
        "enabled": BOOLEAN,
        "include_registry": BOOLEAN,
        "include_all": BOOLEAN,
        "max_skills": POSITIVE_INT,
        "cwd": PATH,
        "exclude_paths": COLLECTION,
    },
    "codemode": {
        "enabled": BOOLEAN,
        **{
            name: POSITIVE_INT
            for name in (
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
            )
        },
        "request_timeout": ValueRule(
            "Positive timeout in seconds", kind="number", minimum=0, exclusive_minimum=True
        ),
    },
    "artifacts": {
        "enabled": BOOLEAN,
        "query_artifacts": ACTIVATION,
        "store": ValueRule("Artifact storage backend", ("memory", "file")),
        "path": PATH,
        "exclude_tools": COLLECTION,
        **{name: NONNEGATIVE_INT for name in ("preview_chars", "read_max_chars")},
        "max_artifacts": POSITIVE_INT,
    },
    "mcp": {
        "transport": ValueRule("MCP transport", ("stdio", "sse", "http", "streamable-http")),
        "command": STRING,
        "url": STRING,
        "args": COLLECTION,
        "env": COLLECTION,
        "headers": COLLECTION,
    },
    "auth": {"scheme": ValueRule("A2A authentication scheme", ("basic", "bearer", "api_key"))},
    "tool": {"risk": ValueRule("Tool side-effect risk", ("read_only", "mutating", "sensitive"))},
    "budget": {"request_limit": POSITIVE_INT, "tool_attempt_limit": POSITIVE_INT},
    "settings": {
        "tool_reason": ACTIVATION,
        "callbacks": PYTHON_OBJECT,
        "rate_limiter": PYTHON_OBJECT,
        "seed": INTEGER,
        "max_turns": INTEGER,
        **{
            name: STRING
            for name in (
                "api_key",
                "base_url",
                "organization",
                "fallback_model",
                "model_provider",
                "model_type",
                "reasoning_effort",
                "personality",
                "summary",
                "permission_mode",
            )
        },
        "cwd": PATH,
        **{
            name: BOOLEAN
            for name in (
                "parallel_tool_calls",
                "response_cache",
                "streaming",
                "stream_usage",
                "cache",
                "use_developer_role",
            )
        },
        "max_tokens": INTEGER,
        "response_cache_ttl": POSITIVE_INT,
        **{name: NONNEGATIVE_INT for name in ("max_retries", "num_retries")},
        **{
            name: NUMBER
            for name in (
                "temperature",
                "top_p",
                "presence_penalty",
                "frequency_penalty",
                "decision_threshold",
                "decision_tool_call_threshold",
                "typesafe_threshold",
                "typesafe_tool_call_threshold",
            )
        },
        **{
            name: COLLECTION
            for name in (
                "stop_sequences",
                "extra_headers",
                "extra_body",
                "logit_bias",
                "model_kwargs",
                "metadata",
                "tags",
                "env",
                "config",
                "allowed_tools",
                "disallowed_tools",
                "tools",
                "launch_kwargs",
                "train_kwargs",
            )
        },
    },
}

# These domains belong to the selected SDK. Do not apply them to other adapters.
ADAPTER_RULES = {
    "pydantic": {
        "tool_choice": ValueRule(
            "Tool selection policy", ("none", "auto", "required"), "tool_choice"
        ),
        "thinking": ValueRule(
            "Thinking budget", ("minimal", "low", "medium", "high", "xhigh"), "bool_or_option"
        ),
        "service_tier": ValueRule("Provider service tier", ("auto", "default", "flex", "priority")),
    },
    "langchain": {
        "disable_streaming": ValueRule(
            "Disable streaming globally or during tool calls", ("tool_calling",), "bool_or_option"
        ),
    },
    "codex": {
        "sandbox": ValueRule(
            "Filesystem sandbox policy",
            (
                "read-only",
                "workspace-write",
                "danger-full-access",
            ),
            aliases=(
                "readOnly",
                "read_only",
                "workspaceWrite",
                "workspace_write",
                "dangerFullAccess",
                "danger_full_access",
            ),
        ),
    },
}


def field_rule(scope: str, name: str, *, adapter: str | None = None) -> ValueRule | None:
    if scope == "settings" and adapter in ADAPTER_RULES:
        rule = ADAPTER_RULES[adapter].get(name)
        if rule is not None:
            return rule
    return FIELD_RULES.get(scope, {}).get(name)


def parse_field(scope: str, name: str, raw: str, *, adapter: str | None = None) -> Any:
    rule = field_rule(scope, name, adapter=adapter)
    try:
        return rule.parse(raw) if rule else parse_scalar(raw)
    except ValueError as exc:
        raise ValueError(f"{scope}.{name}: {exc}") from None


def validate_field(scope: str, name: str, value: Any, *, adapter: str | None = None) -> Any:
    rule = field_rule(scope, name, adapter=adapter)
    if rule is not None:
        try:
            rule.validate(value)
        except ValueError as exc:
            raise ValueError(f"{scope}.{name}: {exc}") from None
    elif value is None:
        raise ValueError(
            f"{scope}.{name}: None is not a reset value; omit the field to use its default or inherited value"
        )
    return value
