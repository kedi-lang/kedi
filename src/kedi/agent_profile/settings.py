from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from kedi.directive_values import ADAPTER_RULES, parse_scalar, validate_field

SettingValue = Any
SettingsMap = Mapping[str, SettingValue]

KEDI_RUNTIME_SETTING_KEYS: Final[frozenset[str]] = frozenset({"tool_reason"})

PYDANTIC_MODEL_SETTING_KEYS: Final[frozenset[str]] = frozenset(
    {
        "decision_threshold",
        "decision_tool_call_threshold",
        "typesafe_threshold",
        "typesafe_tool_call_threshold",
        "max_tokens",
        "temperature",
        "top_p",
        "timeout",
        "parallel_tool_calls",
        "tool_choice",
        "seed",
        "presence_penalty",
        "frequency_penalty",
        "logit_bias",
        "stop_sequences",
        "extra_headers",
        "thinking",
        "service_tier",
        "extra_body",
        "response_cache",
        "response_cache_ttl",
    }
)


LANGCHAIN_MODEL_SETTING_KEYS: Final[frozenset[str]] = frozenset(
    {
        "decision_threshold",
        "decision_tool_call_threshold",
        "typesafe_threshold",
        "typesafe_tool_call_threshold",
        "api_key",
        "base_url",
        "callbacks",
        "configurable_fields",
        "configurable_prefix",
        "disable_streaming",
        "extra_body",
        "extra_headers",
        "response_cache",
        "response_cache_ttl",
        "frequency_penalty",
        "logit_bias",
        "max_retries",
        "max_tokens",
        "metadata",
        "model_kwargs",
        "organization",
        "parallel_tool_calls",
        "presence_penalty",
        "rate_limiter",
        "reasoning",
        "reasoning_effort",
        "seed",
        "stop",
        "stop_sequences",
        "stream_usage",
        "streaming",
        "tags",
        "temperature",
        "timeout",
        "tool_choice",
        "top_p",
    }
)

CLAUDE_AGENT_SETTING_KEYS: Final[frozenset[str]] = frozenset(
    {
        "allowed_tools",
        "cwd",
        "disallowed_tools",
        "env",
        "fallback_model",
        "max_budget_usd",
        "max_turns",
        "permission_mode",
        "tools",
    }
)

CODEX_AGENT_SETTING_KEYS: Final[frozenset[str]] = frozenset(
    {
        "approval_policy",
        "config",
        "cwd",
        "model_provider",
        "personality",
        "sandbox",
        "service_tier",
        "summary",
        "timeout",
    }
)

ACP_AGENT_SETTING_KEYS: Final[frozenset[str]] = frozenset(
    {
        "cwd",
        "env",
        "timeout",
    }
)

MODEL_SETTING_KEYS: Final[frozenset[str]] = (
    KEDI_RUNTIME_SETTING_KEYS
    | PYDANTIC_MODEL_SETTING_KEYS
    | LANGCHAIN_MODEL_SETTING_KEYS
    | CLAUDE_AGENT_SETTING_KEYS
    | CODEX_AGENT_SETTING_KEYS
    | ACP_AGENT_SETTING_KEYS
)


def is_valid_model_setting_key(name: str) -> bool:
    return name in MODEL_SETTING_KEYS


def tool_reason_enabled(settings: SettingsMap, *, default: bool = False) -> bool:
    value = settings.get("tool_reason", default)
    if not isinstance(value, bool):
        raise TypeError(
            "tool_reason must be a boolean; Python expressions must return True or False"
        )
    return value


def parse_plain_setting_value(raw: str) -> SettingValue:
    return parse_scalar(raw)


def pydantic_model_settings(settings: SettingsMap) -> dict[str, SettingValue]:
    return _model_settings(settings, PYDANTIC_MODEL_SETTING_KEYS, "pydantic")


def langchain_model_settings(settings: SettingsMap) -> dict[str, SettingValue]:
    return _model_settings(settings, LANGCHAIN_MODEL_SETTING_KEYS, "langchain")


def claude_agent_settings(settings: SettingsMap) -> dict[str, SettingValue]:
    return _model_settings(settings, CLAUDE_AGENT_SETTING_KEYS, "claude")


def codex_agent_settings(settings: SettingsMap) -> dict[str, SettingValue]:
    return _model_settings(settings, CODEX_AGENT_SETTING_KEYS, "codex")


def acp_agent_settings(settings: SettingsMap) -> dict[str, SettingValue]:
    return _model_settings(settings, ACP_AGENT_SETTING_KEYS, "acp")


def _model_settings(
    settings: SettingsMap, keys: frozenset[str], adapter: str
) -> dict[str, SettingValue]:
    # DSL-owned types are checked during evaluation, not imposed on native SDK
    # objects. Resolve SDK option domains here against the final active adapter.
    rules = ADAPTER_RULES.get(adapter, {})
    return {
        key: (
            validate_field("settings", key, value, adapter=adapter)
            if key in rules and value is not None
            else value
        )
        for key, value in settings.items()
        if key in keys
    }
