from __future__ import annotations

import ast
from collections.abc import Mapping
from typing import Any, Final

SettingValue = Any
SettingsMap = Mapping[str, SettingValue]

PYDANTIC_MODEL_SETTING_KEYS: Final[frozenset[str]] = frozenset(
    {
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

DSPY_LM_SETTING_KEYS: Final[frozenset[str]] = frozenset(
    {
        "model_type",
        "temperature",
        "max_tokens",
        "cache",
        "callbacks",
        "num_retries",
        "provider",
        "finetuning_model",
        "launch_kwargs",
        "train_kwargs",
        "use_developer_role",
        "reasoning_effort",
    }
)

LANGCHAIN_MODEL_SETTING_KEYS: Final[frozenset[str]] = frozenset(
    {
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
    PYDANTIC_MODEL_SETTING_KEYS
    | DSPY_LM_SETTING_KEYS
    | LANGCHAIN_MODEL_SETTING_KEYS
    | CLAUDE_AGENT_SETTING_KEYS
    | CODEX_AGENT_SETTING_KEYS
    | ACP_AGENT_SETTING_KEYS
)


def is_valid_model_setting_key(name: str) -> bool:
    return name in MODEL_SETTING_KEYS


def parse_plain_setting_value(raw: str) -> SettingValue:
    value = raw.strip()
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    try:
        return ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return value


def pydantic_model_settings(settings: SettingsMap) -> dict[str, SettingValue]:
    return {key: value for key, value in settings.items() if key in PYDANTIC_MODEL_SETTING_KEYS}


def dspy_lm_settings(settings: SettingsMap) -> dict[str, SettingValue]:
    return {key: value for key, value in settings.items() if key in DSPY_LM_SETTING_KEYS}


def langchain_model_settings(settings: SettingsMap) -> dict[str, SettingValue]:
    return {key: value for key, value in settings.items() if key in LANGCHAIN_MODEL_SETTING_KEYS}


def claude_agent_settings(settings: SettingsMap) -> dict[str, SettingValue]:
    return {key: value for key, value in settings.items() if key in CLAUDE_AGENT_SETTING_KEYS}


def codex_agent_settings(settings: SettingsMap) -> dict[str, SettingValue]:
    return {key: value for key, value in settings.items() if key in CODEX_AGENT_SETTING_KEYS}


def acp_agent_settings(settings: SettingsMap) -> dict[str, SettingValue]:
    return {key: value for key, value in settings.items() if key in ACP_AGENT_SETTING_KEYS}
