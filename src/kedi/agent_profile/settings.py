from __future__ import annotations

import ast
from collections.abc import Mapping
from typing import Any, Final

SettingValue = Any
SettingsMap = Mapping[str, SettingValue]

CAPABILITY_SETTING_KEYS: Final[frozenset[str]] = frozenset({"codemode"})

PYDANTIC_MODEL_SETTING_KEYS: Final[frozenset[str]] = frozenset(
    {
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
        "api_key",
        "base_url",
        "callbacks",
        "configurable_fields",
        "configurable_prefix",
        "disable_streaming",
        "extra_body",
        "extra_headers",
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

MODEL_SETTING_KEYS: Final[frozenset[str]] = (
    PYDANTIC_MODEL_SETTING_KEYS
    | DSPY_LM_SETTING_KEYS
    | LANGCHAIN_MODEL_SETTING_KEYS
    | CAPABILITY_SETTING_KEYS
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


def parse_codemode_setting(value: SettingValue) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"on", "true"}:
            return True
        if lowered in {"off", "false"}:
            return False
    raise ValueError("codemode setting must be on/off or true/false")


def code_mode_enabled(settings: SettingsMap) -> bool:
    value = settings.get("codemode")
    if value is None:
        return False
    return parse_codemode_setting(value)


def pydantic_model_settings(settings: SettingsMap) -> dict[str, SettingValue]:
    return {key: value for key, value in settings.items() if key in PYDANTIC_MODEL_SETTING_KEYS}


def dspy_lm_settings(settings: SettingsMap) -> dict[str, SettingValue]:
    return {key: value for key, value in settings.items() if key in DSPY_LM_SETTING_KEYS}


def langchain_model_settings(settings: SettingsMap) -> dict[str, SettingValue]:
    return {key: value for key, value in settings.items() if key in LANGCHAIN_MODEL_SETTING_KEYS}
