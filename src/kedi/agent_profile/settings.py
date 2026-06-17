from __future__ import annotations

import ast
from collections.abc import Mapping
from typing import Any, Final

SettingValue = Any
SettingsMap = Mapping[str, SettingValue]

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

MODEL_SETTING_KEYS: Final[frozenset[str]] = PYDANTIC_MODEL_SETTING_KEYS | DSPY_LM_SETTING_KEYS


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
