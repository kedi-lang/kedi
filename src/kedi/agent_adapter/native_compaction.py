from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..agent_profile.compaction import CompactionSettings, CompactionSettingsError

_ANTHROPIC_COMPACTION_BETA = "compact-2026-01-12"
_ANTHROPIC_DEFAULT_THRESHOLD = 150_000


def with_pydantic_native_compaction(
    existing: object | None,
    *,
    model: object,
    settings: CompactionSettings,
) -> list[Any]:
    """Apply Kedi's native compaction policy to Pydantic AI capabilities."""

    if existing is None:
        capabilities: list[Any] = []
    elif isinstance(existing, (list, tuple)):
        capabilities = list(existing)
    else:
        raise TypeError("Pydantic AI capabilities must be a list or tuple.")
    if settings.mode == "disabled":
        return capabilities

    from pydantic_ai.models.anthropic import AnthropicCompaction, AnthropicModel
    from pydantic_ai.models.openai import OpenAICompaction, OpenAIResponsesModel

    if isinstance(model, OpenAIResponsesModel):
        capability = OpenAICompaction(token_threshold=settings.threshold)
    elif isinstance(model, AnthropicModel):
        capability = (
            AnthropicCompaction()
            if settings.threshold is None
            else AnthropicCompaction(token_threshold=settings.threshold)
        )
    else:
        raise CompactionSettingsError(
            "native compaction for Pydantic AI requires OpenAIResponsesModel or "
            f"AnthropicModel; got {type(model).__name__}"
        )

    capabilities = [
        item
        for item in capabilities
        if not isinstance(item, (OpenAICompaction, AnthropicCompaction))
    ]
    capabilities.append(capability)
    return capabilities


def langchain_settings_with_native_compaction(
    base: Mapping[str, Any],
    *,
    model: str | object,
    settings: CompactionSettings,
) -> dict[str, Any]:
    """Add native OpenAI or Anthropic context management to LangChain settings."""

    result = dict(base)
    if settings.mode == "disabled":
        return result
    provider = _langchain_provider(model)
    if provider == "openai":
        _apply_langchain_openai_compaction(result, settings)
    elif provider == "anthropic":
        _apply_langchain_anthropic_compaction(result, settings)
    else:
        raise CompactionSettingsError(
            "native compaction for LangChain requires the OpenAI or Anthropic provider; "
            f"got {provider!r}"
        )
    return result


def _apply_langchain_openai_compaction(
    target: dict[str, Any],
    settings: CompactionSettings,
) -> None:
    configured = target.get("context_management")
    if configured is None:
        entries: list[dict[str, Any]] = []
    elif isinstance(configured, list) and all(isinstance(item, Mapping) for item in configured):
        entries = [dict(item) for item in configured]
    else:
        raise CompactionSettingsError("OpenAI context_management must be a list of mappings")
    entries = [item for item in entries if item.get("type") != "compaction"]
    compaction: dict[str, Any] = {"type": "compaction"}
    if settings.threshold is not None:
        compaction["compact_threshold"] = settings.threshold
    entries.append(compaction)
    target["context_management"] = entries


def _apply_langchain_anthropic_compaction(
    target: dict[str, Any],
    settings: CompactionSettings,
) -> None:
    configured = target.get("context_management")
    if configured is None:
        context_management: dict[str, Any] = {}
    elif isinstance(configured, Mapping):
        context_management = dict(configured)
    else:
        raise CompactionSettingsError("Anthropic context_management must be a mapping")
    configured_edits = context_management.get("edits")
    if configured_edits is None:
        edits: list[dict[str, Any]] = []
    elif isinstance(configured_edits, list) and all(
        isinstance(item, Mapping) for item in configured_edits
    ):
        edits = [dict(item) for item in configured_edits]
    else:
        raise CompactionSettingsError(
            "Anthropic context_management edits must be a list of mappings"
        )
    edits = [item for item in edits if item.get("type") != "compact_20260112"]
    edits.append(
        {
            "type": "compact_20260112",
            "trigger": {
                "type": "input_tokens",
                "value": settings.threshold or _ANTHROPIC_DEFAULT_THRESHOLD,
            },
        }
    )
    context_management["edits"] = edits
    target["context_management"] = context_management

    configured_betas = target.get("betas")
    if configured_betas is None:
        betas: list[str] = []
    elif (
        isinstance(configured_betas, Sequence)
        and not isinstance(configured_betas, (str, bytes))
        and all(isinstance(beta, str) for beta in configured_betas)
    ):
        betas = list(configured_betas)
    else:
        raise CompactionSettingsError("Anthropic betas must be a sequence of strings")
    if _ANTHROPIC_COMPACTION_BETA not in betas:
        betas.append(_ANTHROPIC_COMPACTION_BETA)
    target["betas"] = betas


def _langchain_provider(model: str | object) -> str:
    if isinstance(model, str):
        provider, separator, _ = model.partition(":")
        return provider if separator else "unknown"
    module = type(model).__module__
    name = type(model).__name__
    if module.startswith("langchain_openai") and name == "ChatOpenAI":
        return "openai"
    if module.startswith("langchain_anthropic") and name == "ChatAnthropic":
        return "anthropic"
    return f"{module}.{name}"
