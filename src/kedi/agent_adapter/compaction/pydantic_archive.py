from __future__ import annotations

from collections import Counter
from dataclasses import replace
from typing import Any, cast

from pydantic_ai.capabilities import ProcessHistory
from pydantic_ai.messages import (
    BaseToolCallPart,
    BaseToolReturnPart,
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    ToolReturnPart,
)
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import RunContext

from .archive import (
    PROVIDER_INPUT_BASELINE_KEY as _PROVIDER_INPUT_BASELINE_KEY,
)
from .archive import (
    ArtifactHistoryProcessor,
    cache_epoch_key,
)
from .archive import (
    HistoryArchiveSettings as PydanticHistoryArchiveSettings,
)
from .archive import (
    ProviderInputPressure as _ProviderInputPressure,
)
from .pydantic_codec import PydanticHistoryCodec


class PydanticArtifactHistoryProcessor(ArtifactHistoryProcessor[ModelMessage]):
    def __init__(self, settings: PydanticHistoryArchiveSettings | None = None) -> None:
        super().__init__(PydanticHistoryCodec(), settings)

    def serialize(self, messages: list[ModelMessage]) -> object:
        return ModelMessagesTypeAdapter.dump_python(messages, mode="json")

    def provider_input_pressure(
        self, messages: list[ModelMessage]
    ) -> _ProviderInputPressure | None:
        return _provider_input_pressure(messages)

    def with_provider_input_baseline(
        self, message: ModelMessage, baseline_tokens: int
    ) -> ModelMessage:
        return _with_provider_input_baseline(message, baseline_tokens)

    def tool_outcomes(self, messages: list[ModelMessage]) -> tuple[str, ...]:
        return _tool_outcomes(messages)


class PydanticArtifactHistoryCapability(ProcessHistory[Any]):
    """Pydantic history hook plus a stable prompt-cache epoch per checkpoint."""

    def __init__(self, settings: PydanticHistoryArchiveSettings | None = None) -> None:
        self.archiver = PydanticArtifactHistoryProcessor(settings)
        super().__init__(processor=self.archiver)

    async def before_model_request(
        self,
        ctx: RunContext[Any],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        request_context = await super().before_model_request(ctx, request_context)
        current_settings = request_context.model_settings
        if current_settings is None:
            return request_context
        cache_key = current_settings.get("openai_prompt_cache_key")
        checkpoint_id = _latest_checkpoint_id(request_context.messages)
        if isinstance(cache_key, str) and cache_key and checkpoint_id is not None:
            settings = dict[str, Any](current_settings)
            settings["openai_prompt_cache_key"] = cache_epoch_key(cache_key, checkpoint_id)
            request_context.model_settings = cast("ModelSettings", settings)
        return request_context

    @classmethod
    def get_serialization_name(cls) -> None:
        return None


def _tool_outcomes(messages: list[ModelMessage]) -> tuple[str, ...]:
    calls: Counter[str] = Counter()
    results: Counter[tuple[str, str]] = Counter()
    for message in messages:
        if isinstance(message, ModelResponse):
            calls.update(
                part.tool_name for part in message.parts if isinstance(part, BaseToolCallPart)
            )
        elif isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, ToolReturnPart):
                    results[(part.tool_name, part.outcome)] += 1
                elif isinstance(part, BaseToolReturnPart):
                    results[(part.tool_name, "completed")] += 1
                elif isinstance(part, RetryPromptPart) and part.tool_name:
                    results[(part.tool_name, "retry")] += 1
    summaries: list[str] = []
    for tool_name in sorted(calls):
        statuses = ", ".join(
            f"{status}={count}"
            for (name, status), count in sorted(results.items())
            if name == tool_name
        )
        suffix = f"; {statuses}" if statuses else ""
        summaries.append(f"{tool_name}: calls={calls[tool_name]}{suffix}")
    return tuple(summaries)


def _latest_checkpoint_id(messages: list[ModelMessage]) -> str | None:
    for message in reversed(messages):
        metadata = message.metadata or {}
        checkpoint_id = metadata.get("kedi_compaction_checkpoint")
        if isinstance(checkpoint_id, str) and checkpoint_id:
            return checkpoint_id
    return None


def _provider_input_pressure(
    messages: list[ModelMessage],
) -> _ProviderInputPressure | None:
    baseline: int | None = None
    input_tokens: list[int] = []
    for message in messages:
        metadata = message.metadata or {}
        stored_baseline = metadata.get(_PROVIDER_INPUT_BASELINE_KEY)
        if (
            isinstance(stored_baseline, int)
            and not isinstance(stored_baseline, bool)
            and stored_baseline > 0
        ):
            baseline = stored_baseline
        if isinstance(message, ModelResponse) and (
            message.provider_name is not None or message.provider_url is not None
        ):
            current = message.usage.input_tokens
            if isinstance(current, int) and not isinstance(current, bool) and current > 0:
                input_tokens.append(current)
    if not input_tokens:
        return None
    if baseline is None:
        baseline = input_tokens[0]
    current = input_tokens[-1]
    return _ProviderInputPressure(
        baseline_tokens=baseline,
        current_tokens=current,
        growth_tokens=max(0, current - baseline),
    )


def _with_provider_input_baseline(
    message: ModelMessage,
    baseline_tokens: int,
) -> ModelMessage:
    metadata = dict(message.metadata or {})
    metadata[_PROVIDER_INPUT_BASELINE_KEY] = baseline_tokens
    return replace(message, metadata=metadata)


__all__ = [
    "PydanticArtifactHistoryCapability",
    "PydanticArtifactHistoryProcessor",
    "PydanticHistoryArchiveSettings",
]
