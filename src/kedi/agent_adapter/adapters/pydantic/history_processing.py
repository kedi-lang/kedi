from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from typing import Any, cast

from pydantic_ai.capabilities import AbstractCapability, ProcessHistory
from pydantic_ai.messages import ModelMessage, ModelRequest, ModelResponse
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import RunContext

from kedi.agent_adapter.compaction.pydantic_archive import (
    PydanticArtifactHistoryCapability,
)
from kedi.agent_adapter.compaction.pydantic_codec import PydanticHistoryCodec
from kedi.agent_adapter.history_processing import (
    HistoryProcessingError,
    HistoryProcessorRunner,
    rebase_history_cache_key,
)


class PydanticHistoryMessageCodec:
    adapter_shortname = "pydantic"
    state_key = "pydantic_messages"

    def __init__(self) -> None:
        self._codec = PydanticHistoryCodec()

    def clone(self, messages: Sequence[ModelMessage]) -> list[ModelMessage]:
        if not all(isinstance(message, (ModelRequest, ModelResponse)) for message in messages):
            raise TypeError("all values must be Pydantic ModelRequest or ModelResponse instances")
        return deepcopy(list(messages))

    def inspect(self, messages: Sequence[ModelMessage]):
        return self._codec.inspect({self.state_key: list(messages)})


class PydanticHistoryProcessingCapability(AbstractCapability[Any]):
    """Apply Kedi's validated user callback at Pydantic AI's durable history boundary."""

    def __init__(self, runner: HistoryProcessorRunner[ModelMessage]) -> None:
        self.runner = runner

    async def before_model_request(
        self,
        ctx: RunContext[Any],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        outcome = await self.runner.process(
            request_context.messages,
            model_id=_model_id(request_context),
        )
        request_context.messages = outcome.messages.copy()
        ctx.messages[:] = outcome.messages

        directive = self.runner.prepare_transport()
        if not directive.rewritten:
            return request_context

        settings = dict[str, Any](request_context.model_settings or {})
        if directive.reset_continuation or settings.get("openai_previous_response_id") != "auto":
            settings["openai_previous_response_id"] = None
        # A server-owned conversation cannot forget removed/reordered messages. Keep it
        # disabled for the rest of this run once native history has been rewritten.
        # Explicit None also overrides defaults on the native Model itself.
        settings["openai_conversation_id"] = None
        cache_key = settings.get("openai_prompt_cache_key")
        if directive.cache_key is not None and isinstance(cache_key, str) and cache_key:
            settings["openai_prompt_cache_key"] = rebase_history_cache_key(
                directive.cache_key,
                cache_key,
            )
        request_context.model_settings = cast("ModelSettings", settings)
        return request_context

    @classmethod
    def get_serialization_name(cls) -> None:
        return None


def reject_conflicting_history_capabilities(capabilities: object | None) -> None:
    if capabilities is None:
        return
    if not isinstance(capabilities, (list, tuple)):
        raise TypeError("Pydantic AI capabilities must be a list or tuple.")
    conflicting = [
        capability
        for capability in capabilities
        if isinstance(capability, ProcessHistory)
        and not isinstance(capability, PydanticArtifactHistoryCapability)
    ]
    if conflicting:
        names = ", ".join(type(capability).__name__ for capability in conflicting)
        raise HistoryProcessingError(
            "history_processor cannot be combined with another Pydantic ProcessHistory "
            f"capability ({names}); compose that logic inside the Kedi callback instead"
        )


def _model_id(request_context: ModelRequestContext) -> str | None:
    if isinstance(request_context.model_id, str):
        return request_context.model_id
    for attribute in ("model_name", "model_id"):
        value = getattr(request_context.model, attribute, None)
        if isinstance(value, str):
            return value
    return None
