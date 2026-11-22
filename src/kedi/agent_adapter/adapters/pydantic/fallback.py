"""Kedi fallback policy on the SDK's per-request model selection capability."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any, cast

from pydantic_ai.capabilities import CapabilityOrdering, ModelFallback
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models import Model, ModelRequestContext
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import RunContext

from kedi.agent_adapter.request_retry import transient_provider_error

from .request_retry import RequestRetryCapability


def fallback_models(models: Sequence[Model | str] | None) -> tuple[Model | str, ...]:
    if models is None:
        return ()
    if isinstance(models, (str, bytes)) or not isinstance(models, Sequence):
        raise TypeError("fallback_models must be a sequence of models or model IDs")
    if any(not isinstance(model, (Model, str)) for model in models):
        raise TypeError("fallback_models must contain models or model IDs")
    if any(isinstance(model, str) and not model.strip() for model in models):
        raise ValueError("fallback_models must contain non-empty model IDs")
    return tuple(model.strip() if isinstance(model, str) else model for model in models)


class RequestFallbackCapability(ModelFallback[Any]):
    def __init__(self, models: Sequence[Model | str]) -> None:
        super().__init__(*models, fallback_on=transient_provider_error)

    def get_model(self) -> None:
        # An alternative never supplies an otherwise unresolved primary.
        return None

    def get_ordering(self) -> CapabilityOrdering:
        # Error hooks run inside-out: exhaust this candidate's existing retry
        # owner before the native fallback capability advances to another model.
        return CapabilityOrdering(wraps=[RequestRetryCapability])

    async def before_model_request(
        self, ctx: RunContext[Any], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        self._primary = request_context.model
        return request_context

    async def prepare_model_request(
        self, ctx: RunContext[Any], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        request_context = await super().prepare_model_request(ctx, request_context)
        primary, selected = self._primary, request_context.model
        while isinstance(primary, WrapperModel):
            primary = primary.wrapped
        while isinstance(selected, WrapperModel):
            selected = selected.wrapped
        if selected is primary:
            return request_context

        settings = dict(request_context.model_settings or {})
        for key in (
            "openai_previous_response_id",
            "openai_conversation_id",
            "google_cached_content",
        ):
            value = settings.get(key)
            if value is None:
                continue
            if value == "auto" and key != "google_cached_content":
                settings[key] = None
            else:
                raise ValueError(f"Model fallback cannot preserve server-owned state {key!r}")
        if selected.system != primary.system:
            prefix = f"{primary.system}_"
            incompatible = [
                key
                for key, value in settings.items()
                if key.startswith(prefix) and value is not None
            ]
            if incompatible:
                raise ValueError(
                    f"Model fallback cannot preserve provider settings: {', '.join(incompatible)}"
                )
        if any(
            isinstance(message, ModelResponse) and message.state == "suspended"
            for message in request_context.messages
        ):
            raise ValueError("Model fallback cannot move a suspended provider response")
        request_context.model_settings = cast(ModelSettings, settings) if settings else None
        # Keep portable content and tool results, but not a different model's
        # response/conversation IDs. Preparation never edits durable history.
        request_context.messages = [
            replace(message, provider_response_id=None, provider_details=None)
            if isinstance(message, ModelResponse)
            else message
            for message in request_context.messages
        ]
        return request_context
