"""Scoped native model fallback, with portable replay and narrow recovery boundaries."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Any
from uuid import UUID

from langchain.agents.middleware import ModelFallbackMiddleware
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGenerationChunk, GenerationChunk

from kedi.agent_adapter.compaction.langchain_codec import _langchain_has_signed_reasoning
from kedi.agent_adapter.history_processing import HistoryProcessingError
from kedi.agent_adapter.request_retry import transient_provider_error
from kedi.agent_adapter.responses_session import model_responses_turn

_MODEL_KEY = "kedi_fallback_model"
_CONTINUATION = ("previous_response_id", "conversation", "conversation_id")
_CACHE = ("prompt_cache_key", "prompt_cache_retention", "cache_control")


@dataclass
class _StreamProgress:
    started: bool = False
    reset_continuation: bool = False


_STREAM: ContextVar[_StreamProgress | None] = ContextVar("kedi_fallback_stream", default=None)


def response_started() -> bool:
    state = _STREAM.get()
    return state is not None and state.started


def retain_fallback_continuation_reset(reset: bool) -> bool:
    state = _STREAM.get()
    if state is None:
        return reset
    state.reset_continuation |= reset
    return state.reset_continuation


class _ProgressHandler(AsyncCallbackHandler):
    def __init__(self, state: _StreamProgress) -> None:
        self.state = state

    async def on_llm_new_token(
        self,
        token: str | list[str | dict[str, Any]],
        *,
        chunk: GenerationChunk | ChatGenerationChunk | None = None,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        message = getattr(chunk, "message", None)
        if token or getattr(message, "tool_call_chunks", None):
            self.state.started = True

    async def on_stream_event(self, event: Any, **kwargs: Any) -> None:
        if event.get("type") in {"content-block-start", "content-block-delta"}:
            self.state.started = True


class _StopFallback(BaseException):
    """Carry terminal exceptions through the SDK's broad Exception handler."""

    def __init__(self, error: Exception) -> None:
        self.error = error


def _model_key(model: Any) -> str:
    identity = [
        type(model).__module__,
        type(model).__qualname__,
        *[
            str(getattr(model, name, None))
            for name in ("name", "model_name", "model", "model_id", "openai_api_base")
        ],
        id(getattr(model, "root_async_client", None)),
    ]
    return hashlib.sha256(json.dumps(identity).encode()).hexdigest()


def _without_cache(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _without_cache(item)
            for key, item in value.items()
            if key not in {*_CACHE, "cachePoint", "cache_point"}
        }
    if isinstance(value, list):
        return [_without_cache(item) for item in value]
    return value


def _portable_message(message: Any) -> Any:
    if _langchain_has_signed_reasoning(message):
        raise HistoryProcessingError("Model fallback cannot replay signed or encrypted history")
    content = message.content
    if isinstance(content, list):
        blocks = []
        for block in content:
            if isinstance(block, str):
                blocks.append(block)
                continue
            kind = block.get("type")
            if kind in {"cache_point", "cachePoint"} or "cachePoint" in block:
                continue
            if isinstance(message, AIMessage) and kind in {"tool_call", "tool_use"}:
                if not message.tool_calls:
                    raise HistoryProcessingError("Model fallback requires portable tool calls")
                continue
            if kind not in {"text", "image", "image_url", "audio", "input_audio", "video", "file"}:
                raise HistoryProcessingError(
                    f"Model fallback cannot replay provider-owned content block {kind!r}"
                )
            block = _without_cache(block)
            block.pop("id", None)
            blocks.append(block)
        content = blocks
    additional = {
        key: value for key, value in message.additional_kwargs.items() if key.startswith("kedi_")
    }
    if any(
        message.additional_kwargs.get(key)
        for key in ("reasoning", "reasoning_content", "thinking", "refusal")
    ):
        raise HistoryProcessingError(
            "Model fallback cannot replay provider-owned reasoning/refusal"
        )
    metadata = {key: value for key, value in message.response_metadata.items() if key != "id"}
    return message.model_copy(
        update={"content": content, "additional_kwargs": additional, "response_metadata": metadata}
    )


def prepare_model_history(request: Any, *, switching: bool = False) -> Any:
    """Replay full local history when its provider ownership differs from this model."""
    key = _model_key(request.model)
    foreign = any(
        isinstance(message, AIMessage)
        and message.response_metadata.get(_MODEL_KEY) not in (None, key)
        for message in request.messages
    )
    if not switching and not foreign:
        return request
    model = request.model
    _check_server_history(request)
    native = dict(getattr(model, "model_kwargs", {}) or {})
    settings = dict(request.model_settings)
    for name in (*_CONTINUATION, *_CACHE):
        native.pop(name, None)
        settings.pop(name, None)
    updates: dict[str, Any] = {}
    if hasattr(model, "model_kwargs"):
        updates["model_kwargs"] = native
    if getattr(model, "use_previous_response_id", False):
        updates["use_previous_response_id"] = False
    for name in (*_CONTINUATION, *_CACHE):
        if getattr(model, name, None) is not None:
            updates[name] = None
    if updates:
        model = model.model_copy(update=updates)
    return request.override(
        model=model,
        model_settings=settings,
        messages=[_portable_message(message) for message in request.messages],
        system_message=(
            _portable_message(request.system_message)
            if request.system_message is not None
            else None
        ),
    )


def _check_server_history(request: Any) -> None:
    native = getattr(request.model, "model_kwargs", {}) or {}
    references = any(
        request.model_settings.get(name) or native.get(name) or getattr(request.model, name, None)
        for name in _CONTINUATION
    )
    if references:
        raise HistoryProcessingError(
            "Model fallback cannot reconstruct server-only conversation history"
        )


def _check_contract(request: Any, model: BaseChatModel) -> None:
    profile = getattr(model, "profile", None) or {}
    structured = request.response_format is not None
    if request.tools and profile.get("tool_calling") is False:
        raise ValueError("A fallback model does not support the required tools")
    if (
        structured
        and profile.get("structured_output") is False
        and profile.get("tool_calling") is False
    ):
        raise ValueError("A fallback model does not support structured output")
    if (request.tools or structured) and type(model).bind_tools is BaseChatModel.bind_tools:
        raise ValueError("A fallback model cannot bind the required tools/output schema")
    flags = {
        "image": "image_inputs",
        "image_url": "image_inputs",
        "audio": "audio_inputs",
        "input_audio": "audio_inputs",
        "video": "video_inputs",
        "file": "pdf_inputs",
    }
    for message in request.messages:
        if isinstance(message, HumanMessage) and isinstance(message.content, list):
            for block in message.content:
                kind = block.get("type") if isinstance(block, Mapping) else None
                flag = flags.get(kind) if isinstance(kind, str) else None
                if flag is not None and profile.get(flag) is False:
                    raise ValueError("A fallback model does not support the required media")


class LangChainModelFallbackMiddleware(ModelFallbackMiddleware):
    """Native candidate ordering/cache sanitation, restricted to transient HTTP failures."""

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        for model in self.models:
            _check_contract(request, model)
        attempted: list[str] = []
        reset_continuation = False

        async def eligible(candidate: Any) -> Any:
            nonlocal reset_continuation
            state = _StreamProgress(reset_continuation=reset_continuation)
            token = _STREAM.set(state)
            try:
                switching = candidate.model is not request.model
                if switching and not state.reset_continuation:
                    _check_server_history(request)
                candidate = prepare_model_history(candidate, switching=switching)
                callback = _ProgressHandler(state)
                callbacks = getattr(candidate.model, "callbacks", None)
                if callbacks is None or isinstance(callbacks, list):
                    callbacks = [*(callbacks or []), callback]
                else:
                    callbacks = callbacks.copy()
                    callbacks.add_handler(callback)
                model = candidate.model.model_copy(update={"callbacks": callbacks})
                candidate = candidate.override(model=model)
                attempted.append(
                    str(
                        getattr(model, "model_name", None)
                        or getattr(model, "name", None)
                        or type(model).__name__
                    )
                )
                if switching:
                    async with model_responses_turn(model):
                        response = await handler(candidate)
                else:
                    response = await handler(candidate)
            except Exception as error:
                if state.started or not transient_provider_error(error):
                    raise _StopFallback(error) from error
                raise
            finally:
                reset_continuation |= state.reset_continuation
                _STREAM.reset(token)
            key = _model_key(model)
            return replace(
                response,
                result=[
                    message.model_copy(
                        update={"response_metadata": {**message.response_metadata, _MODEL_KEY: key}}
                    )
                    if isinstance(message, AIMessage)
                    else message
                    for message in response.result
                ],
            )

        try:
            return await super().awrap_model_call(request, eligible)
        except _StopFallback as stopped:
            raise stopped.error from None
        except Exception as error:
            error.add_note("LangChain fallback candidates exhausted: " + " -> ".join(attempted))
            raise
