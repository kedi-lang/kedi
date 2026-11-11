from __future__ import annotations

from collections.abc import AsyncGenerator, Callable, Generator, Sequence
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from typing import Any, Mapping

from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import RunContext

from kedi.agent_adapter.history_provenance import LINEAGE_STATE_KEY, HistoryLineage
from kedi.agent_adapter.request_budget import (
    HISTORY_USAGE_KEY,
    HistoryUsageCheckpoint,
    estimate_content_tokens,
    measured_history_tokens,
    project_media_payloads,
)
from kedi.run_budget import (
    RunBudgetExceeded,
    admit_run_operation,
    assert_no_hidden_model_retries,
    current_run_budget,
)

_MODEL_ADMISSION: ContextVar[Callable[[Sequence[ModelMessage]], None] | None] = ContextVar(
    "kedi_pydantic_model_admission", default=None
)


def estimate_history_tokens(state: Mapping[str, Any]) -> int:
    from pydantic_ai.messages import ModelMessagesTypeAdapter, ModelRequest

    try:
        messages = ModelMessagesTypeAdapter.validate_python(state.get("pydantic_messages", []))
    except (TypeError, ValueError):
        return estimate_content_tokens(state)
    serialized = ModelMessagesTypeAdapter.dump_python(messages, mode="json", exclude_none=True)
    projected = []
    opaque_tokens = 0
    instructions = None
    for message, raw in zip(messages, serialized, strict=True):
        if isinstance(message, ModelRequest) and message.instructions is not None:
            # Pydantic sends current instructions, not all previous instruction snapshots.
            instructions = message.instructions
        output_tokens = message.usage.output_tokens if isinstance(message, ModelResponse) else 0
        hidden = False
        parts = []
        for part in raw["parts"]:
            part = {key: value for key, value in part.items() if key != "timestamp"}
            if part.get("part_kind") == "thinking" and part.get("signature") and output_tokens > 0:
                part["signature"] = "[provider continuation]"
                hidden = True
            parts.append(part)
        if hidden:
            # All output tokens bound the hidden reasoning; ciphertext is not tokenized text.
            opaque_tokens += output_tokens
        projected.append({"kind": raw["kind"], "parts": parts})
    # Provenance is retained for inspection, never sent to the model. Unknown
    # extension state still takes the conservative estimation path.
    extra = {
        key: value
        for key, value in state.items()
        if key != "pydantic_messages"
        and not (key == LINEAGE_STATE_KEY and isinstance(value, HistoryLineage))
        and not (key == HISTORY_USAGE_KEY and isinstance(value, HistoryUsageCheckpoint))
    }
    measured = measured_history_tokens(state, "pydantic_messages")
    if measured is not None:
        return measured + estimate_content_tokens(extra)
    # Inline media is estimated by kind and size, not as base64 text.
    projected, media_tokens = project_media_payloads(projected)
    return estimate_content_tokens(projected, instructions, extra) + opaque_tokens + media_tokens


@contextmanager
def observe_model_admissions(
    observer: Callable[[Sequence[ModelMessage]], None],
) -> Generator[None, None, None]:
    token = _MODEL_ADMISSION.set(observer)
    try:
        yield
    finally:
        _MODEL_ADMISSION.reset(token)


def _admit_request(messages: Sequence[ModelMessage]) -> None:
    admit_run_operation("request")
    observer = _MODEL_ADMISSION.get()
    if observer is not None:
        observer(messages)


class BudgetedModel(WrapperModel):
    """Admit every model generation, including tool-loop and output-repair turns."""

    def __init__(self, wrapped: Model) -> None:
        assert_no_hidden_model_retries(wrapped)
        super().__init__(wrapped)

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        _admit_request(messages)
        return await super().request(messages, model_settings, model_request_parameters)

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncGenerator[Any]:
        _admit_request(messages)
        async with super().request_stream(
            messages, model_settings, model_request_parameters, run_context
        ) as response:
            yield response


async def _fallback_judgment(model: FallbackModel, value: Exception | ModelResponse) -> bool:
    """Ask the wrapped model's own fallback_on policy."""
    return await model._predicates.should_fallback(value)  # pyright: ignore[reportPrivateUsage]


def budget_model(model: Model) -> Model:
    if current_run_budget() is None or isinstance(model, BudgetedModel):
        return model
    if isinstance(model, FallbackModel):

        async def on_exception(value: Exception) -> bool:
            if isinstance(value, RunBudgetExceeded):
                raise value
            return await _fallback_judgment(model, value)

        async def on_response(value: ModelResponse) -> bool:
            return await _fallback_judgment(model, value)

        return FallbackModel(
            *(budget_model(child) for child in model.models),
            fallback_on=[on_exception, on_response],
        )
    if isinstance(model, WrapperModel):
        # copy.copy probes __setstate__ before `wrapped` exists, recursively
        # triggering WrapperModel.__getattr__. Preserve wrapper state directly.
        wrapped = object.__new__(type(model))
        vars(wrapped).update(vars(model))
        wrapped.wrapped = budget_model(model.wrapped)
        return wrapped
    return BudgetedModel(model)
