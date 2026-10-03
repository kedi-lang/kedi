from __future__ import annotations

from collections.abc import AsyncGenerator, Callable, Iterator, Sequence
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from typing import Any

from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import RunContext

from kedi.run_budget import (
    RunBudgetExceeded,
    admit_run_operation,
    assert_no_hidden_model_retries,
    current_run_budget,
)


_MODEL_ADMISSION: ContextVar[Callable[[Sequence[ModelMessage]], None] | None] = ContextVar(
    "kedi_pydantic_model_admission", default=None
)


@contextmanager
def observe_model_admissions(observer: Callable[[Sequence[ModelMessage]], None]) -> Iterator[None]:
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


def budget_model(model: Model) -> Model:
    if current_run_budget() is None or isinstance(model, BudgetedModel):
        return model
    if isinstance(model, FallbackModel):

        async def on_exception(value: Exception) -> bool:
            if isinstance(value, RunBudgetExceeded):
                raise value
            return await model._should_fallback(value)  # pyright: ignore[reportPrivateUsage]

        async def on_response(value: ModelResponse) -> bool:
            return await model._should_fallback(value)  # pyright: ignore[reportPrivateUsage]

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
