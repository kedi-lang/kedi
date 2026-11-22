"""Use Pydantic AI's per-attempt recovery, including stream-opening failures."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Awaitable, Callable, Iterable
from contextlib import asynccontextmanager
from typing import Any

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import FallbackExceptionGroup, RetryModelRequest
from pydantic_ai.messages import ModelMessage, ModelRequestAttempt, ModelResponse
from pydantic_ai.models import Model, ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import RunContext

from kedi.agent_adapter.request_retry import (
    provider_status,
    transient_provider_error,
    wait_for_request_retry,
)


class _RequestFailureModel(WrapperModel):
    """Observe HTTP failures before a native fallback capability handles them."""

    def __init__(self, wrapped: Model, on_failure: Callable[[Exception], None]) -> None:
        super().__init__(wrapped)
        self._on_failure = on_failure

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        try:
            return await super().request(messages, model_settings, model_request_parameters)
        except Exception as error:
            self._on_failure(error)
            raise

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncGenerator[Any]:
        opened = False
        try:
            async with super().request_stream(
                messages, model_settings, model_request_parameters, run_context
            ) as stream:
                opened = True
                yield stream
        except Exception as error:
            if not opened:
                self._on_failure(error)
            raise


class RequestRetryCapability(AbstractCapability[Any]):
    def __init__(self, retries: int) -> None:
        self.retries = retries
        self._attempts: dict[int, ModelRequestAttempt] = {}
        self._pending_http: set[int] = set()
        self._accounted_requests = 0
        self._attempt = 1
        self._fallback_models: set[tuple[str, str]] = set()
        self._candidate: Model | None = None
        self._candidate_attempt = 0

    async def for_run(self, ctx: RunContext[Any]) -> RequestRetryCapability:
        return RequestRetryCapability(self.retries)

    async def before_model_request(
        self, ctx: RunContext[Any], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        self._ctx = ctx
        self._attempts.clear()
        self._pending_http.clear()
        self._accounted_requests = 0
        self._attempt = request_context.attempt
        self._fallback_models.clear()
        self._candidate = None
        self._candidate_attempt = 0
        self._remember_fallbacks(request_context.model)
        return request_context

    def get_wrapper_model(self, model: Model) -> Model:
        return _RequestFailureModel(
            model, lambda error: self._record_http_failure(self._ctx, error)
        )

    def _record_http_failure(self, ctx: RunContext[Any], error: Exception) -> None:
        if provider_status(error) is not None:
            self._pending_http.add(self._attempt)
            self._account_attempts(ctx, ())

    async def prepare_model_request(
        self, ctx: RunContext[Any], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        self._attempt = request_context.attempt
        self._remember_fallbacks(request_context.model)
        candidate = request_context.model
        while isinstance(candidate, WrapperModel):
            candidate = candidate.wrapped
        if candidate is not self._candidate:
            self._candidate = candidate
            self._candidate_attempt = 0
        self._candidate_attempt += 1
        return request_context

    def _remember_fallbacks(self, model: Model) -> None:
        if isinstance(model, FallbackModel):
            self._fallback_models.add((model.model_name, model.system))
            for child in model.models:
                self._remember_fallbacks(child)
        elif isinstance(model, WrapperModel):
            self._remember_fallbacks(model.wrapped)

    def _account_attempts(
        self, ctx: RunContext[Any], attempts: Iterable[ModelRequestAttempt]
    ) -> None:
        for attempt in attempts:
            if id(attempt) in self._attempts:
                continue
            self._attempts[id(attempt)] = attempt
        # Nested groups also describe their wrapper call, not another provider
        # request. A direct HTTP error through that wrapper is still a real call.
        # Records are deduplicated by identity; the SDK's error-kind prefix only
        # distinguishes synthetic group records from direct wrapper failures.
        requests = sum(
            not (
                (attempt.model_name, attempt.provider_name) in self._fallback_models
                and (attempt.error or "").startswith("FallbackExceptionGroup:")
            )
            for attempt in self._attempts.values()
        ) + len(self._pending_http)
        ctx.usage.requests += requests - self._accounted_requests
        self._accounted_requests = requests

    async def wrap_model_request(
        self,
        ctx: RunContext[Any],
        *,
        request_context: ModelRequestContext,
        handler: Callable[[ModelRequestContext], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        self._remember_fallbacks(request_context.model)
        try:
            return await handler(request_context)
        except FallbackExceptionGroup as error:
            self._pending_http.clear()
            self._account_attempts(ctx, error.attempts)
            raise
        finally:
            # The SDK carries attempts here even when cancellation or strict
            # admission stops a fallback before it can return a response/group.
            self._account_attempts(ctx, request_context._usage_attempts)
            if request_context.streaming and request_context._usage_responses:
                # An interrupted stream carries earlier opening failures too.
                # A failure of this attempt itself still needs its direct count.
                self._pending_http.difference_update(
                    attempt for attempt in tuple(self._pending_http) if attempt < self._attempt
                )
            for response in request_context._usage_responses:
                self._account_attempts(ctx, response.failed_attempts or ())

    async def after_model_request(
        self,
        ctx: RunContext[Any],
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        self._remember_fallbacks(request_context.model)
        # A completed response lists all earlier attempts, including direct HTTP
        # failures already provisionally counted before a retry was admitted.
        self._pending_http.clear()
        self._account_attempts(ctx, response.failed_attempts or ())
        return response

    async def on_model_request_error(
        self,
        ctx: RunContext[Any],
        *,
        request_context: ModelRequestContext,
        error: Exception,
    ) -> ModelResponse:
        if provider_status(error) is not None:
            # Native usage counts a step only after a response. HTTP failures are
            # actual requests too; account for them before admitting another attempt.
            self._record_http_failure(ctx, error)
        attempt = self._candidate_attempt or request_context.attempt
        if attempt > self.retries or not transient_provider_error(error):
            raise error
        if ctx.usage_limits is not None:
            ctx.usage_limits.check_before_request(ctx.usage)
        if await wait_for_request_retry(error, attempt=attempt, retries=self.retries):
            raise RetryModelRequest() from error
        raise error
