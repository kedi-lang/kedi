"""Native Pydantic AI pending input and cancellation bridge."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from typing import TypeVar

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.capabilities.abstract import ValidatedToolArgs, WrapModelRequestHandler
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelRequestPart,
    ModelResponse,
    RetryPromptPart,
    ToolAvailabilityDeltaPart,
    ToolCallPart,
    ToolReturn,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.run import AgentRun, AgentRunResult
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai.usage import RunUsage

from kedi.agent_adapter.subagents.control import AgentInput, LiveInput, controlled_attempt
from kedi.run_budget import current_run_budget

from .budget import observe_model_admissions

DepsT = TypeVar("DepsT")
OutputT = TypeVar("OutputT")
ToolResultT = TypeVar("ToolResultT")


@dataclass(kw_only=True)
class InputDeliveryCapability(AbstractCapability[DepsT]):
    control: LiveInput
    requests: int = field(default=0, init=False)
    completed_tools: dict[str, ToolReturn] = field(default_factory=dict, init=False)

    async def wrap_model_request(
        self,
        ctx: RunContext[DepsT],
        *,
        request_context: ModelRequestContext,
        handler: WrapModelRequestHandler,
    ) -> ModelResponse:
        # The adapter wraps budgeted models; their admission also fences fallback requests.
        if current_run_budget() is not None:
            with observe_model_admissions(self._record_request):
                return await handler(request_context)
        self._record_request(request_context.messages)
        return await handler(request_context)

    def _record_request(self, messages: Sequence[ModelMessage]) -> None:
        self.requests += 1
        self.control.confirm_contents(
            tuple(
                part.content
                for message in messages
                if isinstance(message, ModelRequest)
                for part in message.parts
                if isinstance(part, UserPromptPart) and isinstance(part.content, str)
            )
        )

    async def after_tool_execute(
        self,
        ctx: RunContext[DepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        result: ToolResultT,
    ) -> ToolResultT:
        self.completed_tools[call.tool_call_id] = (
            result if isinstance(result, ToolReturn) else ToolReturn(result)
        )
        return result

    @classmethod
    def get_serialization_name(cls) -> None:
        return None


def retain_tool_results(
    run: AgentRun[DepsT, OutputT], capability: InputDeliveryCapability[DepsT]
) -> None:
    messages = run.all_messages()
    last_response = next(
        (i for i in range(len(messages) - 1, -1, -1) if isinstance(messages[i], ModelResponse)),
        None,
    )
    if last_response is None:
        return
    answered = {
        part.tool_call_id
        for message in messages[last_response + 1 :]
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, (ToolReturnPart, RetryPromptPart))
    }
    parts: list[ModelRequestPart] = []
    for call in messages[last_response].parts:
        if not isinstance(call, ToolCallPart) or call.tool_call_id in answered:
            continue
        if call.tool_call_id not in capability.completed_tools:
            parts.append(
                ToolReturnPart(
                    call.tool_name,
                    "This tool call was interrupted. Its result is unavailable; do not assume it succeeded.",
                    call.tool_call_id,
                    outcome="failed",
                )
            )
            continue
        returned = capability.completed_tools[call.tool_call_id]
        parts.append(
            ToolReturnPart(
                call.tool_name, returned.return_value, call.tool_call_id, metadata=returned.metadata
            )
        )
        if returned.content:
            parts.append(UserPromptPart(returned.content))
        if returned.tools:
            parts.append(
                ToolAvailabilityDeltaPart(
                    tools_added=returned.tools, tool_call_id=call.tool_call_id
                )
            )
    if parts:
        messages.append(ModelRequest(parts=parts))


async def run_with_input(
    iterator: AbstractAsyncContextManager[AgentRun[DepsT, OutputT]],
    *,
    control: LiveInput,
    capability: InputDeliveryCapability[DepsT],
    capture: Callable[[AgentRun[DepsT, OutputT]], None],
    usage: RunUsage | None,
) -> AgentRunResult[OutputT]:
    ready = asyncio.Event()
    run: AgentRun[DepsT, OutputT] | None = None
    previous_requests = usage.requests if usage is not None else 0
    previous_admissions = capability.requests

    async def consume() -> AgentRunResult[OutputT]:
        nonlocal run
        async with iterator as active:
            run = active
            ready.set()
            try:
                async for _ in run:
                    pass
                result = run.result
                if result is None:
                    raise RuntimeError("Pydantic AI run ended without a terminal result")
                return result
            finally:
                if usage is not None:
                    usage.requests = max(
                        usage.requests,
                        previous_requests + capability.requests - previous_admissions,
                    )
                retain_tool_results(run, capability)
                capture(run)

    async def deliver(items: tuple[AgentInput, ...]) -> bool:
        await ready.wait()
        assert run is not None
        for item in items:
            run.enqueue(item.render())
        return False

    async def stop() -> None:
        if run is None:
            driver.cancel()
        else:
            run.cancel()

    driver = asyncio.create_task(consume())
    return await controlled_attempt(control, driver, deliver=deliver, stop=stop)
