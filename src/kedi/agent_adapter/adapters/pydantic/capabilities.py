from __future__ import annotations

from dataclasses import replace
from typing import (
    TYPE_CHECKING,
    Any,
)

from pydantic_ai.capabilities import (
    AbstractCapability,
    HandleDeferredToolCalls,
    ValidatedToolArgs,
    WrapRunHandler,
    WrapToolExecuteHandler,
)
from pydantic_ai.exceptions import (
    ApprovalRequired,
)
from pydantic_ai.messages import (
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturn,
)
from pydantic_ai.tools import (
    RunContext,
    ToolDefinition,
)

from kedi.agent_adapter.adapters.pydantic.tools import (
    KEDI_APPROVAL_METADATA_KEY,
    _approval_request_payload,
)
from kedi.agent_adapter.events import AgentMessageResolver
from kedi.agent_adapter.telemetry import (
    native_agent_span_scope,
    native_tool_span_scope,
    reclassify_native_agent_span,
)
from kedi.agent_adapter.tools.execution import (
    admit_current_artifact_result,
    approval_request_for_tool,
)
from kedi.agent_adapter.tools.specs import tool_has_artifact_boundary

from .tools import _validate_required_tool_calls

if TYPE_CHECKING:
    from pydantic_ai.agent import AgentRunResult

    from kedi.agent_adapter.adapters.pydantic.adapter import PydanticAdapter


class _KediTelemetryCapability(AbstractCapability[Any]):
    """Reclassify Pydantic AI's native run span with Kedi semantics."""

    async def wrap_run(
        self,
        ctx: RunContext[Any],
        *,
        handler: WrapRunHandler,
    ) -> AgentRunResult[Any]:
        del ctx
        with native_agent_span_scope(True):
            reclassify_native_agent_span()
            return await handler()

    async def wrap_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        handler: WrapToolExecuteHandler,
    ) -> Any:
        del ctx
        with native_tool_span_scope(call.tool_name):
            return await handler(args)


class _KediArtifactCapability(AbstractCapability[Any]):
    """Admit native and MCP results before Pydantic commits tool history."""

    def __init__(self, adapter: PydanticAdapter[Any, Any]) -> None:
        self.adapter = adapter

    async def after_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        result: Any,
    ) -> Any:
        del ctx, args
        if (tool_def.metadata or {}).get("kedi_codemode_control") is True:
            return result
        tool = self.adapter._tool_spec_for_call(call.tool_name)
        if tool.artifact_result == "never" or tool_has_artifact_boundary(tool):
            return result
        payload = result.return_value if isinstance(result, ToolReturn) else result
        admitted = await admit_current_artifact_result(
            payload,
            tool_name=call.tool_name,
            call_id=call.tool_call_id,
        )
        if isinstance(result, ToolReturn):
            return replace(result, return_value=admitted, content=None)
        return admitted


class _KediRequiredToolCapability(AbstractCapability[Any]):
    """Reject output until every required Kedi tool has completed."""

    async def after_output_process(
        self,
        ctx: RunContext[Any],
        *,
        output_context: Any,
        output: Any,
    ) -> Any:
        del ctx, output_context
        return _validate_required_tool_calls(output)


class _KediAgentEventCapability(AbstractCapability[Any]):
    """Observe completed Pydantic messages without consuming its event stream."""

    def __init__(self, resolver: AgentMessageResolver) -> None:
        self.resolver = resolver
        self._message_index = 0

    async def after_model_request(
        self,
        ctx: RunContext[Any],
        *,
        request_context: Any,
        response: ModelResponse,
    ) -> ModelResponse:
        del ctx, request_context
        text = "".join(part.content for part in response.parts if isinstance(part, TextPart))
        if text:
            self._message_index += 1
            self.resolver.complete_message(
                f"{self.resolver.run_id}:pydantic:{self._message_index}",
                text,
            )
        if any(isinstance(part, ToolCallPart) for part in response.parts):
            self.resolver.mark_tool_boundary()
        return response


class _KediDeferredApprovalCapability(HandleDeferredToolCalls[Any]):
    """Identify Kedi's approval capability so nested preparation can deduplicate it."""


class _KediExternalApprovalCapability(AbstractCapability[Any]):
    """Normalize deferred approvals without taking ownership of the decision."""

    def __init__(self, adapter: PydanticAdapter[Any, Any]) -> None:
        self.adapter = adapter

    async def wrap_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        handler: WrapToolExecuteHandler,
    ) -> Any:
        del ctx
        try:
            return await handler(args)
        except ApprovalRequired as exc:
            tool, _origin = self.adapter._hook_tool_for_call_id(
                call.tool_call_id,
                self.adapter._hook_tool_for_call(call.tool_name, tool_def),
            )
            request = approval_request_for_tool(
                tool,
                self.adapter._hook_arguments_for_call(call.tool_call_id, args),
                self.adapter.shortname,
            )
            call.args = dict(request.arguments)
            metadata = dict(exc.metadata or {})
            metadata[KEDI_APPROVAL_METADATA_KEY] = _approval_request_payload(request)
            raise ApprovalRequired(metadata) from exc
