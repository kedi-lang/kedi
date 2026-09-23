from __future__ import annotations

from contextvars import ContextVar
from time import monotonic
from typing import (
    TYPE_CHECKING,
    Any,
    Iterator,
    Mapping,
    Sequence,
)

from pydantic_ai import (
    ModelRetry,
)
from pydantic_ai.capabilities import (
    AbstractCapability,
    CapabilityOrdering,
    ValidatedToolArgs,
    WrapToolExecuteHandler,
)
from pydantic_ai.exceptions import (
    ApprovalRequired,
    CallDeferred,
    SkipToolExecution,
    ToolFailed,
    ToolFailedError,
    ToolRetryError,
)
from pydantic_ai.messages import (
    ToolCallPart,
)
from pydantic_ai.tools import (
    RunContext,
    ToolDefinition,
)

from kedi.agent_adapter.adapters.pydantic.capabilities import (
    _KediArtifactCapability,
    _KediDeferredApprovalCapability,
    _KediExternalApprovalCapability,
    _KediRequiredToolCapability,
    _KediTelemetryCapability,
)
from kedi.agent_adapter.adapters.pydantic.tools import _external_tool_spec
from kedi.agent_adapter.approvals.policy import ToolApprovalDenied
from kedi.agent_adapter.codemode.pydantic import PydanticCodeModeCapability
from kedi.agent_adapter.hooks import (
    HookDeniedError,
    ToolOrigin,
    current_hook_dispatcher,
    dispatch_post_tool_use,
    dispatch_post_tool_use_failure,
    dispatch_post_tool_result_failure,
    dispatch_pre_tool_use,
    is_tool_execution_interrupted,
)
from kedi.agent_adapter.hooks.tool_wrapper import (
    note_result_observer_failure,
    defer_result_observation,
)
from kedi.agent_adapter.tools.specs import ToolSpec
from kedi.agent_adapter.tools.records import ToolCallTracker, attach_tool_call, track_tool_call

if TYPE_CHECKING:
    from kedi.agent_adapter.adapters.pydantic.adapter import PydanticAdapter


class _KediHookCapability(AbstractCapability[Any]):
    """Project Kedi's tool lifecycle onto Pydantic AI's public capability hooks."""

    def __init__(self, adapter: PydanticAdapter[Any, Any]) -> None:
        self.adapter = adapter
        self._records: dict[str, ToolCallTracker] = {}

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(
            wraps=(
                PydanticCodeModeCapability,
                _KediDeferredApprovalCapability,
                _KediExternalApprovalCapability,
                _KediRequiredToolCapability,
                _KediArtifactCapability,
                _KediTelemetryCapability,
            )
        )

    async def before_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
    ) -> ValidatedToolArgs:
        if ctx.tool_call_approved:
            tool, origin = self.adapter._hook_tool_for_call(call.tool_name, tool_def)
            self.adapter._remember_hook_tool(call.tool_call_id, tool, origin)
            self.adapter._remember_hook_arguments(call.tool_call_id, args)
            return args
        tool, origin = self.adapter._hook_tool_for_call(call.tool_name, tool_def)
        self.adapter._remember_hook_tool(call.tool_call_id, tool, origin)
        try:
            final_arguments = await dispatch_pre_tool_use(
                tool,
                args,
                adapter_shortname=self.adapter.shortname,
                tool_call_id=call.tool_call_id,
                origin=origin,
                native_tool_name=call.tool_name,
            )
        except HookDeniedError as exc:
            raise ModelRetry(str(exc)) from exc
        self.adapter._remember_hook_arguments(call.tool_call_id, final_arguments)
        return final_arguments

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
        self.adapter._remember_hook_started(call.tool_call_id)
        try:
            with track_tool_call(call.tool_call_id) as tracker:
                self._records[call.tool_call_id] = tracker
                result = await handler(args)
                tracker.ready()
                return result
        except (
            ApprovalRequired,
            CallDeferred,
            SkipToolExecution,
        ):
            self._records.pop(call.tool_call_id, None)
            self.adapter._consume_hook_started(call.tool_call_id)
            raise
        except (
            ToolRetryError,
            ToolFailedError,
            ToolFailed,
            ModelRetry,
        ) as exc:
            await self._dispatch_failure(
                call=call,
                tool_def=tool_def,
                args=args,
                error=exc,
            )
            raise
        except BaseException as exc:
            if isinstance(exc, Exception):
                raise
            await self._dispatch_failure(
                call=call,
                tool_def=tool_def,
                args=args,
                error=exc,
            )
            raise

    async def _dispatch_failure(
        self,
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        error: BaseException,
    ) -> None:
        tool, origin = self.adapter._hook_tool_for_call_id(
            call.tool_call_id,
            self.adapter._hook_tool_for_call(call.tool_name, tool_def),
            consume=True,
        )
        final_arguments = self.adapter._hook_arguments_for_call(
            call.tool_call_id,
            args,
            consume=True,
        )
        duration = self.adapter._consume_hook_started(call.tool_call_id)
        tracker = self._records.pop(call.tool_call_id, None)
        record = tracker.latest if tracker is not None else None
        if record is not None:
            assert tracker is not None
            if record.execution_state == "returned":
                stage = getattr(error, "kedi_result_failure_stage", "result_validation")
                record = tracker.result_failed(stage) or record
                attach_tool_call(error, record)
                try:
                    await dispatch_post_tool_result_failure(
                        tool,
                        final_arguments,
                        error,
                        record,
                        adapter_shortname=self.adapter.shortname,
                        tool_call_id=call.tool_call_id,
                        origin=origin,
                        native_tool_name=call.tool_name,
                    )
                except BaseException as observer_error:
                    note_result_observer_failure(error, observer_error)
                return
            attach_tool_call(error, record)
        await dispatch_post_tool_use_failure(
            tool,
            final_arguments,
            error,
            adapter_shortname=self.adapter.shortname,
            tool_call_id=call.tool_call_id,
            origin=origin,
            native_tool_name=call.tool_name,
            duration_seconds=duration,
            interrupted=is_tool_execution_interrupted(error),
        )

    async def after_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        result: Any,
    ) -> Any:
        del ctx
        tool, origin = self.adapter._hook_tool_for_call_id(
            call.tool_call_id,
            self.adapter._hook_tool_for_call(call.tool_name, tool_def),
            consume=True,
        )
        final_arguments = self.adapter._hook_arguments_for_call(
            call.tool_call_id,
            args,
            consume=True,
        )
        self.adapter._consume_hook_started(call.tool_call_id)
        tracker = self._records.pop(call.tool_call_id, None)
        try:
            await dispatch_post_tool_use(
                tool,
                final_arguments,
                result,
                adapter_shortname=self.adapter.shortname,
                tool_call_id=call.tool_call_id,
                origin=origin,
                native_tool_name=call.tool_name,
            )
        except BaseException as exc:
            record = tracker.result_failed("post_hook") if tracker is not None else None
            if record is not None:
                attach_tool_call(exc, record)
                try:
                    await dispatch_post_tool_result_failure(
                        tool,
                        final_arguments,
                        exc,
                        record,
                        adapter_shortname=self.adapter.shortname,
                        tool_call_id=call.tool_call_id,
                        origin=origin,
                        native_tool_name=call.tool_name,
                    )
                except BaseException as observer_error:
                    note_result_observer_failure(exc, observer_error)
            raise
        if tracker is not None and not defer_result_observation(
            tracker, tool, final_arguments, adapter_shortname=self.adapter.shortname, origin=origin
        ):
            if tracker.latest is not None:
                tracker.update(destination="framework")
            tracker.handed_off()
        return result

    async def on_tool_execute_error(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        error: Exception,
    ) -> Any:
        del ctx
        if isinstance(error, (ApprovalRequired, HookDeniedError, ToolApprovalDenied)):
            raise error
        await self._dispatch_failure(
            call=call,
            tool_def=tool_def,
            args=args,
            error=error,
        )
        raise error


def _hook_tool_for_call(
    self: PydanticAdapter[Any, Any],
    tool_name: str,
    tool_def: ToolDefinition,
) -> tuple[ToolSpec, ToolOrigin]:
    tool = self._tool_spec_for_call(tool_name)
    if (tool.metadata or {}).get("source") != "external":
        return tool, "kedi"
    toolset_id = tool_def.toolset_id or ""
    origin: ToolOrigin = "mcp" if "mcp" in toolset_id.lower() else "adapter_native"
    return _external_tool_spec(tool_name, tool_def), origin


def _remember_hook_tool(
    self: PydanticAdapter[Any, Any],
    tool_call_id: str,
    tool: ToolSpec,
    origin: ToolOrigin,
) -> None:
    self._hook_tools_for_current_adapter().get()[tool_call_id] = (tool, origin)


def _hook_tool_for_call_id(
    self: PydanticAdapter[Any, Any],
    tool_call_id: str,
    fallback: tuple[ToolSpec, ToolOrigin],
    *,
    consume: bool = False,
) -> tuple[ToolSpec, ToolOrigin]:
    tools = self._hook_tools_for_current_adapter().get()
    value = tools.pop(tool_call_id, None) if consume else tools.get(tool_call_id)
    return fallback if value is None else value


def _hook_tools_for_current_adapter(
    self: PydanticAdapter[Any, Any],
) -> ContextVar[dict[str, tuple[ToolSpec, ToolOrigin]]]:
    try:
        return self._hook_tools_var
    except AttributeError:
        tools: ContextVar[dict[str, tuple[ToolSpec, ToolOrigin]]] = ContextVar(
            f"kedi_pydantic_hook_tools_{id(self)}",
            default={},
        )
        self._hook_tools_var = tools
        return tools


def _remember_hook_arguments(
    self: PydanticAdapter[Any, Any],
    tool_call_id: str,
    arguments: Mapping[str, Any],
) -> None:
    self._hook_arguments_for_current_adapter().get()[tool_call_id] = dict(arguments)


def _hook_arguments_for_call(
    self: PydanticAdapter[Any, Any],
    tool_call_id: str,
    fallback: Mapping[str, Any],
    *,
    consume: bool = False,
) -> dict[str, Any]:
    arguments = self._hook_arguments_for_current_adapter().get()
    value = arguments.pop(tool_call_id, None) if consume else arguments.get(tool_call_id)
    return dict(fallback) if value is None else dict(value)


def _hook_arguments_for_current_adapter(
    self: PydanticAdapter[Any, Any],
) -> ContextVar[dict[str, dict[str, Any]]]:
    try:
        return self._hook_arguments_var
    except AttributeError:
        arguments: ContextVar[dict[str, dict[str, Any]]] = ContextVar(
            f"kedi_pydantic_hook_arguments_{id(self)}",
            default={},
        )
        self._hook_arguments_var = arguments
        return arguments


def _remember_hook_started(self: PydanticAdapter[Any, Any], tool_call_id: str) -> None:
    dispatcher = current_hook_dispatcher()
    if dispatcher is None or not dispatcher.handlers_for("post_tool_use_failure"):
        return
    self._hook_started_for_current_adapter().get()[tool_call_id] = monotonic()


def _consume_hook_started(self: PydanticAdapter[Any, Any], tool_call_id: str) -> float | None:
    started = self._hook_started_for_current_adapter().get().pop(tool_call_id, None)
    return None if started is None else max(monotonic() - started, 0.0)


def _hook_started_for_current_adapter(
    self: PydanticAdapter[Any, Any],
) -> ContextVar[dict[str, float]]:
    try:
        return self._hook_started_var
    except AttributeError:
        started: ContextVar[dict[str, float]] = ContextVar(
            f"kedi_pydantic_hook_started_{id(self)}",
            default={},
        )
        self._hook_started_var = started
        return started


def _run_tool_spec_scope(
    self: PydanticAdapter[Any, Any], tool_specs: Sequence[ToolSpec]
) -> Iterator[None]:
    token = self._run_tool_specs_var.set(tuple(tool_specs))
    hook_arguments_var = self._hook_arguments_for_current_adapter()
    hook_arguments_token = hook_arguments_var.set({})
    hook_tools_var = self._hook_tools_for_current_adapter()
    hook_tools_token = hook_tools_var.set({})
    hook_started_var = self._hook_started_for_current_adapter()
    hook_started_token = hook_started_var.set({})
    reasons_token = self._tool_reasons_var.set({})
    try:
        with self._required_tool_call_scope():
            yield
    finally:
        self._tool_reasons_var.reset(reasons_token)
        hook_started_var.reset(hook_started_token)
        hook_tools_var.reset(hook_tools_token)
        hook_arguments_var.reset(hook_arguments_token)
        self._run_tool_specs_var.reset(token)
