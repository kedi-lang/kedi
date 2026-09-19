from __future__ import annotations

import json
from dataclasses import replace
from time import monotonic
from typing import (
    Any,
    Mapping,
    Sequence,
)

from kedi.agent_adapter.adapters.langchain.tools import (
    _langchain_external_tool_spec,
    _langchain_tool_error,
)
from kedi.agent_adapter.approvals.policy import ApprovalPolicy, ToolApprovalDenied
from kedi.agent_adapter.approvals.reason import accepts_tool_reason, split_tool_reason
from kedi.agent_adapter.hooks import (
    HookDeniedError,
    ToolOrigin,
    dispatch_post_tool_use,
    dispatch_post_tool_use_failure,
    dispatch_pre_tool_use,
    is_tool_execution_interrupted,
    new_tool_call_id,
)
from kedi.agent_adapter.prefix_cache import langchain_provider_cache_policy
from kedi.agent_adapter.tools.execution import (
    admit_current_artifact_result,
    resolve_tool_arguments_async,
)
from kedi.agent_adapter.tools.specs import ToolSpec, tool_has_artifact_boundary
from kedi.agent_profile import AgentProfile


def _langchain_cache_middleware(model: Any, *, enabled: bool) -> list[Any]:
    if not enabled or langchain_provider_cache_policy(model).backend != "bedrock":
        return []
    try:
        from langchain_aws.middleware.prompt_caching import BedrockPromptCachingMiddleware
    except ImportError as exc:
        raise ModuleNotFoundError(
            "LangChain Bedrock support requires its optional AWS dependencies. "
            "Install with: uv add 'kedi[langchain-aws]'"
        ) from exc
    return [BedrockPromptCachingMiddleware()]


def _langchain_approval_middleware(
    *,
    profile: AgentProfile,
    tools: Sequence[ToolSpec],
    approval_ceiling: Sequence[ApprovalPolicy | None] = (),
    approval_enabled: bool = True,
    tool_reason: bool = False,
) -> Any:
    """Bridge Kedi hook and approval semantics into LangChain's tool lifecycle."""

    try:
        from langchain.agents.middleware import AgentMiddleware
    except ImportError as exc:
        raise ModuleNotFoundError(
            "LangChainAdapter approval support requires langchain. "
            "Install with: uv add 'kedi[langchain]'"
        ) from exc

    known_tools = {tool.name: tool for tool in tools}

    class KediToolLifecycleMiddleware(AgentMiddleware):
        async def awrap_tool_call(self, request: Any, handler: Any) -> Any:
            tool_call = request.tool_call
            tool_name = tool_call.get("name") if isinstance(tool_call, Mapping) else None
            arguments = tool_call.get("args") if isinstance(tool_call, Mapping) else None
            tool_call_id = tool_call.get("id") if isinstance(tool_call, Mapping) else None
            if not isinstance(tool_name, str) or not isinstance(arguments, Mapping):
                return _langchain_tool_error(
                    tool_name if isinstance(tool_name, str) else "unknown",
                    tool_call_id,
                    "LangChain emitted an incomplete tool call.",
                )

            tool = known_tools.get(tool_name, _langchain_external_tool_spec(tool_name))
            call_id = str(tool_call_id or new_tool_call_id())
            origin = _langchain_hook_origin(tool, getattr(request, "tool", None))
            try:
                reason = None
                if tool_reason and accepts_tool_reason(tool):
                    arguments, reason = split_tool_reason(arguments)
                final_arguments = await dispatch_pre_tool_use(
                    tool,
                    arguments,
                    adapter_shortname="langchain",
                    tool_call_id=call_id,
                    origin=origin,
                    native_tool_name=tool_name,
                )
                if approval_enabled:
                    policies = tuple(approval_ceiling)
                    if profile.approval is not None or not policies:
                        policies = (*policies, profile.approval)
                    for policy in policies:
                        final_arguments = await resolve_tool_arguments_async(
                            tool,
                            final_arguments,
                            policy=policy,
                            adapter_shortname="langchain",
                            reason=reason,
                            tool_reason_enabled=tool_reason,
                        )
            except (HookDeniedError, ToolApprovalDenied) as exc:
                return _langchain_tool_error(tool_name, tool_call_id, str(exc))
            except Exception as exc:
                return _langchain_tool_error(tool_name, tool_call_id, str(exc))

            final_request = request.override(
                tool_call={
                    **dict(tool_call),
                    "args": final_arguments,
                }
            )
            started = monotonic()
            try:
                result = await handler(final_request)
            except BaseException as exc:
                await dispatch_post_tool_use_failure(
                    tool,
                    final_arguments,
                    exc,
                    adapter_shortname="langchain",
                    tool_call_id=call_id,
                    origin=origin,
                    native_tool_name=tool_name,
                    duration_seconds=monotonic() - started,
                    interrupted=is_tool_execution_interrupted(exc),
                )
                raise
            result_message = _langchain_tool_message_for_call(result, call_id)
            if result_message is not None and getattr(result_message, "status", None) == "error":
                error = RuntimeError(
                    str(getattr(result_message, "content", "Tool execution failed"))
                )
                await dispatch_post_tool_use_failure(
                    tool,
                    final_arguments,
                    error,
                    adapter_shortname="langchain",
                    tool_call_id=call_id,
                    origin=origin,
                    native_tool_name=tool_name,
                    duration_seconds=monotonic() - started,
                )
                return result
            if (
                tool.artifact_result != "never"
                and not tool_has_artifact_boundary(tool)
                and (tool.metadata or {}).get("kedi_codemode_control") is not True
            ):
                result = await _langchain_admit_native_tool_result(
                    result,
                    tool_name=tool_name,
                    call_id=call_id,
                )
            await dispatch_post_tool_use(
                tool,
                final_arguments,
                result,
                adapter_shortname="langchain",
                tool_call_id=call_id,
                origin=origin,
                native_tool_name=tool_name,
            )
            return result

    return KediToolLifecycleMiddleware()


async def _langchain_admit_native_tool_result(
    result: Any,
    *,
    tool_name: str,
    call_id: str,
) -> Any:
    """Replace causal ToolMessage content while preserving LangGraph commands."""

    try:
        from langchain_core.messages import ToolMessage
        from langgraph.types import Command
    except ImportError:  # pragma: no cover - adapter dependency guard
        return result

    async def admit_message(message: Any) -> tuple[Any, bool]:
        if not isinstance(message, ToolMessage) or message.tool_call_id != call_id:
            return message, False
        content = message.content
        admitted = await admit_current_artifact_result(
            content,
            tool_name=tool_name,
            call_id=call_id,
        )
        if admitted is content:
            return message, False
        model_content = (
            json.dumps(admitted, ensure_ascii=True, separators=(",", ":"))
            if isinstance(admitted, Mapping)
            else admitted
        )
        return message.model_copy(update={"content": model_content}), True

    async def admit_messages(value: Any) -> tuple[Any, bool]:
        if isinstance(value, ToolMessage):
            return await admit_message(value)
        if isinstance(value, list):
            changed = False
            items: list[Any] = []
            for item in value:
                admitted, item_changed = await admit_messages(item)
                items.append(admitted)
                changed = changed or item_changed
            return (items if changed else value), changed
        if isinstance(value, tuple):
            changed = False
            items = []
            for item in value:
                admitted, item_changed = await admit_messages(item)
                items.append(admitted)
                changed = changed or item_changed
            return (tuple(items) if changed else value), changed
        return value, False

    if isinstance(result, ToolMessage):
        admitted, _ = await admit_message(result)
        return admitted
    if isinstance(result, Command):
        update = result.update
        if isinstance(update, Mapping):
            if "messages" not in update:
                return result
            messages, changed = await admit_messages(update["messages"])
            if not changed:
                return result
            return replace(result, update={**update, "messages": messages})

        admitted_update, changed = await admit_messages(update)
        return replace(result, update=admitted_update) if changed else result

    # Keep the existing helper contract for direct/raw results used by adapter-native bridges.
    content = getattr(result, "content", result)
    admitted = await admit_current_artifact_result(
        content,
        tool_name=tool_name,
        call_id=call_id,
    )
    if admitted is content:
        return result
    model_copy = getattr(result, "model_copy", None)
    if not callable(model_copy):
        return admitted
    model_content = (
        json.dumps(admitted, ensure_ascii=True, separators=(",", ":"))
        if isinstance(admitted, Mapping)
        else admitted
    )
    return model_copy(update={"content": model_content})


def _langchain_tool_message_for_call(result: Any, call_id: str) -> Any | None:
    """Find the causal ToolMessage in a direct result or LangGraph Command update."""

    try:
        from langchain_core.messages import ToolMessage
        from langgraph.types import Command
    except ImportError:  # pragma: no cover - adapter dependency guard
        return None

    def find(value: Any) -> Any | None:
        if isinstance(value, ToolMessage):
            return value if value.tool_call_id == call_id else None
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for item in value:
                found = find(item)
                if found is not None:
                    return found
        return None

    if isinstance(result, ToolMessage):
        return find(result)
    if not isinstance(result, Command):
        return None
    update = result.update
    return find(update.get("messages")) if isinstance(update, Mapping) else find(update)


def _langchain_hook_origin(tool: ToolSpec, native_tool: Any) -> ToolOrigin:
    if (tool.metadata or {}).get("source") != "langchain-external":
        return "kedi"
    metadata = getattr(native_tool, "metadata", None)
    if isinstance(metadata, Mapping) and metadata.get("kedi_origin") == "mcp":
        return "mcp"
    return "adapter_native"
