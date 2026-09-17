from __future__ import annotations

from time import monotonic
from typing import Any, Mapping, Sequence

from kedi.agent_adapter.adapters.claude.tools import (
    _claude_native_tool_spec,
    _claude_tool_call_input,
    _claude_tool_specs_by_native_name,
)
from kedi.agent_adapter.approvals.policy import ApprovalPolicy
from kedi.agent_adapter.hooks import (
    HookDeniedError,
    ToolOrigin,
    current_hook_dispatcher,
    dispatch_post_tool_use,
    dispatch_post_tool_use_failure,
    dispatch_pre_tool_use,
    is_tool_execution_interrupted,
    new_tool_call_id,
)
from kedi.agent_adapter.tools.execution import resolve_tool_arguments_async
from kedi.agent_adapter.tools.specs import ToolSpec
from kedi.agent_profile import AgentProfile


def _with_kedi_approval_hook(
    sdk: Any,
    existing_hooks: object,
    *,
    profile: AgentProfile,
    tools: Sequence[ToolSpec],
    approval_ceiling: Sequence[ApprovalPolicy | None] = (),
    passthrough_tool_names: frozenset[str] = frozenset(),
) -> dict[str, list[Any]]:
    """Append Kedi lifecycle bridges without discarding caller-provided hooks."""

    if existing_hooks is None:
        hooks: dict[str, list[Any]] = {}
    elif isinstance(existing_hooks, Mapping):
        hooks = {}
        for event, matchers in existing_hooks.items():
            if not isinstance(matchers, (list, tuple)):
                raise TypeError("Claude hook matchers must be lists or tuples.")
            hooks[str(event)] = list(matchers)
    else:
        raise TypeError("Claude hooks must be a mapping of hook events to matcher lists.")

    matcher_cls = _claude_hook_matcher_class(sdk)
    dispatcher = current_hook_dispatcher()
    post_enabled = dispatcher is not None and bool(dispatcher.handlers_for("post_tool_use"))
    failure_enabled = dispatcher is not None and bool(
        dispatcher.handlers_for("post_tool_use_failure")
    )
    started_at: dict[str, float] | None = {} if failure_enabled else None
    hooks.setdefault("PreToolUse", []).append(
        matcher_cls(
            matcher=None,
            hooks=[
                _claude_pre_tool_use_hook(
                    profile=profile,
                    tools=tools,
                    approval_ceiling=approval_ceiling,
                    passthrough_tool_names=passthrough_tool_names,
                    started_at=started_at,
                )
            ],
        )
    )
    if post_enabled or failure_enabled:
        hooks.setdefault("PostToolUse", []).append(
            matcher_cls(
                matcher=None,
                hooks=[_claude_post_tool_use_hook(tools=tools, started_at=started_at)],
            )
        )
    if failure_enabled:
        hooks.setdefault("PostToolUseFailure", []).append(
            matcher_cls(
                matcher=None,
                hooks=[
                    _claude_post_tool_use_failure_hook(
                        tools=tools,
                        started_at=started_at,
                    )
                ],
            )
        )
    return hooks


def _claude_hook_matcher_class(sdk: Any) -> Any:
    """Return the public Claude SDK hook matcher type."""

    matcher_cls = getattr(sdk, "HookMatcher", None)
    if matcher_cls is None:
        raise RuntimeError("claude-agent-sdk does not expose its public HookMatcher type")
    return matcher_cls


def _claude_pre_tool_use_hook(
    *,
    profile: AgentProfile,
    tools: Sequence[ToolSpec],
    approval_ceiling: Sequence[ApprovalPolicy | None] = (),
    passthrough_tool_names: frozenset[str] = frozenset(),
    started_at: dict[str, float] | None = None,
) -> Any:
    """Build a Claude hook that applies Kedi policy to every native tool call."""

    known_tools = _claude_tool_specs_by_native_name(tools)

    async def decide(hook_input: Any, tool_use_id: str | None, _context: Any) -> dict[str, Any]:
        if not isinstance(hook_input, Mapping):
            return _claude_hook_deny("Claude supplied invalid PreToolUse hook input.")
        tool_name = hook_input.get("tool_name")
        arguments = hook_input.get("tool_input")
        if not isinstance(tool_name, str) or not isinstance(arguments, Mapping):
            return _claude_hook_deny("Claude supplied an incomplete tool permission request.")

        tool, origin = _claude_hook_tool(tool_name, known_tools)
        call_id = tool_use_id or new_tool_call_id()
        try:
            final_arguments = await dispatch_pre_tool_use(
                tool,
                arguments,
                adapter_shortname="claude",
                tool_call_id=call_id,
                origin=origin,
                native_tool_name=tool_name,
            )
            if tool_name not in passthrough_tool_names:
                policies = tuple(approval_ceiling)
                if profile.approval is not None or not policies:
                    policies = (*policies, profile.approval)
                for policy in policies:
                    final_arguments = await resolve_tool_arguments_async(
                        tool,
                        final_arguments,
                        policy=policy,
                        adapter_shortname="claude",
                    )
        except HookDeniedError as exc:
            return _claude_hook_deny(str(exc))
        except Exception as exc:
            return _claude_hook_deny(f"Kedi rejected tool {tool.name!r}: {exc}")

        if started_at is not None:
            started_at[call_id] = monotonic()
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "allow",
                "updatedInput": final_arguments,
            }
        }

    return decide


def _claude_post_tool_use_hook(
    *,
    tools: Sequence[ToolSpec],
    started_at: dict[str, float] | None = None,
) -> Any:
    known_tools = _claude_tool_specs_by_native_name(tools)

    async def observe(hook_input: Any, tool_use_id: str | None, _context: Any) -> dict[str, Any]:
        tool_name, arguments = _claude_tool_call_input(hook_input, "PostToolUse")
        tool, origin = _claude_hook_tool(tool_name, known_tools)
        call_id = tool_use_id or new_tool_call_id()
        if started_at is not None:
            started_at.pop(call_id, None)
        await dispatch_post_tool_use(
            tool,
            arguments,
            hook_input.get("tool_response"),
            adapter_shortname="claude",
            tool_call_id=call_id,
            origin=origin,
            native_tool_name=tool_name,
        )
        return {}

    return observe


def _claude_post_tool_use_failure_hook(
    *,
    tools: Sequence[ToolSpec],
    started_at: dict[str, float] | None = None,
) -> Any:
    known_tools = _claude_tool_specs_by_native_name(tools)

    async def observe(hook_input: Any, tool_use_id: str | None, _context: Any) -> dict[str, Any]:
        tool_name, arguments = _claude_tool_call_input(hook_input, "PostToolUseFailure")
        tool, origin = _claude_hook_tool(tool_name, known_tools)
        error = RuntimeError(str(hook_input.get("error", "Claude tool execution failed")))
        call_id = tool_use_id or new_tool_call_id()
        started = None if started_at is None else started_at.pop(call_id, None)
        await dispatch_post_tool_use_failure(
            tool,
            arguments,
            error,
            adapter_shortname="claude",
            tool_call_id=call_id,
            origin=origin,
            native_tool_name=tool_name,
            duration_seconds=(None if started is None else max(monotonic() - started, 0.0)),
            interrupted=(
                hook_input.get("is_interrupt") is True or is_tool_execution_interrupted(error)
            ),
        )
        return {}

    return observe


def _claude_hook_tool(
    native_name: str,
    known_tools: Mapping[str, ToolSpec],
) -> tuple[ToolSpec, ToolOrigin]:
    tool = known_tools.get(native_name)
    if tool is not None:
        return tool, "kedi"
    origin: ToolOrigin = "mcp" if native_name.startswith("mcp__") else "provider_builtin"
    return _claude_native_tool_spec(native_name), origin


def _claude_hook_deny(reason: str) -> dict[str, Any]:
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }
