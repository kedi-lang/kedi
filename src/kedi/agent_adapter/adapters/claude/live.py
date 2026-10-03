"""Deliver queued input at Claude's tool and stop boundaries."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING

from kedi.agent_adapter.subagents.control import (
    LiveInput,
    RunInterrupted,
    UnconfirmedStop,
    controlled_attempt,
    current_live_revision,
)

if TYPE_CHECKING:
    from claude_agent_sdk import ClaudeAgentOptions, Message, ResultMessage
    from claude_agent_sdk.types import HookContext, HookEvent, HookInput, SyncHookJSONOutput


async def run_query(
    options: ClaudeAgentOptions,
    prompt: str,
    control: LiveInput,
    observe: Callable[[Message], None],
) -> ResultMessage:
    from claude_agent_sdk import ClaudeSDKClient, HookMatcher, ResultMessage

    revision = current_live_revision(control)

    async def deliver(
        input: HookInput, tool_id: str | None, context: HookContext
    ) -> SyncHookJSONOutput:
        del tool_id, context
        event = input["hook_event_name"]
        if event not in ("PostToolUse", "PostToolUseFailure", "Stop"):
            return {}
        items = control.take_pending(revision=revision)
        if not items:
            return {}
        content = "\n\n".join(item.render() for item in items)
        control.mark_delivered(items)
        if event == "Stop":
            return {"decision": "block", "reason": content}
        if event == "PostToolUse":
            return {
                "hookSpecificOutput": {
                    "hookEventName": "PostToolUse",
                    "additionalContext": content,
                }
            }
        return {
            "hookSpecificOutput": {
                "hookEventName": "PostToolUseFailure",
                "additionalContext": content,
            }
        }

    hooks = {name: list(matchers) for name, matchers in (options.hooks or {}).items()}
    input_events: tuple[HookEvent, ...] = ("PostToolUse", "PostToolUseFailure", "Stop")
    for name in input_events:
        hooks.setdefault(name, []).append(HookMatcher(hooks=[deliver]))
    options = replace(options, hooks=hooks)
    async with ClaudeSDKClient(options=options) as client:
        if control.revision != revision:
            raise RunInterrupted

        async def consume() -> ResultMessage:
            result: ResultMessage | None = None
            try:
                await client.query(prompt)
                control.confirm_contents((prompt,))
                async for message in client.receive_response():
                    observe(message)
                    if isinstance(message, ResultMessage):
                        result = message
            except Exception as error:
                raise UnconfirmedStop(
                    "Claude response ended before terminal confirmation"
                ) from error
            if result is None:
                raise UnconfirmedStop("Claude Agent SDK did not return a terminal result")
            return result

        return await controlled_attempt(control, consume(), stop=client.interrupt)
    raise UnconfirmedStop("Claude client closed before terminal confirmation")
