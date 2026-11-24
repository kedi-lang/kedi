from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from kedi.agent_adapter.adapters.claude import ClaudeAdapter
from kedi.agent_adapter.adapters.claude import hooks as claude_hooks
from kedi.agent_adapter.adapters.langchain import middleware as langchain_middleware
from kedi.agent_adapter.approvals.policy import ApprovalCancelled, ToolApprovalDenied
from kedi.agent_adapter.hooks import HookDeniedError, HookExecutionError
from kedi.agent_profile import AgentProfile
from kedi.errors import KediExecutionError
from kedi.run_budget import RunBudgetExceeded
from tests.test_agent_adapters import _install_fake_claude_sdk


@pytest.fixture(params=["hook", "approval", "budget"])
def fatal_error(request: pytest.FixtureRequest) -> Exception:
    if request.param == "hook":
        return HookExecutionError(
            event="pre_tool_use", handler_name="check", cause=RuntimeError("hook failed")
        )
    if request.param == "approval":
        return ApprovalCancelled("approval cancelled")
    return RunBudgetExceeded("request", 0, 0, 0)


def test_claude_fatal_hook_without_a_run_scope_raises_immediately(fatal_error: Exception) -> None:
    with pytest.raises(type(fatal_error)) as caught:
        claude_hooks._claude_stop_hook(fatal_error)
    assert caught.value is fatal_error


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        HookDeniedError(event="pre_tool_use", reason="blocked path"),
        ToolApprovalDenied("Tool 'write' was denied: static deny policy"),
        KediExecutionError(message="tool input could not be prepared"),
    ],
)
async def test_langchain_tool_refusals_reach_the_model_without_a_stack_trace(
    monkeypatch: pytest.MonkeyPatch, error: KediExecutionError
) -> None:
    async def refuse(*_args: Any, **_kwargs: Any) -> Any:
        raise error

    async def tool(_request: Any) -> Any:
        pytest.fail("A refused call must not run the tool")

    monkeypatch.setattr(langchain_middleware, "dispatch_pre_tool_use", refuse)
    middleware = langchain_middleware._langchain_approval_middleware(
        profile=AgentProfile.empty(), tools=()
    )
    message = await middleware.awrap_tool_call(
        SimpleNamespace(tool_call={"name": "write", "args": {}, "id": "call"}), tool
    )

    assert message.status == "error"
    assert message.content == error.message
    assert "Kedi stack trace" not in message.content


@pytest.mark.asyncio
async def test_langchain_pre_tool_fatal_error_escapes_without_running_tool(
    monkeypatch: pytest.MonkeyPatch, fatal_error: Exception
) -> None:
    async def fail(*_args: Any, **_kwargs: Any) -> Any:
        raise fatal_error

    async def tool(_request: Any) -> Any:
        pytest.fail("A fatal pre-tool error must not run the tool")

    monkeypatch.setattr(langchain_middleware, "dispatch_pre_tool_use", fail)
    middleware = langchain_middleware._langchain_approval_middleware(
        profile=AgentProfile.empty(), tools=()
    )
    with pytest.raises(type(fatal_error)) as raised:
        await middleware.awrap_tool_call(
            SimpleNamespace(tool_call={"name": "write", "args": {}, "id": "call"}), tool
        )
    assert raised.value is fatal_error


@pytest.mark.asyncio
@pytest.mark.parametrize("transport_fails", [False, True])
async def test_claude_fatal_hook_stops_native_run_and_preserves_original_error(
    monkeypatch: pytest.MonkeyPatch, fatal_error: Exception, transport_fails: bool
) -> None:
    _install_fake_claude_sdk(monkeypatch)
    sdk = sys.modules["claude_agent_sdk"]
    decisions: list[dict[str, Any]] = []
    attempts = 0

    async def fail(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal attempts
        attempts += 1
        raise fatal_error

    async def query(*, prompt: str, options: Any) -> Any:
        if prompt == "fail":
            hook = options.kwargs["hooks"]["PreToolUse"][0].hooks[0]
            for index in range(2):
                decisions.append(
                    await hook({"tool_name": "Write", "tool_input": {}}, f"call-{index}", None)
                )
            if transport_fails:
                raise RuntimeError("Native run stopped")
        await asyncio.sleep(0)
        yield sdk.ResultMessage(result="ready")

    monkeypatch.setattr(claude_hooks, "dispatch_pre_tool_use", fail)
    monkeypatch.setattr(sdk, "query", query)
    adapter = ClaudeAdapter(model="test-model")
    failed, successful = await asyncio.gather(
        adapter.invoke(prompt="fail"), adapter.invoke(prompt="ok"), return_exceptions=True
    )
    assert failed is fatal_error
    assert successful == "ready"
    assert attempts == 1
    assert len(decisions) == 2
    assert all(decision["continue_"] is False for decision in decisions)
    assert all(decision["stopReason"] == str(fatal_error) for decision in decisions)


@pytest.mark.asyncio
@pytest.mark.parametrize("event", ["post_tool_use", "post_tool_use_failure"])
async def test_claude_post_hook_failures_stop_the_native_run(
    monkeypatch: pytest.MonkeyPatch, event: str
) -> None:
    _install_fake_claude_sdk(monkeypatch)
    sdk = sys.modules["claude_agent_sdk"]
    failure = HookExecutionError(event=event, handler_name="observe", cause=ValueError("failed"))

    async def fail(*_args: Any, **_kwargs: Any) -> Any:
        raise failure

    async def query(*, prompt: str, options: Any) -> Any:
        native_event = "PostToolUse" if event == "post_tool_use" else "PostToolUseFailure"
        hook = options.kwargs["hooks"][native_event][0].hooks[0]
        result = await hook(
            {"tool_name": "Read", "tool_input": {}, "tool_response": "done"}, "call", None
        )
        assert result == {"continue_": False, "stopReason": str(failure)}
        yield sdk.ResultMessage(result="must not succeed")

    monkeypatch.setattr(claude_hooks, "dispatch_" + event, fail)
    monkeypatch.setattr(sdk, "query", query)
    adapter = ClaudeAdapter(model="test-model")
    adapter.on(event)(lambda _event: None)
    with pytest.raises(HookExecutionError) as raised:
        await adapter.invoke(prompt="Read")
    assert raised.value is failure
