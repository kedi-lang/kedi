from __future__ import annotations

import asyncio
from contextvars import ContextVar
from threading import get_ident

import pytest

from kedi.agent_adapter.approvals.policy import (
    ApprovalDecision,
    ApprovalPolicy,
    ApprovalRequest,
    resolve_tool_approval_async,
)
from kedi.agent_adapter.execution import run_sync, use_async_owner_loop


def request():
    return ApprovalRequest.create(
        tool_name="write_draft",
        arguments={},
        risk="mutating",
        adapter_shortname="test",
        description="Write a scratch draft",
        metadata=None,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit_owner", [False, True])
async def test_sync_approval_nested_inference_uses_owner_without_blocking(explicit_owner):
    from contextlib import nullcontext

    owner, host_thread = asyncio.get_running_loop(), get_ident()
    trace = ContextVar("approval_test_trace", default="outside")
    seen = []

    async def infer():
        seen.append((asyncio.get_running_loop(), trace.get()))
        await asyncio.sleep(0)
        return ApprovalDecision.allow(reason="reviewed")

    def handler(call):
        assert call.tool_name == "write_draft"
        assert get_ident() != host_thread
        assert trace.get() == "request"
        trace.set("handler")
        return run_sync(infer())

    token = trace.set("request")
    try:
        with use_async_owner_loop() if explicit_owner else nullcontext():
            decision = await asyncio.wait_for(
                resolve_tool_approval_async(ApprovalPolicy.dynamic(handler), request()), 5
            )
        assert decision.decision == "allow"
        assert seen == [(owner, "handler")]
        assert trace.get() == "request"
    finally:
        trace.reset(token)


@pytest.mark.asyncio
async def test_sync_approval_cancel_drains_nested_model_before_returning():
    entered, cleaned = asyncio.Event(), asyncio.Event()
    finished = []

    async def infer():
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()

    def handler(call):
        try:
            return run_sync(infer())
        finally:
            finished.append(call.tool_name)

    task = asyncio.create_task(
        resolve_tool_approval_async(ApprovalPolicy.dynamic(handler), request())
    )
    await asyncio.wait_for(entered.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 5)
    assert cleaned.is_set()
    assert finished == ["write_draft"]


@pytest.mark.asyncio
@pytest.mark.parametrize("wrapped", [False, True])
async def test_async_approval_and_sync_returning_awaitable_stay_supported(wrapped):
    owner = asyncio.get_running_loop()

    async def handler(call):
        assert asyncio.get_running_loop() is owner
        return ApprovalDecision.deny(reason=call.tool_name)

    policy = ApprovalPolicy.dynamic((lambda call: handler(call)) if wrapped else handler)
    decision = await resolve_tool_approval_async(policy, request())
    assert decision == ApprovalDecision.deny(reason="write_draft")
