import asyncio
import time

import pytest

from kedi.agent_adapter.execution import (
    RunDeadlineExceeded,
    _operation_event_loop,
    run_sync,
    use_async_owner_loop,
    use_run_deadline,
)


@pytest.mark.parametrize("inside_loop", [False, True])
@pytest.mark.parametrize("deadline_offset", [-1.0, 0.02])
def test_deadline_cancels_and_unwinds_before_returning(
    inside_loop: bool, deadline_offset: float
) -> None:
    events: list[str] = []

    async def work() -> None:
        try:
            await asyncio.sleep(10)
        finally:
            await asyncio.sleep(0)
            events.append("closed")

    def invoke() -> None:
        async def start() -> asyncio.Task[None]:
            task = asyncio.create_task(work())
            await asyncio.sleep(0)
            return task

        async def cleanup(task: asyncio.Task[None]) -> None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

        with _operation_event_loop():
            # Start work before arming the deadline; loop startup can exceed 20 ms in CI.
            task = run_sync(start())
            try:
                with use_run_deadline(time.monotonic() + deadline_offset):
                    with pytest.raises(RunDeadlineExceeded):
                        run_sync(task)
                assert events == ["closed"]
                assert task.cancelled()
            finally:
                run_sync(cleanup(task))

    async def outer() -> None:
        invoke()

    if inside_loop:
        asyncio.run(outer())
    else:
        invoke()


def test_deadline_does_not_reclassify_provider_timeout() -> None:
    async def work() -> None:
        raise TimeoutError("provider timeout")

    with use_run_deadline(time.monotonic() + 10):
        with pytest.raises(TimeoutError, match="provider timeout") as error:
            run_sync(work())
    assert not isinstance(error.value, RunDeadlineExceeded)


def test_nested_scope_cannot_extend_deadline_and_resets_afterwards() -> None:
    with use_run_deadline(time.monotonic() + 0.02):
        with use_run_deadline(time.monotonic() + 10), use_run_deadline(None):
            with pytest.raises(RunDeadlineExceeded):
                run_sync(asyncio.sleep(10))
    assert run_sync(asyncio.sleep(0, result="ok")) == "ok"


@pytest.mark.asyncio
async def test_real_pydantic_adapter_cancels_pending_model_request() -> None:
    from pydantic_ai.messages import ModelMessage, ModelResponse
    from pydantic_ai.models.function import AgentInfo, FunctionModel

    from kedi.agent_adapter.adapters import PydanticAdapter

    closed: list[bool] = []
    started = asyncio.Event()

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        started.set()
        try:
            await asyncio.Event().wait()
            raise AssertionError("deadline did not cancel the model")
        finally:
            closed.append(True)

    adapter = PydanticAdapter(FunctionModel(respond))
    task = asyncio.create_task(adapter.invoke(prompt="wait"))

    def wait_with_deadline() -> None:
        with use_run_deadline(time.monotonic() + 0.02):
            with pytest.raises(RunDeadlineExceeded):
                run_sync(task)
        assert closed == [True]

    try:
        # Arm the sync bridge's deadline only after the model is actually pending.
        await asyncio.wait_for(started.wait(), 5)
        with use_async_owner_loop():
            await asyncio.wait_for(asyncio.to_thread(wait_with_deadline), 5)
        assert task.cancelled()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


def test_expired_deadline_does_not_start_pydantic_model_request() -> None:
    from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
    from pydantic_ai.models.function import AgentInfo, FunctionModel

    from kedi.agent_adapter.adapters import PydanticAdapter

    requests: list[str] = []

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        requests.append("started")
        return ModelResponse(parts=[TextPart("unexpected response")])

    adapter = PydanticAdapter(FunctionModel(respond))
    with use_run_deadline(time.monotonic() - 1):
        with pytest.raises(RunDeadlineExceeded):
            adapter.invoke_sync(prompt="wait")
    assert requests == []
