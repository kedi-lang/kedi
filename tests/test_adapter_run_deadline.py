import asyncio
import time

import pytest

from kedi.agent_adapter.utils import RunDeadlineExceeded, run_sync, use_run_deadline


@pytest.mark.parametrize("inside_loop", [False, True])
def test_deadline_cancels_and_unwinds_before_returning(inside_loop: bool) -> None:
    events: list[str] = []

    async def work() -> None:
        try:
            await asyncio.sleep(10)
        finally:
            await asyncio.sleep(0)
            events.append("closed")

    def invoke() -> None:
        with use_run_deadline(time.monotonic() + 0.02):
            with pytest.raises(RunDeadlineExceeded):
                run_sync(work())
        assert events == ["closed"]

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


def test_real_pydantic_adapter_cancels_pending_model_request() -> None:
    from pydantic_ai.messages import ModelMessage, ModelResponse
    from pydantic_ai.models.function import AgentInfo, FunctionModel

    from kedi.agent_adapter.adapters import PydanticAdapter

    closed: list[bool] = []

    async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        try:
            await asyncio.sleep(10)
            raise AssertionError("deadline did not cancel the model")
        finally:
            closed.append(True)

    adapter = PydanticAdapter(FunctionModel(respond))
    with use_run_deadline(time.monotonic() + 0.2):
        with pytest.raises(RunDeadlineExceeded):
            adapter.invoke_sync(prompt="wait")
    assert closed == [True]
