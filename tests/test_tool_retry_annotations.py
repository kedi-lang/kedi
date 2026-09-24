from __future__ import annotations

import asyncio
import inspect
from functools import partial, wraps
from typing import Annotated, get_type_hints

import pytest

from kedi.agent_adapter.tools.retry import ToolRetryPolicy, tool_retry_policy, with_body_retry


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("prebound", [False, True])
def test_callable_retry_annotations_survive_another_wrapper(
    asynchronous: bool, prebound: bool
) -> None:
    class SyncTool:
        def __call__(self, url: Annotated[str, "Target URL"], timeout: int = 10) -> str:
            return f"{url}:{timeout}"

    class AsyncTool:
        async def __call__(self, url: Annotated[str, "Target URL"], timeout: int = 10) -> str:
            return f"{url}:{timeout}"

    instance = AsyncTool() if asynchronous else SyncTool()
    target = partial(instance, timeout=2) if prebound else instance
    policy = ToolRetryPolicy(retries=1)
    retried = with_body_retry(target, policy, isolate_arguments=True)

    @wraps(retried)
    def outer(*args, **kwargs):
        return retried(*args, **kwargs)

    assert get_type_hints(outer, include_extras=True) == {
        "url": Annotated[str, "Target URL"],
        "timeout": int,
        "return": str,
    }
    assert inspect.signature(outer) == inspect.signature(target)
    assert tool_retry_policy(outer) == policy
    result = outer(url="https://example.com")
    if asynchronous:
        result = asyncio.run(result)
    assert result == f"https://example.com:{2 if prebound else 10}"
