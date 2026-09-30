from __future__ import annotations

import asyncio
from typing import Awaitable, TypeVar


_T = TypeVar("_T")


async def finish_storage_operation(awaitable: Awaitable[_T]) -> _T:
    """Drain owned storage work before delivering caller cancellation."""
    operation = asyncio.ensure_future(awaitable)
    cancellation: asyncio.CancelledError | None = None
    while True:
        try:
            result = await asyncio.shield(operation)
            break
        except asyncio.CancelledError as exc:
            if operation.cancelled():
                raise
            cancellation = exc
        except BaseException:
            if cancellation is not None:
                raise cancellation
            raise
    if cancellation is not None:
        raise cancellation
    return result
