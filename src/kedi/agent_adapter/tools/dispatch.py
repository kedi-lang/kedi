"""Invocation-local scheduling shared by native tools and CodeMode."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager


class SequentialDispatchGate:
    """Run sequential tools exclusively without serializing unrelated calls."""

    def __init__(self) -> None:
        self._condition = asyncio.Condition()
        self._active_parallel = 0
        self._exclusive = False
        self._waiting_exclusive = 0

    @asynccontextmanager
    async def enter(self, *, sequential: bool) -> AsyncGenerator[None, None]:
        if sequential:
            async with self._condition:
                self._waiting_exclusive += 1
                try:
                    await self._condition.wait_for(
                        lambda: not self._exclusive and self._active_parallel == 0
                    )
                except BaseException:
                    self._waiting_exclusive -= 1
                    self._condition.notify_all()
                    raise
                self._waiting_exclusive -= 1
                self._exclusive = True
            try:
                yield
            finally:
                self._exclusive = False
                await self._notify_waiters()
            return

        async with self._condition:
            await self._condition.wait_for(
                lambda: not self._exclusive and self._waiting_exclusive == 0
            )
            self._active_parallel += 1
        try:
            yield
        finally:
            self._active_parallel -= 1
            if self._active_parallel == 0:
                await self._notify_waiters()

    async def _notify_waiters(self) -> None:
        async def notify() -> None:
            async with self._condition:
                self._condition.notify_all()

        # Admission state is released synchronously; notification must survive
        # cancellation while another task temporarily owns the condition lock.
        notification = notify()
        if self._condition.locked():
            await asyncio.shield(notification)
        else:
            await notification
