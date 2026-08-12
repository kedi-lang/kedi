from __future__ import annotations

from collections.abc import Awaitable, Callable

from .models import CompactionCapabilities, CompactionOutcome

NativeCompact = Callable[[object, object], Awaitable[CompactionOutcome]]


class NativeCompactionBackend:
    """Thin normalized bridge that leaves history ownership with the native harness."""

    kind = "native"

    def __init__(
        self,
        capabilities: CompactionCapabilities,
        compact: NativeCompact,
    ) -> None:
        if not capabilities.native_managed:
            raise ValueError("native compaction backend requires native-managed capabilities")
        self.capabilities = capabilities
        self._compact = compact

    async def compact(self, state: object, policy: object) -> CompactionOutcome:
        outcome = await self._compact(state, policy)
        if outcome.backend != "native":
            raise ValueError("native compaction callback returned a non-native outcome")
        return outcome
