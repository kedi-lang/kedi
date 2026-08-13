from __future__ import annotations

from typing import Protocol, runtime_checkable

from .models import (
    CompactionBackendKind,
    CompactionCapabilities,
    CompactionMode,
    CompactionOutcome,
    CompactionPreflightError,
)


@runtime_checkable
class CompactionBackend(Protocol):
    """Normalized compaction boundary for native and Kedi-owned implementations."""

    kind: CompactionBackendKind
    capabilities: CompactionCapabilities

    async def compact(self, state: object, policy: object) -> CompactionOutcome:
        """Compact one conversation according to the resolved backend policy."""
        ...


def resolve_compaction_backend(
    mode: CompactionMode,
    capabilities: CompactionCapabilities,
) -> CompactionBackendKind:
    """Resolve compaction ownership or reject unsupported policy before model I/O."""

    if mode == "disabled":
        if capabilities.native_managed and not capabilities.can_disable:
            raise CompactionPreflightError("native compaction cannot be disabled by this adapter")
        return "disabled"
    if mode == "native":
        if not capabilities.native_managed:
            raise CompactionPreflightError("native compaction is not supported by this adapter")
        return "native"
    if mode == "kedi":
        if not capabilities.supports_kedi_backend:
            raise CompactionPreflightError(
                "Kedi compaction requires lossless history export and replacement"
            )
        return "kedi"
    if capabilities.native_managed:
        return "native"
    if capabilities.supports_kedi_backend:
        return "kedi"
    raise CompactionPreflightError(
        "automatic compaction requires a native backend or lossless history replacement"
    )
