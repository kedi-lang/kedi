"""Backward-compatible re-exports for optimizer type resolution helpers."""

from __future__ import annotations

from ..resolved_types import (
    ResolvedKediType,
    as_resolved_kedi_type,
    is_resolved_kedi_type,
    resolved_type_name,
    runtime_isinstance,
)

__all__ = [
    "ResolvedKediType",
    "as_resolved_kedi_type",
    "is_resolved_kedi_type",
    "resolved_type_name",
    "runtime_isinstance",
]
