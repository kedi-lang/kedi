"""Validation metadata shared by Python annotations and Kedi interop."""

from __future__ import annotations

from re import Pattern

from annotated_types import SupportsGe, SupportsGt, SupportsLe, SupportsLt
from pydantic import Field
from pydantic.fields import FieldInfo


def Constraints(
    *,
    ge: SupportsGe | None = None,
    gt: SupportsGt | None = None,
    le: SupportsLe | None = None,
    lt: SupportsLt | None = None,
    multiple_of: float | None = None,
    min_length: int | None = None,
    max_length: int | None = None,
    pattern: str | Pattern[str] | None = None,
    strict: bool | None = None,
    allow_inf_nan: bool | None = None,
    max_digits: int | None = None,
    decimal_places: int | None = None,
) -> FieldInfo:
    """Build validation-only metadata for ``Annotated`` or a Pydantic field.

    Validation and schema generation follow Pydantic's ``Field`` semantics.
    Omitted options leave the underlying type's behavior unchanged. Defaults,
    aliases, descriptions, and serialization settings are intentionally excluded.
    """
    return Field(
        ge=ge,
        gt=gt,
        le=le,
        lt=lt,
        multiple_of=multiple_of,
        min_length=min_length,
        max_length=max_length,
        pattern=pattern,
        strict=strict,
        allow_inf_nan=allow_inf_nan,
        max_digits=max_digits,
        decimal_places=decimal_places,
    )


__all__ = ["Constraints"]
