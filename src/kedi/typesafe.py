"""Optional Jev criteria for Kedi type annotations."""

from __future__ import annotations

try:
    from kedi_typesafe import BooleanCriteria, ChoiceCriteria, Probability, Rubric
except ModuleNotFoundError as exc:
    if exc.name != "kedi_typesafe":
        raise
    raise ModuleNotFoundError(
        "kedi.typesafe requires the optional kedi-typesafe package. "
        "Install it with: uv add 'kedi[typesafe]'",
        name="kedi_typesafe",
    ) from exc

__all__ = ["BooleanCriteria", "ChoiceCriteria", "Probability", "Rubric"]
