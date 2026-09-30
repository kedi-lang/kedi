"""Optional Laya decision types for Kedi and Python annotations."""

from __future__ import annotations

try:
    from kedi_laya import BooleanCriteria, ChoiceCriteria, Probability, Rubric
except ModuleNotFoundError as exc:
    if exc.name != "kedi_laya":
        raise
    raise ModuleNotFoundError(
        "kedi.laya requires the optional kedi-laya package. "
        "Install 'kedi-decisions[laya]' or, on Apple Silicon, 'kedi-decisions[laya-mlx]'.",
        name="kedi_laya",
    ) from exc

__all__ = ["BooleanCriteria", "ChoiceCriteria", "Probability", "Rubric"]
