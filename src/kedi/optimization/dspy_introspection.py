from __future__ import annotations

from typing import Any

import dspy

DEFAULT_INSTRUCTION_PATTERN = "Given the fields"
DEFAULT_INSTRUCTION_MIN_EXTRA = 50

__all__ = [
    "DEFAULT_INSTRUCTION_MIN_EXTRA",
    "DEFAULT_INSTRUCTION_PATTERN",
    "extract_signature_instructions",
    "find_primary_predictor",
    "is_non_default_instruction",
]


def find_primary_predictor(module: Any) -> dspy.Predict | None:
    """Return the student ``dspy.Predict`` on a compiled module, if present."""
    student_predictor = getattr(getattr(module, "student", None), "predictor", None)
    if isinstance(student_predictor, dspy.Predict):
        return student_predictor
    return None


def extract_signature_instructions(predictor: dspy.Predict | None) -> str | None:
    if predictor is None or not hasattr(predictor, "signature"):
        return None
    sig = predictor.signature
    instructions = getattr(sig, "instructions", None) or getattr(sig, "_instructions", None)
    if not isinstance(instructions, str):
        return None
    stripped = instructions.strip()
    return stripped or None


def is_non_default_instruction(instructions: str) -> bool:
    if DEFAULT_INSTRUCTION_PATTERN not in instructions:
        return True
    return len(instructions) > len(DEFAULT_INSTRUCTION_PATTERN) + DEFAULT_INSTRUCTION_MIN_EXTRA
