"""Service for extracting optimized prompts from DSPy modules."""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from typing import Any, cast

import dspy

from ...lang.ast import OptimizeSpan
from ..utils import normalize_optimized_prompt_prefix, span_to_instruction

DSPY_AVAILABLE = True


class DSPyPromptExtractor:
    """Service for extracting optimized instructions from DSPy modules.

    Handles various DSPy module structures to find optimized prompts
    from GEPA-optimized modules.
    """

    def extract_prompts(
        self,
        optimized_module: dspy.Module,
        spans: list[OptimizeSpan],
    ) -> dict[str, str]:
        """Extract optimized prompts from DSPy module.

        Args:
            optimized_module: GEPA-optimized DSPy module
            spans: Original optimize spans (for fallback)

        Returns:
            Dictionary mapping span.name -> optimized prompt string
        """
        if not DSPY_AVAILABLE:
            raise ImportError("DSPy is required")

        optimized_prompts = {}
        predictors = self._find_predictors(optimized_module)

        # Extract prompt for each span
        for i, span in enumerate(spans):
            predictor = self._find_predictor_for_span(predictors, optimized_module, span, i)
            prompt = self._extract_prompt_from_predictor(predictor, span)
            optimized_prompts[span.name] = prompt

        return optimized_prompts

    def _find_predictors(self, module: dspy.Module) -> dict[str, dspy.Predict]:
        """Find all Predict instances in module."""
        predictors = {}

        # Try named_predictors first
        try:
            named = module.named_predictors()
            if isinstance(named, dict):
                items = cast(Iterable[tuple[str, Any]], named.items())
            else:
                items = cast(Iterable[tuple[str, Any]], named)
            for name, predictor in items:
                if isinstance(predictor, dspy.Predict):
                    predictors[str(name)] = predictor
        except Exception:
            pass

        # Check for direct predictor attribute
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            predictor = getattr(module, "predictor", None)
            if isinstance(predictor, dspy.Predict):
                predictors["_main"] = predictor

        # Search all attributes
        if not predictors:
            for attr_name in dir(module):
                if attr_name.startswith("_"):
                    continue
                try:
                    attr = getattr(module, attr_name)
                    if isinstance(attr, dspy.Predict):
                        predictors[attr_name] = attr
                except Exception:
                    continue

        return predictors

    def _find_predictor_for_span(
        self,
        predictors: dict[str, dspy.Predict],
        module: dspy.Module,
        span: OptimizeSpan,
        index: int,
    ) -> dspy.Predict | None:
        """Find predictor for a specific span."""
        # Strategy 1: By span name
        if span.name in predictors:
            return predictors[span.name]

        # Strategy 2: Single predictor
        if len(predictors) == 1:
            return list(predictors.values())[0]

        # Strategy 3: By index
        if index < len(predictors):
            return list(predictors.values())[index]

        # Strategy 4: Common attributes
        predictor = getattr(module, "predictor", None)
        if isinstance(predictor, dspy.Predict):
            return predictor

        student_predictor = getattr(getattr(module, "student", None), "predictor", None)
        if isinstance(student_predictor, dspy.Predict):
            return student_predictor

        return None

    def _extract_prompt_from_predictor(
        self,
        predictor: dspy.Predict | None,
        span: OptimizeSpan,
    ) -> str:
        """Extract stored optimized prefix from predictor instructions."""
        if not predictor:
            return normalize_optimized_prompt_prefix(span, span_to_instruction(span))

        # Extract instructions from signature
        if not hasattr(predictor, "signature"):
            return normalize_optimized_prompt_prefix(span, span_to_instruction(span))

        sig = predictor.signature
        instructions = getattr(sig, "instructions", None) or getattr(sig, "_instructions", None)

        if not isinstance(instructions, str) or not instructions.strip():
            return normalize_optimized_prompt_prefix(span, span_to_instruction(span))

        # Filter out default DSPy signature instructions
        default_pattern = "Given the fields"
        if default_pattern not in instructions or len(instructions) > len(default_pattern) + 50:
            return normalize_optimized_prompt_prefix(span, instructions)

        return normalize_optimized_prompt_prefix(span, span_to_instruction(span))
