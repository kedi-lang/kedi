"""Optimizer adapter interface."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, Protocol

from ..lang.ast import OptimizeSpan
from ..resolved_types import ResolvedKediType


class OptimizerAdapter(Protocol):
    """Interface for optimizers that can optimize prompts in procedures.

    The interface follows the DSPy GEPA pattern where optimizers orchestrate
    training loops using metrics to score candidates. Optimizers are
    implementation-agnostic and receive clean data/metadata.
    """

    def optimize_procedure(
        self,
        proc_name: str,
        proc_params: list[str],
        param_types: dict[str, ResolvedKediType],
        spans: list[OptimizeSpan],
        train_data: Iterable[tuple[Any, Any]],
        test_data: Iterable[tuple[Any, Any]] | None,
        metric: Callable[[Any, Any], tuple[float, str | None]],
    ) -> tuple[dict[str, str], float]:
        """Optimize procedure spans using training data.

        Args:
            proc_name: Procedure name
            proc_params: List of procedure parameter names
            param_types: Mapping of parameter name -> Python type for validation
            spans: List of optimize spans to optimize
            train_data: Iterable of (input, expected_output) tuples.
                - Single param: (value, expected)
                - Multi-param: ((arg1, arg2, ...), expected)
            test_data: Optional test dataset with same format as train_data
            metric: Function (input, expected_output) -> (score, feedback).
                Metric calls procedure internally and compares results.
                Must always return (float, str | None) tuple.

        Returns:
            Tuple of (optimized_prompts, train_score) where:
            - optimized_prompts: Dict mapping span.name -> optimized prompt string
            - train_score: Average metric score over training data (0.0-1.0)

        Raises:
            TypeError: If type validation fails for any training example
            ValueError: If train_data is empty or spans is empty
            RuntimeError: If optimization fails
        """
        ...
