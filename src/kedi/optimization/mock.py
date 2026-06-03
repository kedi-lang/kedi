"""Mock optimizer for testing and development.

Simulates optimization with iterative improvements and score tracking.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from ..lang.ast import OptimizeSpan
from .interfaces import OptimizerAdapter
from .utils import span_to_instruction


class MockOptimizer(OptimizerAdapter):
    """Mock optimizer that simulates optimization with score tracking.

    Features:
    - Type validation as precondition
    - Simulates iterative optimization over training data
    - Tracks scores across multiple "rounds"
    - Returns optimized prompts with subtle improvements
    - Handles single-step and multi-step optimizations
    """

    def optimize_procedure(
        self,
        proc_name: str,
        proc_params: list[str],
        param_types: dict[str, type],
        spans: list[OptimizeSpan],
        train_data: Iterable[tuple[Any, Any]],
        test_data: Iterable[tuple[Any, Any]] | None,
        metric: Callable[[Any, Any], tuple[float, str | None]],
    ) -> tuple[dict[str, str], float]:
        """Optimize procedure spans using training data.

        Simulates optimization by:
        1. Validating types for all training examples
        2. Running multiple optimization rounds
        3. Tracking scores per round
        4. Returning "optimized" prompts with improvements
        """
        # Convert iterable to list for validation and iteration
        train_list = list(train_data)

        # Handle empty train_data (validation before spans check)
        if not train_list:
            raise ValueError(f"No training data provided for '{proc_name}'")

        # Type validation as precondition (5.b)
        self._validate_types(proc_name, proc_params, param_types, train_list)

        # Handle empty spans (after validation)
        if not spans:
            return {}, 0.0

        # Get original prompts
        original_prompts = {span.name: span_to_instruction(span) for span in spans}

        # Simulate optimization rounds (3-5 iterations)
        num_rounds = 3
        all_scores: list[float] = []

        for round_num in range(num_rounds):
            round_scores: list[float] = []

            for input_val, expected_output in train_list:
                try:
                    # Call metric which internally calls procedure
                    score, feedback = metric(input_val, expected_output)

                    # Ensure score is float in [0.0, 1.0] range
                    score = max(0.0, min(1.0, float(score)))
                    round_scores.append(score)
                except Exception:
                    # Handle metric errors - return 0.0 score
                    round_scores.append(0.0)

            # Track scores for this round
            if round_scores:
                all_scores.extend(round_scores)

        # Calculate final training score as average of all metric calls
        train_score = sum(all_scores) / len(all_scores) if all_scores else 0.0

        # Generate "optimized" prompts with subtle improvements
        optimized_prompts = {}
        for span in spans:
            original = original_prompts[span.name]

            # Add improvement prefix (simulating optimization)
            if "Think step by step" not in original:
                optimized = "Think step by step.\n\n" + original
            elif "Consider all aspects carefully" not in original:
                optimized = original + "\n\nConsider all aspects carefully."
            else:
                # Already "optimized", just ensure it's slightly different
                optimized = original + " [optimized]"

            optimized_prompts[span.name] = optimized

        return optimized_prompts, train_score

    def _validate_types(
        self,
        proc_name: str,
        proc_params: list[str],
        param_types: dict[str, type],
        train_data: list[tuple[Any, Any]],
    ) -> None:
        """Validate types for all training examples.

        Raises:
            TypeError: If any example fails type validation
        """
        if not proc_params:
            # No parameters to validate
            return

        for idx, example in enumerate(train_data):
            # Validate example is a tuple with 2 elements
            if not isinstance(example, tuple) or len(example) != 2:
                raise TypeError(
                    f"Training example {idx} for '{proc_name}': "
                    f"Expected tuple (input, expected_output), got {type(example).__name__}"
                )

            input_val, expected_output = example

            # Handle single param vs multi-param
            if len(proc_params) == 1:
                # Single param: input_val is the value itself (not a tuple)
                param_name = proc_params[0]
                expected_type = param_types.get(param_name)
                if expected_type and not isinstance(input_val, expected_type):
                    raise TypeError(
                        f"Training example {idx} for '{proc_name}': "
                        f"Parameter '{param_name}' expected {expected_type.__name__}, "
                        f"got {type(input_val).__name__}"
                    )
            else:
                # Multi-param: input_val must be tuple with matching length
                if not isinstance(input_val, tuple):
                    raise TypeError(
                        f"Training example {idx} for '{proc_name}': "
                        f"Expected tuple of {len(proc_params)} values, got {type(input_val).__name__}"
                    )

                if len(input_val) != len(proc_params):
                    raise TypeError(
                        f"Training example {idx} for '{proc_name}': "
                        f"Expected tuple of length {len(proc_params)}, got {len(input_val)}"
                    )

                # Validate each element matches corresponding param type
                for param_idx, (param_name, arg_val) in enumerate(zip(proc_params, input_val)):
                    expected_type = param_types.get(param_name)
                    if expected_type and not isinstance(arg_val, expected_type):
                        raise TypeError(
                            f"Training example {idx} for '{proc_name}': "
                            f"Parameter '{param_name}' (position {param_idx}) expected "
                            f"{expected_type.__name__}, got {type(arg_val).__name__}"
                        )
