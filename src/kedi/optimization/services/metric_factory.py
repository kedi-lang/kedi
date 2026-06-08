"""Service for creating metric functions that evaluate procedures."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ...lang.ast import EvalMetric, ProcedureDef
from ...lang.compiler import _eval_python_inline, _execute_python_block
from ...lang.compiler.environment import KediRuntime
from ..utils import normalize_metric_result


class MetricFactory:
    """Service for creating metric closures that call procedures.

    Metrics are Python code blocks that receive dataset values and call
    the procedure, then compare results to expected outputs.
    """

    def __init__(self, runtime: KediRuntime):
        """Initialize metric factory.

        Args:
            runtime: Compiled Kedi runtime with access to procedures
        """
        self.runtime = runtime

    def create_metric(
        self,
        metric_def: EvalMetric,
        procedure: ProcedureDef,
        dataset_name: str,
    ) -> Callable[[Any, Any], tuple[float, str | None]]:
        """Create metric function that calls procedure.

        The metric function:
        - Receives (input_val, expected_output) from optimizer
        - Executes metric code with dataset variable bound
        - Metric code internally calls procedure and compares results
        - Returns (score, feedback) tuple

        Args:
            metric_def: Metric definition from eval suite
            procedure: Procedure being optimized
            dataset_name: Name of dataset variable in metric code

        Returns:
            Metric function: (input, expected_output) -> (score, feedback)
        """

        def metric_fn(input_val: Any, expected_output: Any) -> tuple[float, str | None]:
            """Metric function called by optimizer.

            Args:
                input_val: Input value(s) - single value or tuple for multi-param
                expected_output: Expected output for comparison

            Returns:
                (score, feedback) tuple
            """
            try:
                # Prepare metric environment with dataset variable
                # Dataset can be either input_val or (input_val, expected_output) tuple
                metric_env = {dataset_name: input_val}

                # Also provide expected_output as a separate variable
                metric_env["expected"] = expected_output

                # If single param, also bind directly to param name for convenience
                if len(procedure.params) == 1:
                    metric_env[procedure.params[0]] = input_val
                else:
                    # Multi-param: unpack tuple into individual params
                    if isinstance(input_val, tuple):
                        for param_name, arg_val in zip(procedure.params, input_val):
                            metric_env[param_name] = arg_val

                # Execute metric code (which internally calls procedure)
                # Set _current_env before execution so rt.m() calls work
                token = self.runtime._current_env.set(metric_env)
                try:
                    if metric_def.is_block:
                        val = _execute_python_block(metric_def.code, self.runtime, metric_env)
                    else:
                        val = _eval_python_inline(self.runtime, metric_def.code, metric_env)
                finally:
                    self.runtime._current_env.reset(token)

                # Process result - must return (float, str | None)
                return normalize_metric_result(val)

            except Exception as e:
                # Metric execution failed - return 0.0 score with error
                return (0.0, f"error: {e}")

        return metric_fn
