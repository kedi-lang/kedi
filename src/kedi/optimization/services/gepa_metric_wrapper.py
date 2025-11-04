"""Service for wrapping Kedi metrics for GEPA optimization."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    dspy = None  # type: ignore


class GEPAMetricWrapper:
    """Service for wrapping Kedi metrics to work with GEPA.
    
    GEPA metrics receive (gold: Example, pred: Prediction, ...) but
    Kedi metrics expect (input, expected_output). This service bridges
    the gap by extracting outputs from DSPy predictions and calling
    the Kedi metric directly - the metric handles all comparison logic.
    """
    
    def wrap_metric(
        self,
        kedi_metric: Callable[[Any, Any], tuple[float, str | None]],
        proc_params: list[str],
        output_fields: list[tuple[str, str | None]],
        proc_name: str | None = None,
        runtime: Any = None,
    ) -> Callable:
        """Wrap Kedi metric for GEPA.
        
        Args:
            kedi_metric: Kedi metric function (input, expected) -> (score, feedback)
            proc_params: Procedure parameter names
            output_fields: List of (field_name, type_annotation) tuples
            proc_name: Optional procedure name for analytical metric injection
            runtime: Optional Kedi runtime for procedure override
            
        Returns:
            GEPA-compatible metric function
        """
        # Capture proc_name and runtime in closure for analytical metrics
        captured_proc_name = proc_name
        captured_runtime = runtime
        
        def gepa_metric(
            gold: dspy.Example,
            pred: dspy.Prediction,
            trace=None,
            pred_name=None,
            pred_trace=None,
        ):
            """GEPA metric function."""
            try:
                # Extract input and expected output
                input_val = self._extract_input(gold, proc_params)
                expected_output = getattr(gold, "_expected", None)
                pred_output = self._extract_prediction_output(pred, output_fields, proc_params)
                
                # Override procedure to return GEPA's prediction
                if captured_proc_name and captured_runtime:
                    procedures_dict = captured_runtime._procedures
                    original_proc = procedures_dict[captured_proc_name]
                    procedures_dict[captured_proc_name] = lambda *args, **kwargs: pred_output
                    try:
                        score, feedback = kedi_metric(input_val, expected_output)
                    finally:
                        procedures_dict[captured_proc_name] = original_proc
                else:
                    score, feedback = kedi_metric(input_val, expected_output)
                
                # Convert to GEPA format
                return dspy.Prediction(score=float(score), feedback=feedback) if feedback else float(score)
            except Exception as e:
                import traceback
                return dspy.Prediction(score=0.0, feedback=traceback.format_exc())
        
        return gepa_metric
    
    def _extract_input(self, gold: dspy.Example, proc_params: list[str]) -> Any:
        """Extract input value from gold example.
        
        Examples have user_prompt (for DSPy) and original param fields (for metric).
        """
        if len(proc_params) == 1:
            return getattr(gold, proc_params[0])
        return tuple(getattr(gold, param) for param in proc_params)
    
    def _extract_prediction_output(
        self,
        pred: dspy.Prediction,
        output_fields: list[tuple[str, str | None]],
        proc_params: list[str],
    ) -> Any:
        """Extract output value from DSPy Prediction."""
        if not output_fields:
            attrs = {k: v for k, v in pred.__dict__.items() if not k.startswith('_') and k not in proc_params}
            if len(attrs) == 1:
                return list(attrs.values())[0]
            return tuple(attrs.values()) if attrs else str(pred)
        
        output_values = [getattr(pred, out_name, None) for out_name, _ in output_fields]
        if len(output_values) == 1:
            return output_values[0]
        return tuple(output_values)
    

