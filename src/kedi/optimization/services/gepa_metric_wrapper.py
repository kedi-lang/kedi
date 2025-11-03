"""Service for wrapping Kedi metrics for GEPA optimization."""

from __future__ import annotations

import re
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
    the gap by extracting outputs from DSPy predictions and replicating
    the Kedi metric's comparison logic.
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
            # Extract input from gold example
            input_val = self._extract_input(gold, proc_params)
            
            # Get expected output (may be None for analytical metrics)
            expected_output = getattr(gold, "_expected", None)
            
            # Extract prediction output
            pred_output = self._extract_prediction_output(pred, output_fields, proc_params)
            
            # For analytical metrics (expected_output is None), the metric computes expected internally
            # The metric code calls the procedure, but we want to use GEPA's optimized prediction instead
            # Solution: Temporarily override the procedure in the runtime to return GEPA's prediction
            if expected_output is None:
                try:
                    # For analytical metrics, we need to inject GEPA's prediction into the metric execution
                    # The metric code calls the procedure (e.g., find_maximum(A, B)), but we want it to
                    # return GEPA's prediction (pred_output) instead of calling the actual procedure
                    if captured_proc_name and captured_runtime:
                        # Temporarily override the procedure in runtime._procedures to return pred_output
                        # Procedures are stored in rt._procedures dict, not as attributes
                        procedures_dict = getattr(captured_runtime, '_procedures', None)
                        if procedures_dict is not None and captured_proc_name in procedures_dict:
                            original_proc = procedures_dict[captured_proc_name]
                            
                            def override_procedure(*args, **kwargs):
                                return pred_output
                            
                            # Patch the procedure in the procedures dict
                            procedures_dict[captured_proc_name] = override_procedure
                            
                            try:
                                # Call metric - it will call procedure and get pred_output
                                if pred_output is None:
                                    return dspy.Prediction(
                                        score=0.0,
                                        feedback="pred_output is None"
                                    )
                                
                                score, feedback = kedi_metric(input_val, None)
                            except Exception as e:
                                # If metric fails, return error score with full traceback
                                import traceback
                                error_details = f"{type(e).__name__}: {e}\n{traceback.format_exc()[:500]}"
                                return dspy.Prediction(
                                    score=0.0,
                                    feedback=f"Metric execution error: {error_details}"
                                )
                            finally:
                                # Restore original procedure
                                procedures_dict[captured_proc_name] = original_proc
                        else:
                            # Procedure not found in dict, fallback
                            score, feedback = kedi_metric(input_val, None)
                    else:
                        # Fallback: call metric normally (will use original prompt)
                        score, feedback = kedi_metric(input_val, None)
                    
                    # Convert to GEPA format
                    if feedback:
                        return dspy.Prediction(score=float(score), feedback=feedback)
                    return float(score)
                except Exception as e:
                    import traceback
                    error_details = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
                    return dspy.Prediction(score=0.0, feedback=f"Metric evaluation error: {error_details}")
            
            # For non-analytical metrics (with expected outputs), use comparison logic
            return self._compare_outputs(pred_output, expected_output, input_val)
        
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
        """Extract output value from DSPy Prediction.
        
        For multiple output fields, returns a tuple preserving field order.
        For single output field, returns the value directly.
        """
        if not output_fields:
            # No output fields specified - fallback
            attrs = {
                k: v for k, v in pred.__dict__.items()
                if not k.startswith('_') and k not in proc_params
            }
            if attrs:
                if len(attrs) == 1:
                    return list(attrs.values())[0]
                return tuple(attrs.values()) if len(attrs) > 1 else str(pred)
            return str(pred)
        
        # Extract all output field values in order
        output_field_names = [name for name, _ in output_fields]
        output_values = []
        for out_name, type_anno in output_fields:
            if hasattr(pred, out_name):
                value = getattr(pred, out_name)
            elif hasattr(pred, '__dict__') and out_name in pred.__dict__:
                value = pred.__dict__[out_name]
            else:
                # Field not found - try fallback
                value = None
            
            # DSPy should return proper types when using structured signatures
            # No conversion needed - DSPy handles it automatically
            output_values.append(value)
        
        # If we have all values, return tuple for multiple fields, single value for one field
        if all(v is not None for v in output_values):
            if len(output_values) == 1:
                return output_values[0]
            return tuple(output_values)
        
        # Some fields missing - try alternative extraction
        if hasattr(pred, '__dict__'):
            # Get all non-param attributes
            attrs = {
                k: v for k, v in pred.__dict__.items()
                if not k.startswith('_') and k not in proc_params
            }
            if attrs:
                if len(attrs) == 1:
                    return list(attrs.values())[0]
                # Try to match with output field order
                output_field_names = [name for name, _ in output_fields]
                ordered = [attrs.get(name) for name in output_field_names if name in attrs]
                if ordered and len(ordered) == len(output_field_names):
                    return tuple(ordered) if len(ordered) > 1 else ordered[0]
                return tuple(attrs.values()) if len(attrs) > 1 else list(attrs.values())[0]
        
        # Final fallback: return string representation
        return str(pred)
    
    def _compare_outputs(
        self,
        pred_output: Any,
        expected_output: Any,
        input_val: Any,
    ) -> float | dspy.Prediction:
        """Compare predicted output with expected output.
        
        Replicates Kedi metric comparison logic, handling type conversions
        (especially for integer extraction from strings).
        """
        if expected_output is None:
            return dspy.Prediction(score=0.0, feedback="No expected output provided for comparison")
        
        try:
            # Normalize predicted answer
            predicted_answer = self._normalize_answer(pred_output)
            
            # Compare
            correct = predicted_answer == expected_output
            
            if correct:
                return 1.0
            
            feedback = (
                f"Incorrect answer for problem: '{str(input_val)[:50]}...'. "
                f"Expected {expected_output}, got {predicted_answer}. "
                f"Review the reasoning process and verify all computational steps are correct."
            )
            return dspy.Prediction(score=0.0, feedback=feedback)
            
        except Exception as e:
            # Fallback on error
            try:
                predicted_answer = self._normalize_answer(pred_output)
                if predicted_answer == expected_output:
                    return 1.0
            except Exception:
                pass
            
            return dspy.Prediction(score=0.0, feedback=f"Metric evaluation error: {e}")
    
    def _normalize_answer(self, value: Any) -> Any:
        """Normalize answer value for comparison.
        
        Handles type conversion, especially extracting integers from strings.
        """
        # If already an int, use as-is
        if isinstance(value, int):
            return value
        
        # Extract integer from string
        if isinstance(value, str):
            match = re.search(r'\d+', value)
            if match:
                return int(match.group())
            return None
        
        # Convert float to int if whole number
        if isinstance(value, float):
            if value.is_integer():
                return int(value)
            return None
        
        # Try direct conversion
        try:
            return int(value)
        except (ValueError, TypeError):
            return None

