"""Service for calculating training scores from DSPy modules."""

from __future__ import annotations

from collections.abc import Callable
import math

try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    dspy = None  # type: ignore


class DSPyScoreCalculator:
    """Service for calculating average training scores.
    
    Evaluates a module on a sample of training examples using a metric
    to compute average score.
    """
    
    def calculate_score(
        self,
        module: dspy.Module,
        trainset: list[dspy.Example],
        metric: Callable,
        sample_size: int = 10,
    ) -> float:
        """Calculate average training score.
        
        Args:
            module: DSPy module to evaluate
            trainset: List of training examples
            metric: Metric function to use
            sample_size: Number of examples to sample (default: 10)
            
        Returns:
            Average score (0.0-1.0)
        """
        if not DSPY_AVAILABLE:
            raise ImportError("DSPy is required")
        
        scores = []
        sample = trainset[:min(sample_size, len(trainset))]
        
        for example in sample:
            try:
                inputs = self._extract_inputs(example)
                # If inputs are available, evaluate the module to obtain a prediction
                if inputs:
                    pred = module(**inputs)
                    score_result = metric(example, pred)
                else:
                    # Prefix-only program: predictions are irrelevant; call metric directly
                    score_result = metric(example, None)

                # Extract score from result
                if hasattr(score_result, 'score'):
                    score_val = float(score_result.score)
                else:
                    score_val = float(score_result)

                # Sanitize NaN/Inf and clamp into [0.0, 1.0] for stability
                if not math.isfinite(score_val):
                    score_val = 0.0
                # Do not force-clamp if metric is on a different scale; keep within sane bounds
                if score_val < 0.0:
                    score_val = 0.0
                if score_val > 1.0:
                    score_val = min(score_val, 1.0)

                scores.append(score_val)
            except Exception:
                scores.append(0.0)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def _extract_inputs(self, example: dspy.Example) -> dict[str, Any]:
        """Extract input dictionary from example."""
        # Try inputs() method first
        if hasattr(example, 'inputs') and callable(example.inputs):
            inputs = example.inputs()
            if inputs:
                return inputs
        
        # Fallback: extract from __dict__
        if hasattr(example, '__dict__'):
            return {
                k: v for k, v in example.__dict__.items()
                if not k.startswith('_')
            }
        
        return {}
