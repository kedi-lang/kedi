"""Service for wrapping Kedi metrics for GEPA optimization."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Optional
import weakref

try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    dspy = None  # type: ignore

from ...lang.ast import OutputSegment, TemplateLine
from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback
from ..utils import extract_output_fields


class GEPAMetricWrapper:
    """Service for wrapping Kedi metrics to work with GEPA.
    
    GEPA metrics receive (gold: Example, pred: Prediction, ...) but
    Kedi metrics expect (input, expected_output). This service bridges
    the gap by extracting inputs from DSPy examples and calling the Kedi
    metric directly. The metric calls the procedure normally, which uses
    the optimized prompt injected into the span's template.
    """

    def __init__(self) -> None:
        self._prediction_cache: "weakref.WeakKeyDictionary[Any, tuple[float, Optional[str]]]" = (
            weakref.WeakKeyDictionary()
        )
        # Fallback cache keyed by id() for objects that are not weakref-able
        self._prediction_id_cache: dict[int, tuple[float, Optional[str]]] = {}
    
    def wrap_metric(
        self,
        kedi_metric: Callable[[Any, Any], tuple[float, str | None]],
        proc_params: list[str],
        span: Any = None,
        module_ref: Any = None,
        proc_def: Any = None,
    ) -> Callable:
        """Wrap Kedi metric for GEPA.
        
        Args:
            kedi_metric: Kedi metric function (input, expected) -> (score, feedback)
            proc_params: Procedure parameter names
            span: OptimizeSpan to inject optimized prompt into
            module_ref: Reference to GEPA's module (for extracting current instructions)
            
        Returns:
            GEPA-compatible metric function
        """
        def gepa_metric(
            gold: dspy.Example,
            pred: dspy.Prediction,
            trace=None,
            pred_name=None,
            pred_trace=None,
            **context: Any,
        ):
            """GEPA metric function."""
            try:
                example_obj = context.get("module_inputs", gold)
                prediction_obj = context.get("module_outputs", pred)

                # Derive target output field names for extraction/override
                if isinstance(span, list):
                    target_fields = [name for (name, _type) in span if isinstance(name, str)]
                else:
                    target_fields = [name for name, _ in extract_output_fields(span)] if span else []
                
                cached = self._lookup_cached_feedback(prediction_obj) or self._lookup_cached_feedback(pred)
                if cached is not None:
                    cached_score, cached_feedback = cached
                    return ScoreWithFeedback(score=cached_score, feedback=cached_feedback or "")
                
                input_val = self._extract_input(example_obj, proc_params)
                expected_output = getattr(example_obj, "_expected", None)
                
                prompt_handle = None
                override_handle = None
                sibling_overrides = None
                try:
                    if module_ref:
                        optimized_prompt = self._extract_current_instructions(module_ref)
                        prompt_handle = self._apply_prompt_prefix(span, optimized_prompt)
                    
                    predicted_outputs = self._collect_prediction_outputs(prediction_obj or pred, target_fields)
                    if predicted_outputs and hasattr(span, "template_lines"):
                        override_handle = self._apply_output_override(span, predicted_outputs)
                    # Prevent non-active optimize spans from invoking the LLM
                    if proc_def is not None:
                        sibling_overrides = self._apply_default_overrides(proc_def, span)
                    
                    score, feedback = kedi_metric(input_val, expected_output)
                    score_float = float(score)
                    feedback_text = self._stringify_feedback(feedback)
                    payload = ScoreWithFeedback(score=score_float, feedback=feedback_text or "")
                    self._store_cached_feedback(prediction_obj, score_float, feedback_text)
                    self._store_cached_feedback(pred, score_float, feedback_text)
                    return payload
                finally:
                    self._restore_output_override(override_handle)
                    self._restore_sibling_overrides(sibling_overrides)
                    self._restore_prompt_prefix(prompt_handle)
            except Exception as e:
                import traceback
                error_msg = traceback.format_exc()
                payload = ScoreWithFeedback(score=0.0, feedback=f"GEPA metric error: {error_msg}")
                self._store_cached_feedback(pred, 0.0, payload.feedback)
                return payload
        
        return gepa_metric
    
    def _collect_prediction_outputs(self, pred: dspy.Prediction, target_fields: list[str]) -> dict[str, Any]:
        """Collect predicted outputs keyed by output name."""
        outputs: dict[str, Any] = {}
        if pred is None:
            return outputs
        
        
        def _assign(name: str, value: Any) -> None:
            if name and value is not None and not callable(value):
                outputs[name] = value
        
        if target_fields:
            for field in target_fields:
                if hasattr(pred, field):
                    _assign(field, getattr(pred, field))
        
        if outputs:
            return outputs
        
        # Fallback: single value predictions or unnamed attributes
        if not hasattr(pred, "__dict__"):
            if target_fields:
                outputs[target_fields[0]] = pred
            else:
                outputs["value"] = pred
            return outputs
        
        for attr, value in pred.__dict__.items():
            if attr.startswith("_") or attr in ("score", "feedback"):
                continue
            _assign(attr, value)
        
        if not outputs and target_fields:
            # Still nothing - attach first field as raw prediction
            outputs[target_fields[0]] = getattr(pred, target_fields[0], pred)
        
        return outputs
    
    def _extract_input(self, gold: dspy.Example, proc_params: list[str]) -> Any:
        """Extract input value from gold example."""
        if gold is None:
            raise ValueError("Metric requires module inputs to be available")
        if len(proc_params) == 1:
            return getattr(gold, proc_params[0])
        return tuple(getattr(gold, param) for param in proc_params)
    
    def _apply_prompt_prefix(self, span: Any, instructions: str | None):
        """Temporarily apply optimized instructions to the span's first line."""
        if not instructions or not span or not getattr(span, "template_lines", None):
            return None
        first_line = span.template_lines[0]
        original = first_line.optimized_prompt_prefix
        first_line.optimized_prompt_prefix = instructions
        return first_line, original
    
    def _restore_prompt_prefix(self, handle: Any) -> None:
        if not handle:
            return
        line, original = handle
        line.optimized_prompt_prefix = original
    
    def _apply_output_override(
        self,
        span: Any,
        predicted_outputs: dict[str, Any],
    ) -> list[tuple[TemplateLine, dict[str, Any] | None]] | None:
        """Apply deterministic output overrides to each template line in span."""
        if not span or not getattr(span, "template_lines", None):
            return None
        
        handles: list[tuple[TemplateLine, dict[str, Any] | None]] = []
        for template_line in span.template_lines:
            output_names = self._line_output_names(template_line)
            if not output_names:
                continue
            mapping = {
                name: predicted_outputs[name]
                for name in output_names
                if name in predicted_outputs
            }
            if mapping:
                handles.append((template_line, template_line.optimized_output_override))
                template_line.optimized_output_override = mapping
        
        return handles or None
    
    def _restore_output_override(
        self,
        handle: list[tuple[TemplateLine, dict[str, Any] | None]] | None,
    ) -> None:
        if not handle:
            return
        for template_line, original in handle:
            template_line.optimized_output_override = original
    
    def _line_output_names(self, template_line: TemplateLine) -> list[str]:
        """Return output names declared in a template line."""
        names: list[str] = []
        for segment in getattr(template_line, "segments", []):
            if isinstance(segment, OutputSegment):
                names.append(segment.name)
        return names

    def _apply_default_overrides(
        self,
        proc_def: Any,
        active_span: Any,
    ) -> list[tuple[TemplateLine, dict[str, Any] | None]] | None:
        """Apply benign overrides to all spans except the active one.
        Ensures other optimize spans do not trigger LLM calls during metric eval.
        """
        handles: list[tuple[TemplateLine, dict[str, Any] | None]] = []
        spans = getattr(proc_def, "optimize_spans", None)
        if not spans:
            return None
        for sp in spans:
            if sp is active_span:
                continue
            for tl in getattr(sp, "template_lines", []) or []:
                names = self._line_output_names(tl)
                if not names:
                    continue
                mapping = {name: "" for name in names}
                handles.append((tl, tl.optimized_output_override))
                tl.optimized_output_override = mapping
        return handles or None

    def _restore_sibling_overrides(
        self,
        handle: list[tuple[TemplateLine, dict[str, Any] | None]] | None,
    ) -> None:
        if not handle:
            return
        for tl, original in handle:
            tl.optimized_output_override = original
    
    def _stringify_feedback(self, feedback: Any) -> Optional[str]:
        """Convert arbitrary feedback into a printable string."""
        if feedback is None:
            return None
        if isinstance(feedback, str):
            return feedback
        if isinstance(feedback, dspy.Prediction):
            # Avoid recursive nesting; extract textual portion
            text = getattr(feedback, "feedback", None)
            return str(text) if text is not None else str(getattr(feedback, "score", ""))
        return str(feedback)
    
    def _lookup_cached_feedback(
        self,
        prediction: Any,
    ) -> Optional[tuple[float, Optional[str]]]:
        """Fetch cached (score, feedback) tuple for a prediction object."""
        if prediction is None:
            return None
        # Check attribute cache first for robustness
        attr_val = getattr(prediction, "__kedi_metric_cache", None)
        if attr_val is not None:
            return attr_val
        try:
            cached = self._prediction_cache.get(prediction)
        except TypeError:
            cached = None
        if cached is not None:
            return cached
        # Fallback by id()
        try:
            return self._prediction_id_cache.get(id(prediction))
        except Exception:
            return None
    
    def _store_cached_feedback(
        self,
        prediction: Any,
        score: float,
        feedback: Optional[str],
    ) -> None:
        """Persist (score, feedback) tuple for reuse."""
        if prediction is None:
            return
        data = (score, feedback)
        # Always attempt to set an attribute cache for direct retrieval
        try:
            setattr(prediction, "__kedi_metric_cache", data)
        except Exception:
            pass
        try:
            self._prediction_cache[prediction] = data
        except TypeError:
            # Not weakref-able; store by id()
            self._prediction_id_cache[id(prediction)] = data
    
    def _extract_current_instructions(self, module: dspy.Module) -> str | None:
        """Extract current optimized instructions from GEPA's module."""
        if not module:
            return None
        
        try:
            if hasattr(module, 'predictor'):
                predictor = module.predictor
            elif hasattr(module, 'named_predictors'):
                named = module.named_predictors()
                if named:
                    predictor = named[0][1] if isinstance(named[0], tuple) else named[0]
                else:
                    return None
            else:
                return None
            
            if not hasattr(predictor, 'signature'):
                return None
            
            sig = predictor.signature
            instructions = None
            
            if hasattr(sig, 'instructions'):
                instructions = sig.instructions
            elif hasattr(sig, '_instructions'):
                instructions = sig._instructions
            
            if instructions and instructions.strip():
                default_pattern = "Given the fields"
                if default_pattern not in instructions or len(instructions) > len(default_pattern) + 50:
                    return instructions.strip()
            
            return None
        except Exception:
            return None
