"""Service for wrapping Kedi metrics for GEPA optimization."""

from __future__ import annotations

import threading
import time
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
from ..utils import normalize_optimized_prompt_prefix


class GEPAMetricWrapper:
    """Service for wrapping Kedi metrics to work with GEPA.

    GEPA metrics receive (gold: Example, pred: Prediction, ...) but
    Kedi metrics expect (input, expected_output). This service bridges
    the gap by extracting inputs from DSPy examples and calling the Kedi
    metric directly. The metric calls the procedure normally, which uses
    the optimized prompt injected into the span's template.

    Note on concurrency: applying overrides/prefix mutates ``TemplateLine``
    fields on the shared span AST. DSPy's evaluator runs metric callbacks
    on a ``ThreadPoolExecutor`` (default ``dspy.settings.num_threads == 8``),
    so two concurrent callbacks can interleave apply/restore and leak the
    wrong override into the next callback (or the post-compile scoring
    pass). We serialize the entire apply→evaluate→restore region with
    ``_span_lock`` to keep the mutation atomic; the lock is fast because
    the heavy lifting (LM calls) already happened before ``gepa_metric``
    runs.
    """

    def __init__(self) -> None:
        self._prediction_cache: "weakref.WeakKeyDictionary[Any, tuple[float, Optional[str]]]" = weakref.WeakKeyDictionary()
        # Fallback cache keyed by id() for objects that are not weakref-able
        self._prediction_id_cache: dict[int, tuple[float, Optional[str]]] = {}
        # Serializes apply/restore of span.template_lines mutations across
        # concurrent metric callbacks. See class docstring.
        self._span_lock = threading.Lock()

    def wrap_metric(
        self,
        kedi_metric: Callable[[Any, Any], tuple[float, str | None]],
        proc_params: list[str],
        span_or_output_fields: Any = None,  # Can be span object or list of output fields (backward compat)
        module_ref: Any = None,
        proc_def: Any = None,
    ) -> Callable:
        """Wrap Kedi metric for GEPA.

        Args:
            kedi_metric: Kedi metric function (input, expected) -> (score, feedback)
            proc_params: Procedure parameter names
            span_or_output_fields: OptimizeSpan object OR list of (name, type) tuples for backward compatibility
            module_ref: Reference to GEPA's module (for extracting current instructions)
            proc_def: Procedure definition for sibling span overrides

        Returns:
            GEPA-compatible metric function
        """
        # Determine if we have a span or output_fields list
        span = None
        output_fields = None
        if span_or_output_fields is not None:
            if isinstance(span_or_output_fields, list):
                # Backward compatibility: list of output fields
                output_fields = span_or_output_fields
            else:
                # New behavior: span object
                span = span_or_output_fields

        progress = {
            "count": 0,
            "total_score": 0.0,
            "best_score": float("-inf"),
            "start_time": time.monotonic(),
        }

        def gepa_metric(
            gold: dspy.Example,
            pred: dspy.Prediction,
            trace=None,
            pred_name=None,
            pred_trace=None,
            **context: Any,
        ):
            """GEPA metric function."""
            progress["count"] += 1
            cnt = progress["count"]
            # Not compared to max_metric_calls: GEPA's budget counts internal eval units;
            # DSPy may invoke this callback more often than that.
            print(f"   [metric callback {cnt}] starting...", flush=True)

            try:
                example_obj = context.get("module_inputs", gold)
                prediction_obj = context.get("module_outputs", pred)

                cached = self._lookup_cached_feedback(
                    prediction_obj
                ) or self._lookup_cached_feedback(pred)
                if cached is not None:
                    cached_score, cached_feedback = cached
                    return ScoreWithFeedback(
                        score=cached_score, feedback=cached_feedback or ""
                    )

                input_val = self._extract_input(example_obj, proc_params)
                expected_output = getattr(example_obj, "_expected", None)

                prompt_handle = None
                output_override_handle = None
                # The apply/restore region must be atomic with the kedi_metric
                # call (which reads the just-applied prefix/override); otherwise
                # parallel DSPy evaluator threads race and leak overrides past
                # their own restore. See class docstring.
                with self._span_lock:
                    try:
                        # Extract outputs from DSPy prediction and inject them into the span
                        # This ensures the Kedi procedure uses the DSPy-generated outputs
                        # Only apply output override if we have a span (not just output_fields for backward compat)
                        if span and prediction_obj:
                            output_override = self._extract_prediction_outputs(
                                prediction_obj, span, output_fields
                            )
                            if output_override:
                                output_override_handle = self._apply_output_override(
                                    span, output_override
                                )

                        if module_ref:
                            optimized_prompt = self._extract_current_instructions(
                                module_ref
                            )
                            prompt_handle = self._apply_prompt_prefix(
                                span, optimized_prompt
                            )

                        score, feedback = kedi_metric(input_val, expected_output)
                        score_float = float(score)
                        feedback_text = self._stringify_feedback(feedback)

                        progress["total_score"] += score_float
                        cnt = progress["count"]
                        avg = progress["total_score"] / cnt
                        elapsed = time.monotonic() - progress["start_time"]
                        is_new_best = score_float > progress["best_score"]
                        if is_new_best:
                            progress["best_score"] = score_float
                        print(
                            f"   [metric callback {cnt}] score={score_float:.3f}  "
                            f"avg={avg:.3f}  best={progress['best_score']:.3f}  "
                            f"({elapsed:.0f}s)",
                            flush=True,
                        )

                        payload = ScoreWithFeedback(
                            score=score_float, feedback=feedback_text or ""
                        )
                        self._store_cached_feedback(
                            prediction_obj, score_float, feedback_text
                        )
                        self._store_cached_feedback(pred, score_float, feedback_text)
                        return payload
                    finally:
                        self._restore_prompt_prefix(prompt_handle)
                        self._restore_output_override(output_override_handle)
            except Exception:
                import traceback

                error_msg = traceback.format_exc()

                progress["total_score"] += 0.0
                cnt = progress["count"]
                avg = progress["total_score"] / cnt
                elapsed = time.monotonic() - progress["start_time"]
                print(
                    f"   [metric callback {cnt}] score=0.000 (error)  "
                    f"avg={avg:.3f}  best={progress['best_score']:.3f}  "
                    f"({elapsed:.0f}s)",
                    flush=True,
                )

                payload = ScoreWithFeedback(
                    score=0.0, feedback=f"GEPA metric error: {error_msg}"
                )
                self._store_cached_feedback(pred, 0.0, payload.feedback)
                return payload

        gepa_metric._progress = progress
        return gepa_metric

    def _extract_input(self, gold: dspy.Example, proc_params: list[str]) -> Any:
        """Extract input value from gold example.

        Prefers _proc_inputs if available (original procedure parameters),
        otherwise falls back to extracting from proc_params fields.
        """
        if gold is None:
            raise ValueError("Metric requires module inputs to be available")

        # Check if we have stored procedure inputs (from span-aware formatting)
        if hasattr(gold, "_proc_inputs"):
            proc_inputs = gold._proc_inputs
            if len(proc_params) == 1:
                return proc_inputs[proc_params[0]]
            return tuple(proc_inputs[param] for param in proc_params)

        # Fallback: extract from proc_params fields directly
        if len(proc_params) == 1:
            return getattr(gold, proc_params[0])
        return tuple(getattr(gold, param) for param in proc_params)

    def _apply_prompt_prefix(self, span: Any, instructions: str | None):
        """Temporarily apply optimized instructions to the span's first line."""
        if (
            instructions is None
            or not span
            or not getattr(span, "template_lines", None)
        ):
            return None
        first_line = span.template_lines[0]
        original = first_line.optimized_prompt_prefix
        first_line.optimized_prompt_prefix = (
            normalize_optimized_prompt_prefix(span, instructions) or None
        )
        return first_line, original

    def _restore_prompt_prefix(self, handle: Any) -> None:
        if not handle:
            return
        line, original = handle
        line.optimized_prompt_prefix = original

    def _extract_prediction_outputs(
        self,
        prediction: Any,
        span: Any,
        output_fields_override: list[tuple[str, str]] | None = None,
    ) -> dict[str, Any] | None:
        """Extract output values from DSPy prediction that match the span's outputs."""
        if not prediction:
            return None

        # Get output field names from span or override
        if output_fields_override:
            output_fields = output_fields_override
        elif span:
            from ..utils import extract_output_fields

            output_fields = extract_output_fields(span)
        else:
            return None

        if not output_fields:
            return None

        # Extract values from prediction
        outputs = {}
        for field_name, _ in output_fields:
            if hasattr(prediction, field_name):
                outputs[field_name] = getattr(prediction, field_name)

        return outputs if outputs else None

    def _apply_output_override(self, span: Any, outputs: dict[str, Any]) -> Any:
        """Apply output override to ALL template lines in the span."""
        if not span or not outputs:
            return None
        if not hasattr(span, "template_lines") or not span.template_lines:
            return None

        # Store original values for all lines and set override on all of them
        # This is necessary because a span can have multiple template lines
        originals = []
        for line in span.template_lines:
            originals.append((line, line.optimized_output_override))
            line.optimized_output_override = outputs

        return originals

    def _restore_output_override(self, handle: Any) -> None:
        """Restore original output override for all lines."""
        if not handle:
            return
        # Handle is now a list of (line, original) tuples
        for line, original in handle:
            line.optimized_output_override = original

    # Removed prediction-based output override; metric evaluates Kedi directly

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
            return (
                str(text) if text is not None else str(getattr(feedback, "score", ""))
            )
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
            if hasattr(module, "predictor"):
                predictor = module.predictor
            elif hasattr(module, "named_predictors"):
                named = module.named_predictors()
                if named:
                    predictor = named[0][1] if isinstance(named[0], tuple) else named[0]
                else:
                    return None
            else:
                return None

            if not hasattr(predictor, "signature"):
                return None

            sig = predictor.signature
            instructions = None

            if hasattr(sig, "instructions"):
                instructions = sig.instructions
            elif hasattr(sig, "_instructions"):
                instructions = sig._instructions

            if instructions and instructions.strip():
                default_pattern = "Given the fields"
                if (
                    default_pattern not in instructions
                    or len(instructions) > len(default_pattern) + 50
                ):
                    return instructions.strip()

            return None
        except Exception:
            return None
