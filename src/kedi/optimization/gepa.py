"""GEPA optimizer for prompt optimization using DSPy."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable
from typing import Any

try:
    import dspy
    from dspy import LM
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    dspy = None  # type: ignore
    LM = None  # type: ignore

from ..lang.ast import OptimizeSpan
from .interfaces import OptimizerAdapter
from .utils import extract_input_fields, extract_output_fields, span_to_instruction


class GEPAOptimizer(OptimizerAdapter):
    """GEPA optimizer using DSPy's GEPA for reflective prompt evolution.
    
    GEPA uses LLM reflection to evolve prompts based on execution traces
    and feedback. This optimizer bridges Kedi procedures to DSPy modules.
    """

    def __init__(
        self,
        reflection_model: str | None = None,
        max_metric_calls: int = 150,
    ):
        """Initialize GEPA optimizer.
        
        Args:
            reflection_model: Model for reflection LM (defaults to main model)
            max_metric_calls: Maximum metric evaluations during optimization
        """
        if not DSPY_AVAILABLE:
            raise ImportError(
                "DSPy is required for GEPA optimizer. Install with: pip install dspy-ai"
            )
        
        self.reflection_model = reflection_model
        self.max_metric_calls = max_metric_calls

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
        """Optimize procedure spans using GEPA.
        
        Args:
            proc_name: Procedure name
            proc_params: List of procedure parameter names
            param_types: Mapping of parameter name -> Python type
            spans: List of optimize spans to optimize
            train_data: Iterable of (input, expected_output) tuples
            test_data: Optional test dataset
            metric: Function (input, expected_output) -> (score, feedback)
        
        Returns:
            Tuple of (optimized_prompts, train_score)
        """
        if not DSPY_AVAILABLE:
            raise ImportError("DSPy is required for GEPA optimizer")
        
        if not spans:
            return {}, 0.0
        
        train_list = list(train_data)
        if not train_list:
            raise ValueError(f"No training data provided for '{proc_name}'")
        
        # Validate types (basic validation)
        self._validate_types(proc_name, proc_params, param_types, train_list)
        
        # Ensure DSPy is configured (should be done in app.py, but check here too)
        if not hasattr(dspy.settings, 'lm') or dspy.settings.lm is None:
            raise RuntimeError(
                "DSPy LM not configured. GEPA requires DSPy to be configured with a model. "
                "This should be done automatically in run_optimization_cli when using GEPA optimizer."
            )
        
        # Disable DSPy cache during optimization
        original_cache = dspy.settings.cache if hasattr(dspy.settings, 'cache') else None
        dspy.settings.configure(cache=False)
        
        try:
            # Build DSPy module from spans
            dspy_module = self._build_dspy_module(spans, proc_params, param_types)
            
            # Format training data as DSPy Examples
            trainset = self._format_examples(train_list, proc_params)
            valset = None
            if test_data:
                valset = self._format_examples(list(test_data), proc_params)
            else:
                # Use training data as validation if no test data
                valset = trainset[:len(trainset)//4] if len(trainset) >= 4 else trainset
            
            # Extract output fields from first span (for metric wrapper)
            span_outputs = extract_output_fields(spans[0]) if spans else []
            
            # Wrap metric for GEPA
            gepa_metric = self._wrap_metric(metric, proc_params, span_outputs)
            
            # Configure reflection LM if provided
            # OpenAI reasoning models require temperature=1.0 and max_tokens >= 16000
            # gpt-4o supports max 16384, gpt-5 supports 32000
            reflection_lm = None
            if self.reflection_model:
                # Check if it's an OpenAI reasoning model
                if "gpt-5" in self.reflection_model.lower():
                    reflection_lm = LM(
                        model=self.reflection_model,
                        temperature=1.0,
                        max_tokens=32000  # gpt-5 supports 32000
                    )
                elif "gpt-4o" in self.reflection_model.lower():
                    reflection_lm = LM(
                        model=self.reflection_model,
                        temperature=1.0,
                        max_tokens=16384  # gpt-4o supports max 16384
                    )
                else:
                    reflection_lm = LM(model=self.reflection_model)
            
            # Run GEPA optimization
            optimizer = dspy.GEPA(
                metric=gepa_metric,
                max_metric_calls=self.max_metric_calls,
                reflection_lm=reflection_lm,
            )
            
            optimized_module = optimizer.compile(
                student=dspy_module,
                trainset=trainset,
                valset=valset,
            )
            
            # Extract optimized prompts
            optimized_prompts = self._extract_optimized_prompts(optimized_module, spans)
            
            # Debug: Verify we got optimized prompts (log if not found)
            import warnings
            for span in spans:
                if span.name not in optimized_prompts:
                    warnings.warn(f"No optimized prompt found for span '{span.name}'")
                elif optimized_prompts[span.name] == span_to_instruction(span):
                    warnings.warn(
                        f"Optimized prompt for '{span.name}' appears unchanged. "
                        f"GEPA may not have optimized this span."
                    )
            
            # Calculate training score
            train_score = self._calculate_train_score(
                optimized_module, trainset, gepa_metric
            )
            
            return optimized_prompts, train_score
            
        finally:
            # Restore cache setting
            if original_cache is not None:
                dspy.settings.configure(cache=original_cache)
    
    def _build_dspy_module(
        self,
        spans: list[OptimizeSpan],
        proc_params: list[str],
        param_types: dict[str, type],
    ) -> dspy.Module:
        """Build DSPy module from Kedi spans.
        
        For single span: creates a single Predict module.
        For multiple spans: creates a chain of Predict modules.
        """
        if len(spans) == 1:
            return self._build_single_span_module(spans[0], proc_params, param_types)
        else:
            return self._build_multi_span_module(spans, proc_params, param_types)
    
    def _build_single_span_module(
        self,
        span: OptimizeSpan,
        proc_params: list[str],
        param_types: dict[str, type],
    ) -> dspy.Module:
        """Build DSPy module for single span."""
        # Extract outputs from span
        span_outputs = extract_output_fields(span)
        
        if not span_outputs:
            raise ValueError(
                f"Optimize span '{span.name}' must have at least one output field"
            )
        
        # Signature inputs are the procedure parameters
        # Span inputs (<name>) typically reference proc_params or intermediate vars
        # For DSPy, we only need procedure params as inputs
        signature_inputs = proc_params
        
        # Create signature: "param1, param2 -> output1, output2"
        signature = self._create_signature(signature_inputs, span_outputs, param_types)
        
        # Get initial instruction (the prompt template)
        instruction = span_to_instruction(span)
        
        # Create module with Predict
        class SingleSpanModule(dspy.Module):
            def __init__(self, sig: dspy.Signature, inst: str):
                super().__init__()
                self.predictor = dspy.Predict(sig.with_instructions(inst))
            
            def forward(self, **kwargs):
                # Pass kwargs to predictor (DSPy will handle filtering)
                return self.predictor(**kwargs)
        
        return SingleSpanModule(signature, instruction)
    
    def _build_multi_span_module(
        self,
        spans: list[OptimizeSpan],
        proc_params: list[str],
        param_types: dict[str, type],
    ) -> dspy.Module:
        """Build DSPy module for multiple spans (chain)."""
        # For now, treat as independent modules
        # In future, could chain outputs between spans
        # For simplicity, optimize first span only (multi-span optimization is complex)
        warnings.warn(
            f"Multi-span optimization: optimizing first span '{spans[0].name}' only. "
            "Full multi-span support coming soon.",
            UserWarning
        )
        return self._build_single_span_module(spans[0], proc_params, param_types)
    
    def _create_signature(
        self,
        inputs: list[str],
        outputs: list[tuple[str, str | None]],
        param_types: dict[str, type],
    ) -> dspy.Signature:
        """Create DSPy Signature from inputs and outputs.
        
        Uses string-based signature which is simpler and works well with GEPA.
        """
        # Build signature string: "input1, input2 -> output1, output2"
        input_str = ", ".join(inputs)
        output_names = [out_name for out_name, _ in outputs]
        output_str = ", ".join(output_names)
        
        signature_str = f"{input_str} -> {output_str}"
        
        # Create Signature from string
        return dspy.Signature(signature_str)
    
    def _parse_type_annotation(self, type_str: str) -> type:
        """Parse Kedi type annotation to Python type."""
        type_str = type_str.strip().lower()
        
        type_map = {
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
        }
        
        # Handle list[T] syntax
        if type_str.startswith("list["):
            return list  # DSPy handles list types
        
        return type_map.get(type_str, str)
    
    def _format_examples(
        self,
        data: list[tuple[Any, Any]],
        proc_params: list[str],
    ) -> list[dspy.Example]:
        """Format training data as DSPy Examples."""
        examples = []
        
        for input_val, expected_output in data:
            # Create example dict
            example_dict = {}
            
            # Handle single vs multi-param
            if len(proc_params) == 1:
                example_dict[proc_params[0]] = input_val
            else:
                # Multi-param: input_val should be tuple
                if isinstance(input_val, tuple):
                    for param, val in zip(proc_params, input_val):
                        example_dict[param] = val
                else:
                    # Assume single value maps to first param
                    example_dict[proc_params[0]] = input_val
            
            # Add expected output (GEPA will use this in metric)
            example_dict["_expected"] = expected_output
            
            # Create Example and specify which fields are inputs
            # proc_params are the input fields, _expected is metadata
            example = dspy.Example(**example_dict).with_inputs(*proc_params)
            examples.append(example)
        
        return examples
    
    def _wrap_metric(
        self,
        kedi_metric: Callable[[Any, Any], tuple[float, str | None]],
        proc_params: list[str],
        output_fields: list[tuple[str, str | None]],
    ) -> Callable:
        """Wrap Kedi metric for GEPA.
        
        GEPA metric signature:
        metric(gold: Example, pred: Prediction, trace=None, pred_name=None, pred_trace=None)
        
        The Kedi metric expects to call the procedure, but GEPA already executed the module.
        We extract the prediction output and call the Kedi metric with it.
        """
        def gepa_metric(gold: dspy.Example, pred: dspy.Prediction, trace=None, pred_name=None, pred_trace=None):
            # Extract input from gold example
            if len(proc_params) == 1:
                input_val = getattr(gold, proc_params[0])
            else:
                input_val = tuple(getattr(gold, param) for param in proc_params)
            
            # Get expected output
            expected_output = getattr(gold, "_expected", None)
            
            # Extract prediction output from DSPy Prediction
            # Output field names are known from the signature
            output_field_names = [name for name, _ in output_fields]
            
            pred_output = None
            # Try direct attribute access first (most reliable for DSPy Predictions)
            if output_field_names:
                # For multiple outputs, use the LAST one (matches procedure return behavior)
                primary_output_name = output_field_names[-1]
                if hasattr(pred, primary_output_name):
                    pred_output = getattr(pred, primary_output_name)
                elif hasattr(pred, '__dict__') and primary_output_name in pred.__dict__:
                    pred_output = pred.__dict__[primary_output_name]
            
            # Fallback: try all output fields
            if pred_output is None and hasattr(pred, '__dict__'):
                # Try to get output fields by name
                output_values = {}
                for out_name in output_field_names:
                    if hasattr(pred, out_name):
                        output_values[out_name] = getattr(pred, out_name)
                    elif out_name in pred.__dict__:
                        output_values[out_name] = pred.__dict__[out_name]
                
                if output_values:
                    # For single output, use the value directly
                    if len(output_values) == 1:
                        pred_output = list(output_values.values())[0]
                    else:
                        # Multiple outputs - use the LAST output field as primary
                        primary_output_name = output_field_names[-1]
                        pred_output = output_values.get(primary_output_name)
                        if pred_output is None:
                            # Fallback: use tuple
                            pred_output = tuple(output_values.get(name) for name in output_field_names)
                else:
                    # Fallback: get first non-param attribute
                    attrs = {k: v for k, v in pred.__dict__.items() 
                            if not k.startswith('_') and k not in proc_params}
                    if attrs:
                        pred_output = list(attrs.values())[0] if len(attrs) == 1 else dict(attrs)
            
            # Last resort: convert to string
            if pred_output is None:
                pred_output = str(pred)
            
            # The Kedi metric expects to call the procedure, but we already have the prediction from DSPy.
            # We need to replicate the metric's comparison logic without calling the procedure.
            # The metric does: predicted_answer = solve_aime_problem(problem_text), then compares.
            # Since we have pred_output from DSPy, we just need to replicate the metric's comparison.
            try:
                if expected_output is not None:
                    # Replicate the metric's logic: extract int from pred_output if string
                    predicted_answer = pred_output
                    
                    # If it's already an int, use it directly
                    if isinstance(predicted_answer, int):
                        pass  # Use as-is
                    elif isinstance(predicted_answer, str):
                        # Extract first integer found (matches metric logic)
                        import re
                        match = re.search(r'\d+', predicted_answer)
                        if match:
                            predicted_answer = int(match.group())
                        else:
                            predicted_answer = None
                    elif isinstance(predicted_answer, float):
                        # Convert float to int if it's a whole number
                        if predicted_answer.is_integer():
                            predicted_answer = int(predicted_answer)
                        else:
                            predicted_answer = None
                    else:
                        # Try to convert to int
                        try:
                            predicted_answer = int(predicted_answer)
                        except (ValueError, TypeError):
                            predicted_answer = None
                    
                    # Compare (matches metric logic)
                    correct = predicted_answer == expected_output
                    
                    if correct:
                        score = 1.0
                        feedback = None
                    else:
                        score = 0.0
                        feedback = (
                            f"Incorrect answer for problem: '{str(input_val)[:50]}...'. "
                            f"Expected {expected_output}, got {predicted_answer}. "
                            f"Review the reasoning process and verify all computational steps are correct."
                        )
                else:
                    score = 0.0
                    feedback = "No expected output provided for comparison"
                
            except Exception as e:
                # Fallback on error - still try to extract and compare
                try:
                    predicted_answer = pred_output
                    if isinstance(predicted_answer, str):
                        import re
                        match = re.search(r'\d+', str(predicted_answer))
                        if match:
                            predicted_answer = int(match.group())
                        else:
                            predicted_answer = None
                    
                    if expected_output is not None and predicted_answer == expected_output:
                        score = 1.0
                        feedback = None
                    else:
                        score = 0.0
                        feedback = f"Metric evaluation error: {e}"
                except Exception:
                    score = 0.0
                    feedback = f"Metric evaluation error: {e}"
            
            # Return with feedback if available (GEPA benefits from textual feedback)
            if feedback:
                return dspy.Prediction(score=score, feedback=feedback)
            return score
        
        return gepa_metric
    
    def _extract_optimized_prompts(
        self,
        optimized_module: dspy.Module,
        spans: list[OptimizeSpan],
    ) -> dict[str, str]:
        """Extract optimized instructions from DSPy module."""
        optimized_prompts = {}
        
        # Get named predictors - GEPA may structure the module differently
        try:
            predictors = optimized_module.named_predictors()
            # Ensure predictors is a dict-like structure
            if not isinstance(predictors, dict):
                # If it's not a dict, try to convert or create empty dict
                if hasattr(predictors, 'items'):
                    predictors = dict(predictors.items())
                else:
                    predictors = {}
        except Exception:
            predictors = {}
        
        # Debug: print module structure
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Check if module has a direct predictor attribute
            if hasattr(optimized_module, 'predictor'):
                if not isinstance(predictors, dict):
                    predictors = {}
                predictors['_main'] = optimized_module.predictor
        
        # If no named predictors found, try to find any Predict modules
        if not predictors:
            predictors = {}
            # Check all attributes for Predict instances
            for attr_name in dir(optimized_module):
                if attr_name.startswith('_'):
                    continue
                try:
                    attr = getattr(optimized_module, attr_name)
                    if isinstance(attr, dspy.Predict):
                        predictors[attr_name] = attr
                except Exception:
                    continue
        
        # For each span, try to find corresponding predictor
        for i, span in enumerate(spans):
            predictor = None
            
            # Ensure predictors is a dict before accessing
            if not isinstance(predictors, dict):
                predictors = {}
            
            # Strategy 1: Try to find by span name
            if span.name in predictors:
                predictor = predictors[span.name]
            # Strategy 2: Try by index if single predictor
            elif len(predictors) == 1:
                predictor = list(predictors.values())[0]
            # Strategy 3: Try by index position
            elif i < len(predictors):
                predictor = list(predictors.values())[i]
            # Strategy 4: Try common attribute names
            elif hasattr(optimized_module, 'predictor'):
                predictor = optimized_module.predictor
            elif hasattr(optimized_module, 'student') and hasattr(optimized_module.student, 'predictor'):
                predictor = optimized_module.student.predictor
            
            if predictor:
                # Extract instructions - GEPA stores them in signature.instructions
                if hasattr(predictor, 'signature'):
                    sig = predictor.signature
                    # Instructions can be stored directly or in a field
                    instructions = None
                    if hasattr(sig, 'instructions'):
                        instructions = sig.instructions
                    elif hasattr(sig, '_instructions'):
                        instructions = sig._instructions
                    
                    if instructions and instructions.strip():
                        # Check if it's actually different from the default signature template
                        # DSPy adds default instructions like "Given the fields X, produce the fields Y"
                        # We want to skip those and only use custom instructions
                        default_pattern = f"Given the fields"
                        if default_pattern not in instructions or len(instructions) > len(default_pattern) + 50:
                            optimized_prompts[span.name] = instructions
                        else:
                            # Only default instructions found, use original
                            optimized_prompts[span.name] = span_to_instruction(span)
                    else:
                        # No instructions found, use original
                        optimized_prompts[span.name] = span_to_instruction(span)
                else:
                    # No signature, use original
                    optimized_prompts[span.name] = span_to_instruction(span)
            else:
                # No predictor found, use original
                optimized_prompts[span.name] = span_to_instruction(span)
        
        return optimized_prompts
    
    def _calculate_train_score(
        self,
        module: dspy.Module,
        trainset: list[dspy.Example],
        metric: Callable,
    ) -> float:
        """Calculate average training score."""
        scores = []
        for example in trainset[:min(10, len(trainset))]:  # Sample for speed
            try:
                # Create prediction using example.inputs() which returns dict of input fields
                inputs = example.inputs() if hasattr(example, 'inputs') and callable(example.inputs) else {}
                if not inputs:
                    # Fallback: extract inputs manually
                    if hasattr(example, '__dict__'):
                        inputs = {k: v for k, v in example.__dict__.items() if not k.startswith('_')}
                    else:
                        continue  # Skip if we can't get inputs
                pred = module(**inputs)
                score_result = metric(example, pred)
                if hasattr(score_result, 'score'):
                    scores.append(float(score_result.score))
                else:
                    scores.append(float(score_result))
            except Exception:
                scores.append(0.0)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def _validate_types(
        self,
        proc_name: str,
        proc_params: list[str],
        param_types: dict[str, type],
        train_data: list[tuple[Any, Any]],
    ) -> None:
        """Basic type validation (simplified version)."""
        # Skip if no param types specified
        if not param_types:
            return
        
        for idx, example in enumerate(train_data):
            if not isinstance(example, tuple) or len(example) != 2:
                continue  # Let it fail later
            
            input_val, _ = example
            
            if len(proc_params) == 1:
                param_name = proc_params[0]
                expected_type = param_types.get(param_name)
                if expected_type and not isinstance(input_val, expected_type):
                    warnings.warn(
                        f"Training example {idx}: expected {expected_type.__name__}, "
                        f"got {type(input_val).__name__}",
                        UserWarning
                    )
            else:
                if isinstance(input_val, tuple) and len(input_val) == len(proc_params):
                    for param_name, arg_val in zip(proc_params, input_val):
                        expected_type = param_types.get(param_name)
                        if expected_type and not isinstance(arg_val, expected_type):
                            warnings.warn(
                                f"Training example {idx}, param {param_name}: "
                                f"expected {expected_type.__name__}, got {type(arg_val).__name__}",
                                UserWarning
                            )

