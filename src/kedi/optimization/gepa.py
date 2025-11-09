"""GEPA optimizer for prompt optimization using DSPy."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable
from typing import Any

try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    dspy = None  # type: ignore[assignment]

from ..lang.ast import OptimizeSpan
from ..lang.compiler.environment import KediRuntime
from .config import OptimizerConfig
from .interfaces import OptimizerAdapter
from .services.dspy_config import DSPyConfigurationService
from .services.dspy_example_formatter import DSPyExampleFormatter
from .services.dspy_module_builder import DSPyModuleBuilder
from .services.dspy_prompt_extractor import DSPyPromptExtractor
from .services.dspy_score_calculator import DSPyScoreCalculator
from .services.gepa_metric_wrapper import GEPAMetricWrapper
from .utils import span_to_instruction


class GEPAOptimizer(OptimizerAdapter):
    """GEPA optimizer using DSPy's GEPA for reflective prompt evolution.
    
    GEPA uses LLM reflection to evolve prompts based on execution traces
    and feedback. This optimizer bridges Kedi procedures to DSPy modules.
    
    Uses OptimizerConfig and service layer for all operations.
    """

    def __init__(
        self,
        config: OptimizerConfig,
        runtime: KediRuntime | None = None,
    ):
        """Initialize GEPA optimizer.
        
        Args:
            config: Optimizer configuration with model and token settings
            runtime: Optional Kedi runtime for structured type handling
        """
        if not DSPY_AVAILABLE:
            raise ImportError(
                "DSPy is required for GEPA optimizer. Install with: pip install dspy-ai"
            )
        
        self.config = config
        self.runtime = runtime
        
        # Initialize services
        self.dspy_config_service = DSPyConfigurationService(config)
        self.module_builder = DSPyModuleBuilder(runtime=runtime)
        self.example_formatter = DSPyExampleFormatter()
        self.metric_wrapper = GEPAMetricWrapper()
        self.prompt_extractor = DSPyPromptExtractor()
        self.score_calculator = DSPyScoreCalculator()
    
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
        
        # Validate types
        self._validate_types(proc_name, proc_params, param_types, train_list)
        
        # Verify DSPy is configured
        self.dspy_config_service.verify_configured()
        test_list = list(test_data) if test_data is not None else None
        
        # Disable reflection in GEPA for prefix-only integration (no DSPy traces available)
        reflection_lm = None
        
        optimized_prompts: dict[str, str] = {}
        span_scores: list[float] = []
        
        # Preserve original optimized prefixes so we can restore after optimization
        original_prefixes: dict[str, str | None] = {}
        for span in spans:
            if span.template_lines:
                original_prefixes[span.name] = span.template_lines[0].optimized_prompt_prefix
            else:
                original_prefixes[span.name] = None
        
        try:
            for span in spans:
                # Prepare training/validation examples (inputs not required by prefix-only module)
                train_examples = self.example_formatter.format_examples(train_list, proc_params)
                if test_list:
                    val_examples = self.example_formatter.format_examples(test_list, proc_params)
                elif len(train_examples) >= 4:
                    cutoff = max(1, len(train_examples) // 4)
                    val_examples = train_examples[:cutoff]
                else:
                    val_examples = train_examples
                
                dspy_module = self.module_builder.build_module(span, proc_params, param_types)
                
                # Metric used during GEPA optimization - points at the module being optimized
                # Obtain proc_def for sibling span overrides during metric evaluation
                proc_def_for_wrapper = self._get_proc_def(proc_name)
                gepa_metric = self.metric_wrapper.wrap_metric(
                    metric,
                    proc_params,
                    span=span,
                    module_ref=dspy_module,
                    proc_def=proc_def_for_wrapper,
                )
                
                optimizer = dspy.GEPA(
                    metric=gepa_metric,
                    max_metric_calls=self.config.max_metric_calls,
                    reflection_lm=reflection_lm,
                )
                
                optimized_module = optimizer.compile(
                    student=dspy_module,
                    trainset=train_examples,
                    valset=val_examples,
                )
                
                # Extract optimized prompt for this span
                extracted = self.prompt_extractor.extract_prompts(optimized_module, [span])
                optimized_prompt = extracted.get(span.name)
                if optimized_prompt:
                    optimized_prompts[span.name] = optimized_prompt
                    if span.template_lines:
                        span.template_lines[0].optimized_prompt_prefix = optimized_prompt
                else:
                    warnings.warn(f"No optimized prompt found for span '{span.name}'", UserWarning)
                    continue
                
                if optimized_prompt == span_to_instruction(span):
                    warnings.warn(
                        f"Optimized prompt for '{span.name}' appears unchanged. "
                        "GEPA may not have optimized this span.",
                        UserWarning,
                    )
                
                # Re-wrap metric with the optimized module for scoring
                scoring_metric = self.metric_wrapper.wrap_metric(
                    metric,
                    proc_params,
                    span=span,
                    module_ref=optimized_module,
                    proc_def=proc_def_for_wrapper,
                )
                
                span_score = self.score_calculator.calculate_score(
                    optimized_module,
                    train_examples,
                    scoring_metric,
                )
                span_scores.append(span_score)
        finally:
            # Restore original prefixes so caller gets pristine AST
            for span in spans:
                if not span.template_lines:
                    continue
                original_prefix = original_prefixes.get(span.name)
                span.template_lines[0].optimized_prompt_prefix = original_prefix
        
        overall_score = sum(span_scores) / len(span_scores) if span_scores else 0.0
        return optimized_prompts, overall_score

    def _get_proc_def(self, proc_name: str):
        """Best-effort retrieval of ProcedureDef from runtime for the given name."""
        if not self.runtime:
            return None
        try:
            fn = self.runtime._procedures.get(proc_name)
            if not fn:
                return None
            import inspect
            if hasattr(fn, '__code__'):
                closure_vars = inspect.getclosurevars(fn)
                proc = closure_vars.nonlocals.get('proc') or closure_vars.globals.get('proc')
                return proc
        except Exception:
            return None
        return None

    def _validate_types(
        self,
        proc_name: str,
        proc_params: list[str],
        param_types: dict[str, type],
        train_data: list[tuple[Any, Any]],
    ) -> None:
        """Basic type validation."""
        if not param_types:
            return
        
        for idx, example in enumerate(train_data):
            if not isinstance(example, tuple) or len(example) != 2:
                continue
            
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
