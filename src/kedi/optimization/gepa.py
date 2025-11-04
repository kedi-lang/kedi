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
from .utils import extract_output_fields, span_to_instruction


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
        
        # Build DSPy module
        # For now, optimize first span only (multi-span optimization is complex)
        if len(spans) > 1:
            warnings.warn(
                f"Multi-span optimization: optimizing first span '{spans[0].name}' only. "
                "Full multi-span support coming soon.",
                UserWarning
            )
        
        primary_span = spans[0]
        dspy_module = self.module_builder.build_module(
            primary_span, proc_params, param_types
        )
        
        # Format training and validation data
        trainset = self.example_formatter.format_examples(train_list, proc_params)
        valset = (
            self.example_formatter.format_examples(list(test_data), proc_params)
            if test_data
            else trainset[:len(trainset)//4] if len(trainset) >= 4 else trainset
        )
        
        # Extract output fields for metric wrapper
        span_outputs = extract_output_fields(primary_span)
        
        # Wrap metric for GEPA
        gepa_metric = self.metric_wrapper.wrap_metric(
            metric, proc_params, span_outputs, proc_name=proc_name, runtime=self.runtime
        )
        
        # Configure reflection LM
        reflection_lm = None
        if self.config.reflection_model:
            reflection_lm = self.dspy_config_service.configure_reflection_lm()
    
        # Run GEPA optimization
        optimizer = dspy.GEPA(
            metric=gepa_metric,
            max_metric_calls=self.config.max_metric_calls,
            reflection_lm=reflection_lm,
        )
        
        optimized_module = optimizer.compile(
            student=dspy_module,
            trainset=trainset,
            valset=valset,
        )
        
        # Extract optimized prompts
        optimized_prompts = self.prompt_extractor.extract_prompts(
            optimized_module, spans
        )
        
        # Warn if prompts unchanged
        for span in spans:
            if span.name not in optimized_prompts:
                warnings.warn(f"No optimized prompt found for span '{span.name}'")
            elif optimized_prompts[span.name] == span_to_instruction(span):
                warnings.warn(
                    f"Optimized prompt for '{span.name}' appears unchanged. "
                    f"GEPA may not have optimized this span."
                )
        
        # Calculate training score
        train_score = self.score_calculator.calculate_score(
            optimized_module, trainset, gepa_metric
        )
        
        return optimized_prompts, train_score
    
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
