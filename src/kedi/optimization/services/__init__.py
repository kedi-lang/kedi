"""Service layer for optimizer operations.

Keep this package initializer lightweight. Normal `kedi file.kedi` execution
imports `program_loader`, so eager DSPy service imports here would initialize
DSPy/LiteLLM even when optimization is not being used.
"""

from .data_loader import OptimizationDataLoader
from .metric_factory import MetricFactory
from .program_loader import load_program_with_optimized_prompts
from .prompt_loader import OptimizedPromptLoader
from .result_writer import OptimizationResultWriter
from .type_resolver import TypeResolver

__all__ = [
    "OptimizationDataLoader",
    "MetricFactory",
    "TypeResolver",
    "OptimizationResultWriter",
    "OptimizedPromptLoader",
    "load_program_with_optimized_prompts",
]
