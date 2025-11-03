"""Service layer for optimizer operations.

Following SOLID principles, services handle specific responsibilities:
- Data loading and formatting
- Metric creation
- Type resolution
- DSPy configuration
- Result writing
- Structured type handling
"""

from .data_loader import OptimizationDataLoader
from .dspy_config import DSPyConfigurationService
from .dspy_example_formatter import DSPyExampleFormatter
from .dspy_module_builder import DSPyModuleBuilder
from .dspy_prompt_extractor import DSPyPromptExtractor
from .dspy_score_calculator import DSPyScoreCalculator
from .gepa_metric_wrapper import GEPAMetricWrapper
from .metric_factory import MetricFactory
from .program_loader import load_program_with_optimized_prompts
from .prompt_loader import OptimizedPromptLoader
from .result_writer import OptimizationResultWriter
from .type_handler import StructuredTypeHandler
from .type_resolver import TypeResolver

__all__ = [
    "OptimizationDataLoader",
    "MetricFactory",
    "TypeResolver",
    "DSPyConfigurationService",
    "OptimizationResultWriter",
    "StructuredTypeHandler",
    "DSPyModuleBuilder",
    "DSPyExampleFormatter",
    "GEPAMetricWrapper",
    "DSPyPromptExtractor",
    "DSPyScoreCalculator",
    "OptimizedPromptLoader",
    "load_program_with_optimized_prompts",
]

