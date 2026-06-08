"""GEPA optimizer for prompt optimization using DSPy."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, cast

from ..lang.ast import OptimizeSpan
from ..lang.compiler.environment import KediRuntime
from ..reporter import CliReporter, Reporter
from ..resolved_types import ResolvedKediType
from .config import OptimizerConfig
from .gepa_reflection import build_reflection_lm
from .gepa_session import GEPASession
from .interfaces import OptimizerAdapter
from .services.dspy_config import DSPyConfigurationService
from .services.dspy_example_formatter import DSPyExampleFormatter
from .services.dspy_module_builder import DSPyModuleBuilder
from .services.dspy_prompt_extractor import DSPyPromptExtractor
from .services.dspy_score_calculator import DSPyScoreCalculator
from .services.gepa_metric_wrapper import GEPAMetricWrapper


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
        source_path: Path | None = None,
        reporter: Reporter | None = None,
    ):
        """Initialize GEPA optimizer.

        Args:
            config: Optimizer configuration with model and token settings
            runtime: Optional Kedi runtime for structured type handling
            source_path: Path to source .kedi file for saving optimized prompts
        """
        self.config = config
        self.runtime = runtime
        self.source_path = source_path
        self._reporter = reporter or CliReporter()

        self.dspy_config_service = DSPyConfigurationService(config, reporter=self._reporter)
        self.module_builder = DSPyModuleBuilder(runtime=cast(KediRuntime, runtime))
        self.example_formatter = DSPyExampleFormatter(
            cast(KediRuntime, runtime),
            reporter=self._reporter,
        )
        self.metric_wrapper = GEPAMetricWrapper(reporter=self._reporter)
        self.prompt_extractor = DSPyPromptExtractor()
        self.score_calculator = DSPyScoreCalculator()

        if source_path:
            from .services import OptimizationResultWriter

            self.result_writer = OptimizationResultWriter(source_path)
        else:
            self.result_writer = None

    def _gepa_log_dir(self, proc_name: str, span_name: str) -> Path | None:
        """Per-span directory for DSPy GEPA ``log_dir`` (``gepa_state.bin`` resume)."""
        if not self.source_path:
            return None
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in f"{proc_name}__{span_name}")
        return self.source_path.parent / f"{self.source_path.name}.gepa" / safe

    def optimize_procedure(
        self,
        proc_name: str,
        proc_params: list[str],
        param_types: dict[str, ResolvedKediType],
        spans: list[OptimizeSpan],
        train_data: Iterable[tuple[Any, Any]],
        test_data: Iterable[tuple[Any, Any]] | None,
        metric: Callable[[Any, Any], tuple[float, str | None]],
    ) -> tuple[dict[str, str], float]:
        """Optimize procedure spans using GEPA."""
        if not spans:
            return {}, 0.0

        train_list = list(train_data)
        if not train_list:
            raise ValueError(f"No training data provided for '{proc_name}'")

        self._validate_types(proc_name, proc_params, param_types, train_list)
        self.dspy_config_service.verify_configured()

        test_list = list(test_data) if test_data is not None else None
        reflection_lm = build_reflection_lm(self.config, self._reporter)

        optimized_prompts: dict[str, str] = {}
        span_scores: list[float] = []

        original_prefixes: dict[str, str | None] = {}
        for span in spans:
            if span.template_lines:
                original_prefixes[span.name] = span.template_lines[0].optimized_prompt_prefix
            else:
                original_prefixes[span.name] = None

        session = GEPASession(
            proc_name=proc_name,
            proc_params=proc_params,
            param_types=param_types,
            train_list=train_list,
            test_list=test_list,
            metric=metric,
            config=self.config,
            reflection_lm=reflection_lm,
            module_builder=self.module_builder,
            example_formatter=self.example_formatter,
            metric_wrapper=self.metric_wrapper,
            prompt_extractor=self.prompt_extractor,
            score_calculator=self.score_calculator,
            reporter=self._reporter,
            gepa_log_dir_for=self._gepa_log_dir,
            save_prompts=self._save_prompts_immediately,
        )

        try:
            for span_idx, span in enumerate(spans, 1):
                result = session.optimize_single_span(
                    span,
                    span_idx=span_idx,
                    total_spans=len(spans),
                    optimized_prompts=optimized_prompts,
                )
                if result.prompt is not None:
                    span_scores.append(result.score)
        finally:
            for span in spans:
                if not span.template_lines:
                    continue
                original_prefix = original_prefixes.get(span.name)
                span.template_lines[0].optimized_prompt_prefix = original_prefix

        overall_score = sum(span_scores) / len(span_scores) if span_scores else 0.0
        return optimized_prompts, overall_score

    def _save_prompts_immediately(
        self,
        proc_name: str,
        optimized_prompts: dict[str, str],
    ) -> None:
        """Save optimized prompts immediately after extraction."""
        if not self.result_writer:
            return

        try:
            assert self.source_path is not None
            optimized_path = self.source_path.with_suffix(
                self.source_path.suffix + ".optimized.json"
            )
            existing_prompts: dict[str, dict[str, str]] = {}

            if optimized_path.exists():
                try:
                    with open(optimized_path, encoding="utf-8") as f:
                        existing_prompts = json.load(f)
                except (json.JSONDecodeError, OSError):
                    existing_prompts = {}

            existing_prompts[proc_name] = optimized_prompts
            self.result_writer.write_optimized_prompts(existing_prompts)
            self._reporter.emit(f"   💾 Saved optimized prompts to {optimized_path.name}")
        except Exception as e:
            self._reporter.emit(f"   ⚠️  Failed to save prompts: {e}")

    def _validate_types(
        self,
        proc_name: str,
        proc_params: list[str],
        param_types: dict[str, ResolvedKediType],
        train_data: list[tuple[Any, Any]],
    ) -> None:
        """Basic type validation."""
        if not param_types:
            return

        for example in train_data:
            if not isinstance(example, tuple) or len(example) != 2:
                continue
            # Type checking is deferred to the metric / procedure runtime.
