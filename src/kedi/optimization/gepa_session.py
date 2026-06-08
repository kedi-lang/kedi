from __future__ import annotations

import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import dspy

from ..lang.ast import OptimizeSpan
from ..resolved_types import ResolvedKediType
from ..reporter import Reporter
from .config import OptimizerConfig
from .services.dspy_example_formatter import DSPyExampleFormatter
from .services.dspy_module_builder import DSPyModuleBuilder
from .services.dspy_prompt_extractor import DSPyPromptExtractor
from .services.dspy_score_calculator import DSPyScoreCalculator
from .services.gepa_metric_wrapper import GEPAMetricWrapper
from .utils import normalize_optimized_prompt_prefix, span_to_instruction

_GEPA_STATE_NAME = "gepa_state.bin"


@dataclass(frozen=True, kw_only=True)
class GEPASpanResult:
    prompt: str | None
    score: float


class GEPASession:
    """Runs GEPA optimization for a single optimize span within a procedure."""

    def __init__(
        self,
        *,
        proc_name: str,
        proc_params: list[str],
        param_types: dict[str, ResolvedKediType],
        train_list: list[tuple[Any, Any]],
        test_list: list[tuple[Any, Any]] | None,
        metric: Callable[[Any, Any], tuple[float, str | None]],
        config: OptimizerConfig,
        reflection_lm: Any | None,
        module_builder: DSPyModuleBuilder,
        example_formatter: DSPyExampleFormatter,
        metric_wrapper: GEPAMetricWrapper,
        prompt_extractor: DSPyPromptExtractor,
        score_calculator: DSPyScoreCalculator,
        reporter: Reporter,
        gepa_log_dir_for: Callable[[str, str], Path | None],
        save_prompts: Callable[[str, dict[str, str]], None],
    ) -> None:
        self._proc_name = proc_name
        self._proc_params = proc_params
        self._param_types = param_types
        self._train_list = train_list
        self._test_list = test_list
        self._metric = metric
        self._config = config
        self._reflection_lm = reflection_lm
        self._module_builder = module_builder
        self._example_formatter = example_formatter
        self._metric_wrapper = metric_wrapper
        self._prompt_extractor = prompt_extractor
        self._score_calculator = score_calculator
        self._reporter = reporter
        self._gepa_log_dir_for = gepa_log_dir_for
        self._save_prompts = save_prompts

    def optimize_single_span(
        self,
        span: OptimizeSpan,
        *,
        span_idx: int,
        total_spans: int,
        optimized_prompts: dict[str, str],
    ) -> GEPASpanResult:
        self._reporter.emit(f"\n{'=' * 80}")
        self._reporter.emit(f"GEPA Optimization: Span {span_idx}/{total_spans} - '{span.name}'")
        self._reporter.emit(f"{'=' * 80}")

        self._reporter.emit("📊 Preparing datasets...")
        train_examples = self._example_formatter.format_examples(
            self._train_list,
            self._proc_params,
            span=span,
            proc_name=self._proc_name,
        )

        if self._test_list is not None:
            test_list_full = list(self._test_list)
            if self._config.max_validation_examples is not None:
                test_list_full = test_list_full[: self._config.max_validation_examples]
            val_examples = self._example_formatter.format_examples(
                test_list_full,
                self._proc_params,
                span=span,
                proc_name=self._proc_name,
            )
        elif len(train_examples) >= 4:
            val_examples = train_examples[: max(1, len(train_examples) // 4)]
        else:
            val_examples = train_examples

        self._reporter.emit(f"   Training examples: {len(train_examples)}")
        self._reporter.emit(f"   Validation examples: {len(val_examples)}")
        self._reporter.emit(f"   Max metric calls (from config): {self._config.max_metric_calls}")

        self._reporter.emit(f"\n🔧 Building DSPy module for span '{span.name}'...")
        dspy_module = self._module_builder.build_module(span, self._proc_params, self._param_types)

        self._reporter.emit("📝 Wrapping metric function...")
        gepa_metric = self._metric_wrapper.wrap_metric(
            self._metric,
            self._proc_params,
            span=span,
            module_ref=dspy_module,
        )

        self._reporter.emit("\n🚀 Starting GEPA optimization...")
        self._reporter.emit(f"   Optimizer model: {self._config.optimizer_model or 'default'}")
        self._reporter.emit(f"   Reflection model: {self._config.reflection_model or 'default'}")
        self._reporter.emit(f"   Reflection LM object: {self._reflection_lm}")
        if self._reflection_lm:
            self._reporter.emit(
                f"   Reflection LM model: {getattr(self._reflection_lm, 'model', 'unknown')}"
            )
        self._reporter.emit(f"\n{'─' * 80}")

        self._reporter.emit("\n🔨 Creating GEPA optimizer...")
        self._reporter.emit(f"   Passing reflection_lm to GEPA: {self._reflection_lm}")

        if hasattr(dspy.settings, "lm") and dspy.settings.lm:
            model_name = getattr(dspy.settings.lm, "model", "unknown")
            lm_kwargs = getattr(dspy.settings.lm, "kwargs", {})
            self._reporter.emit(f"   DSPy settings.lm: {model_name}")
            self._reporter.emit(f"   DSPy settings.lm kwargs: {lm_kwargs}")
        else:
            self._reporter.emit("   ⚠️  WARNING: dspy.settings.lm is not configured!")

        gepa_log_dir = self._gepa_log_dir_for(self._proc_name, span.name)
        if gepa_log_dir is not None:
            gepa_log_dir.mkdir(parents=True, exist_ok=True)
            state_file = gepa_log_dir / _GEPA_STATE_NAME
            if state_file.exists():
                self._reporter.emit(
                    f"   GEPA checkpoint: resuming from {gepa_log_dir}\n"
                    f"   If optimization ends immediately, raise "
                    f"--optimizer-max-metric-calls above the prior total eval-units.",
                    flush=True,
                )

        optimizer = dspy.GEPA(
            metric=gepa_metric,
            max_metric_calls=self._config.max_metric_calls,
            reflection_lm=cast(Any, self._reflection_lm),
            reflection_minibatch_size=self._config.reflection_minibatch_size,
            track_stats=True,
            log_dir=str(gepa_log_dir) if gepa_log_dir is not None else None,
        )
        self._reporter.emit(
            f"   Reflection minibatch size: {self._config.reflection_minibatch_size}"
        )
        self._reporter.emit("   ✅ GEPA optimizer created successfully")

        est_seconds = len(train_examples) * 25
        self._reporter.emit(
            f"\n⏳ Starting compile ({len(train_examples)} examples, "
            f"budget={self._config.max_metric_calls} calls, "
            f"est ~{est_seconds // 60}m{est_seconds % 60:02d}s for baseline)..."
        )
        self._reporter.emit(f"{'─' * 80}")

        compile_start = time.monotonic()
        optimized_module = optimizer.compile(
            student=dspy_module,
            trainset=train_examples,
            valset=val_examples,
        )
        compile_elapsed = time.monotonic() - compile_start
        compile_min = int(compile_elapsed) // 60
        compile_sec = int(compile_elapsed) % 60

        metric_calls = 0
        progress = getattr(gepa_metric, "_progress", None)
        if isinstance(progress, dict):
            metric_calls = int(progress.get("count", 0))

        self._reporter.emit(f"{'─' * 80}")
        self._reporter.emit(f"✅ GEPA optimization complete for span '{span.name}'")
        self._reporter.emit(
            f"   Duration: {compile_min}m{compile_sec:02d}s  |  "
            f"DSPy metric callbacks: {metric_calls}  "
            f"(GEPA budget: {self._config.max_metric_calls} eval-units; "
            f"DSPy may call the metric more than once per unit)"
        )

        self._reporter.emit("\n🔍 Extracting optimized prompt...")
        previous_instruction = span_to_instruction(span)
        extracted = self._prompt_extractor.extract_prompts(optimized_module, [span])
        optimized_prompt = normalize_optimized_prompt_prefix(span, extracted.get(span.name))
        if optimized_prompt is None:
            self._reporter.emit("   ❌ No optimized prompt found")
            warnings.warn(f"No optimized prompt found for span '{span.name}'", UserWarning)
            return GEPASpanResult(prompt=None, score=0.0)

        optimized_prompts[span.name] = optimized_prompt
        if span.template_lines:
            span.template_lines[0].optimized_prompt_prefix = optimized_prompt or None

        effective_instruction = span_to_instruction(span)
        self._reporter.emit(f"   Previous seed length: {len(previous_instruction)} chars")
        self._reporter.emit(f"   Optimized prefix length: {len(optimized_prompt)} chars")
        self._reporter.emit(f"   Effective prompt length: {len(effective_instruction)} chars")

        if effective_instruction == previous_instruction:
            self._reporter.emit("   ⚠️  Prompt appears unchanged")
            warnings.warn(
                f"Optimized prompt for '{span.name}' appears unchanged. "
                "GEPA may not have optimized this span.",
                UserWarning,
            )
        else:
            self._reporter.emit("   ✨ Prompt successfully optimized")

        self._save_prompts(self._proc_name, optimized_prompts)

        self._reporter.emit("\n📈 Calculating final score on training set...")
        scoring_metric = self._metric_wrapper.wrap_metric(
            self._metric,
            self._proc_params,
            span=span,
            module_ref=optimized_module,
        )
        span_score = self._score_calculator.calculate_score(
            optimized_module,
            train_examples,
            scoring_metric,
        )
        self._reporter.emit(f"   Training score: {span_score:.4f}")
        self._reporter.emit(f"{'=' * 80}\n")
        return GEPASpanResult(prompt=optimized_prompt, score=span_score)
