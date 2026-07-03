from __future__ import annotations

from typing import Any

import dspy

from ..model_normalization import normalize_for_litellm
from ..agent_adapter.llm_wrapper import ThinkingStrippingLM
from ..reporter import Reporter
from .config import OptimizerConfig

__all__ = ["build_reflection_lm"]


def build_reflection_lm(
    config: OptimizerConfig,
    reporter: Reporter,
) -> Any | None:
    """Configure and return a traced reflection LM, or ``None`` if not configured."""
    if not config.reflection_model:
        reporter.emit(
            "\n⚠️  No reflection model configured - GEPA will use default LM for reflection"
        )
        return None

    reporter.emit("\n🤔 Configuring reflection LM...")
    reporter.emit(f"   Model: {config.reflection_model}")
    max_tokens = config.get_reflection_max_tokens()
    reporter.emit(f"   Max tokens: {max_tokens}")
    reporter.emit("   Temperature: 1.0")

    base_lm = ThinkingStrippingLM(
        model=normalize_for_litellm(config.reflection_model),
        temperature=1.0,
        max_tokens=max_tokens,
        timeout=300,
    )

    reflection_call_count = {"count": 0}

    class TracedReflectionLM:
        """Wrapper that traces reflection LM calls while maintaining DSPy compatibility."""

        def __init__(self, base_lm: Any, model_name: str, call_count: dict[str, int]) -> None:
            self.base_lm = base_lm
            self.model_name = model_name
            self.model = model_name
            self.call_count = call_count

        def __call__(self, prompt: Any) -> Any:
            self.call_count["count"] += 1
            reporter.emit(
                f"\n💭 Reflection LM call #{self.call_count['count']}",
                flush=True,
            )
            reporter.emit(f"   Model: {self.model_name}", flush=True)
            result = self.base_lm(prompt)
            reporter.emit("   ✅ Reflection complete", flush=True)
            return result

    reflection_lm = TracedReflectionLM(
        base_lm,
        config.reflection_model,
        reflection_call_count,
    )
    reporter.emit("   ✅ Reflection LM created successfully (with call tracing)")
    return reflection_lm
