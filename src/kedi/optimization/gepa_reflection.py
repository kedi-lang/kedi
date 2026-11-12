from __future__ import annotations

from typing import Any

import dspy

from ..model_normalization import normalize_for_litellm
from ..reporter import Reporter
from .config import OptimizerConfig
from .thinking_lm import ThinkingStrippingLM

__all__ = ["build_reflection_lm"]


def build_reflection_lm(
    config: OptimizerConfig,
    reporter: Reporter,
) -> Any:
    """Use the configured main LM unless a separate reflection model is requested."""
    if not config.reflection_model:
        # DSPy 3.4.1 leaves `lm` out of its default settings until one is configured.
        main_lm = getattr(dspy.settings, "lm", None)
        if main_lm is None:
            raise ValueError("Configure a main DSPy LM or an explicit reflection model for GEPA")
        reporter.emit("\nUsing the configured main DSPy LM for GEPA reflection")
        return main_lm

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

    class TracedReflectionLM(dspy.BaseLM):
        """Trace reflection calls and delegate them to the wrapped LM.

        DSPy runs GEPA reflection inside a ``Predict``, which accepts only ``BaseLM``
        instances; the wrapped LM still owns request execution.
        """

        def __init__(self, base_lm: Any, model_name: str, call_count: dict[str, int]) -> None:
            super().__init__(model=model_name)
            self.base_lm = base_lm
            self.model_name = model_name
            self.call_count = call_count

        def _trace_call(self) -> None:
            self.call_count["count"] += 1
            reporter.emit(
                f"\n💭 Reflection LM call #{self.call_count['count']}",
                flush=True,
            )
            reporter.emit(f"   Model: {self.model_name}", flush=True)

        @staticmethod
        def _forwarded(messages: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
            return kwargs if messages is None else {**kwargs, "messages": messages}

        def __call__(self, prompt: Any = None, *, messages: Any = None, **kwargs: Any) -> Any:
            self._trace_call()
            result = self.base_lm(prompt, **self._forwarded(messages, kwargs))
            reporter.emit("   ✅ Reflection complete", flush=True)
            return result

        async def acall(self, prompt: Any = None, *, messages: Any = None, **kwargs: Any) -> Any:
            self._trace_call()
            result = await self.base_lm.acall(prompt, **self._forwarded(messages, kwargs))
            reporter.emit("   ✅ Reflection complete", flush=True)
            return result

    reflection_lm = TracedReflectionLM(
        base_lm,
        config.reflection_model,
        reflection_call_count,
    )
    reporter.emit("   ✅ Reflection LM created successfully (with call tracing)")
    return reflection_lm
