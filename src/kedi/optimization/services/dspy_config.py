"""Service for managing DSPy LM configuration."""

from __future__ import annotations

import logging
from typing import Any, cast

import dspy
from dspy import LM
from dspy.adapters.chat_adapter import ChatAdapter
from dspy.adapters.json_adapter import JSONAdapter, _get_structured_outputs_response_format

from ...agent_adapter.llm_wrapper import ThinkingStrippingLM
from ...model_normalization import normalize_for_litellm
from ...reporter import CliReporter, Reporter
from ..config import OptimizerConfig

logger = logging.getLogger(__name__)


class AdaptiveJSONAdapter(JSONAdapter):
    """JSONAdapter that caches structured output failures per model.

    First call for a given model tries structured output normally.
    If it fails, the model name is remembered and all subsequent calls
    go straight to json_object mode -- no retry overhead, no log spam.
    Models where structured output succeeds keep using it every time.
    """

    _failed_models: set[str] = set()

    def __call__(
        self,
        lm: Any,
        lm_kwargs: dict[str, Any],
        signature: Any,
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        result = self._json_adapter_call_common(
            lm,
            lm_kwargs,
            signature,
            demos,
            inputs,
            lambda *a, **kw: ChatAdapter.__call__(self, *a, **kw),
        )
        if result:
            return result

        if lm.model in AdaptiveJSONAdapter._failed_models:
            lm_kwargs["response_format"] = {"type": "json_object"}
            return ChatAdapter.__call__(self, lm, lm_kwargs, signature, demos, inputs)

        try:
            structured_model = _get_structured_outputs_response_format(
                signature,
                self.use_native_function_calling,
            )
            lm_kwargs["response_format"] = structured_model
            return ChatAdapter.__call__(self, lm, lm_kwargs, signature, demos, inputs)
        except Exception:
            AdaptiveJSONAdapter._failed_models.add(lm.model)
            logger.info(
                "Model '%s' failed structured output on first attempt; "
                "using json_object mode for all future calls.",
                lm.model,
            )
            lm_kwargs["response_format"] = {"type": "json_object"}
            return ChatAdapter.__call__(self, lm, lm_kwargs, signature, demos, inputs)


class DSPyConfigurationService:
    """Service for managing DSPy LM setup with proper token limits.

    Handles model-specific configurations, especially for OpenAI reasoning models
    that require specific temperature and max_tokens settings.
    """

    def __init__(self, config: OptimizerConfig, *, reporter: Reporter | None = None):
        """Initialize DSPy configuration service.

        Args:
            config: Optimizer configuration with model and token settings
        """
        self.config = config
        self._reporter = reporter or CliReporter()

    def configure_main_lm(self, model: str | None = None) -> None:
        """Configure main DSPy LM with proper token limits.

        Uses model from config or provided model. Applies model-specific
        token limits and temperature settings for reasoning models.

        Args:
            model: Model identifier (uses config.optimizer_model if None)
        """
        model = model or self.config.optimizer_model
        if not model:
            raise ValueError("No model specified for main LM configuration")
        model = normalize_for_litellm(model)

        # Use configured max_tokens
        max_tokens = self.config.get_optimizer_max_tokens()

        self._reporter.emit("🔧 Configuring main DSPy LM:")
        self._reporter.emit(f"   Model: {model}")
        self._reporter.emit(f"   Max tokens: {max_tokens}")

        # WORKAROUND: Monkey-patch DSPy's LM class BEFORE creating any instances
        # This fixes DSPy's bug where internal LM instances use hardcoded max_tokens=4000
        if not hasattr(LM, "_kedi_patched"):
            lm_cls = cast(Any, LM)
            original_init = lm_cls.__init__
            configured_max_tokens = max_tokens

            def patched_init(
                self,
                model,
                model_type="chat",
                temperature=0.0,
                max_tokens=4000,
                cache=True,
                callbacks=None,
                num_retries=3,
                provider=None,
                finetuning_model=None,
                launch_kwargs=None,
                train_kwargs=None,
                **kwargs,
            ):
                # Override the default max_tokens with our configured value
                if max_tokens == 4000:  # If using default, replace with our config
                    max_tokens = configured_max_tokens
                return original_init(
                    self,
                    model,
                    model_type,
                    temperature,
                    max_tokens,
                    cache,
                    callbacks,
                    num_retries,
                    provider,
                    finetuning_model,
                    launch_kwargs,
                    train_kwargs,
                    **kwargs,
                )

            lm_cls.__init__ = patched_init
            setattr(LM, "_kedi_patched", True)
            self._reporter.emit(f"   🔧 Patched DSPy LM to use max_tokens={max_tokens} by default")

        # Create LM with configured settings
        # Enable JSON mode for more reliable structured outputs
        main_lm = ThinkingStrippingLM(
            model=model,
            max_tokens=max_tokens,
            temperature=1.0,
            response_format={"type": "json_object"},
            timeout=300,
        )

        # Configure DSPy with this LM and our adaptive adapter, disable cache
        dspy.settings.configure(
            lm=main_lm,
            cache=False,
            adapter=AdaptiveJSONAdapter(),
        )
        self._reporter.emit("   ✅ Main LM configured (AdaptiveJSONAdapter)")

        # Verify the configuration was applied
        if hasattr(dspy.settings, "lm") and dspy.settings.lm:
            lm_obj = dspy.settings.lm
            self._reporter.emit(f"   Verification: DSPy settings.lm = {lm_obj}")
            if hasattr(lm_obj, "kwargs"):
                self._reporter.emit(f"   LM kwargs: {lm_obj.kwargs}")

        # CRITICAL: Disable DSPy cache (both disk and memory)
        dspy.configure_cache(
            enable_disk_cache=False,
            enable_memory_cache=False,
        )

    def configure_reflection_lm(self, model: str | None = None) -> ThinkingStrippingLM:
        """Configure reflection LM for GEPA.

        Creates and returns an LM instance configured for reflection.
        Does not replace the main DSPy LM.

        Args:
            model: Model identifier (uses config.reflection_model if None)

        Returns:
            Configured LM instance for reflection
        """
        model = model or self.config.reflection_model
        if not model:
            raise ValueError("No model specified for reflection LM configuration")
        model = normalize_for_litellm(model)

        # Use configured max_tokens for reflection
        max_tokens = self.config.get_reflection_max_tokens()

        # Create LM with configured settings
        # Note: Reflection LM generates natural language instructions, NOT JSON
        return ThinkingStrippingLM(
            model=model,
            max_tokens=max_tokens,
            timeout=300,
        )

    def verify_configured(self) -> None:
        """Verify DSPy LM is configured and cache is disabled.

        Raises:
            RuntimeError: If DSPy LM is not configured
        """
        if not hasattr(dspy.settings, "lm") or dspy.settings.lm is None:
            raise RuntimeError("DSPy LM not configured. Call configure_main_lm() first.")

        # Ensure cache is always disabled
        dspy.configure_cache(
            enable_disk_cache=False,
            enable_memory_cache=False,
        )
