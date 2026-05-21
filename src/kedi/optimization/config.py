"""Configuration for optimizer execution."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OptimizerConfig:
    """Configuration for optimizer execution.

    Centralizes all optimizer settings to avoid parameter proliferation
    and ensure consistent configuration across optimizers.
    """

    max_metric_calls: int = 100
    """Maximum number of metric evaluations during optimization."""

    optimizer_model_max_tokens: int = 8000
    """Max tokens for the main optimizer model."""

    reflection_model_max_tokens: int = 32000
    """Max tokens for the reflection model (GEPA)."""

    optimizer_model: str | None = None
    """Main model identifier for optimizer (e.g., 'openai/gpt-4o')."""

    reflection_model: str | None = None
    """Reflection model identifier for GEPA (e.g., 'openai/gpt-5')."""

    reflection_minibatch_size: int = 3
    """Number of examples to evaluate before triggering reflection in GEPA."""

    max_validation_examples: int | None = None
    """Maximum number of validation examples for baseline evaluation. If None, uses entire test set."""

    @classmethod
    def from_cli_options(
        cls,
        max_metric_calls: int = 100,
        optimizer_model_max_tokens: int = 8000,
        reflection_model_max_tokens: int = 32000,
        optimizer_model: str | None = None,
        reflection_model: str | None = None,
        reflection_minibatch_size: int = 3,
        max_validation_examples: int | None = None,
    ) -> OptimizerConfig:
        """Create config from CLI options.

        Args:
            max_metric_calls: Maximum metric evaluations
            optimizer_model_max_tokens: Max tokens for main model
            reflection_model_max_tokens: Max tokens for reflection model
            optimizer_model: Main model identifier
            reflection_model: Reflection model identifier
            reflection_minibatch_size: Number of examples before reflection
            max_validation_examples: Maximum validation examples for baseline

        Returns:
            OptimizerConfig instance
        """
        return cls(
            max_metric_calls=max_metric_calls,
            optimizer_model_max_tokens=optimizer_model_max_tokens,
            reflection_model_max_tokens=reflection_model_max_tokens,
            optimizer_model=optimizer_model,
            reflection_model=reflection_model,
            reflection_minibatch_size=reflection_minibatch_size,
            max_validation_examples=max_validation_examples,
        )

    def get_optimizer_max_tokens(self) -> int:
        """Get max_tokens for the optimizer model.

        Returns:
            Configured max_tokens value
        """
        return self.optimizer_model_max_tokens

    def get_reflection_max_tokens(self) -> int:
        """Get max_tokens for the reflection model.

        Returns:
            Configured max_tokens value
        """
        return self.reflection_model_max_tokens
