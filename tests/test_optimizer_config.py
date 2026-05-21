"""Tests for OptimizerConfig."""

from __future__ import annotations

from kedi.optimization.config import OptimizerConfig


def test_optimizer_config_defaults():
    """Test OptimizerConfig with default values."""
    config = OptimizerConfig()

    assert config.max_metric_calls == 100
    assert config.optimizer_model_max_tokens == 8000
    assert config.reflection_model_max_tokens == 32000
    assert config.optimizer_model is None
    assert config.reflection_model is None
    assert config.reflection_minibatch_size == 3
    assert config.max_validation_examples is None


def test_optimizer_config_from_cli_options():
    """Test OptimizerConfig.from_cli_options factory method."""
    config = OptimizerConfig.from_cli_options(
        max_metric_calls=50,
        optimizer_model_max_tokens=4000,
        reflection_model_max_tokens=64000,
        optimizer_model="openai/gpt-4o",
        reflection_model="openai/gpt-5",
    )

    assert config.max_metric_calls == 50
    assert config.optimizer_model_max_tokens == 4000
    assert config.reflection_model_max_tokens == 64000
    assert config.optimizer_model == "openai/gpt-4o"
    assert config.reflection_model == "openai/gpt-5"


def test_optimizer_config_get_max_tokens():
    """Test get_optimizer_max_tokens and get_reflection_max_tokens."""
    config = OptimizerConfig(optimizer_model_max_tokens=2000, reflection_model_max_tokens=32000)

    assert config.get_optimizer_max_tokens() == 2000
    assert config.get_reflection_max_tokens() == 32000

    # Test with different values
    config2 = OptimizerConfig(optimizer_model_max_tokens=16000, reflection_model_max_tokens=64000)
    assert config2.get_optimizer_max_tokens() == 16000
    assert config2.get_reflection_max_tokens() == 64000


def test_optimizer_config_max_validation_examples():
    """Test max_validation_examples configuration."""
    # Default is None (use entire test set)
    config = OptimizerConfig()
    assert config.max_validation_examples is None

    # Can be set explicitly
    config2 = OptimizerConfig(max_validation_examples=10)
    assert config2.max_validation_examples == 10
