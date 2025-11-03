"""Tests for OptimizerConfig."""

from __future__ import annotations

import pytest

from kedi.optimization.config import OptimizerConfig


def test_optimizer_config_defaults():
    """Test OptimizerConfig with default values."""
    config = OptimizerConfig()
    
    assert config.max_metric_calls == 30
    assert config.optimizer_model_max_tokens == 2000
    assert config.reflection_model_max_tokens == 32000
    assert config.optimizer_model is None
    assert config.reflection_model is None


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


def test_optimizer_config_get_model_max_tokens():
    """Test get_model_max_tokens respects model limits."""
    config = OptimizerConfig(optimizer_model_max_tokens=2000, reflection_model_max_tokens=32000)
    
    # gpt-4o should cap at 16384
    assert config.get_model_max_tokens("openai/gpt-4o") == 2000  # Below limit
    config_high = OptimizerConfig(optimizer_model_max_tokens=20000, reflection_model_max_tokens=32000)
    assert config_high.get_model_max_tokens("openai/gpt-4o") == 16384  # Capped at limit
    
    # gpt-5 should cap at 32000
    assert config.get_model_max_tokens("openai/gpt-5") == 32000  # At limit
    config_higher = OptimizerConfig(optimizer_model_max_tokens=2000, reflection_model_max_tokens=50000)
    assert config_higher.get_model_max_tokens("openai/gpt-5") == 32000  # Capped at limit
    
    # Other models use optimizer_model_max_tokens
    assert config.get_model_max_tokens("openai/gpt-4") == 2000
    assert config.get_model_max_tokens("anthropic/claude") == 2000
    
    # None model uses default
    assert config.get_model_max_tokens(None) == 2000


def test_optimizer_config_is_reasoning_model():
    """Test is_reasoning_model correctly identifies OpenAI reasoning models."""
    config = OptimizerConfig()
    
    assert config.is_reasoning_model("openai/gpt-4o") is True
    assert config.is_reasoning_model("openai/gpt-5") is True
    assert config.is_reasoning_model("openai/gpt-4") is False
    assert config.is_reasoning_model("openai/gpt-3.5-turbo") is False
    assert config.is_reasoning_model("anthropic/claude") is False
    assert config.is_reasoning_model(None) is False
    
    # Case insensitive
    assert config.is_reasoning_model("OPENAI/GPT-4O") is True
    assert config.is_reasoning_model("OpenAI/GPT-5") is True

