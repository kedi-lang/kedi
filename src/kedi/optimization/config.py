"""Configuration for optimizer execution."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OptimizerConfig:
    """Configuration for optimizer execution.
    
    Centralizes all optimizer settings to avoid parameter proliferation
    and ensure consistent configuration across optimizers.
    """
    
    max_metric_calls: int = 30
    """Maximum number of metric evaluations during optimization."""
    
    optimizer_model_max_tokens: int = 2000
    """Max tokens for the main optimizer model."""
    
    reflection_model_max_tokens: int = 32000
    """Max tokens for the reflection model (GEPA)."""
    
    optimizer_model: str | None = None
    """Main model identifier for optimizer (e.g., 'openai/gpt-4o')."""
    
    reflection_model: str | None = None
    """Reflection model identifier for GEPA (e.g., 'openai/gpt-5')."""
    
    @classmethod
    def from_cli_options(
        cls,
        max_metric_calls: int = 30,
        optimizer_model_max_tokens: int = 2000,
        reflection_model_max_tokens: int = 32000,
        optimizer_model: str | None = None,
        reflection_model: str | None = None,
    ) -> OptimizerConfig:
        """Create config from CLI options.
        
        Args:
            max_metric_calls: Maximum metric evaluations
            optimizer_model_max_tokens: Max tokens for main model
            reflection_model_max_tokens: Max tokens for reflection model
            optimizer_model: Main model identifier
            reflection_model: Reflection model identifier
            
        Returns:
            OptimizerConfig instance
        """
        return cls(
            max_metric_calls=max_metric_calls,
            optimizer_model_max_tokens=optimizer_model_max_tokens,
            reflection_model_max_tokens=reflection_model_max_tokens,
            optimizer_model=optimizer_model,
            reflection_model=reflection_model,
        )
    
    def get_model_max_tokens(self, model: str | None) -> int:
        """Get appropriate max_tokens for a given model.
        
        Applies model-specific limits based on known constraints.
        For OpenAI reasoning models, enforces their specific limits.
        
        Args:
            model: Model identifier (e.g., 'openai/gpt-4o')
            
        Returns:
            Appropriate max_tokens value
        """
        if model is None:
            return self.optimizer_model_max_tokens
        
        model_lower = model.lower()
        
        # OpenAI reasoning model specific limits
        if "gpt-5" in model_lower:
            # gpt-5 supports max 32000 tokens
            # Use reflection_model_max_tokens but cap at model limit
            return min(self.reflection_model_max_tokens, 32000)
        elif "gpt-4o" in model_lower:
            # gpt-4o supports max 16384 tokens
            # Use optimizer_model_max_tokens but cap at model limit
            # However, if user wants more than 16384, they should use gpt-5
            return min(self.optimizer_model_max_tokens, 16384)
        elif "gpt-4" in model_lower:
            # gpt-4 (non-o) typically supports less
            return min(self.optimizer_model_max_tokens, 8192)
        
        # Default: use provided max_tokens
        return self.optimizer_model_max_tokens
    
    def is_reasoning_model(self, model: str | None) -> bool:
        """Check if model is an OpenAI reasoning model.
        
        Args:
            model: Model identifier
            
        Returns:
            True if model requires reasoning-specific settings
        """
        if model is None:
            return False
        
        model_lower = model.lower()
        return "gpt-5" in model_lower or "gpt-4o" in model_lower

