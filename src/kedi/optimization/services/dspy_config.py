"""Service for managing DSPy LM configuration."""

from __future__ import annotations

try:
    import dspy
    from dspy import LM
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    dspy = None  # type: ignore
    LM = None  # type: ignore

from ..config import OptimizerConfig


class DSPyConfigurationService:
    """Service for managing DSPy LM setup with proper token limits.
    
    Handles model-specific configurations, especially for OpenAI reasoning models
    that require specific temperature and max_tokens settings.
    """
    
    def __init__(self, config: OptimizerConfig):
        """Initialize DSPy configuration service.
        
        Args:
            config: Optimizer configuration with model and token settings
        """
        if not DSPY_AVAILABLE:
            raise ImportError(
                "DSPy is required for DSPyConfigurationService. "
                "Install with: pip install dspy-ai"
            )
        self.config = config
    
    def configure_main_lm(self, model: str | None = None) -> None:
        """Configure main DSPy LM with proper token limits.
        
        Uses model from config or provided model. Applies model-specific
        token limits and temperature settings for reasoning models.
        
        Args:
            model: Model identifier (uses config.optimizer_model if None)
        """
        if not DSPY_AVAILABLE:
            raise ImportError("DSPy is required")
        
        model = model or self.config.optimizer_model
        if not model:
            raise ValueError("No model specified for main LM configuration")
        
        # Get appropriate max_tokens for this model
        max_tokens = self.config.get_model_max_tokens(model)
        
        # Check if it's a reasoning model (requires temperature=1.0)
        if self.config.is_reasoning_model(model):
            main_lm = LM(model=model, temperature=1.0, max_tokens=max_tokens)
        else:
            main_lm = LM(model=model, max_tokens=max_tokens)
        
        # Configure DSPy with this LM, disable cache
        dspy.settings.configure(lm=main_lm, cache=False)
        
        # CRITICAL: Disable DSPy cache (both disk and memory)
        dspy.configure_cache(
            enable_disk_cache=False,
            enable_memory_cache=False,
        )
    
    def configure_reflection_lm(self, model: str | None = None) -> LM:
        """Configure reflection LM for GEPA.
        
        Creates and returns an LM instance configured for reflection.
        Does not replace the main DSPy LM.
        
        Args:
            model: Model identifier (uses config.reflection_model if None)
            
        Returns:
            Configured LM instance for reflection
        """
        if not DSPY_AVAILABLE:
            raise ImportError("DSPy is required")
        
        model = model or self.config.reflection_model
        if not model:
            raise ValueError("No model specified for reflection LM configuration")
        
        # Get appropriate max_tokens for reflection model
        max_tokens = self.config.get_model_max_tokens(model)
        
        # Reasoning models require temperature=1.0
        if self.config.is_reasoning_model(model):
            return LM(model=model, temperature=1.0, max_tokens=max_tokens)
        else:
            return LM(model=model, max_tokens=max_tokens)
    
    def verify_configured(self) -> None:
        """Verify DSPy LM is configured and cache is disabled.
        
        Raises:
            RuntimeError: If DSPy LM is not configured
        """
        if not DSPY_AVAILABLE:
            raise ImportError("DSPy is required")
        
        if not hasattr(dspy.settings, "lm") or dspy.settings.lm is None:
            raise RuntimeError(
                "DSPy LM not configured. "
                "Call configure_main_lm() first."
            )
        
        # Ensure cache is always disabled
        dspy.configure_cache(
            enable_disk_cache=False,
            enable_memory_cache=False,
        )

