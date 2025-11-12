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
        
        # Use configured max_tokens
        max_tokens = self.config.get_optimizer_max_tokens()
        
        print(f"🔧 Configuring main DSPy LM:")
        print(f"   Model: {model}")
        print(f"   Max tokens: {max_tokens}")
        
        # WORKAROUND: Monkey-patch DSPy's LM class BEFORE creating any instances
        # This fixes DSPy's bug where internal LM instances use hardcoded max_tokens=4000
        if not hasattr(LM, '_kedi_patched'):
            original_init = LM.__init__
            configured_max_tokens = max_tokens
            
            def patched_init(self, model, model_type="chat", temperature=0.0, max_tokens=4000, 
                            cache=True, callbacks=None, num_retries=3, provider=None,
                            finetuning_model=None, launch_kwargs=None, train_kwargs=None, **kwargs):
                # Override the default max_tokens with our configured value
                if max_tokens == 4000:  # If using default, replace with our config
                    max_tokens = configured_max_tokens
                return original_init(self, model, model_type, temperature, max_tokens,
                                   cache, callbacks, num_retries, provider,
                                   finetuning_model, launch_kwargs, train_kwargs, **kwargs)
            
            LM.__init__ = patched_init
            LM._kedi_patched = True
            print(f"   🔧 Patched DSPy LM to use max_tokens={max_tokens} by default")
        
        # Create LM with configured settings
        # Enable JSON mode for more reliable structured outputs
        main_lm = LM(
            model=model,
            max_tokens=max_tokens,
            temperature=1.0,
            response_format={"type": "json_object"}
        )
        
        # Configure DSPy with this LM, disable cache
        dspy.settings.configure(lm=main_lm, cache=False)
        print(f"   ✅ Main LM configured")
        
        # Verify the configuration was applied
        if hasattr(dspy.settings, 'lm') and dspy.settings.lm:
            lm_obj = dspy.settings.lm
            print(f"   Verification: DSPy settings.lm = {lm_obj}")
            if hasattr(lm_obj, 'kwargs'):
                print(f"   LM kwargs: {lm_obj.kwargs}")
        
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
        
        # Use configured max_tokens for reflection
        max_tokens = self.config.get_reflection_max_tokens()
        
        # Create LM with configured settings
        # Note: Reflection LM generates natural language instructions, NOT JSON
        return LM(
            model=model,
            max_tokens=max_tokens,
        )
    
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

