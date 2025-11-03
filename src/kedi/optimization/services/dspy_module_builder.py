"""Service for building DSPy modules from Kedi optimize spans."""

from __future__ import annotations

import warnings
from typing import Any

try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    dspy = None  # type: ignore

from ...lang.ast import OptimizeSpan
from ...lang.compiler.environment import KediRuntime
from ..utils import extract_output_fields, span_to_instruction
from .type_handler import StructuredTypeHandler


class DSPyModuleBuilder:
    """Service for building DSPy modules from Kedi optimize spans.
    
    Handles creation of DSPy Predict modules with proper signatures
    and instructions from Kedi optimize spans.
    """
    
    def __init__(
        self,
        runtime: KediRuntime | None = None,
    ):
        """Initialize module builder.
        
        Args:
            runtime: Optional Kedi runtime for structured type handling
        """
        if not DSPY_AVAILABLE:
            raise ImportError("DSPy is required for DSPyModuleBuilder")
        
        self.type_handler = StructuredTypeHandler(runtime) if runtime else None
    
    def build_module(
        self,
        span: OptimizeSpan,
        proc_params: list[str],
        param_types: dict[str, type],
    ) -> dspy.Module:
        """Build DSPy module for a single span.
        
        Args:
            span: Optimize span to build module from
            proc_params: Procedure parameter names
            param_types: Parameter type mappings
            
        Returns:
            DSPy Module instance
        """
        if not DSPY_AVAILABLE:
            raise ImportError("DSPy is required")
        
        # Extract outputs from span
        span_outputs = extract_output_fields(span)
        
        if not span_outputs:
            raise ValueError(
                f"Optimize span '{span.name}' must have at least one output field"
            )
        
        # Create signature
        # CRITICAL: Make custom types available in the namespace for DSPy Signature parsing
        signature = self._create_signature(proc_params, span_outputs, span=span)
        
        # Get initial instruction
        instruction = span_to_instruction(span)
        
        # Create module with Predict
        class SingleSpanModule(dspy.Module):
            def __init__(self, sig: dspy.Signature, inst: str):
                super().__init__()
                self.predictor = dspy.Predict(sig.with_instructions(inst))
            
            def forward(self, **kwargs):
                return self.predictor(**kwargs)
        
        return SingleSpanModule(signature, instruction)
    
    def _create_signature(
        self,
        inputs: list[str],
        outputs: list[tuple[str, str | None]],
        span: OptimizeSpan | None = None,
    ) -> dspy.Signature:
        """Create DSPy Signature from inputs and outputs.
        
        Uses type handler if available for structured types, otherwise
        creates simple string signature.
        """
        # Use type handler for proper type annotations in signature
        if self.type_handler:
            try:
                signature_str = self.type_handler.create_dspy_signature_sync(
                    inputs,
                    outputs,
                    use_string_signature=True,  # String signatures work fine - DSPy handles types
                )
                
                # CRITICAL: DSPy Signature needs custom types in the calling namespace
                # Make custom types available in __main__ module so DSPy can resolve them
                output_schema = self.type_handler.build_output_schema(outputs)
                import __main__
                for field_name, field_type in output_schema.items():
                    if isinstance(field_type, type) and field_type.__module__ not in ('builtins', 'typing'):
                        # Register custom type in __main__ namespace
                        type_name = field_type.__name__
                        setattr(__main__, type_name, field_type)
                
                # Now create signature - DSPy will find the types in __main__
                return dspy.Signature(signature_str)
            except Exception as e:
                # Fallback to simple signature
                import warnings
                warnings.warn(f"Failed to create typed signature: {e}, using simple signature")
                pass
        
        # Simple string signature
        input_str = ", ".join(inputs)
        output_names = [out_name for out_name, _ in outputs]
        output_str = ", ".join(output_names)
        signature_str = f"{input_str} -> {output_str}"
        return dspy.Signature(signature_str)

