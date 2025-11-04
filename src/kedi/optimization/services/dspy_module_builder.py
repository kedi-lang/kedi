"""Service for building DSPy modules from Kedi optimize spans."""

from __future__ import annotations

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
        runtime: KediRuntime,
    ):
        """Initialize module builder.
        
        Args:
            runtime: Kedi runtime for structured type handling
        """
        if not DSPY_AVAILABLE:
            raise ImportError("DSPy is required for DSPyModuleBuilder")
        
        self.type_handler = StructuredTypeHandler(runtime)
    
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
        
        # Create signature with actual typed inputs
        signature = self._create_signature(span_outputs, proc_params, param_types)
        
        # Get initial instruction (template string) - pass as-is, no modifications
        instruction = span_to_instruction(span)
        
        # Create module with Predict
        class SingleSpanModule(dspy.Module):
            def __init__(self, sig: dspy.Signature):
                super().__init__()
                self.predictor = dspy.Predict(sig)
            
            def forward(self, **kwargs):
                # Accept typed inputs (A, B, etc.), DSPy substitutes them into instructions
                return self.predictor(**kwargs)
        
        return SingleSpanModule(signature.with_instructions(instruction))
    
    def _create_signature(
        self,
        outputs: list[tuple[str, str | None]],
        proc_params: list[str],
        param_types: dict[str, type],
    ) -> dspy.Signature:
        """Create DSPy Signature with typed inputs.
        
        Signature uses actual input types (A: float, B: float) not user_prompt: str.
        """
        # Build output schema from runtime's type environment
        output_schema = self.type_handler.build_output_schema(outputs)
        
        from pydantic import create_model
        
        # Create field definitions with typed inputs
        field_definitions = {}
        
        # Add input fields for each parameter
        for param_name in proc_params:
            param_type = param_types.get(param_name, str)
            field_definitions[param_name] = (param_type, dspy.InputField())
        
        # Add output fields
        for field, type_ in output_schema.items():
            field_definitions[field] = (type_, dspy.OutputField())
        
        return create_model(
            "_OutputSignature",
            __base__=dspy.Signature,
            **field_definitions,
        )

