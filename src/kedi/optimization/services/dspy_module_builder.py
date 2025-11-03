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
        
        # Create signature using DSPyAdapter.type_builder() directly (same as adapters.py)
        signature = self._create_signature(span_outputs)
        
        # Get initial instruction (template string)
        instruction = span_to_instruction(span)
        
        # Create module with Predict
        class SingleSpanModule(dspy.Module):
            def __init__(self, sig: dspy.Signature, template: str, proc_params: list[str]):
                super().__init__()
                self.predictor = dspy.Predict(sig.with_instructions(template))
                self.template = template
                self.proc_params = proc_params
            
            def forward(self, **kwargs):
                # Examples have user_prompt field (substituted template)
                # Pass it directly to predictor (same as DSPyAdapter.produce)
                return self.predictor(user_prompt=kwargs["user_prompt"])
        
        return SingleSpanModule(signature, instruction, proc_params)
    
    def _create_signature(
        self,
        outputs: list[tuple[str, str | None]],
    ) -> dspy.Signature:
        """Create DSPy Signature using exact same logic as DSPyAdapter.type_builder().
        
        Signature is str -> output_types (same as DSPyAdapter.produce).
        Inputs are substituted in the template, not in the signature.
        """
        # Build output schema from runtime's type environment
        output_schema = self.type_handler.build_output_schema(outputs)
        
        # Use exact same logic as DSPyAdapter.type_builder() when string=False (lines 101-110)
        from pydantic import create_model
        
        field_definitions = {
            field: (type_, dspy.OutputField()) for field, type_ in output_schema.items()
        }
        field_definitions.update({"user_prompt": (str, dspy.InputField())})
        
        return create_model(
            "_OutputSignature",
            __base__=dspy.Signature,
            **field_definitions,
        )

