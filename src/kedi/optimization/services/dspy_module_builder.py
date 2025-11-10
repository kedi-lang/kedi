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
from ..utils import extract_output_fields
from .type_handler import StructuredTypeHandler


class _KediRuntimeRef:
    """Lightweight, deepcopy-safe reference holder for KediRuntime."""

    def __init__(self, rt: KediRuntime):
        self.rt = rt

    def __deepcopy__(self, memo):  # noqa: D401
        # Safe to return self; underlying runtime is managed elsewhere
        return self


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
        # Create signature with procedure parameters as inputs and span outputs
        signature = self._create_signature(proc_params, span_outputs, param_types)
        
        # Initial instruction is empty prefix; GEPA will mutate this
        instruction = ""
        
        # Create module with Predict
        rt = self.type_handler.runtime
        class SingleSpanModule(dspy.Module):
            def __init__(self, sig: dspy.Signature):
                super().__init__()
                # Use the globally configured LM (inherits max_tokens from dspy.settings)
                self.predictor = dspy.Predict(sig)
                self._kedi_span = span
                # Keep runtime in a deepcopy-safe wrapper to avoid noisy warnings
                self._kedi_runtime_ref = _KediRuntimeRef(rt)
                
            
            def forward(self, **kwargs):
                # Call the predictor to generate real predictions
                # GEPA needs these predictions to build reflective datasets
                return self.predictor(**kwargs)
        
        return SingleSpanModule(signature.with_instructions(instruction))
    
    def _create_signature(
        self,
        inputs: list[str],
        outputs: list[tuple[str, str | None]],
        param_types: dict[str, type],
    ) -> dspy.Signature:
        """Create DSPy Signature with input and output fields."""
        from pydantic import create_model
        
        # Create field definitions
        field_definitions = {}
        
        # Add input fields (procedure parameters)
        for param in inputs:
            # Use str for all inputs in DSPy module
            field_definitions[param] = (str, dspy.InputField())
        
        # Add output fields
        # IMPORTANT: Use str for all outputs in DSPy module. Kedi handles real typing.
        for field, _type_anno in outputs:
            field_definitions[field] = (str, dspy.OutputField())
        
        return create_model(
            "_ProcedureSignature",
            __base__=dspy.Signature,
            **field_definitions,
        )

