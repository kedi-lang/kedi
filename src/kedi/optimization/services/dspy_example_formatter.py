"""Service for formatting training data as DSPy Examples."""

from __future__ import annotations

from typing import Any

try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    dspy = None  # type: ignore

from ...lang.compiler.environment import KediRuntime
from ...lang.ast import OptimizeSpan


class DSPyExampleFormatter:
    """Service for formatting training data as DSPy Examples.
    
    Converts (input, expected_output) tuples into dspy.Example objects
    with proper input field specification. When a span is provided,
    computes span-level inputs from procedure parameters.
    """
    
    def __init__(self, runtime: KediRuntime):
        """Initialize formatter with runtime for span input computation.
        
        Args:
            runtime: Kedi runtime for executing procedures to get span inputs
        """
        self.runtime = runtime
    
    def format_examples(
        self,
        data: list[tuple[Any, Any]],
        proc_params: list[str],
        span: OptimizeSpan | None = None,
        proc_name: str | None = None,
    ) -> list[dspy.Example]:
        """Format training data as DSPy Examples.
        
        Args:
            data: List of (input, expected_output) tuples
            proc_params: List of procedure parameter names
            span: Optional span to compute span-level inputs from
            proc_name: Optional procedure name for span input computation
            
        Returns:
            List of dspy.Example objects with span input fields (if span provided)
            or procedure parameter fields (fallback)
        """
        if not DSPY_AVAILABLE:
            raise ImportError("DSPy is required for DSPyExampleFormatter")
        
        examples = []
        total = len(data)
        
        for idx, (input_val, expected_output) in enumerate(data, 1):
            # Build input dict from input_val (procedure parameters)
            proc_input_dict = {}
            if len(proc_params) == 1:
                proc_input_dict[proc_params[0]] = input_val
            else:
                # Multi-param: input_val should be tuple
                if isinstance(input_val, tuple):
                    for param, val in zip(proc_params, input_val):
                        proc_input_dict[param] = val
                else:
                    # Assume single value maps to first param
                    proc_input_dict[proc_params[0]] = input_val
            
            # If span provided, compute span inputs from procedure parameters
            if span and proc_name:
                span_inputs = self._compute_span_inputs(
                    span, proc_name, proc_input_dict
                )
                if idx == 1 or idx % 5 == 0 or idx == total:
                    print(f"   Formatting example {idx}/{total}...", flush=True)
                input_fields = list(span_inputs.keys())
            else:
                # Fallback: use procedure parameters
                span_inputs = proc_input_dict
                input_fields = proc_params
            
            # Create Example with span input fields
            example_dict = {
                "_expected": expected_output,  # Metadata for metric
                "_proc_inputs": proc_input_dict,  # Store original proc inputs for metric
            }
            example_dict.update(span_inputs)  # Add span input values
            
            # Mark the span inputs as DSPy input fields
            example = dspy.Example(**example_dict).with_inputs(*input_fields)
            examples.append(example)
        
        return examples
    
    def _compute_span_inputs(
        self,
        span: OptimizeSpan,
        proc_name: str,
        proc_inputs: dict[str, Any],
    ) -> dict[str, str]:
        """Compute span input values from procedure parameters.
        
        Executes the procedure up to the span to get the actual span input values
        (e.g., prettified_architecture from architecture parameter).
        
        Args:
            span: Optimize span
            proc_name: Procedure name
            proc_inputs: Dict of procedure parameter name -> value
            
        Returns:
            Dict of span input name -> computed string value
        """
        from ..utils import extract_input_fields
        
        span_input_names = extract_input_fields(span)
        
        if not span_input_names:
            return {}
        
        # Fast path: if all span inputs are direct procedure parameters, skip the
        # expensive procedure call (which would trigger LLM calls).
        if all(name in proc_inputs for name in span_input_names):
            return {name: str(proc_inputs[name]) for name in span_input_names}
        
        # Slow path: span uses computed intermediate values — must run procedure.
        try:
            # Find the procedure definition in the runtime
            proc_def = None
            if hasattr(self.runtime, '_main') and self.runtime._main:
                # Check if it's the main procedure
                if hasattr(self.runtime._main, '__name__') and self.runtime._main.__name__ == proc_name:
                    # We need to get the AST, not the compiled function
                    # This is tricky - we need access to the original AST
                    pass
            
            # Alternative: execute the procedure and capture environment
            # We'll use a simpler approach: execute assignments before the span
            proc = self.runtime._procedures.get(proc_name)
            if not proc:
                # Fallback: return proc_inputs as strings
                return {k: str(v) for k, v in list(proc_inputs.items())[:len(span_input_names)]}
            
            # Create environment with procedure inputs
            env = dict(proc_inputs)
            
            # Execute the procedure to populate the environment
            # This will compute all the intermediate values including span inputs
            args = list(proc_inputs.values())
            
            # Call the procedure - this will populate the runtime environment
            # We need to intercept the environment at the point of the span
            try:
                # Store original environment
                original_env = self.runtime._current_env.get()
                
                # Set our environment
                self.runtime._current_env.set(env)
                
                # Execute procedure
                proc(*args)
                
                # Get the populated environment
                final_env = self.runtime._current_env.get()
                
                # Extract span input values from environment
                span_input_values = {}
                for input_name in span_input_names:
                    if input_name in final_env:
                        span_input_values[input_name] = str(final_env[input_name])
                
                if span_input_values:
                    return span_input_values
                    
            finally:
                # Restore original environment
                self.runtime._current_env.set(original_env)
            
        except Exception as e:
            # Fallback on error
            pass
        
        # Fallback: return proc_inputs as strings
        return {k: str(v) for k, v in list(proc_inputs.items())[:len(span_input_names)]}


