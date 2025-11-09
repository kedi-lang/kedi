"""Service for formatting training data as DSPy Examples."""

from __future__ import annotations

from typing import Any

try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    dspy = None  # type: ignore


class DSPyExampleFormatter:
    """Service for formatting training data as DSPy Examples.
    
    Converts (input, expected_output) tuples into dspy.Example objects
    with proper input field specification.
    """
    
    def format_examples(
        self,
        data: list[tuple[Any, Any]],
        proc_params: list[str],
    ) -> list[dspy.Example]:
        """Format training data as DSPy Examples.
        
        Args:
            data: List of (input, expected_output) tuples
            proc_params: List of procedure parameter names
            (No specific module inputs are required by prefix-only modules.)
            
        Returns:
            List of dspy.Example objects with typed input fields
        """
        if not DSPY_AVAILABLE:
            raise ImportError("DSPy is required for DSPyExampleFormatter")
        
        examples = []
        
        for input_val, expected_output in data:
            # Build input dict from input_val
            input_dict = {}
            if len(proc_params) == 1:
                input_dict[proc_params[0]] = input_val
            else:
                # Multi-param: input_val should be tuple
                if isinstance(input_val, tuple):
                    for param, val in zip(proc_params, input_val):
                        input_dict[param] = val
                else:
                    # Assume single value maps to first param
                    input_dict[proc_params[0]] = input_val
            
            # Create Example with typed input fields (A, B, etc.)
            example_dict = {
                "_expected": expected_output,  # Metadata for metric
            }
            example_dict.update(input_dict)  # Add input values (A, B, etc.)
            
            # Mark no inputs for prefix-only module; still carry expected for metric
            example = dspy.Example(**example_dict).with_inputs()
            examples.append(example)
        
        return examples

