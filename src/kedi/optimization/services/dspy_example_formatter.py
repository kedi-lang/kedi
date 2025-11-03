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
            
        Returns:
            List of dspy.Example objects with inputs marked
        """
        if not DSPY_AVAILABLE:
            raise ImportError("DSPy is required for DSPyExampleFormatter")
        
        examples = []
        
        for input_val, expected_output in data:
            example_dict = {}
            
            # Handle single vs multi-param
            if len(proc_params) == 1:
                example_dict[proc_params[0]] = input_val
            else:
                # Multi-param: input_val should be tuple
                if isinstance(input_val, tuple):
                    for param, val in zip(proc_params, input_val):
                        example_dict[param] = val
                else:
                    # Assume single value maps to first param
                    example_dict[proc_params[0]] = input_val
            
            # Add expected output (metadata, not an input)
            example_dict["_expected"] = expected_output
            
            # Create Example and specify which fields are inputs
            example = dspy.Example(**example_dict).with_inputs(*proc_params)
            examples.append(example)
        
        return examples

