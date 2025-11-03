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
        template: str,
    ) -> list[dspy.Example]:
        """Format training data as DSPy Examples.
        
        Args:
            data: List of (input, expected_output) tuples
            proc_params: List of procedure parameter names
            template: Template string to substitute inputs into
            
        Returns:
            List of dspy.Example objects with user_prompt field
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
            
            # Substitute inputs into template to create user_prompt
            user_prompt = template
            for param, val in input_dict.items():
                user_prompt = user_prompt.replace(f"<{param}>", str(val))
            
            # Create Example with user_prompt (same as DSPyAdapter.produce)
            # Also include original input values for metric to access
            example_dict = {
                "user_prompt": user_prompt,
                "_expected": expected_output,  # Metadata for metric
            }
            example_dict.update(input_dict)  # Add original input values
            
            # Mark user_prompt as input field (DSPy uses this, not the param fields)
            example = dspy.Example(**example_dict).with_inputs("user_prompt")
            examples.append(example)
        
        return examples

