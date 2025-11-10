"""Service for loading optimization training and test data."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ...lang.compiler import _execute_python_block, _eval_python_inline
from ...lang.compiler.environment import KediRuntime
from ...lang.parser.block_parser import EvalDataset, EvalSuite


class OptimizationDataLoader:
    """Service for loading training and test data from eval suites.
    
    Handles execution of dataset code and conversion to (input, expected_output) tuples.
    """
    
    def __init__(self, runtime: KediRuntime):
        """Initialize data loader.
        
        Args:
            runtime: Compiled Kedi runtime with access to procedures and type environment
        """
        self.runtime = runtime
    
    def load_training_data(
        self,
        eval_suite: EvalSuite,
        dataset_name: str | None = None,
    ) -> list[tuple[Any, Any]]:
        """Load training data from eval suite.
        
        Args:
            eval_suite: Eval suite containing datasets
            dataset_name: Specific dataset name to load (uses first dataset if None)
            
        Returns:
            List of (input, expected_output) tuples
            
        Raises:
            ValueError: If no datasets available or dataset not found
            RuntimeError: If data execution fails or data is not iterable
        """
        if not eval_suite.datasets:
            raise ValueError("No training datasets available in eval suite")
        
        # Get dataset (by name or first one)
        if dataset_name:
            if dataset_name not in eval_suite.datasets:
                raise ValueError(
                    f"Dataset '{dataset_name}' not found in eval suite. "
                    f"Available: {', '.join(eval_suite.datasets.keys())}"
                )
            dataset = eval_suite.datasets[dataset_name]
        else:
            dataset = list(eval_suite.datasets.values())[0]
        
        # Execute dataset code
        try:
            if dataset.is_block:
                val = _execute_python_block(dataset.code, self.runtime, {})
            else:
                val = _eval_python_inline(self.runtime, dataset.code, {})
        except Exception as e:
            raise RuntimeError(
                f"Error loading training data '{dataset.name}': {e}"
            ) from e
        
        # Validate and convert to list
        data_list = self._convert_to_list(val, dataset.name)
        
        # Format as (input, expected_output) tuples
        return self._format_data_as_tuples(data_list)
    
    def load_test_data(
        self,
        eval_suite: EvalSuite,
        dataset_name: str,
    ) -> list[tuple[Any, Any]] | None:
        """Load test data from eval suite.
        
        Args:
            eval_suite: Eval suite containing test_datasets
            dataset_name: Dataset name to load from test_datasets
            
        Returns:
            List of (input, expected_output) tuples, or None if test data not available
            
        Raises:
            RuntimeError: If data execution fails or data is not iterable
        """
        if not eval_suite.test_datasets:
            return None
        
        if dataset_name not in eval_suite.test_datasets:
            return None
        
        test_dataset = eval_suite.test_datasets[dataset_name]
        
        # Execute dataset code
        try:
            if test_dataset.is_block:
                val = _execute_python_block(test_dataset.code, self.runtime, {})
            else:
                val = _eval_python_inline(self.runtime, test_dataset.code, {})
        except Exception as e:
            raise RuntimeError(
                f"Error loading test data '{test_dataset.name}': {e}"
            ) from e
        
        # Validate and convert to list
        data_list = self._convert_to_list(val, test_dataset.name)
        
        # Format as (input, expected_output) tuples
        return self._format_data_as_tuples(data_list)
    
    def _convert_to_list(self, val: Any, dataset_name: str) -> list[Any]:
        """Convert dataset value to list.
        
        Handles dict.items(), lists, tuples, and other iterables.
        Validates that value is iterable and not str/bytes.
        
        Args:
            val: Value returned from dataset code execution
            dataset_name: Name of dataset (for error messages)
            
        Returns:
            List of items
            
        Raises:
            ValueError: If value is not iterable or is str/bytes
        """
        # Validate it's iterable
        if not hasattr(val, "__iter__") or isinstance(val, (str, bytes)):
            raise ValueError(
                f"Data '{dataset_name}' must return an iterable "
                f"(got {type(val).__name__})"
            )
        
        # Convert dict.items() to list
        if isinstance(val, dict):
            return list(val.items())
        
        # Convert other iterables to list
        return list(val)
    
    def _format_data_as_tuples(
        self,
        data_list: list[Any],
    ) -> list[tuple[Any, Any]]:
        """Format data list as (input, expected_output) tuples.
        
        Handles:
        - Tuples with 2 elements: use as-is (input, expected_output)
        - Tuples with other lengths: (item[0], None)
        - Single objects: (obj, None)
        
        Args:
            data_list: List of data items
            
        Returns:
            List of (input, expected_output) tuples
        """
        formatted_data = []
        
        for item in data_list:
            if isinstance(item, tuple):
                if len(item) == 2:
                    # Use tuple as-is: first element is input, second is expected output
                    formatted_data.append(item)
                else:
                    # Single value in tuple - make it (value, None)
                    formatted_data.append((item[0], None))
            else:
                # Single object - wrap as (obj, None)
                formatted_data.append((item, None))
        
        return formatted_data

