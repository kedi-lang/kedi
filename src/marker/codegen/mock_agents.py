from __future__ import annotations

from .interfaces import ProgramContext


class MockPassingAgent:
    """Mock agent that generates Python code (not Marker DSL)."""
    
    def generate_tests(
        self, 
        python_signature: str,
        prompt: str,
        context: ProgramContext,
        previous_error: str | None = None
    ) -> str:
        """Generate Python pytest-style test code."""
        # Extract function name and parameters from signature
        # e.g., "def consolidate_city_info_as_markdown(cities: list[City], country: str) -> str"
        import re
        match = re.match(r'def (\w+)\((.*?)\)', python_signature)
        if not match:
            return "def test_basic():\n    assert True\n"
        
        func_name = match.group(1)
        params_str = match.group(2).strip()
        
        if not params_str:
            # No parameters
            return (
                f"def test_basic():\n"
                f"    result = {func_name}()\n"
                f"    assert result is not None, f\"Expected non-None result, got {{result}}\"\n"
                f"\n"
                f"def test_return_type():\n"
                f"    result = {func_name}()\n"
                f"    assert isinstance(result, str), f\"Expected str result, got {{type(result)}}\"\n"
            )
        
        # Generate test with generic mock values based on type hints
        test_args = []
        empty_args = []
        
        for param in params_str.split(','):
            param = param.strip()
            if ':' in param:
                type_hint = param.split(':')[1].strip()
                # Generate appropriate value based on type
                if 'list' in type_hint:
                    test_args.append("['test']")
                    empty_args.append('[]')
                elif type_hint in ('int', 'float'):
                    test_args.append("1")
                    empty_args.append('0')
                elif type_hint == 'bool':
                    test_args.append("True")
                    empty_args.append('False')
                else:
                    test_args.append("'test'")
                    empty_args.append("''")
            else:
                test_args.append("'test'")
                empty_args.append("''")
        
        args_str = ', '.join(test_args)
        empty_args_str = ', '.join(empty_args)
        
        return (
            f"def test_basic():\n"
            f"    result = {func_name}({args_str})\n"
            f"    assert result is not None, f\"Expected non-None result, got {{result}}\"\n"
            f"\n"
            f"def test_empty_case():\n"
            f"    # Test with empty inputs\n"
            f"    result = {func_name}({empty_args_str})\n"
            f"    assert result is not None, f\"Expected non-None result for empty input, got {{result}}\"\n"
        )
    
    def generate_impl(
        self, 
        python_signature: str,
        prompt: str,
        context: ProgramContext,
        python_tests: str,
        previous_error: str | None = None
    ) -> str:
        """Generate Python function implementation (body only, no signature)."""
        # Extract function name from signature
        import re
        match = re.match(r'def (\w+)\((.*?)\)', python_signature)
        if not match:
            return "result = 'ok'\nreturn result\n"
        
        func_name = match.group(1)
        params_str = match.group(2).strip()
        
        # Generate generic implementation based on return type
        if '->' in python_signature:
            return_type = python_signature.split('->')[-1].strip().rstrip(':')
            if 'str' in return_type:
                return "result = 'test result'\nreturn result\n"
            elif 'int' in return_type:
                return "result = 0\nreturn result\n"
            elif 'bool' in return_type:
                return "result = True\nreturn result\n"
            elif 'list' in return_type:
                return "result = []\nreturn result\n"
        
        # Default generic implementation
        return "result = 'ok'\nreturn result\n"
