"""Utilities to translate between Kedi DSL and Python code."""

from __future__ import annotations

import re

from ..lang.ast import ProcedureDef, TypeDef


def kedi_type_to_python(type_str: str) -> str:
    """Convert Kedi type annotation to Python type annotation.

    Examples:
        list[City] -> list[City]
        str -> str
        City -> City
    """
    return type_str  # Most types are compatible


def kedi_typedef_to_python_class(typedef: TypeDef) -> str:
    """Convert Kedi type definition to Python BaseModel class.

    Example:
        ~City(name, history, economy, cuisine)
        ->
        class City(BaseModel):
            name: str
            history: str
            economy: str
            cuisine: str
    """
    lines = [f"class {typedef.name}(BaseModel):"]

    if not typedef.fields:
        lines.append("    pass")
    else:
        for field in typedef.fields:
            type_hint = (
                kedi_type_to_python(field.type_annotation) if field.type_annotation else "str"
            )
            lines.append(f"    {field.name}: {type_hint}")

    return "\n".join(lines)


def kedi_procedure_to_python_signature(proc: ProcedureDef) -> str:
    """Convert Kedi procedure definition to Python function signature with docstring.

    Example:
        @get_city_info(city) -> City
        ->
        def get_city_info(city: str) -> City:
            '''Available procedure: get_city_info'''
            ...
    """
    # Build parameter list with type hints
    param_list = []
    for param_name in proc.params:
        if proc.param_type_annotations and param_name in proc.param_type_annotations:
            type_hint = proc.param_type_annotations[param_name]
            if type_hint:
                param_list.append(f"{param_name}: {kedi_type_to_python(type_hint)}")
            else:
                param_list.append(f"{param_name}: str")
        else:
            param_list.append(f"{param_name}: str")

    return_type = (
        kedi_type_to_python(proc.return_type_annotation) if proc.return_type_annotation else "str"
    )

    lines = [
        f"def {proc.name}({', '.join(param_list)}) -> {return_type}:",
        f"    '''Available procedure: {proc.name}'''",
        "    ...",
    ]

    return "\n".join(lines)


def kedi_signature_to_python_signature(kedi_sig: str) -> str:
    """Convert Kedi procedure signature to Python function signature.

    Example:
        @consolidate_city_info_as_markdown(cities: list[City], country) -> str
        ->
        def consolidate_city_info_as_markdown(cities: list[City], country: str) -> str
    """
    # Parse: @name(params) -> return_type
    match = re.match(r"@(\w+)\((.*?)\)(?:\s*->\s*(.+))?", kedi_sig)
    if not match:
        raise ValueError(f"Invalid Kedi signature: {kedi_sig}")

    name = match.group(1)
    params_str = match.group(2).strip()
    return_type = match.group(3).strip() if match.group(3) else "str"

    # Parse parameters
    param_list = []
    if params_str:
        for param in params_str.split(","):
            param = param.strip()
            if ":" in param:
                # Has type annotation
                param_list.append(param)
            else:
                # No type, default to str
                param_list.append(f"{param}: str")

    return f"def {name}({', '.join(param_list)}) -> {return_type}"


def python_tests_to_kedi_tests(python_tests: str, procedure_name: str) -> str:
    """Convert Python pytest code to Kedi @test block.

    Example Python:
        def test_basic():
            result = consolidate([City('Istanbul', 'h', 'e', 'c')], 'Turkey')
            assert result is not None
            assert 'Istanbul' in result

    Converts to:
        @test: consolidate:
          > case: basic:
            ```
            result = consolidate([City('Istanbul', 'h', 'e', 'c')], 'Turkey')
            assert result is not None
            assert 'Istanbul' in result
            ```
    """
    lines = [f"@test: {procedure_name}:"]

    # Extract test functions
    test_functions = re.findall(
        r"def (test_\w+)\(\):(.+?)(?=def test_|\Z)", python_tests, re.DOTALL
    )

    if not test_functions:
        # No structured test functions, treat entire code as one test
        code = python_tests.strip()
        if code:
            lines.append("  > case: generated:")
            lines.append("    ```")
            for line in code.split("\n"):
                lines.append(f"    {line}")
            lines.append("    ```")
    else:
        # Convert each test function to a test case
        for test_name, test_body in test_functions:
            case_name = test_name.replace("test_", "")
            lines.append(f"  > case: {case_name}:")
            lines.append("    ```")

            # Preserve relative indentation (important for with/if/for blocks)
            # Don't strip - we need to preserve relative indentation
            body_lines = test_body.split("\n")

            # Find minimum indentation from non-empty lines
            non_empty_lines = [line for line in body_lines if line.strip()]
            if non_empty_lines:
                min_indent = min(len(line) - len(line.lstrip()) for line in non_empty_lines)

                for line in body_lines:
                    stripped = line.strip()
                    if (
                        stripped
                        and not stripped.startswith('"""')
                        and not stripped.startswith("'''")
                    ):
                        # Remove common leading indent, add Kedi indent (4 spaces)
                        dedented = line[min_indent:] if len(line) > min_indent else line.lstrip()
                        lines.append(f"    {dedented}")

            lines.append("    ```")

    return "\n".join(lines)


def python_impl_to_kedi_impl(python_impl: str, kedi_sig: str) -> str:
    """Convert Python function implementation to Kedi procedure.

    Example Python:
        markdown = f"# Cities in {country}\\n"
        for city in cities:
            markdown += f"## {city.name}\\n"
        return markdown

    Converts to:
        @consolidate_city_info_as_markdown(cities: list[City], country) -> str:
          = ```
          markdown = f"# Cities in {country}\\n"
          for city in cities:
              markdown += f"## {city.name}\\n"
          return markdown
          ```
    """
    # Clean up the implementation and normalize indentation
    impl_lines = python_impl.split("\n")

    # Find minimum indentation (dedent to remove any leading spaces LLM added)
    non_empty_lines = [line for line in impl_lines if line.strip()]
    if not non_empty_lines:
        # Empty implementation
        lines = [f"{kedi_sig.rstrip(':')}:"]
        lines.append("  = ```")
        lines.append("  pass")
        lines.append("  ```")
        return "\n".join(lines)

    min_indent = min(len(line) - len(line.lstrip()) for line in non_empty_lines)

    # Build Kedi procedure with native Python block return
    lines = [f"{kedi_sig.rstrip(':')}:"]
    lines.append("  = ```")

    for line in impl_lines:
        if line.strip():
            # Remove common leading indent, add Kedi indent (2 spaces)
            dedented = line[min_indent:] if len(line) > min_indent else line.lstrip()
            lines.append(f"  {dedented}")

    lines.append("  ```")

    return "\n".join(lines)
