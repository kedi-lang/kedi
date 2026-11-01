"""Utilities to translate between Kedi DSL and Python code."""

from __future__ import annotations

import ast
import re

from ..lang.ast import ProcedureDef, TypeDef
from ..lang.type_render import (
    procedure_param_type_to_python,
    procedure_return_type_to_python,
    type_expr_to_python,
)


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
            # Prefer the structured type_expr (falls back to the legacy string) so a backtick
            # Python type like `dict[str, int]` renders as clean Python, not leaked backticks.
            type_hint = type_expr_to_python(field.type_expr, field.type_annotation, default="str")
            default = f" = {field.default_expr}" if field.default_expr is not None else ""
            lines.append(f"    {field.name}: {type_hint}{default}")

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
    # Build parameter list with type hints. Render from the structured type_expr (legacy string
    # fallback) so backtick Python types become clean Python rather than leaked backticks.
    param_list = []
    for param_name in proc.params:
        type_hint = procedure_param_type_to_python(proc, param_name, default="str")
        piece = f"{param_name}: {type_hint}"
        default = (proc.param_defaults or {}).get(param_name)
        if default is not None:
            piece += f" = {default}"
        param_list.append(piece)

    return_type = procedure_return_type_to_python(proc, default="str")

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

    # Parse parameters, splitting only on top-level commas so a generic annotation like
    # `dict[str, int]` is not torn apart at the comma inside its brackets.
    param_list = []
    for param in _split_top_level_commas(params_str):
        param = param.strip()
        if not param:
            continue
        if ":" in param:
            # Has type annotation
            param_list.append(param)
        else:
            # No type, default to str
            param_list.append(f"{param}: str")

    return f"def {name}({', '.join(param_list)}) -> {return_type}"


def _split_top_level_commas(text: str) -> list[str]:
    """Split on commas not nested inside [], (), {}, or a string literal.

    Bracket-depth aware so ``dict[str, int]`` stays intact, and string-aware so a comma or bracket
    inside a default like ``sep: str = ","`` doesn't mis-split or skew the depth count.
    """
    parts: list[str] = []
    depth = 0
    quote = ""  # the open string delimiter, or "" when not inside a string literal
    current: list[str] = []
    for ch in text:
        if quote:
            current.append(ch)
            if ch == quote:
                quote = ""
            continue
        if ch in "'\"":
            quote = ch
            current.append(ch)
            continue
        if ch in "[({":
            depth += 1
        elif ch in "])}":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current))
    return parts


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

    try:
        module = ast.parse(python_tests)
    except SyntaxError as exc:
        raise ValueError(f"Invalid generated Python tests: {exc.msg}") from exc
    test_functions = [
        node
        for node in module.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    ]

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
        # Keep module-level helpers and setup in each independently executed case.
        setup = [node for node in module.body if node not in test_functions]
        for test in test_functions:
            if (
                isinstance(test, ast.AsyncFunctionDef)
                or test.decorator_list
                or test.args.posonlyargs
                or test.args.args
                or test.args.kwonlyargs
                or test.args.vararg
                or test.args.kwarg
            ):
                raise ValueError(
                    f"Generated test {test.name!r} must be synchronous, undecorated, "
                    "and take no parameters; fixtures and parametrization are unsupported"
                )
            lines.append(f"  > case: {test.name.removeprefix('test_')}:")
            lines.append("    ```")
            body = test.body[1:] if ast.get_docstring(test) is not None else test.body
            code = "\n".join(ast.unparse(node) for node in [*setup, *body])
            lines.extend(f"    {line}" for line in code.splitlines())
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
    # The impl prompt asks for ONLY the function body (no `def` line). An LLM that returns a full
    # `def name(...): ...` wrapper would be embedded verbatim into the value block and only DEFINED,
    # never called, so the procedure silently returns None. Reject any top-level function that is
    # never referenced in the body — this catches a bare wrapper AND a wrapper followed by unrelated
    # trailing statements, while keeping the legitimate "define a helper and call it" body (whose
    # helper name IS referenced). Retries with a clear signal instead of caching a dead procedure.
    try:
        tree = ast.parse(python_impl)
    except SyntaxError:
        tree = None
    if tree is not None:
        top_defs = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        if top_defs:
            referenced = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
            unused = sorted(top_defs - referenced)
            if unused:
                raise ValueError(
                    "Generated implementation must be the function body only, not a 'def' wrapper "
                    f"(defined but never called: {unused}). Provide just the body."
                )

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
