"""Shared prompt generation for AI code generation agents."""

from __future__ import annotations

from .interfaces import ProgramContext


def build_context_section(context: ProgramContext) -> str:
    """Build the AVAILABLE CONTEXT section."""
    parts = ["AVAILABLE CONTEXT (use ONLY these):"]

    if context.preamble:
        parts.append(f"# Preamble code:\n{context.preamble}")

    if context.type_definitions:
        parts.append(f"# Available types:\n" + "\n\n".join(context.type_definitions))
    else:
        parts.append("# Available types: (none)")

    if context.procedure_definitions:
        parts.append(f"# Available functions:\n" + "\n\n".join(context.procedure_definitions))
    else:
        parts.append("# Available functions: (none)")

    return "\n\n".join(parts)


def build_test_generation_prompt(
    python_signature: str, prompt: str, context: ProgramContext, previous_error: str | None = None
) -> str:
    """Build complete prompt for test generation."""
    parts = []

    # Context section
    parts.append(build_context_section(context))

    # Main task
    parts.append(
        f"\nWrite pytest-style test code for this function:\n{python_signature}\n\n"
        f"Purpose: {prompt}\n\n"
        f"CRITICAL CONSTRAINTS:\n"
        f"- DO NOT import any packages (no 'import' statements)\n"
        f"- DO NOT use pytest.raises or any pytest features beyond basic test functions\n"
        f"- ONLY use types, functions, and variables from the AVAILABLE CONTEXT above\n"
        f"- DO NOT assume existence of any other packages, modules, or built-in functions beyond Python basics\n\n"
        f"Requirements:\n"
        f"- Write MULTIPLE test functions (def test_name():) covering different scenarios\n"
        f"- Include tests for: normal cases, edge cases (empty inputs, None, etc.), and boundary conditions\n"
        f'- Use assert statements with MEANINGFUL ERROR MESSAGES: assert condition, "Expected X but got Y"\n'
        f'- Example: assert result is not None, f"Expected non-None result, got {{result}}"\n'
        f"- Example: assert 'expected' in result, f\"Expected 'expected' in result, got: {{result}}\"\n"
        f"- For exception testing, use try/except blocks, NOT pytest.raises"
    )

    # Previous error if any
    if previous_error:
        parts.append(
            f"\n# Previous attempt failed with:\n{previous_error}\n\nPlease fix the issues."
        )

    return "\n\n".join(parts)


def build_implementation_prompt(
    python_signature: str,
    prompt: str,
    context: ProgramContext,
    python_tests: str,
    previous_error: str | None = None,
) -> str:
    """Build complete prompt for implementation generation."""
    parts = []

    # Context section
    parts.append(build_context_section(context))

    # Tests that must pass
    parts.append(f"\n# Tests that must pass:\n{python_tests}")

    # Main task
    parts.append(
        f"\nImplement this function:\n{python_signature}\n\n"
        f"Purpose: {prompt}\n\n"
        f"CRITICAL CONSTRAINTS:\n"
        f"- DO NOT import any packages (no 'import' statements)\n"
        f"- ONLY use types, functions, and variables from the AVAILABLE CONTEXT above\n"
        f"- DO NOT use any external libraries, packages, or modules\n"
        f"- DO NOT assume existence of functions not listed in AVAILABLE CONTEXT\n\n"
        f"CRITICAL FORMATTING REQUIREMENTS:\n"
        f"- Provide ONLY the function body (NO 'def' signature line)\n"
        f"- Start code at column 0 (NO leading spaces or indentation)\n"
        f"- Use standard Python indentation (4 spaces) for nested blocks\n"
        f"- Assign final value to a variable before returning: result = ...; return result\n"
        f"- Must pass all tests above\n\n"
        f"Example format:\n"
        f"if condition:\n"
        f"    result = value1\n"
        f"else:\n"
        f"    result = value2\n"
        f"return result"
    )

    # Previous error if any
    if previous_error:
        parts.append(
            f"\n# Previous attempt failed with:\n{previous_error}\n\nPlease fix the issues."
        )

    return "\n\n".join(parts)
