from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class ProgramContext:
    """Structured program context available at the point of procedure definition.

    All context is provided as Python code that the LLM can understand:
    - preamble: Raw Python code from the top of the file
    - type_definitions: Python BaseModel class definitions translated from Marker types
    - procedure_definitions: Python function signatures with docstrings (commented) for available procedures
    """

    preamble: str = ""
    type_definitions: list[str] = field(default_factory=list)  # Python class definitions
    procedure_definitions: list[str] = field(
        default_factory=list
    )  # Python function signatures with docs


@dataclass
class CodegenConfig:
    max_retries: int = 5


class CodegenAgent(Protocol):
    def generate_tests(
        self,
        python_signature: str,
        prompt: str,
        context: ProgramContext,
        previous_error: str | None = None,
    ) -> str:
        """Generate Python test code for a procedure.

        Args:
            python_signature: Python function signature (e.g., "def func(x: str, y: int) -> str:")
            prompt: Natural language description of what the function should do
            context: Available Python context (preamble, types, other functions)
            previous_error: Error from previous attempt, if any

        Returns:
            Python test code (pytest style)
        """
        ...

    def generate_impl(
        self,
        python_signature: str,
        prompt: str,
        context: ProgramContext,
        python_tests: str,
        previous_error: str | None = None,
    ) -> str:
        """Generate Python implementation for a procedure.

        Args:
            python_signature: Python function signature to implement
            prompt: Natural language description of what the function should do
            context: Available Python context (preamble, types, other functions)
            python_tests: Python test code that the implementation must pass
            previous_error: Error from previous attempt, if any

        Returns:
            Python function implementation (body only, without signature)
        """
        ...
