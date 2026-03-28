from __future__ import annotations

import os

from pydantic import BaseModel, Field

from ..agent_adapter.adapters import PydanticAdapter
from .interfaces import CodegenAgent, ProgramContext
from .prompts import build_implementation_prompt, build_test_generation_prompt


class PythonTestOutput(BaseModel):
    """Python test code output."""

    test_code: str = Field(
        description="Complete Python test code in pytest style. Example:\ndef test_basic():\n    result = func('arg')\n    assert result is not None"
    )


class PythonImplOutput(BaseModel):
    """Python implementation output."""

    implementation: str = Field(
        description=(
            "Python function body ONLY (no 'def' signature line). "
            "Start from column 0 with NO leading whitespace. "
            "Use proper Python indentation for blocks. "
            "Always assign final value to a variable and return it. "
            "Example:\nif not items:\n    result = ''\nelse:\n    result = process(items)\nreturn result"
        )
    )


class PydanticAIAgent(CodegenAgent):
    """Agent that generates Python code using Pydantic-AI."""

    def __init__(self, model: str | None = None) -> None:
        self.model = model or os.environ.get("MODEL_NAME") or "openrouter/minimax/minimax-m2.7"
        self.adapter = PydanticAdapter(model=self.model)

    def generate_tests(
        self,
        python_signature: str,
        prompt: str,
        context: ProgramContext,
        previous_error: str | None = None,
    ) -> str:
        """Generate Python test code."""
        instruction = build_test_generation_prompt(
            python_signature, prompt, context, previous_error
        )

        result = self.adapter.produce_sync(template=instruction, output_type=PythonTestOutput)
        return result.test_code

    def generate_impl(
        self,
        python_signature: str,
        prompt: str,
        context: ProgramContext,
        python_tests: str,
        previous_error: str | None = None,
    ) -> str:
        """Generate Python implementation."""
        instruction = build_implementation_prompt(
            python_signature, prompt, context, python_tests, previous_error
        )

        result = self.adapter.produce_sync(template=instruction, output_type=PythonImplOutput)
        return result.implementation
