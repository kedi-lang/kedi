from __future__ import annotations

import ast
import textwrap
from collections.abc import Callable, Mapping
from importlib import import_module
from typing import Any, cast


class MontyInterpreter:
    """DSPy CodeAct-compatible interpreter backed by pydantic-monty.

    DSPy's ``CodeAct`` calls the interpreter once for each tool function source
    before executing model-generated code. This interpreter keeps non-tool
    definitions as a prelude and exposes registered host tools through Monty
    inputs, so generated code can call tools by name.
    """

    def __init__(
        self,
        *,
        tools: Mapping[str, Callable[..., Any]] | None = None,
        skip_sources: set[str] | None = None,
        monty_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        self.tools = dict(tools or {})
        self.skip_sources = set(skip_sources or ())
        self.monty_kwargs = dict(monty_kwargs or {})
        self._prelude: list[str] = []

    def __enter__(self) -> MontyInterpreter:
        return self

    def __exit__(self, *_: object) -> None:
        self.shutdown()

    def __call__(self, code: str, variables: dict[str, Any] | None = None) -> Any:
        return self.execute(code, variables)

    def execute(self, code: str, variables: dict[str, Any] | None = None) -> Any:
        code = textwrap.dedent(code).strip()
        if not code:
            return ""
        if code in self.skip_sources:
            return ""

        definition_name = _single_definition_name(code)
        if definition_name is not None:
            if definition_name not in self.tools:
                self._prelude.append(code)
            return ""

        Monty = _monty_class()
        inputs = dict(variables or {})
        stdout: list[str] = []

        def print_callback(stream: str, value: str) -> None:
            if stream == "stdout":
                stdout.append(value)

        sandbox = Monty(
            _join_code([*self._prelude, code]), inputs=list(inputs), **self.monty_kwargs
        )
        result = sandbox.run(
            inputs=inputs,
            external_functions=self.tools,
            print_callback=print_callback,
        )
        captured = "".join(stdout).rstrip("\n")
        if captured:
            return captured
        return "" if result is None else result

    def shutdown(self) -> None:
        self._prelude.clear()


def _monty_class() -> type[Any]:
    try:
        module = import_module("pydantic_monty")
    except ImportError as exc:
        raise RuntimeError(
            "MontyInterpreter requires pydantic-monty. Install it with "
            '`uv pip install "kedi[monty]"` or `uv add kedi --extra monty`.'
        ) from exc
    return cast("type[Any]", module.Monty)


def _single_definition_name(code: str) -> str | None:
    try:
        module = ast.parse(code)
    except SyntaxError:
        return None
    if len(module.body) != 1:
        return None
    node = module.body[0]
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return node.name
    return None


def _join_code(parts: list[str]) -> str:
    return "\n\n".join(part for part in parts if part.strip())
