from __future__ import annotations

import ast
import sys
import types
from typing import Any

import pytest

import kedi.agent_adapter.monty_interpreter as monty_interpreter
from kedi.agent_adapter import MontyInterpreter


class FakeMonty:
    calls: list[FakeMonty] = []

    def __init__(self, code: str, inputs: list[str], **kwargs: Any) -> None:
        self.code = code
        self.inputs = inputs
        self.kwargs = kwargs
        self.__class__.calls.append(self)

    def run(
        self,
        *,
        inputs: dict[str, Any],
        external_functions: dict[str, Any] | None = None,
        print_callback: Any = None,
    ) -> Any:
        tree = ast.parse(self.code)
        env = {**(external_functions or {}), **inputs}
        if print_callback is not None:
            print_callback("stderr", "ignored")
            env["print"] = lambda *values: _fake_print(print_callback, *values)
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            setup = ast.Module(body=tree.body[:-1], type_ignores=[])
            ast.fix_missing_locations(setup)
            expr = ast.Expression(tree.body[-1].value)
            ast.fix_missing_locations(expr)
            exec(compile(setup, "<fake-monty>", "exec"), env, env)
            return eval(compile(expr, "<fake-monty>", "eval"), env, env)

        ast.fix_missing_locations(tree)
        exec(compile(tree, "<fake-monty>", "exec"), env, env)
        return None


@pytest.fixture
def fake_monty_module(monkeypatch: pytest.MonkeyPatch) -> type[FakeMonty]:
    FakeMonty.calls.clear()
    module = types.ModuleType("pydantic_monty")
    setattr(module, "Monty", FakeMonty)
    monkeypatch.setitem(sys.modules, "pydantic_monty", module)
    return FakeMonty


def _fake_print(print_callback: Any, *values: Any) -> None:
    print_callback("stdout", " ".join(str(value) for value in values))
    print_callback("stdout", "\n")


def test_monty_interpreter_uses_registered_host_tool(
    fake_monty_module: type[FakeMonty],
) -> None:
    def search_schema(query: str) -> str:
        return f"schema:{query}"

    interpreter = MontyInterpreter(tools={"search_schema": search_schema})

    assert interpreter("def search_schema(query: str) -> str:\n    return 'wrong'") == ""
    assert interpreter("print(search_schema(query='users'))") == "schema:users"

    call = fake_monty_module.calls[-1]
    assert call.inputs == []
    assert "return 'wrong'" not in call.code


def test_monty_interpreter_returns_empty_for_empty_code() -> None:
    assert MontyInterpreter()(" \n ") == ""


def test_monty_interpreter_persists_non_tool_definitions(
    fake_monty_module: type[FakeMonty],
) -> None:
    interpreter = MontyInterpreter(monty_kwargs={"strict": True})

    assert interpreter("def helper() -> None:\n    print('ok')") == ""
    assert interpreter("helper()") == "ok"

    call = fake_monty_module.calls[-1]
    assert "def helper" in call.code
    assert call.kwargs == {"strict": True}


def test_monty_interpreter_returns_expression_result(
    fake_monty_module: type[FakeMonty],
) -> None:
    assert MontyInterpreter()("1 + 2") == 3


def test_monty_interpreter_merges_runtime_variables(
    fake_monty_module: type[FakeMonty],
) -> None:
    interpreter = MontyInterpreter()

    assert interpreter("print(name)", {"name": "Ada"}) == "Ada"

    assert fake_monty_module.calls[-1].inputs == ["name"]


def test_monty_interpreter_context_manager_clears_prelude(
    fake_monty_module: type[FakeMonty],
) -> None:
    with MontyInterpreter() as interpreter:
        assert interpreter("def helper() -> None:\n    print('ok')") == ""

    with pytest.raises(NameError):
        interpreter("helper()")


def test_monty_interpreter_missing_dependency_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def blocked_import(name: str) -> Any:
        if name == "pydantic_monty":
            raise ImportError(name)
        raise AssertionError(name)

    monkeypatch.setattr(monty_interpreter, "import_module", blocked_import)

    with pytest.raises(RuntimeError, match="pydantic-monty"):
        MontyInterpreter()("print('x')")


def test_single_definition_name_rejects_non_single_definitions() -> None:
    assert monty_interpreter._single_definition_name("def broken(") is None
    assert monty_interpreter._single_definition_name("x = 1\ny = 2") is None
