from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import pytest

from kedi.agent_adapter.tool_override import ToolScope
from kedi.agent_profile.tools import eval_use_directive
from kedi.core import KediRuntime
from kedi.errors import KediExecutionError
from kedi.lang import compile_program, parse_program
from kedi.lang.ast import ProcedureDef, UseDirectiveStmt
from tests.mock_adapter import MockAdapter

if TYPE_CHECKING:
    from pydantic_ai.tools import Tool


def _adapter() -> MockAdapter:
    return MockAdapter(default_model="default-model")


def _procedure_def(fn: object) -> ProcedureDef:
    proc = getattr(fn, "__kedi_procedure_def__", None)
    assert isinstance(proc, ProcedureDef)
    return proc


def _fake_tool(name: str | None = None) -> Tool[object]:
    if name is None:
        return cast("Tool[object]", SimpleNamespace())
    return cast("Tool[object]", SimpleNamespace(name=name))


def test_eval_use_directive_multiline_returns_tools() -> None:
    merged = eval_use_directive(
        KediRuntime(),
        UseDirectiveStmt(tool_names=["alpha", "beta"]),
        {},
        {},
        available_procedures=set(),
    )
    assert merged.tools == ("alpha", "beta")


def test_use_directive_legacy_name_property_sets_target() -> None:
    stmt = UseDirectiveStmt(target="old")
    stmt.name = "new"
    assert stmt.target == "new"
    assert stmt.name == "new"


def test_tool_scope_registers_deduplicates_and_restores() -> None:
    scope = ToolScope()
    outer = _fake_tool("search")
    inner = _fake_tool("search")
    unnamed = _fake_tool()

    scope.register_tool(outer)
    with scope.tool_scope([outer]):
        assert scope.scoped_tools == (outer,)
        scope.register_tool(inner)
        scope.register_tool(unnamed)
        assert scope.scoped_tools == (inner, unnamed)

    assert scope.scoped_tools == (outer,)


def test_parse_multiline_use_directive() -> None:
    src = """
> use:
    web_browse
    `web_fetch`

@main():
    >> answer [x]
"""
    prog = parse_program(src)
    use_stmt = prog.ordered_toplevel[0]
    assert isinstance(use_stmt, UseDirectiveStmt)
    assert use_stmt.tool_names == ["web_browse", "web_fetch"]


def test_parse_single_line_use_accepts_backtick_target() -> None:
    src = """
@web_fetch():
    = ok

> use: `web_fetch`

@main():
    >> answer [x]
"""
    prog = parse_program(src)
    use_stmt = next(item for item in prog.ordered_toplevel if isinstance(item, UseDirectiveStmt))
    assert use_stmt.name == "web_fetch"
    assert use_stmt.tool_names == []


def test_parse_multiline_use_accepts_backtick_entries() -> None:
    src = """
> use:
    `web_fetch`

@main():
    >> answer [x]
"""
    prog = parse_program(src)
    use_stmt = prog.ordered_toplevel[0]
    assert isinstance(use_stmt, UseDirectiveStmt)
    assert use_stmt.tool_names == ["web_fetch"]


def test_profile_body_single_line_use_adds_tool() -> None:
    src = """
@web_search():
    = ok

> profile: web_agent:
    > model: opus
    > use: web_search

> use: web_agent
@agent():
    >> answer [x]
"""
    prog = parse_program(src)
    profile = prog.profile_defs["web_agent"]
    assert profile.tool_names == ["web_search"]


def test_use_procedure_over_profile_when_both_exist() -> None:
    src = """
> profile: foo:
    > model: profile-model

@foo():
    = done

> use: foo
@agent():
    >> value [x]
"""
    adapter = _adapter()
    rt = compile_program(parse_program(src), adapter=adapter)
    agent_def = _procedure_def(rt._procedures["agent"])
    assert agent_def.captured_agent_profile.tools == ("foo",)
    assert agent_def.captured_agent_profile.model is None

    rt._invoke_procedure(rt._procedures["agent"], [])
    assert adapter.tool_calls == [("foo",)]


def test_nested_tool_override_and_scope_restore() -> None:
    src = """
@bar():
    = something

> use: bar

@foo():
    @bar():
        = foobar
    > use: bar
    >> inner [x]

= <foo()>
>> outer [y]
"""
    adapter = _adapter()
    rt = compile_program(parse_program(src), adapter=adapter)
    rt.run_main()
    assert len(adapter.tool_calls) == 2
    assert adapter.tool_calls == [("bar",), ("bar",)]


def test_sample_nested_tool_override_uses_inner_then_restores_outer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    src = """@bar():
    = top

> use: bar

@foo():
    @bar():
        = inner
    > use: bar
    >> inner [x]
    = done

[done] = <foo()>
>> outer [y]
"""

    def _named_by_declaration_line(_rt: KediRuntime, proc: ProcedureDef) -> SimpleNamespace:
        assert proc.loc is not None
        return SimpleNamespace(name=f"{proc.name}:{proc.loc.start_line}")

    monkeypatch.setattr(
        "kedi.agent_adapter.utils.kedi_procedure_to_pydantic_tool",
        _named_by_declaration_line,
    )
    adapter = _adapter()
    rt = compile_program(parse_program(src), adapter=adapter)

    rt.run_main()

    assert adapter.tool_calls == [("bar:7",), ("bar:1",)]


def test_profile_body_use_registers_tools() -> None:
    src = """
@web_search(query: str) -> str:
    = <query>

> profile: web_agent:
    > model: opus
    > use:
        web_search

> use: web_agent
@agent():
    >> answer [x]
"""
    adapter = _adapter()
    rt = compile_program(parse_program(src), adapter=adapter)
    agent_def = _procedure_def(rt._procedures["agent"])
    assert agent_def.captured_agent_profile.model == "opus"
    assert agent_def.captured_agent_profile.tools == ("web_search",)

    rt._invoke_procedure(rt._procedures["agent"], [])
    assert adapter.tool_calls == [("web_search",)]
    assert adapter.model_calls == ["opus"]


def test_imported_profile_can_use_prelude_python_callable_tool(tmp_path) -> None:
    profile_path = tmp_path / "filesystem_profile.kedi"
    profile_path.write_text(
        "```\n"
        "def read_file_content(file_path: str) -> str:\n"
        "    return 'content'\n"
        "```\n\n"
        "> profile: filesystem:\n"
        "    > use:\n"
        "        `read_file_content`\n\n"
        "> export:\n"
        "    filesystem\n",
        encoding="utf-8",
    )
    agent_path = tmp_path / "agent.kedi"
    agent_source = (
        "> import: filesystem_profile\n\n"
        "> use: filesystem\n\n"
        ">> Read user-requests.md and provided prettified [content]\n\n"
        "= <content>\n"
    )
    agent_path.write_text(agent_source, encoding="utf-8")

    adapter = _adapter()
    rt = compile_program(
        parse_program(agent_source, source_path=str(agent_path)),
        adapter=adapter,
    )

    assert callable(rt._import_env["read_file_content"])
    assert rt.run_main() == "content-value"
    assert adapter.tool_calls == [("read_file_content",)]


def test_imported_profile_can_use_private_kedi_procedure_tool(tmp_path) -> None:
    profile_path = tmp_path / "filesystem_profile.kedi"
    profile_path.write_text(
        "@read_file_content(file_path: str) -> str:\n"
        "    = <file_path>\n\n"
        "> profile: filesystem:\n"
        "    > use:\n"
        "        read_file_content\n\n"
        "> export:\n"
        "    filesystem\n",
        encoding="utf-8",
    )
    agent_path = tmp_path / "agent.kedi"
    agent_source = (
        "> import: filesystem_profile\n\n"
        "> use: filesystem\n\n"
        ">> Read user-requests.md and provided prettified [content]\n\n"
        "= <content>\n"
    )
    agent_path.write_text(agent_source, encoding="utf-8")

    adapter = _adapter()
    rt = compile_program(
        parse_program(agent_source, source_path=str(agent_path)),
        adapter=adapter,
    )

    assert "read_file_content" in rt._procedures
    assert rt.run_main() == "content-value"
    assert adapter.tool_calls == [("read_file_content",)]


def test_imported_profile_tool_dependency_helper_edges() -> None:
    from kedi.agent_profile.profile import AgentProfile
    from kedi.lang.compiler.program_compiler import ProgramCompiler

    compiler = ProgramCompiler(adapter=None)
    imported_rt = KediRuntime()
    existing = object()
    compiler.rt._import_env["existing"] = existing

    assert compiler._profile_tool_names(AgentProfile(tools=("existing", "missing"))) == (
        "existing",
        "missing",
    )

    compiler._merge_profile_tool_dependencies(
        AgentProfile(tools=("existing", "missing")),
        imported_rt,
    )

    assert compiler.rt._import_env["existing"] is existing
    assert "missing" not in compiler.rt._import_env


def test_unknown_use_target_raises() -> None:
    src = "> use: missing\n@main():\n  >> value [x]\n"
    with pytest.raises(KediExecutionError, match="Unknown use target"):
        compile_program(parse_program(src), adapter=_adapter())


def test_empty_use_directive_raises() -> None:
    from kedi.agent_profile.tools import eval_use_directive
    from kedi.core import KediRuntime
    from kedi.lang.ast import UseDirectiveStmt

    with pytest.raises(KediExecutionError, match="Empty '> use:'"):
        eval_use_directive(KediRuntime(), UseDirectiveStmt(), {}, {}, available_procedures=set())


def test_unknown_tool_procedure_at_runtime_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    src = """
@p():
    = ok
> use: p
>> value [x]
"""
    adapter = _adapter()
    rt = compile_program(parse_program(src), adapter=adapter)
    monkeypatch.setattr("kedi.agent_profile.tools.procedure_def_for_tool", lambda _rt, _name: None)
    with pytest.raises(KediExecutionError, match="Unknown tool procedure"):
        rt.run_main()


def test_adapter_without_tool_support_raises() -> None:
    from kedi.agent_adapter.adapters import DSPyAdapter
    from kedi.agent_profile import AgentToolOverrideUnsupported

    src = """
@tool_proc():
    = ok
> use: tool_proc
>> value [x]
"""
    rt = compile_program(parse_program(src), adapter=DSPyAdapter(model="openai/gpt-4o-mini"))
    with pytest.raises(AgentToolOverrideUnsupported, match="does not support tool registration"):
        rt.run_main()


def test_procedure_def_for_tool_handles_missing_metadata() -> None:
    from kedi.agent_profile.tools import procedure_def_for_tool

    rt = compile_program(parse_program("= ok"))
    assert procedure_def_for_tool(rt, "missing") is None
    rt._procedures["plain"] = lambda: None
    assert procedure_def_for_tool(rt, "plain") is None


def test_pydantic_adapter_tool_scope_registers_toolsets() -> None:
    from unittest.mock import AsyncMock, patch

    from kedi.agent_adapter.adapters import PydanticAdapter
    from kedi.agent_adapter.utils import kedi_procedure_to_pydantic_tool

    src = "@search(query: str) -> str:\n  = <query>\n"
    prog = parse_program(src)
    rt = compile_program(prog, adapter=PydanticAdapter(model="test"))
    tool = kedi_procedure_to_pydantic_tool(rt, prog.procedures[0])
    adapter = PydanticAdapter(model="test")

    fake_run = AsyncMock(return_value=type("R", (), {"output": "ok"})())
    with patch.object(adapter, "run", fake_run):
        with adapter.tool_scope():
            adapter.register_tool(tool)
            import asyncio

            asyncio.run(adapter.produce("go", output_type=str))
    assert fake_run.await_args is not None
    assert fake_run.await_args.kwargs.get("toolsets") is not None
