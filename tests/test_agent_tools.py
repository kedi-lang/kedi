from __future__ import annotations

from contextvars import ContextVar
from types import SimpleNamespace
from typing import TYPE_CHECKING, Annotated, Any, cast

import pytest

from kedi.agent_adapter.tools.conversion import kedi_procedure_tool_spec
from kedi.agent_adapter.tools.scope import ToolScope
from kedi.agent_adapter.tools.specs import ToolSpec
from kedi.agent_profile.use_resolution import eval_use_directive
from kedi.core import KediRuntime
from kedi.errors import KediExecutionError
from kedi.lang import compile_program, parse_program
from kedi.lang.ast import ProcedureDef, UseDirectiveStmt
from tests.mock_adapter import MockAdapter

if TYPE_CHECKING:
    from pydantic_ai.tools import Tool


def _adapter() -> MockAdapter:
    return MockAdapter(default_model="default-model")


def _set_empty_profile_stack(adapter: object) -> None:
    setattr(
        adapter,
        "_profile_stack_var",
        ContextVar(f"test_profile_stack_{id(adapter)}", default=()),
    )


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
        available_procedures={"alpha", "beta"},
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
    assert profile.use_targets == ["web_search"]


def test_profile_body_single_line_use_composes_profile() -> None:
    src = """
> profile: filesystem:
    > instructions: Filesystem access.

> profile: orchestrator:
    > use: filesystem
    > model: opus

> use: orchestrator
@agent():
    >> answer [x]
"""
    adapter = _adapter()
    rt = compile_program(parse_program(src), adapter=adapter)
    agent_def = _procedure_def(rt._procedures["agent"])
    assert agent_def.captured_agent_profile.system == "Filesystem access."
    assert agent_def.captured_agent_profile.model == "opus"


def test_profile_body_use_procedure_over_profile_when_both_exist() -> None:
    src = """
> profile: lookup:
    > model: profile-model

@lookup():
    = done

> profile: orchestrator:
    > use: lookup

> use: orchestrator
@agent():
    >> answer [x]
"""
    adapter = _adapter()
    rt = compile_program(parse_program(src), adapter=adapter)
    agent_def = _procedure_def(rt._procedures["agent"])
    assert agent_def.captured_agent_profile.tools == ("lookup",)
    assert agent_def.captured_agent_profile.model is None


def test_profile_body_use_rejects_cycles_and_unknown_targets() -> None:
    cyclic = """
> profile: first:
    > use: second

> profile: second:
    > use: first

> use: first
@main():
    = ok
"""
    with pytest.raises(Exception, match="Cyclic profile use reference"):
        compile_program(parse_program(cyclic), adapter=_adapter())

    unknown = """
> profile: agent:
    > use: missing

> use: agent
@main():
    = ok
"""
    with pytest.raises(Exception, match="Unknown use target 'missing' in profile 'agent'"):
        compile_program(parse_program(unknown), adapter=_adapter())


def test_profile_body_skills_directive_enables_skills() -> None:
    src = """
> profile: agent:
    > skills: enabled

> use: agent
@main():
    >> answer [x]
"""
    rt = compile_program(parse_program(src), adapter=_adapter())
    main = _procedure_def(rt._procedures["main"])
    assert main.captured_agent_profile.skills is not None
    assert main.captured_agent_profile.skills.enabled is True


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
    assert adapter.tool_calls == [
        (
            "foo",
            "read_artifact",
            "search_artifacts",
            "release_artifact",
            "run_artifact_code",
        )
    ]


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

> show: <foo()>
>> outer [y]
"""
    adapter = _adapter()
    rt = compile_program(parse_program(src), adapter=adapter)
    rt.run_main()
    assert len(adapter.tool_calls) == 2
    assert adapter.tool_calls == [
        ("bar", "read_artifact", "search_artifacts", "release_artifact", "run_artifact_code"),
        ("bar", "read_artifact", "search_artifacts", "release_artifact", "run_artifact_code"),
    ]


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

    class LocAwareAdapter(MockAdapter):
        def register_tool(self, tool: ToolSpec) -> None:
            proc = tool.proc
            name = f"{proc.name}:{proc.loc.start_line}" if proc and proc.loc else tool.name
            self._tool_scope.register_tool(SimpleNamespace(name=name))

    adapter = LocAwareAdapter(default_model="default-model")
    rt = compile_program(parse_program(src), adapter=adapter)

    rt.run_main()

    # Each prompt is scoped to exactly its own tool (inner bar:7 for foo, outer bar:1 for the
    # top level). Compared order-independently: under the parallel engine these two independent
    # templates may execute in either order, but each must still see only its correct tool.
    artifact_tools = (
        "read_artifact",
        "search_artifacts",
        "release_artifact",
        "run_artifact_code",
    )
    assert sorted(adapter.tool_calls) == [
        ("bar:1", *artifact_tools),
        ("bar:7", *artifact_tools),
    ]


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
    assert adapter.tool_calls == [
        (
            "web_search",
            "read_artifact",
            "search_artifacts",
            "release_artifact",
            "run_artifact_code",
        )
    ]
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

    assert "read_file_content" not in rt._import_env
    assert rt.run_main() == "content-value"
    assert adapter.tool_calls == [
        (
            "read_file_content",
            "read_artifact",
            "search_artifacts",
            "release_artifact",
            "run_artifact_code",
        )
    ]


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

    assert "read_file_content" not in rt._procedures
    assert rt.run_main() == "content-value"
    assert adapter.tool_calls == [
        (
            "read_file_content",
            "read_artifact",
            "search_artifacts",
            "release_artifact",
            "run_artifact_code",
        )
    ]


def test_imported_profile_can_compose_private_profile(tmp_path) -> None:
    profile_path = tmp_path / "filesystem_profile.kedi"
    profile_path.write_text(
        "@read_file_content(file_path: str) -> str:\n"
        "    = <file_path>\n\n"
        "> profile: readonlyfs:\n"
        "    > use: read_file_content\n\n"
        "> profile: filesystem:\n"
        "    > use: readonlyfs\n\n"
        "> export:\n"
        "    filesystem\n",
        encoding="utf-8",
    )
    agent_path = tmp_path / "agent.kedi"
    agent_source = (
        "> import: filesystem_profile\n\n"
        "> use: filesystem\n\n"
        ">> Read user-requests.md and provide [content]\n\n"
        "= <content>\n"
    )
    agent_path.write_text(agent_source, encoding="utf-8")

    adapter = _adapter()
    rt = compile_program(
        parse_program(agent_source, source_path=str(agent_path)),
        adapter=adapter,
    )

    assert rt.run_main() == "content-value"
    assert adapter.tool_calls == [
        (
            "read_file_content",
            "read_artifact",
            "search_artifacts",
            "release_artifact",
            "run_artifact_code",
        )
    ]


def test_imported_profile_reports_unknown_composed_target(tmp_path) -> None:
    module_path = tmp_path / "profiles.kedi"
    module_path.write_text(
        "> profile: outer:\n    > use: missing\n\n> export:\n    outer\n",
        encoding="utf-8",
    )
    main_path = tmp_path / "main.kedi"
    source = "> import: profiles\n\n> use: outer\n\n@main():\n    = ok\n"
    main_path.write_text(source, encoding="utf-8")

    with pytest.raises(Exception, match="Unknown use target 'missing' in profile 'outer'"):
        compile_program(
            parse_program(source, source_path=str(main_path)),
            adapter=_adapter(),
        )


def test_imported_profile_tool_binding_does_not_publish_or_rebind_dependency() -> None:
    from kedi.agent_profile.definitions import ProfileDefinition, ProfileToolMember
    from kedi.lang.compiler.program_compiler import ProgramCompiler

    compiler = ProgramCompiler(adapter=None)
    imported_rt = KediRuntime()
    imported_rt._prelude_env = {
        "private_lookup": lambda: "private",
    }
    compiler.rt._import_env["private_lookup"] = lambda: "caller"
    definition = ProfileDefinition(
        name="agent",
        members=[ProfileToolMember(tool_names=("private_lookup", "missing"))],
    )

    bound = compiler._bind_imported_profile(definition, imported_rt)

    assert isinstance(bound, ProfileDefinition)
    assert set(bound.tool_bindings) == {"private_lookup"}
    assert bound.tool_bindings["private_lookup"].fn() == "private"
    assert compiler.rt._import_env["private_lookup"]() == "caller"


def test_unknown_use_target_raises() -> None:
    src = "> use: missing\n@main():\n  >> value [x]\n"
    with pytest.raises(KediExecutionError, match="Unknown use target"):
        compile_program(parse_program(src), adapter=_adapter())


def test_empty_use_directive_raises() -> None:
    from kedi.agent_profile.use_resolution import eval_use_directive
    from kedi.core import KediRuntime
    from kedi.lang.ast import UseDirectiveStmt

    with pytest.raises(KediExecutionError, match="Empty '> use:'"):
        eval_use_directive(KediRuntime(), UseDirectiveStmt(), {}, {}, available_procedures=set())


def test_use_directive_accepts_runtime_global_python_callable() -> None:
    from kedi.agent_profile.use_resolution import eval_use_directive
    from kedi.core import KediRuntime
    from kedi.lang.ast import UseDirectiveStmt

    def lookup() -> str:
        return "ok"

    rt = KediRuntime()
    rt.set_initial_globals({"lookup": lookup})

    profile = eval_use_directive(
        rt,
        UseDirectiveStmt(target="lookup"),
        {},
        {},
        available_procedures=set(),
    )

    assert profile.tools == ("lookup",)


def test_unknown_tool_procedure_at_runtime_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    src = """
@p():
    = ok
> use: p
>> value [x]
"""
    adapter = _adapter()
    rt = compile_program(parse_program(src), adapter=adapter)
    monkeypatch.setattr(
        "kedi.agent_profile.use_resolution.procedure_def_for_tool", lambda _rt, _name: None
    )
    with pytest.raises(KediExecutionError, match="Unknown tool procedure"):
        rt.run_main()


def test_tool_use_with_adapter_without_tool_support_raises() -> None:
    class NoToolAdapter:
        async def produce(self, *_args: Any, **_kwargs: Any) -> SimpleNamespace:
            return SimpleNamespace(x="unreachable")

        def produce_sync(self, *_args: Any, **_kwargs: Any) -> SimpleNamespace:
            return SimpleNamespace(x="unreachable")

    src = """
@p():
    = ok
> use: p
>> value [x]
"""
    rt = compile_program(parse_program(src), adapter=NoToolAdapter())
    with pytest.raises(KediExecutionError, match="does not support tool registration"):
        rt.run_main()


def test_procedure_def_for_tool_handles_missing_metadata() -> None:
    from kedi.agent_profile.use_resolution import procedure_def_for_tool

    rt = compile_program(parse_program("= ok"))
    assert procedure_def_for_tool(rt, "missing") is None
    rt._procedures["plain"] = lambda: None
    assert procedure_def_for_tool(rt, "plain") is None


def test_python_callable_for_tool_skips_non_dict_envs() -> None:
    from kedi.agent_profile.use_resolution import python_callable_for_tool

    def tool() -> str:
        return "ok"

    rt = compile_program(parse_program("= ok"))
    cast(Any, rt)._import_env = object()
    cast(Any, rt)._prelude_env = object()
    cast(Any, rt)._globals_env = object()
    rt._initial_globals_env = {"tool": tool}

    assert python_callable_for_tool(rt, "tool") is tool
    assert python_callable_for_tool(rt, "missing") is None


def test_pydantic_adapter_tool_scope_registers_toolsets() -> None:
    import asyncio

    from pydantic_ai.models.test import TestModel

    from kedi.agent_adapter.adapters import PydanticAdapter

    src = "@search(query: str) -> str:\n  = <query>\n"
    prog = parse_program(src)
    rt = compile_program(prog, adapter=PydanticAdapter(model="test"))
    tool = kedi_procedure_tool_spec(rt, prog.procedures[0])
    model = TestModel()
    adapter = PydanticAdapter(model=model)

    with adapter.tool_scope():
        adapter.register_tool(tool)
        asyncio.run(adapter.produce("go", output_type=str))

    assert model.last_model_request_parameters is not None
    assert [tool.name for tool in model.last_model_request_parameters.function_tools] == ["search"]


def test_pydantic_adapter_output_schema_uses_annotated_description() -> None:
    from kedi.agent_adapter.adapters import PydanticAdapter

    model = PydanticAdapter.type_builder(
        {"capital": Annotated[str, "Canonical city name returned by the model."]}
    )

    schema = model.model_json_schema()
    assert schema["properties"]["capital"]["type"] == "string"
    assert (
        schema["properties"]["capital"]["description"]
        == "Canonical city name returned by the model."
    )
    assert schema["required"] == ["capital"]

    ignored = PydanticAdapter.type_builder(
        {"capital": Annotated[str, int, "ignored description"]}
    ).model_json_schema()
    assert "description" not in ignored["properties"]["capital"]
