from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, cast

import pytest
from kedi.agent_adapter import adapters as adapters_module
from kedi.agent_adapter.adapters import DSPyAdapter, PydanticAdapter
from kedi.agent_profile import AgentProfile
from kedi.agent_profile.compile import (
    build_capture_env,
    capture_procedure_profiles,
    effective_profile_registry,
    profile_stmt_to_definition,
)
from kedi.agent_profile.model import (
    AgentProfileOverrideUnsupported,
    _member_to_patch,
    eval_model_value,
    eval_system_value,
    eval_use_profile,
)
from kedi.agent_profile.profile import AgentProfile as AgentProfileType
from kedi.core import KediRuntime
from kedi.lang import compile_program, parse_program
from kedi.lang.ast import (
    InputSegment,
    ModelDirectiveStmt,
    ProcDefStmt,
    ProcedureDef,
    ProfileDefStmt,
    ProfileModelEntry,
    ProfileSystemEntry,
    Program,
    PythonExprSegment,
    Span,
    TemplateLine,
    TextSegment,
    UseDirectiveStmt,
)
from kedi.lang.compiler.statement_compiler import StatementExecutor
from kedi.lang.parser.diagnostics import KediParseError
from kedi.lsp import features
from tests.lsp_helpers import hover_text_at, position_at_line
from tests.mock_adapter import MockAdapter


def test_agent_profile_merge_and_override_flags() -> None:
    assert AgentProfileType.empty().merge(AgentProfileType()).model is None
    assert AgentProfileType(model="a").needs_model_override(None) is True
    assert AgentProfileType(model="a").needs_model_override("a") is False


def test_effective_profile_registry_and_capture_env() -> None:
    rt = KediRuntime()
    assert effective_profile_registry(rt) == {}

    setattr(rt, "_import_env", {"imported": 1})
    env = build_capture_env(rt)
    assert env["imported"] == 1

    rt2 = KediRuntime()
    setattr(rt2, "_import_env", "not-a-dict")
    assert build_capture_env(rt2) == {}


def test_eval_model_value_errors_and_backtick() -> None:
    rt = KediRuntime()

    with pytest.raises(ValueError, match="no value"):
        eval_model_value(rt, plain_value=None, backtick_code=None, env={})

    def _eval_type_int(_rt: KediRuntime, code: str, _env: dict[str, Any]) -> int:
        return 42

    def _eval_type_str(_rt: KediRuntime, code: str, _env: dict[str, Any]) -> str:
        return "gpt"

    rt.executor.evaluate_type_expression = _eval_type_int  # ty: ignore[invalid-assignment]
    with pytest.raises(TypeError, match="must evaluate to str"):
        eval_model_value(rt, plain_value=None, backtick_code="bad", env={})

    rt.executor.evaluate_type_expression = _eval_type_str  # ty: ignore[invalid-assignment]
    assert eval_model_value(rt, plain_value=None, backtick_code="models['x']", env={}) == "gpt"
    assert eval_model_value(rt, plain_value="  haiku  ", backtick_code=None, env={}) == "haiku"


def test_eval_use_profile_unknown_and_bad_member() -> None:
    rt = KediRuntime()
    stmt = UseDirectiveStmt(target="missing")
    with pytest.raises(Exception, match="Unknown profile"):
        eval_use_profile(rt, stmt, {}, {})

    profile_stmt = parse_program("> profile: light:\n    > model: opus\n").profile_defs["light"]
    merged = eval_use_profile(
        rt,
        UseDirectiveStmt(target="light"),
        {"light": profile_stmt_to_definition(profile_stmt)},
        {},
    )
    assert merged.model == "opus"

    with pytest.raises(Exception, match="Unknown profile member kind"):
        _member_to_patch(rt, cast(Any, SimpleNamespace(kind="other")), {})


def test_capture_with_backtick_model_directive() -> None:
    rt = KediRuntime()

    def _eval_type_opus(_rt: KediRuntime, _code: str, _env: dict[str, Any]) -> str:
        return "opus"

    rt.executor.evaluate_type_expression = _eval_type_opus  # ty: ignore[invalid-assignment]
    setattr(rt, "_profile_registry", {})
    prog = Program(
        ordered_toplevel=[
            ModelDirectiveStmt(backtick_code="models['light']"),
            ProcedureDef(name="foo", params=[], body=[]),
        ]
    )
    capture_procedure_profiles(rt, prog)
    captured_proc = prog.ordered_toplevel[1]
    assert isinstance(captured_proc, ProcedureDef)
    assert captured_proc.captured_agent_profile.model == "opus"


def test_procedure_body_use_profile_directive() -> None:
    src = """
> profile: inner:
    > model: inner-model
@foo():
    > use: inner
    >> value [x]
"""
    adapter = MockAdapter(default_model="default")
    rt = compile_program(parse_program(src), adapter=adapter)
    rt._invoke_procedure(rt._procedures["foo"], [])
    assert adapter.model_calls == ["inner-model"]


def test_nested_procedure_registration_branches() -> None:
    rt = KediRuntime()
    rt._procedures = {"inner": lambda: "stale"}
    executor = StatementExecutor(rt)
    saved_procs: dict[str, Callable[[], object]] = {}
    new_proc_names: list[str] = []
    parent = ProcedureDef(name="outer", params=[], body=[])

    inner = ProcedureDef(name="inner", params=[], body=[])
    executor._execute_body_statement(
        ProcDefStmt(proc=inner),
        parent,
        {},
        {},
        saved_procs,
        new_proc_names,
        AgentProfile.empty(),
    )
    assert saved_procs["inner"]() == "stale"

    executor._execute_body_statement(
        ProcDefStmt(proc=None),
        parent,
        {},
        {},
        saved_procs,
        new_proc_names,
        AgentProfile.empty(),
    )

    fresh = ProcedureDef(name="fresh", params=[], body=[])
    executor._execute_body_statement(
        ProcDefStmt(proc=fresh),
        parent,
        {},
        {},
        saved_procs,
        new_proc_names,
        AgentProfile.empty(),
    )
    assert "fresh" in new_proc_names


def test_unsupported_adapter_raises_on_model_override() -> None:
    class BareAdapter:
        def produce_sync(self, *, template: str, output_schema: dict[str, Any], **_: Any) -> Any:
            return SimpleNamespace(x="ok")

    src = "@foo():\n    > model: opus\n    >> value [x]\n"
    rt = compile_program(parse_program(src), adapter=cast(Any, BareAdapter()))
    with pytest.raises(AgentProfileOverrideUnsupported):
        rt._invoke_procedure(rt._procedures["foo"], [])


def test_parse_profile_directive_edge_cases() -> None:
    with pytest.raises(KediParseError):
        parse_program("> profile: first:\n    > model: a\n> profile: first:\n    > model: b\n")

    with pytest.raises(KediParseError):
        parse_program("> profile: empty:\n")


def test_pydantic_adapter_profile_override(monkeypatch: pytest.MonkeyPatch) -> None:
    entered: list[str | None] = []

    class FakeOverride:
        def __init__(self, *, model: str) -> None:
            entered.append(model)

        def __enter__(self) -> None:
            return None

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(
        adapters_module.Agent, "override", lambda self, **kwargs: FakeOverride(**kwargs)
    )

    adapter = object.__new__(PydanticAdapter)
    adapter._model = "openai:gpt-4o-mini"
    with adapter.profile_override(AgentProfile(model="openai:gpt-4o-mini")):
        pass
    assert entered == []

    with adapter.profile_override(AgentProfile(model="anthropic/claude")):
        pass
    assert entered == ["anthropic:claude"]


def test_dspy_adapter_profile_override(monkeypatch: pytest.MonkeyPatch) -> None:
    lm_calls: list[str] = []
    context_calls: list[Any] = []

    class FakeLM:
        def __init__(self, **_: Any) -> None:
            lm_calls.append("created")

    class FakeContext:
        def __init__(self, *, lm: Any) -> None:
            context_calls.append(lm)

        def __enter__(self) -> None:
            return None

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(adapters_module, "LM", FakeLM)
    monkeypatch.setattr(adapters_module.dspy, "context", lambda **kwargs: FakeContext(**kwargs))
    monkeypatch.setattr(
        adapters_module.dspy, "settings", SimpleNamespace(configure=lambda **_: None)
    )

    adapter = object.__new__(DSPyAdapter)
    adapter._lm_model = "openai:gpt-4o-mini"
    adapter._lm_provider = None
    adapter._lm_max_tokens = 1
    adapter._lm_temperature = 0.0
    adapter._lm_extra_kwargs = {}

    with adapter.profile_override(AgentProfile(model="openai:gpt-4o-mini")):
        pass
    assert lm_calls == []

    with adapter.profile_override(AgentProfile(model="openai/gpt-4o")):
        pass
    assert lm_calls == ["created"]
    assert context_calls


def test_lsp_profile_helpers_and_hovers() -> None:
    assert features._directive_name_span(Program(), cast(Any, None), "x") is None
    assert features._directive_name_span(Program(), Span(0, 1, 1, 1, 1, 2), "x") is None

    use_src = "> profile: light:\n    > model: opus\n> use: light\n"
    use_prog = parse_program(use_src)
    use_stmt = next(
        item for item in use_prog.ordered_toplevel if isinstance(item, UseDirectiveStmt)
    )
    use_pos = position_at_line(2, 10)
    resolved = features._resolve_in_stmt(use_prog, use_stmt, use_pos, None, {}, {}, {})
    assert resolved is not None and resolved.kind == "agent_profile"
    assert (
        features._resolve_in_stmt(use_prog, use_stmt, position_at_line(2, 0), None, {}, {}, {})
        is None
    )

    foo_use = parse_program("> use: foo\n").toplevel_body[0]
    assert isinstance(foo_use, UseDirectiveStmt)
    assert foo_use.loc is not None
    assert features._directive_name_span(use_prog, foo_use.loc, "bar") is None

    prog = parse_program("> use: missing\n")
    pos = position_at_line(0, 8)
    sym = features._resolve_symbol_at(prog, pos)
    assert sym is not None and sym.kind == "agent_profile"

    body = hover_text_at("> use: ghost\n", "x.kedi", "ghost")
    assert "definition not found" in body

    src = "> profile: advanced:\n    > model: opus\n\n> export:\n  advanced\n"
    assert "Agent profile" in hover_text_at(src, "x.kedi", "advanced", occurrence=2)

    proc = ProcedureDef(
        name="p",
        params=[],
        captured_agent_profile=AgentProfile(model="opus"),
        loc=Span(0, 1, 1, 1, 1, 2),
    )
    assert "captured model" in features._proc_signature_markdown(proc)
    empty_capture = ProcedureDef(
        name="p",
        params=[],
        captured_agent_profile=AgentProfile(),
        docstring="docs",
        loc=Span(0, 1, 1, 1, 1, 2),
    )
    assert "docs" in features._proc_signature_markdown(empty_capture)
    proc_with_doc = ProcedureDef(
        name="p",
        params=[],
        docstring="docs",
        captured_agent_profile=AgentProfile(model="opus"),
        loc=Span(0, 1, 1, 1, 1, 2),
    )
    assert "docs" in features._proc_signature_markdown(proc_with_doc)

    body_src = "@foo():\n    > use: light\n    >> x [y]\n"
    symbols = features.compute_document_symbols(body_src, "x.kedi")
    foo = next(symbol for symbol in symbols if symbol.name == "foo")
    assert any(child.name.startswith("> use:") for child in foo.children or ())

    profile_span = Span(0, 30, 1, 1, 1, 30)
    member_span = Span(4, 20, 2, 5, 2, 20)
    profile_node = ProfileDefStmt(
        name="p",
        model_members=[
            ProfileModelEntry(plain_value="opus", loc=member_span),
            ProfileModelEntry(backtick_code="dyn", loc=member_span),
        ],
        loc=profile_span,
    )
    profile_prog = Program(
        profile_defs={"p": profile_node},
        ordered_toplevel=[profile_node],
    )
    assert features._smallest_enclosing(profile_prog, position_at_line(0, 12)) is profile_node

    use_hover = features._describe_ast_node(UseDirectiveStmt(target="light"))
    assert use_hover is not None and "> use:" in use_hover
    model_hover = features._describe_ast_node(ModelDirectiveStmt(backtick_code="models['x']"))
    assert model_hover is not None and "models['x']" in model_hover
    assert (
        eval_system_value(KediRuntime(), plain_value="Be concise.", backtick_code=None, env={})
        == "Be concise."
    )

    profile_system_hover = features._describe_ast_node(
        ProfileSystemEntry(
            template_lines=[
                TemplateLine(
                    segments=[
                        TextSegment(text="Use "),
                        InputSegment(name="tone"),
                    ]
                ),
                TemplateLine(
                    segments=[
                        PythonExprSegment(code="args.name"),
                        TextSegment(text=" persona."),
                    ]
                ),
            ]
        )
    )
    assert profile_system_hover is not None
    assert "Use <tone>\\n persona." in profile_system_hover
    profile_system_backtick = features._describe_ast_node(
        ProfileSystemEntry(backtick_code="instructions['short']")
    )
    assert profile_system_backtick is not None
    assert "instructions['short']" in profile_system_backtick

    plain_only = ProfileDefStmt(
        name="plain",
        model_members=[ProfileModelEntry(plain_value="haiku", loc=member_span)],
    )
    profile_hover = features._describe_ast_node(plain_only)
    assert profile_hover is not None and "haiku" in profile_hover
    backtick_only = ProfileDefStmt(
        name="dyn",
        model_members=[ProfileModelEntry(backtick_code="dyn", loc=member_span)],
    )
    assert "dyn" in (features._describe_ast_node(backtick_only) or "")
    sparse_member = ProfileDefStmt(
        name="sparse",
        model_members=[ProfileModelEntry(loc=member_span)],
    )
    assert "(no model members)" in (features._describe_ast_node(sparse_member) or "")

    capture_procedure_profiles(KediRuntime(), Program())
