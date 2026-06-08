from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import BaseModel

from kedi.core import KediRuntime, c, i, o
from kedi.errors import KediTraceFrame
from kedi.lang import compile_program as compile_kedi_program
from kedi.lang import parse_program
from kedi.lang.ast import (
    AssignStmt,
    CallArg,
    CallSegment,
    InputSegment,
    LiteralArg,
    OutputSegment,
    ProcDefStmt,
    ProcedureDef,
    Program,
    PythonBlock,
    PythonExprNativeSegment,
    PythonExprSegment,
    ReturnBlockStmt,
    ReturnStmt,
    Segment,
    Span,
    Stmt,
    TemplateLine,
    TextSegment,
    TypeDefStmt,
    VarArg,
)
from kedi.lang.compiler import environment as compiler_env
from kedi.lang.compiler.environment import build_python_env, build_type_env, lookup_value
from kedi.lang.compiler.evaluation import (
    _evaluate_arg,
    _stringify_segments,
    evaluate_assignment_rhs,
    prepare_segments,
    render_expression,
)
from kedi.lang.compiler.procedure_compiler import ProcedureCompiler
from kedi.lang.compiler.program_compiler import ProgramCompiler
from kedi.lang.compiler.statement_compiler import StatementCompiler, StatementExecutor
from kedi.lang.parser.escapes import (
    _indent_width,
    dedent_block,
    preserve_escaped_whitespace,
    unescape_call_arg_text,
    unescape_text_with_tracking,
)
from kedi.source_loading import CompositeSourceMap, SourceDocument


class Person(BaseModel):
    name: str


def test_environment_lookup_and_type_layers(monkeypatch: pytest.MonkeyPatch) -> None:
    rt = KediRuntime()
    token = rt._current_env.set({"current": "value"})
    try:
        assert lookup_value(rt, "current", {}) == "value"
    finally:
        rt._current_env.reset(token)

    setattr(rt, "_globals_env", {"global_name": "global"})
    setattr(rt, "_prelude_env", {"prelude_name": "prelude"})
    setattr(rt, "_import_env", {"import_name": "import"})

    assert lookup_value(rt, "global_name", {}) == "global"
    assert lookup_value(rt, "prelude_name", {}) == "prelude"
    assert lookup_value(rt, "import_name", {}) == "import"

    class RaisingCurrentEnv:
        def get(self) -> dict[str, Any]:
            raise RuntimeError("context unavailable")

    broken_rt = KediRuntime()
    cast(Any, broken_rt)._current_env = RaisingCurrentEnv()
    assert build_python_env(broken_rt, {"local": 1})["local"] == 1
    assert "List" in build_type_env(broken_rt, None)
    with pytest.raises(KeyError, match="Unknown variable"):
        lookup_value(broken_rt, "missing", {})

    monkeypatch.setattr(compiler_env.typing, "__all__", ("MissingTypingName",), raising=False)
    assert "MissingTypingName" not in build_type_env(KediRuntime(), None)

    monkeypatch.setattr(compiler_env.typing, "__all__", ("_PrivateTypingName",), raising=False)
    assert "_PrivateTypingName" not in build_type_env(KediRuntime(), None)


def test_build_python_env_wraps_procedures_and_coerces_model_values() -> None:
    rt = KediRuntime()

    def add_one(value: int) -> int:
        return value + 1

    rt.procedure(add_one)

    local_env: dict[str, Any] = {"person": '{"name": "Ada"}'}
    setattr(rt, "_type_env_stack", [{"person": Person}])
    env = build_python_env(rt, local_env)

    assert env["add_one"](2) == 3
    assert isinstance(local_env["person"], Person)
    assert local_env["person"].name == "Ada"

    non_model_env: dict[str, Any] = {"thing": "value"}
    setattr(rt, "_type_env_stack", [{"thing": object()}])
    assert build_python_env(rt, non_model_env)["thing"] is not non_model_env["thing"]
    assert non_model_env["thing"] == "value"


def test_prepare_segments_assignment_rhs_and_argument_variants() -> None:
    rt = KediRuntime()

    @rt.procedure
    def echo(value: str) -> str:
        return f"echo:{value}"

    expr, rendered = prepare_segments(
        rt,
        [
            TextSegment(text="Hello "),
            InputSegment(name="name"),
            PythonExprSegment(code="'!'"),
        ],
        {"name": "Ada"},
        context="return",
    )
    assert expr == ["Hello ", "!"]
    assert rendered == "Hello Ada!"

    native, rendered_native = evaluate_assignment_rhs(
        rt,
        [TextSegment(text=" "), PythonExprNativeSegment(code="x + 1")],
        {"x": 2},
    )
    assert native == 3
    assert rendered_native == "3"

    called, rendered_called = evaluate_assignment_rhs(
        rt,
        [CallSegment(name="echo", args=[LiteralArg(value="Ada")])],
        {},
    )
    assert called == "echo:Ada"
    assert rendered_called == "echo:Ada"

    native_arg = LiteralArg(value="")
    setattr(native_arg, "__native_backtick__", "x + 2")
    render_arg = LiteralArg(value="")
    setattr(
        render_arg,
        "__render_segments__",
        [TextSegment(text=" "), InputSegment(name="name")],
    )
    call_arg = LiteralArg(value="")
    setattr(
        call_arg,
        "__call_segment__",
        CallSegment(name="echo", args=[LiteralArg(value="nested")]),
    )
    python_arg = LiteralArg(value="")
    setattr(python_arg, "__python_segment__", PythonExprSegment(code="x + 3"))

    assert _evaluate_arg(rt, native_arg, {"x": 1}) == 3
    assert _evaluate_arg(rt, render_arg, {"name": "Ada"}) == "Ada"
    assert _evaluate_arg(rt, call_arg, {}) == "echo:nested"
    assert _evaluate_arg(rt, python_arg, {"x": 1}) == 4
    assert _evaluate_arg(rt, VarArg(name="name"), {"name": "Ada"}) == "Ada"
    with pytest.raises(ValueError, match="Unknown argument type"):
        _evaluate_arg(rt, CallArg(), {})

    assert prepare_segments(rt, [Segment()], {}, context="template") == ([], "")


def test_stringify_output_segments_uses_current_env_and_empty_fallback() -> None:
    rt = KediRuntime()
    token = rt._current_env.set({"answer": "from-env"})
    try:
        assert _stringify_segments(rt, [OutputSegment(name="answer")], {}, context="return") == (
            "from-env"
        )
    finally:
        rt._current_env.reset(token)

    assert _stringify_segments(rt, [OutputSegment(name="missing")], {}, context="return") == ""
    assert _stringify_segments(rt, [PythonExprSegment(code="'py'")], {}, context="return") == "py"
    assert _stringify_segments(rt, [Segment()], {}, context="return") == ""

    def fake_m(expressions: object) -> dict[str, Any]:
        return {}

    cast(Any, rt).m = fake_m
    assert render_expression(rt, [OutputSegment(name="missing")], {}, context="template") == ""


def test_escape_helpers_cover_error_whitespace_and_dedent_edges() -> None:
    assert unescape_call_arg_text(r"a\,b") == "a,b"
    assert preserve_escaped_whitespace("") == ""
    assert preserve_escaped_whitespace(" ", {99}) == ""
    assert preserve_escaped_whitespace("  x  ", {1}) == " x"
    assert _indent_width("") == 0
    assert dedent_block(["", ""]) == ["", ""]
    assert dedent_block(["alpha"]) == ["alpha"]
    assert dedent_block(["\talpha", "    beta"]) == ["alpha", "beta"]

    unescaped, positions = unescape_text_with_tracking("a\\t\\n\\s\\~")
    assert unescaped == "a\t\n ~"
    assert positions == {1, 2, 3}

    with pytest.raises(ValueError, match="Literal '='"):
        unescape_text_with_tracking("a=b")
    with pytest.raises(ValueError, match="Literal '~'"):
        unescape_text_with_tracking("a~b")
    with pytest.raises(ValueError, match="Invalid escape sequence"):
        unescape_text_with_tracking(r"a\q")


def test_runtime_core_edge_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    rt = KediRuntime()

    @rt.procedure
    def repeated() -> str:
        return "first"

    @rt.procedure
    def repeated() -> str:  # type: ignore[no-redef]  # noqa: F811
        return "second"

    @rt.procedure
    def add_one(value: int) -> int:
        return value + 1

    assert rt._get_proc("repeated")() == "second"
    with pytest.raises(RuntimeError, match="outside of a kedi procedure"):
        rt.m(["prompt"])
    with pytest.raises(KeyError, match="Value not found"):
        rt._evaluate_expression(i("missing"), {})
    assert rt._evaluate_expression(c("add_one", i("value")), {"value": 1}) == (
        "2",
        None,
    )
    assert rt._evaluate_expression(o("answer"), {})[0] == "[answer]"
    assert rt._evaluate_expression(cast(Any, object()), {})[1] is None
    with pytest.raises(KeyError, match="Value not found"):
        rt._evaluate_value(i("missing"), {})
    with pytest.raises(ValueError, match="Output expressions"):
        rt._evaluate_value(o("answer"), {})

    frame = KediTraceFrame(
        kind="test",
        label="test",
        span=Span(0, 1, 1, 1, 1, 2, source_path="sample.kedi"),
    )
    assert rt._hydrate_frame(frame).source_snippet is None

    def fake_evaluate_expression(expr: object, env: dict[str, Any]) -> tuple[None, None]:
        return None, None

    monkeypatch.setattr(rt, "_evaluate_expression", fake_evaluate_expression)
    assert rt._kedi_run(["ignored"], {}) == {}


def test_program_compiler_private_helpers_and_inline_toplevel(tmp_path: Path) -> None:
    compiler = ProgramCompiler(adapter=None)
    rt = compiler.rt

    rt._procedures["visible"] = lambda: "ok"
    rt._procedures["_hidden"] = lambda: "hidden"
    setattr(rt, "_type_env_stack", None)
    assert compiler._public_runtime_names(rt) == ["visible"]

    setattr(rt, "_type_env_stack", [{"TypeName": object(), "_PrivateType": object()}])
    setattr(rt, "_globals_env", {"value": 1, "_secret": 2})
    public = compiler._public_runtime_names(rt)
    assert public == ["visible", "TypeName", "value"]

    imported_rt = KediRuntime()
    setattr(imported_rt, "_type_env_stack", None)
    setattr(imported_rt, "_globals_env", {"answer": 42})
    assert compiler._resolve_export("answer", imported_rt, tmp_path / "mod.kedi") == (
        "value",
        42,
    )

    def imported_fn(x: int = 1) -> int:
        return x + 1

    imported_rt.procedure(imported_fn)
    wrapper = compiler._wrap_imported_procedure("imported_fn", imported_fn, imported_rt)
    assert wrapper() == 2
    assert wrapper(4) == 5

    source_path = tmp_path / "module.kedi"
    source_path.write_text("= ok", encoding="utf-8")
    prog = Program(
        source_map=CompositeSourceMap(
            (SourceDocument(path=str(source_path), text="= ok", start_line_in_merged=1),)
        )
    )
    assert compiler._program_path(prog) == source_path.resolve()

    inline_rt = compile_kedi_program(parse_program("= `1 + 1`\n"), adapter=None)
    assert inline_rt.run_main() == "2"


def test_procedure_compiler_type_and_environment_edges() -> None:
    rt = KediRuntime()
    compiler = ProcedureCompiler(rt)

    assert compiler._evaluate_param_defaults(
        ProcedureDef(name="defaulted", params=["x"], param_defaults={"x": "1 + 1"}),
        None,
    ) == {"x": 2}

    proc = ProcedureDef(name="typed", params=["x"], param_type_annotations={"x": "int"})
    with pytest.raises(TypeError, match="Missing parameter"):
        compiler._enforce_param_types(proc, {})
    with pytest.raises(TypeError, match="does not match annotated type"):
        compiler._enforce_param_types(proc, {"x": "not-int"})
    kwargs: dict[str, Any] = {"x": 2}
    compiler._enforce_param_types(proc, kwargs)
    assert kwargs["x"] == 2
    mixed_kwargs: dict[str, Any] = {"x": 2, "y": "kept"}
    compiler._enforce_param_types(
        ProcedureDef(
            name="mixed",
            params=["x", "y"],
            param_type_annotations={"x": "int", "y": None},
        ),
        mixed_kwargs,
    )
    assert mixed_kwargs == {"x": 2, "y": "kept"}

    with pytest.raises(TypeError, match="Return value does not match annotated type"):
        compiler._enforce_return_type(
            ProcedureDef(name="ret", params=[], return_type_annotation="int"),
            "not-int",
            {},
        )
    with pytest.raises(TypeError, match="Return value must be str"):
        compiler._enforce_return_type(ProcedureDef(name="ret", params=[]), 1, {})

    setattr(rt, "_import_env", {"imported": "from import"})
    setattr(rt, "_globals_env", {"global_value": "from global"})
    token = rt._current_env.set(cast(Any, "not a dict"))
    try:
        prepared = compiler._prepare_kwargs(
            ProcedureDef(name="env", params=[]),
            {"local": "value"},
            {"lexical": "parent", "local": "parent local"},
        )
    finally:
        rt._current_env.reset(token)

    assert prepared == {
        "lexical": "parent",
        "local": "value",
        "imported": "from import",
        "global_value": "from global",
    }

    current: dict[str, Any] = {}
    token = rt._current_env.set(current)
    try:
        compiler._prepare_kwargs(ProcedureDef(name="env", params=[]), {}, None)
    finally:
        rt._current_env.reset(token)
    assert current == {"imported": "from import", "global_value": "from global"}


def test_nested_procedure_scope_restores_saved_and_new_names() -> None:
    rt = KediRuntime()
    compiler = ProcedureCompiler(rt)

    def old_child():
        return "old"

    rt._procedures["child"] = old_child

    child = ProcedureDef(
        name="child",
        params=[],
        body=[ReturnStmt(segments=[TextSegment(text="new")])],
    )
    new_child = ProcedureDef(
        name="new_child",
        params=[],
        body=[ReturnStmt(segments=[TextSegment(text="fresh")])],
    )
    outer = ProcedureDef(
        name="outer",
        params=[],
        body=[
            ReturnStmt(segments=[TextSegment(text="ignored")]),
            ProcDefStmt(proc=None),
            ProcDefStmt(proc=child),
            ProcDefStmt(proc=new_child),
            ProcDefStmt(proc=child),
        ],
    )

    with compiler._nested_procedure_scope(outer, {"captured": "value"}):
        assert rt._procedures["child"] is not old_child
        assert rt._procedures["child"]() == "new"
        assert rt._procedures["new_child"]() == "fresh"
        assert child.lexical_env == {"captured": "value"}
        assert new_child.lexical_env == {"captured": "value"}
        del rt._procedures["new_child"]

    assert rt._procedures["child"] is old_child
    assert "new_child" not in rt._procedures


def test_nested_procedure_scope_swallows_cleanup_delete_errors() -> None:
    class UndeletableProcedures(dict[str, Any]):
        def __delitem__(self, key: str) -> None:
            if key == "new_child":
                raise RuntimeError("cannot delete")
            super().__delitem__(key)

    rt = KediRuntime()
    rt._procedures = UndeletableProcedures()
    compiler = ProcedureCompiler(rt)
    new_child = ProcedureDef(
        name="new_child",
        params=[],
        body=[ReturnStmt(segments=[TextSegment(text="x")])],
    )
    outer = ProcedureDef(name="outer", params=[], body=[ProcDefStmt(proc=new_child)])

    with compiler._nested_procedure_scope(outer, {}):
        assert "new_child" in rt._procedures

    assert "new_child" in rt._procedures


def test_statement_executor_cleanup_skips_missing_new_proc_name() -> None:
    class MissingDuringCleanup(dict[str, Any]):
        def __contains__(self, key: object) -> bool:
            if key == "nested_b":
                return False
            return super().__contains__(key)

    rt = KediRuntime()
    rt._procedures = MissingDuringCleanup()
    executor = StatementExecutor(rt)
    executor.execute(
        ProcedureDef(
            name="outer",
            params=[],
            body=[
                ProcDefStmt(proc=ProcedureDef(name="nested_b", params=[], body=[])),
                ProcDefStmt(proc=ProcedureDef(name="nested_c", params=[], body=[])),
            ],
        ),
        {},
    )

    assert "nested_c" not in rt._procedures


def test_statement_compiler_assignment_template_and_single_statement_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kedi.lang.compiler import statement_compiler as statement_compiler_module

    rt = KediRuntime()
    compiler = StatementCompiler(rt)

    template = TemplateLine(
        segments=[TextSegment(text="ignored")],
        optimized_output_override={"answer": "ok"},
    )
    with pytest.raises(RuntimeError, match="active runtime environment"):
        compiler.compile_template_line(template)()

    env: dict[str, Any] = {}
    token = rt._current_env.set(env)
    try:
        assert compiler.compile_template_line(template)() == {"answer": "ok"}
    finally:
        rt._current_env.reset(token)
    assert env["answer"] == "ok"

    class RecordingRuntime(KediRuntime):
        def __init__(self) -> None:
            super().__init__()
            self.expressions: list[object] = []

        def m(self, expressions: object) -> dict[str, Any]:
            self.expressions.append(expressions)
            return {"done": "yes"}

    recording = RecordingRuntime()
    prefixed = TemplateLine(
        segments=[TextSegment(text="seed")],
        optimized_prompt_prefix="better",
    )
    assert StatementCompiler(recording).compile_template_line(prefixed)() == {"done": "yes"}
    assert recording.expressions == [["better\n\n", "seed"]]

    closure: dict[str, Any] = {"x": 1}
    compiler.execute_assignment(
        AssignStmt(name="angle", expr=[PythonExprSegment(code="x + 1")]),
        closure,
    )
    assert closure["angle"] == "2"

    monkeypatch.setattr(
        statement_compiler_module,
        "evaluate_assignment_rhs",
        lambda rt_arg, segments, env_arg: (2, "2"),
    )
    monkeypatch.setattr(
        statement_compiler_module,
        "detect_rhs_kind",
        lambda segments: (False, True, False),
    )
    compiler.execute_assignment(AssignStmt(name="native_angle", expr=[]), closure)
    assert closure["native_angle"] == "2"

    monkeypatch.setattr(
        statement_compiler_module,
        "detect_rhs_kind",
        lambda segments: (False, False, False),
    )
    compiler.execute_assignment(AssignStmt(name="native_rendered", expr=[]), closure)
    assert closure["native_rendered"] == "2"

    monkeypatch.setattr(
        statement_compiler_module,
        "evaluate_assignment_rhs",
        lambda rt_arg, segments, env_arg: (1, "not-int"),
    )
    monkeypatch.setattr(
        statement_compiler_module,
        "detect_rhs_kind",
        lambda segments: (False, False, False),
    )
    with pytest.raises(TypeError, match="Native assignment value does not match annotated type"):
        compiler.execute_assignment(
            AssignStmt(
                name="bad",
                expr=[TextSegment(text=" "), PythonExprNativeSegment(code="1")],
                type_annotation="int",
            ),
            closure,
        )
    monkeypatch.undo()

    executor = StatementExecutor(rt)
    nested_a = ProcedureDef(name="nested_a", params=[], body=[])
    nested_b = ProcedureDef(name="nested_b", params=[], body=[])
    nested_c = ProcedureDef(name="nested_c", params=[], body=[])
    rt._procedures["nested_a"] = lambda: "old"
    executor.execute(
        ProcedureDef(
            name="with_edges",
            params=[],
            body=[
                TypeDefStmt(typedef=None),
                ProcDefStmt(proc=nested_a),
                ProcDefStmt(proc=nested_a),
                ProcDefStmt(proc=nested_b),
                ProcDefStmt(proc=nested_c),
                ReturnBlockStmt(code="'middle'"),
                Stmt(),
                ReturnStmt(segments=[TextSegment(text="last")]),
            ],
        ),
        {},
    )
    assert "nested_b" not in rt._procedures
    assert "nested_c" not in rt._procedures
    top_proc = ProcedureDef(
        name="top_proc",
        params=[],
        body=[ReturnStmt(segments=[TextSegment(text="ok")])],
    )
    executor.execute_single_statement(ProcDefStmt(proc=top_proc), {}, is_toplevel=True)
    assert rt._get_proc("top_proc")() == "ok"
    executor.execute_single_statement(ProcDefStmt(proc=top_proc), {}, is_toplevel=False)

    side_effect_env: dict[str, Any] = {"x": 0}
    executor.execute_single_statement(PythonBlock(code="x = 1"), side_effect_env)
    assert side_effect_env["x"] == 1

    single_env: dict[str, Any] = {}
    executor.execute_single_statement(
        AssignStmt(name="value", expr=[TextSegment(text="ok")]), single_env
    )
    assert single_env["value"] == "ok"
    executor.execute_single_statement(ReturnStmt(segments=[TextSegment(text="ignored")]), {})


def test_procedure_kwargs_skip_non_dict_runtime_envs() -> None:
    rt = compile_kedi_program(parse_program("@p():\n  = ok\n"), adapter=None)
    setattr(rt, "_import_env", object())
    setattr(rt, "_globals_env", object())

    assert rt._get_proc("p")() == "ok"
