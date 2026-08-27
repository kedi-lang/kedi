from __future__ import annotations

import builtins
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic import BaseModel

from kedi.core import KediRuntime, c, i, o
from kedi.errors import KediTraceFrame
from kedi.lang import compile_program as compile_kedi_program
from kedi.lang import parse_program
from kedi.lang.ast import (
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
    VariableInitStmt,
)
from kedi.lang.compiler import environment as compiler_env
from kedi.lang.compiler.environment import (
    BindingMetadata,
    KediEnv,
    ScopeFrame,
    assert_not_cli_args_identifier,
    build_python_env,
    build_type_env,
    cli_args_identifier,
    lookup_value,
    write_owned_value,
)
from kedi.lang.compiler.evaluation import (
    _evaluate_arg,
    _stringify_segments,
    evaluate_binding_rhs,
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
from kedi.utils import AttrDict


class Person(BaseModel):
    name: str


def test_kedienv_reserves_configured_cli_args_identifier(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(compiler_env, "CLI_ARGS_IDENTIFIER", "cli_args")
    cli_args = SimpleNamespace(name="Ada")
    env = KediEnv()
    env._set_cli_args(cli_args)

    assert cli_args_identifier() == "cli_args"
    assert env["cli_args"] is cli_args
    assert_not_cli_args_identifier("ordinary")

    for mutate in (
        lambda: env.__setitem__("cli_args", "override"),
        lambda: env.update({"cli_args": "override"}),
        lambda: env.update(cli_args="override"),
        lambda: env.setdefault("cli_args", "override"),
        lambda: env.__ior__({"cli_args": "override"}),
        lambda: env.__or__({"cli_args": "override"}),
        lambda: KediEnv({"ordinary": "value"}).__ror__({"cli_args": "override"}),
        lambda: KediEnv.fromkeys(["cli_args"], "override"),
        lambda: env.__delitem__("cli_args"),
        lambda: env.pop("cli_args"),
        env.popitem,
        env.clear,
    ):
        with pytest.raises(ValueError, match="Reserved identifier|reserved CLI"):
            mutate()

    with pytest.raises(ValueError, match="Reserved identifier"):
        KediEnv({"cli_args": "override"})

    env["args"] = "ordinary value"
    assert env["args"] == "ordinary value"

    rt = KediRuntime()
    rt.set_initial_globals({"cli_args": cli_args})
    assert build_python_env(rt, {})["cli_args"] is cli_args
    with pytest.raises(ValueError, match="Reserved identifier"):
        build_python_env(rt, {"cli_args": "override"})


@pytest.mark.parametrize("identifier", [None, "not-valid", "class"])
def test_cli_args_identifier_requires_a_valid_python_name(
    monkeypatch: pytest.MonkeyPatch, identifier: object
) -> None:
    monkeypatch.setattr(compiler_env, "CLI_ARGS_IDENTIFIER", identifier)

    with pytest.raises(ValueError, match="valid, non-keyword Python identifier"):
        cli_args_identifier()


def test_kedienv_keeps_regular_dictionary_operations_available() -> None:
    env = KediEnv({"existing": 1})

    assert env.setdefault("existing", 2) == 1
    assert env.setdefault("added", 3) == 3
    assert env | {"right": 4} == {"existing": 1, "added": 3, "right": 4}
    assert {"left": 0} | env == {"left": 0, "existing": 1, "added": 3}
    assert env.copy() == {"existing": 1, "added": 3}
    assert KediEnv.fromkeys(["first", "second"], 5) == {"first": 5, "second": 5}

    env |= {"merged": 6}
    assert env.pop("merged") == 6
    del env["added"]
    assert env.popitem() == ("existing", 1)
    env.clear()
    assert env == {}

    with pytest.raises(KeyError, match="dictionary is empty"):
        env.popitem()


def test_scope_frame_dict_mutations_preserve_binding_ownership_and_metadata() -> None:
    parent = ScopeFrame()
    parent.declare("outer", 1, BindingMetadata(kind="outer"))
    child = ScopeFrame(parent=parent)

    assert child.setdefault("outer", 2) == 1
    assert not child.owns("outer")
    child.update({"local": 3})
    assert child.binding_metadata("local") == BindingMetadata()
    child |= {"second": 4}
    assert child.pop("second") == 4
    assert child.binding_metadata("second") is None
    child.clear()
    assert child.raw_value("outer") == 1
    assert not child.owns("local")


def test_scope_frame_rejects_invalid_restore_assignment_and_mutation_edges() -> None:
    frame = ScopeFrame()
    with pytest.raises(ValueError, match="same names"):
        frame.restore_locals({"value": 1}, {})

    reserved = cli_args_identifier()
    with pytest.raises(ValueError, match="runtime-owned"):
        frame.restore_locals(
            {reserved: object()},
            {reserved: BindingMetadata(kind="value")},
        )

    with pytest.raises(NameError, match="unknown Kedi binding"):
        frame.assign("missing", 1)

    frame.declare("readonly", 1, BindingMetadata(assignable=False))
    with pytest.raises(ValueError, match="read-only"):
        frame.assign("readonly", 2)

    plain_parent = {"outer": 1}
    child = ScopeFrame(parent=plain_parent)
    child.assign("outer", 2)
    assert plain_parent["outer"] == 2
    assert child.setdefault("new", 3) == 3
    assert child.owns("new")
    assert child.popitem() == ("new", 3)
    child["new"] = 3

    with pytest.raises(TypeError, match="at most 2 arguments"):
        child.pop("new", 1, 2)
    with pytest.raises(KeyError, match="missing"):
        child.pop("missing")

    empty = ScopeFrame()
    with pytest.raises(KeyError, match="dictionary is empty"):
        empty.popitem()

    reserved_frame = ScopeFrame()
    reserved_frame.restore_locals(
        {reserved: object()},
        {
            reserved: BindingMetadata(
                kind="runtime",
                assignable=False,
                synthetic=True,
            )
        },
    )
    with pytest.raises(ValueError, match="reserved CLI binding"):
        reserved_frame.clear()


def test_owned_value_writes_validate_scope_ownership_and_readonly_metadata() -> None:
    frame = ScopeFrame()
    with pytest.raises(RuntimeError, match="does not own"):
        write_owned_value(frame, "missing", 1)

    frame.declare("readonly", 1, BindingMetadata(assignable=False))
    with pytest.raises(ValueError, match="read-only"):
        write_owned_value(frame, "readonly", 2)

    plain: dict[str, Any] = {}
    write_owned_value(plain, "value", 3)
    assert plain == {"value": 3}


def test_python_writeback_ignores_dirty_binding_removed_from_source() -> None:
    runtime = KediRuntime()
    owner = {"value": 1}
    source = KediEnv()
    source._set_binding("value", 1, owner)
    source["value"] = 2
    dict.__delitem__(source, "value")

    runtime.executor._sync_existing_variables(runtime, owner, source)

    assert owner == {"value": 1}


def test_reserved_cli_args_cannot_be_overwritten_by_kedi_or_python() -> None:
    cli_args = SimpleNamespace(name="Ada")

    initialization_runtime = compile_kedi_program(
        parse_program("[args] = overwritten\n"),
        runtime_globals={"args": cli_args},
    )
    with pytest.raises(Exception, match="Reserved identifier 'args'"):
        initialization_runtime.run_main()

    output_runtime = compile_kedi_program(
        parse_program(">> [args]\n"),
        runtime_globals={"args": cli_args},
    )
    with pytest.raises(Exception, match="Reserved identifier 'args'"):
        output_runtime.run_main()

    raw_invoke_runtime = compile_kedi_program(
        parse_program("[args] << hello\n"),
        runtime_globals={"args": cli_args},
    )
    with pytest.raises(Exception, match="Reserved identifier 'args'"):
        raw_invoke_runtime.run_main()

    python_runtime = KediRuntime()
    python_runtime.set_initial_globals({"args": cli_args})
    with pytest.raises(ValueError, match="Reserved identifier 'args'"):
        python_runtime.executor.execute_side_effects(python_runtime, "args = 'override'", {})


def test_empty_cli_args_retain_their_runtime_identity() -> None:
    from kedi.app import _runtime_globals

    cli_args = AttrDict({}, default=None)
    assert _runtime_globals(cli_args)["args"] is cli_args

    runtime = compile_kedi_program(
        parse_program("@identity(value: int) -> int:\n  = `value`\n= `identity(1)`\n"),
        runtime_globals=_runtime_globals(cli_args),
    )
    runtime.set_initial_globals(_runtime_globals(cli_args))
    assert runtime.run_main() == 1


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("= `1`\n", 1),
        ("= ```\nreturn [1, 2, 3]\n```\n", [1, 2, 3]),
        ("= <`1`>\n", "1"),
    ],
)
def test_run_main_preserves_native_top_level_python_returns(source: str, expected: Any) -> None:
    runtime = compile_kedi_program(parse_program(source))

    assert runtime.run_main() == expected


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
    type_env = build_type_env(broken_rt, None)
    assert type_env["__builtins__"] is builtins.__dict__
    assert "builtins" not in type_env
    assert "typing" not in type_env
    assert "typing_extensions" not in type_env
    assert "List" in type_env
    assert "Protocol" not in type_env
    assert "Generic" not in type_env
    assert build_type_env(broken_rt, None)["datetime"].__name__ == "datetime"
    assert build_type_env(broken_rt, None)["date"].__name__ == "date"
    assert build_type_env(broken_rt, None)["time"].__name__ == "time"
    assert build_type_env(broken_rt, None)["timedelta"].__name__ == "timedelta"
    assert build_type_env(broken_rt, None)["Regex"].__origin__.__name__ == "Pattern"
    assert build_type_env(broken_rt, None)["Email"].__name__ == "EmailStr"
    assert build_type_env(broken_rt, None)["HttpUrl"].__name__ == "HttpUrl"
    assert build_type_env(broken_rt, None)["FileUrl"].__name__ == "FileUrl"
    assert build_type_env(broken_rt, None)["ApprovalRequest"].__name__ == "ApprovalRequest"
    assert build_type_env(broken_rt, None)["ApprovalDecision"].__name__ == "ApprovalDecision"
    cast(Any, broken_rt)._type_env_stack = None
    assert "List" in build_type_env(broken_rt, None)
    with pytest.raises(KeyError, match="Unknown variable"):
        lookup_value(broken_rt, "missing", {})

    monkeypatch.setattr(compiler_env.typing, "__all__", ("MissingTypingName",), raising=False)
    assert "MissingTypingName" not in build_type_env(KediRuntime(), None)

    monkeypatch.setattr(compiler_env.typing, "__all__", ("_PrivateTypingName",), raising=False)
    assert "_PrivateTypingName" not in build_type_env(KediRuntime(), None)

    monkeypatch.setattr(compiler_env, "typing", SimpleNamespace(List=list))
    assert build_type_env(KediRuntime(), None)["List"] is list

    no_globals = KediRuntime()
    no_globals._globals_env = None
    assert build_python_env(no_globals, {"local": 1})["local"] == 1

    plain_env: dict[str, Any] = {}
    no_globals._type_env_stack = [{"Person": Person}]
    compiler_env._add_type_env(no_globals, plain_env, {})
    assert plain_env["Person"] is Person


def test_runtime_lookup_reads_current_env() -> None:
    rt = KediRuntime()
    token = rt._current_env.set({"current": "value"})
    try:
        assert rt._lookup_value("current", {}) == "value"
    finally:
        rt._current_env.reset(token)


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


def test_proc_wrapper_allows_metadata_without_procedure_def() -> None:
    from kedi.lang.procedure_introspection import ProcedurePythonMetadata

    rt = KediRuntime()

    def loose() -> str:
        return "ok"

    setattr(
        loose,
        "__kedi_procedure_metadata__",
        ProcedurePythonMetadata(
            name="loose",
            params=(),
            param_types=(),
            param_defaults=(),
            default_values=(),
            return_type="str",
            docstring=None,
        ),
    )
    rt._procedures["loose"] = loose

    wrapper = compiler_env._make_proc_wrapper(rt, "loose", loose, {})

    assert wrapper() == "ok"
    assert getattr(wrapper, "__kedi_procedure_metadata__").name == "loose"
    assert not hasattr(wrapper, "__kedi_procedure_def__")


def test_proc_wrapper_falls_back_when_generated_source_is_invalid() -> None:
    from kedi.lang.procedure_introspection import ProcedurePythonMetadata

    rt = KediRuntime()

    def loose() -> str:
        return "ok"

    setattr(
        loose,
        "__kedi_procedure_metadata__",
        ProcedurePythonMetadata(
            name="invalid-name",
            params=(),
            param_types=(),
            param_defaults=(),
            default_values=(),
            return_type="str",
            docstring=None,
        ),
    )
    rt._procedures["loose"] = loose

    wrapper = compiler_env._make_proc_wrapper(rt, "loose", loose, {})

    assert wrapper() == "ok"


def test_statement_compiler_plain_mapping_assignment_helpers() -> None:
    env: dict[str, Any] = {"value": 1}
    assert StatementCompiler._assignment_type("value", env) is None
    StatementCompiler._assign_value("value", 2, env)
    assert env["value"] == 2

    with pytest.raises(NameError, match="unknown Kedi binding"):
        StatementCompiler._assignment_type("missing", env)
    with pytest.raises(NameError, match="unknown Kedi binding"):
        StatementCompiler._assign_value("missing", 1, env)


def test_statement_executor_legacy_sync_updates_only_existing_names() -> None:
    runtime = KediRuntime()
    current = {"kept": 2, "new": 3}
    closure = {"kept": 1, "untouched": 4}
    token = runtime._current_env.set(current)
    try:
        StatementExecutor(runtime)._sync_env(closure)
    finally:
        runtime._current_env.reset(token)

    assert closure == {"kept": 2, "untouched": 4}


def test_prepare_segments_binding_rhs_and_argument_variants() -> None:
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

    native, rendered_native = evaluate_binding_rhs(
        rt,
        [TextSegment(text=" "), PythonExprNativeSegment(code="x + 1")],
        {"x": 2},
    )
    assert native == 3
    assert rendered_native == "3"

    called, rendered_called = evaluate_binding_rhs(
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

    def _register_repeated(value: str) -> None:
        @rt.procedure
        def repeated() -> str:
            return value

    _register_repeated("first")
    _register_repeated("second")

    @rt.procedure
    def add_one(value: int) -> int:
        return value + 1

    assert rt._get_proc("repeated")() == "second"
    with pytest.raises(RuntimeError, match="outside of a kedi procedure"):
        rt.m(["prompt"])
    with pytest.raises(NameError, match="Value not found"):
        rt._evaluate_expression(i("missing"), {})
    assert rt._evaluate_expression(c("add_one", i("value")), {"value": 1}) == (
        "2",
        None,
    )
    assert rt._evaluate_expression(o("answer"), {})[0] == "[answer]"
    assert rt._evaluate_expression(cast(Any, object()), {})[1] is None
    with pytest.raises(NameError, match="Value not found"):
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


def test_no_tool_template_does_not_import_pydantic_ai_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Adapter:
        def produce_sync(self, *, template: str, output_schema: dict[str, Any], **_: Any) -> Any:
            class Result:
                answer = "ok"

            return Result()

    real_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "pydantic_ai" or name.startswith("pydantic_ai."):
            raise AssertionError(f"unexpected pydantic_ai import: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    rt = compile_kedi_program(parse_program(">> answer [answer]\n= <answer>\n"), adapter=Adapter())

    assert rt.run_main() == "ok"


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
    assert inline_rt.run_main() == 2


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

    from kedi.lang.compiler.environment import ScopeFrame

    parent = ScopeFrame({"lexical": "parent", "local": "parent local"})
    prepared = ScopeFrame({"local": "value"}, parent=parent)
    assert prepared["lexical"] == "parent"
    assert prepared["local"] == "value"
    assert dict(prepared) == {"local": "value"}


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


def test_statement_compiler_initialization_template_and_single_statement_edges(
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
    compiler.execute_variable_init(
        VariableInitStmt(name="angle", expr=[PythonExprSegment(code="x + 1")]),
        closure,
    )
    assert closure["angle"] == "2"

    monkeypatch.setattr(
        statement_compiler_module,
        "evaluate_binding_rhs",
        lambda rt_arg, segments, env_arg: (2, "2"),
    )
    monkeypatch.setattr(
        statement_compiler_module,
        "detect_rhs_kind",
        lambda segments: (False, True, False),
    )
    compiler.execute_variable_init(VariableInitStmt(name="native_angle", expr=[]), closure)
    assert closure["native_angle"] == "2"

    monkeypatch.setattr(
        statement_compiler_module,
        "detect_rhs_kind",
        lambda segments: (False, False, False),
    )
    compiler.execute_variable_init(VariableInitStmt(name="native_rendered", expr=[]), closure)
    assert closure["native_rendered"] == "2"

    monkeypatch.setattr(
        statement_compiler_module,
        "evaluate_binding_rhs",
        lambda rt_arg, segments, env_arg: (1, "not-int"),
    )
    monkeypatch.setattr(
        statement_compiler_module,
        "detect_rhs_kind",
        lambda segments: (False, False, False),
    )
    with pytest.raises(
        TypeError, match="Native initialization value does not match annotated type"
    ):
        compiler.execute_variable_init(
            VariableInitStmt(
                name="bad",
                expr=[TextSegment(text=" "), PythonExprNativeSegment(code="1")],
                type_annotation="int",
            ),
            closure,
        )
    monkeypatch.undo()

    executor = StatementExecutor(rt)
    template_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        executor.compiler,
        "compile_template_line",
        lambda _stmt: lambda **kwargs: template_calls.append(kwargs),
    )
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
                TemplateLine(segments=[TextSegment(text="legacy template")]),
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
    assert template_calls == [{}]
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
        VariableInitStmt(name="value", expr=[TextSegment(text="ok")]), single_env
    )
    assert single_env["value"] == "ok"
    executor.execute_single_statement(ReturnStmt(segments=[TextSegment(text="ignored")]), {})


def test_procedure_kwargs_skip_non_dict_runtime_envs() -> None:
    rt = compile_kedi_program(parse_program("@p():\n  = ok\n"), adapter=None)
    setattr(rt, "_import_env", object())
    setattr(rt, "_globals_env", object())

    assert rt._get_proc("p")() == "ok"
