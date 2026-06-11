">> Tests for the tree-sitter-backed parser pipeline (Phase 6).\n\n>> These are NEW tests added during the refactor — they verify the parts\nof the public API that the pre-refactor parser did not expose:\n\n  - ``Span`` is populated on every AST node.\n  - Structured ``TypeExpr`` is produced (``TypeRef`` / ``TypeApply`` /\n    ``TypeUnion`` / ``TypePython``), not just opaque strings.\n  - ``parse_program`` returns ``ProgramWithValidation`` whenever\n    ``@test:`` or ``@eval:`` blocks are present.\n  - ``KediParseError`` is a ``ValueError`` subclass and carries\n    ``KediDiagnostic`` records with source locations.\n\n>> The pre-existing ``test_kedi_lang_*`` suites still cover the\nsource-in / behavior-out side; this file complements them by asserting\non the structured AST shape itself.\n"

from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from kedi.lang import (
    KediDiagnostic,
    KediParseError,
    Program,
    ProgramWithValidation,
    Span,
    TypeApply,
    TypePython,
    TypeRef,
    TypeUnion,
    parse_program,
)
from kedi.lang.ast import (
    AssignStmt,
    InputSegment,
    OutputSegment,
    PythonExprSegment,
    ReturnBlockStmt,
    ReturnStmt,
    SystemDirectiveStmt,
    TemplateBlock,
    TextSegment,
    UseDirectiveStmt,
)

cst_module = importlib.import_module("kedi.lang.parser.cst_to_ast")

# ---------------------------------------------------------------------
# Span propagation
# ---------------------------------------------------------------------


class _FakeNode:
    _next_id = 1

    def __init__(
        self,
        type_: str,
        *,
        children: list["_FakeNode"] | None = None,
        fields: dict[str, "_FakeNode"] | None = None,
        named: bool = True,
        has_error: bool = False,
        is_error: bool = False,
        is_missing: bool = False,
        start: int = 0,
        end: int = 0,
    ) -> None:
        self.type = type_
        self.children = children or []
        self._fields = fields or {}
        self.is_named = named
        self.has_error = has_error
        self.is_error = is_error
        self.is_missing = is_missing
        self.start_byte = start
        self.end_byte = end
        self.start_point = (0, start)
        self.end_point = (0, end)
        self.id = _FakeNode._next_id
        _FakeNode._next_id += 1
        self._field_names_by_id = {child.id: name for name, child in self._fields.items()}

    def child_by_field_name(self, name: str):
        return self._fields.get(name)

    def children_by_field_name(self, name: str):
        child = self._fields.get(name)
        if child is None:
            return []
        if isinstance(child, list):
            return child
        return [child]

    def field_name_for_child(self, child_id: int):
        return self._field_names_by_id.get(child_id)


def _fake_walker(source: bytes = b"name") -> Any:
    tree = cast(Any, SimpleNamespace(root_node=_FakeNode("source_file")))
    return cst_module._Walker(source, tree, None, None)


def test_cst_walker_defensive_edges(monkeypatch):
    assert cst_module._decode_block_comment_body("###") == ""

    field_child = _FakeNode("field_child")
    other_child = _FakeNode("other_child")
    field_parent = _FakeNode(
        "parent", children=[field_child, other_child], fields={"target": field_child}
    )
    assert cst_module._fields(cast(Any, field_parent), "target") == [field_child]

    bad_root = _FakeNode("not_source")
    with pytest.raises(KediParseError):
        cst_module._Walker(
            b"", cast(Any, SimpleNamespace(root_node=bad_root)), None, None
        ).walk_program()
    odd_root = _FakeNode("source_file", children=[_FakeNode("odd_top_level")])
    with pytest.raises(KediParseError):
        cst_module._Walker(
            b"", cast(Any, SimpleNamespace(root_node=odd_root)), None, None
        ).walk_program()

    walker = _fake_walker(b"\xff")
    walker._validate_block_comments()
    walker._line_error(0, "bad")
    assert walker.diagnostics == []

    missing = _FakeNode("identifier", has_error=True, is_missing=True)
    error_walker = _fake_walker()
    error_walker._collect_tree_errors(missing)
    assert error_walker.diagnostics[-1].message == "missing 'identifier'"
    missing_newline = _FakeNode("_newline", has_error=True, is_missing=True)
    error_walker._collect_tree_errors(missing_newline)
    assert error_walker.diagnostics[-1].message == "missing 'identifier'"

    export_walker = _fake_walker(b"> export:\n  name\n")
    export_body = _FakeNode(
        "body",
        children=[
            _FakeNode("line_comment"),
            _FakeNode(
                "module_export_name",
                fields={"name": _FakeNode("identifier", start=12, end=16)},
            ),
            _FakeNode(
                "module_export_name",
                fields={"name": _FakeNode("identifier", start=12, end=16)},
            ),
        ],
    )
    export = export_walker._walk_module_export(
        _FakeNode("module_export", fields={"body": export_body}, end=16)
    )
    assert export.names == ["name", "name"]
    export_body_missing_name = _FakeNode(
        "body",
        children=[
            _FakeNode("module_export_name"),
            _FakeNode(
                "module_export_name",
                fields={"name": _FakeNode("identifier", start=12, end=16)},
            ),
        ],
    )
    export_missing = export_walker._walk_module_export(
        _FakeNode("module_export", fields={"body": export_body_missing_name}, end=16)
    )
    assert export_missing.names == ["name"]

    proc_walker = _fake_walker(b"proc")
    proc = proc_walker._walk_procedure_def(
        _FakeNode(
            "procedure_def",
            fields={"name": _FakeNode("identifier", start=0, end=4)},
            end=4,
        )
    )
    assert proc.body == []
    param_noise = _FakeNode("param_list", children=[_FakeNode("line_comment")])
    proc_with_noise = proc_walker._walk_procedure_def(
        _FakeNode(
            "procedure_def",
            fields={
                "name": _FakeNode("identifier", start=0, end=4),
                "params": param_noise,
            },
            end=4,
        )
    )
    assert proc_with_noise.params == []

    stmt_walker = _fake_walker(b"stmt")
    assert stmt_walker._walk_stmt(_FakeNode("type_def_stmt"), None) is None
    assert stmt_walker._walk_stmt(_FakeNode("auto_directive"), None) is None
    assert stmt_walker._walk_stmt(_FakeNode("optimize_directive"), None) is None
    assert stmt_walker._walk_stmt(_FakeNode("unknown_stmt"), None) is None
    assert len(stmt_walker.diagnostics) >= 4

    assert stmt_walker._walk_backtick_line_stmt(_FakeNode("backtick_line_stmt")).code == ""
    backtick_with_noise = stmt_walker._walk_backtick_line_stmt(
        _FakeNode(
            "backtick_line_stmt",
            children=[
                _FakeNode("line_comment"),
                _FakeNode(
                    "inline_python_expr",
                    fields={"code": _FakeNode("python_inline_body", start=0, end=4)},
                ),
            ],
        )
    )
    assert backtick_with_noise.code == "stmt"
    assert stmt_walker._walk_template_line(_FakeNode("template_line")).segments == []
    empty_template_block = stmt_walker._walk_template_block_stmt(_FakeNode("template_block_stmt"))
    assert empty_template_block.lines == []
    template_with_noise = stmt_walker._walk_template_line(
        _FakeNode(
            "template_line",
            children=[
                _FakeNode("line_comment"),
                _FakeNode(
                    "template_expr",
                    children=[_FakeNode("text_segment", start=0, end=4)],
                ),
            ],
        )
    )
    assert template_with_noise.segments[0].text == "stmt"
    whitespace_only = _fake_walker(b"   ")._walk_template_expr(
        _FakeNode(
            "template_expr",
            children=[_FakeNode("text_segment", start=0, end=3)],
        )
    )
    assert whitespace_only[0].text == ""
    escaped_leading = _fake_walker(b"\\s hello")._walk_template_expr(
        _FakeNode(
            "template_expr",
            children=[_FakeNode("text_segment", start=0, end=8)],
        )
    )
    assert escaped_leading[0].escaped_whitespace == {0}
    assert escaped_leading[0].text == "  hello"
    assert stmt_walker._walk_type_expr(_FakeNode("type_expr")) is None
    weird_type = _FakeNode("type_expr", children=[_FakeNode("weird_type")])
    assert stmt_walker._walk_type_expr(weird_type) is None
    assert stmt_walker._walk_type_term(_FakeNode("weird_term")) is None

    type_apply = _FakeNode(
        "type_apply",
        fields={"name": _FakeNode("identifier", start=0, end=4)},
        children=[_FakeNode("not_type_expr")],
        end=4,
    )
    assert cst_module.TypeApply == TypeApply
    assert stmt_walker._walk_type_term(type_apply).args == []

    type_def = stmt_walker._walk_type_def(
        _FakeNode(
            "type_def",
            fields={
                "name": _FakeNode("identifier", start=0, end=4),
                "fields": _FakeNode("fields", children=[_FakeNode("comment")]),
            },
            end=4,
        )
    )
    assert type_def.fields == []
    assert (
        stmt_walker._walk_type_def(
            _FakeNode(
                "type_def",
                fields={"name": _FakeNode("identifier", start=0, end=4)},
                end=4,
            )
        ).fields
        == []
    )

    assert stmt_walker._walk_segment(_FakeNode("python_expr_segment")).code == ""
    python_expr_with_noise = stmt_walker._walk_segment(
        _FakeNode(
            "python_expr_segment",
            children=[
                _FakeNode("line_comment"),
                _FakeNode(
                    "inline_python_expr",
                    fields={"code": _FakeNode("python_inline_body", start=0, end=4)},
                ),
            ],
        )
    )
    assert python_expr_with_noise.code == "stmt"
    assert isinstance(stmt_walker._walk_segment(_FakeNode("unknown_segment")), TextSegment)
    assert stmt_walker._walk_call_arg(_FakeNode("call_arg")).value == ""
    call_arg_with_noise = stmt_walker._walk_call_segment(
        _FakeNode(
            "call_segment",
            fields={
                "name": _FakeNode("identifier", start=0, end=4),
                "args": _FakeNode(
                    "args",
                    children=[
                        _FakeNode("line_comment"),
                        _FakeNode("call_arg", fields={}, end=4),
                    ],
                ),
            },
        )
    )
    assert len(call_arg_with_noise.args) == 1
    render_arg = stmt_walker._walk_call_arg(
        _FakeNode(
            "call_arg",
            children=[
                _FakeNode(
                    "template_arg",
                    children=[
                        _FakeNode("text_in_call_segment", start=0, end=4),
                        _FakeNode("text_in_call_segment", start=0, end=4),
                    ],
                )
            ],
        )
    )
    assert hasattr(render_arg, "__render_segments__")
    unknown_arg = _FakeNode("call_arg", children=[_FakeNode("unknown", start=0, end=4)], end=4)
    assert stmt_walker._walk_call_arg(unknown_arg).value == "stmt"
    inline_without_code = _FakeNode("inline_python_expr")
    assert stmt_walker._walk_inline_python_expr(inline_without_code) == ""
    assert stmt_walker._dedent_fenced(_FakeNode("python_code", start=0, end=4)) == "stmt"

    auto = stmt_walker._walk_auto_directive(_FakeNode("auto_directive"))
    assert auto.prompt == ""
    auto_body = _FakeNode(
        "body",
        children=[
            _FakeNode("comment"),
            _FakeNode("template_line"),
            _FakeNode("template_line", children=[_FakeNode("template_expr", start=0, end=4)]),
        ],
    )
    auto_with_body = stmt_walker._walk_auto_directive(
        _FakeNode("auto_directive", fields={"body": auto_body})
    )
    assert auto_with_body.prompt == "stmt"

    opt_proc = cst_module.ProcedureDef("proc", [])
    empty_opt = stmt_walker._walk_optimize_directive(
        _FakeNode(
            "optimize_directive",
            fields={"name": _FakeNode("identifier", start=0, end=4)},
        ),
        opt_proc,
    )
    assert empty_opt.template_lines == []
    noisy_opt = stmt_walker._walk_optimize_directive(
        _FakeNode(
            "optimize_directive",
            fields={
                "name": _FakeNode("identifier", start=0, end=4),
                "body": _FakeNode("body", children=[_FakeNode("line_comment")]),
            },
        ),
        opt_proc,
    )
    assert noisy_opt.template_lines == []

    backtick_model = stmt_walker._walk_model_directive(
        _FakeNode(
            "model_directive",
            fields={
                "value": _FakeNode(
                    "inline_python_expr",
                    fields={"code": _FakeNode("python_inline_body", start=0, end=4)},
                )
            },
        )
    )
    assert backtick_model.backtick_code == "stmt"
    backtick_system = stmt_walker._walk_system_directive(
        _FakeNode(
            "system_directive",
            fields={
                "value": _FakeNode(
                    "inline_python_expr",
                    fields={"code": _FakeNode("python_inline_body", start=0, end=4)},
                )
            },
        )
    )
    assert backtick_system.backtick_code == "stmt"
    noisy_system = stmt_walker._walk_system_directive(
        _FakeNode(
            "system_directive",
            fields={
                "body": _FakeNode(
                    "body",
                    children=[
                        _FakeNode("line_comment"),
                        _FakeNode("system_line"),
                    ],
                )
            },
        )
    )
    assert noisy_system.template_lines == []
    stmt_walker._system_line_from_expr(
        _FakeNode("template_expr", children=[_FakeNode("unknown_segment")])
    )
    assert any(d.code == "K-SYSTEM-SEGMENT" for d in stmt_walker.diagnostics)

    profile_body = _FakeNode(
        "body",
        children=[
            _FakeNode("line_comment"),
            _FakeNode(
                "model_directive",
                fields={
                    "value": _FakeNode(
                        "inline_python_expr",
                        fields={"code": _FakeNode("python_inline_body", start=0, end=4)},
                    )
                },
            ),
            _FakeNode(
                "model_directive",
                fields={"value": _FakeNode("identifier", start=0, end=4)},
            ),
            _FakeNode(
                "system_directive",
                fields={"value": _FakeNode("identifier", start=0, end=4)},
            ),
            _FakeNode(
                "system_directive",
                fields={
                    "value": _FakeNode(
                        "inline_python_expr",
                        fields={"code": _FakeNode("python_inline_body", start=0, end=4)},
                    )
                },
            ),
            _FakeNode(
                "system_directive",
                fields={},
            ),
            _FakeNode(
                "model_directive",
                fields={},
            ),
        ],
    )
    profile = stmt_walker._walk_profile_directive(
        _FakeNode(
            "profile_directive",
            fields={
                "name": _FakeNode("identifier", start=0, end=4),
                "body": profile_body,
            },
        )
    )
    assert profile.model_members[0].backtick_code == "stmt"
    assert profile.model_members[1].plain_value == "stmt"
    assert profile.system_members[0].plain_value == "stmt"
    assert profile.system_members[1].backtick_code == "stmt"

    profile_with_tool_use = stmt_walker._walk_profile_directive(
        _FakeNode(
            "profile_directive",
            fields={
                "name": _FakeNode("identifier", start=0, end=4),
                "body": _FakeNode(
                    "body",
                    children=[
                        _FakeNode(
                            "use_directive",
                            fields={"name": _FakeNode("identifier", start=0, end=4)},
                        ),
                    ],
                ),
            },
        )
    )
    assert profile_with_tool_use.tool_names == ["stmt"]

    profile_with_multiline_tool_use = stmt_walker._walk_profile_directive(
        _FakeNode(
            "profile_directive",
            fields={
                "name": _FakeNode("identifier", start=0, end=4),
                "body": _FakeNode(
                    "body",
                    children=[
                        _FakeNode(
                            "use_directive",
                            fields={
                                "body": _FakeNode(
                                    "body",
                                    children=[
                                        _FakeNode(
                                            "use_tool_name",
                                            fields={
                                                "name": _FakeNode("identifier", start=0, end=4)
                                            },
                                        ),
                                    ],
                                )
                            },
                        ),
                    ],
                ),
            },
        )
    )
    assert profile_with_multiline_tool_use.tool_names == ["stmt"]

    profile_with_both_use_forms = stmt_walker._walk_profile_directive(
        _FakeNode(
            "profile_directive",
            fields={
                "name": _FakeNode("identifier", start=0, end=4),
                "body": _FakeNode(
                    "body",
                    children=[
                        _FakeNode(
                            "use_directive",
                            fields={"name": _FakeNode("identifier", start=0, end=4)},
                        ),
                        _FakeNode(
                            "use_directive",
                            fields={
                                "body": _FakeNode(
                                    "body",
                                    children=[
                                        _FakeNode(
                                            "use_tool_name",
                                            fields={
                                                "name": _FakeNode("identifier", start=0, end=4)
                                            },
                                        ),
                                    ],
                                )
                            },
                        ),
                    ],
                ),
            },
        )
    )
    assert profile_with_both_use_forms.tool_names == ["stmt", "stmt"]

    empty_name_use_walker = _fake_walker(b"stmt")
    calls = 0

    def _fake_walk_use(_node: _FakeNode) -> UseDirectiveStmt:
        nonlocal calls
        calls += 1
        if calls == 1:
            return UseDirectiveStmt(target="", tool_names=[])
        return UseDirectiveStmt(target="tool")

    empty_name_use_walker._walk_use_directive = _fake_walk_use  # type: ignore[method-assign]
    profile_with_empty_use = empty_name_use_walker._walk_profile_directive(
        _FakeNode(
            "profile_directive",
            fields={
                "name": _FakeNode("identifier", start=0, end=4),
                "body": _FakeNode(
                    "body",
                    children=[
                        _FakeNode("use_directive"),
                        _FakeNode("use_directive"),
                    ],
                ),
            },
        )
    )
    assert profile_with_empty_use.tool_names == ["tool"]

    use_body_with_noise = stmt_walker._walk_use_tool_names(
        _FakeNode(
            "body",
            children=[
                _FakeNode("line_comment"),
                _FakeNode(
                    "use_tool_name",
                    fields={"name": _FakeNode("identifier", start=0, end=4)},
                ),
            ],
        )
    )
    assert use_body_with_noise == ["stmt"]

    empty_use_walker = _fake_walker()
    empty_use_walker._walk_use_tool_names(_FakeNode("body", children=[]))
    assert any(d.code == "K-USE-EMPTY" for d in empty_use_walker.diagnostics)

    mixed_use_walker = _fake_walker(b"alpha\n`beta`")
    tool_names = mixed_use_walker._walk_use_tool_names(
        _FakeNode(
            "body",
            children=[
                _FakeNode(
                    "use_tool_name",
                    fields={"name": _FakeNode("identifier", start=0, end=5)},
                ),
                _FakeNode(
                    "use_tool_backtick",
                    fields={"name": _FakeNode("identifier", start=7, end=11)},
                ),
            ],
        )
    )
    assert tool_names == ["alpha", "beta"]

    empty_profile_walker = _fake_walker(b"name")
    empty_profile_walker._walk_profile_directive(
        _FakeNode(
            "profile_directive",
            fields={"name": _FakeNode("identifier", start=0, end=4)},
        )
    )
    assert any(d.code == "K-PROFILE-EMPTY" for d in empty_profile_walker.diagnostics)

    nested_proc = cst_module.ProcedureDef("proc", [])
    assert stmt_walker._walk_stmt(_FakeNode("profile_directive"), nested_proc) is None
    assert any(d.code == "K-PROFILE-SCOPE" for d in stmt_walker.diagnostics)

    validation_walker = _fake_walker(b"test")
    assert validation_walker._walk_validation_block(_FakeNode("validation_block")) is None
    assert (
        validation_walker._walk_validation_block(
            _FakeNode("validation_block", fields={"kw": _FakeNode("kw", start=0, end=4)})
        )
        is None
    )

    test_body = _FakeNode(
        "body",
        children=[_FakeNode("eval_data"), _FakeNode("ERROR", is_error=True)],
    )
    test_suite = validation_walker._walk_validation_block(
        _FakeNode(
            "validation_block",
            fields={
                "kw": _FakeNode("kw", start=0, end=4),
                "procedure": _FakeNode("identifier", start=0, end=4),
                "body": test_body,
            },
        )
    )
    assert isinstance(test_suite, cst_module.TestSuite)
    ignored_test_body = _FakeNode("body", children=[_FakeNode("line_comment")])
    ignored_test_suite = validation_walker._walk_validation_block(
        _FakeNode(
            "validation_block",
            fields={
                "kw": _FakeNode("kw", start=0, end=4),
                "procedure": _FakeNode("identifier", start=0, end=4),
                "body": ignored_test_body,
            },
        )
    )
    assert isinstance(ignored_test_suite, cst_module.TestSuite)
    assert ignored_test_suite.cases == []

    eval_walker = _fake_walker(b"eval")
    eval_body = _FakeNode(
        "body",
        children=[_FakeNode("test_case"), _FakeNode("ERROR", is_error=True)],
    )
    eval_suite = eval_walker._walk_validation_block(
        _FakeNode(
            "validation_block",
            fields={
                "kw": _FakeNode("kw", start=0, end=4),
                "procedure": _FakeNode("identifier", start=0, end=4),
                "body": eval_body,
            },
        )
    )
    assert isinstance(eval_suite, cst_module.EvalSuite)
    ignored_eval_body = _FakeNode("body", children=[_FakeNode("line_comment")])
    ignored_eval_suite = eval_walker._walk_validation_block(
        _FakeNode(
            "validation_block",
            fields={
                "kw": _FakeNode("kw", start=0, end=4),
                "procedure": _FakeNode("identifier", start=0, end=4),
                "body": ignored_eval_body,
            },
        )
    )
    assert isinstance(ignored_eval_suite, cst_module.EvalSuite)
    assert (
        validation_walker._walk_test_case(
            _FakeNode("test_case", fields={"name": _FakeNode("identifier", start=0, end=4)})
        ).code
        == ""
    )
    backtick_without_inline = validation_walker._walk_test_case(
        _FakeNode(
            "test_case",
            fields={
                "name": _FakeNode("identifier", start=0, end=4),
                "code": _FakeNode(
                    "code",
                    children=[
                        _FakeNode("backtick_line_stmt", children=[_FakeNode("line_comment")])
                    ],
                ),
            },
        )
    )
    assert backtick_without_inline.code == ""
    block_test_case = validation_walker._walk_test_case(
        _FakeNode(
            "test_case",
            fields={
                "name": _FakeNode("identifier", start=0, end=4),
                "code": _FakeNode(
                    "code",
                    children=[
                        _FakeNode(
                            "python_block",
                            fields={"code": _FakeNode("python_code", start=0, end=4)},
                        )
                    ],
                ),
            },
        )
    )
    assert block_test_case.is_block is True
    assert block_test_case.code == "test"

    unknown_validation = _fake_walker(b"what")._walk_validation_block(
        _FakeNode(
            "validation_block",
            fields={
                "kw": _FakeNode("kw", start=0, end=4),
                "procedure": _FakeNode("identifier", start=0, end=4),
            },
        )
    )
    assert unknown_validation is None

    assert validation_walker._walk_eval_entry_body(None) == ("", False)
    return_value = _FakeNode("template_expr", start=0, end=4)
    assert validation_walker._walk_eval_entry_body(
        _FakeNode("body", children=[_FakeNode("return_stmt", fields={"value": return_value})])
    ) == ("test", False)
    assert validation_walker._walk_eval_entry_body(
        _FakeNode("body", children=[_FakeNode("return_stmt")])
    ) == ("", False)
    assert validation_walker._walk_eval_entry_body(
        _FakeNode("body", children=[_FakeNode("return_block_stmt")])
    ) == ("", True)
    assert validation_walker._walk_eval_entry_body(
        _FakeNode("body", children=[_FakeNode("line_comment")])
    ) == ("", False)


def test_span_populated_on_root():
    prog = parse_program("= ok\n", source_path="example.kedi")
    assert isinstance(prog.loc, Span)
    assert prog.loc.start_line == 1
    assert prog.loc.start_col == 1
    assert prog.loc.source_path == "example.kedi"


def test_span_propagated_to_procedures_and_body():
    src = "@hello(name):\n  >> Hi, [reply: str]\n  = <reply>\n"
    prog = parse_program(src)
    proc = prog.procedures[0]
    assert proc.loc is not None
    assert proc.loc.start_line == 1
    # Procedure body is the indented block, starting at line 2.
    assert proc.body[0].loc is not None
    assert proc.body[0].loc.start_line == 2
    assert proc.body[1].loc is not None
    assert proc.body[1].loc.start_line == 3


def test_span_source_path_forwarded_to_all_nodes():
    prog = parse_program("@p():\n  = ok\n", source_path="/tmp/example.kedi")
    proc = prog.procedures[0]
    assert proc.loc is not None
    assert proc.loc.source_path == "/tmp/example.kedi"
    assert proc.body[0].loc is not None
    assert proc.body[0].loc.source_path == "/tmp/example.kedi"


def test_tools_fixture_parses_multiline_backtick_use() -> None:
    src = Path("tools.kedi").read_text(encoding="utf-8")
    prog = parse_program(src, source_path=str(Path("tools.kedi").resolve()))

    multiline_uses = [
        stmt
        for stmt in prog.ordered_toplevel
        if isinstance(stmt, UseDirectiveStmt) and stmt.tool_names
    ]

    assert multiline_uses
    assert multiline_uses[0].tool_names == ["web_browse", "web_fetch"]


def test_system_directive_accepts_text_and_input_segments_only() -> None:
    prog = parse_program(
        "> system:\n    You are <style>.\n    Your name is <`args.name`>.\n    Use <``> fences.\n"
    )
    stmt = prog.ordered_toplevel[0]
    assert isinstance(stmt, SystemDirectiveStmt)
    assert len(stmt.template_lines) == 3
    first_line = stmt.template_lines[0]
    second_line = stmt.template_lines[1]
    third_line = stmt.template_lines[2]
    assert isinstance(first_line.segments[0], TextSegment)
    assert isinstance(first_line.segments[1], InputSegment)
    assert any(
        isinstance(seg, PythonExprSegment) and seg.code == "args.name"
        for seg in second_line.segments
    )
    assert any(isinstance(seg, TextSegment) and seg.text == "<``>" for seg in third_line.segments)
    assert stmt.plain_value == "You are .\nYour name is .\nUse <``> fences."


@pytest.mark.parametrize(
    "src",
    [
        "> system: fill [answer]\n",
        "> system: call <tool()>\n",
    ],
)
def test_system_directive_rejects_llm_output_and_calls(src: str) -> None:
    with pytest.raises(KediParseError):
        parse_program(src)


# ---------------------------------------------------------------------
# Structured TypeExpr
# ---------------------------------------------------------------------


def test_type_ref_for_bare_identifier():
    prog = parse_program("@p(x: int) -> str:\n  = ok\n")
    proc = prog.procedures[0]
    assert proc.param_type_exprs is not None
    assert isinstance(proc.return_type_expr, TypeRef)
    assert proc.return_type_expr.name == "str"
    assert isinstance(proc.param_type_exprs["x"], TypeRef)
    assert proc.param_type_exprs["x"].name == "int"


def test_type_apply_for_generics():
    prog = parse_program("@xs(items: list[int]) -> dict[str, int]:\n  = ok\n")
    proc = prog.procedures[0]
    params = proc.param_type_exprs
    assert params is not None
    assert isinstance(params["items"], TypeApply)
    assert params["items"].name == "list"
    assert isinstance(params["items"].args[0], TypeRef)
    assert params["items"].args[0].name == "int"

    ret = proc.return_type_expr
    assert isinstance(ret, TypeApply)
    assert ret.name == "dict"
    assert len(ret.args) == 2
    assert isinstance(ret.args[0], TypeRef) and ret.args[0].name == "str"
    assert isinstance(ret.args[1], TypeRef) and ret.args[1].name == "int"


def test_type_apply_nested():
    prog = parse_program("@p(g: list[list[int]]):\n  = ok\n")
    proc = prog.procedures[0]
    assert proc.param_type_exprs is not None
    g = proc.param_type_exprs["g"]
    assert isinstance(g, TypeApply) and g.name == "list"
    inner = g.args[0]
    assert isinstance(inner, TypeApply) and inner.name == "list"
    assert isinstance(inner.args[0], TypeRef) and inner.args[0].name == "int"


def test_type_union():
    prog = parse_program("@p(v: int | str | float):\n  = ok\n")
    proc = prog.procedures[0]
    assert proc.param_type_exprs is not None
    v = proc.param_type_exprs["v"]
    # The union is left-associative: ((int | str) | float).
    assert isinstance(v, TypeUnion)
    assert isinstance(v.right, TypeRef) and v.right.name == "float"
    assert isinstance(v.left, TypeUnion)
    assert isinstance(v.left.left, TypeRef) and v.left.left.name == "int"
    assert isinstance(v.left.right, TypeRef) and v.left.right.name == "str"


def test_type_python_carries_raw_code():
    prog = parse_program("[x: `Optional[int]`] = `42`\n")
    stmt = prog.toplevel_body[0]
    assert isinstance(stmt, AssignStmt)
    assert isinstance(stmt.type_expr, TypePython)
    assert stmt.type_expr.code == "Optional[int]"


def test_legacy_string_annotation_still_populated():
    """Backward-compat: the string ``type_annotation`` field is filled
    so unmigrated consumers (codegen, optimizer utilities) keep working.
    """
    prog = parse_program("@p(x: list[int]) -> str:\n  = ok\n")
    proc = prog.procedures[0]
    assert proc.param_type_annotations is not None
    assert proc.return_type_annotation == "str"
    assert proc.param_type_annotations["x"] == "list[int]"


def test_output_segment_carries_type_expr():
    prog = parse_program("@p():\n  >> Pick a [x: int].\n  = <x>\n")
    block = prog.procedures[0].body[0]
    assert isinstance(block, TemplateBlock)
    out = next(s for s in block.lines[0].segments if isinstance(s, OutputSegment))
    assert out.name == "x"
    assert isinstance(out.type_expr, TypeRef)
    assert out.type_expr.name == "int"
    # Legacy string still set.
    assert out.type_annotation == "int"


# ---------------------------------------------------------------------
# ProgramWithValidation discrimination
# ---------------------------------------------------------------------


def test_program_returns_plain_program_without_validation():
    prog = parse_program("@p():\n  = ok\n")
    assert isinstance(prog, Program)
    assert not isinstance(prog, ProgramWithValidation)


def test_program_returns_with_validation_when_test_block_present():
    src = '@p():\n  = ok\n\n@test: p:\n  > case: c1:\n    `assert p() == "ok"`\n'
    prog = parse_program(src)
    assert isinstance(prog, ProgramWithValidation)
    assert len(prog.tests) == 1
    assert prog.tests[0].procedure_name == "p"
    assert len(prog.tests[0].cases) == 1


def test_program_returns_with_validation_when_eval_block_present():
    src = (
        "@p(x: int) -> int:\n"
        "  = `x * 2`\n"
        "\n"
        "@eval: p:\n"
        "  > data: cases:\n"
        "    = `[(1, 2), (3, 6)]`\n"
        "  > metric: acc(cases):\n"
        '    = `(1.0, "ok")`\n'
    )
    prog = parse_program(src)
    assert isinstance(prog, ProgramWithValidation)
    assert len(prog.evals) == 1
    suite = prog.evals[0]
    assert suite.procedure_name == "p"
    assert "cases" in suite.datasets
    assert len(suite.metrics) == 1
    assert suite.metrics[0].dataset_name == "cases"


def test_validation_block_kw_is_validated():
    ">> Walker discriminates ``test`` vs ``eval`` from the kw text and\n    >> rejects everything else (the kw is matched as a plain identifier\n    in the grammar to avoid keyword-extraction conflicts with\n    procedure names ``@test(...)`` / ``@eval(...)``).\n"
    with pytest.raises(KediParseError):
        parse_program("@bogus: foo:\n  > case: c:\n    `pass`\n")


# ---------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------


def test_kedi_parse_error_is_value_error():
    """Backward compatibility for ``pytest.raises(ValueError)`` callers."""
    with pytest.raises(ValueError):
        parse_program("[42]\n")  # invalid identifier


def test_diagnostic_has_span_with_line_col():
    try:
        parse_program("~Item(a, a)\n", source_path="dup.kedi")
    except KediParseError as exc:
        assert exc.diagnostics
        d = exc.diagnostics[0]
        assert isinstance(d, KediDiagnostic)
        assert d.span is not None
        assert d.span.start_line >= 1
        assert d.span.start_col >= 1
        assert "duplicate field" in d.message.lower()
    else:
        pytest.fail("expected KediParseError")


def test_unterminated_block_comment_raises():
    with pytest.raises(KediParseError) as info:
        parse_program("###\nbody never closed\n")
    assert any("unterminated" in d.message.lower() for d in info.value.diagnostics)


def test_optimize_empty_body_raises_with_legacy_substring():
    """The diagnostic message must contain the legacy substring so
    callers using ``pytest.raises(ValueError, match="must have an
    indented block")`` keep working.
    """
    src = "@p():\n  > optimize: span:\n\n  = done\n"
    with pytest.raises(KediParseError, match="must have an indented block"):
        parse_program(src)


# ---------------------------------------------------------------------
# Return-statement classification
# ---------------------------------------------------------------------


def test_sole_backtick_return_produces_return_block_stmt():
    """``= \\`code\\``` is a ReturnBlockStmt — preserves the native
    Python value at runtime, so non-string returns without an explicit
    annotation are rejected (legacy invariant the runtime relies on).
    """
    prog = parse_program("@p() -> int:\n  = `42`\n")
    body = prog.procedures[0].body
    assert len(body) == 1
    assert isinstance(body[0], ReturnBlockStmt)
    assert body[0].code == "42"


def test_mixed_return_is_returnstmt_not_block():
    """A return value that mixes text and inline-python segments is a
    ReturnStmt (rendered to string at runtime), not a ReturnBlockStmt.
    The distinction lets the runtime preserve native values for the
    ``= \\`code\\``` form while stringifying everything else.
    """
    from kedi.lang.ast import ReturnStmt

    prog = parse_program("@p(x: int) -> str:\n  = result is <`x`> here\n")
    body = prog.procedures[0].body
    assert isinstance(body[0], ReturnStmt)
    # Segments: text + python_expr_segment + text.
    assert len(body[0].segments) >= 2


# ---------------------------------------------------------------------
# Escape handling
# ---------------------------------------------------------------------


def test_template_text_escapes_decoded():
    prog = parse_program("@p():\n  >> Use \\<angle\\> \\[brackets\\] \\@at \\= eq\n  = ok\n")
    block = prog.procedures[0].body[0]
    assert isinstance(block, TemplateBlock)
    text = "".join(s.text for s in block.lines[0].segments if isinstance(s, TextSegment))
    assert "<angle>" in text
    assert "[brackets]" in text
    assert "@at" in text
    assert "= eq" in text


def test_tilde_double_escape_decoded():
    prog = parse_program("= literal ~~Person\n")
    assert isinstance(prog.toplevel_return, ReturnStmt)
    seg = prog.toplevel_return.segments[0]
    assert isinstance(seg, TextSegment)
    assert "~Person" in seg.text


def test_line_continuation_collapses_to_single_space():
    src = "= Start \\\n  middle \\\n  end\n"
    prog = parse_program(src)
    assert isinstance(prog.toplevel_return, ReturnStmt)
    seg = prog.toplevel_return.segments[0]
    assert isinstance(seg, TextSegment)
    # The trailing-WS-rstrip + `\<newline>` + leading-WS-strip rule
    # collapses to single spaces.
    assert seg.text == "Start middle end"
