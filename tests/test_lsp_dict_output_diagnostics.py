from __future__ import annotations

from pathlib import Path

import pytest
from lsprotocol import types as lsp

import kedi.lsp.features.diagnostics as diagnostics_module
from kedi.lang.ast import OutputSegment, Program, TemplateLine, TypeApply, TypeExpr, TypeRef
from kedi.lsp.features.diagnostics import compute_diagnostics


@pytest.mark.parametrize(
    "capture", [">> Extract [result: {annotation}].", "[result: {annotation}] << Extract data."]
)
@pytest.mark.parametrize(
    ("annotation", "dict_type"),
    [
        ("dict", "dict"),
        ("dict[str, int]", "dict[str, int]"),
        ("dict[str, Any]", "dict[str, Any]"),
        ("Dict[str, int]", "Dict[str, int]"),
        ('Annotated[dict, "Dynamic keys and values"]', "dict"),
        ('Annotated[dict[str, int], "Return name and age keys"]', "dict[str, int]"),
        ('dict[str, Annotated[str, "Full name"]]', 'dict[str, Annotated[str, "Full name"]]'),
        ("Optional[dict[str, int]]", "dict[str, int]"),
        ("Union[str, dict]", "dict"),
        ("dict | None", "dict"),
        ("str | dict", "dict"),
        ("list[dict[str, int]]", "dict[str, int]"),
        ('Annotated[list[dict], "Records"]', "dict"),
    ],
)
def test_open_dict_model_output_warns_on_capture_type(
    capture: str, annotation: str, dict_type: str
) -> None:
    line = "  " + capture.format(annotation=annotation)
    source = "@extract():\n" + line + "\n"

    diagnostics = compute_diagnostics(source, "example.kedi")

    assert len(diagnostics) == 1
    diagnostic = diagnostics[0]
    assert diagnostic.code == "K-OUTPUT-OPEN-DICT"
    assert diagnostic.severity == lsp.DiagnosticSeverity.Warning
    assert diagnostic.source == "kedi"
    assert diagnostic.range == lsp.Range(
        start=lsp.Position(line=1, character=line.index(dict_type)),
        end=lsp.Position(line=1, character=line.index(dict_type) + len(dict_type)),
    )
    assert "TypedDict" in diagnostic.message
    assert "custom type" in diagnostic.message
    assert "keys" in diagnostic.message
    assert "values" in diagnostic.message
    assert "dynamic" in diagnostic.message


@pytest.mark.parametrize(
    "source",
    [
        ">> Extract [result: dict].\n",
        "> if: `True`:\n  >> Extract [result: dict].\n",
        "> loop [item]: `[1]`:\n  [result: dict] << Extract data.\n",
        "@outer():\n  @inner():\n    >> Extract [result: dict].\n  = done\n",
        "@extract():\n  > optimize: extract:\n    >> Extract [result: dict].\n",
    ],
)
def test_open_dict_model_output_warns_once_in_each_scope(source: str) -> None:
    diagnostics = compute_diagnostics(source, "example.kedi")

    assert [diagnostic.code for diagnostic in diagnostics] == ["K-OUTPUT-OPEN-DICT"]


@pytest.mark.parametrize(
    "type_expr",
    [TypeRef(name="dict"), TypeApply(name="dict", args=[TypeRef(name="str"), TypeRef(name="int")])],
)
def test_open_dict_output_without_type_span_is_skipped(
    type_expr: TypeExpr, monkeypatch: pytest.MonkeyPatch
) -> None:
    program = Program(
        toplevel_body=[TemplateLine(segments=[OutputSegment(name="result", type_expr=type_expr)])]
    )
    monkeypatch.setattr(diagnostics_module, "parse_program", lambda *_args, **_kwargs: program)

    assert type_expr.loc is None
    assert compute_diagnostics(">> Extract [result: dict].\n", "example.kedi") == []


@pytest.mark.parametrize("annotation", ["Record", "`Record`", 'Annotated[Record, "Known keys"]'])
@pytest.mark.parametrize(
    "declaration",
    [
        "~Record(name, age: int)\n",
        "```\nfrom typing_extensions import TypedDict\n"
        "class Record(TypedDict):\n    name: str\n    age: int\n```\n",
    ],
)
def test_fixed_shape_model_outputs_do_not_warn(annotation: str, declaration: str) -> None:
    source = declaration + f">> Extract [result: {annotation}].\n"

    assert compute_diagnostics(source, "example.kedi") == []


def test_deterministic_dicts_and_non_capture_types_do_not_warn() -> None:
    source = (
        "```\nplain = {'name': 'Ada'}\n"
        "def make_record() -> dict:\n    return plain\n```\n"
        "~Record(data: dict)\n"
        "> profile: worker:\n  > output: dict\n"
        "[local: dict[str, str]] = `{'name': 'Ada'}`\n"
        "[block: dict] = ```\nreturn {'name': 'Ada'}\n```\n"
        "[local] := `{'name': 'Grace'}`\n"
        "@identity(value: dict) -> dict:\n  = `value`\n"
    )

    assert compute_diagnostics(source, "example.kedi") == []


@pytest.mark.parametrize(
    "source",
    [
        ">> Extract [result].\n",
        "[result] << Extract data.\n",
        ">> Extract [result: str].\n",
        "[result: str] << Extract data.\n",
        '>> Extract [result: Literal["dict"]].\n',
    ],
)
def test_non_dict_model_outputs_do_not_get_dict_warning(source: str) -> None:
    diagnostics = compute_diagnostics(source, "example.kedi")

    assert all(diagnostic.code != "K-OUTPUT-OPEN-DICT" for diagnostic in diagnostics)
    assert all(diagnostic.severity != lsp.DiagnosticSeverity.Error for diagnostic in diagnostics)


def test_dict_in_annotated_metadata_is_not_a_dict_output() -> None:
    diagnostics = compute_diagnostics(
        ">> Extract [result: Annotated[str, dict]].\n", "example.kedi"
    )

    assert [diagnostic.code for diagnostic in diagnostics] == ["K-ANNOTATED-DESCRIPTION"]


def test_open_dict_output_lint_does_not_execute_python(tmp_path: Path) -> None:
    marker = tmp_path / "lint-executed"
    side_effect = f"__import__('pathlib').Path({str(marker)!r}).touch()"
    source = (
        f"```\n{side_effect}\n```\n"
        f">> Extract [result: `{side_effect} or dict`].\n"
        f'>> Extract [described: Annotated[dict, "Dynamic map", `{side_effect}`]].\n'
    )

    diagnostics = compute_diagnostics(source, "example.kedi")

    assert not marker.exists()
    assert [diagnostic.code for diagnostic in diagnostics] == [
        "K-ANNOTATED-EXTRA",
        "K-OUTPUT-OPEN-DICT",
    ]
