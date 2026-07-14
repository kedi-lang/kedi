from __future__ import annotations

import copy
import importlib
import runpy
from pathlib import Path
from typing import Any, cast

import pytest

from kedi.lang.ast import ProcedureDef, Span, TypeApply, TypePython, TypeRef, TypeUnion
from kedi.lang.parser import parser as parser_module
from kedi.lang.parser.diagnostics import KediDiagnostic, KediParseError
from kedi.lang.type_introspection import TypeInitMetadata
from kedi.lang.type_render import (
    procedure_param_type_to_python,
    procedure_return_type_to_python,
    type_expr_to_python,
)
from kedi.source_loading import (
    SourceDocument,
    load_kedi_source_bundle,
    merge_source_documents,
)
from kedi.utils import AttrDict, parse_args


def test_kedi_diagnostic_format_without_and_with_snippet() -> None:
    span = Span(0, 4, 2, 3, 2, 7, source_path="sample.kedi")
    diagnostic = KediDiagnostic(
        severity="error",
        message="bad source",
        span=span,
        code="K123",
    )

    assert diagnostic.format() == "sample.kedi:2:3: error: bad source [K123]"

    snippet = KediDiagnostic(
        severity="warning",
        message="odd source",
        span=Span(0, 1, 1, 0, 1, 0),
        snippet="abc",
    ).format()

    assert "<source>:1:0: warning: odd source" in snippet
    assert "abc" in snippet
    assert "^" in snippet


def test_kedi_parse_error_preserves_empty_and_primary_messages() -> None:
    assert str(KediParseError([])) == "kedi parse error"

    error = KediParseError(
        [
            KediDiagnostic(
                severity="error",
                message="first",
                span=Span(0, 1, 1, 1, 1, 2),
            ),
            KediDiagnostic(
                severity="error",
                message="second",
                span=Span(2, 3, 2, 1, 2, 2),
            ),
        ]
    )

    assert str(error) == "first"
    assert [diagnostic.message for diagnostic in error.diagnostics] == [
        "first",
        "second",
    ]


def test_type_render_handles_structured_legacy_and_default_types() -> None:
    assert type_expr_to_python(TypePython(code="tuple[int, str]"), None) == "tuple[int, str]"
    assert type_expr_to_python(TypePython(code=""), None, default="Any") == "Any"
    assert type_expr_to_python(TypeRef(name="Person"), None) == "Person"
    assert type_expr_to_python(TypeRef(name=""), None, default="str") == "str"
    assert (
        type_expr_to_python(
            TypeApply(name="dict", args=[TypeRef(name="str"), TypeRef(name="int")]),
            None,
        )
        == "dict[str, int]"
    )
    assert type_expr_to_python(TypeApply(name="", args=[]), None, default="str") == "str"
    assert (
        type_expr_to_python(TypeUnion(left=TypeRef(name="int"), right=TypeRef(name="str")), None)
        == "int | str"
    )
    assert type_expr_to_python(None, "list[str]") == "list[str]"
    assert type_expr_to_python(None, None, default="Any") == "Any"


def test_procedure_type_render_uses_param_and_return_metadata() -> None:
    proc = ProcedureDef(
        name="make",
        params=["name", "age", "raw"],
        param_type_exprs={"name": TypeRef(name="str"), "raw": None},
        param_type_annotations={"age": "int"},
        return_type_expr=TypeApply(name="list", args=[TypeRef(name="str")]),
    )

    assert procedure_param_type_to_python(proc, "name") == "str"
    assert procedure_param_type_to_python(proc, "age") == "int"
    assert procedure_param_type_to_python(proc, "raw", default="Any") == "Any"
    assert procedure_param_type_to_python(proc, "missing", default="object") == "object"
    assert procedure_return_type_to_python(proc) == "list[str]"


def test_tree_sitter_parser_handles_parser_initialized_inside_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_parser = parser_module._parser
    fake_parser = object()

    class RaceLock:
        def __enter__(self) -> None:
            cast(Any, parser_module)._parser = fake_parser

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

    monkeypatch.setattr(parser_module, "_parser", None)
    monkeypatch.setattr(parser_module, "_lock", RaceLock())

    assert parser_module._get_parser() is fake_parser

    monkeypatch.setattr(parser_module, "_parser", original_parser)


def test_utils_parse_args_and_attrdict() -> None:
    parsed = parse_args(
        ["--count", "1", "--count", "2", "positional", "--enabled", "--name", "Ada"]
    )

    assert isinstance(parsed, AttrDict)
    assert parsed.count == "1"
    assert parsed.enabled is True
    assert parsed.name == "Ada"
    assert parsed.missing is None

    plain = AttrDict({"present": "ok"})
    assert plain.present == "ok"
    with pytest.raises(AttributeError):
        _ = plain.missing
    with pytest.raises(AttributeError):
        _ = plain.__deepcopy__

    copied = copy.copy(AttrDict())
    with pytest.raises(AttributeError):
        _ = copied.missing


def test_source_loading_maps_empty_and_missing_lines(tmp_path: Path) -> None:
    empty = SourceDocument(path="empty.kedi", text="", start_line_in_merged=1)
    assert empty.line_count == 1

    first = tmp_path / "first.kedi"
    second = tmp_path / "second.kedi"
    loaded = merge_source_documents([(first, "one\ntwo"), (second, "three")])

    assert loaded.source_map.resolve_line(0) is None
    assert loaded.source_map.get_line(str(tmp_path / "absent.kedi"), 1) is None
    assert loaded.source_map.get_line(str(first), 99) is None
    assert loaded.source_map.has_path(str(second))
    assert not loaded.source_map.has_path(str(tmp_path / "missing.kedi"))

    root = tmp_path / "main.kedi"
    root.write_text("= ok", encoding="utf-8")
    root.with_suffix(".val.kedi").write_text("@test: main\n", encoding="utf-8")
    root.with_suffix(".cache.kedi").write_text("@cached():\n  = ok\n", encoding="utf-8")
    bundled = load_kedi_source_bundle(root, include_val=False, include_cache=True)

    assert bundled.text == "= ok\n\n@cached():\n  = ok\n"
    assert bundled.source_map.has_path(str(root.resolve()))
    assert not bundled.source_map.has_path(str(root.with_suffix(".val.kedi").resolve()))
    assert bundled.source_map.has_path(str(root.with_suffix(".cache.kedi").resolve()))

    without_cache = load_kedi_source_bundle(root, include_val=False, include_cache=False)
    assert without_cache.text == "= ok"


def test_tree_sitter_parser_public_helpers_accept_str_and_bytes() -> None:
    language = parser_module.get_language()
    tree_from_str = parser_module.parse("= ok\n")
    tree_from_bytes = parser_module.parse(b"= ok\n")

    assert language is parser_module.get_language()
    assert tree_from_str.root_node.type == "source_file"
    assert tree_from_bytes.root_node.type == "source_file"


def test_type_init_metadata_handles_empty_and_defaulted_fields() -> None:
    empty = TypeInitMetadata(
        name="Empty",
        fields=(),
        field_types=(),
        default_sources=(),
        default_values=(),
    )

    assert empty.display_source() == (
        "def __init__(self) -> None:\n"
        '    """This stub is generated for Python introspection."""\n'
        "    __kedi_model_init__(self)\n"
    )
    assert "class Empty:" in empty.display_class_source()
    assert "def __init__(self) -> None:" in empty.executable_source()
    assert empty.default_namespace() == {}

    defaulted = TypeInitMetadata(
        name="Person",
        fields=("name",),
        field_types=("str",),
        default_sources=("'Ada'",),
        default_values=("Ada",),
    )
    namespace = defaulted.default_namespace()

    assert repr(namespace["__kedi_default_0"]) == "'Ada'"
    value = namespace["__kedi_default_0"].value()
    assert isinstance(value, str)


def test_type_init_metadata_deep_copies_default_values() -> None:
    metadata = TypeInitMetadata(
        name="Bag",
        fields=("items",),
        field_types=("list[str]",),
        default_sources=("['a']",),
        default_values=(["a"],),
    )
    default_value = metadata.default_namespace()["__kedi_default_0"]

    first = default_value.value()
    second = default_value.value()
    first.append("b")

    assert first == ["a", "b"]
    assert second == ["a"]


def test_package_main_and_settings_module(monkeypatch: pytest.MonkeyPatch) -> None:
    import kedi.cli
    import kedi.settings

    monkeypatch.setattr(kedi.cli, "main", lambda: 7)

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_module("kedi.__main__", run_name="__main__")

    assert exc_info.value.code == 7

    main_module = importlib.import_module("kedi.__main__")
    dotenv_calls: list[bool] = []
    monkeypatch.setattr(main_module.dotenv, "load_dotenv", lambda: dotenv_calls.append(True))

    assert main_module.main() == 7
    assert dotenv_calls == [True]

    settings_module = importlib.reload(kedi.settings)
    assert settings_module.settings.adapter == "pydantic"
    assert settings_module.settings.debug is False
