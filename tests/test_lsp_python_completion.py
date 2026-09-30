"""Completion edits must be executable in the source, not the shadow file."""

from copy import deepcopy
import io
import json
import sys

import pytest

from kedi.lang.parser import parse_program
from kedi.lsp.coordinates import utf16_to_index
from kedi.lsp.python_virtual.completion import (
    _import_addition,
    _imports_edit,
    _map_edit,
    _mapped_range,
    map_python_completions,
)
from kedi.lsp.python_virtual.document import compute_python_virtual_document


def _range(line, start, end):
    return {"start": {"line": line, "character": start}, "end": {"line": line, "character": end}}


def _edit(virtual, token, new_text):
    for index, line in enumerate(virtual["text"].splitlines()):
        if token in line:
            start = line.index(token)
            return {"range": _range(index, start, start + len(token)), "newText": new_text}
    raise AssertionError(token)


def _apply(source, edits):
    lines = source.splitlines(keepends=True)

    def offset(position):
        line, character = position["line"], position["character"]
        return sum(map(len, lines[:line])) + utf16_to_index(lines[line], character)

    for edit in sorted(edits, key=lambda e: offset(e["range"]["start"]), reverse=True):
        a, b = edit["range"]["start"], edit["range"]["end"]
        source = source[: offset(a)] + edit["newText"] + source[offset(b) :]
    return source


@pytest.mark.parametrize(
    "source,focus",
    [
        ("```\nvalue = 'hello'\nvalue.upp\n```\n", 2),
        ("@name(value: str) -> str:\n  = `value.upp`\n", 1),
        ("@name(value: str) -> str:\n  [answer] = ```\n  return value.upp\n  ```\n  = answer\n", 2),
    ],
)
def test_completion_preserves_snippets_and_source_scope(source, focus):
    virtual = compute_python_virtual_document(source, focus_line=focus)
    item = {
        "label": "upper",
        "insertTextFormat": 2,
        "textEdit": _edit(virtual, "upp", "upper(${1:arg})"),
        "command": {"command": "shadow-only"},
    }
    before = deepcopy(item)
    result = map_python_completions(source, virtual, [item])
    assert result[0]["insertTextFormat"] == 2
    assert "command" not in result[0]
    assert _apply(source, [result[0]["textEdit"]]) == source.replace("upp", "upper(${1:arg})")
    assert item == before


@pytest.mark.parametrize("tail", ["> show: `capital.upp", "> show: <`capital.upp"])
def test_unfinished_python_keeps_native_bindings_and_exact_mapping(tail):
    source = "[capital: str] = Ankara\n" + tail
    virtual = compute_python_virtual_document(source, focus_line=1)
    assert "capital: str" in virtual["text"]
    item = {"label": "upper", "textEdit": _edit(virtual, "upp", "upper")}
    mapped = map_python_completions(source, virtual, [item])
    assert _apply(source, [mapped[0]["textEdit"]]) == source.replace("upp", "upper")


def test_unfinished_python_retains_procedure_parameters_and_locals():
    source = "@echo(value: str):\n  [local] = ready\n  > show: `value.upp"
    virtual = compute_python_virtual_document(source, focus_line=2)
    assert "__kedi_scope_echo(value: str)" in virtual["text"]
    assert "local: str" in virtual["text"]
    assert "value.upp" in virtual["text"]
    assert virtual["focusKey"] == "proc:echo:1"


def test_completion_recovery_does_not_change_strict_parser():
    from kedi.lang.parser import KediParseError
    from kedi.lsp.completion_source import parse_completion_source

    source = "[capital] = Ankara\n> show: <cap"
    assert parse_completion_source(source, None, 1).toplevel_body
    for line in (None, -1, 50):
        with pytest.raises(KediParseError):
            parse_completion_source(source, None, line)
    with pytest.raises(KediParseError):
        parse_program(source)


@pytest.mark.parametrize(
    "source,focus",
    [
        ("= `Path('.')`\n", 0),
        ("```\nfrom __future__ import annotations\nvalue = 1\n```\n= `Path('.')`\n", 4),
        ("@make() -> str:\n  = `Path('.')`\n", 1),
    ],
)
def test_auto_imports_go_to_module_prelude_even_when_focus_is_elsewhere(source, focus):
    virtual = compute_python_virtual_document(source, focus_line=focus)
    item = {
        "label": "Path",
        "textEdit": _edit(virtual, "Path", "Path"),
        "additionalTextEdits": [
            {"range": _range(0, 0, 0), "newText": "from pathlib import Path\n"}
        ],
    }
    result = map_python_completions(source, virtual, [item])[0]
    output = _apply(source, [result["textEdit"], *result["additionalTextEdits"]])
    prelude = parse_program(output).prelude
    assert prelude is not None
    assert "from pathlib import Path" in prelude
    if "__future__" in source:
        assert prelude.index("__future__") < prelude.index("pathlib")
        assert output.count("```") == 2


def test_auto_import_addition_to_synthetic_typing_import():
    source = "= `Sequence`\n"
    virtual = compute_python_virtual_document(source)
    virtual["text"] = "from typing import Any\n" + virtual["text"]
    for entry in virtual["mappings"]:
        for point in entry["virtualRange"].values():
            point["line"] += 1
    item = {
        "label": "Sequence",
        "textEdit": _edit(virtual, "Sequence", "Sequence"),
        "additionalTextEdits": [{"range": _range(0, 22, 22), "newText": ", Sequence"}],
    }
    mapped = map_python_completions(source, virtual, [item])[0]
    assert (
        mapped["additionalTextEdits"][0]["newText"] == "```\nfrom typing import Sequence\n```\n\n"
    )


def test_unmappable_edit_rejects_entire_suggestion():
    source = "= `str(1)`\n"
    virtual = compute_python_virtual_document(source)
    item = {
        "label": "str",
        "textEdit": _edit(virtual, "str(1)", "str(2)"),
        "additionalTextEdits": [{"range": _range(0, 0, 1), "newText": "unsafe"}],
    }
    assert map_python_completions(source, virtual, [item]) == []
    item.pop("additionalTextEdits")
    item["textEdit"] = {"range": _range(0, 0, 1), "newText": "unsafe"}
    assert map_python_completions(source, virtual, [item]) == []


def test_insert_replace_at_end_and_additional_mapped_edit():
    source = "```\nvalue = 'test'\nvalue.upp\n```\n"
    virtual = compute_python_virtual_document(source)
    edit = _edit(virtual, "upp", "upper")
    item = {
        "label": "upper",
        "textEdit": {"newText": "upper", "insert": edit["range"], "replace": edit["range"]},
        "additionalTextEdits": [_edit(virtual, "'test'", "'example'")],
    }
    mapped = map_python_completions(source, virtual, [item])[0]
    assert mapped["textEdit"]["insert"] == _range(2, 6, 9)
    assert mapped["textEdit"]["replace"] == _range(2, 6, 9)
    assert mapped["additionalTextEdits"][0]["range"] == _range(1, 8, 14)


def test_unicode_before_inline_python_keeps_utf16_offsets():
    source = '> show: "\U0001f680" <`str(1)`>\n'
    virtual = compute_python_virtual_document(source)
    assert "str(1)" in virtual["text"]
    item = {"label": "str", "textEdit": _edit(virtual, "str(1)", "str(2)")}
    mapped = map_python_completions(source, virtual, [item])[0]
    assert _apply(source, [mapped["textEdit"]]) == source.replace("str(1)", "str(2)")


def test_multiline_snippet_reindents_procedure_body():
    source = "@make() -> int:\n  ```\n  val\n  ```\n  = `1`\n"
    virtual = compute_python_virtual_document(source, focus_line=2)
    edit = _edit(virtual, "val", "value = (\n        ${1:1}\n    )")
    mapped = map_python_completions(source, virtual, [{"label": "value", "textEdit": edit}])[0]
    assert mapped["textEdit"]["newText"] == "value = (\n      ${1:1}\n  )"


def test_completion_projection_custom_request_uses_deserialized_protocol_params():
    from kedi.lsp.server import make_server

    source = "= `str(1)`\n"
    virtual = compute_python_virtual_document(source)
    item = {"label": "str", "textEdit": _edit(virtual, "str(1)", "str(2)")}
    server = make_server()
    message = server.protocol.structure_message(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "kedi/pythonCompletions",
            "params": {
                "textDocument": {"uri": "file:///tmp/example.kedi"},
                "text": source,
                "virtual": virtual,
                "items": [item],
            },
        }
    )
    result = server.protocol.fm.features["kedi/pythonCompletions"](message.params)
    assert _apply(source, [result[0]["textEdit"]]) == "= `str(2)`\n"


def test_completion_custom_request_without_a_document_is_empty():
    from kedi.lsp.server import make_server

    server = make_server()
    assert server.protocol.fm.features["kedi/pythonCompletions"]({}) == []


def test_persistent_projection_worker_handles_completion_requests(monkeypatch):
    from kedi.lsp.python_virtual.document import main_loop

    source = "= `str(1)`\n"
    virtual = compute_python_virtual_document(source)
    payload = {
        "id": 2,
        "method": "completion",
        "text": source,
        "virtual": virtual,
        "items": [{"label": "str", "textEdit": _edit(virtual, "str(1)", "str(2)")}],
    }
    output = io.StringIO()
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(payload) + "\n"))
    monkeypatch.setattr(sys, "stdout", output)
    main_loop()
    response = json.loads(output.getvalue())
    assert response["id"] == 2
    assert response["ok"]
    assert _apply(source, [response["result"][0]["textEdit"]]) == "= `str(2)`\n"


def test_import_at_primary_insertion_is_merged_without_overlapping_edits():
    source = "```\nPath\n```\n"
    virtual = compute_python_virtual_document(source)
    item = {
        "label": "Path",
        "textEdit": _edit(virtual, "Path", "Path"),
        "additionalTextEdits": [
            {"range": _range(0, 0, 0), "newText": "from pathlib import Path\n"}
        ],
    }
    result = map_python_completions(source, virtual, [item])[0]
    assert result["additionalTextEdits"] == []
    assert _apply(source, [result["textEdit"]]) == "```\nfrom pathlib import Path\nPath\n```\n"


def _two_line_mapping():
    return {
        "mappings": [
            {"virtualRange": _range(0, 0, 3), "sourceRange": _range(2, 2, 5)},
            {"virtualRange": _range(1, 0, 3), "sourceRange": _range(3, 2, 5)},
        ]
    }


def test_multiline_edits_require_contiguous_source_lines_with_identical_indentation():
    virtual = _two_line_mapping()
    range_ = {"start": {"line": 0, "character": 1}, "end": {"line": 1, "character": 2}}
    assert _mapped_range(virtual, range_) == {
        "start": {"line": 2, "character": 3},
        "end": {"line": 3, "character": 4},
    }
    for change in ("missing", "source_gap", "indent"):
        broken = deepcopy(virtual)
        if change == "missing":
            broken["mappings"].pop()
        elif change == "source_gap":
            broken["mappings"][1]["sourceRange"] = _range(4, 2, 5)
        else:
            broken["mappings"][1]["sourceRange"] = _range(3, 3, 6)
        assert _mapped_range(broken, range_) is None
    outside = deepcopy(range_)
    outside["end"]["character"] = 4
    assert _mapped_range(virtual, outside) is None


@pytest.mark.parametrize(
    "text,expected",
    [("one\ntwo\n", "one\n  two\n"), ("one\n\ntwo", "one\n\n  two")],
)
def test_multiline_snippet_adds_only_source_indentation_to_nonempty_lines(text, expected):
    mapped = _map_edit(_two_line_mapping(), {"range": _range(0, 0, 3), "newText": text})
    assert mapped["newText"] == expected


def test_multiline_snippet_does_not_strip_content_without_expected_indentation():
    virtual = {"mappings": [{"virtualRange": _range(0, 2, 5), "sourceRange": _range(0, 0, 3)}]}
    mapped = _map_edit(virtual, {"range": _range(0, 2, 5), "newText": "one\ntwo"})
    assert mapped["newText"] == "one\ntwo"


@pytest.mark.parametrize(
    "text,range_,new_text,expected",
    [
        ("import os", _range(5, 0, 0), "import sys\n", None),
        (
            "import os\nimport sys",
            {"start": {"line": 0, "character": 0}, "end": {"line": 1, "character": 0}},
            "import pathlib\n",
            None,
        ),
        ("import os", _range(0, 0, 9), "from", None),
        ("value = 1", _range(0, 0, 9), "value = 2", None),
        ("import os", _range(0, 0, 9), "import os", None),
        ("import os", _range(0, 0, 9), "import os, sys", "import sys"),
        ("import os", _range(0, 0, 0), "from __future__ import annotations\n", None),
        ("import os", _range(0, 0, 9), "", None),
    ],
)
def test_only_new_valid_imports_can_leave_the_shadow_document(text, range_, new_text, expected):
    assert _import_addition(text, {"range": range_, "newText": new_text}) == expected


def test_auto_import_handles_invalid_kedi_and_incomplete_python_without_a_loop():
    invalid = "> completely_unknown: broken\n"
    edit = _imports_edit(invalid, ["import pathlib"])
    assert edit["range"] == _range(0, 0, 0)
    assert edit["newText"] == "```\nimport pathlib\n```\n\n"
    incomplete = "```\nimport os\nvalue = (\n```\n"
    edit = _imports_edit(incomplete, ["import pathlib"])
    assert edit["range"] == _range(2, 0, 0)
    assert edit["newText"] == "import pathlib\n"


@pytest.mark.parametrize(
    "source,line",
    [
        ("```\n```\n", 1),
        ('```\n"Module documentation."\nfrom __future__ import annotations\n```\n', 3),
    ],
)
def test_auto_import_preserves_prelude_docstring_and_future_header(source, line):
    edit = _imports_edit(source, ["import pathlib"])
    assert edit["range"] == _range(line, 0, 0)
    assert edit["newText"] == "import pathlib\n"


def test_label_only_completion_and_crlf_import_insertion():
    item = {"label": "Path"}
    source = "= `1`\r\n"
    result = map_python_completions(source, compute_python_virtual_document(source), [item])
    assert result == [{"label": "Path", "additionalTextEdits": []}]
    assert item == {"label": "Path"}
    edit = _imports_edit(source, ["import pathlib", "import pathlib"])
    assert edit["newText"] == "```\r\nimport pathlib\r\n```\r\n\r\n"
