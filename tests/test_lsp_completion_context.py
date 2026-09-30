"""Completions must not replace syntax or hijack newline and fence entry."""

from __future__ import annotations

import pytest
from lsprotocol import types as lsp

from kedi.lsp.features.completion import compute_completion


def complete(source: str, character: int | None = None) -> list[lsp.CompletionItem]:
    lines = source.split("\n")
    return compute_completion(
        source,
        "/tmp/example.kedi",
        lsp.Position(
            line=len(lines) - 1, character=character if character is not None else len(lines[-1])
        ),
    )


@pytest.mark.parametrize("source", [">", "> ", ">mo", "> mo", "  > mo", "\t> mo"])
def test_directive_edit_preserves_indent_and_exactly_one_marker(source: str) -> None:
    item = next(item for item in complete(source) if item.label == "model")
    assert isinstance(item.text_edit, lsp.TextEdit)
    edit = item.text_edit
    applied = (
        source[: edit.range.start.character] + edit.new_text + source[edit.range.end.character :]
    )
    indent = source[: len(source) - len(source.lstrip())]
    assert applied == indent + "> model: "
    assert item.detail == "Select the model for this lexical scope"
    assert item.filter_text is not None and item.filter_text.startswith(source.lstrip())


def test_directive_edit_replaces_remaining_name_and_existing_colon() -> None:
    source = "  > model: test"
    item = next(item for item in complete(source, 6) if item.label == "model")
    assert isinstance(item.text_edit, lsp.TextEdit)
    edit = item.text_edit
    assert (
        source[: edit.range.start.character] + edit.new_text + source[edit.range.end.character :]
        == source
    )


def test_every_directive_explains_its_purpose() -> None:
    items = complete("> ")
    assert len({item.detail for item in items}) == len(items)
    assert all(item.detail and "Kedi directive" not in item.detail for item in items)


@pytest.mark.parametrize(
    "source",
    [
        "> model:",
        "> model: ",
        "> model: openai:",
        "> profile: worker:",
        "> if: ready:",
        "> task [job]: ",
        "> await: ",
        "> await [result]: ",
        "> history:\n    enabled: ",
        "> history:\n    enabled: en",
        "> codemode:\n    preload_tools:",
        "> budget:\n    request_limit: ",
        "@work():\n    > tool:\n        description: ",
        "`",
        "``",
        "```",
        "[x] = `",
        "[x] = `value`",
        "> settings:\n    temperature: `",
        "# > mo",
        ">> Plain template text",
        "> system: Be concise",
        "> model: openai",
    ],
)
def test_no_native_completions_in_values_fences_or_plain_template_text(source: str) -> None:
    assert complete(source) == []


@pytest.mark.parametrize("source", ["[result: ", ">> The result is [result: ", "> if: Ready <res"])
def test_capture_types_and_template_inputs_still_complete(source: str) -> None:
    assert "str" in {item.label for item in complete(source)}


@pytest.mark.parametrize("source", ["[x] = `value`", "> settings:\n    temperature: `value`"])
def test_native_completions_do_not_compete_with_python_at_expression_end(source: str) -> None:
    assert complete(source, len(source.split("\n")[-1]) - 1) == []


@pytest.mark.parametrize("tail", [">> The city is <cap", "> show: <cap", "[copy] = <cap"])
def test_unfinished_substitution_keeps_prior_bindings(tail):
    source = (
        "```\nfrom pathlib import Path\nlabel = 'ready'\n```\n"
        "[capital] = Ankara\n"
        ">> The population is [population: int].\n"
        "[summary] << Summarize the city.\n"
        "@echo(value: str):\n  = <value>\n" + tail
    )
    names = {item.label for item in complete(source)}
    assert {"capital", "population", "summary", "echo", "Path", "label"} <= names


def test_completion_keeps_lexical_ancestors_without_leaking_sibling_locals():
    source = (
        "[global_value] = ready\n"
        "@unrelated(secret):\n  [hidden] = private\n  @hidden_proc():\n    = private\n"
        "@outer(parent):\n"
        "  [outer_value] = ready\n"
        "  @inner(child):\n"
        "    [local_value] = ready\n"
        "    > show: <loc"
    )
    names = {item.label for item in complete(source)}
    assert {
        "global_value",
        "parent",
        "outer_value",
        "child",
        "local_value",
        "outer",
        "inner",
    } <= names
    assert not {"secret", "hidden", "hidden_proc"} & names


def test_completion_recovers_before_an_unfinished_later_statement():
    source = "[capital] = Ankara\n> show: <cap\n>> The next result is ["
    names = {item.label for item in compute_completion(source, None, lsp.Position(1, 12))}
    assert "capital" in names


def test_completion_does_not_execute_python_to_discover_names(tmp_path):
    marker = tmp_path / "must-not-exist"
    source = f"```\nfrom pathlib import Path\nPath({str(marker)!r}).touch()\n```\n> show: <Pa"
    assert "Path" in {item.label for item in complete(source)}
    assert not marker.exists()


def test_unrecoverable_prior_source_still_offers_builtin_types():
    names = {item.label for item in complete("~Broken(\n>> The value is [value: fl")}
    assert "float" in names


def test_procedure_python_block_names_are_visible_only_in_its_scope():
    source = "@outer():\n  ```\n  python_value = 'ready'\n  ```\n  > show: <python_v"
    assert "python_value" in {item.label for item in complete(source)}


def test_control_flow_type_discovery_and_completion_agree():
    from kedi.lang.parser import parse_program
    from kedi.lsp.features.scope import _all_type_defs

    source = (
        "> if: `True`:\n"
        "  ~Visible(value: str)\n"
        "  @factory():\n"
        "    ~Local(value: int)\n"
        "    = ready\n"
        "> show: <value>\n"
    )
    assert {t.name for t in _all_type_defs(parse_program(source))} == {"Visible", "Local"}
    items = compute_completion(source, None, lsp.Position(6, 14))
    assert {"Visible", "factory"} <= {item.label for item in items}
    assert "Local" not in {item.label for item in items}
