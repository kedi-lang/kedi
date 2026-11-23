"""Kedi #214 (6.23): procedures defined inside top-level blocks.

A procedure inside a top-level ``> if:`` / ``> loop:`` block is a
``ProcDefStmt`` in the top-level body, not an entry of
``prog.procedures``. Every consumer of ``_procedure_scope_entries``
(assignment and substitution-type diagnostics, completion, media)
must see it, exactly like a directly declared procedure. Type
diagnostics already covered these procedures via ``_all_procedures``;
these tests pin the ones that did not.
"""

from __future__ import annotations

from kedi.lsp.features.scope import _procedure_scope_entries
from kedi.lsp.parsing import parse_program
from tests.lsp_helpers import diagnostics_for

PATH = "/tmp/x.kedi"


def test_scope_entries_include_procedure_in_top_level_block() -> None:
    src = "> if: `True`:\n  @inner():\n    = ok\n\n@direct():\n  = ok\n"
    prog = parse_program(src, source_path=PATH)
    names = [proc.name for proc, _ in _procedure_scope_entries(prog)]
    assert names == ["direct", "inner"]


def test_unknown_assignment_in_top_level_block_procedure_is_flagged() -> None:
    src = "> if: `True`:\n  @inner():\n    [nope] := `1`\n    = ok\n"
    diags = diagnostics_for(src, PATH)
    assert [(d.code, d.range.start.line) for d in diags] == [("K-ASSIGN-UNKNOWN", 2)]


def test_substitution_type_in_top_level_block_procedure_is_flagged() -> None:
    direct = "~Item(value: int)\n@p():\n  [d: Item] = `Item(value=1)`\n  >> Value <d>\n  = ok\n"
    assert [(d.code, d.range.start.line) for d in diagnostics_for(direct, PATH)] == [
        ("K-SUBSTITUTION-TYPE", 3)
    ]
    topif = (
        "~Item(value: int)\n"
        "> if: `True`:\n"
        "  @p():\n"
        "    [d: Item] = `Item(value=1)`\n"
        "    >> Value <d>\n"
        "    = ok\n"
    )
    assert [(d.code, d.range.start.line) for d in diagnostics_for(topif, PATH)] == [
        ("K-SUBSTITUTION-TYPE", 4)
    ]


def test_unsupported_type_in_top_level_block_procedure_is_flagged() -> None:
    src = "> if: `True`:\n  @inner(x: Tuple):\n    = <x>\n"
    diags = diagnostics_for(src, PATH)
    assert any(d.code == "K-TYPE-UNSUPPORTED" for d in diags)


def test_legal_cases_in_top_level_block_procedure_are_not_flagged() -> None:
    # Local typedef, raw invoke, and a conditional loop are all legal
    # inside a procedure in a top-level block.
    src = (
        "> if: `True`:\n"
        "  @p():\n"
        "    ~Local(name: str)\n"
        "    [v: Local] = ```\n"
        '    return Local(name="x")\n'
        "    ```\n"
        "    > loop: `True`:\n"
        "      [w] = <`v.name`>\n"
        "    = <`v.name`>\n"
    )
    diags = diagnostics_for(src, PATH)
    assert [d.code for d in diags] == []
