"""Kedi #214 (6.12): signature help label offsets and control-body calls.

LSP signature-label offsets are UTF-16 code units, so a non-BMP
character in a parameter default must not shift later parameters. And
a legal call inside a control-flow body (``> if:``) is still a call the
user is typing: signature help and inlay hints must find it.
"""

from __future__ import annotations

from tests.lsp_helpers import inlay_hints_for_lines, signature_help_at

PATH = "/tmp/x.kedi"


def _utf16_slice(label: str, start: int, end: int) -> str:
    return label.encode("utf-16-le")[start * 2 : end * 2].decode("utf-16-le")


def test_signature_label_offsets_are_utf16_units() -> None:
    src = '@p(a: str = `"😀"`, b: int = `2`) -> int:\n  = `a`\n\n= <p("x", 2)>\n'
    sig = signature_help_at(src, PATH, "p", occurrence=2)
    assert sig is not None
    label = sig.signatures[0].label
    params = sig.signatures[0].parameters or []
    assert len(params) == 2
    assert _utf16_slice(label, *params[0].label) == 'a: str = `"😀"`'
    assert _utf16_slice(label, *params[1].label) == "b: int = `2`"


def test_signature_help_inside_if_body() -> None:
    src = "@inner(x: int) -> int:\n  = `x`\n\n@outer():\n  > if: `True`:\n    = <inner(1)>\n"
    sig = signature_help_at(src, PATH, "inner", occurrence=2)
    assert sig is not None
    assert sig.signatures[0].label.startswith("@inner(")
    assert sig.active_parameter == 0


def test_signature_help_inside_top_level_if_body() -> None:
    src = "@inner(x: int) -> int:\n  = `x`\n\n> if: `True`:\n  = <inner(1)>\n"
    sig = signature_help_at(src, PATH, "inner", occurrence=2)
    assert sig is not None
    assert sig.signatures[0].label.startswith("@inner(")


def test_signature_help_inside_loop_body() -> None:
    src = "@inner(x: int) -> int:\n  = `x`\n\n@outer():\n  > loop: `True`:\n    = <inner(1)>\n"
    sig = signature_help_at(src, PATH, "inner", occurrence=2)
    assert sig is not None


def test_inlay_hints_inside_if_body() -> None:
    src = "@inner(x: int) -> int:\n  = `x`\n\n@outer():\n  > if: `True`:\n    = <inner(1)>\n"
    hints = inlay_hints_for_lines(src, PATH, 0, 10)
    assert any(hint.label == "x: " for hint in hints)
