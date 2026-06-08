"""Shared helpers for LSP feature tests."""

from __future__ import annotations

from collections.abc import Sequence

from lsprotocol import types as lsp


def decode_semantic_tokens(
    data: Sequence[int],
) -> list[tuple[int, int, int, int, int]]:
    out: list[tuple[int, int, int, int, int]] = []
    line = 0
    col = 0
    for d_line, d_col, length, token_type, token_mods in zip(
        data[0::5], data[1::5], data[2::5], data[3::5], data[4::5], strict=False
    ):
        line += d_line
        col = col + d_col if d_line == 0 else d_col
        out.append((line, col, length, token_type, token_mods))
    return out


def hover_value(hover: lsp.Hover | None) -> str:
    assert hover is not None
    assert isinstance(hover.contents, lsp.MarkupContent)
    return hover.contents.value


def position_of(source: str, needle: str, *, occurrence: int = 1, offset: int = 0) -> lsp.Position:
    start = 0
    for _ in range(occurrence):
        index = source.index(needle, start)
        start = index + len(needle)
    index += offset
    before = source[:index]
    return lsp.Position(
        line=before.count("\n"),
        character=len(before.rsplit("\n", 1)[-1]),
    )
