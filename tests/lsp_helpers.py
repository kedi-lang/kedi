"""Shared helpers for LSP feature tests."""

from __future__ import annotations

from collections.abc import Sequence

from lsprotocol import types as lsp

from kedi.lsp import features


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


def span_at(source: str, needle: str, *, occurrence: int = 1) -> tuple[lsp.Position, lsp.Position]:
    """Return start/end positions covering ``needle`` in ``source``."""
    start = position_of(source, needle, occurrence=occurrence)
    end = position_of(source, needle, occurrence=occurrence, offset=len(needle))
    return start, end


def hover_at(
    source: str,
    source_path: str,
    needle: str,
    *,
    occurrence: int = 1,
    offset: int = 0,
) -> lsp.Hover | None:
    """Compute hover at the first occurrence of ``needle`` in ``source``."""
    pos = position_of(source, needle, occurrence=occurrence, offset=offset)
    return features.compute_hover(source, source_path, pos)


def hover_text_at(
    source: str,
    source_path: str,
    needle: str,
    *,
    occurrence: int = 1,
    offset: int = 0,
) -> str:
    return hover_value(hover_at(source, source_path, needle, occurrence=occurrence, offset=offset))


def diagnostics_for(source: str, source_path: str) -> list:
    return features.compute_diagnostics(source, source_path)


def position_at_line(line: int, character: int = 0) -> lsp.Position:
    """Explicit coordinates for edge tests that are not tied to source needles."""
    return lsp.Position(line=line, character=character)


def position_of(source: str, needle: str, *, occurrence: int = 1, offset: int = 0) -> lsp.Position:
    start = 0
    index = 0
    for _ in range(occurrence):
        index = source.index(needle, start)
        start = index + len(needle)
    index += offset
    before = source[:index]
    return lsp.Position(
        line=before.count("\n"),
        character=len(before.rsplit("\n", 1)[-1]),
    )


def line_range(start_line: int, end_line: int) -> lsp.Range:
    return lsp.Range(
        start=lsp.Position(line=start_line, character=0),
        end=lsp.Position(line=end_line, character=0),
    )


def definition_at(
    source: str,
    uri: str,
    source_path: str,
    needle: str,
    *,
    occurrence: int = 1,
    offset: int = 0,
) -> lsp.Location | None:
    pos = position_of(source, needle, occurrence=occurrence, offset=offset)
    return features.compute_definition(source, uri, source_path, pos)


def references_at(
    source: str,
    uri: str,
    source_path: str,
    needle: str,
    include_declaration: bool,
    *,
    occurrence: int = 1,
    offset: int = 0,
) -> list[lsp.Location]:
    pos = position_of(source, needle, occurrence=occurrence, offset=offset)
    return features.compute_references(source, uri, source_path, pos, include_declaration)


def prepare_rename_at(
    source: str,
    source_path: str,
    needle: str,
    *,
    occurrence: int = 1,
    offset: int = 0,
) -> lsp.Range | None:
    pos = position_of(source, needle, occurrence=occurrence, offset=offset)
    return features.compute_prepare_rename(source, source_path, pos)


def rename_at(
    source: str,
    uri: str,
    source_path: str,
    needle: str,
    new_name: str,
    *,
    occurrence: int = 1,
    offset: int = 0,
) -> lsp.WorkspaceEdit | None:
    pos = position_of(source, needle, occurrence=occurrence, offset=offset)
    return features.compute_rename(source, uri, source_path, pos, new_name)


def signature_help_at(
    source: str,
    source_path: str,
    needle: str,
    *,
    occurrence: int = 1,
    offset: int = 0,
) -> lsp.SignatureHelp | None:
    pos = position_of(source, needle, occurrence=occurrence, offset=offset)
    return features.compute_signature_help(source, source_path, pos)


def inlay_hints_for_lines(
    source: str,
    source_path: str,
    start_line: int,
    end_line: int,
) -> list[lsp.InlayHint]:
    return features.compute_inlay_hints(source, source_path, line_range(start_line, end_line))


def document_symbols_for(source: str, source_path: str) -> list:
    return features.compute_document_symbols(source, source_path)
