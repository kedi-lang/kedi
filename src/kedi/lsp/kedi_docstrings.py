"""Embedded-Kedi support for Python docstrings.

The Python API treats a function docstring whose first cleaned line is
``kedi`` as an inline Kedi program.  This module mirrors that extraction
for editor features and remaps Kedi LSP ranges back into the Python file.
"""

from __future__ import annotations

import ast
import inspect
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Iterable

from lsprotocol import types as lsp

from . import features
from .coordinates import byte_column_to_utf16, utf16_length, utf16_to_index

_TRIPLE_QUOTED_RE = re.compile(r"^[rRuUbB]*(?P<quote>'''|\"\"\")")


@dataclass(frozen=True)
class _LineMap:
    kedi_line: int
    source_line: int
    source_start_character: int
    text: str


@dataclass(frozen=True)
class _KediDocstring:
    source_range: lsp.Range
    kedi_source: str
    line_maps: tuple[_LineMap, ...]
    external_symbols: dict[str, "_ExternalSymbol"]


@dataclass(frozen=True)
class _ExternalSymbol:
    range: lsp.Range
    type_annotation: str


def compute_kedi_docstring_ranges(source: str) -> list[dict[str, Any]]:
    return [_docstring_to_payload(doc) for doc in _iter_kedi_docstrings(source)]


def compute_kedi_docstring_diagnostics(
    source: str,
    source_path: str | None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for doc in _iter_kedi_docstrings(source):
        diagnostics = features.compute_diagnostics(doc.kedi_source, source_path)
        for diagnostic in diagnostics:
            mapped = _map_range_to_source(doc, diagnostic.range)
            if mapped is None:
                continue
            out.append(
                {
                    "range": _range_to_dict(mapped),
                    "message": diagnostic.message,
                    "severity": diagnostic.severity,
                    "source": "kedi",
                }
            )
    return out


def compute_kedi_docstring_hover(
    source: str,
    source_path: str | None,
    pos: lsp.Position,
) -> dict[str, Any] | None:
    found = _docstring_at_position(source, pos)
    if found is None:
        return None
    doc, kedi_pos = found
    hover = features.compute_hover(doc.kedi_source, source_path, kedi_pos)
    if hover is not None:
        payload: dict[str, Any] = {"contents": _hover_contents_to_markdown(hover.contents)}
        if hover.range is not None:
            mapped = _map_range_to_source(doc, hover.range)
            if mapped is not None:
                payload["range"] = _range_to_dict(mapped)
        return payload

    external = _external_symbol_at_position(doc, kedi_pos)
    if external is None:
        return None
    return {
        "contents": {
            "kind": "markdown",
            "value": (
                f"**Query parameter** `{external.name}: {external.type_annotation}` — "
                "available inside this Kedi docstring."
            ),
        },
        "range": _range_to_dict(external.source_range),
    }


def compute_kedi_docstring_definition(
    source: str,
    uri: str,
    source_path: str | None,
    pos: lsp.Position,
) -> list[dict[str, Any]]:
    found = _docstring_at_position(source, pos)
    if found is None:
        return []
    doc, kedi_pos = found
    location = features.compute_definition(doc.kedi_source, uri, source_path, kedi_pos)
    if location is None:
        external = _external_symbol_location(doc, uri, kedi_pos)
        return [external] if external is not None else []
    mapped = _map_location_to_source(doc, location, uri)
    return [mapped] if mapped is not None else [_location_to_dict(location)]


def compute_kedi_docstring_references(
    source: str,
    uri: str,
    source_path: str | None,
    pos: lsp.Position,
    *,
    include_declaration: bool,
) -> list[dict[str, Any]]:
    found = _docstring_at_position(source, pos)
    if found is None:
        return []
    doc, kedi_pos = found
    refs = features.compute_references(
        doc.kedi_source,
        uri,
        source_path,
        kedi_pos,
        include_declaration,
    )
    out: list[dict[str, Any]] = []
    for ref in refs:
        out.append(_map_location_to_source(doc, ref, uri) or _location_to_dict(ref))
    out.extend(_external_symbol_references(doc, uri, kedi_pos, include_declaration))
    return out


def compute_kedi_docstring_semantic_tokens(source: str) -> dict[str, list[int]]:
    raw: list[tuple[int, int, int, int, int]] = []
    for doc in _iter_kedi_docstrings(source):
        tokens = features.compute_semantic_tokens_full(doc.kedi_source, None)
        for line, col, length, token_type, token_mods in _decode_semantic_tokens(tokens.data):
            mapped = _map_range_to_source(
                doc,
                lsp.Range(
                    start=lsp.Position(line=line, character=col),
                    end=lsp.Position(line=line, character=col + length),
                ),
            )
            if mapped is None or mapped.start.line != mapped.end.line:
                continue
            raw.append(
                (
                    mapped.start.line,
                    mapped.start.character,
                    mapped.end.character - mapped.start.character,
                    token_type,
                    token_mods,
                )
            )
    raw.sort(key=lambda token: (token[0], token[1]))
    return {"data": _encode_semantic_tokens(raw)}


def _iter_kedi_docstrings(source: str) -> Iterable[_KediDocstring]:
    try:
        module = ast.parse(source)
    except SyntaxError:
        return []

    docs: list[_KediDocstring] = []
    for node in ast.walk(module):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not node.body:
            continue
        first = node.body[0]
        if not isinstance(first, ast.Expr) or not isinstance(first.value, ast.Constant):
            continue
        if not isinstance(first.value.value, str):
            continue
        doc = _extract_kedi_docstring(source, node, first.value)
        if doc is not None:
            docs.append(doc)
    return docs


def _extract_kedi_docstring(
    source: str,
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    node: ast.Constant,
) -> _KediDocstring | None:
    value = node.value
    if not isinstance(value, str):
        return None
    segment = ast.get_source_segment(source, node)
    if segment is None:
        return None
    match = _TRIPLE_QUOTED_RE.match(segment)
    if match is None:
        return None

    cleaned = inspect.cleandoc(value)
    cleaned_lines = cleaned.splitlines()
    if not cleaned_lines or cleaned_lines[0].strip() != "kedi":
        return None

    quote = match.group("quote")
    content = segment[match.end() : -len(quote)]
    content_lines = content.split("\n")
    header_index = _find_header_line(content_lines)
    if header_index is None:
        return None

    body_start = header_index + 1
    body_end = len(content_lines)
    while body_start < body_end and not content_lines[body_start].strip():
        body_start += 1
    while body_end > body_start and not content_lines[body_end - 1].strip():
        body_end -= 1
    body_lines = content_lines[body_start:body_end]
    if not body_lines:
        return None

    indent = _common_indent(body_lines)
    kedi_lines = [_dedent_line(line, indent) for line in body_lines]
    kedi_source = "\n".join(kedi_lines)
    if not kedi_source.strip():
        return None

    source_lines = source.splitlines()
    base_line = node.lineno - 1
    base_character = byte_column_to_utf16(source_lines[base_line], node.col_offset)
    base_character += utf16_length(segment[: match.end()])
    line_maps = []
    for kedi_line, raw_index in enumerate(range(body_start, body_end)):
        source_line = base_line + raw_index
        source_start_character = (base_character if raw_index == 0 else 0) + min(
            indent,
            len(content_lines[raw_index]) - len(content_lines[raw_index].lstrip(" ")),
        )
        line_maps.append(
            _LineMap(
                kedi_line=kedi_line,
                source_line=source_line,
                source_start_character=source_start_character,
                text=kedi_lines[kedi_line],
            )
        )

    source_start = lsp.Position(
        line=line_maps[0].source_line,
        character=line_maps[0].source_start_character,
    )
    last = line_maps[-1]
    source_end = lsp.Position(
        line=last.source_line,
        character=last.source_start_character + utf16_length(last.text),
    )
    return _KediDocstring(
        source_range=lsp.Range(start=source_start, end=source_end),
        kedi_source=kedi_source,
        line_maps=tuple(line_maps),
        external_symbols=_function_param_ranges(function, source),
    )


def _find_header_line(lines: list[str]) -> int | None:
    for index, line in enumerate(lines):
        if line.strip() == "kedi":
            return index
        if line.strip():
            return None
    return None


def _common_indent(lines: list[str]) -> int:
    indents = [len(line) - len(line.lstrip(" ")) for line in lines if line.strip()]
    return min(indents, default=0)


def _dedent_line(line: str, indent: int) -> str:
    return line[indent:] if len(line) >= indent else ""


def _docstring_at_position(
    source: str,
    pos: lsp.Position,
) -> tuple[_KediDocstring, lsp.Position] | None:
    for doc in _iter_kedi_docstrings(source):
        kedi_pos = _map_position_to_kedi(doc, pos)
        if kedi_pos is not None:
            return doc, kedi_pos
    return None


def _map_position_to_kedi(doc: _KediDocstring, pos: lsp.Position) -> lsp.Position | None:
    for line_map in doc.line_maps:
        if line_map.source_line != pos.line:
            continue
        start = line_map.source_start_character
        end = start + utf16_length(line_map.text)
        if pos.character < start or pos.character > end:
            return None
        return lsp.Position(line=line_map.kedi_line, character=pos.character - start)
    return None


def _map_range_to_source(doc: _KediDocstring, range_: lsp.Range) -> lsp.Range | None:
    start = _map_position_to_source(doc, range_.start)
    end = _map_position_to_source(doc, range_.end)
    if start is None or end is None:
        return None
    return lsp.Range(start=start, end=end)


def _map_position_to_source(doc: _KediDocstring, pos: lsp.Position) -> lsp.Position | None:
    for line_map in doc.line_maps:
        if line_map.kedi_line != pos.line:
            continue
        if pos.character < 0 or pos.character > utf16_length(line_map.text):
            return None
        return lsp.Position(
            line=line_map.source_line,
            character=line_map.source_start_character + pos.character,
        )
    return None


def _map_location_to_source(
    doc: _KediDocstring,
    location: lsp.Location,
    original_uri: str,
) -> dict[str, Any] | None:
    if location.uri != original_uri:
        return None
    mapped = _map_range_to_source(doc, location.range)
    if mapped is None:
        return None
    return {"uri": original_uri, "range": _range_to_dict(mapped)}


def _function_param_ranges(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    source: str,
) -> dict[str, _ExternalSymbol]:
    ranges: dict[str, _ExternalSymbol] = {}
    args = [
        *function.args.posonlyargs,
        *function.args.args,
        *function.args.kwonlyargs,
    ]
    if function.args.vararg is not None:
        args.append(function.args.vararg)
    if function.args.kwarg is not None:
        args.append(function.args.kwarg)
    source_lines = source.splitlines()
    for arg in args:
        line = source_lines[arg.lineno - 1] if arg.lineno - 1 < len(source_lines) else ""
        start = byte_column_to_utf16(line, arg.col_offset)
        ranges[arg.arg] = _ExternalSymbol(
            range=lsp.Range(
                start=lsp.Position(line=arg.lineno - 1, character=start),
                end=lsp.Position(
                    line=arg.lineno - 1,
                    character=start + utf16_length(arg.arg),
                ),
            ),
            type_annotation=ast.unparse(arg.annotation) if arg.annotation is not None else "str",
        )
    return ranges


def _external_symbol_location(
    doc: _KediDocstring,
    uri: str,
    pos: lsp.Position,
) -> dict[str, Any] | None:
    word = _word_at(doc.kedi_source, pos)
    if word is None:
        return None
    symbol = doc.external_symbols.get(word)
    if symbol is None:
        return None
    return {"uri": uri, "range": _range_to_dict(symbol.range)}


def _external_symbol_references(
    doc: _KediDocstring,
    uri: str,
    pos: lsp.Position,
    include_declaration: bool,
) -> list[dict[str, Any]]:
    word = _word_at(doc.kedi_source, pos)
    if word is None or word not in doc.external_symbols:
        return []
    refs: list[dict[str, Any]] = []
    if include_declaration:
        refs.append({"uri": uri, "range": _range_to_dict(doc.external_symbols[word].range)})
    for range_ in _word_ranges(doc.kedi_source, word):
        mapped = _map_range_to_source(doc, range_)
        if mapped is not None:
            refs.append({"uri": uri, "range": _range_to_dict(mapped)})
    return refs


def _word_at(source: str, pos: lsp.Position) -> str | None:
    word = _word_and_range_at(source, pos)
    return word[0] if word is not None else None


def _word_and_range_at(source: str, pos: lsp.Position) -> tuple[str, lsp.Range] | None:
    lines = source.splitlines()
    if pos.line < 0 or pos.line >= len(lines):
        return None
    line = lines[pos.line]
    index = min(utf16_to_index(line, pos.character), len(line) - 1)
    if index < 0:
        return None
    if not _is_identifier_char(line[index]):
        index -= 1
    if index < 0 or not _is_identifier_char(line[index]):
        return None
    start = index
    while start > 0 and _is_identifier_char(line[start - 1]):
        start -= 1
    end = index + 1
    while end < len(line) and _is_identifier_char(line[end]):
        end += 1
    return (
        line[start:end],
        lsp.Range(
            start=lsp.Position(line=pos.line, character=utf16_length(line[:start])),
            end=lsp.Position(line=pos.line, character=utf16_length(line[:end])),
        ),
    )


@dataclass(frozen=True)
class _ExternalSymbolAtPosition:
    name: str
    type_annotation: str
    source_range: lsp.Range


def _external_symbol_at_position(
    doc: _KediDocstring,
    pos: lsp.Position,
) -> _ExternalSymbolAtPosition | None:
    word = _word_and_range_at(doc.kedi_source, pos)
    if word is None:
        return None
    name, kedi_range = word
    symbol = doc.external_symbols.get(name)
    if symbol is None:
        return None
    mapped = _map_range_to_source(doc, kedi_range)
    if mapped is None:
        return None
    return _ExternalSymbolAtPosition(name, symbol.type_annotation, mapped)


def _word_ranges(source: str, word: str) -> list[lsp.Range]:
    ranges: list[lsp.Range] = []
    pattern = re.compile(rf"\b{re.escape(word)}\b")
    for line_no, line in enumerate(source.splitlines()):
        for match in pattern.finditer(line):
            ranges.append(
                lsp.Range(
                    start=lsp.Position(
                        line=line_no,
                        character=utf16_length(line[: match.start()]),
                    ),
                    end=lsp.Position(
                        line=line_no,
                        character=utf16_length(line[: match.end()]),
                    ),
                )
            )
    return ranges


def _is_identifier_char(char: str) -> bool:
    return bool(re.match(r"[A-Za-z0-9_]", char))


def _docstring_to_payload(doc: _KediDocstring) -> dict[str, Any]:
    return {
        "range": _range_to_dict(doc.source_range),
        "text": doc.kedi_source,
    }


def _hover_contents_to_markdown(contents: Any) -> dict[str, str]:
    if isinstance(contents, lsp.MarkupContent):
        return {"kind": contents.kind, "value": contents.value}
    return {"kind": "markdown", "value": str(contents)}


def _location_to_dict(location: lsp.Location) -> dict[str, Any]:
    return {"uri": location.uri, "range": _range_to_dict(location.range)}


def _range_to_dict(range_: lsp.Range) -> dict[str, dict[str, int]]:
    return {
        "start": _position_to_dict(range_.start),
        "end": _position_to_dict(range_.end),
    }


def _position_to_dict(pos: lsp.Position) -> dict[str, int]:
    return {"line": pos.line, "character": pos.character}


def _decode_semantic_tokens(
    data: Sequence[int],
) -> list[tuple[int, int, int, int, int]]:
    out: list[tuple[int, int, int, int, int]] = []
    line = 0
    col = 0
    for d_line, d_col, length, token_type, token_mods in zip(
        data[0::5],
        data[1::5],
        data[2::5],
        data[3::5],
        data[4::5],
        strict=False,
    ):
        line += d_line
        col = col + d_col if d_line == 0 else d_col
        out.append((line, col, length, token_type, token_mods))
    return out


def _encode_semantic_tokens(tokens: list[tuple[int, int, int, int, int]]) -> list[int]:
    data: list[int] = []
    prev_line = 0
    prev_col = 0
    for line, col, length, token_type, token_mods in tokens:
        data.extend(
            [
                line - prev_line,
                col - prev_col if line == prev_line else col,
                length,
                token_type,
                token_mods,
            ]
        )
        prev_line = line
        prev_col = col
    return data
