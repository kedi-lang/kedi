"""Coordinate conversion helpers shared by Kedi LSP surfaces.

Tree-sitter reports UTF-8 byte columns while LSP clients use UTF-16 code-unit
columns. Keeping the conversion here prevents each feature and the Python
docstring bridge from implementing subtly different position math.
"""

from __future__ import annotations

from bisect import bisect_right
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import lru_cache, wraps
from pathlib import Path
from typing import Callable, Iterator, Optional, ParamSpec, TypeVar

from lsprotocol import types as lsp

from ..lang.ast import Span

P = ParamSpec("P")
R = TypeVar("R")


def utf16_length(text: str) -> int:
    """Return the number of LSP UTF-16 code units in ``text``."""
    return len(text.encode("utf-16-le")) // 2


def utf16_to_index(text: str, character: int) -> int:
    """Convert an LSP UTF-16 offset to the nearest Python string index."""
    if character <= 0:
        return 0
    units = 0
    for index, char in enumerate(text):
        next_units = units + utf16_length(char)
        if next_units > character:
            return index
        units = next_units
    return len(text)


def byte_column_to_utf16(line: str, byte_column: int) -> int:
    """Convert a tree-sitter byte column within ``line`` to an LSP column."""
    raw = line.encode("utf-8")
    prefix = raw[: max(0, min(byte_column, len(raw)))].decode("utf-8", "replace")
    return utf16_length(prefix)


@dataclass(frozen=True)
class CoordinateSource:
    source: str
    source_path: Optional[str]

    @property
    def lines(self) -> tuple[str, ...]:
        return tuple(self.source.split("\n"))

    def _line_starts(self) -> tuple[int, ...]:
        raw = self.source.encode("utf-8")
        starts = [0]
        starts.extend(index + 1 for index, value in enumerate(raw) if value == 10)
        return tuple(starts)

    def byte_offset_to_position(self, offset: int) -> lsp.Position:
        raw = self.source.encode("utf-8")
        offset = max(0, min(offset, len(raw)))
        starts = self._line_starts()
        line = bisect_right(starts, offset) - 1
        line_start = starts[line]
        return lsp.Position(
            line=line,
            character=byte_column_to_utf16(self.lines[line], offset - line_start),
        )

    def tree_sitter_point_to_position(self, line: int, byte_column: int) -> lsp.Position:
        lines = self.lines
        if line < 0:
            return lsp.Position(line=0, character=0)
        if line >= len(lines):
            line = len(lines) - 1
        return lsp.Position(
            line=line,
            character=byte_column_to_utf16(lines[line], byte_column),
        )


_CURRENT_SOURCE: ContextVar[CoordinateSource | None] = ContextVar(
    "kedi_lsp_coordinate_source", default=None
)


def current_source() -> CoordinateSource | None:
    """Return the source currently bound to a public LSP feature call."""
    return _CURRENT_SOURCE.get()


@contextmanager
def source_coordinates(source: str, source_path: Optional[str]) -> Iterator[None]:
    token = _CURRENT_SOURCE.set(CoordinateSource(source, source_path))
    try:
        yield
    finally:
        _CURRENT_SOURCE.reset(token)


def with_source_coordinates(
    source_path_index: int | None = 1,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorate a public feature so span helpers see the current source."""

    def decorate(function: Callable[P, R]) -> Callable[P, R]:
        @wraps(function)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
            source = args[0] if args else kwargs.get("source", "")
            source_path = kwargs.get("source_path")
            if not isinstance(source_path, str):
                source_path = None
            if source_path is None and source_path_index is not None:
                if len(args) > source_path_index:
                    candidate = args[source_path_index]
                    source_path = candidate if isinstance(candidate, str) else None
            with source_coordinates(str(source), source_path):
                return function(*args, **kwargs)

        return wrapped

    return decorate


@lru_cache(maxsize=32)
def _read_source(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None


def _span_source(span: Span) -> CoordinateSource | None:
    current = _CURRENT_SOURCE.get()
    if current is None or not span.source_path:
        return current
    if current.source_path is not None:
        try:
            if Path(current.source_path).resolve() == Path(span.source_path).resolve():
                return current
        except OSError:
            if current.source_path == span.source_path:
                return current
    external = _read_source(span.source_path)
    return CoordinateSource(external, span.source_path) if external is not None else None


def span_to_range(span: Span) -> lsp.Range:
    source = _span_source(span)
    if source is not None and span.end_byte >= span.start_byte:
        return lsp.Range(
            start=source.byte_offset_to_position(span.start_byte),
            end=source.byte_offset_to_position(span.end_byte),
        )
    return lsp.Range(
        start=lsp.Position(line=span.start_line - 1, character=span.start_col - 1),
        end=lsp.Position(line=span.end_line - 1, character=max(span.end_col - 1, 0)),
    )


def position_inside(pos: lsp.Position, span: Span) -> bool:
    range_ = span_to_range(span)
    if pos.line < range_.start.line or pos.line > range_.end.line:
        return False
    if pos.line == range_.start.line and pos.character < range_.start.character:
        return False
    if pos.line == range_.end.line and pos.character >= range_.end.character:
        return False
    return True


def tree_sitter_point_to_position(source: str, line: int, byte_column: int) -> lsp.Position:
    return CoordinateSource(source, None).tree_sitter_point_to_position(line, byte_column)
