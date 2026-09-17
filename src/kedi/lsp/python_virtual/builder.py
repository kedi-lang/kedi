from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from kedi.lang.ast import Span

Position = dict[str, int]


Range = dict[str, Position]


@dataclass
class PythonRegion:
    kind: str
    range: Range
    text: str
    used: bool = False


@dataclass
class ScopeName:
    typ: str
    source_range: Optional[Range]


class VirtualBuilder:
    def __init__(self, source: str, source_path: Optional[str]) -> None:
        self.source = source
        self.source_lines = source.splitlines()
        self.lines: list[str] = []
        self.mappings: list[dict[str, Any]] = []
        self.regions: list[dict[str, Any]] = []
        self.symbols: list[dict[str, Any]] = []

    @property
    def line(self) -> int:
        return len(self.lines)

    def add(self, text: str = "") -> int:
        line = self.line
        self.lines.append(text)
        return line

    def add_symbol(
        self,
        *,
        kind: str,
        name: str,
        source_range: Optional[Range],
        virtual_range: Range,
    ) -> None:
        if source_range is None:
            return
        self.symbols.append(
            {
                "kind": kind,
                "name": name,
                "sourceRange": source_range,
                "virtualRange": virtual_range,
            }
        )

    def add_synthetic_inline_region(
        self,
        *,
        source_range: Range,
        virtual_range: Range,
        text: str,
    ) -> None:
        self.mappings.append(
            {
                "sourceRange": source_range,
                "virtualRange": virtual_range,
                "kind": "inline",
            }
        )
        self.regions.append(
            {
                "kind": "inline",
                "range": source_range,
                "sourceRange": source_range,
                "virtualRange": virtual_range,
                "text": text,
            }
        )

    def add_mapped_region(self, region: PythonRegion, prefix: str = "") -> Range:
        virtual_start: Optional[Position] = None
        virtual_end: Optional[Position] = None
        emitted_substantive_line = False
        src_start = region.range["start"]
        src_end = region.range["end"]
        start_line = src_start["line"]
        end_line = src_end["line"]
        dedent = self._common_indent(start_line, end_line) if region.kind == "fenced" else 0

        for source_line in range(start_line, end_line + 1):
            if source_line >= len(self.source_lines):
                continue
            line_text = self.source_lines[source_line]
            start_char = src_start["character"] if source_line == start_line else 0
            end_char = src_end["character"] if source_line == end_line else len(line_text)
            if source_line == end_line and end_char == 0:
                continue
            if dedent:
                start_char = min(end_char, start_char + min(dedent, _indent_width(line_text)))
            if end_char < start_char:
                continue

            body = line_text[start_char:end_char]
            if _has_python_substance(body):
                emitted_substantive_line = True
            virtual_line = self.line
            virtual_start_char = len(prefix)
            self.add(prefix + body)
            source_range = {
                "start": {"line": source_line, "character": start_char},
                "end": {"line": source_line, "character": end_char},
            }
            virtual_range = {
                "start": {"line": virtual_line, "character": virtual_start_char},
                "end": {
                    "line": virtual_line,
                    "character": virtual_start_char + max(0, end_char - start_char),
                },
            }
            self.mappings.append(
                {
                    "sourceRange": source_range,
                    "virtualRange": virtual_range,
                    "kind": region.kind,
                }
            )
            if virtual_start is None:
                virtual_start = dict(virtual_range["start"])
            virtual_end = dict(virtual_range["end"])

        if virtual_start is None:
            virtual_start = {"line": self.line, "character": len(prefix)}
            virtual_end = dict(virtual_start)
            self.add(prefix + "...")
            emitted_substantive_line = True
        elif not emitted_substantive_line:
            virtual_line = self.line
            self.add(prefix + "...")
            virtual_end = {
                "line": virtual_line,
                "character": len(prefix) + len("..."),
            }

        virtual_range = {"start": virtual_start, "end": virtual_end or virtual_start}
        self.regions.append(
            {
                "kind": region.kind,
                "range": region.range,
                "sourceRange": region.range,
                "virtualRange": virtual_range,
                "text": region.text,
            }
        )
        region.used = True
        return virtual_range

    def _common_indent(self, start_line: int, end_line: int) -> int:
        indents: list[int] = []
        for source_line in range(start_line, end_line + 1):
            if source_line >= len(self.source_lines):
                continue
            line_text = self.source_lines[source_line]
            if line_text.strip():
                indents.append(_indent_width(line_text))
        return min(indents) if indents else 0

    def finish(self) -> str:
        return "\n".join(self.lines) + "\n"


class RegionCursor:
    def __init__(self, ranges: list[dict[str, Any]]) -> None:
        self.regions = [
            PythonRegion(kind=r["kind"], range=r["range"], text=r.get("text", "")) for r in ranges
        ]

    def next(self, kind: str, after: Optional[Span] = None) -> Optional[PythonRegion]:
        for region in self.regions:
            if region.used or region.kind != kind:
                continue
            if after is not None:
                if not _range_within_span(region.range, after):
                    continue
            region.used = True
            return region
        return None


def _line_range(line: int, start: int, text: str) -> Range:
    return {
        "start": {"line": line, "character": start},
        "end": {"line": line, "character": start + len(text)},
    }


def _span_range(span: Span) -> Range:
    return {
        "start": {"line": span.start_line - 1, "character": span.start_col - 1},
        "end": {"line": span.end_line - 1, "character": max(span.end_col - 1, 0)},
    }


def _range_within_span(range_: Range, span: Span) -> bool:
    start = range_["start"]
    end = range_["end"]
    start_line = start["line"] + 1
    end_line = end["line"] + 1
    if start_line < span.start_line or end_line > span.end_line:
        return False
    if start_line == span.start_line and start["character"] + 1 < span.start_col:
        return False
    if end_line == span.end_line and end["character"] + 1 > span.end_col:
        return False
    return True


def _indent_width(line: str) -> int:
    width = 0
    for char in line:
        if char == " ":
            width += 1
        elif char == "\t":
            width += 4
        else:
            break
    return width


def _has_python_substance(text: str) -> bool:
    stripped = text.strip()
    return bool(stripped) and not stripped.startswith("#")
