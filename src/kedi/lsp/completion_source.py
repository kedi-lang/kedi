"""Recover surrounding declarations while a completion line is unfinished."""

from __future__ import annotations

from kedi.lang.ast import Program
from kedi.lang.parser import KediParseError
from kedi.lsp.parsing import parse_program


def parse_completion_source(source: str, source_path: str | None, line: int | None) -> Program:
    """Keep editor recovery separate from the strict execution parser.

    A neutral statement retains the current indentation and physical line so
    enclosing scopes survive an unfinished capture or Python delimiter. Python
    tooling still maps the original expression, never this placeholder.
    """
    try:
        return parse_program(source, source_path=source_path)
    except KediParseError:
        if line is None or not 0 <= line < len(source.split("\n")):
            raise

    lines = source.split("\n")
    current = lines[line]
    indent = current[: len(current) - len(current.lstrip(" \t"))]
    width = len(current.encode())
    placeholder = indent + "> show: "
    # Replacements keep the line's byte length, so spans after it match the source.
    if width > len(placeholder):
        candidates = [placeholder + "_" * (width - len(placeholder))]
    else:
        # Too short for a statement: a comment of the same length keeps the scopes when
        # the block has other lines; otherwise accept a statement that shifts later spans.
        candidates = [indent + "#" + "_" * (width - len(indent) - 1)] if width > len(indent) else []
        candidates.append(placeholder + "_")
    for candidate in candidates:
        lines[line] = candidate
        try:
            return parse_program("\n".join(lines) + "\n", source_path=source_path)
        except KediParseError:
            continue
    # An unfinished later statement must not hide an otherwise valid prefix.
    return parse_program("\n".join(lines[: line + 1]) + "\n", source_path=source_path)
