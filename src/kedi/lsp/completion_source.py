"""Recover surrounding declarations while a completion line is unfinished."""

from __future__ import annotations

from kedi.lang.ast import Program
from kedi.lang.parser import KediParseError, parse_program


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
    placeholder = indent + "> show: "
    # Match bytes as well as line numbers for spans after the recovered line.
    lines[line] = placeholder + "_" * max(1, len(current.encode()) - len(placeholder.encode()))
    try:
        return parse_program("\n".join(lines) + "\n", source_path=source_path)
    except KediParseError:
        # An unfinished later statement must not hide an otherwise valid prefix.
        return parse_program("\n".join(lines[: line + 1]) + "\n", source_path=source_path)
