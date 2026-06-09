#!/usr/bin/env python3
"""Add ``>>`` to bare template rows in Kedi snippets embedded in test strings."""

from __future__ import annotations

import ast
import re
import sys
import tokenize
from io import BytesIO
from pathlib import Path

ASSIGN_RE = re.compile(r"^\[.+\]\s*=")
SKIP_PREFIXES = ("@", "=", "`", ">>", ">", "~", "#", "###", "```")
OPT_DIRECTIVE_RE = re.compile(
    r"> (optimize|auto|test|eval|case|data|test_data|metric|import|export):"
)


def _maybe_kedi_snippet(value: str) -> bool:
    if "\n" not in value:
        return False
    if any(
        marker in value for marker in ("@", ">>", "> optimize:", "> auto:", "> test:", "> eval:")
    ):
        return True
    if re.search(r"(^|\n)\s*=", value):
        return True
    if re.search(r"(^|\n)\s*>>", value):
        return True
    return bool(re.search(r"\[[A-Za-z_][A-Za-z0-9_]*", value))


def _looks_like_template(stripped: str) -> bool:
    return bool(re.search(r"[\[<]", stripped) or re.match(r"[A-Za-z'\"]", stripped))


def _should_skip(stripped: str) -> bool:
    return not stripped or stripped.startswith(SKIP_PREFIXES) or bool(ASSIGN_RE.match(stripped))


def _opens_python_block(stripped: str) -> bool:
    return stripped == "```" or "= ```" in stripped


def _closes_python_block(stripped: str) -> bool:
    return stripped == "```"


def fix_kedi_source(src: str) -> str:
    lines = src.split("\n")
    out: list[str] = []
    optimize_at: int | None = None
    in_block = False
    block_indent = -1
    in_fenced_python = False

    for line in lines:
        stripped = line.lstrip()
        indent = len(line) - len(stripped) if stripped else len(line.rstrip("\n"))

        if in_fenced_python:
            if _closes_python_block(stripped):
                in_fenced_python = False
            out.append(line)
            continue

        if _opens_python_block(stripped):
            in_fenced_python = True
            in_block = False
            out.append(line)
            continue

        if OPT_DIRECTIVE_RE.match(stripped):
            optimize_at = indent
            in_block = False
            out.append(line)
            continue

        if (
            optimize_at is not None
            and stripped
            and indent <= optimize_at
            and not stripped.startswith(">")
        ):
            optimize_at = None

        in_opt = optimize_at is not None and indent > optimize_at

        if stripped.startswith(">>"):
            in_block = True
            block_indent = indent
            out.append(line)
            continue

        if in_block and not stripped:
            in_block = False
            out.append(line)
            continue

        if in_block and stripped and indent == block_indent:
            out.append(line)
            continue

        if in_block and stripped:
            in_block = False

        if not in_opt and not _should_skip(stripped) and _looks_like_template(stripped):
            line = f"{' ' * indent}>> {stripped}"
            in_block = True
            block_indent = indent

        out.append(line)

    return "\n".join(out)


def _string_prefix(raw: str) -> str:
    idx = 0
    while idx < len(raw) and raw[idx] in "rRbBuU":
        idx += 1
    return raw[:idx]


def _string_literals(source: str) -> list[tuple[int, int, str, str]]:
    """Return (start_line, end_line, decoded_value, raw_token) for plain strings."""
    literals: list[tuple[int, int, str, str]] = []
    tokens = tokenize.tokenize(BytesIO(source.encode()).readline)
    for tok in tokens:
        if tok.type != tokenize.STRING:
            continue
        prefix = _string_prefix(tok.string)
        if "f" in prefix.lower():
            continue
        try:
            value = ast.literal_eval(tok.string)
        except (ValueError, SyntaxError):
            continue
        if isinstance(value, str) and _maybe_kedi_snippet(value):
            literals.append((tok.start[0], tok.end[0], value, tok.string))
    return literals


def fix_file(path: Path) -> bool:
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines(keepends=True)
    changed = False

    for start_line, end_line, value, raw in reversed(_string_literals(source)):
        fixed = fix_kedi_source(value)
        if fixed == value:
            continue
        prefix = _string_prefix(raw)
        new_raw = repr(fixed)
        if prefix and prefix.lower() != "r":
            new_raw = prefix + new_raw
        start_idx = start_line - 1
        end_idx = end_line
        chunk = "".join(lines[start_idx:end_idx])
        new_chunk = chunk.replace(raw, new_raw, 1)
        if new_chunk == chunk:
            continue
        lines[start_idx:end_idx] = [new_chunk]
        changed = True

    if not changed:
        return False
    path.write_text("".join(lines), encoding="utf-8")
    return True


def main(argv: list[str]) -> int:
    root = Path(argv[1] if len(argv) > 1 else "tests")
    changed = [p for p in sorted(root.rglob("test_*.py")) if fix_file(p)]
    for path in changed:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
