"""Project Python completion edits back into executable Kedi source."""

from __future__ import annotations

import ast
from copy import deepcopy
from typing import Any

from kedi.lang.parser import KediParseError
from kedi.lsp.coordinates import utf16_to_index
from kedi.lsp.features.python_regions import compute_python_ranges
from kedi.lsp.parsing import parse_program

from .builder import Position, Range


def _key(position: Position) -> tuple[int, int]:
    return position["line"], position["character"]


def _mapped_range(virtual: dict[str, Any], range_: Range) -> Range | None:
    """Only copied Python is writable; synthetic symbols are navigation-only."""
    mappings = virtual.get("mappings", [])
    for first in mappings:
        start = range_["start"]
        a = first["virtualRange"]
        if not (_key(a["start"]) <= _key(start) <= _key(a["end"])):
            continue
        source = first["sourceRange"]
        delta = source["start"]["character"] - a["start"]["character"]
        end = range_["end"]
        current = first
        for line in range(start["line"] + 1, end["line"] + 1):
            following = next(
                (item for item in mappings if item["virtualRange"]["start"]["line"] == line),
                None,
            )
            if (
                following is None
                or following["sourceRange"]["start"]["line"]
                != current["sourceRange"]["start"]["line"] + 1
                or following["sourceRange"]["start"]["character"]
                - following["virtualRange"]["start"]["character"]
                != delta
            ):
                break
            current = following
        else:
            b = current["virtualRange"]
            if _key(b["start"]) <= _key(end) <= _key(b["end"]):
                return {
                    "start": {
                        "line": source["start"]["line"],
                        "character": start["character"] + delta,
                    },
                    "end": {
                        "line": current["sourceRange"]["start"]["line"],
                        "character": end["character"] + delta,
                    },
                }
    return None


def _map_edit(virtual: dict[str, Any], edit: dict[str, Any]) -> dict[str, Any] | None:
    keys = ("insert", "replace") if "insert" in edit else ("range",)
    result = dict(edit)
    for key in keys:
        mapped = _mapped_range(virtual, edit[key])
        if mapped is None:
            return None
        result[key] = mapped
    if "\n" in edit["newText"]:
        delta = result[keys[0]]["start"]["character"] - edit[keys[0]]["start"]["character"]
        lines = edit["newText"].splitlines(keepends=True)
        for index in range(1, len(lines)):
            line = lines[index]
            if delta < 0 and line.startswith(" " * -delta):
                lines[index] = line[-delta:]
            elif delta > 0 and line.strip():
                lines[index] = " " * delta + line
        result["newText"] = "".join(lines)
    return result


def _import_addition(text: str, edit: dict[str, Any]) -> str | None:
    """Recognize an import insertion or additions to a synthetic import line."""
    range_ = edit["range"]
    start, end = range_["start"], range_["end"]
    lines = text.splitlines()
    if start["line"] != end["line"] or start["line"] >= len(lines):
        return None
    line = lines[start["line"]]
    left = utf16_to_index(line, start["character"])
    right = utf16_to_index(line, end["character"])
    original = line.strip()
    replacement = (line[:left] + edit["newText"] + line[right:]).strip()
    if left == right == 0 and edit["newText"].endswith("\n"):
        original = ""
        replacement = edit["newText"].strip()
    try:
        before = ast.parse(original).body
        after = ast.parse(replacement).body
    except SyntaxError:
        return None
    if not after or not all(
        isinstance(node, (ast.Import, ast.ImportFrom)) for node in before + after
    ):
        return None
    prior = {
        (getattr(node, "module", None), getattr(node, "level", 0), alias.name, alias.asname)
        for node in before
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    result = []
    for node in after:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            continue
        assert isinstance(node, (ast.Import, ast.ImportFrom))
        node.names = [
            alias
            for alias in node.names
            if (getattr(node, "module", None), getattr(node, "level", 0), alias.name, alias.asname)
            not in prior
        ]
        if node.names:
            result.append(ast.unparse(node))
    return "\n".join(result) or None


def _imports_edit(source: str, imports: list[str]) -> dict[str, Any]:
    newline = "\r\n" if "\r\n" in source else "\n"
    text = newline.join(dict.fromkeys(imports)) + newline
    try:
        program = parse_program(source)
        prelude = program.prelude
        location = program.prelude_loc
    except KediParseError:
        prelude = None
        location = None
    first = next(
        (
            r
            for r in compute_python_ranges(source)
            if r["kind"] == "fenced"
            and location is not None
            and r["range"]["start"]["line"] == location.start_line
        ),
        None,
    )
    if prelude is not None and first is not None:
        # Read the valid prefix while a user is still typing an incomplete expression.
        prefix = prelude
        while True:
            try:
                nodes = ast.parse(prefix).body
                break
            except SyntaxError as exc:
                prefix = "\n".join(prefix.splitlines()[: max(0, (exc.lineno or 1) - 1)])
        header = 0
        for index, node in enumerate(nodes):
            if isinstance(node, (ast.Import, ast.ImportFrom)) or (
                index == 0
                and isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                header = node.end_lineno or node.lineno
            else:
                break
        start = first["range"]["start"]
        position = {"line": start["line"] + header, "character": 0}
    else:
        position = {"line": 0, "character": 0}
        text = "```" + newline + text + "```" + newline + newline
    return {"range": {"start": position, "end": dict(position)}, "newText": text}


def map_python_completions(
    source: str,
    virtual: dict[str, Any],
    items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return LSP completion items whose entire edit set can be applied safely."""
    result = []
    for original in items:
        item = deepcopy(original)
        edit = item.get("textEdit")
        if edit is not None:
            mapped = _map_edit(virtual, edit)
            if mapped is None:
                continue
            item["textEdit"] = mapped
        additional = []
        imports = []
        for extra in item.get("additionalTextEdits", []):
            mapped = _map_edit(virtual, extra)
            if mapped is not None:
                additional.append(mapped)
            else:
                addition = _import_addition(virtual.get("text", ""), extra)
                if addition is None:
                    break
                imports.append(addition)
        else:
            if imports:
                import_edit = _imports_edit(source, imports)
                primary = item.get("textEdit")
                primary_range = primary.get("range", primary.get("insert")) if primary else None
                if (
                    primary is not None
                    and primary_range
                    and primary_range["start"] == import_edit["range"]["start"]
                ):
                    # LSP forbids additional edits overlapping the primary edit.
                    primary["newText"] = import_edit["newText"] + primary["newText"]
                else:
                    additional.append(import_edit)
            item["additionalTextEdits"] = additional
            # Python commands may target the shadow document. Only edits cross this boundary.
            item.pop("command", None)
            result.append(item)
    return result
