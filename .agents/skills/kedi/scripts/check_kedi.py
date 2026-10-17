#!/usr/bin/env python3
"""Parse Kedi files without compiling/executing them; use the target venv."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import sys


def check(path: Path, *, tree_only: bool = False) -> dict:
    row = {"path": str(path), "ok": False, "mode": "tree" if tree_only else "kedi"}
    try:
        source = path.read_text(encoding="utf-8")
        if tree_only:
            import tree_sitter_kedi
            from tree_sitter import Language, Parser

            tree = Parser(Language(tree_sitter_kedi.language())).parse(source.encode("utf-8"))
            errors = []
            stack = [tree.root_node]
            while stack:
                node = stack.pop()
                if node.is_error or node.is_missing:
                    errors.append(
                        {
                            "line": node.start_point.row + 1,
                            "column": node.start_point.column + 1,
                            "type": node.type,
                            "missing": node.is_missing,
                        }
                    )
                stack.extend(reversed(node.children))
            row["errors"] = errors
            row["ok"] = not tree.root_node.has_error and not errors
        else:
            from kedi.lang import parse_program

            # A real filename is required for package.kedi placement validation.
            parse_program(source, source_path=str(path))
            row["ok"] = True
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        if hasattr(exc, "diagnostics"):
            row["diagnostics"] = [
                {
                    "message": d.message,
                    "line": d.span.start_line,
                    "column": d.span.start_col,
                    "code": d.code,
                }
                for d in exc.diagnostics
            ]
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument(
        "--tree-only",
        action="store_true",
        help="Check CST grammar only; skip Kedi structural validation",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable results")
    args = parser.parse_args()
    files: set[Path] = set()
    errors = []
    for path in args.paths:
        if path.is_dir():
            found = list(path.rglob("*.kedi"))
            if not found:
                errors.append({"path": str(path), "ok": False, "error": "No .kedi files"})
            files.update(p.resolve() for p in found if p.is_file())
        elif path.is_file():
            files.add(path.resolve())
        else:
            errors.append({"path": str(path), "ok": False, "error": "Path does not exist"})
    results = errors + [check(p, tree_only=args.tree_only) for p in sorted(files)]
    versions = {}
    for name in ("kedi", "tree-sitter", "tree-sitter-kedi"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    if args.json:
        print(json.dumps({"versions": versions, "results": results}, indent=2))
    else:
        print("Versions: " + ", ".join(f"{k}={v}" for k, v in versions.items()))
        for row in results:
            print(f"{'PASS' if row['ok'] else 'FAIL'} {row['path']}")
            if not row["ok"]:
                print(row.get("error", row.get("errors")), file=sys.stderr)
        print(
            f"{sum(r['ok'] for r in results)}/{len(results)} files passed ({'CST only' if args.tree_only else 'Kedi AST'})."
        )
    return 0 if results and all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
