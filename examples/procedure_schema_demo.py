#!/usr/bin/env python3
"""Print the input JSON schema for a Kedi procedure."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from kedi.lang import compile_program, parse_program, procedure_json_schema

DEFAULT_SOURCE = Path(__file__).with_name("procedure_schema_demo.kedi")
DEFAULT_PROCEDURE = "search_docs"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print the JSON schema for a procedure in a Kedi source file."
    )
    parser.add_argument(
        "--source",
        default=str(DEFAULT_SOURCE),
        help="Path to the .kedi source file.",
    )
    parser.add_argument(
        "--procedure",
        default=DEFAULT_PROCEDURE,
        help="Procedure name to convert to JSON Schema.",
    )
    args = parser.parse_args()

    source_path = Path(args.source).resolve()
    if not source_path.exists():
        print(f"Source file not found: {source_path}", file=sys.stderr)
        sys.exit(1)

    source = source_path.read_text(encoding="utf-8")
    program = parse_program(source, source_path=str(source_path))
    runtime = compile_program(program)

    procedure = next(
        (proc for proc in program.procedures if proc.name == args.procedure),
        None,
    )
    if procedure is None:
        available = ", ".join(proc.name for proc in program.procedures) or "<none>"
        print(
            f"Procedure {args.procedure!r} not found in {source_path.name}. Available: {available}",
            file=sys.stderr,
        )
        sys.exit(1)
    assert procedure is not None

    schema = procedure_json_schema(runtime, procedure)
    print(json.dumps(schema, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
