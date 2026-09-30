"""Scope-aware virtual Python documents for embedded Kedi Python.

The editor extensions use this module to ask Python tooling about code
that lives inside Kedi files.  The generated document is ordinary Python
with Kedi procedures, parameters, and local variables represented as
static stubs.  Source maps keep editor positions tied to the original
Kedi regions.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Optional
from urllib.parse import unquote, urlparse

from kedi.lang.ast import ProgramWithValidation
from kedi.lang.parser import KediParseError
from kedi.lsp.completion_source import parse_completion_source
from kedi.lsp.features import compute_header_python_ranges, compute_python_ranges
from kedi.lsp.module_surface import imported_surface_for_program
from kedi.lsp.python_virtual.builder import PythonRegion, RegionCursor, ScopeName, VirtualBuilder
from kedi.lsp.python_virtual.completion import map_python_completions
from kedi.lsp.python_virtual.scope import (
    _all_procedures,
    _collect_toplevel_scope,
    _find_focused_procedure,
    _index_header_ranges,
    _range_key,
    _should_emit_full_module_python,
    _split_leading_toplevel_python,
    _top_level_procedure_for,
)
from kedi.lsp.python_virtual.statements import (
    _emit_procedure_body,
    _emit_return,
    _emit_statements,
    _emit_unconsumed_inline_regions,
    _emit_validation,
)
from kedi.lsp.python_virtual.stubs import (
    _emit_imported_surface_stubs,
    _emit_proc_stub,
    _emit_python_module_stub,
    _emit_runtime_args_stub,
    _emit_toplevel_scope_stubs,
    _emit_type_stub,
    _typing_import_lines,
)


def compute_python_virtual_document(
    source: str,
    source_path: Optional[str] = None,
    *,
    focus_line: Optional[int] = None,
) -> dict[str, Any]:
    """Return a virtual Python document plus Kedi/Python source maps."""
    try:
        program = parse_completion_source(source, source_path, focus_line)
    except KediParseError:
        ranges = compute_python_ranges(source)
        return _fallback_virtual_document(source, ranges)

    header_regions = compute_header_python_ranges(source)
    header_range_keys = {_range_key(item["range"]) for item in header_regions}
    cursor = RegionCursor(
        [
            item
            for item in compute_python_ranges(source)
            if _range_key(item["range"]) not in header_range_keys
        ]
    )
    header_ranges = _index_header_ranges(header_regions)
    builder = VirtualBuilder(source, source_path)
    focused_proc = _find_focused_procedure(program.procedures, focus_line)
    focused_root = _top_level_procedure_for(program.procedures, focused_proc)
    focus_key = (
        f"proc:{focused_proc.name}:{focused_proc.loc.start_line}"
        if focused_proc is not None and focused_proc.loc is not None
        else None
    )
    imported_surface = imported_surface_for_program(program)
    leading_toplevel_python, remaining_toplevel = _split_leading_toplevel_python(
        program.toplevel_body
    )
    module_python = [
        code
        for code in (
            program.prelude,
            *(statement.code for statement in leading_toplevel_python),
        )
        if code is not None
    ]
    builder.add("from __future__ import annotations")
    for import_line in _typing_import_lines(program, imported_surface, module_python):
        builder.add(import_line)
    _emit_runtime_args_stub(builder)
    builder.add("")

    _emit_imported_surface_stubs(builder, imported_surface)

    # Prelude symbols participate in Kedi type annotations and procedure
    # defaults, so they must exist before the generated Kedi declarations.
    if program.prelude is not None:
        prelude_region = cursor.next("fenced")
        if _should_emit_full_module_python(prelude_region, focus_line):
            if prelude_region is not None:
                builder.add_mapped_region(prelude_region)
            else:
                for line in program.prelude.splitlines():
                    builder.add(line)
        else:
            _emit_python_module_stub(builder, program.prelude)
        builder.add("")

    for stmt in leading_toplevel_python:
        region = cursor.next("fenced", stmt.loc)
        if _should_emit_full_module_python(region, focus_line):
            if region is not None:
                builder.add_mapped_region(region)
            else:
                for line in stmt.code.splitlines():
                    builder.add(line)
        else:
            _emit_python_module_stub(builder, stmt.code)
        builder.add("")

    for profile in program.profile_defs.values():
        _emit_unconsumed_inline_regions(builder, cursor, profile.loc, "")

    for typedef in program.type_defs:
        _emit_type_stub(builder, typedef, header_ranges)
    if program.type_defs:
        builder.add("")

    top_level_scope = _collect_toplevel_scope(remaining_toplevel)
    _emit_toplevel_scope_stubs(builder, top_level_scope)

    for proc in program.procedures:
        _emit_proc_stub(builder, proc)
    if program.procedures:
        builder.add("")

    if focused_proc is None and (program.toplevel_body or program.toplevel_return is not None):
        builder.add("def __kedi_toplevel__() -> Any:")
        scope: dict[str, ScopeName] = {}
        before_body = builder.line
        _emit_statements(
            builder,
            cursor,
            remaining_toplevel,
            scope,
            "    ",
            header_ranges,
        )
        if program.toplevel_return is not None:
            _emit_return(builder, cursor, program.toplevel_return, scope, "    ")
        if builder.line == before_body:
            builder.add("    pass")
        builder.add("")

    for proc in program.procedures:
        if focused_proc is not None and proc is not focused_root:
            continue
        _emit_procedure_body(builder, cursor, proc, header_ranges)

    if focused_proc is None and isinstance(program, ProgramWithValidation):
        procedure_map = {proc.name: proc for proc in _all_procedures(program.procedures)}
        _emit_validation(builder, cursor, program, procedure_map)

    return {
        "text": builder.finish(),
        "ranges": builder.regions,
        "mappings": builder.mappings,
        "symbols": builder.symbols,
        "focusKey": focus_key,
    }


def _fallback_virtual_document(source: str, ranges: list[dict[str, Any]]) -> dict[str, Any]:
    builder = VirtualBuilder(source, None)
    builder.add("from __future__ import annotations")
    _emit_runtime_args_stub(builder)
    builder.add("")
    for r in ranges:
        builder.add_mapped_region(
            PythonRegion(kind=r["kind"], range=r["range"], text=r.get("text", ""))
        )
    return {
        "text": builder.finish(),
        "ranges": builder.regions,
        "mappings": builder.mappings,
        "symbols": builder.symbols,
        "focusKey": None,
    }


def main() -> None:
    payload = json.loads(sys.stdin.read() or "{}")
    text = payload.get("text", "")
    source_path = payload.get("sourcePath") or _path_from_uri(payload.get("uri"))
    focus_line = payload.get("focusLine")
    json.dump(
        compute_python_virtual_document(text, source_path, focus_line=focus_line),
        sys.stdout,
    )


def main_loop() -> None:
    for line in sys.stdin:
        if not line.strip():
            continue
        request_id = None
        try:
            payload = json.loads(line)
            request_id = payload.get("id")
            text = payload.get("text", "")
            source_path = payload.get("sourcePath") or _path_from_uri(payload.get("uri"))
            focus_line = payload.get("focusLine")
            if payload.get("method") == "completion":
                result = map_python_completions(text, payload["virtual"], payload["items"])
            else:
                result = compute_python_virtual_document(
                    text,
                    source_path,
                    focus_line=focus_line,
                )
            print(
                json.dumps({"id": request_id, "ok": True, "result": result}),
                flush=True,
            )
        except Exception as exc:
            print(
                json.dumps(
                    {
                        "id": request_id,
                        "ok": False,
                        "error": str(exc),
                    }
                ),
                flush=True,
            )


def _path_from_uri(uri: Any) -> Optional[str]:
    if not isinstance(uri, str) or not uri.startswith("file://"):
        return None
    parsed = urlparse(uri)
    return unquote(parsed.path)


if __name__ == "__main__":  # pragma: no cover
    main()
