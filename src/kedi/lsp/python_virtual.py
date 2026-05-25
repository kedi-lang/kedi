"""Scope-aware virtual Python documents for embedded Kedi Python.

The editor extensions use this module to ask Python tooling about code
that lives inside Kedi files.  The generated document is ordinary Python
with Kedi procedures, parameters, and local variables represented as
static stubs.  Source maps keep editor positions tied to the original
Kedi regions.
"""

from __future__ import annotations

import ast as py_ast
import json
import re
import sys
from dataclasses import dataclass
from typing import Any, Iterable, Optional
from urllib.parse import unquote, urlparse

from ..lang.ast import (
    AssignBlockStmt,
    AssignStmt,
    BacktickLineStmt,
    EvalDataset,
    EvalMetric,
    EvalSuite,
    OutputSegment,
    ProcDefStmt,
    ProcedureDef,
    ProgramWithValidation,
    PythonBlock,
    PythonExprNativeSegment,
    PythonExprSegment,
    ReturnBlockStmt,
    ReturnStmt,
    Span,
    Stmt,
    TemplateLine,
    TestCase,
    TypeApply,
    TypeDef,
    TypeExpr,
    TypeField,
    TypePython,
    TypeRef,
    TypeUnion,
)
from ..lang.parser import KediParseError, parse_program
from ..lang.type_render import type_expr_to_python
from .features import (
    _known_lsp_type_origin,
    compute_header_python_ranges,
    compute_python_ranges,
)

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
        dedent = (
            self._common_indent(start_line, end_line) if region.kind == "fenced" else 0
        )

        for source_line in range(start_line, end_line + 1):
            if source_line >= len(self.source_lines):
                continue
            line_text = self.source_lines[source_line]
            start_char = src_start["character"] if source_line == start_line else 0
            end_char = (
                src_end["character"] if source_line == end_line else len(line_text)
            )
            if source_line == end_line and end_char == 0:
                continue
            if dedent:
                start_char = min(
                    end_char, start_char + min(dedent, _indent_width(line_text))
                )
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
            PythonRegion(kind=r["kind"], range=r["range"], text=r.get("text", ""))
            for r in ranges
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


def compute_python_virtual_document(
    source: str,
    source_path: Optional[str] = None,
    *,
    focus_line: Optional[int] = None,
) -> dict[str, Any]:
    """Return a virtual Python document plus Kedi/Python source maps."""
    try:
        program = parse_program(source, source_path=source_path)
    except KediParseError:
        ranges = compute_python_ranges(source)
        return _fallback_virtual_document(source, ranges)

    cursor = RegionCursor(compute_python_ranges(source))
    header_defaults = _index_header_default_ranges(compute_header_python_ranges(source))
    builder = VirtualBuilder(source, source_path)
    focused_proc = _find_focused_procedure(program.procedures, focus_line)
    focus_key = (
        f"proc:{focused_proc.name}:{focused_proc.loc.start_line}"
        if focused_proc is not None and focused_proc.loc is not None
        else None
    )
    builder.add("from __future__ import annotations")
    for import_line in _typing_import_lines(program):
        builder.add(import_line)
    builder.add("")

    for typedef in program.type_defs:
        _emit_type_stub(builder, typedef, header_defaults)
    if program.type_defs:
        builder.add("")

    for proc in program.procedures:
        _emit_proc_stub(builder, proc)
    if program.procedures:
        builder.add("")

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

    leading_toplevel_python, remaining_toplevel = _split_leading_toplevel_python(
        program.toplevel_body
    )
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

    if focused_proc is None and (
        program.toplevel_body or program.toplevel_return is not None
    ):
        builder.add("def __kedi_toplevel__() -> Any:")
        scope: dict[str, ScopeName] = {}
        before_body = builder.line
        _emit_statements(
            builder,
            cursor,
            remaining_toplevel,
            scope,
            "    ",
            header_defaults,
        )
        if program.toplevel_return is not None:
            _emit_return(builder, cursor, program.toplevel_return, scope, "    ")
        if builder.line == before_body:
            builder.add("    pass")
        builder.add("")

    for proc in _all_procedures(program.procedures):
        if focused_proc is not None and proc is not focused_proc:
            continue
        _emit_procedure_body(builder, cursor, proc, header_defaults)

    if focused_proc is None and isinstance(program, ProgramWithValidation):
        procedure_map = {
            proc.name: proc for proc in _all_procedures(program.procedures)
        }
        _emit_validation(builder, cursor, program, procedure_map)

    return {
        "text": builder.finish(),
        "ranges": builder.regions,
        "mappings": builder.mappings,
        "symbols": builder.symbols,
        "focusKey": focus_key,
    }


def _fallback_virtual_document(
    source: str, ranges: list[dict[str, Any]]
) -> dict[str, Any]:
    builder = VirtualBuilder(source, None)
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


def _find_focused_procedure(
    procedures: list[ProcedureDef], focus_line: Optional[int]
) -> Optional[ProcedureDef]:
    if focus_line is None:
        return None
    source_line = focus_line + 1
    return _find_deepest_procedure_at_line(procedures, source_line)


def _find_deepest_procedure_at_line(
    procedures: Iterable[ProcedureDef], source_line: int
) -> Optional[ProcedureDef]:
    for proc in procedures:
        if proc.loc is None or not (
            proc.loc.start_line <= source_line <= proc.loc.end_line
        ):
            continue
        nested = _find_deepest_procedure_at_line(_nested_procedures(proc), source_line)
        return nested or proc
    return None


def _split_leading_toplevel_python(
    statements: list[Stmt],
) -> tuple[list[PythonBlock], list[Stmt]]:
    leading: list[PythonBlock] = []
    index = 0
    while index < len(statements) and isinstance(statements[index], PythonBlock):
        leading.append(statements[index])
        index += 1
    return leading, statements[index:]


def _should_emit_full_module_python(
    region: Optional[PythonRegion], focus_line: Optional[int]
) -> bool:
    if focus_line is None:
        return True
    if region is None:
        return False
    return region.range["start"]["line"] <= focus_line < region.range["end"]["line"]


def _emit_python_module_stub(builder: VirtualBuilder, code: str) -> None:
    try:
        module = py_ast.parse(code)
    except SyntaxError:
        builder.add("# Kedi Python module block omitted from this focused view.")
        return

    emitted = False
    for stmt in module.body:
        emitted = _emit_python_stub_stmt(builder, stmt) or emitted
    if not emitted:
        builder.add("# Kedi Python module block omitted from this focused view.")


def _emit_python_stub_stmt(builder: VirtualBuilder, stmt: py_ast.stmt) -> bool:
    if isinstance(stmt, py_ast.Import):
        emitted = False
        for alias in stmt.names:
            name = alias.asname or alias.name.split(".", 1)[0]
            builder.add(f"{name}: Any = ...")
            emitted = True
        return emitted
    if isinstance(stmt, py_ast.ImportFrom):
        emitted = False
        for alias in stmt.names:
            if alias.name == "*":
                continue
            name = alias.asname or alias.name
            if name in {"Field"}:
                builder.add(f"def {name}(*args: Any, **kwargs: Any) -> Any: ...")
            elif name[:1].isupper():
                builder.add(f"class {name}: ...")
            else:
                builder.add(f"{name}: Any = ...")
            emitted = True
        return emitted
    if isinstance(stmt, py_ast.ClassDef):
        _emit_python_class_stub(builder, stmt)
        return True
    if isinstance(stmt, (py_ast.FunctionDef, py_ast.AsyncFunctionDef)):
        builder.add(_python_function_stub(stmt))
        return True
    names = _assigned_python_names(stmt)
    for name in names:
        builder.add(f"{name}: Any = ...")
    return bool(names)


def _emit_python_class_stub(builder: VirtualBuilder, stmt: py_ast.ClassDef) -> None:
    builder.add(f"class {stmt.name}:")
    fields = _python_class_fields(stmt)
    methods = [
        item
        for item in stmt.body
        if isinstance(item, (py_ast.FunctionDef, py_ast.AsyncFunctionDef))
    ]
    if not fields and not methods:
        builder.add("    pass")
        return
    for name, typ, _has_default in fields:
        builder.add(f"    {name}: {typ}")
    if fields:
        params = ", ".join(
            _python_class_init_param(name, typ, has_default)
            for name, typ, has_default in fields
        )
        builder.add(f"    def __init__(self, {params}) -> None: ...")
    for method in methods:
        builder.add(f"    {_python_function_stub(method)}")


def _python_class_fields(stmt: py_ast.ClassDef) -> list[tuple[str, str, bool]]:
    fields: list[tuple[str, str, bool]] = []
    for item in stmt.body:
        if isinstance(item, py_ast.AnnAssign) and isinstance(item.target, py_ast.Name):
            fields.append(
                (
                    item.target.id,
                    _unparse_or_any(item.annotation),
                    _python_field_has_default(item.value),
                )
            )
        elif isinstance(item, py_ast.Assign):
            for target in item.targets:
                if isinstance(target, py_ast.Name):
                    fields.append((target.id, "Any", True))
    return fields


def _python_field_has_default(value: Optional[py_ast.expr]) -> bool:
    if value is None:
        return False
    if isinstance(value, py_ast.Call) and _python_call_name(value.func) == "Field":
        if not value.args:
            return any(keyword.arg == "default" for keyword in value.keywords)
        first = value.args[0]
        return not (isinstance(first, py_ast.Constant) and first.value is Ellipsis)
    return True


def _python_class_init_param(name: str, typ: str, has_default: bool) -> str:
    piece = f"{name}: {typ}"
    if has_default:
        piece += " = ..."
    return piece


def _python_function_stub(stmt: py_ast.FunctionDef | py_ast.AsyncFunctionDef) -> str:
    prefix = "async def" if isinstance(stmt, py_ast.AsyncFunctionDef) else "def"
    args = py_ast.unparse(stmt.args)
    ret = f" -> {_unparse_or_any(stmt.returns)}" if stmt.returns is not None else ""
    return f"{prefix} {stmt.name}({args}){ret}: ..."


def _assigned_python_names(stmt: py_ast.stmt) -> list[str]:
    names: list[str] = []
    if isinstance(stmt, py_ast.Assign):
        for target in stmt.targets:
            names.extend(_target_python_names(target))
    elif isinstance(stmt, py_ast.AnnAssign):
        names.extend(_target_python_names(stmt.target))
    return names


def _target_python_names(target: py_ast.expr) -> list[str]:
    if isinstance(target, py_ast.Name):
        return [target.id]
    if isinstance(target, (py_ast.Tuple, py_ast.List)):
        names: list[str] = []
        for item in target.elts:
            names.extend(_target_python_names(item))
        return names
    return []


def _python_call_name(expr: py_ast.expr) -> str:
    if isinstance(expr, py_ast.Name):
        return expr.id
    if isinstance(expr, py_ast.Attribute):
        return expr.attr
    return ""


def _unparse_or_any(expr: Optional[py_ast.AST]) -> str:
    if expr is None:
        return "Any"
    try:
        return py_ast.unparse(expr)
    except Exception:
        return "Any"


def _nested_procedures(proc: ProcedureDef) -> list[ProcedureDef]:
    return [
        stmt.proc
        for stmt in proc.body
        if isinstance(stmt, ProcDefStmt) and stmt.proc is not None
    ]


def _all_procedures(procedures: Iterable[ProcedureDef]) -> list[ProcedureDef]:
    out: list[ProcedureDef] = []

    def visit(proc: ProcedureDef) -> None:
        out.append(proc)
        for child in _nested_procedures(proc):
            visit(child)

    for proc in procedures:
        visit(proc)
    return out


def _emit_type_stub(
    builder: VirtualBuilder,
    typedef: TypeDef,
    header_defaults: dict[str, dict[int, list[Range]]],
) -> None:
    line = builder.add(f"class {typedef.name}:")
    builder.add_symbol(
        kind="type",
        name=typedef.name,
        source_range=_type_name_range(typedef),
        virtual_range=_line_range(line, len("class "), typedef.name),
    )
    if not typedef.fields:
        builder.add("    pass")
    else:
        default_ranges = list(
            header_defaults.get("type_field", {}).get(_container_line(typedef), [])
        )
        for field in typedef.fields:
            field_line_text = f"    {_type_field_to_python(field)}"
            field_line = builder.add(field_line_text)
            if field.default_expr is not None and default_ranges:
                source_range = default_ranges.pop(0)
                _add_default_expr_region(
                    builder,
                    source_range=source_range,
                    line=field_line,
                    line_text=field_line_text,
                    expr=field.default_expr,
                )
        builder.add(f"    def __init__({_type_init_params(typedef)}) -> None: ...")


def _emit_proc_stub(
    builder: VirtualBuilder, proc: ProcedureDef, prefix: str = ""
) -> None:
    params = ", ".join(_proc_param_to_python(proc, name) for name in proc.params)
    ret = _type_to_python(
        proc.return_type_expr, proc.return_type_annotation, default="str"
    )
    line = builder.add(f"{prefix}def {proc.name}({params}) -> {ret}: ...")
    builder.add_symbol(
        kind="procedure",
        name=proc.name,
        source_range=_proc_name_range(proc),
        virtual_range=_line_range(line, len(prefix) + len("def "), proc.name),
    )


def _emit_procedure_body(
    builder: VirtualBuilder,
    cursor: RegionCursor,
    proc: ProcedureDef,
    header_defaults: dict[str, dict[int, list[Range]]],
) -> None:
    params = ", ".join(_proc_param_to_python(proc, name) for name in proc.params)
    ret = _type_to_python(
        proc.return_type_expr, proc.return_type_annotation, default="str"
    )
    signature = f"def __kedi_scope_{proc.name}({params}) -> {ret}:"
    signature_line = builder.add(signature)
    _add_proc_default_regions(
        builder,
        proc,
        signature_line=signature_line,
        signature=signature,
        header_defaults=header_defaults,
    )
    scope = {
        name: ScopeName(
            typ=_type_to_python(
                (proc.param_type_exprs or {}).get(name),
                (proc.param_type_annotations or {}).get(name),
                default="Any",
            ),
            source_range=_param_name_range(builder.source_lines, proc, name),
        )
        for name in proc.params
    }
    _add_scope_param_symbols(builder, scope, signature_line, signature)
    before_body = builder.line
    _emit_statements(builder, cursor, proc.body, scope, "    ", header_defaults)
    if builder.line == before_body:
        builder.add("    pass")
    builder.add("")


def _emit_nested_procedure(
    builder: VirtualBuilder,
    cursor: RegionCursor,
    proc: ProcedureDef,
    indent: str,
    header_defaults: dict[str, dict[int, list[Range]]],
) -> None:
    params = ", ".join(_proc_param_to_python(proc, name) for name in proc.params)
    ret = _type_to_python(
        proc.return_type_expr, proc.return_type_annotation, default="str"
    )
    signature = f"{indent}def {proc.name}({params}) -> {ret}:"
    signature_line = builder.add(signature)
    _add_proc_default_regions(
        builder,
        proc,
        signature_line=signature_line,
        signature=signature,
        header_defaults=header_defaults,
    )
    scope = {
        name: ScopeName(
            typ=_type_to_python(
                (proc.param_type_exprs or {}).get(name),
                (proc.param_type_annotations or {}).get(name),
                default="Any",
            ),
            source_range=_param_name_range(builder.source_lines, proc, name),
        )
        for name in proc.params
    }
    builder.add_symbol(
        kind="procedure",
        name=proc.name,
        source_range=_proc_name_range(proc),
        virtual_range=_line_range(signature_line, len(indent) + len("def "), proc.name),
    )
    _add_scope_param_symbols(builder, scope, signature_line, signature)
    before_body = builder.line
    _emit_statements(
        builder,
        cursor,
        proc.body,
        scope,
        indent + "    ",
        header_defaults,
    )
    if builder.line == before_body:
        builder.add(f"{indent}    pass")


def _emit_statements(
    builder: VirtualBuilder,
    cursor: RegionCursor,
    statements: Iterable[Stmt],
    scope: dict[str, ScopeName],
    indent: str,
    header_defaults: dict[str, dict[int, list[Range]]],
) -> None:
    for stmt in statements:
        if isinstance(stmt, ProcDefStmt) and stmt.proc is not None:
            _emit_nested_procedure(builder, cursor, stmt.proc, indent, header_defaults)
            continue
        if isinstance(stmt, AssignBlockStmt):
            _emit_assign_block(builder, cursor, stmt, scope, indent)
            continue
        if isinstance(stmt, AssignStmt):
            _emit_assign_stmt(builder, cursor, stmt, scope, indent)
            continue
        if isinstance(stmt, PythonBlock):
            region = cursor.next("fenced", stmt.loc)
            if region is not None:
                builder.add_mapped_region(region, indent)
            continue
        if isinstance(stmt, BacktickLineStmt):
            region = cursor.next("inline", stmt.loc)
            if region is not None:
                builder.add_mapped_region(region, indent)
            continue
        if isinstance(stmt, TemplateLine):
            _emit_segments(builder, cursor, stmt, scope, indent)
            continue
        if isinstance(stmt, ReturnStmt):
            _emit_return(builder, cursor, stmt, scope, indent)
            continue
        if isinstance(stmt, ReturnBlockStmt):
            _emit_return(builder, cursor, stmt, scope, indent)


def _emit_assign_block(
    builder: VirtualBuilder,
    cursor: RegionCursor,
    stmt: AssignBlockStmt,
    scope: dict[str, ScopeName],
    indent: str,
) -> None:
    typ = _type_to_python(stmt.type_expr, stmt.type_annotation)
    helper = (
        f"__kedi_block_{stmt.name}_{stmt.loc.start_line if stmt.loc else len(scope)}"
    )
    params = _scope_params(scope)
    signature = f"{indent}def {helper}({params}) -> {typ}:"
    signature_line = builder.add(signature)
    _add_scope_param_symbols(builder, scope, signature_line, signature)
    region = cursor.next("fenced", stmt.loc)
    if region is not None:
        builder.add_mapped_region(region, indent + "    ")
    else:
        builder.add(f"{indent}    ...")
    args = ", ".join(scope.keys())
    line = builder.add(f"{indent}{stmt.name}: {typ} = {helper}({args})")
    source_range = _assign_name_range(stmt)
    builder.add_symbol(
        kind="local",
        name=stmt.name,
        source_range=source_range,
        virtual_range=_line_range(line, len(indent), stmt.name),
    )
    scope[stmt.name] = ScopeName(typ=typ, source_range=source_range)


def _emit_assign_stmt(
    builder: VirtualBuilder,
    cursor: RegionCursor,
    stmt: AssignStmt,
    scope: dict[str, ScopeName],
    indent: str,
) -> None:
    typ = _type_to_python(stmt.type_expr, stmt.type_annotation)
    source_range = _assign_name_range(stmt)
    inline = _single_python_segment(stmt.expr)
    if inline is not None:
        line = builder.add(f"{indent}{stmt.name}: {typ} = (")
        builder.add_symbol(
            kind="local",
            name=stmt.name,
            source_range=source_range,
            virtual_range=_line_range(line, len(indent), stmt.name),
        )
        region = cursor.next("inline", inline.loc)
        if region is not None:
            builder.add_mapped_region(region, indent + "    ")
        else:
            builder.add(f"{indent}    ...")
        builder.add(f"{indent})")
    else:
        line = builder.add(f"{indent}{stmt.name}: {typ} = ...")
        builder.add_symbol(
            kind="local",
            name=stmt.name,
            source_range=source_range,
            virtual_range=_line_range(line, len(indent), stmt.name),
        )
        _emit_segments(builder, cursor, stmt, scope, indent)
    scope[stmt.name] = ScopeName(typ=typ, source_range=source_range)


def _emit_return(
    builder: VirtualBuilder,
    cursor: RegionCursor,
    stmt: ReturnStmt | ReturnBlockStmt,
    scope: dict[str, ScopeName],
    indent: str,
) -> None:
    if isinstance(stmt, ReturnBlockStmt):
        region = cursor.next("fenced", stmt.loc)
        if region is not None:
            builder.add_mapped_region(region, indent)
        return
    _emit_segments(builder, cursor, stmt, scope, indent)


def _emit_segments(
    builder: VirtualBuilder,
    cursor: RegionCursor,
    stmt: Any,
    scope: dict[str, ScopeName],
    indent: str,
) -> None:
    for segment in list(getattr(stmt, "segments", []) or []) + list(
        getattr(stmt, "expr", []) or []
    ):
        if isinstance(segment, OutputSegment):
            typ = _type_to_python(segment.type_expr, segment.type_annotation)
            line = builder.add(f"{indent}{segment.name}: {typ} = ...")
            source_range = _output_name_range(segment)
            builder.add_symbol(
                kind="local",
                name=segment.name,
                source_range=source_range,
                virtual_range=_line_range(line, len(indent), segment.name),
            )
            scope[segment.name] = ScopeName(typ=typ, source_range=source_range)
        elif isinstance(segment, (PythonExprSegment, PythonExprNativeSegment)):
            region = cursor.next("inline", segment.loc)
            if region is not None:
                builder.add_mapped_region(region, indent)


def _emit_validation(
    builder: VirtualBuilder,
    cursor: RegionCursor,
    program: ProgramWithValidation,
    procedure_map: dict[str, ProcedureDef],
) -> None:
    for suite in program.tests:
        for case in suite.cases:
            _emit_validation_code(
                builder, cursor, case, f"__kedi_test_{suite.procedure_name}_{case.name}"
            )
    for suite in program.evals:
        for dataset in list(suite.datasets.values()) + list(
            suite.test_datasets.values()
        ):
            _emit_validation_code(
                builder,
                cursor,
                dataset,
                f"__kedi_data_{suite.procedure_name}_{dataset.name}",
            )
        for metric in suite.metrics:
            _emit_metric(
                builder,
                cursor,
                suite,
                metric,
                procedure_map.get(suite.procedure_name),
            )


def _emit_validation_code(
    builder: VirtualBuilder,
    cursor: RegionCursor,
    item: TestCase | EvalDataset,
    name: str,
) -> None:
    builder.add(f"def {name}() -> Any:")
    region = cursor.next("fenced" if item.is_block else "inline", item.loc)
    if region is not None:
        builder.add_mapped_region(region, "    ")
    else:
        builder.add("    pass")
    builder.add("")


def _emit_metric(
    builder: VirtualBuilder,
    cursor: RegionCursor,
    suite: EvalSuite,
    metric: EvalMetric,
    procedure: Optional[ProcedureDef],
) -> None:
    builder.add(f"def __kedi_metric_{suite.procedure_name}_{metric.name}() -> Any:")
    if metric.dataset_name:
        builder.add(f"    {metric.dataset_name}: Any = ...")
    builder.add("    expected: Any = ...")
    if procedure is not None:
        for name in procedure.params:
            if name == metric.dataset_name:
                continue
            builder.add(f"    {name}: Any = ...")
    region = cursor.next("fenced" if metric.is_block else "inline", metric.loc)
    if region is not None:
        builder.add_mapped_region(region, "    ")
    else:
        builder.add("    pass")
    builder.add("")


def _single_python_segment(
    expr: list[Any],
) -> Optional[PythonExprNativeSegment | PythonExprSegment]:
    if len(expr) != 1:
        return None
    segment = expr[0]
    if isinstance(segment, (PythonExprNativeSegment, PythonExprSegment)):
        return segment
    return None


def _scope_params(scope: dict[str, ScopeName]) -> str:
    return ", ".join(f"{name}: {info.typ}" for name, info in scope.items())


def _add_scope_param_symbols(
    builder: VirtualBuilder,
    scope: dict[str, ScopeName],
    line: int,
    text: str,
) -> None:
    search_from = text.find("(") + 1
    if search_from <= 0:
        return
    for name, info in scope.items():
        start = text.find(name, search_from)
        if start == -1:
            continue
        builder.add_symbol(
            kind="parameter",
            name=name,
            source_range=info.source_range,
            virtual_range=_line_range(line, start, name),
        )
        search_from = start + len(name)


def _line_range(line: int, start: int, text: str) -> Range:
    return {
        "start": {"line": line, "character": start},
        "end": {"line": line, "character": start + len(text)},
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


def _proc_name_range(proc: ProcedureDef) -> Optional[Range]:
    if proc.loc is None:
        return None
    line = proc.loc.start_line - 1
    start = proc.loc.start_col
    return _line_range(line, start, proc.name)


def _type_name_range(typedef: TypeDef) -> Optional[Range]:
    if typedef.loc is None:
        return None
    line = typedef.loc.start_line - 1
    start = typedef.loc.start_col
    return _line_range(line, start, typedef.name)


def _assign_name_range(stmt: AssignStmt | AssignBlockStmt) -> Optional[Range]:
    if stmt.loc is None:
        return None
    line = stmt.loc.start_line - 1
    start = stmt.loc.start_col
    return _line_range(line, start, stmt.name)


def _output_name_range(segment: OutputSegment) -> Optional[Range]:
    if segment.loc is None:
        return None
    line = segment.loc.start_line - 1
    start = segment.loc.start_col
    return _line_range(line, start, segment.name)


def _param_name_range(
    source_lines: list[str],
    proc: ProcedureDef,
    name: str,
) -> Optional[Range]:
    if proc.loc is None:
        return None
    line = proc.loc.start_line - 1
    if line < 0 or line >= len(source_lines):
        return None
    header = source_lines[line]
    open_paren = header.find("(")
    close_paren = header.find(")", open_paren + 1)
    if open_paren == -1:
        return None
    if close_paren == -1:
        close_paren = len(header)
    params = header[open_paren + 1 : close_paren]
    offset = 0
    for chunk in params.split(","):
        stripped = chunk.lstrip()
        leading = len(chunk) - len(stripped)
        candidate = stripped.split(":", 1)[0].strip()
        if candidate == name:
            start = open_paren + 1 + offset + leading
            return _line_range(line, start, name)
        offset += len(chunk) + 1
    return None


def _index_header_default_ranges(
    ranges: list[dict[str, Any]],
) -> dict[str, dict[int, list[Range]]]:
    indexed: dict[str, dict[int, list[Range]]] = {}
    for item in ranges:
        owner_kind = item.get("ownerKind")
        container_line = item.get("containerLine")
        if owner_kind is None or container_line is None:
            continue
        indexed.setdefault(owner_kind, {}).setdefault(container_line, []).append(
            item["range"]
        )
    return indexed


def _container_line(node: ProcedureDef | TypeDef) -> int:
    if node.loc is None:
        return -1
    return node.loc.start_line - 1


def _add_proc_default_regions(
    builder: VirtualBuilder,
    proc: ProcedureDef,
    *,
    signature_line: int,
    signature: str,
    header_defaults: dict[str, dict[int, list[Range]]],
) -> None:
    default_ranges = list(
        header_defaults.get("param", {}).get(_container_line(proc), [])
    )
    if not default_ranges:
        return
    search_from = signature.find("(") + 1
    for name in proc.params:
        piece = _proc_param_to_python(proc, name)
        start = signature.find(piece, search_from)
        if start == -1:
            continue
        default_expr = (proc.param_defaults or {}).get(name)
        if default_expr is not None and default_ranges:
            source_range = default_ranges.pop(0)
            expr_start = signature.find(default_expr, start, start + len(piece))
            if expr_start != -1:
                builder.add_synthetic_inline_region(
                    source_range=source_range,
                    virtual_range=_line_range(signature_line, expr_start, default_expr),
                    text=default_expr,
                )
        search_from = start + len(piece)


def _add_default_expr_region(
    builder: VirtualBuilder,
    *,
    source_range: Range,
    line: int,
    line_text: str,
    expr: str,
) -> None:
    start = line_text.find(expr)
    if start == -1:
        return
    builder.add_synthetic_inline_region(
        source_range=source_range,
        virtual_range=_line_range(line, start, expr),
        text=expr,
    )


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


def _typing_import_lines(program: Any) -> list[str]:
    imports_by_origin: dict[str, set[str]] = {"typing": {"Any"}}
    for expr in _iter_type_exprs(program):
        for origin, name in _typing_symbols_in_type_expr(expr):
            imports_by_origin.setdefault(origin, set()).add(name)

    lines: list[str] = []
    for origin in ("typing", "typing_extensions"):
        names = sorted(imports_by_origin.get(origin, ()))
        if names:
            lines.append(f"from {origin} import {', '.join(names)}")
    return lines


def _iter_type_exprs(program: Any) -> Iterable[Optional[TypeExpr]]:
    for typedef in getattr(program, "type_defs", ()):
        for field in typedef.fields:
            yield field.type_expr
    for proc in getattr(program, "procedures", ()):
        yield proc.return_type_expr
        for expr in (proc.param_type_exprs or {}).values():
            yield expr
        for stmt in proc.body:
            yield from _iter_stmt_type_exprs(stmt)
    for stmt in getattr(program, "toplevel_body", ()):
        yield from _iter_stmt_type_exprs(stmt)


def _iter_stmt_type_exprs(stmt: Stmt) -> Iterable[Optional[TypeExpr]]:
    if isinstance(stmt, (AssignStmt, AssignBlockStmt)):
        yield stmt.type_expr
    if isinstance(stmt, ProcDefStmt) and stmt.proc is not None:
        yield stmt.proc.return_type_expr
        for expr in (stmt.proc.param_type_exprs or {}).values():
            yield expr
        for child in stmt.proc.body:
            yield from _iter_stmt_type_exprs(child)
    for segment in list(getattr(stmt, "segments", []) or []) + list(
        getattr(stmt, "expr", []) or []
    ):
        if isinstance(segment, OutputSegment):
            yield segment.type_expr


def _typing_symbols_in_type_expr(expr: Optional[TypeExpr]) -> set[tuple[str, str]]:
    if expr is None:
        return set()
    if isinstance(expr, TypeRef):
        origin = _known_lsp_type_origin(expr.name)
        return {(origin, expr.name)} if origin is not None else set()
    if isinstance(expr, TypeApply):
        found = set()
        origin = _known_lsp_type_origin(expr.name)
        if origin is not None:
            found.add((origin, expr.name))
        for arg in expr.args:
            found.update(_typing_symbols_in_type_expr(arg))
        return found
    if isinstance(expr, TypeUnion):
        found = set()
        found.update(_typing_symbols_in_type_expr(expr.left))
        found.update(_typing_symbols_in_type_expr(expr.right))
        return found
    if isinstance(expr, TypePython):
        return _typing_symbols_in_python_type(expr.code)
    return set()


def _typing_symbols_in_python_type(code: str) -> set[tuple[str, str]]:
    found: set[tuple[str, str]] = set()
    for name in set(re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\b", code)):
        origin = _known_lsp_type_origin(name)
        if origin is not None:
            found.add((origin, name))
    return found


def _type_to_python(
    expr: Optional[TypeExpr], annotation: Optional[str], *, default: str = "str"
) -> str:
    return type_expr_to_python(expr, annotation, default=default)


def _proc_param_to_python(proc: ProcedureDef, name: str) -> str:
    typ = _type_to_python(
        (proc.param_type_exprs or {}).get(name),
        (proc.param_type_annotations or {}).get(name),
        default="Any",
    )
    piece = f"{name}: {typ}"
    default = (proc.param_defaults or {}).get(name)
    if default is not None:
        piece += f" = {default}"
    return piece


def _type_field_to_python(field: TypeField) -> str:
    typ = _type_to_python(field.type_expr, field.type_annotation)
    piece = f"{field.name}: {typ}"
    if field.default_expr is not None:
        piece += f" = {field.default_expr}"
    return piece


def _type_init_params(typedef: TypeDef) -> str:
    if not typedef.fields:
        return "self"
    fields = ", ".join(_type_field_to_python(field) for field in typedef.fields)
    return f"self, {fields}"


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
            source_path = payload.get("sourcePath") or _path_from_uri(
                payload.get("uri")
            )
            focus_line = payload.get("focusLine")
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


if __name__ == "__main__":
    main()
