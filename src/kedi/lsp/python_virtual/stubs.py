from __future__ import annotations

import ast as py_ast
import re
from typing import Any, Iterable, Optional

from kedi.lang.ast import (
    ProcedureDef,
    TypeApply,
    TypeDef,
    TypeExpr,
    TypeField,
    TypePython,
    TypeRef,
    TypeUnion,
)
from kedi.lsp.features.type_support import _known_lsp_type_origin
from kedi.lsp.module_surface import ImportedSurface, ImportedValueInfo
from kedi.lsp.python_virtual.builder import ScopeName, VirtualBuilder, _line_range, _span_range
from kedi.lsp.python_virtual.scope import (
    _add_default_expr_region,
    _container_line,
    _imported_value_name_start,
    _iter_imported_type_exprs,
    _iter_type_exprs,
    _proc_name_range,
    _proc_param_to_python,
    _type_name_range,
    _type_to_python,
    _unparse_or_any,
)


def _emit_runtime_args_stub(builder: VirtualBuilder) -> None:
    builder.add("from typing import Any, Generic, Iterable, TypeAlias, TypeVar, Union, cast")
    builder.add("_K = TypeVar('_K')")
    builder.add("_V = TypeVar('_V')")
    builder.add("ArgType: TypeAlias = Union[str, bool, None]")
    builder.add("def _kedi_placeholder() -> Any: return None")
    builder.add("class AttrDict(dict[_K, _V], Generic[_K, _V]):")
    builder.add("    def __getattr__(self, key: _K) -> _V: ...")
    builder.add("args: AttrDict[str, ArgType] = cast(AttrDict[str, ArgType], _kedi_placeholder())")


def _emit_imported_surface_stubs(
    builder: VirtualBuilder,
    imported_surface: ImportedSurface,
) -> None:
    emitted = False
    for typedef in imported_surface.types.values():
        _emit_type_stub(builder, typedef, {})
        emitted = True
    for proc in imported_surface.procedures.values():
        _emit_proc_stub(builder, proc)
        emitted = True
    for value in imported_surface.values.values():
        _emit_imported_value_stub(builder, value)
        emitted = True
    if emitted:
        builder.add("")


def _emit_imported_value_stub(
    builder: VirtualBuilder,
    value: ImportedValueInfo,
) -> None:
    line_text = value.python_stub
    if line_text is None:
        typ = _type_to_python(value.type_expr, None)
        line_text = f"{value.name}: {typ} = cast({typ}, _kedi_placeholder())"
    else:
        line_text = _checker_safe_value_stub(line_text)
    line = builder.line
    for stub_line in line_text.splitlines() or [line_text]:
        builder.add(stub_line)
    builder.add_symbol(
        kind="imported_value",
        name=value.name,
        source_range=_span_range(value.declaration_span),
        virtual_range=_line_range(
            line, _imported_value_name_start(line_text, value.name), value.name
        ),
    )


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
        if all(
            alias.name in {"typing", "typing_extensions", "datetime", "re"} for alias in stmt.names
        ):
            builder.add(py_ast.unparse(stmt))
            return True
        emitted = False
        for alias in stmt.names:
            name = alias.asname or alias.name.split(".", 1)[0]
            builder.add(f"{name}: Any = cast(Any, _kedi_placeholder())")
            emitted = True
        return emitted
    if isinstance(stmt, py_ast.ImportFrom):
        if stmt.module in {"typing", "typing_extensions", "datetime", "re"}:
            builder.add(py_ast.unparse(stmt))
            return True
        emitted = False
        for alias in stmt.names:
            if alias.name == "*":
                continue
            name = alias.asname or alias.name
            if name in {"Field"}:
                builder.add(
                    f"def {name}(*args: Any, **kwargs: Any) -> Any: "
                    "return cast(Any, _kedi_placeholder())"
                )
            elif name[:1].isupper():
                builder.add(f"class {name}: pass")
            else:
                builder.add(f"{name}: Any = cast(Any, _kedi_placeholder())")
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
        annotation = _assigned_python_annotation(stmt, name)
        builder.add(f"{name}: {annotation} = cast({annotation}, _kedi_placeholder())")
    return bool(names)


def _assigned_python_annotation(stmt: py_ast.stmt, name: str) -> str:
    if (
        isinstance(stmt, py_ast.AnnAssign)
        and isinstance(stmt.target, py_ast.Name)
        and stmt.target.id == name
    ):
        return _unparse_or_any(stmt.annotation)
    return "Any"


def _checker_safe_value_stub(line_text: str) -> str:
    """Replace value ellipses with a typed placeholder accepted by Pyright."""
    try:
        statement = py_ast.parse(line_text).body[0]
    except (SyntaxError, IndexError):
        return line_text
    if not isinstance(statement, py_ast.AnnAssign) or not isinstance(statement.target, py_ast.Name):
        return line_text
    typ = _unparse_or_any(statement.annotation)
    return f"{statement.target.id}: {typ} = cast({typ}, _kedi_placeholder())"


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
    has_explicit_init = any(method.name == "__init__" for method in methods)
    if fields and not has_explicit_init:
        params = ", ".join(
            _python_class_init_param(name, typ, has_default) for name, typ, has_default in fields
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
        piece += f" = cast({typ}, _kedi_placeholder())"
    return piece


def _python_function_stub(stmt: py_ast.FunctionDef | py_ast.AsyncFunctionDef) -> str:
    prefix = "async def" if isinstance(stmt, py_ast.AsyncFunctionDef) else "def"
    args = py_ast.unparse(stmt.args)
    ret = f" -> {_unparse_or_any(stmt.returns)}" if stmt.returns is not None else ""
    return_type = _unparse_or_any(stmt.returns)
    return f"{prefix} {stmt.name}({args}){ret}: return cast({return_type}, _kedi_placeholder())"


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


def _emit_type_stub(
    builder: VirtualBuilder,
    typedef: TypeDef,
    header_ranges: dict[str, dict[int, list[dict[str, Any]]]],
    prefix: str = "",
) -> None:
    line = builder.add(f"{prefix}class {typedef.name}:")
    builder.add_symbol(
        kind="type",
        name=typedef.name,
        source_range=_type_name_range(typedef),
        virtual_range=_line_range(line, len(prefix) + len("class "), typedef.name),
    )
    if not typedef.fields:
        builder.add(f"{prefix}    pass")
    else:
        field_ranges = header_ranges.get("type_field", {}).get(_container_line(typedef), [])
        type_ranges = [item["range"] for item in field_ranges if item["expressionKind"] == "type"]
        default_ranges = [
            item["range"] for item in field_ranges if item["expressionKind"] == "default"
        ]
        for field in typedef.fields:
            field_line_text = f"{prefix}    {_type_field_to_python(field)}"
            field_line = builder.add(field_line_text)
            if isinstance(field.type_expr, TypePython) and type_ranges:
                _add_default_expr_region(
                    builder,
                    source_range=type_ranges.pop(0),
                    line=field_line,
                    line_text=field_line_text,
                    expr=_type_to_python(field.type_expr, field.type_annotation),
                )
            if field.default_expr is not None and default_ranges:
                source_range = default_ranges.pop(0)
                _add_default_expr_region(
                    builder,
                    source_range=source_range,
                    line=field_line,
                    line_text=field_line_text,
                    expr=field.default_expr,
                )
        builder.add(f"{prefix}    def __init__({_type_init_params(typedef)}) -> None: ...")


def _emit_proc_stub(builder: VirtualBuilder, proc: ProcedureDef, prefix: str = "") -> None:
    params = ", ".join(_proc_param_to_python(proc, name) for name in proc.params)
    ret = _type_to_python(proc.return_type_expr, proc.return_type_annotation, default="str")
    if proc.docstring:
        line = builder.add(f"{prefix}def {proc.name}({params}) -> {ret}:")
        for doc_line in _docstring_lines(proc.docstring, prefix + "    "):
            builder.add(doc_line)
        builder.add(f"{prefix}    return cast({ret}, _kedi_placeholder())")
    else:
        line = builder.add(
            f"{prefix}def {proc.name}({params}) -> {ret}: return cast({ret}, _kedi_placeholder())"
        )
    builder.add_symbol(
        kind="procedure",
        name=proc.name,
        source_range=_proc_name_range(proc),
        virtual_range=_line_range(line, len(prefix) + len("def "), proc.name),
    )


def _docstring_lines(docstring: str, indent: str) -> list[str]:
    escaped = docstring.replace('"""', '\\"""')
    lines = escaped.splitlines() or [""]
    if len(lines) == 1:
        return [f'{indent}"""{lines[0]}"""']
    out = [f'{indent}"""{lines[0]}']
    out.extend(f"{indent}{line}" for line in lines[1:-1])
    out.append(f'{indent}{lines[-1]}"""')
    return out


def _emit_toplevel_scope_stubs(builder: VirtualBuilder, scope: dict[str, ScopeName]) -> None:
    for name, info in scope.items():
        builder.add(f"{name}: {info.typ} = cast({info.typ}, _kedi_placeholder())")
    if scope:
        builder.add("")


def _typing_import_lines(
    program: Any,
    imported_surface: ImportedSurface | None = None,
    module_python: Iterable[str] = (),
) -> list[str]:
    imports_by_origin: dict[str, set[str]] = {"typing": {"Any"}}
    for expr in _iter_type_exprs(program):
        for origin, name in _typing_symbols_in_type_expr(expr):
            imports_by_origin.setdefault(origin, set()).add(name)
    if imported_surface is not None:
        for expr in _iter_imported_type_exprs(imported_surface):
            for origin, name in _typing_symbols_in_type_expr(expr):
                imports_by_origin.setdefault(origin, set()).add(name)
        for value in imported_surface.values.values():
            for origin, name in _typing_symbols_in_python_stub(value.python_stub):
                imports_by_origin.setdefault(origin, set()).add(name)

    provided_names, star_origins = _provided_python_type_imports(module_python)
    for origin, names in imports_by_origin.items():
        if origin in star_origins:
            names.clear()
        else:
            names.difference_update(provided_names)

    lines: list[str] = []
    for origin in ("typing", "typing_extensions"):
        names = sorted(imports_by_origin.get(origin, ()))
        if names:
            lines.append(f"from {origin} import {', '.join(names)}")
    datetime_names = sorted(imports_by_origin.get("datetime", ()))
    if datetime_names:
        lines.append(f"from datetime import {', '.join(datetime_names)}")
    if "Regex" in imports_by_origin.get("re", ()):
        lines.append("from re import Pattern as Regex")
    pydantic_names = imports_by_origin.get("pydantic", set())
    pydantic_imports = []
    if "Email" in pydantic_names:
        pydantic_imports.append("EmailStr as Email")
    for name in ("FileUrl", "HttpUrl"):
        if name in pydantic_names:
            pydantic_imports.append(name)
    if pydantic_imports:
        lines.append(f"from pydantic import {', '.join(pydantic_imports)}")
    kedi_names = sorted(imports_by_origin.get("kedi", ()))
    if kedi_names:
        lines.append(f"from kedi import {', '.join(kedi_names)}")
    return lines


def _typing_symbols_in_python_stub(stub: str | None) -> set[tuple[str, str]]:
    if not stub:
        return set()
    try:
        module = py_ast.parse(stub)
    except SyntaxError:
        return set()
    annotations: list[py_ast.expr] = []
    for node in py_ast.walk(module):
        if isinstance(node, py_ast.AnnAssign):
            annotations.append(node.annotation)
        elif isinstance(node, py_ast.arg) and node.annotation is not None:
            annotations.append(node.annotation)
        elif isinstance(node, (py_ast.FunctionDef, py_ast.AsyncFunctionDef)):
            if node.returns is not None:
                annotations.append(node.returns)
    found: set[tuple[str, str]] = set()
    for annotation in annotations:
        found.update(_typing_symbols_in_python_type(py_ast.unparse(annotation)))
    return found


def _provided_python_type_imports(codes: Iterable[str]) -> tuple[set[str], set[str]]:
    names: set[str] = set()
    star_origins: set[str] = set()
    known_origins = {"typing", "typing_extensions", "datetime", "re", "pydantic", "kedi"}
    for code in codes:
        try:
            module = py_ast.parse(code)
        except SyntaxError:
            continue
        for statement in module.body:
            if (
                not isinstance(statement, py_ast.ImportFrom)
                or statement.module not in known_origins
            ):
                continue
            for alias in statement.names:
                if alias.name == "*":
                    star_origins.add(statement.module)
                else:
                    names.add(alias.asname or alias.name)
    return names, star_origins


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
