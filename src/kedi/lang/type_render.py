"""Helpers for rendering Kedi type annotations as Python source."""

from __future__ import annotations

from typing import Optional

from .ast import ProcedureDef, TypeApply, TypeExpr, TypePython, TypeRef, TypeUnion


def type_expr_to_python(
    expr: Optional[TypeExpr],
    annotation: Optional[str],
    *,
    default: str = "str",
) -> str:
    if isinstance(expr, TypePython):
        return expr.code or default
    if isinstance(expr, TypeRef):
        return expr.name or default
    if isinstance(expr, TypeApply):
        args = ", ".join(type_expr_to_python(arg, None) for arg in expr.args)
        return f"{expr.name}[{args}]" if args else expr.name or default
    if isinstance(expr, TypeUnion):
        left = type_expr_to_python(expr.left, None)
        right = type_expr_to_python(expr.right, None)
        return f"{left} | {right}"
    if annotation:
        return annotation
    return default


def procedure_param_type_to_python(
    proc: ProcedureDef,
    param_name: str,
    *,
    default: str = "str",
) -> str:
    return type_expr_to_python(
        (proc.param_type_exprs or {}).get(param_name),
        (proc.param_type_annotations or {}).get(param_name),
        default=default,
    )


def procedure_return_type_to_python(proc: ProcedureDef, *, default: str = "str") -> str:
    return type_expr_to_python(proc.return_type_expr, proc.return_type_annotation, default=default)
