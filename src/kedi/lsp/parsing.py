"""Programs as editor features see them: each group's statements in their enclosing scope.

A group is transparent at run time, so scope, completion, navigation and hover treat its
statements as statements of the block around it. ``Program.groups`` still lists every group.
"""

from __future__ import annotations

from typing import Any, List

from kedi.lang.ast import GroupStmt, ProcDefStmt, Program, TaskGroupStmt
from kedi.lang.parser import parse_program as _parse_program


def parse_program(source: str, **kwargs: Any) -> Program:
    program = _parse_program(source, **kwargs)
    program.toplevel_body = _inline(program.toplevel_body)
    program.ordered_toplevel = _inline(program.ordered_toplevel)
    for procedure in program.procedures:
        procedure.body = _inline(procedure.body)
    return program


def _inline(statements: List[Any]) -> List[Any]:
    inlined: List[Any] = []
    for statement in statements:
        if isinstance(statement, GroupStmt):
            inlined.extend(_inline(statement.body))
            continue
        for attribute in ("body", "else_body", "map_body"):
            nested = getattr(statement, attribute, None)
            if isinstance(nested, list):
                setattr(statement, attribute, _inline(nested))
        if isinstance(statement, ProcDefStmt) and statement.proc is not None:
            statement.proc.body = _inline(statement.proc.body)
        if isinstance(statement, TaskGroupStmt):
            for arm in statement.arms:
                if arm.body is not None:
                    arm.body = _inline(arm.body)
        inlined.append(statement)
    return inlined


__all__ = ["parse_program"]
