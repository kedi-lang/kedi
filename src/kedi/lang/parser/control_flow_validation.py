"""Shared lexical return validation, without evaluating program expressions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..ast import (
    ConditionalLoopStmt,
    IfStmt,
    LoopStmt,
    ProcDefStmt,
    ProcedureDef,
    Program,
    ReturnBlockStmt,
    ReturnStmt,
    ShowStmt,
    OutputSegment,
    Span,
)
from .diagnostics import KediDiagnostic, KediParseError


def validate_control_flow(program: Program, *, interactive: bool = False) -> None:
    """Reject unreachable statements and returns outside their execution boundary."""
    diagnostics: list[KediDiagnostic] = []

    def error(statement: Any, message: str, code: str) -> None:
        diagnostics.append(
            KediDiagnostic(
                severity="error",
                message=message,
                span=statement.loc or program.loc or Span(0, 0, 1, 1, 1, 1),
                code=code,
            )
        )

    def block(statements: Sequence[Any], *, root: bool) -> bool:
        terminal: Any = None
        for statement in statements:
            # Generated companion-file procedures are compiled declarations,
            # not executable statements appended to the root file's flow.
            if (
                terminal is not None
                and isinstance(statement, ProcedureDef)
                and statement.loc is not None
                and terminal.loc is not None
                and statement.loc.source_path != terminal.loc.source_path
            ):
                block(statement.body, root=False)
                continue
            if terminal is not None:
                location = terminal.loc
                origin = f" at line {location.start_line}" if location else ""
                error(
                    statement,
                    f"Unreachable statement after return{origin}; use '> show:' for display",
                    "kedi.unreachable_statement",
                )
                break
            if isinstance(statement, (ReturnStmt, ReturnBlockStmt)):
                if interactive and root:
                    error(
                        statement,
                        "Interactive root returns are not allowed; use '> show:' instead",
                        "kedi.interactive_return",
                    )
                terminal = statement
            elif isinstance(statement, ShowStmt):
                if any(isinstance(segment, OutputSegment) for segment in statement.segments):
                    error(statement, "Show cannot declare output captures", "kedi.show_capture")
            elif isinstance(statement, ProcedureDef):
                block(statement.body, root=False)
            elif isinstance(statement, ProcDefStmt) and statement.proc is not None:
                block(statement.proc.body, root=False)
            elif isinstance(statement, IfStmt):
                body_returns = block(statement.body, root=root)
                else_returns = block(statement.else_body or [], root=root)
                if body_returns and else_returns:
                    terminal = statement
            elif isinstance(statement, (LoopStmt, ConditionalLoopStmt)):
                block(statement.body, root=root)
                if isinstance(statement, LoopStmt) and statement.map_body is not None:
                    block(statement.map_body, root=root)
        return terminal is not None

    sequence = list(program.ordered_toplevel or program.toplevel_body)
    if not program.ordered_toplevel:
        for procedure in program.procedures:
            block(procedure.body, root=False)
    if program.toplevel_return is not None and not any(
        item is program.toplevel_return for item in sequence
    ):
        sequence.append(program.toplevel_return)
    block(sequence, root=True)
    if diagnostics:
        raise KediParseError(diagnostics)
