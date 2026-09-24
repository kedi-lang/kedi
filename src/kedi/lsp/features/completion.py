from __future__ import annotations

import re
from typing import List, Optional

from lsprotocol import types as lsp

from kedi.agent_profile import CODEMODE_SETTING_FIELDS
from kedi.lang.ast import (
    AwaitStmt,
    LoopStmt,
    OutputSegment,
    ProcedureDef,
    TaskStmt,
    UseDirectiveStmt,
    VariableInitBlockStmt,
    VariableInitStmt,
)
from kedi.lang.compiler.environment import KEDI_RUNTIME_TYPE_ORIGINS, KEDI_TYPING_TYPE_NAMES
from kedi.lang.parser import KediParseError, parse_program
from kedi.lsp.coordinates import position_inside as _position_inside
from kedi.lsp.coordinates import with_source_coordinates
from kedi.lsp.features.python_regions import _position_in_python_hover_region
from kedi.lsp.features.scope import (
    _all_procedures,
    _all_type_defs,
    _iter_control_statements,
    _segments_of,
)
from kedi.lsp.features.symbols import _merged_profile_defs
from kedi.lsp.module_surface import imported_surface_for_program

_DIRECTIVE_COMPLETIONS = (
    "adapter",
    "agent",
    "await",
    "approval",
    "artifacts",
    "auto",
    "budget",
    "case",
    "codemode",
    "data",
    "effort",
    "else",
    "export",
    "history",
    "hooks",
    "if",
    "import",
    "loop",
    "map",
    "max_agents",
    "mcp",
    "metric",
    "model",
    "optimize",
    "output",
    "package",
    "profile",
    "requires",
    "show",
    "settings",
    "skills",
    "subagent",
    "system",
    "task",
    "test_data",
    "tool",
    "use",
    "workflow",
)

_TOOL_SETTING_FIELDS = ("description", "name", "retries", "retry_on", "risk")


_BUILTIN_TYPE_COMPLETIONS = (
    "bool",
    "bytes",
    "complex",
    "dict",
    "float",
    "frozenset",
    "int",
    "list",
    "set",
    "str",
    "type",
)


@with_source_coordinates(1)
def compute_completion(
    source: str,
    source_path: Optional[str],
    pos: lsp.Position,
) -> List[lsp.CompletionItem]:
    """Return syntax and in-scope symbol completions for Kedi source."""
    if _position_in_python_hover_region(source, pos):
        return []

    line = source.splitlines()[pos.line] if pos.line < len(source.splitlines()) else ""
    before_cursor = line[: pos.character]
    if _inside_directive_block(source, pos, "history"):
        return [
            lsp.CompletionItem(
                label=name,
                kind=lsp.CompletionItemKind.Property,
                insert_text=f"{name}: ",
                detail="Native history setting",
                sort_text=f"0-{name}",
            )
            for name in (
                "enabled",
                "processor",
                "processor_condition",
                "compaction_mode",
                "compaction_threshold",
            )
        ]
    if _inside_directive_block(source, pos, "codemode"):
        return [
            lsp.CompletionItem(
                label=name,
                kind=lsp.CompletionItemKind.Property,
                insert_text=f"{name}: ",
                detail="CodeMode setting",
                sort_text=f"0-{name}",
            )
            for name in sorted(CODEMODE_SETTING_FIELDS)
        ]
    if _inside_directive_block(source, pos, "budget"):
        return [
            lsp.CompletionItem(
                label=name,
                kind=lsp.CompletionItemKind.Property,
                insert_text=f"{name}: ",
                detail="Run budget ceiling",
                sort_text=f"0-{name}",
            )
            for name in ("request_limit", "tool_attempt_limit")
        ]
    if _inside_directive_block(source, pos, "tool"):
        return [
            lsp.CompletionItem(
                label=name,
                kind=lsp.CompletionItemKind.Property,
                insert_text=f"{name}: ",
                detail="Tool metadata field",
                sort_text=f"0-{name}",
            )
            for name in _TOOL_SETTING_FIELDS
        ]
    if re.fullmatch(r"\s*>\s*task\s+\[\w+\]\s*:\s*\w*", before_cursor):
        return _native_task_completions(source, source_path, pos, child=True)
    if re.fullmatch(r"\s*>\s*await(?:\s+\[\w+\])?\s*:\s*\w*", before_cursor):
        return _native_task_completions(source, source_path, pos, child=False)
    if before_cursor.lstrip().startswith(">") and not before_cursor.lstrip().startswith("> show:"):
        in_procedure_header = _inside_direct_procedure_body(source, pos)
        return [
            lsp.CompletionItem(
                label=name,
                kind=lsp.CompletionItemKind.Keyword,
                insert_text=(
                    "task [job]: "
                    if name == "task"
                    else "await: "
                    if name == "await"
                    else f"{name}: "
                ),
                detail="Kedi directive",
                sort_text=f"0-{name}",
            )
            for name in _DIRECTIVE_COMPLETIONS
            if name != "tool" or in_procedure_header
        ]

    items: dict[str, lsp.CompletionItem] = {}

    def add(
        name: str,
        kind: lsp.CompletionItemKind,
        detail: str,
        *,
        insert_text: str | None = None,
    ) -> None:
        items.setdefault(
            name,
            lsp.CompletionItem(
                label=name,
                kind=kind,
                detail=detail,
                insert_text=insert_text,
            ),
        )

    for name in _BUILTIN_TYPE_COMPLETIONS:
        add(name, lsp.CompletionItemKind.Class, "built-in Kedi type")
    for name in sorted(KEDI_TYPING_TYPE_NAMES | KEDI_RUNTIME_TYPE_ORIGINS.keys()):
        add(name, lsp.CompletionItemKind.Class, "Kedi type")

    try:
        prog = parse_program(source, source_path=source_path)
    except KediParseError:
        return list(items.values())

    imported_surface = imported_surface_for_program(prog)
    for typedef in [*_all_type_defs(prog), *imported_surface.types.values()]:
        add(typedef.name, lsp.CompletionItemKind.Class, "Kedi type")
    for procedure in [*_all_procedures(prog), *imported_surface.procedures.values()]:
        parameters = ", ".join(procedure.params)
        add(
            procedure.name,
            lsp.CompletionItemKind.Function,
            f"Kedi procedure ({parameters})",
            insert_text=procedure.name,
        )
    for name in _merged_profile_defs(prog, imported_surface):
        add(name, lsp.CompletionItemKind.Struct, "agent profile")
    for name in imported_surface.values:
        add(name, lsp.CompletionItemKind.Variable, "imported value")

    containing_procedures = [
        procedure
        for procedure in _all_procedures(prog)
        if procedure.loc is not None and _position_inside(pos, procedure.loc)
    ]

    def procedure_span_size(procedure: ProcedureDef) -> int:
        assert procedure.loc is not None
        return procedure.loc.end_byte - procedure.loc.start_byte

    current_procedure = min(
        containing_procedures,
        key=procedure_span_size,
        default=None,
    )
    if current_procedure is not None:
        for parameter in current_procedure.params:
            add(parameter, lsp.CompletionItemKind.Variable, "procedure parameter")
        statements = current_procedure.body
    else:
        statements = prog.toplevel_body

    for statement in _iter_control_statements(statements):
        loc = getattr(statement, "loc", None)
        if loc is not None and (loc.start_line - 1, loc.start_col - 1) > (
            pos.line,
            pos.character,
        ):
            continue
        if isinstance(statement, (VariableInitStmt, VariableInitBlockStmt)):
            add(statement.name, lsp.CompletionItemKind.Variable, "Kedi value")
        if isinstance(statement, LoopStmt) and statement.binder:
            add(statement.binder, lsp.CompletionItemKind.Variable, "loop value")
        if isinstance(statement, TaskStmt):
            add(statement.binder, lsp.CompletionItemKind.Variable, "subagent task")
        if isinstance(statement, AwaitStmt) and statement.binder:
            add(statement.binder, lsp.CompletionItemKind.Variable, "subagent result")
        for segment in _segments_of(statement):
            if isinstance(segment, OutputSegment) and not isinstance(statement, TaskStmt):
                add(segment.name, lsp.CompletionItemKind.Variable, "template output")

    return sorted(items.values(), key=lambda item: item.label.lower())


def _native_task_completions(
    source: str, source_path: Optional[str], pos: lsp.Position, *, child: bool
) -> List[lsp.CompletionItem]:
    try:
        program = parse_program(source, source_path=source_path)
    except KediParseError:
        prefix = "\n".join(source.splitlines()[: pos.line]) + "\n"
        try:
            program = parse_program(prefix, source_path=source_path)
        except KediParseError:
            current_line = source.splitlines()[pos.line]
            indent = current_line[: len(current_line) - len(current_line.lstrip())]
            try:
                program = parse_program(
                    prefix + indent + "> show: ready\n", source_path=source_path
                )
            except KediParseError:
                return []

    procedures = [
        proc
        for proc in _all_procedures(program)
        if proc.loc is not None
        and proc.loc.start_line - 1 <= pos.line
        and proc.loc.end_line >= pos.line
    ]
    current_procedure = procedures[-1] if procedures else None
    if child:
        selected = [
            stmt.name
            for stmt in program.ordered_toplevel
            if isinstance(stmt, UseDirectiveStmt)
            and not stmt.tool_names
            and stmt.loc is not None
            and stmt.loc.start_line - 1 < pos.line
        ]
        if current_procedure is not None:
            selected.extend(
                stmt.name
                for stmt in _iter_control_statements(current_procedure.body)
                if isinstance(stmt, UseDirectiveStmt)
                and not stmt.tool_names
                and stmt.loc is not None
                and stmt.loc.start_line - 1 < pos.line
            )
        profiles = _merged_profile_defs(program, imported_surface_for_program(program))
        parent = profiles.get(selected[-1]) if selected else None
        names = (member.name for member in parent.subagent_members) if parent else ()
        kind = lsp.CompletionItemKind.Struct
    else:
        statements = list(program.toplevel_body)
        if current_procedure is not None:
            statements.extend(current_procedure.body)
        names = (
            stmt.binder
            for stmt in _iter_control_statements(statements)
            if isinstance(stmt, TaskStmt)
            and stmt.loc is not None
            and stmt.loc.start_line - 1 < pos.line
        )
        kind = lsp.CompletionItemKind.Variable
    return [lsp.CompletionItem(label=name, kind=kind) for name in dict.fromkeys(names)]


def _inside_directive_block(source: str, pos: lsp.Position, directive: str) -> bool:
    return _enclosing_header(source, pos) == f"> {directive}:"


def _inside_direct_procedure_body(source: str, pos: lsp.Position) -> bool:
    """Return whether the cursor is at the direct body indent of a procedure."""
    header = _enclosing_header(source, pos)
    return header is not None and header.startswith("@") and header.endswith(":")


def _enclosing_header(source: str, pos: lsp.Position) -> str | None:
    lines = source.splitlines()
    if pos.line >= len(lines):
        return None
    current = lines[pos.line][: pos.character]
    current_indent = len(current) - len(current.lstrip())
    if current_indent == 0:
        return None
    for line in reversed(lines[: pos.line]):
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        if indent >= current_indent:
            continue
        return line.strip()
    return None
