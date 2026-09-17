from __future__ import annotations

from typing import List, Optional

from lsprotocol import types as lsp

from kedi.lang.ast import (
    LoopStmt,
    OutputSegment,
    ProcedureDef,
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
    "approval",
    "artifacts",
    "auto",
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
    "settings",
    "skills",
    "subagent",
    "system",
    "test_data",
    "use",
    "workflow",
)


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
    if before_cursor.lstrip().startswith(">"):
        return [
            lsp.CompletionItem(
                label=name,
                kind=lsp.CompletionItemKind.Keyword,
                insert_text=f"{name}: ",
                detail="Kedi directive",
                sort_text=f"0-{name}",
            )
            for name in _DIRECTIVE_COMPLETIONS
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
        for segment in _segments_of(statement):
            if isinstance(segment, OutputSegment):
                add(segment.name, lsp.CompletionItemKind.Variable, "template output")

    return sorted(items.values(), key=lambda item: item.label.lower())
