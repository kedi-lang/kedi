from __future__ import annotations

import re
from typing import List, Optional

from lsprotocol import types as lsp

from kedi.agent_profile import CODEMODE_SETTING_FIELDS
from kedi.agent_profile.capabilities import CAPABILITY_DIRECTIVES
from kedi.lang.ast import (
    AwaitStmt,
    LoopStmt,
    OutputSegment,
    ProcDefStmt,
    ProcedureDef,
    PythonBlock,
    RawInvokeStmt,
    TaskStmt,
    TypeDefStmt,
    UseDirectiveStmt,
    VariableInitBlockStmt,
    VariableInitStmt,
)
from kedi.lang.compiler.environment import KEDI_RUNTIME_TYPE_ORIGINS, KEDI_TYPING_TYPE_NAMES
from kedi.lang.parser import KediParseError, parse_program
from kedi.lsp.completion_source import parse_completion_source
from kedi.lsp.coordinates import position_inside as _position_inside
from kedi.lsp.coordinates import utf16_length, utf16_to_index, with_source_coordinates
from kedi.lsp.features.directive_options import option_completions
from kedi.lsp.features.python_regions import compute_python_ranges
from kedi.lsp.features.scope import (
    _all_procedures,
    _direct_type_defs,
    _iter_control_statements,
    _nested_procedures,
    _prelude_defined_names,
    _procedure_scope_entries,
    _segments_of,
    _visible_control_statements,
)
from kedi.lsp.features.symbols import _merged_profile_defs
from kedi.lsp.module_surface import imported_surface_for_program

_DIRECTIVE_COMPLETIONS = {
    "adapter": "Select the model integration for this scope",
    "agent": "Select an agent harness or remote agent",
    "await": "Wait for a subagent task and bind its result",
    "send": "Send context to a running subagent task",
    "interrupt": "Redirect a running subagent task with updated input",
    "approval": "Control permission for tool execution",
    "artifacts": "Keep large tool results outside model context",
    "auto": "Generate a procedure implementation from its specification",
    "budget": "Limit model requests and tool attempts",
    "case": "Define an evaluation case",
    "capability": "Define composable tools, guidance, and owned tool policies",
    "codemode": "Let the agent call tools through Python code",
    "data": "Supply evaluation or optimization data",
    "defer": "Load this capability on demand; defaults to false",
    "effort": "Set the model reasoning effort",
    "else": "Run the alternative branch of a condition",
    "export": "Expose module definitions to importers",
    "history": "Configure conversation history and processing",
    "hooks": "Attach callbacks to agent lifecycle events",
    "if": "Run a block when a condition holds",
    "import": "Import a Kedi module or package",
    "loop": "Repeat a block over values or a condition",
    "map": "Process each result of the preceding iterable loop",
    "max_agents": "Limit concurrent subagent execution",
    "mcp": "Connect tools from an MCP server",
    "metric": "Define how evaluation results are scored",
    "model": "Select the model for this lexical scope",
    "optimize": "Optimize a prompt against evaluation data",
    "output": "Declare the default structured result type of a subagent",
    "package": "Declare package metadata and dependencies",
    "profile": "Define a reusable agent configuration",
    "process": "Run independent program branches concurrently",
    "requires": "Require adapter capabilities before invocation",
    "show": "Display a value without returning from the program",
    "settings": "Set model and runtime options for this scope",
    "skills": "Configure discovery of agent skill documents",
    "subagent": "Make a child agent available for delegation",
    "instructions": "Set the agent's instructions for this lexical scope",
    "task": "Start a subagent task and bind its handle",
    "task_group": "Group subagent tasks for structured execution",
    "test_data": "Supply held-out evaluation data",
    "tool": "Describe a procedure's tool metadata and retry policy",
    "use": "Apply a profile, add a capability, or expose a tool",
    "workflow": "Select direct delegation or code-based subagent orchestration",
}

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


_WORKFLOW_COMPLETIONS = {
    "mode": "Select direct delegation, blocking delegation or code-based orchestration",
    "tools": "Choose which subagent control tools the model sees (delegate mode)",
}


@with_source_coordinates(1)
def compute_completion(
    source: str,
    source_path: Optional[str],
    pos: lsp.Position,
    *,
    explicit: bool = False,
) -> List[lsp.CompletionItem]:
    """Return syntax and in-scope symbol completions for Kedi source."""
    for region in compute_python_ranges(source):
        start, end = region["range"]["start"], region["range"]["end"]
        if (
            (start["line"], start["character"])
            <= (pos.line, pos.character)
            <= (end["line"], end["character"])
        ):
            return []

    line = source.splitlines()[pos.line] if pos.line < len(source.splitlines()) else ""
    before_cursor = line[: utf16_to_index(line, pos.character)]
    # Delimiters are syntax, not completion prefixes. In particular, an empty
    # directive value must not reopen the menu and consume the next Enter.
    if before_cursor.rstrip().endswith("`") or before_cursor.lstrip().startswith("#"):
        return []
    options = option_completions(source, pos, explicit=explicit)
    if options is not None:
        return options
    if before_cursor.rstrip().endswith(":") and not re.search(r"\[[^\]]*$", before_cursor):
        return []
    field_prefix = re.fullmatch(r"\s*[A-Za-z_]*", before_cursor) is not None
    if not field_prefix and any(
        _inside_directive_block(source, pos, name)
        for name in ("history", "codemode", "budget", "tool")
    ):
        return []
    if field_prefix and _inside_directive_block(source, pos, "history"):
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
    if field_prefix and _inside_directive_block(source, pos, "codemode"):
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
    if field_prefix and _inside_directive_block(source, pos, "budget"):
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
    if field_prefix and _inside_directive_block(source, pos, "tool"):
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
    if re.fullmatch(r"\s*>\s*task\s+\[\w+\]\s*:\s*\w+", before_cursor):
        return _native_task_completions(source, source_path, pos, child=True)
    if re.fullmatch(r"\s*>\s*(?:await(?:\s+\[\w+\])?|send|interrupt)\s*:\s*\w+", before_cursor):
        return _native_task_completions(source, source_path, pos, child=False)
    directive_prefix = re.fullmatch(r"\s*(>\s*)[A-Za-z_]*", before_cursor)
    header = _enclosing_header(source, pos)
    if directive_prefix and header is not None and re.fullmatch(r">\s*tools\s*:", header):
        # A `> tools:` body lists tool names, never directives.
        return []
    if directive_prefix and header is not None and re.fullmatch(r">\s*workflow\s*:", header):
        return [
            lsp.CompletionItem(
                label=name,
                kind=lsp.CompletionItemKind.Keyword,
                insert_text=f"{name}: ",
                detail=description,
                sort_text=f"0-{name}",
            )
            for name, description in _WORKFLOW_COMPLETIONS.items()
        ]
    owner = _enclosing_configuration(source, pos)
    in_capability = owner == "capability"
    if directive_prefix:
        in_procedure_header = _inside_direct_procedure_body(source, pos)
        start = len(before_cursor) - len(before_cursor.lstrip())
        suffix = re.match(r"[A-Za-z_]*", line[len(before_cursor) :])
        end = len(before_cursor) + (suffix.end() if suffix else 0)
        existing_colon = re.match(r"\s*:\s*", line[end:])
        if existing_colon:
            end += existing_colon.end()
        edit_range = lsp.Range(
            start=lsp.Position(line=pos.line, character=utf16_length(line[:start])),
            end=lsp.Position(line=pos.line, character=utf16_length(line[:end])),
        )
        return [
            lsp.CompletionItem(
                label=name,
                kind=lsp.CompletionItemKind.Keyword,
                text_edit=lsp.TextEdit(
                    range=edit_range,
                    new_text="> task [job]: " if name == "task" else f"> {name}: ",
                ),
                detail=description,
                filter_text=f"{directive_prefix[1]}{name}",
                sort_text=f"0-{name}",
            )
            for name, description in _DIRECTIVE_COMPLETIONS.items()
            if (name != "tool" or in_procedure_header)
            and (not in_capability or name in CAPABILITY_DIRECTIVES or name in {"if", "else"})
            and (owner is None or name in _PROFILE_DIRECTIVES)
            and (name != "defer" or in_capability)
        ]
    use_target = re.fullmatch(r"\s*>\s*use\s*:\s*([A-Za-z_]\w*)", before_cursor)
    if (
        before_cursor.lstrip().startswith(">")
        and not use_target
        and not re.search(r"[<\[][^>\]]*$", before_cursor)
    ):
        return []

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
        prog = parse_completion_source(source, source_path, pos.line)
    except KediParseError:
        return list(items.values())

    # The cursor follows the typed prefix; its final character can coincide
    # with an end-exclusive scope span at EOF.
    scope_pos = lsp.Position(line=pos.line, character=max(0, pos.character - 1))
    containing_procedures = [
        procedure
        for procedure in _all_procedures(prog)
        if procedure.loc is not None and _position_inside(scope_pos, procedure.loc)
    ]

    def procedure_span_size(procedure: ProcedureDef) -> int:
        assert procedure.loc is not None
        return procedure.loc.end_byte - procedure.loc.start_byte

    current_procedure = min(
        containing_procedures,
        key=procedure_span_size,
        default=None,
    )
    parents = next(
        (
            ancestors
            for proc, ancestors in _procedure_scope_entries(prog)
            if proc is current_procedure
        ),
        (),
    )
    scopes = [*parents, *([current_procedure] if current_procedure is not None else [])]
    imported_surface = imported_surface_for_program(prog)
    types = [*prog.type_defs, *imported_surface.types.values()]
    procedures = [*prog.procedures, *imported_surface.procedures.values()]
    for scope in scopes:
        types.extend(_direct_type_defs(scope, scope_pos))
        procedures.extend(_nested_procedures(scope, scope_pos))
        for parameter in scope.params:
            add(parameter, lsp.CompletionItemKind.Variable, "procedure parameter")
    for typedef in types:
        add(typedef.name, lsp.CompletionItemKind.Class, "Kedi type")
    for procedure in procedures:
        parameters = ", ".join(procedure.params)
        add(
            procedure.name,
            lsp.CompletionItemKind.Function,
            f"Kedi procedure ({parameters})",
            insert_text=procedure.name,
        )
    for name, profile in _merged_profile_defs(prog, imported_surface).items():
        if use_target and in_capability and profile.kind != "capability":
            continue
        add(name, lsp.CompletionItemKind.Struct, f"agent {profile.kind}")
    for name in imported_surface.values:
        add(name, lsp.CompletionItemKind.Variable, "imported value")
    for name in sorted(_prelude_defined_names(prog.prelude)):
        add(name, lsp.CompletionItemKind.Variable, "Python binding")

    statements = [*prog.toplevel_body, *(stmt for scope in scopes for stmt in scope.body)]
    for statement in _visible_control_statements(statements, scope_pos):
        loc = getattr(statement, "loc", None)
        if loc is not None and (loc.start_line - 1, loc.start_col - 1) > (
            pos.line,
            pos.character,
        ):
            continue
        if isinstance(statement, (VariableInitStmt, VariableInitBlockStmt)):
            add(statement.name, lsp.CompletionItemKind.Variable, "Kedi value")
        if isinstance(statement, RawInvokeStmt):
            add(statement.name, lsp.CompletionItemKind.Variable, "model output")
        if isinstance(statement, ProcDefStmt) and statement.proc is not None:
            add(statement.proc.name, lsp.CompletionItemKind.Function, "Kedi procedure")
        if isinstance(statement, TypeDefStmt) and statement.typedef is not None:
            add(statement.typedef.name, lsp.CompletionItemKind.Class, "Kedi type")
        if isinstance(statement, PythonBlock):
            for name in sorted(_prelude_defined_names(statement.code)):
                add(name, lsp.CompletionItemKind.Variable, "Python binding")
        if isinstance(statement, LoopStmt) and statement.binder:
            add(statement.binder, lsp.CompletionItemKind.Variable, "loop value")
        if isinstance(statement, TaskStmt):
            add(statement.binder, lsp.CompletionItemKind.Variable, "subagent task")
        if isinstance(statement, AwaitStmt) and statement.binder:
            add(statement.binder, lsp.CompletionItemKind.Variable, "subagent result")
        for segment in _segments_of(statement):
            if isinstance(segment, OutputSegment) and not isinstance(statement, TaskStmt):
                add(segment.name, lsp.CompletionItemKind.Variable, "template output")

    if use_target:
        items = {
            name: item
            for name, item in items.items()
            if item.kind in (lsp.CompletionItemKind.Function, lsp.CompletionItemKind.Struct)
            or item.detail == "Python binding"
        }
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
            for stmt in _visible_control_statements(statements, pos)
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


_PROFILE_DIRECTIVES = {
    "agent",
    "adapter",
    "model",
    "effort",
    "approval",
    "hooks",
    "history",
    "skills",
    "codemode",
    "requires",
    "budget",
    "instructions",
    "mcp",
    "settings",
    "artifacts",
    "output",
    "subagent",
    "max_agents",
    "workflow",
    "use",
    "defer",
    "if",
    "else",
}


def _enclosing_configuration(source: str, pos: lsp.Position) -> str | None:
    lines = source.splitlines()
    if pos.line >= len(lines):
        return None
    indent = len(lines[pos.line]) - len(lines[pos.line].lstrip())
    for line in reversed(lines[: pos.line]):
        if not line.strip():
            continue
        previous_indent = len(line) - len(line.lstrip())
        if previous_indent >= indent:
            continue
        match = re.match(r"\s*>\s*(profile|capability)\s*:", line)
        if match:
            return match[1]
        if not re.match(r"\s*>\s*(if|else)\s*:", line):
            return None
        indent = previous_indent
    return None


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
