"""LSP feature implementations for the Kedi language server.

The split keeps :mod:`kedi.lsp.server` thin — it owns the protocol
surface and dispatches to the functions in this module. Each function
takes raw source plus an LSP position / range and returns the LSP
response shape.

Features implemented here:

  - :func:`compute_diagnostics`
  - :func:`compute_semantic_tokens_full`  (B.1)
  - :func:`compute_document_symbols`      (B.2)
  - :func:`compute_completion`
  - :func:`compute_references`            (B.3)
  - :func:`compute_prepare_rename`        (B.4)
  - :func:`compute_rename`                (B.4)
  - :func:`compute_formatting`            (B.5)
  - :func:`compute_signature_help`        (B.6)
  - :func:`compute_inlay_hints`           (B.7)
  - :func:`compute_definition`            (B.8 — extended)
  - :func:`compute_hover`
  - :func:`compute_python_ranges`         (B.9 — custom request)

Plus the constants ``SEMANTIC_TOKEN_TYPES`` / ``SEMANTIC_TOKEN_MODIFIERS``
the server publishes as a semantic-tokens legend.
"""

from __future__ import annotations

import ast as _py_ast
import builtins as _builtins
import importlib.resources
import re
import typing as _typing
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import List, Optional, Tuple, Union, cast

from lsprotocol import types as lsp
from tree_sitter import Language as _TSLanguage
from tree_sitter import Node, Tree
from tree_sitter import Parser as _TSParser
from tree_sitter import Query as _TSQuery
from tree_sitter import QueryCursor as _TSQueryCursor

try:
    import tree_sitter_python as _ts_python_mod

    _PY_LANGUAGE = _TSLanguage(_ts_python_mod.language())
    _PY_PARSER = _TSParser(_PY_LANGUAGE)
    _PY_HIGHLIGHTS_SRC = (
        importlib.resources.files("tree_sitter_python") / "queries" / "highlights.scm"
    ).read_text()
    _PY_HIGHLIGHTS_QUERY = _TSQuery(_PY_LANGUAGE, _PY_HIGHLIGHTS_SRC)
except Exception:  # pragma: no cover — degraded mode if the wheel is missing
    _PY_PARSER = None
    _PY_LANGUAGE = None
    _PY_HIGHLIGHTS_QUERY = None

from ..agent_adapter.meta import (
    AdapterCapabilities,
    AdapterCapability,
    AgentAdapterKind,
    HookCapabilityEvent,
)
from ..agent_adapter.registry import (
    adapter_capabilities,
    adapter_factory,
    known_adapter_shortnames,
)
from ..agent_profile.settings import (
    ACP_AGENT_SETTING_KEYS,
    CLAUDE_AGENT_SETTING_KEYS,
    CODEX_AGENT_SETTING_KEYS,
    DSPY_LM_SETTING_KEYS,
    LANGCHAIN_MODEL_SETTING_KEYS,
    MODEL_SETTING_KEYS,
    PYDANTIC_MODEL_SETTING_KEYS,
)
from ..lang.ast import (
    AdapterDirectiveStmt,
    AgentDirectiveStmt,
    ApprovalDirectiveStmt,
    ArtifactsDirectiveStmt,
    ArtifactsField,
    AssignmentBlockStmt,
    AssignmentStmt,
    AutoSpec,
    BacktickLineStmt,
    CallArg,
    CallSegment,
    CodeModeDirectiveStmt,
    ConditionalLoopStmt,
    EffortDirectiveStmt,
    EvalDataset,
    EvalMetric,
    EvalSuite,
    HistoryDirectiveStmt,
    HooksDirectiveStmt,
    IfStmt,
    InputSegment,
    LiteralArg,  # noqa: F401
    LoopStmt,
    McpDirectiveStmt,
    McpField,
    ModelDirectiveStmt,
    OptimizeSpan,
    OutputSegment,
    PackageMetadata,
    ProcDefStmt,
    ProcedureDef,
    ProfileAdapterEntry,
    ProfileAgentEntry,
    ProfileApprovalEntry,
    ProfileArtifactsEntry,
    ProfileCodeModeEntry,
    ProfileDefStmt,
    ProfileEffortEntry,
    ProfileHistoryEntry,
    ProfileHooksEntry,
    ProfileMaxAgentsEntry,
    ProfileMcpEntry,
    ProfileModelEntry,
    ProfileOutputEntry,
    ProfileSettingsEntry,
    ProfileSkillsEntry,
    ProfileSystemEntry,
    ProfileWorkflowEntry,
    Program,
    ProgramWithValidation,
    PythonBlock,
    PythonExprSegment,
    RawInvokeStmt,
    ReturnBlockStmt,
    ReturnStmt,
    SettingsDirectiveStmt,
    SettingsField,
    SkillsDirectiveStmt,
    Span,
    SystemDirectiveStmt,
    TemplateBlock,
    TemplateLine,
    TestCase,
    TestSuite,
    TextSegment,
    TypeApply,
    TypeDef,
    TypeDefStmt,
    TypeExpr,
    TypeField,
    TypeLiteral,
    TypePython,
    TypeRef,
    TypeUnion,
    UseDirectiveStmt,
    VariableInitBlockStmt,
    VariableInitStmt,
)
from ..lang.call_arg_access import iter_call_arg_nested
from ..lang.compiler.environment import (
    CLI_ARGS_IDENTIFIER,
    KEDI_RUNTIME_TYPE_JSON_SCHEMA_FORMATS,
    KEDI_RUNTIME_TYPE_ORIGINS,
    KEDI_TYPING_TYPE_NAMES,
)
from ..lang.compiler.type_system import UNSUPPORTED_KEDI_TYPE_NAMES
from ..lang.parser import KediDiagnostic, KediParseError, parse_program
from ..lang.parser.parser import get_language as _get_kedi_language
from ..lang.parser.parser import parse as _ts_parse
from ..module_registry import describe_module_search_paths, module_search_paths
from .ast_helpers import binding_name_span
from .ast_spans import name_token_span
from .coordinates import (
    byte_column_to_utf16,
    current_source,
    tree_sitter_point_to_position,
    utf16_length,
    with_source_coordinates,
)
from .coordinates import (
    position_inside as _position_inside,
)
from .coordinates import (
    span_to_range as _span_to_range,
)
from .module_surface import (
    ImportedSurface,
    ImportedValueInfo,
    import_resolution_error,
    imported_surface_for_program,
    load_imported_surface,
    program_path,
    resolved_import_path,
    top_level_value_infos,
)

# Kedi highlights + injection queries, loaded from the tree-sitter-kedi
# wheel's bundled ``queries/`` directory. These are the single source of
# truth for what the editor highlights — features.py just maps capture
# names to LSP token types.
try:
    _KEDI_LANGUAGE = _get_kedi_language()
    _KEDI_HIGHLIGHTS_SRC = (
        importlib.resources.files("tree_sitter_kedi") / "queries" / "highlights.scm"
    ).read_text()
    _KEDI_INJECTIONS_SRC = (
        importlib.resources.files("tree_sitter_kedi") / "queries" / "injections.scm"
    ).read_text()
    _KEDI_HIGHLIGHTS_QUERY = _TSQuery(_KEDI_LANGUAGE, _KEDI_HIGHLIGHTS_SRC)
    _KEDI_INJECTIONS_QUERY = _TSQuery(_KEDI_LANGUAGE, _KEDI_INJECTIONS_SRC)
except Exception:  # pragma: no cover
    _KEDI_HIGHLIGHTS_QUERY = None
    _KEDI_INJECTIONS_QUERY = None

try:
    import typing_extensions as _typing_extensions
except Exception:  # pragma: no cover - optional runtime dependency in degraded envs
    _typing_extensions: ModuleType | None = None


def _build_lsp_type_registry() -> dict[str, str]:
    registry: dict[str, str] = {}
    for name in dir(_builtins):
        if name.startswith("_") or name in UNSUPPORTED_KEDI_TYPE_NAMES:
            continue
        value = vars(_builtins)[name]
        if isinstance(value, type) and not name.startswith("_"):
            registry[name] = "builtins"
    for module, origin in (
        (_typing, "typing"),
        (_typing_extensions, "typing_extensions"),
    ):
        if module is None:
            continue
        for name in KEDI_TYPING_TYPE_NAMES:
            if hasattr(module, name):
                registry.setdefault(name, origin)
    for name, origin in KEDI_RUNTIME_TYPE_ORIGINS.items():
        registry[name] = origin
    return registry


_LSP_TYPE_REGISTRY = _build_lsp_type_registry()


def _known_lsp_type_origin(name: str) -> Optional[str]:
    return _LSP_TYPE_REGISTRY.get(name)


# =====================================================================
# Position / range helpers
# =====================================================================


def _segments_of(stmt) -> list:
    """All segment children of a statement (``segments`` or ``expr``)."""
    if isinstance(stmt, TemplateBlock):
        segs: list = []
        for line in stmt.lines:
            segs.extend(line.segments)
        return segs
    if isinstance(stmt, TemplateLine):
        return list(stmt.segments)
    if isinstance(stmt, RawInvokeStmt):
        return list(stmt.segments)
    if isinstance(stmt, ReturnStmt):
        return list(stmt.segments)
    if isinstance(stmt, (VariableInitStmt, AssignmentStmt)):
        return list(stmt.expr)
    if isinstance(stmt, (IfStmt, ConditionalLoopStmt)):
        return list(stmt.condition_segments or [])
    return []


def _control_body_statements(stmt) -> list:
    if isinstance(stmt, IfStmt):
        return [*stmt.body, *(stmt.else_body or [])]
    if isinstance(stmt, LoopStmt):
        return [*stmt.body, *(stmt.map_body or [])]
    if isinstance(stmt, ConditionalLoopStmt):
        return list(stmt.body)
    return []


def _iter_control_statements(statements):
    for stmt in statements:
        yield stmt
        yield from _iter_control_statements(_control_body_statements(stmt))


# =====================================================================
# Diagnostics
# =====================================================================


@with_source_coordinates(1)
def compute_diagnostics(source: str, source_path: Optional[str]) -> List[lsp.Diagnostic]:
    """Parse ``source`` and surface every ``KediDiagnostic`` as LSP."""
    try:
        prog = parse_program(source, source_path=source_path)
        out = _missing_import_diagnostics(prog)
        out.extend(_subagent_profile_diagnostics(prog))
        out.extend(_adapter_capability_diagnostics(prog))
        out.extend(_annotated_type_diagnostics(prog))
        out.extend(_assignment_diagnostics(prog))
        for warning in prog.parser_warnings:
            out.append(_diag_to_lsp(warning))
        return out
    except KediParseError as exc:
        return [_diag_to_lsp(d) for d in exc.diagnostics]


def _diag_to_lsp(d: KediDiagnostic) -> lsp.Diagnostic:
    severity = (
        lsp.DiagnosticSeverity.Error if d.severity == "error" else lsp.DiagnosticSeverity.Warning
    )
    return lsp.Diagnostic(
        range=_span_to_range(d.span),
        severity=severity,
        code=d.code,
        source="kedi",
        message=d.message,
    )


def _annotated_type_diagnostics(
    prog: Union[Program, ProgramWithValidation],
) -> List[lsp.Diagnostic]:
    out: List[lsp.Diagnostic] = []
    for expr in _iter_type_exprs(prog):
        _collect_annotated_type_diagnostics(expr, out)
    return out


def _assignment_diagnostics(
    prog: Program | ProgramWithValidation,
) -> List[lsp.Diagnostic]:
    imported_values = imported_surface_for_program(prog).values
    out: List[lsp.Diagnostic] = []

    def visit(
        statements: Sequence[object],
        proc: ProcedureDef | None,
        ancestors: tuple[ProcedureDef, ...],
    ) -> None:
        for stmt in _iter_control_statements(statements):
            if not isinstance(stmt, (AssignmentStmt, AssignmentBlockStmt)):
                continue
            target_span = binding_name_span(stmt)
            if target_span is None:
                continue
            if stmt.name == CLI_ARGS_IDENTIFIER:
                out.append(
                    _error(
                        target_span,
                        "K-ASSIGN-RESERVED",
                        f"Reserved CLI argument binding '{CLI_ARGS_IDENTIFIER}' cannot be assigned.",
                    )
                )
                continue
            if stmt.name in imported_values:
                continue
            if (
                _resolve_value_declaration(
                    prog,
                    proc,
                    ancestors,
                    stmt.name,
                    target_span,
                )
                is None
            ):
                out.append(
                    _error(
                        target_span,
                        "K-ASSIGN-UNKNOWN",
                        f"Cannot assign unknown Kedi binding '{stmt.name}'.",
                    )
                )

    visit(prog.toplevel_body, None, ())
    for proc, ancestors in _procedure_scope_entries(prog):
        visit(proc.body, proc, ancestors)
    return out


def _iter_type_exprs(prog: Union[Program, ProgramWithValidation]):
    for typedef in prog.type_defs:
        for field in typedef.fields:
            yield field.type_expr
    for profile in prog.profile_defs.values():
        for member in profile.output_members:
            yield member.type_expr
    for proc in _all_procedures(prog):
        yield proc.return_type_expr
        for expr in (proc.param_type_exprs or {}).values():
            yield expr
        for stmt in proc.body:
            yield from _iter_stmt_type_exprs(stmt)
    for stmt in prog.toplevel_body:
        yield from _iter_stmt_type_exprs(stmt)
    if prog.toplevel_return is not None:
        yield from _iter_stmt_type_exprs(prog.toplevel_return)


def _iter_stmt_type_exprs(stmt):
    if isinstance(stmt, (VariableInitStmt, VariableInitBlockStmt)):
        yield stmt.type_expr
    for segment in _segments_of(stmt):
        if isinstance(segment, OutputSegment):
            yield segment.type_expr
    for child in _control_body_statements(stmt):
        yield from _iter_stmt_type_exprs(child)


def _collect_annotated_type_diagnostics(
    expr: Optional[TypeExpr],
    out: List[lsp.Diagnostic],
) -> None:
    if expr is None:
        return
    if isinstance(expr, (TypeRef, TypeApply)):
        if expr.name in UNSUPPORTED_KEDI_TYPE_NAMES and expr.loc is not None:
            out.append(
                _error(
                    expr.loc,
                    "K-TYPE-UNSUPPORTED",
                    f"Unsupported Kedi type annotation '{expr.name}'.",
                )
            )
    if isinstance(expr, TypeApply):
        if expr.name == "Annotated" and expr.loc is not None:
            if len(expr.args) == 1:
                out.append(
                    _warning(
                        expr.loc,
                        "K-ANNOTATED-DESCRIPTION",
                        "Annotated[type] is accepted as the runtime type, but Kedi "
                        "Annotated metadata should include a string description: "
                        'Annotated[type, "description"].',
                    )
                )
            elif len(expr.args) > 2:
                out.append(
                    _warning(
                        expr.loc,
                        "K-ANNOTATED-EXTRA",
                        'Kedi uses only Annotated[type, "description"]; extra Annotated '
                        "metadata is ignored.",
                    )
                )
            elif len(expr.args) == 2 and not _is_string_type_literal(expr.args[1]):
                out.append(
                    _warning(
                        expr.loc,
                        "K-ANNOTATED-DESCRIPTION",
                        'Kedi Annotated metadata expects Annotated[type, "description"]; '
                        "the second argument should be a string literal.",
                    )
                )
        for arg in expr.args:
            _collect_annotated_type_diagnostics(arg, out)
    elif isinstance(expr, TypeUnion):
        _collect_annotated_type_diagnostics(expr.left, out)
        _collect_annotated_type_diagnostics(expr.right, out)


def _is_string_type_literal(expr: TypeExpr) -> bool:
    return isinstance(expr, TypeLiteral) and isinstance(expr.value, str)


def _warning(span: Span, code: str, message: str) -> lsp.Diagnostic:
    return lsp.Diagnostic(
        range=_span_to_range(span),
        severity=lsp.DiagnosticSeverity.Warning,
        code=code,
        source="kedi",
        message=message,
    )


def _error(span: Span, code: str, message: str) -> lsp.Diagnostic:
    return lsp.Diagnostic(
        range=_span_to_range(span),
        severity=lsp.DiagnosticSeverity.Error,
        code=code,
        source="kedi",
        message=message,
    )


# =====================================================================
# Symbol resolution at a cursor position
# =====================================================================


@dataclass
class SymbolRef:
    """The symbol under the cursor, used by references / rename / hover."""

    kind: str
    name: str
    declaration_span: Optional[Span]
    scope_proc: Optional[ProcedureDef]
    # The span of the identifier token under the cursor (for prepareRename).
    cursor_span: Span


@dataclass(frozen=True)
class _ValueDeclaration:
    kind: str
    name: str
    declaration_span: Span
    scope_proc: Optional[ProcedureDef]


def _merged_profile_defs(
    prog: Program | ProgramWithValidation,
    imported_surface: ImportedSurface,
) -> dict[str, ProfileDefStmt]:
    merged = dict(prog.profile_defs)
    merged.update(imported_surface.profiles)
    return merged


def _validation_suites(
    prog: Program | ProgramWithValidation,
) -> tuple[TestSuite | EvalSuite, ...]:
    if not isinstance(prog, ProgramWithValidation):
        return ()
    return (*prog.tests, *prog.evals)


def _output_name_span(seg: OutputSegment) -> Optional[Span]:
    if seg.loc is None:
        return None
    return name_token_span(seg.loc, seg.name, prefix_len=1)


def _input_name_span(seg: InputSegment) -> Optional[Span]:
    if seg.loc is None:
        return None
    return name_token_span(seg.loc, seg.name, prefix_len=1)


def _program_line_text(
    prog: Program | ProgramWithValidation,
    line_number: int,
    source_path: Optional[str] = None,
) -> Optional[str]:
    if prog.source_text:
        lines = prog.source_text.splitlines()
        if 1 <= line_number <= len(lines):
            return lines[line_number - 1]
    source_map = prog.source_map
    if source_map is not None:
        path = source_path or (prog.loc.source_path if prog.loc is not None else None)
        if path:
            return source_map.get_line(path, line_number)
    return None


def _line_start_byte(
    prog: Program | ProgramWithValidation,
    line_number: int,
) -> Optional[int]:
    if not prog.source_text:
        return None
    offset = 0
    for index, line in enumerate(prog.source_text.splitlines(keepends=True), start=1):
        if index == line_number:
            return offset
        offset += len(line.encode("utf-8"))
    return None


def _line_start_byte_for_program(
    prog: Program | ProgramWithValidation,
    line_number: int,
) -> Optional[int]:
    """Return the UTF-8 byte offset of a source line in ``prog``."""
    return _line_start_byte(prog, line_number)


def _name_token_span_on_line(
    prog: Program | ProgramWithValidation,
    *,
    line_number: int,
    source_path: Optional[str],
    name: str,
    search_start: int = 0,
) -> Optional[Span]:
    line = _program_line_text(prog, line_number, source_path)
    line_start = _line_start_byte(prog, line_number)
    if line is None or line_start is None:
        return None
    for candidate in (f"`{name}`", name):
        name_index = line.find(candidate, search_start)
        if name_index < 0:
            continue
        start_col = name_index + 1
        start_byte = line_start + len(line[:name_index].encode("utf-8"))
        return Span(
            start_byte=start_byte,
            end_byte=start_byte + len(candidate.encode("utf-8")),
            start_line=line_number,
            start_col=start_col,
            end_line=line_number,
            end_col=start_col + len(candidate),
            source_path=source_path,
        )
    return None


def _use_tool_entry_spans(
    prog: Program | ProgramWithValidation,
    stmt: UseDirectiveStmt,
) -> list[tuple[str, Span]]:
    if stmt.loc is None or not stmt.tool_names:
        return []
    spans: list[tuple[str, Span]] = []
    seen: set[tuple[str, int]] = set()
    for line_number in range(stmt.loc.start_line, stmt.loc.end_line + 1):
        for tool_name in stmt.tool_names:
            name_span = _name_token_span_on_line(
                prog,
                line_number=line_number,
                source_path=stmt.loc.source_path,
                name=tool_name,
            )
            if name_span is None:
                continue
            key = (tool_name, name_span.start_line)
            if key in seen:
                continue
            seen.add(key)
            spans.append((tool_name, name_span))
    return spans


def _profile_use_entry_spans(
    prog: Program | ProgramWithValidation,
    profile: ProfileDefStmt,
) -> list[tuple[str, Span]]:
    """Return target spans from a profile's ``> use:`` members.

    Single-line targets may resolve to either a tool or another profile.
    Multiline targets are explicit tools. Reconstruct declaration-like spans
    only from directive lines and their indented entries so profile prose
    cannot become a false reference.
    """
    target_names = (*profile.use_targets, *profile.tool_names)
    if profile.loc is None or not target_names:
        return []

    spans: list[tuple[str, Span]] = []
    seen: set[tuple[str, int, int]] = set()
    in_use_body = False
    for line_number in range(profile.loc.start_line, profile.loc.end_line + 1):
        line = _program_line_text(prog, line_number, profile.loc.source_path)
        if line is None:
            continue
        stripped = line.strip()
        if stripped.startswith("> use:"):
            in_use_body = True
            search_start = line.find(":") + 1
        elif stripped.startswith(">"):
            in_use_body = False
            continue
        elif not in_use_body or not stripped:
            continue
        else:
            search_start = len(line) - len(line.lstrip())

        for target_name in target_names:
            name_span = _name_token_span_on_line(
                prog,
                line_number=line_number,
                source_path=profile.loc.source_path,
                name=target_name,
                search_start=search_start,
            )
            if name_span is None:
                continue
            key = (target_name, name_span.start_line, name_span.start_col)
            if key not in seen:
                seen.add(key)
                spans.append((target_name, name_span))
    return spans


def _use_single_target_span(
    prog: Program | ProgramWithValidation,
    stmt: UseDirectiveStmt,
) -> Optional[Span]:
    if stmt.loc is None or not stmt.name:
        return None
    line = _program_line_text(prog, stmt.loc.start_line, stmt.loc.source_path)
    if line is None:
        return None
    colon_index = line.find(":")
    search_start = colon_index + 1 if colon_index >= 0 else 0
    return _name_token_span_on_line(
        prog,
        line_number=stmt.loc.start_line,
        source_path=stmt.loc.source_path,
        name=stmt.name,
        search_start=search_start,
    )


def _validation_procedure_span(
    prog: Program | ProgramWithValidation,
    suite: TestSuite | EvalSuite,
) -> Optional[Span]:
    """Return the procedure-name token in a validation-suite header."""
    if suite.loc is None:
        return None
    line = _program_line_text(prog, suite.loc.start_line, suite.loc.source_path)
    if line is None:
        return None
    colon_index = line.find(":")
    search_start = colon_index + 1 if colon_index >= 0 else 0
    return _name_token_span_on_line(
        prog,
        line_number=suite.loc.start_line,
        source_path=suite.loc.source_path,
        name=suite.procedure_name,
        search_start=search_start,
    )


def _directive_name_span(
    prog: Program | ProgramWithValidation,
    loc: Span,
    name: str,
) -> Optional[Span]:
    if loc is None:
        return None
    line = _program_line_text(prog, loc.start_line, loc.source_path)
    if line is None:
        return None
    colon_index = line.find(":")
    search_start = colon_index + 1 if colon_index >= 0 else 0
    name_index = line.find(name, search_start)
    if name_index < 0:
        return None
    prefix = line[:name_index]
    start_col = name_index + 1
    line_start = _line_start_byte_for_program(prog, loc.start_line)
    if line_start is None:
        return None
    start_byte = line_start + len(prefix.encode("utf-8"))
    return Span(
        start_byte=start_byte,
        end_byte=start_byte + len(name.encode("utf-8")),
        start_line=loc.start_line,
        start_col=start_col,
        end_line=loc.start_line,
        end_col=start_col + len(name),
        source_path=loc.source_path,
    )


def _import_name_span(
    prog: Program | ProgramWithValidation,
    module_import,
) -> Optional[Span]:
    if module_import.loc is None:
        return None
    line = _program_line_text(prog, module_import.loc.start_line, module_import.loc.source_path)
    if line is None:
        return None
    colon_index = line.find(":")
    search_start = colon_index + 1 if colon_index >= 0 else 0
    name_index = line.find(module_import.module, search_start)
    if name_index < 0:
        return None
    prefix = line[:name_index]
    start_col = name_index + 1
    line_start = _line_start_byte_for_program(prog, module_import.loc.start_line)
    if line_start is None:
        return None
    start_byte = line_start + len(prefix.encode("utf-8"))
    return Span(
        start_byte=start_byte,
        end_byte=start_byte + len(module_import.module.encode("utf-8")),
        start_line=module_import.loc.start_line,
        start_col=start_col,
        end_line=module_import.loc.start_line,
        end_col=start_col + len(module_import.module),
        source_path=module_import.loc.source_path,
    )


def _exported_name_at_position(
    prog: Program | ProgramWithValidation,
    pos: lsp.Position,
) -> Optional[tuple[str, Span]]:
    for export in prog.exports:
        if export.loc is None or not _position_inside(pos, export.loc):
            continue
        if pos.line + 1 <= export.loc.start_line:
            continue
        line = _program_line_text(prog, pos.line + 1, export.loc.source_path)
        if line is None:
            continue
        export_name = line.split("#", 1)[0].strip()
        if not export_name or export_name == "*" or not _is_identifier(export_name):
            continue
        name_index = line.find(export_name)
        if name_index < 0:
            continue
        start_col = name_index + 1
        line_start = _line_start_byte_for_program(prog, pos.line + 1)
        if line_start is None:
            continue
        start_byte = line_start + len(line[:name_index].encode("utf-8"))
        span = Span(
            start_byte=start_byte,
            end_byte=start_byte + len(export_name.encode("utf-8")),
            start_line=pos.line + 1,
            start_col=start_col,
            end_line=pos.line + 1,
            end_col=start_col + len(export_name),
            source_path=export.loc.source_path,
        )
        if _position_inside(pos, span):
            return export_name, span
    return None


def _missing_import_diagnostics(
    prog: Program | ProgramWithValidation,
) -> List[lsp.Diagnostic]:
    diagnostics: list[lsp.Diagnostic] = []
    for module_import in prog.imports:
        resolution_error = import_resolution_error(prog, module_import.module)
        if resolution_error is not None:
            span = _import_name_span(prog, module_import) or module_import.loc
            if span is not None:
                diagnostics.append(
                    lsp.Diagnostic(
                        range=_span_to_range(span),
                        severity=lsp.DiagnosticSeverity.Error,
                        code="K-IMPORT-CORRUPT",
                        source="kedi",
                        message=(
                            f"installed package for import {module_import.module!r} is invalid: "
                            f"{resolution_error}"
                        ),
                    )
                )
            continue
        module_path = resolved_import_path(prog, module_import.module)
        if module_path is not None:
            imported = load_imported_surface(module_path, set(), module_import.module)
            exported_names = imported.exported_names()
            for name in dict.fromkeys(module_import.names):
                if name in exported_names:
                    continue
                span = module_import.name_locs[name]
                diagnostics.append(
                    lsp.Diagnostic(
                        range=_span_to_range(span),
                        severity=lsp.DiagnosticSeverity.Error,
                        code="K-IMPORT-NAME",
                        source="kedi",
                        message=f"module '{module_import.module}' does not export '{name}'",
                    )
                )
            continue
        span = _import_name_span(prog, module_import) or module_import.loc
        if span is None:
            continue
        paths = module_search_paths(module_import.module, source_path=program_path(prog))
        diagnostics.append(
            lsp.Diagnostic(
                range=_span_to_range(span),
                severity=lsp.DiagnosticSeverity.Error,
                code="K-IMPORT-NOT-FOUND",
                source="kedi",
                message=(
                    f"imported module '{module_import.module}' not found: "
                    f"{describe_module_search_paths(paths)}"
                ),
            )
        )
    return diagnostics


def _subagent_profile_diagnostics(
    prog: Program | ProgramWithValidation,
) -> list[lsp.Diagnostic]:
    profile_defs = _merged_profile_defs(prog, imported_surface_for_program(prog))
    diagnostics: list[lsp.Diagnostic] = []
    graph = {
        name: tuple(entry.name for entry in profile.subagent_members)
        for name, profile in prog.profile_defs.items()
    }

    for profile in prog.profile_defs.values():
        for entry in profile.subagent_members:
            if entry.name in profile_defs or entry.loc is None:
                continue
            diagnostics.append(
                lsp.Diagnostic(
                    range=_span_to_range(entry.loc),
                    severity=lsp.DiagnosticSeverity.Error,
                    code="K-SUBAGENT-UNKNOWN",
                    source="kedi",
                    message=(
                        f"Unknown subagent profile '{entry.name}' referenced by "
                        f"profile '{profile.name}'"
                    ),
                )
            )

    visited: set[str] = set()
    active: list[str] = []
    active_set: set[str] = set()

    def visit(profile_name: str) -> None:
        if profile_name in visited:
            return
        if profile_name in active_set:
            cycle_start = active.index(profile_name)
            cycle = (*active[cycle_start:], profile_name)
            parent = prog.profile_defs[cycle[0]]
            entry = next(item for item in parent.subagent_members if item.name == cycle[1])
            assert entry.loc is not None
            diagnostics.append(
                lsp.Diagnostic(
                    range=_span_to_range(entry.loc),
                    severity=lsp.DiagnosticSeverity.Error,
                    code="K-SUBAGENT-CYCLE",
                    source="kedi",
                    message=("Cyclic subagent profile reference: " + " -> ".join(cycle)),
                )
            )
            return
        active.append(profile_name)
        active_set.add(profile_name)
        for child_name in graph.get(profile_name, ()):
            if child_name in graph:
                visit(child_name)
        active.pop()
        active_set.remove(profile_name)
        visited.add(profile_name)

    for profile_name in graph:
        visit(profile_name)
    return diagnostics


@dataclass(frozen=True)
class _AdapterDiagnosticState:
    shortname: str | None = None
    kind: AgentAdapterKind | None = None
    approval_loc: Span | None = None
    approval_plain_value: str | None = None
    hooks_loc: Span | None = None
    hooks_enabled: bool | None = None
    hook_events: frozenset[HookCapabilityEvent] = frozenset()
    cleared_hook_events: frozenset[HookCapabilityEvent] = frozenset()
    history_loc: Span | None = None
    history_enabled: bool | None = None
    codemode_loc: Span | None = None
    codemode_enabled: bool | None = None


def _adapter_capability_diagnostics(
    prog: Program | ProgramWithValidation,
) -> list[lsp.Diagnostic]:
    imported_surface = imported_surface_for_program(prog)
    profile_defs = _merged_profile_defs(prog, imported_surface)
    proc_names = {proc.name for proc in _all_procedures(prog)}
    proc_names.update(imported_surface.procedures)
    profile_states = {
        name: _profile_adapter_state(profile) for name, profile in profile_defs.items()
    }
    type_defs = {typedef.name: typedef for typedef in prog.type_defs}
    type_defs.update(imported_surface.types)

    diagnostics: list[lsp.Diagnostic] = []
    warned_approval_spans: set[tuple[str | None, int, int]] = set()
    for profile in prog.profile_defs.values():
        diagnostics.extend(
            _profile_capability_diagnostics(
                prog,
                profile,
                _profile_adapter_state(profile),
                warned_approval_spans,
            )
        )

    diagnostics.extend(
        _scan_capability_scope(
            prog,
            prog.ordered_toplevel or prog.toplevel_body,
            _AdapterDiagnosticState(),
            profile_defs,
            profile_states,
            proc_names,
            type_defs,
            warned_approval_spans,
        )
    )
    return diagnostics


def _profile_adapter_state(profile: ProfileDefStmt) -> _AdapterDiagnosticState:
    members: list[
        tuple[
            int,
            AgentAdapterKind | None,
            str | None,
            Span | None,
        ]
    ] = []
    for member in profile.agent_members:
        if member.loc is not None:
            members.append((member.loc.start_byte, "agent-harness", member.plain_value, None))
    for member in profile.adapter_members:
        if member.loc is not None:
            members.append((member.loc.start_byte, "agent-framework", member.plain_value, None))
    for member in profile.approval_members:
        if member.loc is not None:
            members.append((member.loc.start_byte, None, member.plain_value, member.loc))

    state = _AdapterDiagnosticState()
    for _, expected_kind, plain_value, approval_loc in sorted(members):
        if approval_loc is not None:
            state = replace(
                state,
                approval_loc=approval_loc,
                approval_plain_value=plain_value,
            )
            continue
        assert expected_kind is not None
        shortname = plain_value
        selected = _adapter_state_from_plain_value(shortname, expected_kind)
        if selected is not None:
            state = _merge_adapter_state(state, selected)
    hooks_members = sorted(
        (member for member in profile.hooks_members if member.loc is not None),
        key=lambda member: member.loc.start_byte if member.loc is not None else -1,
    )
    for member in hooks_members:
        state = _with_hooks_state(state, member)
    history_members = [member for member in profile.history_members if member.loc is not None]
    if history_members:
        latest = max(
            history_members,
            key=lambda member: member.loc.start_byte if member.loc is not None else -1,
        )
        state = replace(
            state,
            history_loc=latest.loc,
            history_enabled=latest.enabled,
        )
    codemode_members = [member for member in profile.codemode_members if member.loc is not None]
    if codemode_members:
        latest = max(
            codemode_members,
            key=lambda member: member.loc.start_byte if member.loc is not None else -1,
        )
        state = replace(
            state,
            codemode_loc=latest.loc,
            codemode_enabled=latest.enabled,
        )
    return state


def _adapter_state_from_plain_value(
    shortname: str | None,
    expected_kind: AgentAdapterKind,
) -> _AdapterDiagnosticState | None:
    if shortname is None:
        return None
    try:
        factory = adapter_factory(shortname)
    except Exception:  # pragma: no cover - parser/compiler surfaces unknown adapters elsewhere
        return None
    if factory.kind != expected_kind:
        return None
    return _AdapterDiagnosticState(shortname=shortname, kind=factory.kind)


def _merge_adapter_state(
    outer: _AdapterDiagnosticState,
    inner: _AdapterDiagnosticState,
) -> _AdapterDiagnosticState:
    selected = inner if inner.shortname is not None else outer
    approval_owner = inner if inner.approval_loc is not None else outer
    hooks_owner = inner if inner.hooks_enabled is not None else outer
    hook_events = outer.hook_events
    cleared_hook_events = outer.cleared_hook_events
    if inner.hooks_enabled is not None:
        hook_events = (hook_events - inner.cleared_hook_events) | inner.hook_events
        cleared_hook_events = (cleared_hook_events | inner.cleared_hook_events) - inner.hook_events
    history_owner = inner if inner.history_enabled is not None else outer
    codemode_owner = inner if inner.codemode_enabled is not None else outer
    return _AdapterDiagnosticState(
        shortname=selected.shortname,
        kind=selected.kind,
        approval_loc=approval_owner.approval_loc,
        approval_plain_value=approval_owner.approval_plain_value,
        hooks_loc=hooks_owner.hooks_loc,
        hooks_enabled=hooks_owner.hooks_enabled,
        hook_events=hook_events,
        cleared_hook_events=cleared_hook_events,
        history_loc=history_owner.history_loc,
        history_enabled=history_owner.history_enabled,
        codemode_loc=codemode_owner.codemode_loc,
        codemode_enabled=codemode_owner.codemode_enabled,
    )


def _adapter_supports(
    state: _AdapterDiagnosticState,
    feature: AdapterCapability,
) -> bool | None:
    if state.shortname is None:
        return None
    try:
        capabilities = adapter_capabilities(state.shortname)
    except Exception:  # pragma: no cover - keep LSP diagnostics resilient
        capabilities = AdapterCapabilities()
    return capabilities.supports(feature)


def _hook_event_changes(
    fields: Sequence[SettingsField],
) -> tuple[frozenset[HookCapabilityEvent], frozenset[HookCapabilityEvent]]:
    enabled: set[HookCapabilityEvent] = set()
    cleared: set[HookCapabilityEvent] = set()
    for field in fields:
        if field.name not in {
            "user_prompt_submit",
            "pre_tool_use",
            "post_tool_use",
            "post_tool_use_failure",
        }:
            continue
        event = cast(HookCapabilityEvent, field.name)
        if (field.backtick_code or "").strip() == "[]":
            enabled.discard(event)
            cleared.add(event)
        else:
            enabled.add(event)
            cleared.discard(event)
    return frozenset(enabled), frozenset(cleared)


def _with_hooks_state(
    state: _AdapterDiagnosticState,
    directive: HooksDirectiveStmt | ProfileHooksEntry,
) -> _AdapterDiagnosticState:
    enabled, cleared = _hook_event_changes(directive.fields)
    return replace(
        state,
        hooks_loc=directive.loc,
        hooks_enabled=directive.enabled,
        hook_events=(state.hook_events - cleared) | enabled,
        cleared_hook_events=(state.cleared_hook_events | cleared) - enabled,
    )


def _profile_capability_diagnostics(
    prog: Program | ProgramWithValidation,
    profile: ProfileDefStmt,
    state: _AdapterDiagnosticState,
    warned_approval_spans: set[tuple[str | None, int, int]],
) -> list[lsp.Diagnostic]:
    diagnostics: list[lsp.Diagnostic] = []
    for member in profile.agent_members:
        diagnostics.extend(
            _adapter_selection_diagnostics(
                prog,
                shortname=member.plain_value,
                expected_kind="agent-harness",
                loc=member.loc,
            )
        )
    for member in profile.adapter_members:
        diagnostics.extend(
            _adapter_selection_diagnostics(
                prog,
                shortname=member.plain_value,
                expected_kind="agent-framework",
                loc=member.loc,
            )
        )
    for member in profile.mcp_members:
        diagnostics.extend(
            _unsupported_capability_diagnostics(
                state,
                "mcp",
                [_directive_keyword_span(prog, member.loc, "mcp") or member.loc],
                "MCP server directives",
            )
        )
    for member in profile.artifacts_members:
        diagnostics.extend(
            _unsupported_capability_diagnostics(
                state,
                "artifacts",
                [_directive_keyword_span(prog, member.loc, "artifacts") or member.loc],
                "model-facing artifact tools",
            )
        )
    for member in profile.codemode_members:
        if member.enabled:
            diagnostics.extend(
                _unsupported_capability_diagnostics(
                    state,
                    "codemode",
                    [_directive_keyword_span(prog, member.loc, "codemode") or member.loc],
                    "CodeMode tool execution",
                    severity=lsp.DiagnosticSeverity.Error,
                    code="K-ADAPTER-CODEMODE",
                )
            )
    diagnostics.extend(_hooks_capability_diagnostics(state))
    diagnostics.extend(_history_capability_diagnostics(state))
    for member in profile.history_members:
        diagnostics.extend(_compaction_capability_diagnostics(state, member.fields))
    diagnostics.extend(
        _unsupported_capability_diagnostics(
            state,
            "subagents",
            [entry.loc for entry in profile.subagent_members],
            "Subagent delegation",
            severity=lsp.DiagnosticSeverity.Error,
            code="K-ADAPTER-SUBAGENTS",
        )
    )
    diagnostics.extend(
        _native_approval_capability_diagnostics(
            prog,
            state,
            warned_approval_spans,
        )
    )
    return diagnostics


def _scan_capability_scope(
    prog: Program | ProgramWithValidation,
    statements: Sequence[object],
    state: _AdapterDiagnosticState,
    profile_defs: dict[str, ProfileDefStmt],
    profile_states: dict[str, _AdapterDiagnosticState],
    proc_names: set[str],
    type_defs: dict[str, TypeDef],
    warned_approval_spans: set[tuple[str | None, int, int]] | None = None,
) -> list[lsp.Diagnostic]:
    diagnostics: list[lsp.Diagnostic] = []
    active = state
    if warned_approval_spans is None:
        warned_approval_spans = set()

    for stmt in statements:
        if isinstance(stmt, AgentDirectiveStmt):
            diagnostics.extend(
                _adapter_selection_diagnostics(
                    prog,
                    shortname=stmt.plain_value,
                    expected_kind="agent-harness",
                    loc=stmt.loc,
                )
            )
            selected = _adapter_state_from_plain_value(stmt.plain_value, "agent-harness")
            if selected is not None:
                active = _merge_adapter_state(active, selected)
                diagnostics.extend(
                    _native_approval_capability_diagnostics(
                        prog,
                        active,
                        warned_approval_spans,
                    )
                )
                diagnostics.extend(_history_capability_diagnostics(active))
                diagnostics.extend(_codemode_capability_diagnostics(active))
                diagnostics.extend(_hooks_capability_diagnostics(active))
            continue
        if isinstance(stmt, AdapterDirectiveStmt):
            diagnostics.extend(
                _adapter_selection_diagnostics(
                    prog,
                    shortname=stmt.plain_value,
                    expected_kind="agent-framework",
                    loc=stmt.loc,
                )
            )
            selected = _adapter_state_from_plain_value(stmt.plain_value, "agent-framework")
            if selected is not None:
                active = _merge_adapter_state(active, selected)
                diagnostics.extend(
                    _native_approval_capability_diagnostics(
                        prog,
                        active,
                        warned_approval_spans,
                    )
                )
                diagnostics.extend(_history_capability_diagnostics(active))
                diagnostics.extend(_codemode_capability_diagnostics(active))
                diagnostics.extend(_hooks_capability_diagnostics(active))
            continue
        if isinstance(stmt, ApprovalDirectiveStmt):
            active = replace(
                active,
                approval_loc=stmt.loc,
                approval_plain_value=stmt.plain_value,
            )
            diagnostics.extend(
                _native_approval_capability_diagnostics(
                    prog,
                    active,
                    warned_approval_spans,
                )
            )
            continue
        if isinstance(stmt, HooksDirectiveStmt):
            active = _with_hooks_state(active, stmt)
            diagnostics.extend(_hooks_capability_diagnostics(active))
            continue
        if isinstance(stmt, HistoryDirectiveStmt):
            active = replace(
                active,
                history_loc=stmt.loc,
                history_enabled=stmt.enabled,
            )
            diagnostics.extend(_history_capability_diagnostics(active))
            diagnostics.extend(_compaction_capability_diagnostics(active, stmt.fields))
            continue
        if isinstance(stmt, SkillsDirectiveStmt):
            if stmt.enabled:
                diagnostics.extend(
                    _unsupported_capability_diagnostics(
                        active,
                        "tool_registration",
                        [_directive_keyword_span(prog, stmt.loc, "skills") or stmt.loc],
                        "skill tool registration",
                    )
                )
            continue
        if isinstance(stmt, CodeModeDirectiveStmt):
            active = replace(
                active,
                codemode_loc=stmt.loc,
                codemode_enabled=stmt.enabled,
            )
            diagnostics.extend(_codemode_capability_diagnostics(active))
            continue
        if isinstance(stmt, McpDirectiveStmt):
            diagnostics.extend(
                _unsupported_capability_diagnostics(
                    active,
                    "mcp",
                    [_directive_keyword_span(prog, stmt.loc, "mcp") or stmt.loc],
                    "MCP server directives",
                )
            )
            continue
        if isinstance(stmt, ArtifactsDirectiveStmt):
            diagnostics.extend(
                _unsupported_capability_diagnostics(
                    active,
                    "artifacts",
                    [_directive_keyword_span(prog, stmt.loc, "artifacts") or stmt.loc],
                    "model-facing artifact tools",
                )
            )
            continue
        if isinstance(stmt, SettingsDirectiveStmt):
            continue
        if isinstance(stmt, UseDirectiveStmt):
            diagnostics.extend(
                _use_capability_diagnostics(
                    prog,
                    stmt,
                    active,
                    profile_defs,
                    profile_states,
                    proc_names,
                )
            )
            if stmt.name and not stmt.tool_names and stmt.name in profile_defs:
                active = _merge_adapter_state(
                    active,
                    profile_states.get(stmt.name, _AdapterDiagnosticState()),
                )
                diagnostics.extend(
                    _native_approval_capability_diagnostics(
                        prog,
                        active,
                        warned_approval_spans,
                    )
                )
                diagnostics.extend(_history_capability_diagnostics(active))
            continue
        if isinstance(stmt, TemplateBlock):
            diagnostics.extend(_template_capability_diagnostics(active, stmt, type_defs))
            continue
        if isinstance(stmt, TemplateLine):
            diagnostics.extend(_template_line_capability_diagnostics(active, stmt, type_defs))
            continue
        if isinstance(stmt, IfStmt):
            diagnostics.extend(
                _scan_capability_scope(
                    prog,
                    stmt.body,
                    active,
                    profile_defs,
                    profile_states,
                    proc_names,
                    type_defs,
                    warned_approval_spans,
                )
            )
            if stmt.else_body is not None:
                diagnostics.extend(
                    _scan_capability_scope(
                        prog,
                        stmt.else_body,
                        active,
                        profile_defs,
                        profile_states,
                        proc_names,
                        type_defs,
                        warned_approval_spans,
                    )
                )
            continue
        if isinstance(stmt, LoopStmt):
            loop_statements = list(stmt.body)
            if stmt.map_body is not None:
                loop_statements.extend(stmt.map_body)
            diagnostics.extend(
                _scan_capability_scope(
                    prog,
                    loop_statements,
                    active,
                    profile_defs,
                    profile_states,
                    proc_names,
                    type_defs,
                    warned_approval_spans,
                )
            )
            continue
        if isinstance(stmt, ConditionalLoopStmt):
            diagnostics.extend(
                _scan_capability_scope(
                    prog,
                    stmt.body,
                    active,
                    profile_defs,
                    profile_states,
                    proc_names,
                    type_defs,
                    warned_approval_spans,
                )
            )
            continue
        if isinstance(stmt, ProcedureDef):
            diagnostics.extend(
                _scan_capability_scope(
                    prog,
                    stmt.body,
                    active,
                    profile_defs,
                    profile_states,
                    proc_names,
                    type_defs,
                    warned_approval_spans,
                )
            )
            continue
        if isinstance(stmt, ProcDefStmt) and stmt.proc is not None:
            diagnostics.extend(
                _scan_capability_scope(
                    prog,
                    stmt.proc.body,
                    active,
                    profile_defs,
                    profile_states,
                    proc_names,
                    type_defs,
                    warned_approval_spans,
                )
            )
            continue
    return diagnostics


def _compaction_capability_diagnostics(
    state: _AdapterDiagnosticState,
    fields: Sequence[SettingsField],
) -> list[lsp.Diagnostic]:
    mode = next((field for field in fields if field.name == "compaction_mode"), None)
    threshold = next(
        (field for field in fields if field.name == "compaction_threshold"),
        None,
    )
    configured = mode or threshold
    if configured is None or configured.loc is None:
        return []
    if mode is not None and mode.plain_value == "disabled":
        return []
    if mode is not None and mode.backtick_code is not None:
        return []
    return _unsupported_capability_diagnostics(
        state,
        "native_compaction",
        [configured.loc],
        "provider-native conversation compaction",
        severity=lsp.DiagnosticSeverity.Error,
        code="K-ADAPTER-COMPACTION",
    )


def _history_capability_diagnostics(
    state: _AdapterDiagnosticState,
) -> list[lsp.Diagnostic]:
    if state.history_enabled is not True or state.history_loc is None:
        return []
    return _unsupported_capability_diagnostics(
        state,
        "stateful_history",
        [state.history_loc],
        "stateful conversation history",
        severity=lsp.DiagnosticSeverity.Error,
        code="K-ADAPTER-HISTORY",
    )


def _codemode_capability_diagnostics(
    state: _AdapterDiagnosticState,
) -> list[lsp.Diagnostic]:
    if state.codemode_enabled is not True or state.codemode_loc is None:
        return []
    return _unsupported_capability_diagnostics(
        state,
        "codemode",
        [state.codemode_loc],
        "CodeMode tool execution",
        severity=lsp.DiagnosticSeverity.Error,
        code="K-ADAPTER-CODEMODE",
    )


def _hooks_capability_diagnostics(
    state: _AdapterDiagnosticState,
    *,
    span: Span | None = None,
) -> list[lsp.Diagnostic]:
    if state.hooks_enabled is False or state.hooks_loc is None:
        return []
    diagnostic_span = span or state.hooks_loc
    generic = _unsupported_capability_diagnostics(
        state,
        "hooks",
        [diagnostic_span],
        "agent lifecycle hooks",
        severity=lsp.DiagnosticSeverity.Error,
        code="K-ADAPTER-HOOKS",
    )
    if generic or state.shortname is None or not state.hook_events:
        return generic
    try:
        hook_capabilities = adapter_capabilities(state.shortname).hooks
    except Exception:  # pragma: no cover - keep LSP diagnostics resilient
        hook_capabilities = None
    if hook_capabilities is None:
        return generic
    unsupported = hook_capabilities.unsupported(state.hook_events)
    if not unsupported:
        return []
    names = ", ".join(unsupported)
    return [
        lsp.Diagnostic(
            range=_span_to_range(diagnostic_span),
            severity=lsp.DiagnosticSeverity.Error,
            code="K-ADAPTER-HOOKS",
            source="kedi",
            message=(f"Adapter '{state.shortname}' does not support agent hook events: {names}."),
        )
    ]


def _adapter_selection_diagnostics(
    prog: Program | ProgramWithValidation,
    *,
    shortname: str | None,
    expected_kind: AgentAdapterKind,
    loc: Span | None,
) -> list[lsp.Diagnostic]:
    """Report invalid literal ``> agent:`` / ``> adapter:`` selections.

    Backtick values are resolved by the runtime, so only literal shortnames can
    be diagnosed statically here.
    """
    if shortname is None or loc is None:
        return []

    directive = "agent" if expected_kind == "agent-harness" else "adapter"
    kind_label = "agent harness" if expected_kind == "agent-harness" else "agent framework"
    span = _directive_name_span(prog, loc, shortname) or loc
    try:
        factory = adapter_factory(shortname)
    except Exception:
        known = ", ".join(
            name
            for name in known_adapter_shortnames()
            if adapter_factory(name).kind == expected_kind
        )
        return [
            lsp.Diagnostic(
                range=_span_to_range(span),
                severity=lsp.DiagnosticSeverity.Error,
                code="K-ADAPTER-NOT-FOUND",
                source="kedi",
                message=(
                    f"Unknown {kind_label} adapter {shortname!r}. "
                    f"Known {kind_label} adapters: {known}"
                ),
            )
        ]
    if factory.kind == expected_kind:
        return []

    actual = "agent harness" if factory.kind == "agent-harness" else "agent framework"
    return [
        lsp.Diagnostic(
            range=_span_to_range(span),
            severity=lsp.DiagnosticSeverity.Error,
            code="K-ADAPTER-KIND",
            source="kedi",
            message=(f"> {directive}: cannot select {shortname!r}; it is an {actual} adapter."),
        )
    ]


def _use_capability_diagnostics(
    prog: Program | ProgramWithValidation,
    stmt: UseDirectiveStmt,
    state: _AdapterDiagnosticState,
    profile_defs: dict[str, ProfileDefStmt],
    profile_states: dict[str, _AdapterDiagnosticState],
    proc_names: set[str],
) -> list[lsp.Diagnostic]:
    if stmt.tool_names:
        spans = [span for _, span in _use_tool_entry_spans(prog, stmt)]
        return _unsupported_capability_diagnostics(
            state,
            "tool_registration",
            spans or [stmt.loc],
            "agent tool registration",
        )
    if not stmt.name:
        return []

    target_span = _use_single_target_span(prog, stmt)
    if target_span is None:
        target_span = stmt.loc
    if target_span is None:
        return []
    if stmt.name == "skills" and stmt.name not in proc_names and stmt.name not in profile_defs:
        return [
            lsp.Diagnostic(
                range=_span_to_range(target_span),
                severity=lsp.DiagnosticSeverity.Error,
                code="K-SKILLS-USE",
                source="kedi",
                message="Skills use a standalone directive; replace '> use: skills' with '> skills: enabled'.",
            )
        ]
    if stmt.name in proc_names:
        return _unsupported_capability_diagnostics(
            state,
            "tool_registration",
            [target_span],
            "agent tool registration",
        )

    profile = profile_defs.get(stmt.name)
    if profile is None:
        return []
    merged = _merge_adapter_state(state, profile_states.get(stmt.name, _AdapterDiagnosticState()))
    capabilities = _profile_capability_usage(
        profile,
        profile_defs=profile_defs,
        proc_names=proc_names,
    )
    diagnostics: list[lsp.Diagnostic] = []
    if "tool_registration" in capabilities:
        diagnostics.extend(
            _unsupported_capability_diagnostics(
                merged,
                "tool_registration",
                [target_span],
                "agent tool registration from this profile",
            )
        )
    if "mcp" in capabilities:
        diagnostics.extend(
            _unsupported_capability_diagnostics(
                merged,
                "mcp",
                [target_span],
                "MCP server directives from this profile",
            )
        )
    if "subagents" in capabilities:
        diagnostics.extend(
            _unsupported_capability_diagnostics(
                merged,
                "subagents",
                [target_span],
                "subagent delegation from this profile",
                severity=lsp.DiagnosticSeverity.Error,
                code="K-ADAPTER-SUBAGENTS",
            )
        )
    if "codemode" in capabilities:
        diagnostics.extend(
            _unsupported_capability_diagnostics(
                merged,
                "codemode",
                [target_span],
                "CodeMode tool execution from this profile",
                severity=lsp.DiagnosticSeverity.Error,
                code="K-ADAPTER-CODEMODE",
            )
        )
    if "hooks" in capabilities:
        diagnostics.extend(_hooks_capability_diagnostics(merged, span=target_span))
    return diagnostics


def _profile_capability_usage(
    profile: ProfileDefStmt,
    *,
    profile_defs: dict[str, ProfileDefStmt],
    proc_names: set[str],
    active: frozenset[str] = frozenset(),
) -> set[AdapterCapability]:
    """Return capabilities exercised by a profile and composed profiles."""
    if profile.name in active:
        return set()

    capabilities: set[AdapterCapability] = set()
    if profile.tool_names:
        capabilities.add("tool_registration")
    if profile.mcp_members:
        capabilities.add("mcp")
    if profile.subagent_members:
        capabilities.add("subagents")
    if any(member.enabled for member in profile.skills_members):
        capabilities.add("tool_registration")
    if any(member.enabled for member in profile.codemode_members):
        capabilities.add("codemode")
    if any(member.enabled is not False for member in profile.hooks_members):
        capabilities.add("hooks")

    nested_active = active | {profile.name}
    for target in profile.use_targets:
        if target in proc_names:
            capabilities.add("tool_registration")
            continue
        nested = profile_defs.get(target)
        if nested is not None:
            capabilities.update(
                _profile_capability_usage(
                    nested,
                    profile_defs=profile_defs,
                    proc_names=proc_names,
                    active=nested_active,
                )
            )
    return capabilities


def _template_capability_diagnostics(
    state: _AdapterDiagnosticState,
    stmt: TemplateBlock,
    type_defs: dict[str, TypeDef],
) -> list[lsp.Diagnostic]:
    diagnostics: list[lsp.Diagnostic] = []
    for line in stmt.lines:
        diagnostics.extend(_template_line_capability_diagnostics(state, line, type_defs))
    return diagnostics


def _template_line_capability_diagnostics(
    state: _AdapterDiagnosticState,
    line: TemplateLine,
    type_defs: dict[str, TypeDef],
) -> list[lsp.Diagnostic]:
    output_segments = [seg for seg in line.segments if isinstance(seg, OutputSegment)]
    diagnostics = _unsupported_capability_diagnostics(
        state,
        "structured_output",
        [seg.loc for seg in output_segments],
        "structured template output fields",
    )
    if diagnostics or state.shortname is None:
        return diagnostics

    try:
        supported_formats = adapter_capabilities(state.shortname).supported_json_schema_formats
    except Exception:  # pragma: no cover - keep LSP diagnostics resilient
        supported_formats = None
    if supported_formats is None:
        return diagnostics

    for segment in output_segments:
        unsupported = _unsupported_type_formats(
            segment.type_expr,
            type_defs,
            supported_formats,
        )
        if not unsupported:
            continue
        type_name, schema_format = unsupported[0]
        span = segment.type_expr.loc if segment.type_expr is not None else segment.loc
        if span is None:
            continue
        diagnostics.append(
            lsp.Diagnostic(
                range=_span_to_range(span),
                severity=lsp.DiagnosticSeverity.Error,
                code="K-ADAPTER-SCHEMA-FORMAT",
                source="kedi",
                message=(
                    f"Adapter '{state.shortname}' does not support Kedi type "
                    f"'{type_name}' in structured output (JSON Schema format "
                    f"'{schema_format}'). Use Annotated[str, \"field description\"] "
                    "when a string with semantic guidance is sufficient."
                ),
            )
        )
    return diagnostics


def _unsupported_type_formats(
    expr: TypeExpr | None,
    type_defs: dict[str, TypeDef],
    supported_formats: frozenset[str],
    seen_types: frozenset[str] = frozenset(),
) -> list[tuple[str, str]]:
    if expr is None or isinstance(expr, (TypeLiteral, TypePython)):
        return []
    if isinstance(expr, TypeRef):
        schema_format = KEDI_RUNTIME_TYPE_JSON_SCHEMA_FORMATS.get(expr.name)
        if schema_format is not None and schema_format not in supported_formats:
            return [(expr.name, schema_format)]
        typedef = type_defs.get(expr.name)
        if typedef is None or expr.name in seen_types:
            return []
        nested_seen = seen_types | {expr.name}
        return [
            unsupported
            for field in typedef.fields
            for unsupported in _unsupported_type_formats(
                field.type_expr,
                type_defs,
                supported_formats,
                nested_seen,
            )
        ]
    if isinstance(expr, TypeApply):
        return [
            unsupported
            for arg in expr.args
            for unsupported in _unsupported_type_formats(
                arg,
                type_defs,
                supported_formats,
                seen_types,
            )
        ]
    if isinstance(expr, TypeUnion):
        return [
            *_unsupported_type_formats(
                expr.left,
                type_defs,
                supported_formats,
                seen_types,
            ),
            *_unsupported_type_formats(
                expr.right,
                type_defs,
                supported_formats,
                seen_types,
            ),
        ]
    return []


def _unsupported_capability_diagnostics(
    state: _AdapterDiagnosticState,
    feature: AdapterCapability,
    spans: Sequence[Span | None],
    description: str,
    *,
    severity: lsp.DiagnosticSeverity | None = None,
    code: str = "K-ADAPTER-CAPABILITY",
) -> list[lsp.Diagnostic]:
    supported = _adapter_supports(state, feature)
    if supported is not False or state.shortname is None:
        return []
    if severity is None:
        severity = (
            lsp.DiagnosticSeverity.Error
            if feature == "structured_output"
            else lsp.DiagnosticSeverity.Warning
        )
    return [
        lsp.Diagnostic(
            range=_span_to_range(span),
            severity=severity,
            code=code,
            source="kedi",
            message=(
                f"Adapter '{state.shortname}' does not support {description}; "
                "this construct will not be handled by that adapter."
            ),
        )
        for span in spans
        if span is not None
    ]


def _native_approval_capability_diagnostics(
    prog: Program | ProgramWithValidation,
    state: _AdapterDiagnosticState,
    warned_spans: set[tuple[str | None, int, int]],
) -> list[lsp.Diagnostic]:
    """Warn when a selected adapter cannot govern its own builtin or MCP tools.

    Kedi-registered ``ToolSpec`` callables still receive the common approval
    wrapper. The warning is intentionally limited to adapter-native calls,
    whose approval surface is unavailable for the selected adapter.
    """
    loc = state.approval_loc
    if loc is None or state.shortname is None:
        return []
    if _adapter_supports(state, "native_approvals") is not False:
        return []

    span = _approval_value_span(prog, loc, state.approval_plain_value) or loc
    key = (span.source_path, span.start_byte, span.end_byte)
    if key in warned_spans:
        return []
    warned_spans.add(key)
    return [
        lsp.Diagnostic(
            range=_span_to_range(span),
            severity=lsp.DiagnosticSeverity.Warning,
            code="K-ADAPTER-NATIVE-APPROVAL",
            source="kedi",
            message=(
                f"Adapter '{state.shortname}' cannot apply this approval policy to its "
                "builtin or MCP tools. Kedi-registered tools remain protected by Kedi."
            ),
        )
    ]


def _approval_value_span(
    prog: Program | ProgramWithValidation,
    loc: Span,
    plain_value: str | None,
) -> Span | None:
    if plain_value:
        return _directive_name_span(prog, loc, plain_value)
    return _directive_value_span(prog, loc)


def _directive_value_span(
    prog: Program | ProgramWithValidation,
    loc: Span,
) -> Span | None:
    """Return the non-whitespace value portion of one single-line directive."""
    line = _program_line_text(prog, loc.start_line, loc.source_path)
    line_start = _line_start_byte(prog, loc.start_line)
    if line is None or line_start is None:
        return None
    colon_index = line.find(":")
    if colon_index < 0:
        return None
    start_index = colon_index + 1
    while start_index < len(line) and line[start_index].isspace():
        start_index += 1
    end_index = len(line)
    while end_index > start_index and line[end_index - 1].isspace():
        end_index -= 1
    if start_index == end_index:
        return None
    start_byte = line_start + len(line[:start_index].encode("utf-8"))
    end_byte = line_start + len(line[:end_index].encode("utf-8"))
    return Span(
        start_byte=start_byte,
        end_byte=end_byte,
        start_line=loc.start_line,
        start_col=start_index + 1,
        end_line=loc.start_line,
        end_col=end_index + 1,
        source_path=loc.source_path,
    )


def _directive_keyword_span(
    prog: Program | ProgramWithValidation,
    loc: Span | None,
    keyword: str,
) -> Span | None:
    if loc is None:
        return None
    line = _program_line_text(prog, loc.start_line, loc.source_path)
    line_start = _line_start_byte(prog, loc.start_line)
    if line is None or line_start is None:
        return None
    name_index = line.find(keyword)
    if name_index < 0:
        return None
    start_byte = line_start + len(line[:name_index].encode("utf-8"))
    start_col = name_index + 1
    return Span(
        start_byte=start_byte,
        end_byte=start_byte + len(keyword.encode("utf-8")),
        start_line=loc.start_line,
        start_col=start_col,
        end_line=loc.start_line,
        end_col=start_col + len(keyword),
        source_path=loc.source_path,
    )


def _resolve_symbol_at(
    prog: Union[Program, ProgramWithValidation], pos: lsp.Position
) -> Optional[SymbolRef]:
    """Identify what the cursor is on. Returns ``None`` if nothing useful."""
    imported_surface = imported_surface_for_program(prog)
    proc_by_name = {p.name: p for p in prog.procedures}
    proc_by_name.update(imported_surface.procedures)
    type_by_name = {t.name: t for t in prog.type_defs}
    type_by_name.update(imported_surface.types)
    top_level_values = top_level_value_infos(prog)

    for module_import in prog.imports:
        name_span = _import_name_span(prog, module_import)
        if name_span is None or not _position_inside(pos, name_span):
            for selected_name in module_import.names:
                selected_span = module_import.name_locs[selected_name]
                if not _position_inside(pos, selected_span):
                    continue
                module_path = resolved_import_path(prog, module_import.module)
                selected_surface = (
                    load_imported_surface(module_path, set(), module_import.module)
                    if module_path is not None
                    else ImportedSurface()
                )
                if selected_name in selected_surface.procedures:
                    target = selected_surface.procedures[selected_name]
                    kind = "procedure"
                    declaration_span = target.loc
                elif selected_name in selected_surface.types:
                    target = selected_surface.types[selected_name]
                    kind = "type"
                    declaration_span = target.loc
                elif selected_name in selected_surface.values:
                    target = selected_surface.values[selected_name]
                    kind = "imported_value"
                    declaration_span = target.declaration_span
                elif selected_name in selected_surface.profiles:
                    target = selected_surface.profiles[selected_name]
                    kind = "agent_profile"
                    declaration_span = target.loc
                else:
                    kind = "imported_value"
                    declaration_span = None
                return SymbolRef(
                    kind=kind,
                    name=selected_name,
                    declaration_span=declaration_span,
                    scope_proc=None,
                    cursor_span=selected_span,
                )
            continue
        module_info = imported_surface.modules.get(module_import.module)
        return SymbolRef(
            kind="module",
            name=module_import.module,
            declaration_span=(module_info.declaration_span if module_info is not None else None),
            scope_proc=None,
            cursor_span=name_span,
        )

    profile_defs = _merged_profile_defs(prog, imported_surface)
    for profile_name, profile_def in profile_defs.items():
        name_span = (
            _directive_name_span(prog, profile_def.loc, profile_name) if profile_def.loc else None
        )
        if name_span is not None and _position_inside(pos, name_span):
            return SymbolRef(
                kind="agent_profile",
                name=profile_name,
                declaration_span=profile_def.loc,
                scope_proc=None,
                cursor_span=name_span,
            )

    for profile in prog.profile_defs.values():
        for entry in profile.output_members:
            resolved = _resolve_in_type_expr(
                entry.type_expr,
                pos,
                type_by_name,
                imported_surface.values,
            )
            if resolved is not None:
                return resolved
        for entry in profile.subagent_members:
            assert entry.loc is not None
            if not _position_inside(pos, entry.loc):
                continue
            target = profile_defs.get(entry.name)
            return SymbolRef(
                kind="agent_profile",
                name=entry.name,
                declaration_span=target.loc if target is not None else None,
                scope_proc=None,
                cursor_span=entry.loc,
            )
        for target_name, name_span in _profile_use_entry_spans(prog, profile):
            if not _position_inside(pos, name_span):
                continue
            target_proc = proc_by_name.get(target_name)
            if target_proc is not None:
                return SymbolRef(
                    kind="agent_tool",
                    name=target_name,
                    declaration_span=target_proc.loc,
                    scope_proc=None,
                    cursor_span=name_span,
                )
            target_profile = profile_defs.get(target_name)
            if target_profile is not None:
                return SymbolRef(
                    kind="agent_profile",
                    name=target_name,
                    declaration_span=target_profile.loc,
                    scope_proc=None,
                    cursor_span=name_span,
                )
            return SymbolRef(
                kind="agent_tool",
                name=target_name,
                declaration_span=None,
                scope_proc=None,
                cursor_span=name_span,
            )

    for item in prog.ordered_toplevel:
        if isinstance(item, UseDirectiveStmt) and item.loc is not None:
            if item.tool_names:
                for tool_name, name_span in _use_tool_entry_spans(prog, item):
                    if not _position_inside(pos, name_span):
                        continue
                    target = proc_by_name.get(tool_name)
                    if target is not None:
                        return SymbolRef(
                            kind="agent_tool",
                            name=tool_name,
                            declaration_span=target.loc,
                            scope_proc=None,
                            cursor_span=name_span,
                        )
                    return SymbolRef(
                        kind="agent_tool",
                        name=tool_name,
                        declaration_span=None,
                        scope_proc=None,
                        cursor_span=name_span,
                    )
            elif item.name:
                name_span = _use_single_target_span(prog, item)
                if name_span is not None and _position_inside(pos, name_span):
                    if item.name in proc_by_name:
                        target = proc_by_name[item.name]
                        return SymbolRef(
                            kind="agent_tool",
                            name=item.name,
                            declaration_span=target.loc,
                            scope_proc=None,
                            cursor_span=name_span,
                        )
                    profile_def = profile_defs.get(item.name)
                    return SymbolRef(
                        kind="agent_profile",
                        name=item.name,
                        declaration_span=(profile_def.loc if profile_def is not None else None),
                        scope_proc=None,
                        cursor_span=name_span,
                    )

    for suite in _validation_suites(prog):
        name_span = _validation_procedure_span(prog, suite)
        if name_span is None or not _position_inside(pos, name_span):
            continue
        target = proc_by_name.get(suite.procedure_name)
        return SymbolRef(
            kind="procedure",
            name=suite.procedure_name,
            declaration_span=(target.loc if target is not None else None),
            scope_proc=None,
            cursor_span=name_span,
        )

    export_name_hit = _exported_name_at_position(prog, pos)
    if export_name_hit is not None:
        export_name, export_span = export_name_hit
        if export_name in proc_by_name:
            target = proc_by_name[export_name]
            return SymbolRef(
                kind="procedure",
                name=export_name,
                declaration_span=target.loc,
                scope_proc=None,
                cursor_span=export_span,
            )
        if export_name in type_by_name:
            target = type_by_name[export_name]
            return SymbolRef(
                kind="type",
                name=export_name,
                declaration_span=target.loc,
                scope_proc=None,
                cursor_span=export_span,
            )
        if export_name in top_level_values:
            target = top_level_values[export_name]
            return SymbolRef(
                kind="module_value",
                name=export_name,
                declaration_span=target.declaration_span,
                scope_proc=None,
                cursor_span=export_span,
            )
        if export_name in imported_surface.values:
            target = imported_surface.values[export_name]
            return SymbolRef(
                kind="imported_value",
                name=export_name,
                declaration_span=target.declaration_span,
                scope_proc=None,
                cursor_span=export_span,
            )
        if export_name in profile_defs:
            profile_def = profile_defs[export_name]
            return SymbolRef(
                kind="agent_profile",
                name=export_name,
                declaration_span=profile_def.loc,
                scope_proc=None,
                cursor_span=export_span,
            )

    # Walk procedures first — they own the deepest spans.
    for proc in prog.procedures:
        r = _resolve_in_procedure(
            prog, proc, pos, proc_by_name, type_by_name, imported_surface.values
        )
        if r is not None:
            return r

    # Top-level
    for stmt in prog.toplevel_body:
        r = _resolve_in_stmt(
            prog,
            stmt,
            pos,
            None,
            proc_by_name,
            type_by_name,
            imported_surface.values,
        )
        if r is not None:
            return r
    if prog.toplevel_return is not None:
        r = _resolve_in_stmt(
            prog,
            prog.toplevel_return,
            pos,
            None,
            proc_by_name,
            type_by_name,
            imported_surface.values,
        )
        if r is not None:
            return r
    # Type defs (top-level).
    for td in prog.type_defs:
        r = _resolve_in_type_def(td, pos, type_by_name, imported_surface.values)
        if r is not None:
            return r
    return None


def _resolve_in_procedure(
    prog: Program | ProgramWithValidation,
    proc: ProcedureDef,
    pos: lsp.Position,
    proc_by_name: dict,
    type_by_name: dict,
    imported_values: dict[str, ImportedValueInfo],
    ancestor_procs: tuple[ProcedureDef, ...] = (),
) -> Optional[SymbolRef]:
    if proc.loc is None or not _position_inside(pos, proc.loc):
        return None
    scoped_type_by_name = {
        **type_by_name,
        **{typedef.name: typedef for typedef in _direct_type_defs(proc)},
    }
    scoped_proc_by_name = {
        **proc_by_name,
        **{child.name: child for child in _nested_procedures(proc)},
    }
    for child in _nested_procedures(proc):
        r = _resolve_in_procedure(
            prog,
            child,
            pos,
            scoped_proc_by_name,
            scoped_type_by_name,
            imported_values,
            ancestor_procs + (proc,),
        )
        if r is not None:
            return r
    if pos.line + 1 == proc.loc.start_line and _position_in_proc_name(proc, pos):
        return SymbolRef(
            kind="procedure",
            name=proc.name,
            declaration_span=proc.loc,
            scope_proc=None,
            cursor_span=_proc_name_span(proc),
        )
    for pname, pspan in _param_decls(proc):
        if _position_inside(pos, pspan):
            return SymbolRef(
                kind="param_decl",
                name=pname,
                declaration_span=pspan,
                scope_proc=proc,
                cursor_span=pspan,
            )
    r = _resolve_in_proc_signature_types(proc, pos, scoped_type_by_name, imported_values)
    if r is not None:
        return r
    for stmt in proc.body:
        r = _resolve_in_stmt(
            prog,
            stmt,
            pos,
            proc,
            scoped_proc_by_name,
            scoped_type_by_name,
            imported_values,
            ancestor_procs,
        )
        if r is not None:
            return r
    return None


def _position_in_proc_name(proc: ProcedureDef, pos: lsp.Position) -> bool:
    """True iff the cursor is exactly on the procedure's name token.

    The header `@<name>(...)` starts at column 1 in 1-based terms. The
    name occupies columns 2..2+len(name).
    """
    assert proc.loc is not None
    if pos.line + 1 != proc.loc.start_line:
        return False
    return _position_inside(pos, _proc_name_span(proc))


def _proc_name_span(proc: ProcedureDef) -> Span:
    assert proc.loc is not None
    return name_token_span(proc.loc, proc.name, prefix_len=1)


def _param_decls(proc: ProcedureDef) -> List[Tuple[str, Span]]:
    """Best-effort span for each parameter declaration in the header.

    We don't carry parameter spans in the AST (the walker captures the
    procedure-level Span only), so we reconstruct them by searching
    the procedure's first physical line. Cheap and good enough for
    cursor hit-testing on a single line.
    """
    source = current_source()
    if proc.loc is None or not proc.params or source is None:
        return []
    header_bytes = source.source.encode("utf-8")[proc.loc.start_byte :]
    header = header_bytes.split(b"\n", 1)[0].decode("utf-8")
    cursor = header.find("(") + 1
    declarations: List[Tuple[str, Span]] = []
    for name in proc.params:
        match = re.search(rf"(?<!\w){re.escape(name)}(?!\w)", header[cursor:])
        if match is None:
            continue
        start_char = cursor + match.start()
        end_char = cursor + match.end()
        start_offset = len(header[:start_char].encode("utf-8"))
        end_offset = len(header[:end_char].encode("utf-8"))
        declarations.append(
            (
                name,
                Span(
                    start_byte=proc.loc.start_byte + start_offset,
                    end_byte=proc.loc.start_byte + end_offset,
                    start_line=proc.loc.start_line,
                    start_col=start_offset + 1,
                    end_line=proc.loc.start_line,
                    end_col=end_offset + 1,
                    source_path=proc.loc.source_path,
                    source_id=proc.loc.source_id,
                ),
            )
        )
        cursor = end_char
    return declarations


def _param_decl_span(proc: ProcedureDef, name: str) -> Optional[Span]:
    return next((span for param_name, span in _param_decls(proc) if param_name == name), None)


def _resolve_lexical_name(
    proc: Optional[ProcedureDef],
    ancestor_procs: tuple[ProcedureDef, ...],
    name: str,
) -> tuple[str, Span, ProcedureDef] | None:
    """Find the nearest Kedi-local declaration visible from a procedure body."""
    if proc is None:
        return None
    for candidate in reversed((*ancestor_procs, proc)):
        if name in candidate.params:
            declaration_span = _param_decl_span(candidate, name)
            if declaration_span is not None:
                return "parameter", declaration_span, candidate
        declaration_span = _find_local_declaration(candidate, name)
        if declaration_span is not None:
            return "local", declaration_span, candidate
    return None


def _direct_statement_declarations(
    stmt,
    name: str,
    proc: Optional[ProcedureDef],
) -> list[_ValueDeclaration]:
    declarations: list[_ValueDeclaration] = []
    if isinstance(stmt, (VariableInitStmt, VariableInitBlockStmt)) and stmt.name == name:
        span = binding_name_span(stmt)
        if span is not None:
            declarations.append(
                _ValueDeclaration(
                    kind="local" if proc is not None else "module_value",
                    name=name,
                    declaration_span=span,
                    scope_proc=proc,
                )
            )
    for segment in _segments_of(stmt):
        if isinstance(segment, OutputSegment) and segment.name == name:
            span = _output_name_span(segment)
            if span is not None:
                declarations.append(
                    _ValueDeclaration(
                        kind="output_decl",
                        name=name,
                        declaration_span=span,
                        scope_proc=proc,
                    )
                )
    return declarations


def _statements_contain_byte(statements: Sequence[object], byte: int) -> bool:
    for stmt in statements:
        loc = getattr(stmt, "loc", None)
        if isinstance(loc, Span) and loc.start_byte <= byte < loc.end_byte:
            return True
    return False


def _declaration_in_scope_path(
    statements: Sequence[object],
    name: str,
    reference_byte: int,
    proc: Optional[ProcedureDef],
) -> _ValueDeclaration | None:
    """Resolve ``name`` in the lexical block containing ``reference_byte``.

    Control-flow bodies are child scopes. A loop body and its attached map
    share one iteration scope, with the binder acting as that scope's first
    declaration.
    """

    latest: _ValueDeclaration | None = None
    for stmt in statements:
        loc = getattr(stmt, "loc", None)
        if loc is None:
            continue
        if loc.start_byte >= reference_byte:
            break
        if loc.start_byte <= reference_byte < loc.end_byte:
            if isinstance(stmt, IfStmt):
                for branch in (stmt.body, stmt.else_body or []):
                    if _statements_contain_byte(branch, reference_byte):
                        return (
                            _declaration_in_scope_path(branch, name, reference_byte, proc) or latest
                        )
                return latest
            if isinstance(stmt, LoopStmt):
                iteration_body = [*stmt.body, *(stmt.map_body or [])]
                if not _statements_contain_byte(iteration_body, reference_byte):
                    return latest
                nested = _declaration_in_scope_path(
                    iteration_body,
                    name,
                    reference_byte,
                    proc,
                )
                if nested is not None:
                    return nested
                if stmt.binder == name and stmt.binder_loc is not None:
                    return _ValueDeclaration(
                        kind="loop_binder",
                        name=name,
                        declaration_span=stmt.binder_loc,
                        scope_proc=proc,
                    )
                return latest
            if isinstance(stmt, ConditionalLoopStmt):
                if _statements_contain_byte(stmt.body, reference_byte):
                    return (
                        _declaration_in_scope_path(
                            stmt.body,
                            name,
                            reference_byte,
                            proc,
                        )
                        or latest
                    )
                return latest
            return latest
        declarations = _direct_statement_declarations(stmt, name, proc)
        if declarations:
            latest = max(declarations, key=lambda item: item.declaration_span.start_byte)
    return latest


def _resolve_value_declaration(
    prog: Program | ProgramWithValidation,
    proc: Optional[ProcedureDef],
    ancestor_procs: tuple[ProcedureDef, ...],
    name: str,
    reference_span: Span,
) -> _ValueDeclaration | None:
    reference_byte = reference_span.start_byte
    if proc is not None:
        declaration = _declaration_in_scope_path(proc.body, name, reference_byte, proc)
        if declaration is not None:
            return declaration
        parameter_span = _param_decl_span(proc, name) if name in proc.params else None
        if parameter_span is not None:
            return _ValueDeclaration("parameter", name, parameter_span, proc)

        for ancestor in reversed(ancestor_procs):
            declaration = _declaration_in_scope_path(
                ancestor.body,
                name,
                reference_byte,
                ancestor,
            )
            if declaration is not None:
                return declaration
            parameter_span = _param_decl_span(ancestor, name) if name in ancestor.params else None
            if parameter_span is not None:
                return _ValueDeclaration("parameter", name, parameter_span, ancestor)

        top_level_reference = (
            max(
                (
                    loc.end_byte
                    for stmt in prog.toplevel_body
                    if isinstance((loc := getattr(stmt, "loc", None)), Span)
                ),
                default=reference_byte,
            )
            + 1
        )
        return _declaration_in_scope_path(
            prog.toplevel_body,
            name,
            top_level_reference,
            None,
        )

    return _declaration_in_scope_path(
        prog.toplevel_body,
        name,
        reference_byte,
        None,
    )


def _resolve_in_proc_signature_types(
    proc: ProcedureDef,
    pos: lsp.Position,
    type_by_name: dict,
    imported_values: dict[str, ImportedValueInfo],
) -> Optional[SymbolRef]:
    """Resolve type names used in a procedure signature."""
    for expr in (proc.param_type_exprs or {}).values():
        r = _resolve_in_type_expr(expr, pos, type_by_name, imported_values)
        if r is not None:
            return r
    return _resolve_in_type_expr(proc.return_type_expr, pos, type_by_name, imported_values)


def _resolve_in_stmt(
    prog: Program | ProgramWithValidation,
    stmt,
    pos: lsp.Position,
    proc: Optional[ProcedureDef],
    proc_by_name: dict,
    type_by_name: dict,
    imported_values: dict[str, ImportedValueInfo],
    ancestor_procs: tuple[ProcedureDef, ...] = (),
) -> Optional[SymbolRef]:
    if not hasattr(stmt, "loc") or stmt.loc is None or not _position_inside(pos, stmt.loc):
        return None
    if (
        isinstance(stmt, LoopStmt)
        and stmt.binder_loc is not None
        and _position_inside(pos, stmt.binder_loc)
    ):
        return SymbolRef(
            kind="loop_binder",
            name=stmt.binder,
            declaration_span=stmt.binder_loc,
            scope_proc=proc,
            cursor_span=stmt.binder_loc,
        )
    if isinstance(stmt, TypeDefStmt) and stmt.typedef is not None:
        return _resolve_in_type_def(stmt.typedef, pos, type_by_name, imported_values)
    # VariableInitStmt / VariableInitBlockStmt target name
    if isinstance(stmt, (VariableInitStmt, VariableInitBlockStmt)) and stmt.loc is not None:
        name_span = binding_name_span(stmt)
        if name_span is not None and _position_inside(pos, name_span):
            return SymbolRef(
                kind=(
                    "module_value"
                    if proc is None and any(stmt is item for item in prog.toplevel_body)
                    else "local"
                ),
                name=stmt.name,
                declaration_span=name_span,
                scope_proc=proc,
                cursor_span=name_span,
            )
    if isinstance(stmt, (AssignmentStmt, AssignmentBlockStmt)) and stmt.loc is not None:
        name_span = binding_name_span(stmt)
        if name_span is not None and _position_inside(pos, name_span):
            declaration = _resolve_value_declaration(
                prog,
                proc,
                ancestor_procs,
                stmt.name,
                name_span,
            )
            if declaration is not None:
                return SymbolRef(
                    kind=declaration.kind,
                    name=stmt.name,
                    declaration_span=declaration.declaration_span,
                    scope_proc=declaration.scope_proc,
                    cursor_span=name_span,
                )
            if stmt.name in imported_values:
                return SymbolRef(
                    kind="imported_value",
                    name=stmt.name,
                    declaration_span=imported_values[stmt.name].declaration_span,
                    scope_proc=proc,
                    cursor_span=name_span,
                )
            return SymbolRef(
                kind="runtime_value" if stmt.name == "args" else "local",
                name=stmt.name,
                declaration_span=None,
                scope_proc=proc,
                cursor_span=name_span,
            )
    if isinstance(stmt, VariableInitStmt) and stmt.type_expr is not None:
        r = _resolve_in_type_expr(stmt.type_expr, pos, type_by_name, imported_values)
        if r is not None:
            return r
    if isinstance(stmt, VariableInitBlockStmt) and stmt.type_expr is not None:
        r = _resolve_in_type_expr(stmt.type_expr, pos, type_by_name, imported_values)
        if r is not None:
            return r
    if isinstance(stmt, UseDirectiveStmt) and stmt.loc is not None:
        if stmt.tool_names:
            for tool_name, name_span in _use_tool_entry_spans(prog, stmt):
                if not _position_inside(pos, name_span):
                    continue
                target = proc_by_name.get(tool_name)
                if target is not None:
                    return SymbolRef(
                        kind="agent_tool",
                        name=tool_name,
                        declaration_span=target.loc,
                        scope_proc=proc,
                        cursor_span=name_span,
                    )
                return SymbolRef(
                    kind="agent_tool",
                    name=tool_name,
                    declaration_span=None,
                    scope_proc=proc,
                    cursor_span=name_span,
                )
        elif stmt.name:
            name_span = _use_single_target_span(prog, stmt)
            if name_span is not None and _position_inside(pos, name_span):
                if stmt.name in proc_by_name:
                    target = proc_by_name[stmt.name]
                    return SymbolRef(
                        kind="agent_tool",
                        name=stmt.name,
                        declaration_span=target.loc,
                        scope_proc=proc,
                        cursor_span=name_span,
                    )
                imported_surface = imported_surface_for_program(prog)
                profile_def = _merged_profile_defs(prog, imported_surface).get(stmt.name)
                return SymbolRef(
                    kind="agent_profile",
                    name=stmt.name,
                    declaration_span=(profile_def.loc if profile_def is not None else None),
                    scope_proc=proc,
                    cursor_span=name_span,
                )
    for seg in _segments_of(stmt):
        r = _resolve_in_segment(
            seg,
            pos,
            proc,
            proc_by_name,
            type_by_name,
            imported_values,
            ancestor_procs,
            prog=prog,
        )
        if r is not None:
            return r
    for child in _control_body_statements(stmt):
        r = _resolve_in_stmt(
            prog,
            child,
            pos,
            proc,
            proc_by_name,
            type_by_name,
            imported_values,
            ancestor_procs,
        )
        if r is not None:
            return r
    return None


def _resolve_in_segment(
    seg,
    pos: lsp.Position,
    proc: Optional[ProcedureDef],
    proc_by_name: dict,
    type_by_name: dict,
    imported_values: dict[str, ImportedValueInfo],
    ancestor_procs: tuple[ProcedureDef, ...] = (),
    *,
    prog: Program | ProgramWithValidation | None = None,
) -> Optional[SymbolRef]:
    if not hasattr(seg, "loc") or seg.loc is None or not _position_inside(pos, seg.loc):
        return None
    if isinstance(seg, InputSegment):
        # Could be a parameter, an output, or an assign-target reference.
        name_span = _input_name_span(seg) or seg.loc
        declaration = (
            _resolve_value_declaration(
                prog,
                proc,
                ancestor_procs,
                seg.name,
                name_span,
            )
            if prog is not None
            else None
        )
        if declaration is None and prog is None:
            lexical = _resolve_lexical_name(proc, ancestor_procs, seg.name)
            if lexical is not None:
                kind, declaration_span, owner = lexical
                declaration = _ValueDeclaration(kind, seg.name, declaration_span, owner)
        if declaration is not None:
            return SymbolRef(
                kind=declaration.kind,
                name=seg.name,
                declaration_span=declaration.declaration_span,
                scope_proc=declaration.scope_proc,
                cursor_span=name_span,
            )
        if seg.name in imported_values:
            return SymbolRef(
                kind="imported_value",
                name=seg.name,
                declaration_span=imported_values[seg.name].declaration_span,
                scope_proc=proc,
                cursor_span=name_span,
            )
        if seg.name == "args":
            return SymbolRef(
                kind="runtime_value",
                name=seg.name,
                declaration_span=None,
                scope_proc=proc,
                cursor_span=name_span,
            )
        return SymbolRef(
            kind="local",
            name=seg.name,
            declaration_span=None,
            scope_proc=proc,
            cursor_span=name_span,
        )
    if isinstance(seg, CallSegment):
        name_span = name_token_span(seg.loc, seg.name, prefix_len=1)
        if _position_inside(pos, name_span):
            target = proc_by_name.get(seg.name)
            if target is None and seg.name in imported_values:
                imported_value = imported_values[seg.name]
                return SymbolRef(
                    kind="imported_value",
                    name=seg.name,
                    declaration_span=imported_value.declaration_span,
                    scope_proc=proc,
                    cursor_span=name_span,
                )
            return SymbolRef(
                kind="procedure",
                name=seg.name,
                declaration_span=target.loc if target is not None else None,
                scope_proc=None,
                cursor_span=name_token_span(seg.loc, seg.name, prefix_len=1),
            )
        # Recurse into args.

        for arg in seg.args:
            for inner in iter_call_arg_nested(arg):
                r = _resolve_in_segment(
                    inner,
                    pos,
                    proc,
                    proc_by_name,
                    type_by_name,
                    imported_values,
                    ancestor_procs,
                    prog=prog,
                )
                if r is not None:
                    return r
    if isinstance(seg, OutputSegment):
        # The name occupies the first identifier after `[`.
        name_span = _output_name_span(seg)
        if name_span is not None and _position_inside(pos, name_span):
            return SymbolRef(
                kind="output_decl",
                name=seg.name,
                declaration_span=name_span,
                scope_proc=proc,
                cursor_span=name_span,
            )
        # type annotation
        if seg.type_expr is not None:
            r = _resolve_in_type_expr(seg.type_expr, pos, type_by_name, imported_values)
            if r is not None:
                return r
    return None


def _resolve_in_type_def(
    td: TypeDef,
    pos: lsp.Position,
    type_by_name: dict,
    imported_values: dict[str, ImportedValueInfo] | None = None,
) -> Optional[SymbolRef]:
    if td.loc is None or not _position_inside(pos, td.loc):
        return None
    # `~Name(...)` — the name follows the sigil.
    name_span = name_token_span(td.loc, td.name, prefix_len=1)
    if _position_inside(pos, name_span):
        return SymbolRef(
            kind="type",
            name=td.name,
            declaration_span=td.loc,
            scope_proc=None,
            cursor_span=name_token_span(td.loc, td.name, prefix_len=1),
        )
    for f in td.fields:
        r = _resolve_in_type_expr(f.type_expr, pos, type_by_name, imported_values)
        if r is not None:
            return r
        if f.loc is not None and _position_inside(pos, f.loc):
            return SymbolRef(
                kind="type_field",
                name=f.name,
                declaration_span=f.loc,
                scope_proc=None,
                cursor_span=f.loc,
            )
    return None


def _resolve_in_type_expr(
    expr: Optional[TypeExpr],
    pos: lsp.Position,
    type_by_name: dict,
    imported_values: dict[str, ImportedValueInfo] | None = None,
) -> Optional[SymbolRef]:
    if expr is None or expr.loc is None or not _position_inside(pos, expr.loc):
        return None
    if isinstance(expr, TypeRef):
        target = type_by_name.get(expr.name)
        imported = (imported_values or {}).get(expr.name)
        if target is None and imported is not None and _is_imported_python_class(imported):
            return SymbolRef(
                kind="imported_value",
                name=expr.name,
                declaration_span=imported.declaration_span,
                scope_proc=None,
                cursor_span=expr.loc,
            )
        return SymbolRef(
            kind="type",
            name=expr.name,
            declaration_span=target.loc if target is not None else None,
            scope_proc=None,
            cursor_span=expr.loc,
        )
    if isinstance(expr, TypeApply):
        # `Name[arg,...]` — cursor on Name?
        name_span = name_token_span(expr.loc, expr.name, prefix_len=0)
        on_name = _position_inside(pos, name_span)
        if on_name:
            target = type_by_name.get(expr.name)
            imported = (imported_values or {}).get(expr.name)
            if target is None and imported is not None and _is_imported_python_class(imported):
                return SymbolRef(
                    kind="imported_value",
                    name=expr.name,
                    declaration_span=imported.declaration_span,
                    scope_proc=None,
                    cursor_span=name_span,
                )
            return SymbolRef(
                kind="type",
                name=expr.name,
                declaration_span=target.loc if target is not None else None,
                scope_proc=None,
                cursor_span=name_span,
            )
        for arg in expr.args:
            r = _resolve_in_type_expr(arg, pos, type_by_name, imported_values)
            if r is not None:
                return r
    if isinstance(expr, TypeUnion):
        for child in (expr.left, expr.right):
            r = _resolve_in_type_expr(child, pos, type_by_name, imported_values)
            if r is not None:
                return r
    return None


def _is_imported_python_class(value: ImportedValueInfo | None) -> bool:
    return (
        value is not None
        and value.python_stub is not None
        and value.python_stub.startswith("class ")
    )


def _find_local_declaration(proc: Optional[ProcedureDef], name: str) -> Optional[Span]:
    """Find an output/assign-target named ``name`` in a procedure body."""
    if proc is None:
        return None
    for stmt in _iter_control_statements(proc.body):
        if isinstance(stmt, (VariableInitStmt, VariableInitBlockStmt)) and stmt.name == name:
            return binding_name_span(stmt)
        for seg in _segments_of(stmt):
            if isinstance(seg, OutputSegment) and seg.name == name:
                return _output_name_span(seg)
    return None


def _nested_procedures(proc: ProcedureDef) -> list[ProcedureDef]:
    from ..lang.ast import ProcDefStmt

    return [
        stmt.proc
        for stmt in _iter_control_statements(proc.body)
        if isinstance(stmt, ProcDefStmt) and stmt.proc is not None
    ]


def _direct_type_defs(proc: ProcedureDef) -> list[TypeDef]:
    return [
        stmt.typedef
        for stmt in _iter_control_statements(proc.body)
        if isinstance(stmt, TypeDefStmt) and stmt.typedef is not None
    ]


def _all_type_defs(prog: Program | ProgramWithValidation) -> list[TypeDef]:
    type_defs = list(prog.type_defs)

    def visit(proc: ProcedureDef) -> None:
        type_defs.extend(_direct_type_defs(proc))
        for child in _nested_procedures(proc):
            visit(child)

    for proc in prog.procedures:
        visit(proc)
    return type_defs


def _type_def_for_declaration(
    prog: Program | ProgramWithValidation,
    declaration_span: Optional[Span],
) -> Optional[TypeDef]:
    if declaration_span is None:
        return None
    return next(
        (typedef for typedef in _all_type_defs(prog) if typedef.loc == declaration_span),
        None,
    )


def _procedure_for_declaration(
    prog: Program | ProgramWithValidation,
    declaration_span: Optional[Span],
) -> Optional[ProcedureDef]:
    if declaration_span is None:
        return None
    return next(
        (proc for proc in _all_procedures(prog) if proc.loc == declaration_span),
        None,
    )


def _all_procedures(prog: Union[Program, ProgramWithValidation]) -> list[ProcedureDef]:
    out: list[ProcedureDef] = []

    def visit(proc: ProcedureDef) -> None:
        out.append(proc)
        for child in _nested_procedures(proc):
            visit(child)

    for proc in prog.procedures:
        visit(proc)
    return out


# =====================================================================
# Hover
# =====================================================================


@with_source_coordinates(1)
def compute_hover(
    source: str, source_path: Optional[str], pos: lsp.Position
) -> Optional[lsp.Hover]:
    """Markdown hover for the symbol under the cursor."""
    # Embedded Python owns hover inside its exact source regions. Returning a Kedi
    # fallback here races the editor's Pyright/Pylance forwarding provider and can
    # mask the Python symbol/type information with a generic statement hover.
    if _position_in_python_hover_region(source, pos):
        return None
    try:
        prog = parse_program(source, source_path=source_path)
    except KediParseError:
        return None
    if (doc_hover := _docstring_hover(prog, pos)) is not None:
        return doc_hover
    if _position_in_comment(source, pos):
        return None
    sym = _resolve_symbol_at(prog, pos)
    if sym is None:
        # Fall back to "generic statement" hover so the cursor inside
        # known regions at least returns something descriptive.
        return _generic_hover(prog, pos)

    range_ = _span_to_range(sym.cursor_span)
    imported_surface = imported_surface_for_program(prog)
    proc_by_name = {p.name: p for p in prog.procedures}
    proc_by_name.update(imported_surface.procedures)
    type_by_name = {t.name: t for t in prog.type_defs}
    type_by_name.update(imported_surface.types)

    if sym.kind == "module":
        module_info = imported_surface.modules.get(sym.name)
        if module_info is None:
            text = (
                f"`> import: {sym.name}`\n\n"
                "Module could not be resolved from the current file or Kedi's built-in modules."
            )
        else:
            text = f"**Module** `{sym.name}`"
            if module_info.docstring:
                text += f"\n\n{module_info.docstring}"
    elif sym.kind == "agent_tool":
        target = _procedure_for_declaration(prog, sym.declaration_span) or proc_by_name.get(
            sym.name
        )
        if target is not None:
            text = _agent_tool_markdown(sym.name, target)
        else:
            text = (
                f"**Agent tool** `> use: {sym.name}`\n\n"
                f"No procedure named `{sym.name}` is available in the current module or its imports."
            )
    elif sym.kind == "procedure":
        target = _procedure_for_declaration(prog, sym.declaration_span) or proc_by_name.get(
            sym.name
        )
        if target is not None:
            text = _proc_signature_markdown(target)
        else:
            text = f"**{sym.name}** — procedure not found in this module."
    elif sym.kind == "type":
        target = _type_def_for_declaration(prog, sym.declaration_span) or type_by_name.get(sym.name)
        if target is not None:
            text = _type_signature_markdown(target)
        elif (origin := _known_lsp_type_origin(sym.name)) is not None:
            text = _known_type_markdown(sym.name, origin)
        elif sym.name in _prelude_defined_names(prog.prelude):
            text = _prelude_type_markdown(sym.name)
        elif sym.name in _module_scope_python_names(prog):
            text = _python_module_scope_type_markdown(sym.name)
        else:
            text = (
                f"`{sym.name}`\n\n"
                "Not resolved statically. This type must be provided by the runtime environment."
            )
    elif sym.kind in ("parameter", "param_decl"):
        if sym.scope_proc is not None:
            text = (
                f"`{sym.name}: {_proc_param_annotation(sym.scope_proc, sym.name)}`\n\n"
                f"Parameter of `@{sym.scope_proc.name}`."
            )
        else:
            text = f"`{sym.name}`\n\nParameter."
    elif sym.kind == "loop_binder":
        text = (
            f"**Loop binder** `{sym.name}: int`\n\n"
            "Available only while this sequential range-loop iteration executes. "
            "The previous binding is restored after the loop."
        )
    elif sym.kind == "local":
        text = (
            f"**Local** `{sym.name}`\n\n"
            "Local value produced by an initialization or template output in this scope."
        )
    elif sym.kind == "runtime_value":
        text = (
            "**Runtime value** `args: AttrDict[str, ArgType]`\n\n"
            "Command-line arguments passed after the Kedi source path. Values are strings, "
            "a bare flag resolves to `True`, and missing names resolve to `None`."
        )
    elif sym.kind == "module_value":
        text = f"**Module value** `{sym.name}`\n\nTop-level value exported by this module."
    elif sym.kind == "imported_value":
        imported_value = imported_surface.values.get(sym.name)
        text = _imported_value_markdown(sym.name, imported_value)
        if imported_value is not None and imported_value.module_name:
            text += f"\n\nImported from `{imported_value.module_name}`."
        else:
            text += "\n\nImported from another Kedi module."
    elif sym.kind == "output_decl":
        text = (
            f"**Output** `{sym.name}`\n\n"
            "Structured output generated by this template call. Available after the call completes."
        )
    elif sym.kind == "type_field":
        target_field = next(
            (
                field
                for typedef in _all_type_defs(prog)
                for field in typedef.fields
                if field.loc == sym.declaration_span and field.name == sym.name
            ),
            None,
        )
        if target_field is not None:
            text = f"**Field** `{sym.name}: {_type_field_annotation(target_field)}` — type definition member."
        else:
            text = f"**Field** `{sym.name}` — type definition member."
    elif sym.kind == "agent_profile":
        profile_def = _merged_profile_defs(prog, imported_surface).get(sym.name)
        if profile_def is not None:
            text = _profile_signature_markdown(profile_def)
        else:
            text = _missing_agent_profile_hover(sym.name, imported_surface)
    else:
        text = f"**{sym.kind}** `{sym.name}`"
    return lsp.Hover(
        contents=lsp.MarkupContent(kind=lsp.MarkupKind.Markdown, value=text),
        range=range_,
    )


def _imported_value_markdown(name: str, value: ImportedValueInfo | None) -> str:
    if value is None:
        return f"**Imported value** `{name}`"
    if value.type_expr is not None:
        return f"**Imported value** `{name}: {_type_expr_to_str(value.type_expr)}`"
    if value.python_stub is None:
        return f"**Imported value** `{name}`"

    stub = value.python_stub
    if stub.startswith("async def "):
        signature = stub.removeprefix("async def ").removesuffix(": ...")
        return f"**Imported value**\n\n```python\nasync {signature}\n```\n\nPython async function."
    if stub.startswith("def "):
        signature = stub.removeprefix("def ").removesuffix(": ...")
        return f"**Imported value**\n\n```python\n{signature}\n```\n\nPython function."
    if stub.startswith("class "):
        type_name = stub.splitlines()[0].removeprefix("class ").removesuffix(":")
        return f"**Imported value**\n\n```python\nclass {type_name}\n```\n\nPython class."
    return f"**Imported value** `{stub.removesuffix(' = ...')}`\n\nPython value."


def _generic_hover(prog, pos: lsp.Position) -> Optional[lsp.Hover]:
    """Fallback hover when no named symbol is under the cursor.

    Drill into the smallest enclosing AST node so the hover range
    matches the statement / segment the cursor is on — *not* the
    entire enclosing procedure. Multi-line statements that end lines
    with ``\\`` are handled naturally because the parser already
    records their ``loc.end_line`` past the first physical line.
    """
    target = _smallest_enclosing(prog, pos)
    if target is None or target.loc is None:
        return None
    text = _describe_ast_node(target)
    if text is None:
        return None
    return lsp.Hover(
        contents=lsp.MarkupContent(kind=lsp.MarkupKind.Markdown, value=text),
        range=_span_to_range(target.loc),
    )


def _docstring_hover(
    prog: Program | ProgramWithValidation, pos: lsp.Position
) -> Optional[lsp.Hover]:
    for proc in _all_procedures(prog):
        if proc.docstring_loc is None or not _position_inside(pos, proc.docstring_loc):
            continue
        return lsp.Hover(
            contents=lsp.MarkupContent(
                kind=lsp.MarkupKind.Markdown,
                value=_proc_signature_markdown(proc),
            ),
            range=_span_to_range(proc.docstring_loc),
        )
    for profile in prog.profile_defs.values():
        if profile.docstring_loc is None or not _position_inside(pos, profile.docstring_loc):
            continue
        return lsp.Hover(
            contents=lsp.MarkupContent(
                kind=lsp.MarkupKind.Markdown,
                value=_profile_signature_markdown(profile),
            ),
            range=_span_to_range(profile.docstring_loc),
        )
    return None


def _position_in_comment(source: str, pos: lsp.Position) -> bool:
    tree = _ts_parse(source.encode("utf-8"))

    def visit(node: Node) -> bool:
        if node.type in {"line_comment", "block_comment"}:
            span = Span(
                start_byte=node.start_byte,
                end_byte=node.end_byte,
                start_line=node.start_point[0] + 1,
                start_col=node.start_point[1] + 1,
                end_line=node.end_point[0] + 1,
                end_col=node.end_point[1] + 1,
            )
            return _position_inside(pos, span)
        for child in node.children:
            if child.is_named and visit(child):
                return True
        return False

    return visit(tree.root_node)


def _smallest_enclosing(prog, pos: lsp.Position):
    """Walk the AST and return the deepest node whose ``loc`` contains ``pos``.

    Used by both ``compute_hover`` (fallback) and any future feature
    that wants "what's under the cursor" granularity. The walk is
    explicit per AST class to stay fast on large files — no generic
    dataclass reflection.
    """
    best = None

    def visit(node) -> None:
        nonlocal best
        if node is None:
            return
        loc = node.loc
        if loc is None or not _position_inside(pos, loc):
            return
        # Provisionally accept this node, then try children for a
        # tighter match.
        best = node
        # Procedure → body + optimize spans + auto spec
        if isinstance(node, ProcedureDef):
            for stmt in node.body:
                visit(stmt)
            if node.auto_spec is not None:
                visit(node.auto_spec)
            for span in node.optimize_spans or []:
                visit(span)
            return
        if isinstance(node, ProfileDefStmt):
            for member in node.agent_members:
                visit(member)
            for member in node.adapter_members:
                visit(member)
            for member in node.model_members:
                visit(member)
            for member in node.effort_members:
                visit(member)
            for member in node.approval_members:
                visit(member)
            for member in node.hooks_members:
                visit(member)
            for member in node.history_members:
                visit(member)
            for member in node.skills_members:
                visit(member)
            for member in node.codemode_members:
                visit(member)
            for member in node.system_members:
                visit(member)
            for member in node.mcp_members:
                visit(member)
            for member in node.settings_members:
                visit(member)
            for member in node.artifacts_members:
                visit(member)
            for member in node.output_members:
                visit(member)
            for member in node.subagent_members:
                visit(member)
            for member in node.max_agents_members:
                visit(member)
            for member in node.workflow_members:
                visit(member)
            return
        if isinstance(node, SystemDirectiveStmt):
            for line in node.template_lines:
                visit(line)
            return
        if isinstance(node, HistoryDirectiveStmt):
            for field in node.fields:
                visit(field)
            return
        if isinstance(node, HooksDirectiveStmt):
            for field in node.fields:
                visit(field)
            return
        if isinstance(node, SkillsDirectiveStmt):
            for field in node.fields:
                visit(field)
            return
        if isinstance(node, CodeModeDirectiveStmt):
            for field in node.fields:
                visit(field)
            return
        if isinstance(node, McpDirectiveStmt):
            for field in node.fields:
                visit(field)
            return
        if isinstance(node, SettingsDirectiveStmt):
            for field in node.fields:
                visit(field)
            return
        if isinstance(node, ArtifactsDirectiveStmt):
            for field in node.fields:
                visit(field)
            return
        if isinstance(node, ProfileMcpEntry):
            for field in node.fields:
                visit(field)
            return
        if isinstance(node, ProfileSettingsEntry):
            for field in node.fields:
                visit(field)
            return
        if isinstance(node, ProfileHistoryEntry):
            for field in node.fields:
                visit(field)
            return
        if isinstance(node, ProfileHooksEntry):
            for field in node.fields:
                visit(field)
            return
        if isinstance(node, ProfileSkillsEntry):
            for field in node.fields:
                visit(field)
            return
        if isinstance(node, ProfileCodeModeEntry):
            for field in node.fields:
                visit(field)
            return
        if isinstance(node, ProfileArtifactsEntry):
            for field in node.fields:
                visit(field)
            return
        if isinstance(node, ProfileSystemEntry):
            for line in node.template_lines:
                visit(line)
            return
        if isinstance(node, ProcDefStmt) and node.proc is not None:
            visit(node.proc)
            return
        if isinstance(node, IfStmt):
            for segment in node.condition_segments or []:
                visit(segment)
            for stmt in node.body:
                visit(stmt)
            for stmt in node.else_body or []:
                visit(stmt)
            return
        if isinstance(node, LoopStmt):
            for stmt in node.body:
                visit(stmt)
            for stmt in node.map_body or []:
                visit(stmt)
            return
        if isinstance(node, ConditionalLoopStmt):
            for segment in node.condition_segments or []:
                visit(segment)
            for stmt in node.body:
                visit(stmt)
            return
        if isinstance(node, OptimizeSpan):
            for tl in node.template_lines or []:
                visit(tl)
            return
        # Statements with segment children
        if isinstance(node, TemplateBlock):
            for line in node.lines:
                visit(line)
            return
        if isinstance(node, TemplateLine):
            for seg in node.segments:
                visit(seg)
            return
        if isinstance(node, ReturnStmt):
            for seg in node.segments:
                visit(seg)
            return
        if isinstance(node, (VariableInitStmt, AssignmentStmt)):
            for seg in node.expr:
                visit(seg)
            return
        # Type defs descend into fields
        if isinstance(node, TypeDef):
            for f in node.fields:
                visit(f)
            return
        # Validation suites (only present on ProgramWithValidation)
        if isinstance(node, TestSuite):
            for case in node.cases:
                visit(case)
            return
        if isinstance(node, EvalSuite):
            for ds in node.datasets.values():
                visit(ds)
            for ds in node.test_datasets.values():
                visit(ds)
            for m in node.metrics:
                visit(m)
            return
        # Segment containers: CallSegment may have nested args
        if isinstance(node, CallSegment):
            for arg in node.args:
                for inner in iter_call_arg_nested(arg):
                    visit(inner)
            return

    ordered_toplevel = prog.ordered_toplevel if hasattr(prog, "ordered_toplevel") else []
    if ordered_toplevel:
        for item in ordered_toplevel:
            visit(item)
    else:
        for proc in prog.procedures if hasattr(prog, "procedures") else []:
            visit(proc)
        toplevel_body = (
            prog.toplevel_body
            if hasattr(prog, "toplevel_body")
            else getattr(prog, "top_level_statements", [])
        )
        for stmt in toplevel_body or []:
            visit(stmt)
    for td in prog.type_defs if hasattr(prog, "type_defs") else []:
        visit(td)
    package = getattr(prog, "package", None)
    if package is not None:
        visit(package)
    toplevel_return = prog.toplevel_return if hasattr(prog, "toplevel_return") else None
    if toplevel_return is not None:
        visit(toplevel_return)
    if isinstance(prog, ProgramWithValidation):
        for s in prog.tests or []:
            visit(s)
        for s in prog.evals or []:
            visit(s)
    return best


def _describe_ast_node(node) -> Optional[str]:
    """Per-node hover description used by :func:`_generic_hover`."""
    if isinstance(node, ProcedureDef):
        return _proc_signature_markdown(node)
    if isinstance(node, TypeDef):
        return _type_signature_markdown(node)
    if isinstance(node, TypeField):
        ann = node.type_annotation or "str"
        default = f" = `{node.default_expr}`" if node.default_expr is not None else ""
        return f"**Type field** `{node.name}: {ann}{default}` — member of a Kedi type."
    if isinstance(node, PackageMetadata):
        lines = [f"**Kedi package** `{node.name}`"]
        if node.version is not None:
            lines.append(f"Version: `{node.version}`")
        if node.source is not None:
            lines.append(f"Source: `{node.source}`; `main.kedi` is the package root module.")
        if node.python_requirement is not None:
            lines.append(f"Python: `{node.python_requirement}`")
        if node.python_dependencies:
            lines.append("Python dependencies: `" + "`, `".join(node.python_dependencies) + "`")
        return "\n\n".join(lines)
    if isinstance(node, TemplateBlock):
        line_count = len(node.lines)
        return (
            f"**Template block** · {line_count} lines · one model call\n\n"
            "Lines are joined into a single prompt. Fields from this block become available "
            "after the call completes."
        )
    if isinstance(node, TemplateLine):
        return "**Template line**\n\nPart of the surrounding template block's single model call."
    if isinstance(node, RawInvokeStmt):
        return (
            f"**Raw model invoke** `{node.name}: str`\n\n"
            "Raw model response from the active adapter. No structured output schema is used."
        )
    if isinstance(node, VariableInitStmt):
        ann = node.type_annotation or "str"
        return f"`{node.name}: {ann}`\n\nVariable initialized in the current scope."
    if isinstance(node, VariableInitBlockStmt):
        ann = node.type_annotation or "str"
        return (
            f"`{node.name}: {ann}`\n\n"
            "The Python block result initializes this variable in the current scope."
        )
    if isinstance(node, AssignmentStmt):
        return (
            f"**Assignment** `{node.name}`\n\n"
            "Updates the nearest visible Kedi binding while preserving its declared type."
        )
    if isinstance(node, AssignmentBlockStmt):
        return (
            f"**Python assignment** `{node.name}`\n\n"
            "The Python block result updates the nearest visible Kedi binding."
        )
    if isinstance(node, ReturnStmt):
        return "**Return**\n\nFinal value emitted by this procedure or top-level program."
    if isinstance(node, ReturnBlockStmt):
        return (
            "**Python return**\n\n"
            "The Python value becomes the procedure or program result without string rendering."
        )
    if isinstance(node, IfStmt):
        if node.condition_segments is not None:
            return (
                "**Model-classified condition** `> if: claim`\n\n"
                "Renders the claim through Kedi template substitutions and continues only when "
                "the active adapter establishes it as true."
            )
        return (
            "**Conditional branch** ``> if: `condition`:``\n\n"
            "Evaluates the embedded Python expression once. It must return an exact `bool`; "
            "only the selected Kedi body executes."
        )
    if isinstance(node, LoopStmt):
        continuation = (
            " Its attached map stage continues each iteration in that iteration's isolated scope."
            if node.map_body is not None
            else ""
        )
        return (
            f"**Iterable loop** `{node.binder}`\n\n"
            "Evaluates the embedded Python expression once and requires an `Iterable`. "
            "Each iteration has an isolated child scope."
            f"{continuation}"
        )
    if isinstance(node, ConditionalLoopStmt):
        if node.condition_segments is not None:
            return (
                "**Model-classified loop** `> loop: claim`\n\n"
                "Re-renders and classifies the claim before every iteration. The loop stops when "
                "the claim cannot be established as true."
            )
        return (
            "**Conditional loop** ``> loop: `condition`:``\n\n"
            "Re-evaluates the exact-`bool` Python condition before every iteration."
        )
    if isinstance(node, BacktickLineStmt):
        return "**Inline Python statement** `` `…` `` — side-effecting Python line."
    if isinstance(node, PythonBlock):
        return "**Python block** — top-level ```python``` runs once as the module prelude."
    if isinstance(node, AutoSpec):
        return "**`> auto:`** — AI-generated procedure prompt; body is the spec."
    if isinstance(node, OptimizeSpan):
        return f"**`> optimize: {node.name}:`** — GEPA / DSPy optimisation target."
    if isinstance(node, ModelDirectiveStmt):
        return _model_directive_markdown(node)
    if isinstance(node, AgentDirectiveStmt):
        return _agent_directive_markdown(node)
    if isinstance(node, AdapterDirectiveStmt):
        return _adapter_directive_markdown(node)
    if isinstance(node, EffortDirectiveStmt):
        return _effort_directive_markdown(node)
    if isinstance(node, ApprovalDirectiveStmt):
        return _approval_directive_markdown(node)
    if isinstance(node, HooksDirectiveStmt):
        return _hooks_directive_markdown(node)
    if isinstance(node, HistoryDirectiveStmt):
        return _history_directive_markdown(node)
    if isinstance(node, SkillsDirectiveStmt):
        return _skills_directive_markdown(node)
    if isinstance(node, CodeModeDirectiveStmt):
        return _codemode_directive_markdown(node)
    if isinstance(node, SystemDirectiveStmt):
        return _system_directive_markdown(node)
    if isinstance(node, McpDirectiveStmt):
        return _mcp_directive_markdown(node)
    if isinstance(node, McpField):
        return _mcp_field_markdown(node)
    if isinstance(node, SettingsDirectiveStmt):
        return _settings_directive_markdown(node)
    if isinstance(node, SettingsField):
        return _settings_field_markdown(node)
    if isinstance(node, ArtifactsDirectiveStmt):
        return _artifacts_directive_markdown(node)
    if isinstance(node, ArtifactsField):
        return _artifacts_field_markdown(node)
    if isinstance(node, ProfileAgentEntry):
        return _profile_agent_entry_markdown(node)
    if isinstance(node, ProfileAdapterEntry):
        return _profile_adapter_entry_markdown(node)
    if isinstance(node, ProfileModelEntry):
        return _profile_model_entry_markdown(node)
    if isinstance(node, ProfileEffortEntry):
        return _profile_effort_entry_markdown(node)
    if isinstance(node, ProfileApprovalEntry):
        return _profile_approval_entry_markdown(node)
    if isinstance(node, ProfileHooksEntry):
        return _profile_hooks_entry_markdown(node)
    if isinstance(node, ProfileHistoryEntry):
        return _profile_history_entry_markdown(node)
    if isinstance(node, ProfileSkillsEntry):
        return _profile_skills_entry_markdown(node)
    if isinstance(node, ProfileCodeModeEntry):
        return _profile_codemode_entry_markdown(node)
    if isinstance(node, ProfileSystemEntry):
        return _profile_system_entry_markdown(node)
    if isinstance(node, ProfileMcpEntry):
        return _profile_mcp_entry_markdown(node)
    if isinstance(node, ProfileSettingsEntry):
        return _profile_settings_entry_markdown(node)
    if isinstance(node, ProfileArtifactsEntry):
        return _profile_artifacts_entry_markdown(node)
    if isinstance(node, ProfileOutputEntry):
        return (
            f"**Subagent output** `> output: {node.type_annotation}`\n\n"
            "Default structured result type used when `delegate_task` omits `final_schema`. "
            "An explicit delegation schema takes precedence."
        )
    if isinstance(node, ProfileMaxAgentsEntry):
        return (
            f"**Subagent budget** `> max_agents: {node.value}`\n\n"
            "Maximum descendant subagent invocations that this profile invocation may start."
        )
    if isinstance(node, ProfileWorkflowEntry):
        if node.mode == "dynamic":
            return (
                "**Dynamic subagent workflow** `> workflow: dynamic`\n\n"
                "Exposes one `run_workflow(code: str)` tool. Workflow code may call the "
                "profile's direct subagents concurrently and returns its final expression."
            )
        return (
            "**Delegated subagent workflow** `> workflow: delegate`\n\n"
            "Exposes the standard delegation and subagent lifecycle tools. This is the "
            "default when `> workflow:` is omitted."
        )
    if isinstance(node, UseDirectiveStmt):
        return _use_directive_markdown(node)
    if isinstance(node, ProfileDefStmt):
        return _profile_signature_markdown(node)
    if isinstance(node, TextSegment):
        text = (node.text or "").strip()
        preview = text[:40] + ("…" if len(text) > 40 else "")
        return f"**Text segment** — literal text in the prompt: `{preview}`."
    if isinstance(node, InputSegment):
        return f"**Input segment** `<{node.name}>` — substituted from a parameter or local."
    if isinstance(node, OutputSegment):
        ann = node.type_annotation or "str"
        return (
            f"`{node.name}: {ann}`\n\n"
            "Structured field generated by the containing model call. Available after the call completes."
        )
    if isinstance(node, CallSegment):
        return f"**Call segment** `<{node.name}(...)>` — invokes another procedure inline."
    if isinstance(node, PythonExprSegment):
        return (
            "**Inline Python expression**\n\n"
            "Evaluated in the current Kedi environment and inserted into the surrounding expression or prompt."
        )
    if isinstance(node, EvalSuite):
        return f"**`@eval: {node.procedure_name}:`** — evaluation block."
    if isinstance(node, TestSuite):
        return f"**`@test: {node.procedure_name}:`** — test block."
    if isinstance(node, TestCase):
        return f"**Test case** `{node.name}` — assertion body evaluated by its test suite."
    if isinstance(node, EvalDataset):
        return f"**Dataset** `> data: {node.name}` — input rows for evaluation."
    if isinstance(node, EvalMetric):
        return f"**Metric** `> metric: {node.name}` — scoring function for `@eval:`."
    return None


def _agent_tool_markdown(name: str, proc: ProcedureDef) -> str:
    return (
        f"**Agent tool** `> use: {name}`\n\n{_proc_signature_markdown(proc)}\n\n"
        "Available to agent calls in the current scope."
    )


def _use_directive_markdown(stmt: UseDirectiveStmt) -> str:
    if stmt.tool_names:
        tools = ", ".join(f"`{name}`" for name in stmt.tool_names)
        return (
            f"`> use:`\n\nRegisters Kedi procedures as agent tools: {tools}.\n\n"
            "Inner `> use:` directives override outer tools in the current scope."
        )
    return (
        f"`> use: {stmt.name}`\n\n"
        "Configures the current scope for subsequent agent calls. A procedure target is "
        "registered as an agent tool; a profile target applies its agent configuration."
    )


def _model_directive_markdown(stmt: ModelDirectiveStmt) -> str:
    value = _model_value_preview(stmt)
    return f"`> model:` {value}\n\nUses this model for subsequent agent calls in the current scope."


def _agent_directive_markdown(stmt: AgentDirectiveStmt) -> str:
    value = _adapter_value_preview(stmt)
    command = _agent_command_preview(stmt)
    if command is not None:
        return (
            f"`> agent:` {value} `:` {command}\n\n"
            "Uses this command to run the selected agent harness for subsequent agent calls "
            "in the current scope."
        )
    return (
        f"`> agent:` {value}\n\n"
        "Routes subsequent agent calls in the current scope through this agent harness."
    )


def _adapter_directive_markdown(stmt: AdapterDirectiveStmt) -> str:
    value = _adapter_value_preview(stmt)
    return (
        f"`> adapter:` {value}\n\n"
        "Uses this model framework for subsequent model calls in the current scope."
    )


def _effort_directive_markdown(stmt: EffortDirectiveStmt) -> str:
    value = _effort_value_preview(stmt)
    return (
        f"`> effort:` {value}\n\n"
        "Requests this reasoning effort for subsequent model calls in the current scope. "
        "Support depends on the active adapter and model."
    )


def _approval_directive_markdown(stmt: ApprovalDirectiveStmt) -> str:
    value = _approval_value_preview(stmt)
    return (
        f"`> approval:` {value}\n\n"
        "Controls registered tool calls in the current scope. `allow` permits mutating "
        "tools; `deny` refuses them. An inline Python callable may return an "
        "`ApprovalDecision` to allow, deny, or edit arguments."
    )


def _hooks_directive_markdown(stmt: HooksDirectiveStmt) -> str:
    if stmt.enabled is False:
        value = "disabled"
    elif stmt.fields:
        value = ", ".join(f"`{field.name}`" for field in stmt.fields)
    else:
        value = "enabled"
    return (
        f"`> hooks:` {value}\n\n"
        "Registers ordered agent lifecycle handlers in the current scope. "
        "Handlers may inspect prompt submission and tool execution, edit prompts or tool "
        "arguments before transport, deny requests, and observe tool success or failure."
    )


def _history_directive_markdown(stmt: HistoryDirectiveStmt) -> str:
    value = "enabled" if stmt.enabled else "disabled"
    compaction = next(
        (field for field in stmt.fields if field.name.startswith("compaction_")),
        None,
    )
    configured = (
        " The expanded form also configures provider-native compaction."
        if compaction is not None
        else ""
    )
    return (
        f"`> history: {value}`\n\n"
        "Controls conversation continuity through stateful history for subsequent model calls "
        "in this lexical scope. "
        "`enabled` reuses adapter-scoped model history and provider cache identity; `disabled` "
        "runs stateless calls without mutating the surrounding history. The default is disabled."
        f"{configured}"
    )


def _skills_directive_markdown(stmt: SkillsDirectiveStmt) -> str:
    value = "enabled" if stmt.enabled else "disabled"
    return (
        f"`> skills: {value}`\n\n"
        "Controls on-demand skill discovery for subsequent agent calls in this lexical scope. "
        "Enabled scopes expose `list_skills` and `read_skill`. Sources are checked in order: "
        "the Kedi registry, project `.agents/skills`, then user `.agents/skills`. The expanded "
        "form configures `cwd`, `max_skills`, source merging, and excluded paths."
    )


def _codemode_directive_markdown(stmt: CodeModeDirectiveStmt) -> str:
    value = "enabled" if stmt.enabled else "disabled"
    return (
        f"`> codemode: {value}`\n\n"
        "Controls progressive tool discovery and bounded Monty execution for subsequent agent "
        "calls in this lexical scope. Enabled scopes expose `search_tools`, `get_tool_schema`, "
        "and `execute_code`; the expanded form configures discovery, hydration, execution, "
        "result, concurrency, and timeout limits. The default is disabled."
    )


def _system_directive_markdown(stmt: SystemDirectiveStmt) -> str:
    value = _system_value_preview(stmt)
    return (
        f"**`> system:`** {value}\n\n"
        "System instructions for subsequent model calls in the current scope. "
        "Text and `<substitutions>` are evaluated when this directive is applied."
    )


def _mcp_directive_markdown(stmt: McpDirectiveStmt) -> str:
    fields = ", ".join(f"`{field.name}`" for field in stmt.fields) or "`<empty>`"
    return (
        "**`> mcp:`**\n\n"
        "Adds tools from this MCP server to agent calls in the current scope. "
        f"Configured fields: {fields}."
    )


def _mcp_field_markdown(field: McpField) -> str:
    value = _mcp_field_value_preview(field)
    descriptions = {
        "transport": (
            "Connection transport for this MCP server. `http` is an alias for `streamable-http`."
        ),
        "command": "Command used to start a stdio MCP server.",
        "args": "String argument list passed to the stdio MCP command.",
        "env": "String environment dictionary passed to the stdio MCP process.",
        "url": "HTTP/SSE MCP server URL.",
        "headers": "String header dictionary sent to HTTP/SSE MCP servers.",
    }
    desc = descriptions.get(field.name, "MCP server configuration field.")
    if field.name == "transport":
        return (
            f'`transport: Literal["stdio", "sse", "http", "streamable-http"] =` {value}\n\n{desc}'
        )
    return f"`{field.name}` = {value}\n\n{desc}"


def _settings_directive_markdown(stmt: SettingsDirectiveStmt) -> str:
    fields = ", ".join(f"`{field.name}`" for field in stmt.fields) or "`<empty>`"
    return (
        "**`> settings:`**\n\nSet active model configuration for subsequent "
        "agent calls in the current scope. Configured fields: "
        f"{fields}."
    )


def _settings_field_markdown(field: SettingsField) -> str:
    value = _settings_field_value_preview(field)
    hook_descriptions = {
        "user_prompt_submit": "Runs before the rendered user prompt crosses the model transport boundary.",
        "pre_tool_use": "Runs before approval and tool execution; it may edit or deny canonical arguments.",
        "post_tool_use": "Observes the final model-visible result after successful tool execution.",
        "post_tool_use_failure": "Observes execution failures, excluding hook and approval denials.",
    }
    if field.name in hook_descriptions:
        return f"`{field.name}` = {value}\n\n{hook_descriptions[field.name]}"
    skill_descriptions = {
        "max_skills": "Maximum number of skill names exposed by `list_skills` in this scope.",
        "include_registry": "Include the user Kedi skill registry as the highest-priority source.",
        "include_all": "Merge all configured skill sources instead of using the first non-empty source.",
        "exclude_paths": "Paths excluded from skill discovery and direct skill reads.",
    }
    if field.name in skill_descriptions:
        return f"`{field.name}` = {value}\n\n{skill_descriptions[field.name]}"
    codemode_descriptions = {
        "default_search_limit": "Default number of tool names returned by `search_tools`.",
        "max_search_limit": "Maximum page size accepted by `search_tools`.",
        "max_hydrated_tools": "Maximum exact tool schemas hydrated during one agent run.",
        "max_discovery_result_bytes": "Maximum serialized result size for discovery controls.",
        "max_code_chars": "Maximum Monty source length accepted by `execute_code`.",
        "max_nested_calls": "Maximum host tool calls made by one `execute_code` invocation.",
        "max_concurrent_calls": "Maximum concurrently active host tool calls.",
        "max_tool_result_bytes": "Maximum serialized size of one nested host tool result.",
        "max_total_tool_result_bytes": "Maximum aggregate nested tool-result size per execution.",
        "max_print_bytes": "Maximum captured standard output from one Monty execution.",
        "request_timeout": "Timeout in seconds for one nested host tool call.",
    }
    if field.name in codemode_descriptions:
        return f"`{field.name}` = {value}\n\n{codemode_descriptions[field.name]}"
    if field.name == "compaction_mode":
        return (
            f'`compaction_mode: Literal["native", "disabled"]` = {value}\n\n'
            "Select provider-native conversation compaction or disable the inherited "
            "compaction policy while retaining stateful history."
        )
    if field.name == "compaction_threshold":
        return (
            f"`compaction_threshold: int | None` = {value}\n\n"
            "Positive input-token threshold that triggers native compaction. When omitted, "
            "the provider default is used."
        )
    adapters = _setting_adapter_label(field.name)
    return f"`{field.name}` = {value}\n\nSupported by: {adapters}."


def _artifacts_directive_markdown(stmt: ArtifactsDirectiveStmt) -> str:
    fields = ", ".join(f"`{field.name}`" for field in stmt.fields) or "`<empty>`"
    return (
        "**`> artifacts:`**\n\nConfigure large-value storage for subsequent "
        "model calls in the current scope. Native Kedi and Python reads retain "
        f"the original logical value. Configured fields: {fields}."
    )


def _artifacts_field_markdown(field: ArtifactsField) -> str:
    value = _artifacts_field_value_preview(field)
    descriptions = {
        "enabled": "Enable artifact creation and agent-facing artifact tools.",
        "query_artifacts": (
            "Expose the bounded lexical `query_artifact` tool: `enabled` or `disabled`. "
            "Defaults to `disabled`; an explicit scoped value overrides the adapter default."
        ),
        "store": "Payload backend: `memory` or `file`.",
        "path": "Root directory used by the file artifact store.",
        "threshold": "Minimum payload size before a value becomes an artifact.",
        "ttl": "Maximum lifetime measured from artifact creation.",
        "idle_ttl": "Optional expiry duration measured from the last access.",
        "preview_chars": "Maximum model-visible preview length.",
        "read_max_chars": "Hard character limit for one `read_artifact` call.",
        "session_quota": "Maximum active payload bytes in this artifact session.",
        "max_artifacts": "Maximum active artifact count in this session.",
        "cleanup_interval": "Interval used by background expiry cleanup.",
    }
    return f"`{field.name}` = {value}\n\n{descriptions[field.name]}"


def _profile_mcp_entry_markdown(entry: ProfileMcpEntry) -> str:
    fields = ", ".join(f"`{field.name}`" for field in entry.fields) or "`<empty>`"
    return (
        f"**`> mcp:`**\n\nMCP server configuration stored in this profile. "
        f"Fields: {fields}.\n\nApplied when the profile is used."
    )


def _profile_settings_entry_markdown(entry: ProfileSettingsEntry) -> str:
    fields = ", ".join(f"`{field.name}`" for field in entry.fields) or "`<empty>`"
    return (
        f"**`> settings:`**\n\nModel configuration stored in this profile. "
        f"Fields: {fields}.\n\nApplied when the profile is used."
    )


def _profile_artifacts_entry_markdown(entry: ProfileArtifactsEntry) -> str:
    fields = ", ".join(f"`{field.name}`" for field in entry.fields) or "`<empty>`"
    return (
        "**`> artifacts:`**\n\nArtifact policy stored in this profile. "
        f"Fields: {fields}.\n\nApplied when the profile is used."
    )


def _profile_agent_entry_markdown(entry: ProfileAgentEntry) -> str:
    value = _adapter_value_preview(entry)
    command = _agent_command_preview(entry)
    if command is not None:
        return (
            f"`> agent:` {value} `:` {command}\n\n"
            "Agent harness stored in this profile. Applied when the profile is used."
        )
    return (
        f"`> agent:` {value}\n\n"
        "Agent harness stored in this profile. Applied when the profile is used."
    )


def _profile_adapter_entry_markdown(entry: ProfileAdapterEntry) -> str:
    value = _adapter_value_preview(entry)
    return (
        f"`> adapter:` {value}\n\n"
        "Model framework stored in this profile. Applied when the profile is used."
    )


def _profile_model_entry_markdown(entry: ProfileModelEntry) -> str:
    value = _model_value_preview(entry)
    return (
        f"`> model:` {value}\n\n"
        "Model override stored in this profile. Applied when the profile is used."
    )


def _profile_system_entry_markdown(entry: ProfileSystemEntry) -> str:
    value = _system_value_preview(entry)
    return (
        f"**`> system:`** {value}\n\n"
        "System instructions stored in this profile. Applied when the profile is used."
    )


def _profile_effort_entry_markdown(entry: ProfileEffortEntry) -> str:
    value = _effort_value_preview(entry)
    return (
        f"`> effort:` {value}\n\n"
        "Reasoning effort stored in this profile. Applied when the profile is used."
    )


def _profile_approval_entry_markdown(entry: ProfileApprovalEntry) -> str:
    value = _approval_value_preview(entry)
    return (
        f"`> approval:` {value}\n\n"
        "Tool-call approval policy stored in this profile. Applied when the profile is used."
    )


def _profile_hooks_entry_markdown(entry: ProfileHooksEntry) -> str:
    if entry.enabled is False:
        value = "disabled"
    elif entry.fields:
        value = ", ".join(f"`{field.name}`" for field in entry.fields)
    else:
        value = "enabled"
    return (
        f"`> hooks:` {value}\n\n"
        "Ordered agent lifecycle handlers stored in this profile. Applied when the profile is used."
    )


def _profile_history_entry_markdown(entry: ProfileHistoryEntry) -> str:
    value = "enabled" if entry.enabled else "disabled"
    configured = " Includes a compaction policy." if entry.fields else ""
    return (
        f"`> history: {value}`\n\n"
        "Conversation continuity policy stored in this profile. Applied when the profile is used."
        f"{configured}"
    )


def _profile_skills_entry_markdown(entry: ProfileSkillsEntry) -> str:
    value = "enabled" if entry.enabled else "disabled"
    return (
        f"`> skills: {value}`\n\n"
        "Skill discovery policy stored in this profile. Applied when the profile is used."
    )


def _profile_codemode_entry_markdown(entry: ProfileCodeModeEntry) -> str:
    value = "enabled" if entry.enabled else "disabled"
    configured = " Includes bounded execution settings." if entry.fields else ""
    return (
        f"`> codemode: {value}`\n\n"
        "CodeMode tool-discovery and execution policy stored in this profile. "
        f"Applied when the profile is used.{configured}"
    )


def _effort_value_preview(stmt: EffortDirectiveStmt | ProfileEffortEntry) -> str:
    if stmt.backtick_code is not None:
        return f"`{stmt.backtick_code}`"
    return f"`{stmt.plain_value or ''}`"


def _approval_value_preview(stmt: ApprovalDirectiveStmt | ProfileApprovalEntry) -> str:
    if stmt.backtick_code is not None:
        return f"`{stmt.backtick_code}`"
    return f"`{stmt.plain_value or ''}`"


def _model_value_preview(stmt: ModelDirectiveStmt | ProfileModelEntry) -> str:
    if stmt.backtick_code is not None:
        return f"`{stmt.backtick_code}`"
    return f"`{stmt.plain_value or ''}`"


def _adapter_value_preview(
    stmt: (AgentDirectiveStmt | AdapterDirectiveStmt | ProfileAgentEntry | ProfileAdapterEntry),
) -> str:
    if stmt.backtick_code is not None:
        return f"`{stmt.backtick_code}`"
    return f"`{stmt.plain_value or ''}`"


def _agent_command_preview(stmt: AgentDirectiveStmt | ProfileAgentEntry) -> str | None:
    if stmt.command_backtick_code is not None:
        return f"`{stmt.command_backtick_code}`"
    if stmt.command_plain_value is not None:
        return f"`{stmt.command_plain_value}`"
    return None


def _system_value_preview(stmt: SystemDirectiveStmt | ProfileSystemEntry) -> str:
    if stmt.template_lines:
        parts: list[str] = []
        for index, line in enumerate(stmt.template_lines):
            if index:
                parts.append("\\n")
            for seg in line.segments:
                if isinstance(seg, TextSegment):
                    parts.append(seg.text)
                elif isinstance(seg, InputSegment):
                    parts.append(f"<{seg.name}>")
        text = "".join(parts).strip()
        return f"`{text}`" if text else "`<empty>`"
    if stmt.backtick_code is not None:
        return f"`{stmt.backtick_code}`"
    return f"`{stmt.plain_value or ''}`"


def _mcp_field_value_preview(field: McpField) -> str:
    if field.backtick_code is not None:
        return f"`{field.backtick_code}`"
    return f"`{field.plain_value or ''}`"


def _settings_field_value_preview(field: SettingsField) -> str:
    if field.backtick_code is not None:
        return f"`{field.backtick_code}`"
    return f"`{field.plain_value or ''}`"


def _artifacts_field_value_preview(field: ArtifactsField) -> str:
    if field.backtick_code is not None:
        return f"`{field.backtick_code}`"
    return f"`{field.plain_value or ''}`"


def _setting_adapter_label(name: str) -> str:
    labels: list[str] = []
    if name in PYDANTIC_MODEL_SETTING_KEYS:
        labels.append("Pydantic AI `ModelSettings`")
    if name in DSPY_LM_SETTING_KEYS:
        labels.append("DSPy `LM`")
    if name in LANGCHAIN_MODEL_SETTING_KEYS:
        labels.append("LangChain model settings")
    if name in CLAUDE_AGENT_SETTING_KEYS:
        labels.append("Claude Agent SDK settings")
    if name in CODEX_AGENT_SETTING_KEYS:
        labels.append("Codex App Server settings")
    if name in ACP_AGENT_SETTING_KEYS:
        labels.append("ACP agent settings")
    if len(labels) == 1:
        return f"{labels[0]} only"
    if labels:
        return ", ".join(labels[:-1]) + f", and {labels[-1]}"
    expected = ", ".join(sorted(MODEL_SETTING_KEYS))
    return f"no adapter; valid keys are {expected}"


def _missing_agent_profile_hover(name: str, imported_surface: ImportedSurface) -> str:
    module_info = imported_surface.modules.get(name)
    if module_info is not None:
        from .module_surface import load_imported_surface

        exported = sorted(load_imported_surface(module_info.path, set(), name).profiles.keys())
        if exported:
            options = ", ".join(f"`> use: {profile}`" for profile in exported)
            return (
                f"**Agent profile** `{name}` — not found. "
                f"`{name}` is an imported **module**, not a profile name. "
                f"Exported profiles from that module: {options}."
            )
        return (
            f"**Agent profile** `{name}` — not found. "
            f"`{name}` is an imported module with no exported agent profiles."
        )
    return f"**Agent profile** `{name}` — referenced by `> use:`; definition not found."


def _profile_signature_markdown(profile: ProfileDefStmt) -> str:
    model_parts: list[str] = []
    for member in profile.model_members:
        if member.backtick_code is not None:
            model_parts.append(f"`{member.backtick_code}`")
        elif member.plain_value is not None:
            model_parts.append(f"`{member.plain_value}`")
    effort_parts: list[str] = []
    for member in profile.effort_members:
        effort_parts.append(_effort_value_preview(member))
    summary: list[str] = []
    if model_parts:
        summary.append(f"Model: {', '.join(model_parts)}")
    if effort_parts:
        summary.append(f"Effort: {', '.join(effort_parts)}")
    if any(
        member.template_lines or member.backtick_code is not None or member.plain_value is not None
        for member in profile.system_members
    ):
        summary.append("System instructions: configured")
    explicit_tools = (*profile.use_targets, *profile.tool_names)
    if explicit_tools:
        targets = ", ".join(f"`{name}`" for name in explicit_tools)
        summary.append(f"Uses: {targets}")
    if profile.mcp_members:
        summary.append(f"MCP servers: {len(profile.mcp_members)}")
    setting_count = sum(len(member.fields) for member in profile.settings_members)
    if setting_count:
        summary.append(f"Settings: {setting_count}")
    artifact_field_count = sum(len(member.fields) for member in profile.artifacts_members)
    if artifact_field_count:
        summary.append(f"Artifact settings: {artifact_field_count}")
    if profile.output_members:
        summary.append(f"Subagent output: `{profile.output_members[-1].type_annotation}`")
    workflow_mode = profile.workflow_members[-1].mode if profile.workflow_members else "delegate"
    summary.append(f"Subagent workflow: `{workflow_mode}`")

    text = f"**Agent profile** `> profile: {profile.name}:`"
    if profile.docstring:
        text += f"\n\n{profile.docstring}"
    else:
        text += "\n\nReusable agent profile."
    text += "\n\n" + "\n\n".join(summary)
    return text + "\n\nApplied when `> use:` runs."


def _proc_signature_markdown(proc: ProcedureDef) -> str:
    params = ", ".join(_format_proc_param(proc, p) for p in proc.params)
    sig = f"`@{proc.name}({params})`"
    if proc.return_type_annotation:
        sig += f" → `{proc.return_type_annotation}`"
    desc = "Kedi procedure."
    if proc.auto_spec is not None:
        desc = "AI-generated procedure (`> auto:`)."
    elif proc.optimize_spans:
        desc = (
            "Procedure with optimization spans: "
            + ", ".join(s.name for s in proc.optimize_spans)
            + "."
        )
    elif proc.captured_agent_profile is not None:
        captured = proc.captured_agent_profile
        profile_details: list[str] = []
        if captured.model is not None:
            profile_details.append(f"Model in this scope: `{captured.model}`.")
        if captured.effort is not None:
            profile_details.append(f"Reasoning effort in this scope: `{captured.effort}`.")
        if profile_details:
            desc = "Kedi procedure.\n\n" + "\n\n".join(profile_details)
    if proc.docstring:
        return f"{sig}\n\n{desc}\n\n{proc.docstring}"
    return f"{sig}\n\n{desc}"


def _type_signature_markdown(td: TypeDef) -> str:
    fields = ", ".join(_format_type_field(f) for f in td.fields)
    return (
        f"`~{td.name}({fields})`\n\n"
        "Structured Kedi type definition used for validation and output schema generation."
    )


def _type_expr_to_str(expr: TypeExpr | None) -> str:
    if expr is None:
        return "None"
    if isinstance(expr, TypeRef):
        return expr.name
    if isinstance(expr, TypeApply):
        return f"{expr.name}[{', '.join(_type_expr_to_str(arg) for arg in expr.args)}]"
    if isinstance(expr, TypeUnion):
        return f"{_type_expr_to_str(expr.left)} | {_type_expr_to_str(expr.right)}"
    return repr(expr)


def _format_proc_param(proc: ProcedureDef, name: str) -> str:
    piece = f"{name}: {_proc_param_annotation(proc, name)}"
    default = (proc.param_defaults or {}).get(name)
    if default is not None:
        piece += f" = `{default}`"
    return piece


def _format_type_field(field: TypeField) -> str:
    piece = f"{field.name}: {_type_field_annotation(field)}"
    if field.default_expr is not None:
        piece += f" = `{field.default_expr}`"
    return piece


def _proc_param_annotation(proc: ProcedureDef, name: str) -> str:
    return ((proc.param_type_annotations or {}).get(name) or "str").strip()


def _type_field_annotation(field: TypeField) -> str:
    return (field.type_annotation or "str").strip()


def _known_type_markdown(name: str, origin: str) -> str:
    if name == "Annotated":
        return (
            '`Annotated[T, "description"]`\n\n'
            "Adds field documentation without changing the runtime type `T`. The description "
            "is included in generated schemas and shown to compatible model adapters.\n\n"
            'Example: `Annotated[str, "A valid ISO country code"]`'
        )
    if origin == "datetime":
        descriptions = {
            "date": "Calendar date serialized as `YYYY-MM-DD`.",
            "time": "Time of day serialized in ISO 8601 format.",
            "datetime": "Date and time serialized in ISO 8601 format.",
            "timedelta": "Duration represented by Python's `datetime.timedelta`.",
        }
        return f"`{name}`\n\n{descriptions.get(name, f'Python datetime.{name} type.')}"
    if origin == "re":
        return "`Regex`\n\nCompiled regular-expression pattern."
    if origin == "pydantic":
        descriptions = {
            "Email": "Validated email address.",
            "HttpUrl": "Validated HTTP or HTTPS URL.",
            "FileUrl": "Validated file URL.",
        }
        return f"`{name}`\n\n{descriptions.get(name, 'Pydantic-validated value.')}"
    return f"`{name}`\n\nKedi type resolved to Python's {origin}.{name}."


def _prelude_type_markdown(name: str) -> str:
    return f"`{name}`\n\nPython type defined in this module's prelude and available to Kedi annotations."


def _python_module_scope_type_markdown(name: str) -> str:
    return (
        f"`{name}`\n\nPython type defined in a top-level Python block and available "
        "to Kedi annotations."
    )


def _prelude_defined_names(code: str | None) -> set[str]:
    if not code:
        return set()
    try:
        module = _py_ast.parse(code)
    except SyntaxError:
        return set()

    names: set[str] = set()
    for stmt in module.body:
        if isinstance(stmt, (_py_ast.ClassDef, _py_ast.FunctionDef, _py_ast.AsyncFunctionDef)):
            names.add(stmt.name)
        elif isinstance(stmt, (_py_ast.Import, _py_ast.ImportFrom)):
            for alias in stmt.names:
                names.add(alias.asname or alias.name.split(".")[-1])
        elif isinstance(stmt, (_py_ast.Assign, _py_ast.AnnAssign)):
            targets = stmt.targets if isinstance(stmt, _py_ast.Assign) else [stmt.target]
            for target in targets:
                if isinstance(target, _py_ast.Name):
                    names.add(target.id)
    return names


def _module_scope_python_names(prog: Program | ProgramWithValidation) -> set[str]:
    names: set[str] = set()
    for stmt in prog.toplevel_body:
        if not isinstance(stmt, PythonBlock):
            break
        names.update(_prelude_defined_names(stmt.code))
    return names


# =====================================================================
# Completion
# =====================================================================


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


# =====================================================================
# Definition
# =====================================================================


def _location_for_span(uri: str, source_path: Optional[str], span: Span) -> lsp.Location:
    target_uri = uri
    if span.source_path:
        span_path = Path(span.source_path).resolve()
        if source_path is None or span_path != Path(source_path).resolve():
            target_uri = span_path.as_uri()
    return lsp.Location(uri=target_uri, range=_span_to_range(span))


@with_source_coordinates(2)
def compute_definition(
    source: str, uri: str, source_path: Optional[str], pos: lsp.Position
) -> Optional[lsp.Location]:
    try:
        prog = parse_program(source, source_path=source_path)
    except KediParseError:
        return None
    sym = _resolve_symbol_at(prog, pos)
    if sym is None or sym.declaration_span is None:
        return None
    return _location_for_span(uri, source_path, sym.declaration_span)


# =====================================================================
# References
# =====================================================================


@with_source_coordinates(2)
def compute_references(
    source: str,
    uri: str,
    source_path: Optional[str],
    pos: lsp.Position,
    include_declaration: bool,
) -> List[lsp.Location]:
    try:
        prog = parse_program(source, source_path=source_path)
    except KediParseError:
        return []
    sym = _resolve_symbol_at(prog, pos)
    if sym is None:
        return []
    spans = _find_all_references(prog, sym, include_declaration=include_declaration)
    return [_location_for_span(uri, source_path, span) for span in spans]


def _find_all_references(
    prog: Union[Program, ProgramWithValidation],
    sym: SymbolRef,
    *,
    include_declaration: bool,
) -> List[Span]:
    out: List[Span] = []
    if include_declaration and sym.declaration_span is not None:
        out.append(_declaration_identifier_span(prog, sym))

    if sym.kind in ("procedure", "agent_tool"):
        procedure_candidates: List[Span] = []
        for profile in prog.profile_defs.values():
            for tool_name, tool_span in _profile_use_entry_spans(prog, profile):
                if tool_name == sym.name:
                    procedure_candidates.append(tool_span)
        for suite in _validation_suites(prog):
            if suite.procedure_name == sym.name:
                suite_span = _validation_procedure_span(prog, suite)
                if suite_span is not None:
                    procedure_candidates.append(suite_span)
        for p in _all_procedures(prog):
            for stmt in _iter_control_statements(p.body):
                for seg in _segments_of(stmt):
                    _collect_call_refs(seg, sym.name, procedure_candidates)
        for stmt in _iter_control_statements(prog.toplevel_body):
            for seg in _segments_of(stmt):
                _collect_call_refs(seg, sym.name, procedure_candidates)
        if prog.toplevel_return is not None:
            for seg in _segments_of(prog.toplevel_return):
                _collect_call_refs(seg, sym.name, procedure_candidates)
        out.extend(
            candidate
            for candidate in procedure_candidates
            if _reference_resolves_to(prog, candidate, sym)
        )
    elif sym.kind == "type":
        type_candidates: List[Span] = []
        for td in _all_type_defs(prog):
            _collect_type_refs_in_type_def(td, sym.name, type_candidates)
        for profile in prog.profile_defs.values():
            for member in profile.output_members:
                _collect_type_refs_in_expr(member.type_expr, sym.name, type_candidates)
        for stmt in prog.toplevel_body:
            _collect_type_refs_in_stmt(stmt, sym.name, type_candidates)
        for p in _all_procedures(prog):
            if p.return_type_expr is not None:
                _collect_type_refs_in_expr(p.return_type_expr, sym.name, type_candidates)
            for v in (p.param_type_exprs or {}).values():
                if v is not None:
                    _collect_type_refs_in_expr(v, sym.name, type_candidates)
            for stmt in p.body:
                _collect_type_refs_in_stmt(stmt, sym.name, type_candidates)
        out.extend(
            candidate
            for candidate in type_candidates
            if _reference_resolves_to(prog, candidate, sym)
        )
    elif sym.kind == "agent_profile":
        for profile in prog.profile_defs.values():
            for target_name, target_span in _profile_use_entry_spans(prog, profile):
                if target_name == sym.name:
                    out.append(target_span)
            for member in profile.subagent_members:
                if member.name == sym.name and member.loc is not None:
                    out.append(member.loc)
        for stmt in prog.ordered_toplevel:
            if isinstance(stmt, UseDirectiveStmt) and stmt.name == sym.name and not stmt.tool_names:
                target_span = _use_single_target_span(prog, stmt)
                if target_span is not None:
                    out.append(target_span)
    elif sym.kind in {
        "parameter",
        "param_decl",
        "local",
        "output_decl",
        "loop_binder",
        "module_value",
        "imported_value",
    }:
        value_candidates: list[Span] = []
        _collect_value_reference_candidates(prog.toplevel_body, sym.name, value_candidates)
        if prog.toplevel_return is not None:
            for segment in _segments_of(prog.toplevel_return):
                _collect_input_refs(segment, sym.name, value_candidates)
        for procedure in _all_procedures(prog):
            _collect_value_reference_candidates(procedure.body, sym.name, value_candidates)
        out.extend(
            candidate
            for candidate in value_candidates
            if _reference_resolves_to(prog, candidate, sym)
        )
    # Deduplicate spans by (start_byte, end_byte) — extra-safe.
    seen: set = set()
    uniq: List[Span] = []
    for s in out:
        key = (s.start_byte, s.end_byte, s.start_line, s.start_col)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(s)
    return uniq


def _declaration_identifier_span(
    prog: Program | ProgramWithValidation,
    sym: SymbolRef,
) -> Span:
    assert sym.declaration_span is not None
    if current_source() is None:
        return sym.declaration_span
    if sym.kind in ("procedure", "agent_tool"):
        proc = _procedure_for_declaration(prog, sym.declaration_span)
        if proc is not None:
            return _proc_name_span(proc)
        return name_token_span(sym.declaration_span, sym.name, prefix_len=1)
    if sym.kind == "type":
        return name_token_span(sym.declaration_span, sym.name, prefix_len=1)
    if sym.kind == "type_field":
        return name_token_span(sym.declaration_span, sym.name, prefix_len=0)
    return sym.declaration_span


def _reference_resolves_to(
    prog: Program | ProgramWithValidation,
    candidate: Span,
    target: SymbolRef,
) -> bool:
    if current_source() is None:
        return True
    resolved = _resolve_symbol_at(
        prog,
        lsp.Position(line=candidate.start_line - 1, character=candidate.start_col - 1),
    )
    if resolved is None:  # pragma: no cover - collectors only emit resolvable syntax spans
        return False
    procedure_kinds = {"procedure", "agent_tool"}
    parameter_kinds = {"parameter", "param_decl"}
    same_kind = (
        resolved.kind == target.kind
        or resolved.kind in procedure_kinds
        and target.kind in procedure_kinds
        or resolved.kind in parameter_kinds
        and target.kind in parameter_kinds
    )
    return same_kind and resolved.declaration_span == target.declaration_span


def _collect_literal_arg_refs(arg: CallArg, name: str, out: List[Span], collector) -> None:
    for nested in iter_call_arg_nested(arg):
        collector(nested, name, out)


def _collect_call_refs(seg, name: str, out: List[Span]) -> None:
    if isinstance(seg, CallSegment):
        if seg.name == name and seg.loc is not None:
            out.append(name_token_span(seg.loc, seg.name, prefix_len=1))
        for arg in seg.args:
            _collect_literal_arg_refs(arg, name, out, _collect_call_refs)


def _collect_input_refs(seg, name: str, out: List[Span]) -> None:
    """Collect matching template substitutions, including nested call arguments."""
    if isinstance(seg, InputSegment) and seg.name == name and seg.loc is not None:
        out.append(name_token_span(seg.loc, seg.name, prefix_len=1))
    elif isinstance(seg, CallSegment):
        for arg in seg.args:
            _collect_literal_arg_refs(arg, name, out, _collect_input_refs)


def _collect_value_reference_candidates(
    statements: Sequence[object],
    name: str,
    out: List[Span],
) -> None:
    for stmt in _iter_control_statements(statements):
        if isinstance(stmt, (AssignmentStmt, AssignmentBlockStmt)) and stmt.name == name:
            target_span = binding_name_span(stmt)
            if target_span is not None:
                out.append(target_span)
        for segment in _segments_of(stmt):
            _collect_input_refs(segment, name, out)


def _procedure_scope_entries(
    prog: Union[Program, ProgramWithValidation],
) -> list[tuple[ProcedureDef, tuple[ProcedureDef, ...]]]:
    entries: list[tuple[ProcedureDef, tuple[ProcedureDef, ...]]] = []

    def visit(proc: ProcedureDef, ancestors: tuple[ProcedureDef, ...]) -> None:
        entries.append((proc, ancestors))
        for child in _nested_procedures(proc):
            visit(child, ancestors + (proc,))

    for proc in prog.procedures:
        visit(proc, ())
    return entries


def _collect_lexical_input_refs(
    seg,
    name: str,
    proc: ProcedureDef,
    ancestors: tuple[ProcedureDef, ...],
    owner: ProcedureDef,
    declaration_span: Optional[Span],
    out: List[Span],
) -> None:
    if isinstance(seg, InputSegment) and seg.name == name and seg.loc is not None:
        resolved = _resolve_lexical_name(proc, ancestors, name)
        if resolved is not None and resolved[2] is owner and resolved[1] == declaration_span:
            out.append(name_token_span(seg.loc, seg.name, prefix_len=1))
    elif isinstance(seg, CallSegment):
        for arg in seg.args:
            for nested in iter_call_arg_nested(arg):
                _collect_lexical_input_refs(
                    nested,
                    name,
                    proc,
                    ancestors,
                    owner,
                    declaration_span,
                    out,
                )


def _collect_loop_binder_refs(
    prog: Program | ProgramWithValidation,
    target: SymbolRef,
    out: List[Span],
) -> None:
    def visit(statements, active: bool = False) -> None:
        for stmt in statements:
            if isinstance(stmt, LoopStmt):
                is_target = stmt.binder_loc == target.declaration_span
                if is_target:
                    visit(stmt.body, True)
                    visit(stmt.map_body or [], True)
                elif not (active and stmt.binder == target.name):
                    visit(stmt.body, active)
                    visit(stmt.map_body or [], active)
                continue

            if active:
                for segment in _segments_of(stmt):
                    _collect_input_refs(segment, target.name, out)
            if isinstance(stmt, IfStmt):
                visit(stmt.body, active)
                visit(stmt.else_body or [], active)
            elif isinstance(stmt, ConditionalLoopStmt):
                visit(stmt.body, active)

    visit(prog.toplevel_body)
    for proc in _all_procedures(prog):
        visit(proc.body)


def _collect_type_refs_in_stmt(stmt, name: str, out: List[Span]) -> None:
    if isinstance(stmt, (VariableInitStmt, VariableInitBlockStmt)) and stmt.type_expr is not None:
        _collect_type_refs_in_expr(stmt.type_expr, name, out)
    for seg in _segments_of(stmt):
        if isinstance(seg, OutputSegment) and seg.type_expr is not None:
            _collect_type_refs_in_expr(seg.type_expr, name, out)
    for child in _control_body_statements(stmt):
        _collect_type_refs_in_stmt(child, name, out)


def _collect_type_refs_in_type_def(td: TypeDef, name: str, out: List[Span]) -> None:
    for type_field in td.fields:
        _collect_type_refs_in_expr(type_field.type_expr, name, out)


def _collect_type_refs_in_expr(expr: Optional[TypeExpr], name: str, out: List[Span]) -> None:
    if expr is None:
        return
    if isinstance(expr, TypeRef) and expr.name == name and expr.loc is not None:
        out.append(expr.loc)
    elif isinstance(expr, TypeApply):
        if expr.name == name and expr.loc is not None:
            out.append(name_token_span(expr.loc, expr.name, prefix_len=0))
        for arg in expr.args:
            _collect_type_refs_in_expr(arg, name, out)
    elif isinstance(expr, TypeUnion):
        _collect_type_refs_in_expr(expr.left, name, out)
        _collect_type_refs_in_expr(expr.right, name, out)


# =====================================================================
# Rename
# =====================================================================


@with_source_coordinates(1)
def compute_prepare_rename(
    source: str, source_path: Optional[str], pos: lsp.Position
) -> Optional[lsp.Range]:
    """Return the precise identifier range for the editor's rename popup.

    Only renameable kinds are returned; everything else is ``None``
    (which tells the editor "can't rename here").
    """
    try:
        prog = parse_program(source, source_path=source_path)
    except KediParseError:
        return None
    sym = _resolve_symbol_at(prog, pos)
    if sym is None:
        return None
    if (
        sym.kind == "type"
        and sym.declaration_span is None
        and _known_lsp_type_origin(sym.name) is not None
    ):
        return None
    if (
        source_path is not None
        and sym.declaration_span is not None
        and sym.declaration_span.source_path is not None
        and Path(sym.declaration_span.source_path).resolve() != Path(source_path).resolve()
    ):
        return None
    if sym.kind not in (
        "procedure",
        "agent_tool",
        "type",
        "type_field",
        "parameter",
        "param_decl",
        "local",
        "loop_binder",
        "output_decl",
        "module_value",
        "imported_value",
    ):
        return None
    return _span_to_range(sym.cursor_span)


@with_source_coordinates(2)
def compute_rename(
    source: str,
    uri: str,
    source_path: Optional[str],
    pos: lsp.Position,
    new_name: str,
) -> Optional[lsp.WorkspaceEdit]:
    try:
        prog = parse_program(source, source_path=source_path)
    except KediParseError:
        return None
    sym = _resolve_symbol_at(prog, pos)
    if sym is None:
        return None
    if (
        sym.kind == "type"
        and sym.declaration_span is None
        and _known_lsp_type_origin(sym.name) is not None
    ):
        return None
    if (
        source_path is not None
        and sym.declaration_span is not None
        and sym.declaration_span.source_path is not None
        and Path(sym.declaration_span.source_path).resolve() != Path(source_path).resolve()
    ):
        return None
    # Reject non-identifier new names.
    if not _is_identifier(new_name):
        return None
    spans = _find_all_references(prog, sym, include_declaration=True)
    # For procedure / type / output declarations we also need to
    # rewrite the declaration identifier itself, not the whole declaring
    # node — synthesize a name-only span if needed.
    edits: List[lsp.TextEdit] = []
    for s in spans:
        edits.append(lsp.TextEdit(range=_span_to_range(s), new_text=new_name))
    return lsp.WorkspaceEdit(changes={uri: edits})


def _is_identifier(s: str) -> bool:
    import re

    return bool(re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", s))


# =====================================================================
# Formatting
# =====================================================================


def compute_formatting(source: str) -> List[lsp.TextEdit]:
    """Conservative: trim trailing WS per line, single trailing newline."""
    lines = source.splitlines(keepends=False)
    cleaned = [ln.rstrip(" \t") for ln in lines]
    # Drop trailing empty lines, then ensure exactly one final newline.
    while cleaned and cleaned[-1] == "":
        cleaned.pop()
    formatted = "\n".join(cleaned) + "\n" if cleaned else ""

    if formatted == source:
        return []

    # Compute end position of the original document.
    end_line = source.count("\n")
    last_line_start = source.rfind("\n") + 1
    end_char = len(source) - last_line_start
    return [
        lsp.TextEdit(
            range=lsp.Range(
                start=lsp.Position(line=0, character=0),
                end=lsp.Position(line=end_line, character=end_char),
            ),
            new_text=formatted,
        )
    ]


# =====================================================================
# Signature help
# =====================================================================


@with_source_coordinates(1)
def compute_signature_help(
    source: str, source_path: Optional[str], pos: lsp.Position
) -> Optional[lsp.SignatureHelp]:
    """Provide parameter-list info while typing inside ``<func(...)>``."""
    try:
        prog = parse_program(source, source_path=source_path)
    except KediParseError:
        return None
    imported_surface = imported_surface_for_program(prog)
    proc_by_name = {p.name: p for p in _all_procedures(prog)}
    proc_by_name.update(imported_surface.procedures)

    # Find the innermost call_segment containing the cursor.
    target_call: Optional[CallSegment] = None
    for proc in _all_procedures(prog):
        if proc.loc is None or not _position_inside(pos, proc.loc):
            continue
        for stmt in proc.body:
            call = _innermost_call_at(stmt, pos)
            if call is not None:
                target_call = call
    for stmt in prog.toplevel_body:
        call = _innermost_call_at(stmt, pos)
        if call is not None:
            target_call = call
    if prog.toplevel_return is not None:
        call = _innermost_call_at(prog.toplevel_return, pos)
        if call is not None:
            target_call = call

    if target_call is None:
        return None
    target = proc_by_name.get(target_call.name)
    if target is None or target.loc is None:
        return None

    # Build signature label and parameter ranges.
    params_label_parts: List[str] = []
    param_infos: List[lsp.ParameterInformation] = []
    cursor_byte_offset = 0  # within the label string
    label_prefix = f"@{target.name}("
    cursor_byte_offset = len(label_prefix)
    for i, pname in enumerate(target.params):
        piece = _format_proc_param(target, pname)
        params_label_parts.append(piece)
        start = cursor_byte_offset
        end = cursor_byte_offset + len(piece)
        param_infos.append(lsp.ParameterInformation(label=(start, end)))
        cursor_byte_offset = end + 2  # ", "
    label = f"{label_prefix}{', '.join(params_label_parts)})"
    if target.return_type_annotation:
        label += f" -> {target.return_type_annotation}"

    active_param = _active_param_index(target_call, pos)
    return lsp.SignatureHelp(
        signatures=[
            lsp.SignatureInformation(
                label=label,
                parameters=param_infos,
                active_parameter=active_param,
            )
        ],
        active_signature=0,
        active_parameter=active_param,
    )


def _innermost_call_at(stmt, pos: lsp.Position) -> Optional[CallSegment]:
    if not hasattr(stmt, "loc") or stmt.loc is None or not _position_inside(pos, stmt.loc):
        return None
    found: Optional[CallSegment] = None
    for seg in _segments_of(stmt):
        c = _innermost_call_in_segment(seg, pos)
        if c is not None:
            found = c
    return found


def _innermost_call_in_segment(seg, pos: lsp.Position) -> Optional[CallSegment]:
    if not hasattr(seg, "loc") or seg.loc is None or not _position_inside(pos, seg.loc):
        return None
    if isinstance(seg, CallSegment):
        # Recurse into nested calls first to find the innermost.
        deepest: Optional[CallSegment] = seg
        for arg in seg.args:
            for inner in iter_call_arg_nested(arg):
                r = _innermost_call_in_segment(inner, pos)
                if r is not None:
                    deepest = r
        return deepest
    return None


def _active_param_index(call: CallSegment, pos: lsp.Position) -> int:
    """Approximate: count commas BEFORE the cursor between the call's parens."""
    # The call spans <name(arg1, arg2)>. We don't have per-arg fine-
    # grained spans, so count commas in the source slice from after the
    # opening `(` to the cursor.
    # The position byte we'd need to map; we just clamp at param count.
    if call.args is None:
        return 0
    # Best effort: index of the arg whose span contains the cursor.
    for i, arg in enumerate(call.args):
        loc = arg.loc
        if loc is not None and _position_inside(pos, loc):
            return i
    # Past the end — last param.
    return max(0, len(call.args) - 1)


# =====================================================================
# Inlay hints
# =====================================================================


@with_source_coordinates(1)
def compute_inlay_hints(
    source: str, source_path: Optional[str], rng: lsp.Range
) -> List[lsp.InlayHint]:
    try:
        prog = parse_program(source, source_path=source_path)
    except KediParseError:
        return []
    out: List[lsp.InlayHint] = []
    proc_by_name = {p.name: p for p in _all_procedures(prog)}

    def visit_stmt(stmt, enclosing_proc: Optional[ProcedureDef]) -> None:
        for seg in _segments_of(stmt):
            visit_segment(seg, enclosing_proc)

    def visit_segment(seg, enclosing_proc: Optional[ProcedureDef]) -> None:
        if isinstance(seg, CallSegment):
            target = proc_by_name.get(seg.name)
            if target is not None and seg.args and target.params:
                for i, arg in enumerate(seg.args):
                    if i >= len(target.params):
                        break
                    loc = arg.loc
                    if loc is None or not _in_range(loc, rng):
                        continue
                    out.append(
                        lsp.InlayHint(
                            position=_span_to_range(loc).start,
                            label=f"{target.params[i]}: ",
                            kind=lsp.InlayHintKind.Parameter,
                            padding_right=False,
                            padding_left=False,
                        )
                    )
            for arg in seg.args:
                for inner in iter_call_arg_nested(arg):
                    visit_segment(inner, enclosing_proc)
        elif isinstance(seg, OutputSegment):
            if seg.type_expr is None and seg.type_annotation is None and seg.loc is not None:
                if not _in_range(seg.loc, rng):
                    return
                # Position the hint AT the closing `]` so it reads as
                # `[name: str]` visually.
                segment_range = _span_to_range(seg.loc)
                out.append(
                    lsp.InlayHint(
                        position=lsp.Position(
                            line=segment_range.end.line,
                            character=max(segment_range.end.character - 1, 0),
                        ),
                        label=": str",
                        kind=lsp.InlayHintKind.Type,
                        padding_right=False,
                        padding_left=False,
                    )
                )

    for proc in _all_procedures(prog):
        for stmt in proc.body:
            visit_stmt(stmt, proc)
    for stmt in prog.toplevel_body:
        visit_stmt(stmt, None)
    if prog.toplevel_return is not None:
        visit_stmt(prog.toplevel_return, None)
    return out


def _in_range(loc: Span, rng: lsp.Range) -> bool:
    """Loose intersection check between an AST span and an LSP range."""
    if loc.end_line - 1 < rng.start.line:
        return False
    if loc.start_line - 1 > rng.end.line:
        return False
    return True


# =====================================================================
# Document symbols
# =====================================================================


@with_source_coordinates(1)
def compute_document_symbols(source: str, source_path: Optional[str]) -> List[lsp.DocumentSymbol]:
    try:
        prog = parse_program(source, source_path=source_path)
    except KediParseError:
        return []
    symbols: List[lsp.DocumentSymbol] = []
    for td in prog.type_defs:
        symbols.append(_type_def_symbol(td))
    for profile in prog.profile_defs.values():
        symbols.append(_profile_def_symbol(profile))
    for proc in prog.procedures:
        symbols.append(_proc_symbol(proc))
    if isinstance(prog, ProgramWithValidation):
        for suite in prog.tests:
            symbols.append(_test_suite_symbol(suite))
        for suite in prog.evals:
            symbols.append(_eval_suite_symbol(suite))
    return symbols


def _proc_symbol(proc: ProcedureDef) -> lsp.DocumentSymbol:
    children: List[lsp.DocumentSymbol] = []
    for stmt in proc.body:
        from ..lang.ast import ProcDefStmt

        if isinstance(stmt, ProcDefStmt) and stmt.proc is not None:
            children.append(_proc_symbol(stmt.proc))
        elif isinstance(stmt, TypeDefStmt) and stmt.typedef is not None:
            children.append(_type_def_symbol(stmt.typedef))
    for span in proc.optimize_spans:
        if span.loc is not None:
            children.append(
                lsp.DocumentSymbol(
                    name=f"> optimize: {span.name}",
                    detail="optimize span",
                    kind=lsp.SymbolKind.Constructor,
                    range=_span_to_range(span.loc),
                    selection_range=_span_to_range(span.loc),
                    children=[],
                )
            )
    for stmt in proc.body:
        if isinstance(stmt, AgentDirectiveStmt) and stmt.loc is not None:
            value = stmt.plain_value or (f"`{stmt.backtick_code}`" if stmt.backtick_code else "")
            children.append(
                lsp.DocumentSymbol(
                    name=f"> agent: {value}",
                    detail="agent harness override",
                    kind=lsp.SymbolKind.Event,
                    range=_span_to_range(stmt.loc),
                    selection_range=_span_to_range(stmt.loc),
                    children=[],
                )
            )
        elif isinstance(stmt, AdapterDirectiveStmt) and stmt.loc is not None:
            value = stmt.plain_value or (f"`{stmt.backtick_code}`" if stmt.backtick_code else "")
            children.append(
                lsp.DocumentSymbol(
                    name=f"> adapter: {value}",
                    detail="agent framework override",
                    kind=lsp.SymbolKind.Event,
                    range=_span_to_range(stmt.loc),
                    selection_range=_span_to_range(stmt.loc),
                    children=[],
                )
            )
        elif isinstance(stmt, ModelDirectiveStmt) and stmt.loc is not None:
            value = stmt.plain_value or (f"`{stmt.backtick_code}`" if stmt.backtick_code else "")
            children.append(
                lsp.DocumentSymbol(
                    name=f"> model: {value}",
                    detail="agent model override",
                    kind=lsp.SymbolKind.Event,
                    range=_span_to_range(stmt.loc),
                    selection_range=_span_to_range(stmt.loc),
                    children=[],
                )
            )
        elif isinstance(stmt, EffortDirectiveStmt) and stmt.loc is not None:
            value = stmt.plain_value or (f"`{stmt.backtick_code}`" if stmt.backtick_code else "")
            children.append(
                lsp.DocumentSymbol(
                    name=f"> effort: {value}",
                    detail="agent reasoning effort",
                    kind=lsp.SymbolKind.Event,
                    range=_span_to_range(stmt.loc),
                    selection_range=_span_to_range(stmt.loc),
                    children=[],
                )
            )
        elif isinstance(stmt, SystemDirectiveStmt) and stmt.loc is not None:
            value = _system_value_preview(stmt).strip("`")
            children.append(
                lsp.DocumentSymbol(
                    name=f"> system: {value}",
                    detail="agent instructions",
                    kind=lsp.SymbolKind.Event,
                    range=_span_to_range(stmt.loc),
                    selection_range=_span_to_range(stmt.loc),
                    children=[],
                )
            )
        elif isinstance(stmt, UseDirectiveStmt) and stmt.loc is not None:
            if stmt.tool_names:
                use_label = "> use: " + ", ".join(stmt.tool_names)
                use_detail = "agent tools"
            else:
                use_label = f"> use: {stmt.name}"
                use_detail = "agent profile or tool"
            children.append(
                lsp.DocumentSymbol(
                    name=use_label,
                    detail=use_detail,
                    kind=lsp.SymbolKind.Event,
                    range=_span_to_range(stmt.loc),
                    selection_range=_span_to_range(stmt.loc),
                    children=[],
                )
            )
    detail_parts = []
    if proc.params:
        detail_parts.append(
            "(" + ", ".join(f"{p}: {_proc_param_annotation(proc, p)}" for p in proc.params) + ")"
        )
    if proc.return_type_annotation:
        detail_parts.append(f"-> {proc.return_type_annotation}")
    return lsp.DocumentSymbol(
        name=proc.name,
        detail=" ".join(detail_parts),
        kind=lsp.SymbolKind.Function,
        range=_span_to_range(proc.loc) if proc.loc else _empty_range(),
        selection_range=(_span_to_range(_proc_name_span(proc)) if proc.loc else _empty_range()),
        children=children,
    )


def _profile_def_symbol(profile: ProfileDefStmt) -> lsp.DocumentSymbol:
    children: list[lsp.DocumentSymbol] = []
    for member in profile.agent_members:
        if member.loc is None:
            continue
        value = member.plain_value or (
            f"`{member.backtick_code}`" if member.backtick_code is not None else ""
        )
        children.append(
            lsp.DocumentSymbol(
                name=f"> agent: {value}",
                detail="agent harness override",
                kind=lsp.SymbolKind.Event,
                range=_span_to_range(member.loc),
                selection_range=_span_to_range(member.loc),
                children=[],
            )
        )
    for member in profile.adapter_members:
        if member.loc is None:
            continue
        value = member.plain_value or (
            f"`{member.backtick_code}`" if member.backtick_code is not None else ""
        )
        children.append(
            lsp.DocumentSymbol(
                name=f"> adapter: {value}",
                detail="agent framework override",
                kind=lsp.SymbolKind.Event,
                range=_span_to_range(member.loc),
                selection_range=_span_to_range(member.loc),
                children=[],
            )
        )
    for member in profile.model_members:
        if member.loc is None:
            continue
        value = member.plain_value or (
            f"`{member.backtick_code}`" if member.backtick_code is not None else ""
        )
        children.append(
            lsp.DocumentSymbol(
                name=f"> model: {value}",
                detail="agent model override",
                kind=lsp.SymbolKind.Event,
                range=_span_to_range(member.loc),
                selection_range=_span_to_range(member.loc),
                children=[],
            )
        )
    for member in profile.effort_members:
        if member.loc is None:
            continue
        value = member.plain_value or (
            f"`{member.backtick_code}`" if member.backtick_code is not None else ""
        )
        children.append(
            lsp.DocumentSymbol(
                name=f"> effort: {value}",
                detail="agent reasoning effort",
                kind=lsp.SymbolKind.Event,
                range=_span_to_range(member.loc),
                selection_range=_span_to_range(member.loc),
                children=[],
            )
        )
    for member in profile.system_members:
        if member.loc is None:
            continue
        value = _system_value_preview(member).strip("`")
        children.append(
            lsp.DocumentSymbol(
                name=f"> system: {value}",
                detail="agent instructions",
                kind=lsp.SymbolKind.Event,
                range=_span_to_range(member.loc),
                selection_range=_span_to_range(member.loc),
                children=[],
            )
        )
    for member in profile.artifacts_members:
        if member.loc is None:
            continue
        fields = ", ".join(field.name for field in member.fields)
        children.append(
            lsp.DocumentSymbol(
                name=f"> artifacts: {fields}",
                detail="artifact policy",
                kind=lsp.SymbolKind.Event,
                range=_span_to_range(member.loc),
                selection_range=_span_to_range(member.loc),
                children=[],
            )
        )
    for member in profile.output_members:
        if member.loc is None:
            continue
        children.append(
            lsp.DocumentSymbol(
                name=f"> output: {member.type_annotation}",
                detail="default subagent output type",
                kind=lsp.SymbolKind.TypeParameter,
                range=_span_to_range(member.loc),
                selection_range=_span_to_range(member.loc),
                children=[],
            )
        )
    use_names = (*profile.use_targets, *profile.tool_names)
    if use_names and profile.loc is not None:
        children.append(
            lsp.DocumentSymbol(
                name="> use: " + ", ".join(use_names),
                detail="agent profiles or tools",
                kind=lsp.SymbolKind.Event,
                range=_span_to_range(profile.loc),
                selection_range=_span_to_range(profile.loc),
                children=[],
            )
        )
    return lsp.DocumentSymbol(
        name=f"> profile: {profile.name}",
        detail="agent profile",
        kind=lsp.SymbolKind.Interface,
        range=_span_to_range(profile.loc) if profile.loc else _empty_range(),
        selection_range=(_span_to_range(profile.loc) if profile.loc else _empty_range()),
        children=children,
    )


def _type_def_symbol(td: TypeDef) -> lsp.DocumentSymbol:
    children = [
        lsp.DocumentSymbol(
            name=f.name,
            detail=_type_field_annotation(f),
            kind=lsp.SymbolKind.Field,
            range=_span_to_range(f.loc) if f.loc else _empty_range(),
            selection_range=_span_to_range(f.loc) if f.loc else _empty_range(),
            children=[],
        )
        for f in td.fields
    ]
    return lsp.DocumentSymbol(
        name=td.name,
        detail=f"({len(td.fields)} fields)",
        kind=lsp.SymbolKind.Struct,
        range=_span_to_range(td.loc) if td.loc else _empty_range(),
        selection_range=_span_to_range(td.loc) if td.loc else _empty_range(),
        children=children,
    )


def _test_suite_symbol(suite: TestSuite) -> lsp.DocumentSymbol:
    children = [
        lsp.DocumentSymbol(
            name=c.name,
            detail="test case",
            kind=lsp.SymbolKind.Method,
            range=_span_to_range(c.loc) if c.loc else _empty_range(),
            selection_range=_span_to_range(c.loc) if c.loc else _empty_range(),
            children=[],
        )
        for c in suite.cases
    ]
    return lsp.DocumentSymbol(
        name=f"@test: {suite.procedure_name}",
        detail=f"{len(suite.cases)} cases",
        kind=lsp.SymbolKind.Namespace,
        range=_span_to_range(suite.loc) if suite.loc else _empty_range(),
        selection_range=_span_to_range(suite.loc) if suite.loc else _empty_range(),
        children=children,
    )


def _eval_suite_symbol(suite: EvalSuite) -> lsp.DocumentSymbol:
    children: List[lsp.DocumentSymbol] = []
    for name, ds in suite.datasets.items():
        children.append(
            lsp.DocumentSymbol(
                name=f"> data: {name}",
                detail="dataset",
                kind=lsp.SymbolKind.Variable,
                range=_span_to_range(ds.loc) if ds.loc else _empty_range(),
                selection_range=_span_to_range(ds.loc) if ds.loc else _empty_range(),
                children=[],
            )
        )
    for name, ds in suite.test_datasets.items():
        children.append(
            lsp.DocumentSymbol(
                name=f"> test_data: {name}",
                detail="test dataset",
                kind=lsp.SymbolKind.Variable,
                range=_span_to_range(ds.loc) if ds.loc else _empty_range(),
                selection_range=_span_to_range(ds.loc) if ds.loc else _empty_range(),
                children=[],
            )
        )
    for m in suite.metrics:
        children.append(
            lsp.DocumentSymbol(
                name=f"> metric: {m.name}",
                detail=f"(dataset={m.dataset_name})" if m.dataset_name else "metric",
                kind=lsp.SymbolKind.Function,
                range=_span_to_range(m.loc) if m.loc else _empty_range(),
                selection_range=_span_to_range(m.loc) if m.loc else _empty_range(),
                children=[],
            )
        )
    return lsp.DocumentSymbol(
        name=f"@eval: {suite.procedure_name}",
        detail=f"{len(suite.metrics)} metrics, {len(suite.datasets)} data",
        kind=lsp.SymbolKind.Namespace,
        range=_span_to_range(suite.loc) if suite.loc else _empty_range(),
        selection_range=_span_to_range(suite.loc) if suite.loc else _empty_range(),
        children=children,
    )


def _empty_range() -> lsp.Range:
    return lsp.Range(
        start=lsp.Position(line=0, character=0),
        end=lsp.Position(line=0, character=0),
    )


# =====================================================================
# Semantic tokens
# =====================================================================


# Standard LSP semantic-token type list. The ORDER matters: clients
# reference token types by their index in the legend the server returns
# at initialization.
SEMANTIC_TOKEN_TYPES: List[str] = [
    "namespace",  # 0
    "type",  # 1
    "class",  # 2
    "enum",  # 3
    "interface",  # 4
    "struct",  # 5
    "typeParameter",  # 6
    "parameter",  # 7
    "variable",  # 8
    "property",  # 9
    "enumMember",  # 10
    "event",  # 11
    "function",  # 12
    "method",  # 13
    "macro",  # 14
    "keyword",  # 15
    "modifier",  # 16
    "comment",  # 17
    "string",  # 18
    "number",  # 19
    "regexp",  # 20
    "operator",  # 21
    "decorator",  # 22
]
SEMANTIC_TOKEN_MODIFIERS: List[str] = [
    "declaration",  # bit 0  = 1
    "definition",  # bit 1  = 2
    "readonly",  # bit 2  = 4
    "static",  # bit 3  = 8
    "deprecated",  # bit 4  = 16
    "abstract",  # bit 5  = 32
    "async",  # bit 6  = 64
    "modification",  # bit 7  = 128
    "documentation",  # bit 8  = 256
    "defaultLibrary",  # bit 9  = 512
]

_TT = {name: i for i, name in enumerate(SEMANTIC_TOKEN_TYPES)}
_TM = {name: 1 << i for i, name in enumerate(SEMANTIC_TOKEN_MODIFIERS)}


# Map tree-sitter highlight-capture names → (LSP token type, modifiers).
#
# This is the *single* place feature semantics turn into LSP wire
# encoding. Both the Kedi grammar's ``highlights.scm`` and
# ``tree-sitter-python``'s upstream ``highlights.scm`` are run as
# tree-sitter queries; their captures share the standard nvim-treesitter
# / Helix / Zed naming convention (``@function.call``,
# ``@variable.parameter``, …), so one mapping covers both.
#
# Lookup is **longest-prefix** in capture-name space: a node tagged
# ``@function.method`` will match the ``function.method`` entry; if no
# entry exists for the exact name, we walk down dotted segments
# (``function.method`` → ``function``) until we hit one. That way new
# refinements in upstream highlights.scm files degrade gracefully.
_CAPTURE_TO_LSP: dict = {
    # Comments
    "comment": ("comment", 0),
    "comment.line": ("comment", 0),
    "comment.block": ("comment", 0),
    # Strings
    "string": ("string", 0),
    "string.special": ("string", 0),
    "string.escape": ("string", 0),
    "escape": ("string", 0),
    # Numbers / literals
    "number": ("number", 0),
    "boolean": ("keyword", 0),
    "constant": ("variable", 0),
    "constant.builtin": ("keyword", 0),
    # Operators / punctuation
    "operator": ("operator", 0),
    # Only punctuation.special is semantically meaningful in Kedi (`@`,
    # `~`, `>`, ``` and backticks). Delimiters stay uncoloured to avoid
    # noisy paren / colon / bracket highlighting.
    "punctuation.special": ("operator", 0),
    # We deliberately leave ``punctuation.delimiter`` out — colouring
    # every paren / colon / bracket looks noisy.
    # Keywords
    "keyword": ("keyword", 0),
    "keyword.return": ("keyword", 0),
    "keyword.function": ("keyword", 0),
    "keyword.operator": ("keyword", 0),
    "keyword.import": ("keyword", 0),
    "keyword.control": ("keyword", 0),
    # Functions / methods
    "function": ("function", 0),
    "function.call": ("function", 0),
    "function.method": ("method", 0),
    "function.builtin": ("function", 0),
    "function.macro": ("macro", 0),
    "method": ("method", 0),
    # Types
    "type": ("type", 0),
    "type.builtin": ("type", 0),
    "type.definition": ("type", 0),
    "constructor": ("class", 0),
    "class": ("class", 0),
    # Variables / params / properties
    "variable": ("variable", 0),
    "variable.parameter": ("parameter", 0),
    "variable.builtin": ("variable", 0),
    "parameter": ("parameter", 0),
    "property": ("property", 0),
    "attribute": ("property", 0),
    "field": ("property", 0),
    # Misc identifiers
    "decorator": ("decorator", 0),
    "namespace": ("namespace", 0),
    "module": ("namespace", 0),
    "label": ("enumMember", 0),
    "tag": ("variable", 0),
    "embedded": ("operator", 0),
}


def _capture_to_token(name: str) -> Optional[Tuple[int, int]]:
    """Resolve a tree-sitter highlight capture to (token_type_idx, mods).

    Falls back through dotted prefixes (``a.b.c`` → ``a.b`` → ``a``)
    when the exact capture name isn't mapped.
    """
    parts = name.split(".")
    while parts:
        key = ".".join(parts)
        hit = _CAPTURE_TO_LSP.get(key)
        if hit is not None:
            tt, mods = hit
            return _TT[tt], mods
        parts.pop()
    return None


def _procedure_docstring_nodes(root: Node) -> List[Node]:
    out: List[Node] = []

    def visit(node: Node) -> None:
        if node.type == "procedure_def":
            body = node.child_by_field_name("body")
            if body is not None:
                named = [child for child in body.children if child.is_named]
                if named and named[0].type == "block_comment":
                    out.append(named[0])
        for child in node.children:
            if child.is_named:
                visit(child)

    visit(root)
    return out


def _python_code_lines(text: str) -> list[str]:
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    return lines


def _emit_token_spans_for_range(
    *,
    src_lines: Sequence[str],
    emit: Callable[[int, int, int, int, int], None],
    start_line: int,
    start_col: int,
    end_line: int,
    end_col: int,
    token_type: int,
    token_mods: int,
) -> None:
    """Emit one or more LSP semantic-token spans for a possibly multi-line range."""
    if start_line == end_line:
        emit(start_line, start_col, end_col - start_col, token_type, token_mods)
        return
    if start_line < len(src_lines):
        emit(
            start_line,
            start_col,
            max(0, utf16_length(src_lines[start_line]) - start_col),
            token_type,
            token_mods,
        )
    for line_idx in range(start_line + 1, end_line):
        if line_idx < len(src_lines):
            emit(line_idx, 0, utf16_length(src_lines[line_idx]), token_type, token_mods)
    if end_col > 0 and end_line < len(src_lines):
        emit(end_line, 0, end_col, token_type, token_mods)


@with_source_coordinates(1)
def compute_semantic_tokens_full(source: str, source_path: Optional[str]) -> lsp.SemanticTokens:
    """Run tree-sitter highlight queries and emit LSP semantic tokens.

    The pipeline mirrors what editors like Neovim/Helix do natively
    with tree-sitter:

      1. Parse the document with tree-sitter-kedi.
      2. Run ``tree-sitter-kedi/queries/highlights.scm`` → Kedi tokens.
      3. Run ``tree-sitter-kedi/queries/injections.scm`` → list of
         (range, embedded language) injection sites.
      4. For each Python injection, parse the region's raw bytes with
         tree-sitter-python and run *its own*
         ``queries/highlights.scm`` against the sub-tree.
      5. Translate sub-tree positions back into the Kedi document's
         coordinate space; merge with the Kedi tokens.
      6. Sort + delta-encode per the LSP spec.

    There is no hand-rolled Python (or Kedi) classifier — the SCM
    query files are the single source of truth.
    """
    src_bytes = source.encode("utf-8") if isinstance(source, str) else source
    tree: Tree = _ts_parse(src_bytes)
    src_lines = source.split("\n")
    raw: List[Tuple[int, int, int, int, int]] = []

    # Collect Python-injection byte ranges *first* so any Kedi-level
    # token that falls inside them (e.g. the fallback ``(python_code)
    # @string.special`` capture in kedi's highlights.scm) can be
    # punched out — the upstream tree-sitter-python tokens are finer
    # and the LSP spec forbids overlapping semantic tokens.
    python_byte_ranges: List[Tuple[int, int]] = []
    if _KEDI_INJECTIONS_QUERY is not None:
        cursor = _TSQueryCursor(_KEDI_INJECTIONS_QUERY)
        for _pat_idx, caps in cursor.matches(tree.root_node):
            for region_node in caps.get("injection.content", []):
                python_byte_ranges.append((region_node.start_byte, region_node.end_byte))
    docstring_nodes = _procedure_docstring_nodes(tree.root_node)
    docstring_byte_ranges = [(node.start_byte, node.end_byte) for node in docstring_nodes]

    def in_python_region(node: Node) -> bool:
        for s, e in python_byte_ranges:
            if node.start_byte >= s and node.end_byte <= e:
                return True
        return False

    def in_docstring_region(node: Node) -> bool:
        for s, e in docstring_byte_ranges:
            if node.start_byte >= s and node.end_byte <= e:
                return True
        return False

    def emit(line: int, col: int, length: int, ttype: int, mods: int) -> None:
        if length <= 0:  # pragma: no cover - tree-sitter captures are non-empty.
            return
        raw.append((line, col, length, ttype, mods))

    def emit_node(node: Node, ttype: int, mods: int, base_line: int = 0, base_col: int = 0) -> None:
        sl, sc = node.start_point
        el, ec = node.end_point
        # Translate from the embedded sub-tree's coordinates back to
        # Kedi-document coordinates. The Python sub-tree was parsed in
        # isolation, so its line 0 col 0 corresponds to (base_line,
        # base_col) in the Kedi document. Lines past line 0 in the
        # sub-tree always start at col 0 (newlines are 1:1).
        kedi_sl = base_line + sl
        kedi_sc = (base_col + sc) if sl == 0 else sc
        kedi_el = base_line + el
        kedi_ec = (base_col + ec) if el == 0 else ec
        _emit_token_spans_for_range(
            src_lines=src_lines,
            emit=emit,
            start_line=kedi_sl,
            start_col=byte_column_to_utf16(src_lines[kedi_sl], kedi_sc),
            end_line=kedi_el,
            end_col=byte_column_to_utf16(src_lines[kedi_el], kedi_ec),
            token_type=ttype,
            token_mods=mods,
        )

    # ---- 1. Kedi tokens via highlights.scm ----------------------------
    if _KEDI_HIGHLIGHTS_QUERY is not None:
        _emit_query_captures(
            _KEDI_HIGHLIGHTS_QUERY,
            tree.root_node,
            emit_node,
            base_line=0,
            base_col=0,
            skip_if=lambda node: in_python_region(node) or in_docstring_region(node),
        )
    docstring_token = _capture_to_token("variable")
    if docstring_token is not None:
        ttype, mods = docstring_token
        for node in docstring_nodes:
            emit_node(node, ttype, mods)

    # ---- 2. Embedded Python regions via injections.scm ---------------
    if python_byte_ranges and _PY_PARSER is not None and _PY_HIGHLIGHTS_QUERY is not None:
        assert _KEDI_INJECTIONS_QUERY is not None
        cursor = _TSQueryCursor(_KEDI_INJECTIONS_QUERY)
        for _pat_idx, caps in cursor.matches(tree.root_node):
            for region_node in caps.get("injection.content", []):
                py_bytes = region_node.text
                if not py_bytes:  # pragma: no cover - injection captures are non-empty.
                    continue
                try:
                    py_tree = _PY_PARSER.parse(py_bytes)
                except Exception:  # pragma: no cover — defensive
                    continue
                py_base_line, py_base_col = region_node.start_point
                _emit_query_captures(
                    _PY_HIGHLIGHTS_QUERY,
                    py_tree.root_node,
                    emit_node,
                    base_line=py_base_line,
                    base_col=py_base_col,
                )

    # ---- 3. Resolve overlaps + LSP delta-encode ----------------------
    # The LSP spec forbids overlapping semantic tokens. Highlights
    # queries routinely produce overlap — e.g. Python's f-string
    # ``(string)`` covers the whole literal while ``(interpolation)``
    # and identifier captures sit inside it, and pattern refinement
    # (``(identifier) @variable`` vs ``(call ... @function)``) emits
    # two tokens on the same node. Resolve by "paint with specificity":
    # for each character column, the *smallest* covering token wins.
    resolved = _resolve_overlaps(raw)
    resolved.sort(key=lambda t: (t[0], t[1]))

    data: List[int] = []
    prev_line = 0
    prev_col = 0
    for line, col, length, ttype, mods in resolved:
        d_line = line - prev_line
        d_col = col - prev_col if d_line == 0 else col
        data.extend([d_line, d_col, length, ttype, mods])
        prev_line = line
        prev_col = col
    return lsp.SemanticTokens(data=data)


def _resolve_overlaps(
    tokens: List[Tuple[int, int, int, int, int]],
) -> List[Tuple[int, int, int, int, int]]:
    """Collapse overlapping tokens into non-overlapping runs.

    For each character position, the token with the *smallest* length
    that covers it wins (smaller = more specific). Ties go to whichever
    was emitted first. Output runs of equal classification are merged
    so we don't fragment unnecessarily.
    """
    by_line: dict = {}
    for line, col, length, ttype, mods in tokens:
        by_line.setdefault(line, []).append((col, length, ttype, mods))

    out: List[Tuple[int, int, int, int, int]] = []
    for line, line_tokens in by_line.items():
        max_col = max(c + n for c, n, _, _ in line_tokens)
        # `paint[i]` = (length, ttype, mods) where length acts as the
        # specificity tag. Smaller length wins on tie-break.
        paint: List[Optional[Tuple[int, int, int]]] = [None] * max_col
        # Apply tokens longest-first so shorter ones overwrite.
        for col, length, ttype, mods in sorted(line_tokens, key=lambda t: -t[1]):
            for i in range(col, col + length):
                if 0 <= i < max_col:
                    paint[i] = (length, ttype, mods)
        # Collapse contiguous runs of identical (ttype, mods).
        i = 0
        while i < max_col:
            slot = paint[i]
            if slot is None:
                i += 1
                continue
            _, ttype, mods = slot
            j = i
            while j < max_col:
                next_slot = paint[j]
                if next_slot is None or next_slot[1:] != (ttype, mods):
                    break
                j += 1
            out.append((line, i, j - i, ttype, mods))
            i = j
    return out


def _emit_query_captures(
    query: _TSQuery,
    root: Node,
    emit_node,
    *,
    base_line: int,
    base_col: int,
    skip_if=None,
) -> None:
    """Run a highlights.scm-style query and emit tokens for each capture.

    Pattern order matters: tree-sitter returns matches in the order
    patterns appear in the .scm file, and the convention is that
    later patterns refine earlier ones (``(identifier) @variable``
    early, ``(function_definition name: (identifier) @function)``
    later). We feed all captures into the emit pipeline and let the
    dedup pass in :func:`compute_semantic_tokens_full` keep the last
    one for any identical (line, col, length).

    ``skip_if(node) -> bool`` lets the caller drop captures that fall
    inside an injection region (so an outer-language fallback
    capture doesn't overlap the inner-language injection tokens).
    """
    cursor = _TSQueryCursor(query)
    for _pat_idx, caps in cursor.matches(root):
        for cap_name, nodes_list in caps.items():
            mapped = _capture_to_token(cap_name)
            if mapped is None:
                continue
            ttype, mods = mapped
            for n in nodes_list:
                if skip_if is not None and skip_if(n):
                    continue
                emit_node(n, ttype, mods, base_line, base_col)


# =====================================================================
# kedi/pythonRanges — custom request
# =====================================================================


@with_source_coordinates(None)
def compute_python_ranges(source: str) -> List[dict]:
    """Return byte / line ranges of every Python region in the document.

    Each entry is::

        {
          "kind": "fenced" | "inline",
          "range": {"start": {"line": ..., "character": ...},
                     "end":   {"line": ..., "character": ...}},
          "text":  "<the Python source (dedented for fenced, raw for inline)>"
        }

    Consumed by the VS Code extension to assemble a virtual Python
    document that mirrors the Kedi file's coordinate space so requests
    can be forwarded to a Python language server without translation.
    """
    return _collect_python_ranges(source)


@with_source_coordinates(None)
def compute_header_python_ranges(source: str) -> List[dict]:
    return _collect_python_ranges(source, include_regular_inline=False)


def _collect_python_ranges(
    source: str,
    *,
    include_header_inline: bool = True,
    include_regular_inline: bool = True,
    include_directive_inline: bool = True,
) -> List[dict]:
    tree: Tree = _ts_parse(source.encode("utf-8") if isinstance(source, str) else source)
    src_bytes = source.encode("utf-8") if isinstance(source, str) else source
    out: List[dict] = []

    def visit(node: Node, parents: tuple[Node, ...] = ()) -> None:
        if node.type == "python_code":
            if not include_regular_inline:
                return
            text = src_bytes[node.start_byte : node.end_byte].decode("utf-8", "replace")
            from ..lang.parser.escapes import dedent_block

            dedented = "\n".join(dedent_block(_python_code_lines(text)))
            out.append(
                {
                    "kind": "fenced",
                    "range": _node_range(node, source),
                    "text": dedented,
                }
            )
        elif node.type == "python_inline_body":
            if not include_directive_inline and any(
                parent.type.endswith("_directive") for parent in parents
            ):
                return
            header_owner = _header_inline_owner(parents)
            if header_owner is not None and not include_header_inline:
                return
            if header_owner is None and not include_regular_inline:
                return
            text = src_bytes[node.start_byte : node.end_byte].decode("utf-8", "replace")
            item = {
                "kind": "inline",
                "range": _node_range(node, source),
                "text": text,
            }
            if header_owner is not None:
                item.update(header_owner)
            out.append(item)
        for c in node.children:
            visit(c, parents + (node,))

    visit(tree.root_node)
    return out


def _position_in_python_hover_region(source: str, pos: lsp.Position) -> bool:
    return any(
        _position_in_range(pos, region["range"])
        for region in _collect_python_ranges(source, include_directive_inline=False)
    )


def _position_in_range(pos: lsp.Position, range_: dict) -> bool:
    start = range_["start"]
    end = range_["end"]
    if pos.line < start["line"] or pos.line > end["line"]:
        return False
    if pos.line == start["line"] and pos.character < start["character"]:
        return False
    if pos.line == end["line"] and pos.character >= end["character"]:
        return False
    return True


def _node_range(node: Node, source: str) -> dict:
    start = tree_sitter_point_to_position(source, *node.start_point)
    end = tree_sitter_point_to_position(source, *node.end_point)
    return {
        "start": {"line": start.line, "character": start.character},
        "end": {"line": end.line, "character": end.character},
    }


def _header_inline_owner(parents: tuple[Node, ...]) -> Optional[dict]:
    is_type_expression = any(parent.type == "type_python" for parent in parents)
    owner = next(
        (parent for parent in reversed(parents) if parent.type in {"param", "type_field"}),
        None,
    )
    container = next(
        (parent for parent in reversed(parents) if parent.type in {"procedure_def", "type_def"}),
        None,
    )
    if owner is None:
        is_output_type = any(parent.type == "output_segment" for parent in parents)
        if not is_type_expression or is_output_type or container is None:
            return None
        owner_kind = "return_type"
    else:
        owner_kind = owner.type
    return {
        "ownerKind": owner_kind,
        "containerKind": container.type if container is not None else None,
        "containerLine": container.start_point[0] if container is not None else None,
        "expressionKind": "type" if is_type_expression else "default",
    }
