"""Document symbols and conservative source formatting for the language server."""

from __future__ import annotations

from typing import List, Optional

from lsprotocol import types as lsp

from kedi.lang.ast import (
    AdapterDirectiveStmt,
    AgentDirectiveStmt,
    EffortDirectiveStmt,
    EvalSuite,
    ModelDirectiveStmt,
    ProcDefStmt,
    ProcedureDef,
    ProfileDefStmt,
    ProgramWithValidation,
    SystemDirectiveStmt,
    TestSuite,
    TypeDef,
    TypeDefStmt,
    UseDirectiveStmt,
)
from kedi.lang.parser import KediParseError, parse_program
from kedi.lsp.coordinates import span_to_range as _span_to_range
from kedi.lsp.coordinates import with_source_coordinates
from kedi.lsp.features.directive_help import (
    _proc_param_annotation,
    _system_value_preview,
    _type_field_annotation,
)
from kedi.lsp.features.scope import _proc_name_span


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
                    name=f"> instructions: {value}",
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
    for member in profile.defer_members:
        if member.loc is not None:
            value = (
                f"`{member.backtick_code}`"
                if member.backtick_code is not None
                else str(member.enabled).lower()
            )
            children.append(
                lsp.DocumentSymbol(
                    name=f"> defer: {value}",
                    detail="capability activation policy",
                    kind=lsp.SymbolKind.Event,
                    range=_span_to_range(member.loc),
                    selection_range=_span_to_range(member.loc),
                    children=[],
                )
            )
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
                name=f"> instructions: {value}",
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
        name=f"> {profile.kind}: {profile.name}",
        detail=f"agent {profile.kind}",
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
