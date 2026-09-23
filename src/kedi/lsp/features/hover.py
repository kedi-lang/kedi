from __future__ import annotations

from typing import Optional

from lsprotocol import types as lsp
from tree_sitter import Node

from kedi.lang.ast import (
    AdapterDirectiveStmt,
    AgentConnectionField,
    AgentDirectiveStmt,
    ApprovalDirectiveStmt,
    ArtifactsDirectiveStmt,
    ArtifactsField,
    AssignmentBlockStmt,
    AssignmentStmt,
    AutoSpec,
    BacktickLineStmt,
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
    ProfileRequiresEntry,
    ProfileSettingsEntry,
    ProfileSkillsEntry,
    ProfileSystemEntry,
    ProfileWorkflowEntry,
    Program,
    ProgramWithValidation,
    PythonBlock,
    PythonExprSegment,
    RawInvokeStmt,
    RequiresDirectiveStmt,
    ReturnBlockStmt,
    ReturnStmt,
    ShowStmt,
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
    ToolDeclaration,
    TypeDef,
    TypeField,
    UseDirectiveStmt,
    VariableInitBlockStmt,
    VariableInitStmt,
)
from kedi.lang.call_arg_access import iter_call_arg_nested
from kedi.lang.parser import KediParseError, parse_program
from kedi.lang.parser.parser import parse as _ts_parse
from kedi.lsp.coordinates import position_inside as _position_inside
from kedi.lsp.coordinates import span_to_range as _span_to_range
from kedi.lsp.coordinates import with_source_coordinates
from kedi.lsp.features.directive_help import (
    _adapter_directive_markdown,
    _agent_connection_field_markdown,
    _agent_directive_markdown,
    _agent_tool_markdown,
    _approval_directive_markdown,
    _artifacts_directive_markdown,
    _artifacts_field_markdown,
    _codemode_directive_markdown,
    _effort_directive_markdown,
    _history_directive_markdown,
    _hooks_directive_markdown,
    _known_type_markdown,
    _mcp_directive_markdown,
    _mcp_field_markdown,
    _model_directive_markdown,
    _prelude_type_markdown,
    _proc_param_annotation,
    _proc_signature_markdown,
    _profile_adapter_entry_markdown,
    _profile_agent_entry_markdown,
    _profile_approval_entry_markdown,
    _profile_artifacts_entry_markdown,
    _profile_codemode_entry_markdown,
    _profile_effort_entry_markdown,
    _profile_history_entry_markdown,
    _profile_hooks_entry_markdown,
    _profile_mcp_entry_markdown,
    _profile_model_entry_markdown,
    _requires_directive_markdown,
    _profile_settings_entry_markdown,
    _profile_signature_markdown,
    _profile_skills_entry_markdown,
    _profile_system_entry_markdown,
    _python_module_scope_type_markdown,
    _settings_directive_markdown,
    _settings_field_markdown,
    _skills_directive_markdown,
    _system_directive_markdown,
    _tool_declaration_markdown,
    _type_expr_to_str,
    _type_field_annotation,
    _type_signature_markdown,
    _use_directive_markdown,
)
from kedi.lsp.features.python_regions import _position_in_python_hover_region
from kedi.lsp.features.resolution import _resolve_symbol_at
from kedi.lsp.features.scope import (
    _all_procedures,
    _all_type_defs,
    _module_scope_python_names,
    _prelude_defined_names,
    _procedure_for_declaration,
    _type_def_for_declaration,
)
from kedi.lsp.features.symbols import _merged_profile_defs
from kedi.lsp.features.type_support import _known_lsp_type_origin
from kedi.lsp.module_surface import ImportedSurface, ImportedValueInfo, imported_surface_for_program


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
            if node.tool is not None:
                visit(node.tool)
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
            for member in node.requires_members:
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
        if isinstance(node, (AgentDirectiveStmt, ProfileAgentEntry)):
            for field in node.connection_fields:
                visit(field)
            return
        if isinstance(node, AgentConnectionField):
            for field in node.fields:
                visit(field)
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
        if isinstance(node, (ReturnStmt, ShowStmt)):
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
    if toplevel_return is not None and not any(
        item is toplevel_return for item in ordered_toplevel
    ):
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
        return "**Return**\n\nEnds this procedure or regular program invocation immediately."
    if isinstance(node, ShowStmt):
        return "**Display** `> show:`\n\nDisplays a value once and continues execution without returning."
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
    if isinstance(node, AgentConnectionField):
        return _agent_connection_field_markdown(node)
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
    if isinstance(node, (RequiresDirectiveStmt, ProfileRequiresEntry)):
        return _requires_directive_markdown(node)
    if isinstance(node, ToolDeclaration):
        return _tool_declaration_markdown(node)
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


def _missing_agent_profile_hover(name: str, imported_surface: ImportedSurface) -> str:
    module_info = imported_surface.modules.get(name)
    if module_info is not None:
        from kedi.lsp.module_surface import load_imported_surface

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
