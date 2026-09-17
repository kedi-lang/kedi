from __future__ import annotations

from typing import List, Optional

from lsprotocol import types as lsp

from kedi.lang.ast import CallSegment, OutputSegment, ProcedureDef, Span
from kedi.lang.call_arg_access import iter_call_arg_nested
from kedi.lang.parser import KediParseError, parse_program
from kedi.lsp.coordinates import position_inside as _position_inside
from kedi.lsp.coordinates import span_to_range as _span_to_range
from kedi.lsp.coordinates import with_source_coordinates
from kedi.lsp.features.directive_help import _format_proc_param
from kedi.lsp.features.scope import _all_procedures, _segments_of
from kedi.lsp.module_surface import imported_surface_for_program


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
