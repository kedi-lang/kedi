"""Expression and segment evaluation for Kedi compiler."""

from __future__ import annotations

from typing import Any

from ...core import KediRuntime
from ...errors import KediTraceFrame
from ..ast import (
    CallArg,
    CallSegment,
    InputSegment,
    LiteralArg,
    OutputSegment,
    PythonExprNativeSegment,
    PythonExprSegment,
    Segment,
    Span,
    TemplateLine,
    TextSegment,
    VarArg,
)
from ..call_arg_access import (
    call_arg_call_segment,
    call_arg_native_backtick,
    call_arg_python_segment,
    call_arg_render_segments,
)
from ..parser.escapes import preserve_escaped_whitespace
from .environment import lookup_value
from .type_system import KediType


def evaluate_call(rt: KediRuntime, seg: CallSegment, local_env: dict[str, Any]) -> Any:
    """Evaluate a call segment with arguments."""
    eval_args = [_evaluate_arg(rt, arg, local_env) for arg in seg.args]
    callee = rt._get_proc(seg.name)
    return rt._invoke_procedure(callee, eval_args)


def evaluate_python_inline(
    rt: KediRuntime,
    code: str,
    local_env: dict[str, Any],
    *,
    span=None,
) -> Any:
    """Evaluate inline Python expression or statement."""
    with rt.trace_frame(KediTraceFrame(kind="python_inline", label="inline python", span=span)):
        return rt.executor.evaluate_inline(rt, code, local_env)


def execute_python_block(code: str, rt: KediRuntime, kwargs: dict[str, Any], *, span=None) -> Any:
    """Execute a multi-line Python block with return value."""
    with rt.trace_frame(KediTraceFrame(kind="python_block", label="python block", span=span)):
        return rt.executor.execute_block(rt, code, kwargs)


def execute_inline_side_effects(
    rt: KediRuntime,
    code: str,
    env_map: dict[str, Any],
    *,
    span=None,
    kedi_line_offset: int = 0,
) -> None:
    """Execute Python code for side effects, syncing back existing vars only."""
    __tracebackhide__ = True
    with rt.trace_frame(
        KediTraceFrame(kind="python_side_effects", label="python statement", span=span)
    ):
        rt.executor.execute_side_effects(
            rt,
            code,
            env_map,
            kedi_line_offset=kedi_line_offset,
        )


def merge_template_block_lines(
    lines: list[TemplateLine],
    *,
    loc: Span | None = None,
) -> TemplateLine:
    """Merge ``>>`` block rows into one template (newline-joined, single LLM run)."""
    if not lines:
        return TemplateLine(segments=[], loc=loc)
    if len(lines) == 1:
        line = lines[0]
        if loc is None:
            return line
        return TemplateLine(
            segments=line.segments,
            optimized_prompt_prefix=line.optimized_prompt_prefix,
            optimized_output_override=line.optimized_output_override,
            loc=loc,
        )

    output_names: set[str] = set()
    for line in lines:
        for seg in line.segments:
            if isinstance(seg, OutputSegment):
                output_names.add(seg.name)

    merged_segments: list[Segment] = []
    for index, line in enumerate(lines):
        if index > 0:
            merged_segments.append(TextSegment(text="\n"))
        for seg in line.segments:
            if isinstance(seg, InputSegment) and seg.name in output_names:
                merged_segments.append(TextSegment(text=f"<{seg.name}>"))
            else:
                merged_segments.append(seg)

    merged_override: dict[str, Any] | None = None
    for line in lines:
        if line.optimized_output_override:
            merged_override = {
                **(merged_override or {}),
                **line.optimized_output_override,
            }

    prefix = next(
        (line.optimized_prompt_prefix for line in lines if line.optimized_prompt_prefix),
        None,
    )

    return TemplateLine(
        segments=merged_segments,
        optimized_prompt_prefix=prefix,
        optimized_output_override=merged_override,
        loc=loc or lines[0].loc,
    )


def prepare_segments(
    rt: KediRuntime,
    segments: list[Segment],
    kwargs: dict[str, Any],
    *,
    context: str = "template",
) -> tuple[list[Any], str]:
    """Prepare segments for rt.m() call, building expression and string parts."""
    expr: list[Any] = []
    parts: list[str] = []

    for seg in segments:
        if isinstance(seg, TextSegment):
            expr.append(seg.text)
            parts.append(seg.text)
        elif isinstance(seg, InputSegment):
            if context == "template":
                expr.append(rt.i(seg.name))
                parts.append("")
            else:
                value = lookup_value(rt, seg.name, kwargs)
                parts.append(str(value))
        elif isinstance(seg, CallSegment):
            value = evaluate_call(rt, seg, kwargs)
            expr.append(str(value))
            parts.append(str(value))
        elif isinstance(seg, OutputSegment):
            kedi_type = KediType.from_node(seg.type_expr, seg.type_annotation)
            # None (no annotation) still defaults to str downstream; a written
            # but unresolvable annotation raises instead of silently becoming str.
            pytype = kedi_type.resolve_required(rt, kwargs)
            expr.append(rt.o(seg.name, pytype))
        elif isinstance(seg, (PythonExprSegment, PythonExprNativeSegment)):
            value = evaluate_python_inline(rt, seg.code, kwargs, span=seg.loc)
            expr.append(str(value))
            parts.append(str(value))

    return expr, "".join(parts)


def render_expression(
    rt: KediRuntime, segments: list[Segment], kwargs: dict[str, Any], *, context: str
) -> str:
    """Render segments to string, invoking rt.m() only if outputs present."""
    if _has_outputs(segments):
        expr, _ = prepare_segments(rt, segments, kwargs, context=context)
        token = rt._current_env.set(kwargs)
        try:
            rt.m(expr)
            env = rt._current_env.get()
            if env:
                kwargs.update(env)
        finally:
            rt._current_env.reset(token)
        return _stringify_segments(rt, segments, kwargs, context=context)

    return _stringify_segments(rt, segments, kwargs, context=context)


def evaluate_assignment_rhs(
    rt: KediRuntime, segments: list[Segment], kwargs: dict[str, Any]
) -> tuple[Any | None, str]:
    """Evaluate assignment RHS, returning native value if pure expression."""
    non_ws = [s for s in segments if not (isinstance(s, TextSegment) and s.text.strip() == "")]

    # Single native expression or call
    if len(non_ws) == 1:
        seg = non_ws[0]
        if isinstance(seg, PythonExprNativeSegment):
            value = evaluate_python_inline(rt, seg.code, kwargs, span=seg.loc)
            return value, str(value)
        if isinstance(seg, CallSegment):
            value = evaluate_call(rt, seg, kwargs)
            return value, str(value)

    # Fallback to rendered string
    return None, render_expression(rt, segments, kwargs, context="assign")


# Private helpers


def _evaluate_arg(rt: KediRuntime, arg: CallArg, local_env: dict[str, Any]) -> Any:
    """Evaluate a single call argument."""
    if isinstance(arg, LiteralArg):
        native = call_arg_native_backtick(arg)
        if native is not None:
            return evaluate_python_inline(rt, native, local_env, span=arg.loc)
        render_segments = call_arg_render_segments(arg)
        if render_segments:
            return render_expression(rt, render_segments, local_env, context="return")
        call_segment = call_arg_call_segment(arg)
        if call_segment is not None:
            return evaluate_call(rt, call_segment, local_env)
        python_segment = call_arg_python_segment(arg)
        if python_segment is not None:
            return evaluate_python_inline(
                rt, python_segment.code, local_env, span=python_segment.loc
            )
        return arg.value
    elif isinstance(arg, VarArg):
        return lookup_value(rt, arg.name, local_env)
    raise ValueError("Unknown argument type")


def _has_outputs(segments: list[Segment]) -> bool:
    """Check if any segment is an output."""
    return any(isinstance(s, OutputSegment) for s in segments)


def _stringify_segments(
    rt: KediRuntime, segments: list[Segment], kwargs: dict[str, Any], *, context: str
) -> str:
    """Convert segments to string, resolving all values."""
    parts: list[str] = []

    for i, seg in enumerate(segments):
        if isinstance(seg, TextSegment):
            # Apply preserve_escaped_whitespace to boundary TextSegments or those with escaped whitespace
            is_first = i == 0
            is_last = i == len(segments) - 1

            if seg.escaped_whitespace:
                # Has escaped whitespace, always apply preserve_escaped_whitespace
                processed_text = preserve_escaped_whitespace(seg.text, seg.escaped_whitespace)
                parts.append(processed_text)
            elif is_first and is_last:
                # Only TextSegment, trim leading and trailing spaces
                parts.append(seg.text.strip())
            elif is_first:
                # First segment, trim leading spaces only
                parts.append(seg.text.lstrip())
            elif is_last:
                # Last segment, trim trailing spaces only
                parts.append(seg.text.rstrip())
            else:
                # Internal TextSegment, keep as-is
                parts.append(seg.text)
        elif isinstance(seg, InputSegment):
            value = lookup_value(rt, seg.name, kwargs)
            parts.append(str(value))
        elif isinstance(seg, CallSegment):
            value = evaluate_call(rt, seg, kwargs)
            parts.append(str(value))
        elif isinstance(seg, OutputSegment):
            if seg.name in kwargs:
                parts.append(str(kwargs[seg.name]))
            else:
                env = rt._current_env.get()
                if env and seg.name in env:
                    parts.append(str(env[seg.name]))
                else:
                    parts.append("")
        elif isinstance(seg, (PythonExprSegment, PythonExprNativeSegment)):
            value = evaluate_python_inline(rt, seg.code, kwargs, span=seg.loc)
            parts.append(str(value))

    return "".join(parts)
