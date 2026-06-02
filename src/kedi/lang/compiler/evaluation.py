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
    TextSegment,
    VarArg,
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
    with rt.trace_frame(
        KediTraceFrame(kind="python_side_effects", label="python statement", span=span)
    ):
        rt.executor.execute_side_effects(
            rt,
            code,
            env_map,
            kedi_line_offset=kedi_line_offset,
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
            pytype = kedi_type.resolve(rt, env=kwargs)
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
        if hasattr(arg, "__native_backtick__"):
            code = getattr(arg, "__native_backtick__")
            return evaluate_python_inline(rt, code, local_env, span=arg.loc)
        if hasattr(arg, "__render_segments__"):
            segs = getattr(arg, "__render_segments__")
            return render_expression(rt, segs, local_env, context="return")
        if hasattr(arg, "__call_segment__"):
            nested = getattr(arg, "__call_segment__")
            return evaluate_call(rt, nested, local_env)
        if hasattr(arg, "__python_segment__"):
            pyseg = getattr(arg, "__python_segment__")
            return evaluate_python_inline(rt, pyseg.code, local_env, span=pyseg.loc)
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
