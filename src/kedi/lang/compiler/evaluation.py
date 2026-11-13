"""Expression and segment evaluation for Kedi compiler."""

from __future__ import annotations

from typing import Any

from ...core import KediRuntime
from ...core.expressions import ComputedExpr, ReferenceExpr, binding_reference
from ...errors import KediTraceFrame
from ...media import Reference, reference
from ...promise import force
from ..ast import (
    CallArg,
    CallReference,
    CallSegment,
    InputReference,
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
from .environment import lookup_value, native_value, raw_lookup_value, substitution_text
from .type_system import KediType

NO_NATIVE_VALUE = object()


def evaluate_call(
    rt: KediRuntime,
    seg: CallSegment,
    local_env: dict[str, Any],
    *,
    cache: dict[int, Any] | None = None,
) -> Any:
    """Evaluate a call segment with arguments.

    ``cache`` memoizes the result per CallSegment within a single ``render_expression`` pass so a
    call that co-occurs with an output isn't evaluated twice (once to build the prompt and again
    to render the result) — which would otherwise invoke the procedure, and its LLM calls, twice.
    """
    if cache is not None and id(seg) in cache:
        return cache[id(seg)]
    eval_args = [_evaluate_arg(rt, arg, local_env) for arg in seg.args]
    callee = rt._get_proc(seg.name)
    result = rt._invoke_procedure(callee, eval_args)
    if cache is not None:
        cache[id(seg)] = result
    return result


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

    merged_segments: list[Segment] = []
    for index, line in enumerate(lines):
        if index > 0:
            merged_segments.append(TextSegment(text="\n"))
        merged_segments.extend(line.segments)

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
    call_cache: dict[int, Any] | None = None,
) -> tuple[list[Any], str]:
    """Prepare segments for rt.m() call, building expression and string parts."""
    expr: list[Any] = []
    parts: list[str] = []

    for index, seg in enumerate(segments):
        if isinstance(seg, TextSegment):
            text = preserve_escaped_whitespace(
                seg.text,
                seg.escaped_whitespace,
                trim_left=index == 0,
                trim_right=index == len(segments) - 1,
            )
            expr.append(text)
            parts.append(text)
        elif isinstance(seg, InputReference):
            expr.append(ReferenceExpr(key=seg.name))
            parts.append("")
        elif isinstance(seg, CallReference):
            value = force(evaluate_call(rt, seg, kwargs, cache=call_cache))
            expr.append(ReferenceExpr(reference=_call_reference(seg, value)))
            parts.append("")
        elif isinstance(seg, InputSegment):
            if context == "template":
                expr.append(rt.i(seg.name))
                parts.append("")
            else:
                parts.append(substitution_text(lookup_value(rt, seg.name, kwargs), seg.name))
        elif isinstance(seg, CallSegment):
            value = force(evaluate_call(rt, seg, kwargs, cache=call_cache))
            _append_computed(expr, parts, substitution_text(value, _call_subject(seg)), context)
        elif isinstance(seg, OutputSegment):
            kedi_type = KediType.from_node(seg.type_expr, seg.type_annotation)
            # None (no annotation) still defaults to str downstream; a written
            # but unresolvable annotation raises instead of silently becoming str.
            pytype = kedi_type.resolve_required(rt, kwargs)
            expr.append(rt.o(seg.name, pytype, binding_contract=kedi_type))
        elif isinstance(seg, (PythonExprSegment, PythonExprNativeSegment)):
            value = force(evaluate_python_inline(rt, seg.code, kwargs, span=seg.loc))
            _append_computed(expr, parts, substitution_text(value, f"`{seg.code}`"), context)

    return expr, "".join(parts)


def _append_computed(expr: list[Any], parts: list[str], text: str, context: str) -> None:
    """Add an evaluated call or inline Python text to a prompt being prepared."""
    expr.append(ComputedExpr(text) if context == "template" else text)
    parts.append(text)


def _call_subject(seg: CallSegment) -> str:
    return f"{seg.name}(...)" if seg.args else f"{seg.name}()"


def _call_reference(seg: CallSegment, value: Any) -> Reference:
    return reference(native_value(value), call=seg.name)


def render_prompt_parts(
    rt: KediRuntime, segments: list[Segment], kwargs: dict[str, Any]
) -> list[Any]:
    """Render a template condition's claim: text parts and references."""
    return _segment_parts(rt, segments, kwargs, context="condition", prompt=True)


def render_expression(
    rt: KediRuntime, segments: list[Segment], kwargs: dict[str, Any], *, context: str
) -> str:
    """Render segments to string, invoking rt.m() only if outputs present."""
    if _has_outputs(segments):
        # Memoize call evaluation across the two passes (prompt build + result render) so a
        # CallSegment co-occurring with an output is evaluated exactly once.
        call_cache: dict[int, Any] = {}
        expr, _ = prepare_segments(rt, segments, kwargs, context=context, call_cache=call_cache)
        token = rt._current_env.set(kwargs)
        try:
            rt.m(expr)
            env = rt._current_env.get()
            if env:
                kwargs.update(env)
        finally:
            rt._current_env.reset(token)
        return _stringify_segments(rt, segments, kwargs, context=context, call_cache=call_cache)

    return _stringify_segments(rt, segments, kwargs, context=context)


def evaluate_binding_rhs(
    rt: KediRuntime, segments: list[Segment], kwargs: dict[str, Any]
) -> tuple[Any, str]:
    """Return a backtick's native value (including None), or NO_NATIVE_VALUE with text.

    Only a sole single-backtick expression keeps its native value; every template, including a
    sole ``<proc()>``, renders text.
    """
    non_ws = [
        s
        for s in segments
        if not (isinstance(s, TextSegment) and not s.escaped_whitespace and s.text.strip() == "")
    ]
    if len(non_ws) == 1 and isinstance(non_ws[0], PythonExprNativeSegment):
        seg = non_ws[0]
        value = force(evaluate_python_inline(rt, seg.code, kwargs, span=seg.loc))
        return value, str(value)
    return NO_NATIVE_VALUE, render_expression(rt, segments, kwargs, context="assign")


# Private helpers


def _evaluate_arg(rt: KediRuntime, arg: CallArg, local_env: dict[str, Any]) -> Any:
    """Evaluate a single call argument: a bare backtick is native, everything else text."""
    if isinstance(arg, LiteralArg):
        native = call_arg_native_backtick(arg)
        if native is not None:
            return evaluate_python_inline(rt, native, local_env, span=arg.loc)
        call_segment = call_arg_call_segment(arg)
        python_segment = call_arg_python_segment(arg)
        segments: list[Segment] = (
            call_arg_render_segments(arg)
            or ([call_segment] if call_segment is not None else [])
            or ([python_segment] if python_segment is not None else [])
        )
        if segments:
            return render_expression(rt, segments, local_env, context="return")
        return arg.value
    elif isinstance(arg, VarArg):
        return lookup_value(rt, arg.name, local_env)
    raise ValueError("Unknown argument type")


def _has_outputs(segments: list[Segment]) -> bool:
    """Check if any segment is an output."""
    return any(isinstance(s, OutputSegment) for s in segments)


def _stringify_segments(
    rt: KediRuntime,
    segments: list[Segment],
    kwargs: dict[str, Any],
    *,
    context: str,
    call_cache: dict[int, Any] | None = None,
) -> str:
    """Convert segments to string, resolving all values."""
    return "".join(_segment_parts(rt, segments, kwargs, context=context, call_cache=call_cache))


def _segment_parts(
    rt: KediRuntime,
    segments: list[Segment],
    kwargs: dict[str, Any],
    *,
    context: str,
    call_cache: dict[int, Any] | None = None,
    prompt: bool = False,
) -> list[Any]:
    """Render segments to text parts; a model prompt (a claim) also keeps its references."""
    parts: list[Any] = []

    for i, seg in enumerate(segments):
        if isinstance(seg, TextSegment):
            parts.append(
                preserve_escaped_whitespace(
                    seg.text,
                    seg.escaped_whitespace,
                    trim_left=i == 0,
                    trim_right=i == len(segments) - 1,
                )
            )
        elif isinstance(seg, (InputReference, CallReference)):
            # The parser accepts references only in model prompts.
            assert prompt, "a reference outside a model prompt"
            if isinstance(seg, InputReference):
                parts.append(binding_reference(seg.name, raw_lookup_value(rt, seg.name, kwargs)))
            else:
                value = force(evaluate_call(rt, seg, kwargs, cache=call_cache))
                parts.append(_call_reference(seg, value))
        elif isinstance(seg, InputSegment):
            parts.append(substitution_text(lookup_value(rt, seg.name, kwargs), seg.name))
        elif isinstance(seg, CallSegment):
            value = force(evaluate_call(rt, seg, kwargs, cache=call_cache))
            parts.append(substitution_text(value, _call_subject(seg)))
        elif isinstance(seg, OutputSegment):
            if seg.name in kwargs:
                parts.append(str(force(kwargs[seg.name])))
            else:
                env = rt._current_env.get()
                if env and seg.name in env:
                    parts.append(str(force(env[seg.name])))
                else:
                    parts.append("")
        elif isinstance(seg, PythonExprNativeSegment) and context == "show":
            # `> show: `value`` displays any value, the way a single backtick always has.
            parts.append(str(force(evaluate_python_inline(rt, seg.code, kwargs, span=seg.loc))))
        elif isinstance(seg, (PythonExprSegment, PythonExprNativeSegment)):
            value = force(evaluate_python_inline(rt, seg.code, kwargs, span=seg.loc))
            parts.append(substitution_text(value, f"`{seg.code}`"))

    return parts
