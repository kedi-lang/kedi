"""Expression and segment evaluation for Marker compiler."""
from __future__ import annotations

from typing import Any
from ..ast import (
    Segment, TextSegment, InputSegment, CallSegment,
    OutputSegment, PythonExprSegment, PythonExprNativeSegment, CallArg, LiteralArg, VarArg
)
from ...core import MarkerRuntime
from .environment import lookup_value, build_python_env


def evaluate_call(rt: MarkerRuntime, seg: CallSegment, local_env: dict[str, Any]) -> Any:
    """Evaluate a call segment with arguments."""
    eval_args = [_evaluate_arg(rt, arg, local_env) for arg in seg.args]
    callee = rt._get_proc(seg.name)
    return rt._invoke_procedure(callee, eval_args)


def evaluate_python_inline(rt: MarkerRuntime, code: str, local_env: dict[str, Any]) -> Any:
    """Evaluate inline Python expression or statement."""
    env = build_python_env(rt, local_env)
    try:
        return eval(code, env, env)
    except SyntaxError:
        # Try exec for statements
        try:
            exec(code, env, env)
            return ""
        except Exception as e:
            raise RuntimeError(f"Inline Python error: {type(e).__name__}: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Inline Python error: {type(e).__name__}: {e}") from e


def execute_python_block(code: str, rt: MarkerRuntime, kwargs: dict[str, Any]) -> Any:
    """Execute a multi-line Python block with return value."""
    env_exec = build_python_env(rt, kwargs)
    
    # Wrap in function to handle parameter scoping correctly
    param_names = list(kwargs.keys())
    params_str = ", ".join(param_names)
    wrapper_src = (
        f"def __marker_block__({params_str}):\n" + 
        "\n".join("    " + line for line in code.splitlines())
    )
    
    try:
        exec(wrapper_src, env_exec, env_exec)
        return env_exec["__marker_block__"](**kwargs)
    except Exception as e:
        raise RuntimeError(f"Python block error: {e}") from e


def execute_inline_side_effects(rt: MarkerRuntime, code: str, env_map: dict[str, Any]) -> None:
    """Execute Python code for side effects, syncing back existing vars only."""
    env_exec = build_python_env(rt, env_map)
    try:
        exec(code, env_exec, env_exec)
    except Exception as e:
        raise RuntimeError(f"Inline Python error: {type(e).__name__}: {e}") from e
    
    # Sync back only existing variables
    env_rt = rt._current_env.get()
    for var_name in list(env_map.keys()):
        if var_name in env_exec:
            env_map[var_name] = env_exec[var_name]
            if env_rt is not None:
                env_rt[var_name] = env_exec[var_name]


def prepare_segments(
    rt: MarkerRuntime,
    segments: list[Segment],
    kwargs: dict[str, Any],
    *,
    context: str = "template"
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
            from .type_system import resolve_type
            pytype = resolve_type(rt, seg.type_annotation, env=kwargs)
            expr.append(rt.o(seg.name, pytype))
        elif isinstance(seg, (PythonExprSegment, PythonExprNativeSegment)):
            value = evaluate_python_inline(rt, seg.code, kwargs)
            expr.append(str(value))
            parts.append(str(value))
    
    return expr, "".join(parts)


def render_expression(
    rt: MarkerRuntime,
    segments: list[Segment],
    kwargs: dict[str, Any],
    *,
    context: str
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
    rt: MarkerRuntime,
    segments: list[Segment],
    kwargs: dict[str, Any]
) -> tuple[Any | None, str]:
    """Evaluate assignment RHS, returning native value if pure expression."""
    non_ws = [s for s in segments if not (isinstance(s, TextSegment) and s.text.strip() == "")]
    
    # Single native expression or call
    if len(non_ws) == 1:
        seg = non_ws[0]
        if isinstance(seg, PythonExprNativeSegment):
            value = evaluate_python_inline(rt, seg.code, kwargs)
            return value, str(value)
        if isinstance(seg, CallSegment):
            value = evaluate_call(rt, seg, kwargs)
            return value, str(value)
    
    # Fallback to rendered string
    return None, render_expression(rt, segments, kwargs, context="assign")


# Private helpers

def _evaluate_arg(rt: MarkerRuntime, arg: CallArg, local_env: dict[str, Any]) -> Any:
    """Evaluate a single call argument."""
    if isinstance(arg, LiteralArg):
        if hasattr(arg, "__native_backtick__"):
            code = getattr(arg, "__native_backtick__")
            return evaluate_python_inline(rt, code, local_env)
        if hasattr(arg, "__render_segments__"):
            segs = getattr(arg, "__render_segments__")
            return render_expression(rt, segs, local_env, context="return")
        if hasattr(arg, "__call_segment__"):
            nested = getattr(arg, "__call_segment__")
            return evaluate_call(rt, nested, local_env)
        if hasattr(arg, "__python_segment__"):
            pyseg = getattr(arg, "__python_segment__")
            return evaluate_python_inline(rt, pyseg.code, local_env)
        return arg.value
    elif isinstance(arg, VarArg):
        return lookup_value(rt, arg.name, local_env)
    raise ValueError("Unknown argument type")


def _has_outputs(segments: list[Segment]) -> bool:
    """Check if any segment is an output."""
    return any(isinstance(s, OutputSegment) for s in segments)


def _stringify_segments(
    rt: MarkerRuntime,
    segments: list[Segment],
    kwargs: dict[str, Any],
    *,
    context: str
) -> str:
    """Convert segments to string, resolving all values."""
    parts: list[str] = []
    for seg in segments:
        if isinstance(seg, TextSegment):
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
            value = evaluate_python_inline(rt, seg.code, kwargs)
            parts.append(str(value))
    return "".join(parts)