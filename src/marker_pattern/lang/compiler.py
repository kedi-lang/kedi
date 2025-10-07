from __future__ import annotations

from typing import Any, List

from ..core import MarkerRuntime
from .ast import (
    Program,
    ProcedureDef,
    TemplateLine,
    ReturnStmt,
    PythonBlock,
    AssignStmt,
    Segment,
    TextSegment,
    InputSegment,
    CallSegment,
    CallArg,
    LiteralArg,
    VarArg,
    OutputSegment,
    PythonExprSegment,
)
from .types import TypeValidator


def _eval_call(rt: MarkerRuntime, seg: CallSegment, local_env: dict[str, Any]) -> Any:
    # Evaluate args: literals stay as strings, variables read from env, nested calls evaluated recursively
    eval_args: List[Any] = []
    for a in seg.args:
        if isinstance(a, LiteralArg) and hasattr(a, "__call_segment__"):
            nested: CallSegment = getattr(a, "__call_segment__")  # type: ignore[attr-defined]
            eval_args.append(_eval_call(rt, nested, local_env))
        elif isinstance(a, LiteralArg) and hasattr(a, "__python_segment__"):
            pyseg: PythonExprSegment = getattr(a, "__python_segment__")  # type: ignore[attr-defined]
            eval_args.append(_eval_python_inline(rt, pyseg.code, local_env))
        elif isinstance(a, LiteralArg):
            eval_args.append(a.value)
        elif isinstance(a, VarArg):
            if a.name in local_env:
                eval_args.append(local_env[a.name])
            else:
                # Try reading from runtime environment
                env = rt._current_env.get()  # type: ignore[attr-defined]
                if env and a.name in env:
                    eval_args.append(env[a.name])
                else:
                    raise KeyError(f"Unknown variable: {a.name}")
        else:
            raise ValueError("Unknown argument type")

    callee = rt._get_proc(seg.name)  # type: ignore[attr-defined]
    return rt._invoke_procedure(callee, eval_args)  # type: ignore[attr-defined]


def _resolve_pytype(type_annotation: str | None) -> Any | None:
    if not type_annotation:
        return None
    try:
        res = TypeValidator(type_annotation).validate_names()
        return res.pytype if res.is_valid else None
    except Exception:
        return None


def _coerce_value_to_type(value: Any, pytype: Any) -> Any:
    # Best-effort coercion for simple builtins and list[...] cases
    try:
        origin = getattr(pytype, "__origin__", None)
        args = getattr(pytype, "__args__", ())
        if origin in (list, List):  # type: ignore[name-defined]
            item_t = args[0] if args else Any
            if isinstance(value, list):
                def cast_item(v: Any) -> Any:
                    try:
                        if item_t in (int, float, str, bool):
                            return item_t(v)
                        return v
                    except Exception:
                        return v
                return [cast_item(v) for v in value]
            return value
        if pytype in (int, float, str, bool):
            return pytype(value)
    except Exception:
        pass
    return value


def _prepare_segments(rt: MarkerRuntime, segments: list[Segment], kwargs: dict[str, Any]) -> tuple[list[Any], str]:
    expr: list[Any] = []
    parts: list[str] = []
    for s in segments:
        if isinstance(s, TextSegment):
            expr.append(s.text)
            parts.append(s.text)
        elif isinstance(s, InputSegment):
            expr.append(rt.i(s.name))
            parts.append("" if s.name not in kwargs else str(kwargs[s.name]))
        elif isinstance(s, CallSegment):
            value = _eval_call(rt, s, kwargs)
            expr.append(str(value))
            parts.append(str(value))
        elif isinstance(s, OutputSegment):
            pytype = _resolve_pytype(s.type_annotation)
            expr.append(rt.o(s.name, pytype))
        elif isinstance(s, PythonExprSegment):
            value = _eval_python_inline(rt, s.code, kwargs)
            expr.append(str(value))
            parts.append(str(value))
    return expr, "".join(parts)


def _evaluate_assignment_rhs(rt: MarkerRuntime, segments: list[Segment], kwargs: dict[str, Any]) -> tuple[Any | None, str]:
    # If RHS is a pure single python expression, return its native value
    non_ws_segments: list[Segment] = []
    for s in segments:
        if isinstance(s, TextSegment) and s.text.strip() == "":
            continue
        non_ws_segments.append(s)

    if len(non_ws_segments) == 1 and isinstance(non_ws_segments[0], PythonExprSegment):
        value = _eval_python_inline(rt, non_ws_segments[0].code, kwargs)
        # Rendered string representation too
        return value, str(value)

    # Fallback to rendered string
    _, rendered = _prepare_segments(rt, segments, kwargs)
    return None, rendered


def _build_python_env(rt: MarkerRuntime, local_env: dict[str, Any]) -> dict[str, Any]:
    # Expose variables and procedures as callables
    env: dict[str, Any] = {}
    env.update(local_env)

    # Wrap registered procedures to be callable from inline Python
    for name, fn in rt._procedures.items():  # type: ignore[attr-defined]
        def make_wrapper(proc_name: str):
            def _callable(*args: Any):
                callee = rt._get_proc(proc_name)  # type: ignore[attr-defined]
                return rt._invoke_procedure(callee, list(args))  # type: ignore[attr-defined]
            return _callable
        env[name] = make_wrapper(name)

    # Full Python builtins access as requested
    import builtins as _builtins  # local import to avoid polluting env
    env["__builtins__"] = _builtins.__dict__
    return env


def _eval_python_inline(rt: MarkerRuntime, code: str, local_env: dict[str, Any]) -> Any:
    env = _build_python_env(rt, local_env)
    try:
        return eval(code, env, env)
    except SyntaxError as e_syn:
        # Fall back to exec for statements like 'assert', assignments, etc.
        try:
            exec(code, env, env)
            return ""
        except Exception as e:
            raise RuntimeError(f"Inline Python error: {type(e).__name__}: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Inline Python error: {type(e).__name__}: {e}") from e


def _compile_template_line(rt: MarkerRuntime, tl: TemplateLine):
    def _fn(**kwargs: Any) -> Any:
        expr, _ = _prepare_segments(rt, tl.segments, kwargs)
        return rt.m(expr)

    return _fn


def _compile_return(rt: MarkerRuntime, ret: ReturnStmt):
    def _fn(**kwargs: Any) -> Any:
        _, rendered = _prepare_segments(rt, ret.segments, kwargs)
        return rendered.strip()

    return _fn


def compile_program(prog: Program) -> MarkerRuntime:
    rt = MarkerRuntime()

    # Register procedures
    for proc in prog.procedures:
        def make_proc(p: ProcedureDef):  # capture
            def _body(**kwargs: Any) -> Any:
                # Execute template lines in order; last return decides
                last_return: Any = None
                for stmt in p.body:
                    if isinstance(stmt, TemplateLine):
                        _compile_template_line(rt, stmt)(**kwargs)
                        # Sync kwargs with current runtime env (capturing outputs)
                        env = rt._current_env.get()  # type: ignore[attr-defined]
                        if env:
                            for k, v in env.items():
                                kwargs[k] = v
                    elif isinstance(stmt, PythonBlock):
                        # Execute user python code with env injection
                        env_exec = _build_python_env(rt, kwargs)
                        try:
                            exec(stmt.code, env_exec, env_exec)
                        except Exception as e:
                            raise RuntimeError(f"Python block error: {e}") from e
                        # sync back any modified/created names to kwargs
                        for k, v in env_exec.items():
                            if k.startswith("__"):
                                continue
                            kwargs[k] = v
                    elif isinstance(stmt, AssignStmt):
                        # Evaluate the RHS; if pure python expr, capture native value
                        native, rendered = _evaluate_assignment_rhs(rt, stmt.expr, kwargs)
                        value = native if native is not None else rendered.strip()
                        # Coerce to annotated type if provided
                        pytype = _resolve_pytype(stmt.type_annotation)
                        if pytype is not None:
                            value = _coerce_value_to_type(value, pytype)
                        kwargs[stmt.name] = value
                        # Also update runtime env if present (so subsequent rt.m can see it)
                        env = rt._current_env.get()  # type: ignore[attr-defined]
                        if env is not None:
                            env[stmt.name] = value
                    elif isinstance(stmt, ReturnStmt):
                        last_return = _compile_return(rt, stmt)(**kwargs)
                return last_return

            # Dynamically create a function with the exact parameter names
            param_list = ", ".join(p.params)
            args_map = ", ".join(f"{n}={n}" for n in p.params)
            src = (
                "def _impl(" + param_list + "):\n"
                "    return _body(" + args_map + ")\n"
            )
            ns: dict[str, Any] = {"_body": _body}
            exec(src, ns, ns)
            fn = ns["_impl"]
            fn.__name__ = p.name
            return fn

        rt.procedure(make_proc(proc))

    # Top-level main: execute accumulated top-level statements before rendering return
    if prog.toplevel_return is not None:
        ret_stmt = prog.toplevel_return

        @rt.main
        def _main() -> Any:
            # Execute top-level template lines and python blocks to populate env
            kwargs: dict[str, Any] = {}
            for stmt in prog.toplevel_body:
                if isinstance(stmt, TemplateLine):
                    _compile_template_line(rt, stmt)(**kwargs)
                    env = rt._current_env.get()  # type: ignore[attr-defined]
                    if env:
                        for k, v in env.items():
                            kwargs[k] = v
                elif isinstance(stmt, PythonBlock):
                    env_exec = _build_python_env(rt, kwargs)
                    try:
                        exec(stmt.code, env_exec, env_exec)
                    except Exception as e:
                        raise RuntimeError(f"Python block error: {e}") from e
                    for k, v in env_exec.items():
                        if k.startswith("__"):
                            continue
                        kwargs[k] = v
                elif isinstance(stmt, AssignStmt):
                    native, rendered = _evaluate_assignment_rhs(rt, stmt.expr, kwargs)
                    value = native if native is not None else rendered.strip()
                    pytype = _resolve_pytype(stmt.type_annotation)
                    if pytype is not None:
                        value = _coerce_value_to_type(value, pytype)
                    kwargs[stmt.name] = value
                    env = rt._current_env.get()  # type: ignore[attr-defined]
                    if env is not None:
                        env[stmt.name] = value
            # Now render the final return using the accumulated env
            return _compile_return(rt, ret_stmt)(**kwargs)

    return rt


