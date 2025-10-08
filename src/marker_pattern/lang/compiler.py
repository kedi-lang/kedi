from __future__ import annotations

from typing import Any, List

from ..core import MarkerRuntime
from .ast import (
    Program,
    ProcedureDef,
    TemplateLine,
    ReturnStmt,
    PythonBlock,
    AssignBlockStmt,
    ReturnBlockStmt,
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
    PythonExprNativeSegment,
)
from .types import TypeValidator


def _eval_call(rt: MarkerRuntime, seg: CallSegment, local_env: dict[str, Any]) -> Any:
    # Evaluate args: literals stay as strings, variables read from env, nested calls evaluated recursively
    eval_args: List[Any] = []
    for a in seg.args:
        if isinstance(a, LiteralArg) and hasattr(a, "__native_backtick__"):
            code = getattr(a, "__native_backtick__")  # type: ignore[attr-defined]
            eval_args.append(_eval_python_inline(rt, code, local_env))
        elif isinstance(a, LiteralArg) and hasattr(a, "__render_segments__"):
            segs: list[Segment] = getattr(a, "__render_segments__")  # type: ignore[attr-defined]
            # Render segments to string using current env, honoring outputs
            rendered = _render_expression(rt, segs, local_env, context="return")
            eval_args.append(rendered)
        elif isinstance(a, LiteralArg) and hasattr(a, "__call_segment__"):
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


def _is_value_compatible(pytype: Any, value: Any) -> bool:
    try:
        origin = getattr(pytype, "__origin__", None)
        args = getattr(pytype, "__args__", ())
        if origin in (list, List):  # type: ignore[name-defined]
            if not isinstance(value, list):
                return False
            item_t = args[0] if args else Any
            if item_t in (int, float, str, bool):
                return all(isinstance(v, item_t) for v in value)
            return True
        if pytype in (int, float, str, bool):
            return isinstance(value, pytype)
    except Exception:
        return True
    return True


def _prepare_segments(rt: MarkerRuntime, segments: list[Segment], kwargs: dict[str, Any], *, context: str = "template") -> tuple[list[Any], str]:
    expr: list[Any] = []
    parts: list[str] = []
    for s in segments:
        if isinstance(s, TextSegment):
            expr.append(s.text)
            parts.append(s.text)
        elif isinstance(s, InputSegment):
            if context == "template":
                expr.append(rt.i(s.name))
                # parts for template lines are not used; keep minimal placeholder
                parts.append("")
            else:
                # Resolve from kwargs or current runtime env; raise if missing
                if s.name in kwargs:
                    value = kwargs[s.name]
                else:
                    env = rt._current_env.get()  # type: ignore[attr-defined]
                    if env and s.name in env:
                        value = env[s.name]
                    else:
                        raise KeyError(f"Unknown variable: {s.name}")
                parts.append(str(value))
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
        elif isinstance(s, PythonExprNativeSegment):
            # Native value appears only in assignment-return detection; when rendering, coerce to str
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

    # Only treat backtick-native as native; angle-inline is string-rendered
    if len(non_ws_segments) == 1 and isinstance(non_ws_segments[0], PythonExprNativeSegment):
        value = _eval_python_inline(rt, non_ws_segments[0].code, kwargs)
        # Rendered string representation too
        return value, str(value)

    # Fallback to rendered string (with outputs filled)
    rendered = _render_expression(rt, segments, kwargs, context="assign")
    return None, rendered


def _segments_have_outputs(segments: list[Segment]) -> bool:
    for s in segments:
        if isinstance(s, OutputSegment):
            return True
    return False


def _stringify_segments(rt: MarkerRuntime, segments: list[Segment], kwargs: dict[str, Any], *, context: str) -> str:
    parts: list[str] = []
    for s in segments:
        if isinstance(s, TextSegment):
            parts.append(s.text)
        elif isinstance(s, InputSegment):
            # Strict resolution in non-template contexts
            if s.name in kwargs:
                parts.append(str(kwargs[s.name]))
            else:
                env = rt._current_env.get()  # type: ignore[attr-defined]
                if env and s.name in env:
                    parts.append(str(env[s.name]))
                else:
                    raise KeyError(f"Unknown variable: {s.name}")
        elif isinstance(s, CallSegment):
            value = _eval_call(rt, s, kwargs)
            parts.append(str(value))
        elif isinstance(s, OutputSegment):
            # After rt.m(), outputs live in env/kwargs
            if s.name in kwargs:
                parts.append(str(kwargs[s.name]))
            else:
                env = rt._current_env.get()  # type: ignore[attr-defined]
                if env and s.name in env:
                    parts.append(str(env[s.name]))
                else:
                    parts.append("")
        elif isinstance(s, PythonExprSegment):
            value = _eval_python_inline(rt, s.code, kwargs)
            parts.append(str(value))
        elif isinstance(s, PythonExprNativeSegment):
            value = _eval_python_inline(rt, s.code, kwargs)
            parts.append(str(value))
    return "".join(parts)


def _render_expression(rt: MarkerRuntime, segments: list[Segment], kwargs: dict[str, Any], *, context: str) -> str:
    # Build expr for potential rt.m() and detect outputs
    expr, _ = _prepare_segments(rt, segments, kwargs, context=context)
    if _segments_have_outputs(segments):
        # Ensure current env is set during m()
        token = rt._current_env.set(kwargs)
        try:
            rt.m(expr)
            # Sync kwargs with env
            env = rt._current_env.get()  # type: ignore[attr-defined]
            if env:
                for k, v in env.items():
                    kwargs[k] = v
        finally:
            rt._current_env.reset(token)
    # Now stringify with outputs substituted
    return _stringify_segments(rt, segments, kwargs, context=context)


def _build_python_env(rt: MarkerRuntime, local_env: dict[str, Any]) -> dict[str, Any]:
    # Expose variables and procedures as callables
    env: dict[str, Any] = {}
    # Inject global/top-level environment if present
    try:
        globals_env = getattr(rt, "_globals_env")  # type: ignore[attr-defined]
    except Exception:
        globals_env = None
    if isinstance(globals_env, dict):
        env.update(globals_env)
    # Inject current marker runtime environment (procedure locals)
    try:
        current_env = rt._current_env.get()  # type: ignore[attr-defined]
    except Exception:
        current_env = None
    if current_env:
        env.update(current_env)
    # Then overlay with local_env (procedure/top-level kwargs passed in this call)
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
        # Propagate errors including NameError; tests expect failure
        raise RuntimeError(f"Inline Python error: {type(e).__name__}: {e}") from e


# ------------------------
# New helpers to reduce duplication
# ------------------------

def _detect_rhs_kind(segments: list[Segment]) -> tuple[bool, bool]:
    non_ws: list[Segment] = []
    for s in segments:
        if isinstance(s, TextSegment) and s.text.strip() == "":
            continue
        non_ws.append(s)
    is_backtick_native = len(non_ws) == 1 and isinstance(non_ws[0], PythonExprNativeSegment)
    is_angle_pure = len(non_ws) == 1 and isinstance(non_ws[0], PythonExprSegment)
    return is_backtick_native, is_angle_pure


def _assign_value_from_native(native: Any, pytype: Any | None, *, strict_native: bool) -> Any:
    # strict_native=True for backtick-native; False for angle-inline pure
    if strict_native:
        # backtick-native strict typing
        if isinstance(native, str):
            if pytype is not None and pytype is not str:
                raise TypeError("Annotation type mismatch: got str value")
            return native
        # non-str requires explicit and compatible annotation
        if pytype is None:
            raise TypeError("Native non-string assignment requires explicit type annotation")
        if not _is_value_compatible(pytype, native):
            raise TypeError("Native assignment value does not match annotated type")
        return native
    # angle-inline pure → always string; annotation must not be non-str
    if pytype is not None:
        raise TypeError("Angle-inline pure python cannot have a non-str annotation")
    return str(native)


def _execute_python_block(stmt_code: str, rt: MarkerRuntime, kwargs: dict[str, Any]) -> Any:
    env_exec = _build_python_env(rt, kwargs)
    wrapper_src = (
        "def __marker_block__():\n" + "\n".join("    " + line for line in stmt_code.splitlines())
    )
    try:
        exec(wrapper_src, env_exec, env_exec)
        return env_exec["__marker_block__"]()
    except Exception as e:
        raise RuntimeError(f"Python block error: {e}") from e


# -----------------------------
# Compiler entry points
# -----------------------------

def _compile_template_line(rt: MarkerRuntime, tl: TemplateLine):
    def _fn(**kwargs: Any) -> Any:
        expr, _ = _prepare_segments(rt, tl.segments, kwargs)
        return rt.m(expr)

    return _fn


def _compile_return(rt: MarkerRuntime, ret: ReturnStmt):
    def _fn(**kwargs: Any) -> Any:
        rendered = _render_expression(rt, ret.segments, kwargs, context="return")
        return rendered.strip()

    return _fn


def compile_program(prog: Program) -> MarkerRuntime:
    rt = MarkerRuntime()

    # Register procedures
    for proc in prog.procedures:
        def make_proc(p: ProcedureDef):  # capture
            def _body(**kwargs: Any) -> Any:
                # Enforce parameter type annotations if provided
                if p.param_type_annotations is not None:
                    for param_name in p.params:
                        expected = p.param_type_annotations.get(param_name)
                        if expected:
                            pytype = _resolve_pytype(expected)
                        else:
                            pytype = str  # default to string when unspecified
                        # Retrieve provided value
                        if param_name not in kwargs:
                            raise TypeError(f"Missing parameter: {param_name}")
                        provided = kwargs[param_name]
                        # No conversions allowed: require exact type compatibility
                        if pytype is not None and not _is_value_compatible(pytype, provided):
                            raise TypeError(f"Parameter '{param_name}' does not match annotated type")
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
                        # Execute user python code with env injection (no leakage back into marker env)
                        env_exec = _build_python_env(rt, kwargs)
                        try:
                            exec(stmt.code, env_exec, env_exec)
                        except Exception as e:
                            raise RuntimeError(f"Python block error: {e}") from e
                        # Sync back modifications to existing marker variables only
                        env_rt = rt._current_env.get()  # type: ignore[attr-defined]
                        for var_name in list(kwargs.keys()):
                            if var_name in env_exec:
                                kwargs[var_name] = env_exec[var_name]
                                if env_rt is not None:
                                    env_rt[var_name] = env_exec[var_name]
                    elif isinstance(stmt, AssignBlockStmt):
                        result = _execute_python_block(stmt.code, rt, kwargs)
                        pytype = _resolve_pytype(stmt.type_annotation)
                        # Allow coercion when annotation exists
                        value = result
                        if pytype is not None:
                            # Coerce simple types and list[...] when possible
                            value = _coerce_value_to_type(value, pytype)
                            if not _is_value_compatible(pytype, value):
                                raise TypeError("Native assignment value does not match annotated type")
                        else:
                            # Without annotation only strings are allowed
                            if not isinstance(value, str):
                                raise TypeError("Native non-string assignment requires explicit type annotation")
                        kwargs[stmt.name] = value
                        env = rt._current_env.get()  # type: ignore[attr-defined]
                        if env is not None:
                            env[stmt.name] = value
                    elif isinstance(stmt, AssignStmt):
                        native, rendered = _evaluate_assignment_rhs(rt, stmt.expr, kwargs)
                        is_backtick_native, is_angle_pure = _detect_rhs_kind(stmt.expr)
                        pytype = _resolve_pytype(stmt.type_annotation)
                        if native is not None:
                            if is_backtick_native:
                                value = _assign_value_from_native(native, pytype, strict_native=True)
                            elif is_angle_pure:
                                value = _assign_value_from_native(native, pytype, strict_native=False)
                            else:
                                value = rendered.strip()
                                if pytype is not None:
                                    value = _coerce_value_to_type(value, pytype)
                        else:
                            value = rendered.strip()
                            if pytype is not None:
                                value = _coerce_value_to_type(value, pytype)
                        kwargs[stmt.name] = value
                        env = rt._current_env.get()  # type: ignore[attr-defined]
                        if env is not None:
                            env[stmt.name] = value
                    elif isinstance(stmt, ReturnStmt):
                        last_return = _compile_return(rt, stmt)(**kwargs)
                    elif isinstance(stmt, ReturnBlockStmt):
                        # Execute wrapper if code is a block or contains 'return'; else eval inline expr
                        if ("\n" in stmt.code) or stmt.code.strip().startswith("return"):
                            result = _execute_python_block(stmt.code, rt, kwargs)
                            last_return = result
                        else:
                            last_return = _eval_python_inline(rt, stmt.code, kwargs)
                # Enforce return type at end based on procedure annotation
                anno = _resolve_pytype(p.return_type_annotation)
                if anno is None:
                    # default to str: non-string native returns are illegal
                    if last_return is not None and not isinstance(last_return, str):
                        raise TypeError("Return value must be str; annotate function to return non-string")
                    return "" if last_return is None else str(last_return)
                # annotated non-str: allow native returns; validate type
                if not _is_value_compatible(anno, last_return):
                    raise TypeError("Return value does not match annotated type")
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
            # Expose top-level env to python blocks and called procedures
            try:
                setattr(rt, "_globals_env", kwargs)
            except Exception:
                pass
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
                    # Sync back modifications to existing marker variables only
                    env_rt = rt._current_env.get()  # type: ignore[attr-defined]
                    for var_name in list(kwargs.keys()):
                        if var_name in env_exec:
                            kwargs[var_name] = env_exec[var_name]
                            if env_rt is not None:
                                env_rt[var_name] = env_exec[var_name]
                elif isinstance(stmt, AssignBlockStmt):
                    result = _execute_python_block(stmt.code, rt, kwargs)
                    pytype = _resolve_pytype(stmt.type_annotation)
                    value = result
                    if pytype is not None:
                        value = _coerce_value_to_type(value, pytype)
                        if not _is_value_compatible(pytype, value):
                            raise TypeError("Native assignment value does not match annotated type")
                    else:
                        if not isinstance(value, str):
                            raise TypeError("Native non-string assignment requires explicit type annotation")
                    kwargs[stmt.name] = value
                    env = rt._current_env.get()  # type: ignore[attr-defined]
                    if env is not None:
                        env[stmt.name] = value
                elif isinstance(stmt, AssignStmt):
                    native, rendered = _evaluate_assignment_rhs(rt, stmt.expr, kwargs)
                    is_backtick_native, is_angle_pure = _detect_rhs_kind(stmt.expr)
                    pytype = _resolve_pytype(stmt.type_annotation)
                    if native is not None:
                        if is_backtick_native:
                            value = _assign_value_from_native(native, pytype, strict_native=True)
                        elif is_angle_pure:
                            value = _assign_value_from_native(native, pytype, strict_native=False)
                        else:
                            value = rendered.strip()
                            if pytype is not None:
                                value = _coerce_value_to_type(value, pytype)
                    else:
                        value = rendered.strip()
                        if pytype is not None:
                            value = _coerce_value_to_type(value, pytype)
                    kwargs[stmt.name] = value
                    env = rt._current_env.get()  # type: ignore[attr-defined]
                    if env is not None:
                        env[stmt.name] = value
            # Now render the final return using the accumulated env
            if isinstance(ret_stmt, ReturnStmt):
                return _compile_return(rt, ret_stmt)(**kwargs)
            else:
                # Top-level: always return string rendering for native returns
                if ("\n" in ret_stmt.code) or ret_stmt.code.strip().startswith("return"):
                    result = _execute_python_block(ret_stmt.code, rt, kwargs)
                    return "" if result is None else str(result)
                # Inline python native return
                value = _eval_python_inline(rt, ret_stmt.code, kwargs)
                return "" if value is None else str(value)

    return rt


