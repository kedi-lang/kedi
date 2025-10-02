from __future__ import annotations

from typing import Any, List

from ..core import MarkerRuntime
from .ast import (
    Program,
    ProcedureDef,
    TemplateLine,
    ReturnStmt,
    Segment,
    TextSegment,
    InputSegment,
    CallSegment,
    CallArg,
    LiteralArg,
    VarArg,
    OutputSegment,
)


def _eval_call(rt: MarkerRuntime, seg: CallSegment, local_env: dict[str, Any]) -> Any:
    # Evaluate args: literals stay as strings, variables read from env, nested calls evaluated recursively
    eval_args: List[Any] = []
    for a in seg.args:
        if isinstance(a, LiteralArg) and hasattr(a, "__call_segment__"):
            nested: CallSegment = getattr(a, "__call_segment__")  # type: ignore[attr-defined]
            eval_args.append(_eval_call(rt, nested, local_env))
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


def _compile_template_line(rt: MarkerRuntime, tl: TemplateLine):
    def _fn(**kwargs: Any) -> Any:
        # This helper composes an rt.m expression and executes it
        expr: list[Any] = []
        for s in tl.segments:
            if isinstance(s, TextSegment):
                expr.append(s.text)
            elif isinstance(s, InputSegment):
                expr.append(rt.i(s.name))
            elif isinstance(s, CallSegment):
                # Evaluate call to string and append
                value = _eval_call(rt, s, kwargs)
                expr.append(str(value))
            elif isinstance(s, OutputSegment):
                expr.append(rt.o(s.name))
        return rt.m(expr)

    return _fn


def _compile_return(rt: MarkerRuntime, ret: ReturnStmt):
    def _fn(**kwargs: Any) -> Any:
        # Build a string or a single value depending on segments
        if len(ret.segments) == 1:
            s = ret.segments[0]
            if isinstance(s, TextSegment):
                return s.text.strip()
            if isinstance(s, InputSegment):
                return kwargs.get(s.name)
            if isinstance(s, CallSegment):
                return _eval_call(rt, s, kwargs)

        parts: list[str] = []
        for s in ret.segments:
            if isinstance(s, TextSegment):
                parts.append(s.text)
            elif isinstance(s, InputSegment):
                val = kwargs.get(s.name)
                parts.append("" if val is None else str(val))
            elif isinstance(s, CallSegment):
                parts.append(str(_eval_call(rt, s, kwargs)))
            else:
                # Outputs shouldn't appear in returns, ignore if present
                continue
        composed = "".join(parts).strip()
        return composed

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

    # Top-level main
    if prog.toplevel_return is not None:
        ret_stmt = prog.toplevel_return

        @rt.main
        def _main() -> Any:
            return _compile_return(rt, ret_stmt)()

    return rt


