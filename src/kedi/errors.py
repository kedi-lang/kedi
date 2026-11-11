from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .lang.ast import Span


@dataclass(frozen=True)
class KediTraceFrame:
    kind: str
    label: str
    span: Span | None = None
    procedure_name: str | None = None
    source_snippet: str | None = None


@dataclass(frozen=True)
class KediPythonTrace:
    filename: str
    kind: str
    code: str
    span: Span | None
    traceback_lineno: int
    python_lineno: int
    python_col: int | None = None
    kedi_lineno: int | None = None
    procedure_name: str | None = None
    source_snippet: str | None = None


@dataclass(frozen=True)
class KediPythonOrigin:
    filename: str
    kind: str
    code: str
    span: Span | None
    compile_line_offset: int = 0
    kedi_line_offset: int = 0
    procedure_name: str | None = None

    def to_trace(
        self,
        traceback_lineno: int,
        *,
        source_map: Any | None = None,
        python_col: int | None = None,
    ) -> KediPythonTrace:
        python_lineno = max(
            1,
            traceback_lineno - self.compile_line_offset,
        )
        kedi_lineno = None
        source_snippet = None
        if self.span is not None:
            kedi_lineno = self.span.start_line + self.kedi_line_offset + python_lineno - 1
            if (
                source_map is not None
                and self.span.source_path is not None
                and hasattr(source_map, "get_line")
            ):
                source_snippet = source_map.get_line(self.span.source_path, kedi_lineno)
        return KediPythonTrace(
            filename=self.filename,
            kind=self.kind,
            code=self.code,
            span=self.span,
            traceback_lineno=traceback_lineno,
            python_lineno=python_lineno,
            python_col=python_col,
            kedi_lineno=kedi_lineno,
            procedure_name=self.procedure_name,
            source_snippet=source_snippet,
        )


@dataclass
class KediExecutionError(RuntimeError):
    message: str
    frames: tuple[KediTraceFrame, ...] = field(default_factory=tuple)
    python_traces: tuple[KediPythonTrace, ...] = field(default_factory=tuple)
    original: BaseException | None = None
    tool_call: Any | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        RuntimeError.__init__(self, self.message)

    def render(self, *, use_color: bool = False) -> str:
        from .error_rendering import render_execution_error

        return render_execution_error(self, use_color=use_color)

    def __str__(self) -> str:
        return self.render()

    def __repr__(self) -> str:
        return self.render()


class KediTypeExecutionError(KediExecutionError, TypeError):
    pass


class KediValueExecutionError(KediExecutionError, ValueError):
    pass


class KediKeyExecutionError(KediExecutionError, KeyError):
    pass


class KediNameError(KediExecutionError, NameError):
    pass


class LoopIterationLimitError(RuntimeError):
    """Raised when a conditional loop remains true beyond its configured limit."""


class KediLoopIterationLimitError(KediExecutionError, LoopIterationLimitError):
    pass


def python_error_message(exc: BaseException, kind: str) -> str:
    prefix_map = {
        "inline": "Inline Python error",
        "block": "Python block error",
        "side_effects": "Inline Python error",
        "prelude": "Python prelude error",
        "type_expression": "Python type expression error",
    }
    prefix = prefix_map.get(kind, "Python error")
    detail = str(exc)
    if detail:
        return f"{prefix}: {type(exc).__name__}: {detail}"
    return f"{prefix}: {type(exc).__name__}"


def collect_python_traces(
    exc: BaseException,
    origins: dict[str, KediPythonOrigin],
    *,
    source_map: Any | None = None,
) -> tuple[KediPythonTrace, ...]:
    traces: list[KediPythonTrace] = []
    if isinstance(exc, SyntaxError) and exc.filename in origins and exc.lineno is not None:
        traces.append(
            origins[exc.filename].to_trace(
                exc.lineno,
                source_map=source_map,
                python_col=exc.offset,
            )
        )
    tb = exc.__traceback__
    while tb is not None:
        filename = tb.tb_frame.f_code.co_filename
        origin = origins.get(filename)
        if origin is not None:
            traces.append(origin.to_trace(tb.tb_lineno, source_map=source_map))
        tb = tb.tb_next
    return tuple(traces)


def build_execution_error(
    exc: BaseException,
    *,
    frames: tuple[KediTraceFrame, ...],
    python_origins: dict[str, KediPythonOrigin],
    source_map: Any | None = None,
) -> KediExecutionError:
    python_traces = collect_python_traces(exc, python_origins, source_map=source_map)
    if python_traces:
        message = python_error_message(exc, python_traces[-1].kind)
    else:
        message = str(exc) or type(exc).__name__
    error_cls: type[KediExecutionError] = KediExecutionError
    if isinstance(exc, NameError):
        error_cls = KediNameError
    elif isinstance(exc, LoopIterationLimitError):
        error_cls = KediLoopIterationLimitError
    elif isinstance(exc, KeyError):
        error_cls = KediKeyExecutionError
    elif isinstance(exc, TypeError):
        error_cls = KediTypeExecutionError
    elif isinstance(exc, ValueError):
        error_cls = KediValueExecutionError
    return error_cls(
        message=message,
        frames=frames,
        python_traces=python_traces,
        original=exc,
        tool_call=getattr(exc, "tool_call", None),
    )


def frame_label(kind: str, procedure_name: str | None = None) -> str:
    labels = {
        "program": "top-level program",
        "procedure": f"procedure {procedure_name}" if procedure_name else "procedure",
        "template": "template line",
        "variable_init": "variable initialization",
        "variable_init_block": "variable initialization block",
        "assignment": "assignment",
        "assignment_block": "assignment block",
        "return": "return",
        "return_block": "return block",
        "python_inline": "inline python",
        "python_block": "python block",
        "python_side_effects": "python statement",
        "prelude": "prelude",
        "type_expression": "type expression",
    }
    return labels.get(kind, kind.replace("_", " "))
