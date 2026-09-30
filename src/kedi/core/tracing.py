from __future__ import annotations

from dataclasses import replace
from types import TracebackType
from typing import (
    TYPE_CHECKING,
    Any,
)

from kedi.errors import KediExecutionError, KediPythonOrigin, KediTraceFrame
from kedi.errors import build_execution_error as _build_execution_error
from kedi.debugging import checkpoint

if TYPE_CHECKING:
    from kedi.core.runtime import KediRuntime


def trace_frame(self: KediRuntime, frame: KediTraceFrame):
    __tracebackhide__ = True
    frame = self._hydrate_frame(frame)
    stack = self._trace_stack.get()
    token = self._trace_stack.set(stack + (frame,))
    try:
        yield
    except KediExecutionError:
        raise
    except Exception as exc:
        kedi_exc = self.build_execution_error(exc)
        traceback = self._last_python_origin_traceback(exc) or exc.__traceback__
        raise kedi_exc.with_traceback(traceback) from None
    finally:
        self._trace_stack.reset(token)


def current_trace_frames(self: KediRuntime) -> tuple[KediTraceFrame, ...]:
    return self._trace_stack.get()


def build_execution_error(self: KediRuntime, exc: BaseException) -> KediExecutionError:
    checkpoint(self, "exception", payload=exc)
    return _build_execution_error(
        exc,
        frames=self.current_trace_frames(),
        python_origins=self._python_origins,
        source_map=self._source_map,
    )


def _last_python_origin_traceback(self: KediRuntime, exc: BaseException) -> TracebackType | None:
    tb = exc.__traceback__
    last = None
    while tb is not None:
        if tb.tb_frame.f_code.co_filename in self._python_origins:
            last = tb
        tb = tb.tb_next
    if last is None:
        return None
    return TracebackType(
        None,
        last.tb_frame,
        last.tb_lasti,
        last.tb_lineno,
    )


def register_python_origin(
    self: KediRuntime,
    *,
    kind: str,
    code: str,
    span: Any,
    compile_line_offset: int = 0,
    wrapper_line_offset: int = 0,
    kedi_line_offset: int = 0,
    procedure_name: str | None = None,
    traceback_code: str | None = None,
) -> str:
    import linecache

    origin_span = span
    if span is not None and span.source_path is not None:
        origin_span = span
    elif (
        span is not None
        and self._source_map is not None
        and hasattr(self._source_map, "resolve_line")
    ):
        resolved = self._source_map.resolve_line(span.start_line)
        if resolved is not None:
            origin_span = replace(
                span,
                start_line=resolved.source_line,
                end_line=resolved.source_line + max(0, span.end_line - span.start_line),
                source_path=resolved.source_path,
                source_id=resolved.source_id,
            )
    span_key = (
        None
        if origin_span is None
        else (
            origin_span.start_byte,
            origin_span.end_byte,
            origin_span.start_line,
            origin_span.start_col,
            origin_span.end_line,
            origin_span.end_col,
            origin_span.source_path,
            origin_span.source_id,
        )
    )
    origin_key = (
        kind,
        code,
        span_key,
        compile_line_offset,
        wrapper_line_offset,
        kedi_line_offset,
        procedure_name,
        traceback_code,
    )
    with self._origin_counter_lock:
        cached = self._python_origin_keys.get(origin_key)
        if cached is not None:
            return cached

        self._python_origin_counter += 1
        filename = f"<kedi-python:{kind}:{self._python_origin_counter}>"
        self._python_origins[filename] = KediPythonOrigin(
            filename=filename,
            kind=kind,
            code=code,
            span=origin_span,
            compile_line_offset=compile_line_offset,
            wrapper_line_offset=wrapper_line_offset,
            kedi_line_offset=kedi_line_offset,
            procedure_name=procedure_name,
        )
        self._python_origin_keys[origin_key] = filename
        cache_text = "\n" * compile_line_offset + (traceback_code or code)
        lines = [line + "\n" for line in cache_text.splitlines()] or ["\n"]
        linecache.cache[filename] = (len(code), None, lines, filename)
        return filename


def _hydrate_frame(self: KediRuntime, frame: KediTraceFrame) -> KediTraceFrame:
    if frame.source_snippet is not None or frame.span is None or frame.span.source_path is None:
        return frame
    source_map = self._source_map
    line = None
    if source_map is not None and hasattr(source_map, "get_line"):
        line = source_map.get_line(frame.span.source_path, frame.span.start_line)
    return KediTraceFrame(
        kind=frame.kind,
        label=frame.label,
        span=frame.span,
        procedure_name=frame.procedure_name,
        source_snippet=line,
    )
