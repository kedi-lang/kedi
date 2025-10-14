"""Template segment parsing."""
from __future__ import annotations
from ..ast import (
    Segment, TextSegment, InputSegment, OutputSegment,
    CallSegment, PythonExprSegment, CallArg, LiteralArg, VarArg
)
from .preprocessing import unescape_text
import re


def parse_segments(body: str) -> list[Segment]:
    """Parse a template line body into segments."""
    segments: list[Segment] = []
    i = 0
    buf: list[str] = []
    
    def flush_text():
        if buf:
            segments.append(TextSegment(unescape_text("".join(buf))))
            buf.clear()
    
    while i < len(body):
        ch = body[i]
        nxt = body[i + 1] if i + 1 < len(body) else ""

        # Parse angle brackets (substitutions)
        if ch == "<" and (i == 0 or body[i - 1] != "\\"):
            flush_text()
            end = find_matching_delimiter(body, i, "<", ">")
            inner = body[i + 1:end]
            segments.append(parse_angle_expression(inner.strip()))
            i = end + 1
            continue
        
        # Parse square brackets (outputs)
        if ch == "[" and (i == 0 or body[i - 1] != "\\"):
            flush_text()
            end = find_matching_delimiter(body, i, "[", "]")
            inner = body[i + 1:end].strip()
            segments.append(_parse_output(inner))
            i = end + 1
            continue
        
        # Validate unmatched closers
        if ch in ("]", ">") and (i == 0 or body[i - 1] != "\\"):
            raise ValueError(f"Unmatched closer '{ch}'")
        
        # Validate literal equals
        if ch == "=" and (i == 0 or body[i - 1] != "\\"):
            raise ValueError("Literal '=' must be escaped as '\\='")
        
        buf.append(ch)
        i += 1
    
    flush_text()
    return segments


def parse_angle_expression(inner: str) -> Segment:
    """Parse <...> expression (variable, call, or Python)."""
    # Inline Python
    if _is_wrapped_backticks(inner):
        code = inner[1:-1].replace("\\`", "`")
        return PythonExprSegment(code=code)
    
    # Call or variable
    if "(" in inner:
        if not inner.endswith(")"):
            raise ValueError("Call expression missing ')'")
        name, args_src = inner.split("(", 1)
        name = name.strip()
        args_src = args_src[:-1]
        args = parse_call_arguments(args_src)
        return CallSegment(name=name, args=args)
    
    # Variable
    return InputSegment(name=inner.strip())


def parse_call_arguments(args_src: str) -> list[CallArg]:
    """Parse comma-separated call arguments."""
    raw_args = _split_arguments(args_src)
    args: list[CallArg] = []
    
    for raw in raw_args:
        raw_trim = raw.strip()
        
        # Native backtick argument
        if _is_wrapped_backticks(raw_trim):
            code = raw_trim[1:-1].replace("\\`", "`")
            lit = LiteralArg("__NATIVE_BACKTICK__")
            setattr(lit, "__native_backtick__", code)
            args.append(lit)
            continue
        
        # Parse generic segments; if it's a single pure python or call, preserve natively
        segs = parse_segments(raw_trim)
        lit = LiteralArg("__RENDER__")
        if len(segs) == 1:
            only = segs[0]
            from ..ast import PythonExprSegment, CallSegment
            if isinstance(only, PythonExprSegment):
                setattr(lit, "__python_segment__", only)
                args.append(lit)
                continue
            if isinstance(only, CallSegment):
                setattr(lit, "__call_segment__", only)
                args.append(lit)
                continue
        setattr(lit, "__render_segments__", segs)
        args.append(lit)
    
    return args


def find_matching_delimiter(s: str, start: int, open_ch: str, close_ch: str) -> int:
    """Find matching delimiter considering backticks."""
    depth = 0
    i = start
    in_backticks = False
    
    while i < len(s):
        ch = s[i]
        nxt = s[i + 1] if i + 1 < len(s) else ""
        
        # Skip escaped delimiters in text context (outside backticks)
        if not in_backticks and ch == "\\" and nxt:
            i += 2
            continue
        
        if ch == "`":
            in_backticks = not in_backticks
            i += 1
            continue
        
        if not in_backticks:
            if ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    return i
        i += 1
    
    raise ValueError("Unmatched delimiter")


def _parse_output(inner: str) -> OutputSegment:
    """Parse output specification [name] or [name: type]."""
    if ":" in inner:
        name_part, type_part = inner.split(":", 1)
        name = name_part.strip()
        type_anno = type_part.strip()
        if not name or not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
            raise ValueError("Output name must be identifier")
        return OutputSegment(name=name, type_annotation=type_anno)
    
    name = inner
    if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
        raise ValueError("Output name must be identifier")
    return OutputSegment(name=name)


def _split_arguments(src: str) -> list[str]:
    """Split arguments by comma, respecting nesting."""
    parts: list[str] = []
    buf: list[str] = []
    i = 0
    in_backticks = False
    depth_angle = 0
    
    while i < len(src):
        ch = src[i]
        nxt = src[i + 1] if i + 1 < len(src) else ""
        
        # Escaped comma becomes literal comma
        if ch == "\\" and nxt == ",":
            buf.append(",")
            i += 2
            continue
        
        if ch == "`":
            in_backticks = not in_backticks
            buf.append(ch)
            i += 1
            continue
        
        if not in_backticks and ch == "<":
            end = find_matching_delimiter(src, i, "<", ">")
            buf.append(src[i:end + 1])
            i = end + 1
            continue
        
        if not in_backticks and ch == "," and depth_angle == 0:
            parts.append("".join(buf).strip())
            buf.clear()
            i += 1
            continue
        
        buf.append(ch)
        i += 1
    
    if buf:
        parts.append("".join(buf).strip())
    
    return [p for p in parts if p]


def _is_wrapped_backticks(s: str) -> bool:
    """Return True if the string is wrapped by unescaped backticks at both ends,
    and contains no other unescaped backticks inside.
    """
    if len(s) >= 2 and s[0] == "`" and s[-1] == "`" and (len(s) < 2 or s[-2] != "\\"):
        # ensure no other unescaped backticks inside
        i = 1
        while i < len(s) - 1:
            if s[i] == "\\":
                i += 2
                continue
            if s[i] == "`":
                return False
            i += 1
        return True
    return False