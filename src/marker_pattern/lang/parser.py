from __future__ import annotations

import re
from typing import List, Tuple

from .ast import (
    Program,
    ProcedureDef,
    Stmt,
    TemplateLine,
    ReturnStmt,
    PythonBlock,
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
from .preprocess import preprocess_blocks, BEGIN, END


SPECIALS = {"<", ">", "[", "]", "=", "@", ","}


def _unescape(text: str) -> str:
    # Replace doubled specials with single literal
    text = text.replace("<<", "<").replace(">>", ">")
    text = text.replace("[[", "[").replace("]]", "]")
    text = text.replace("==", "=").replace("@@", "@")
    # ',,' is handled in arg parsing; keep here for plain text
    text = text.replace(",,", ",")
    return text


def _parse_segments(body: str) -> List[Segment]:
    segments: List[Segment] = []
    i = 0
    buf: List[str] = []

    def flush_text() -> None:
        if buf:
            segments.append(TextSegment(_unescape("".join(buf))))
            buf.clear()

    while i < len(body):
        ch = body[i]
        nxt = body[i + 1] if i + 1 < len(body) else ""

        # Handle escaped doubles in plain text
        if ch + nxt in ("<<", ">>", "[[", "]]", "==", "@@"):
            buf.append(ch)  # leave as doubled for _unescape to process
            buf.append(nxt)
            i += 2
            continue

        if ch == "<":
            # substitution or call
            flush_text()
            end = _find_matching(body, i, "<", ">")
            inner = body[i + 1 : end]
            seg = _parse_angle(inner.strip())
            segments.append(seg)
            i = end + 1
            continue
        if ch == "[":
            # output
            flush_text()
            end = _find_matching(body, i, "[", "]")
            name = body[i + 1 : end].strip()
            if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
                raise ValueError("Output name must be an identifier")
            segments.append(OutputSegment(name))
            i = end + 1
            continue

        # stray unmatched closers are errors (unless escaped by doubling which was handled above)
        if ch in ("]", ">"):
            raise ValueError(f"Unmatched closer '{ch}' in expression")

        buf.append(ch)
        i += 1

    flush_text()
    return segments


def _find_matching(s: str, start: int, open_ch: str, close_ch: str) -> int:
    depth = 0
    i = start
    while i < len(s):
        ch = s[i]
        nxt = s[i + 1] if i + 1 < len(s) else ""
        if ch + nxt in ("<<", ">>", "[[", "]]", "==", "@@", ",,"):
            i += 2
            continue
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise ValueError("Unmatched delimiter in expression")


def _parse_angle(inner: str) -> Segment:
    # Inline Python: backtick-delimited single-line expression
    if inner.startswith("`") and inner.endswith("`") and len(inner) >= 2:
        code = inner[1:-1]
        return PythonExprSegment(code=code)
    # Determine if call or variable
    if "(" in inner:
        if not inner.endswith(")"):
            raise ValueError("Call expression missing closing ')'")
        name, args_src = inner.split("(", 1)
        name = name.strip()
        args_src = args_src[:-1]  # drop closing )
        args = _parse_args(args_src)
        return CallSegment(name=name, args=args)
    # variable
    name = inner.strip()
    return InputSegment(name=name)


def _parse_args(src: str) -> List[CallArg]:
    args: List[CallArg] = []
    i = 0
    buf: List[str] = []

    def push_literal() -> None:
        if buf:
            lit = _unescape("".join(buf)).strip()
            buf.clear()
            if lit != "":
                args.append(LiteralArg(lit))

    while i < len(src):
        ch = src[i]
        nxt = src[i + 1] if i + 1 < len(src) else ""
        if ch + nxt == ",,":
            buf.append(",")
            i += 2
            continue
        if ch + nxt in ("<<", ">>", "[[", "]]", "==", "@@"):
            buf.append(ch)
            buf.append(nxt)
            i += 2
            continue
        if ch == "<":
            # flush current literal
            push_literal()
            end = _find_matching(src, i, "<", ">")
            inner = src[i + 1 : end]
            seg = _parse_angle(inner.strip())
            if isinstance(seg, InputSegment):
                args.append(VarArg(name=seg.name))
            elif isinstance(seg, CallSegment):
                # Represent nested call result as a literal placeholder to be evaluated later
                # We'll handle this in the compiler by evaluating nested calls at runtime
                # Encode as a special LiteralArg with a marker
                args.append(LiteralArg(f"__CALL__:{seg.name}:{len(seg.args)}"))
                # Also stash the actual segment for the compiler by attaching a private attribute
                setattr(args[-1], "__call_segment__", seg)  # type: ignore[attr-defined]
            elif isinstance(seg, PythonExprSegment):
                # Placeholder literal with attached python segment for runtime eval
                args.append(LiteralArg("__PYEXPR__"))
                setattr(args[-1], "__python_segment__", seg)  # type: ignore[attr-defined]
            else:
                raise ValueError("Unsupported segment in args")
            i = end + 1
            continue
        if ch == ",":
            push_literal()
            i += 1
            continue
        buf.append(ch)
        i += 1

    push_literal()
    return args


def parse_program(source: str) -> Program:
    pre = preprocess_blocks(source)
    lines = pre.splitlines()

    program = Program()
    proc_stack: List[ProcedureDef] = []
    current_body: List[Stmt] | None = None

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped == "":
            i += 1
            continue
        if stripped == BEGIN:
            # Start of a block; current_body is already set by the line before
            i += 1
            continue
        if stripped == END:
            # End of current block
            if proc_stack:
                proc = proc_stack.pop()
                if not proc_stack:
                    program.procedures.append(proc)
                else:
                    # Nested blocks are not expected beyond procedure scope for now
                    pass
                current_body = proc_stack[-1].body if proc_stack else None
            i += 1
            continue

        if stripped.startswith("@"):
            if not stripped.endswith(":"):
                raise ValueError("Procedure header must end with ':'")
            header = stripped[1:-1]
            if "(" not in header:
                raise ValueError("Procedure header missing '('")
            name, params_src = header.split("(", 1)
            name = name.strip()
            params_src = params_src.strip()
            if not params_src.endswith(")"):
                raise ValueError("Procedure header malformed")
            params_csv = params_src[:-1]
            params = [p.strip() for p in params_csv.split(",") if p.strip()]
            if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
                raise ValueError("Procedure name must be an identifier")
            proc = ProcedureDef(name=name, params=params)
            proc_stack.append(proc)
            current_body = proc.body
            i += 1
            continue

        # Inside a procedure body or at top-level
        # Multiline python block fenced by lines containing only ```
        if stripped == "```":
            i += 1
            block_lines: List[str] = []
            while i < len(lines):
                cur = lines[i]
                cur_stripped = cur.strip()
                if cur_stripped == "```":
                    break
                # Skip preprocess markers inside python blocks so they don't leak
                if cur_stripped in (BEGIN, END):
                    i += 1
                    continue
                block_lines.append(cur)
                i += 1
            if i >= len(lines) or lines[i].strip() != "```":
                raise ValueError("Unterminated python block; expected closing ``` line")
            # consume closing fence
            i += 1

            # Normalize indentation relative to minimal indent in block
            # Keep original content otherwise
            # Find minimal leading spaces among non-empty lines
            def indent_of(s: str) -> int:
                n = 0
                for ch in s:
                    if ch == " ":
                        n += 1
                    elif ch == "\t":
                        n += 4
                    else:
                        break
                return n

            non_empty = [s for s in block_lines if s.strip() != ""]
            min_indent = min((indent_of(s) for s in non_empty), default=0)
            dedented = []
            for s in block_lines:
                # remove up to min_indent spaces/tabs-equivalent
                j = 0
                removed = 0
                while j < len(s) and removed < min_indent:
                    if s[j] == " ":
                        removed += 1
                        j += 1
                    elif s[j] == "\t":
                        removed += 4
                        j += 1
                    else:
                        break
                dedented.append(s[j:])

            stmt_block = PythonBlock(code="\n".join(dedented))
            if current_body is None:
                # Top-level python block allowed: treat as no-op or future hook
                pass
            else:
                current_body.append(stmt_block)
            continue

        # Inside a procedure body or at top-level
        if stripped.startswith("="):
            expr_src = stripped[1:]
            segments = _parse_segments(expr_src)
            ret = ReturnStmt(segments=segments)
            if current_body is None:
                program.toplevel_return = ret
            else:
                current_body.append(ret)
            i += 1
            continue

        # Template line
        segments = _parse_segments(stripped)
        stmt = TemplateLine(segments=segments)
        if current_body is None:
            # Allow template lines at top-level to drive outputs before a return
            # For now, treat them as no-ops
            pass
        else:
            current_body.append(stmt)
        i += 1

    # Close any open procedures
    while proc_stack:
        program.procedures.append(proc_stack.pop())

    return program


