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
            # output: [name] or [name: type]
            flush_text()
            end = _find_matching(body, i, "[", "]")
            inner = body[i + 1 : end].strip()
            if ":" in inner:
                name_part, type_part = inner.split(":", 1)
                name = name_part.strip()
                type_anno = type_part.strip()
                if not name or not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
                    raise ValueError("Output name must be an identifier")
                segments.append(OutputSegment(name=name, type_annotation=type_anno))
            else:
                name = inner
                if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
                    raise ValueError("Output name must be an identifier")
                segments.append(OutputSegment(name=name))
            i = end + 1
            continue

        # stray unmatched closers are errors (unless escaped by doubling which was handled above)
        if ch in ("]", ">"):
            raise ValueError(f"Unmatched closer '{ch}' in expression")

        # Enforce escaping for literal '=' anywhere inside expressions/text
        # Return lines are handled at the parser level (line starts with '=')
        # so any '=' encountered here must be a literal and therefore escaped as '=='
        if ch == "=" and nxt != "=":
            raise ValueError("Literal '=' must be written as '==' in template or return expressions")

        buf.append(ch)
        i += 1

    flush_text()
    return segments


def _find_matching(s: str, start: int, open_ch: str, close_ch: str) -> int:
    depth = 0
    i = start
    in_backticks = False
    while i < len(s):
        ch = s[i]
        nxt = s[i + 1] if i + 1 < len(s) else ""

        # Toggle backtick mode; inside backticks, we ignore delimiters
        if ch == "`":
            in_backticks = not in_backticks
            i += 1
            continue

        if not in_backticks:
            # Do not treat '[[', ']]' as escapes when matching brackets; they may be nested types like list[str]
            if ch + nxt in ("<<", ">>", "==", "@@", ",,"):
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
        # NEW ARG PARSING: split into raw argument elements
        def _split_args(src: str) -> List[str]:
            parts: List[str] = []
            buf: List[str] = []
            i = 0
            in_backticks = False
            depth_angle = 0
            while i < len(src):
                ch = src[i]
                nxt = src[i + 1] if i + 1 < len(src) else ""
                if ch + nxt == ",,":
                    buf.append(",")
                    i += 2
                    continue
                if ch == "`":
                    in_backticks = not in_backticks
                    buf.append(ch)
                    i += 1
                    continue
                if not in_backticks and ch == "<":
                    # copy entire <...> block verbatim
                    end = _find_matching(src, i, "<", ">")
                    buf.append(src[i : end + 1])
                    i = end + 1
                    continue
                if not in_backticks and ch == "," and depth_angle == 0:
                    parts.append("".join(buf).strip())
                    buf.clear()
                    i += 1
                    continue
                if not in_backticks and ch == "<":
                    depth_angle += 1
                elif not in_backticks and ch == ">" and depth_angle > 0:
                    depth_angle -= 1
                buf.append(ch)
                i += 1
            if buf:
                parts.append("".join(buf).strip())
            return [p for p in parts if p != ""]

        raw_args = _split_args(args_src)
        args: List[CallArg] = []
        for raw in raw_args:
            # Single backtick-native => native argument
            raw_trim = raw.strip()
            if raw_trim.startswith("`") and raw_trim.endswith("`") and raw_trim.count("`") == 2:
                code = raw_trim[1:-1]
                lit = LiteralArg("__NATIVE_BACKTICK__")
                setattr(lit, "__native_backtick__", code)  # type: ignore[attr-defined]
                args.append(lit)
                continue
            # Otherwise treat as string-rendered mini-template
            segs = _parse_segments(raw)
            lit = LiteralArg("__RENDER__")
            setattr(lit, "__render_segments__", segs)  # type: ignore[attr-defined]
            args.append(lit)
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
            # Support optional return type: name(params) -> type
            ret_anno: str | None = None
            if "->" in header:
                header, ret_anno_src = header.split("->", 1)
                ret_anno = ret_anno_src.strip()
            if "(" not in header:
                raise ValueError("Procedure header missing '('")
            name, params_src = header.split("(", 1)
            name = name.strip()
            params_src = params_src.strip()
            if not params_src.endswith(")"):
                raise ValueError("Procedure header malformed")
            params_csv = params_src[:-1]
            params: List[str] = []
            param_type_annos: dict[str, str | None] = {}
            for p in [pp.strip() for pp in params_csv.split(",") if pp.strip()]:
                if ":" in p:
                    pname, pty = p.split(":", 1)
                    pname = pname.strip()
                    pty = pty.strip()
                    params.append(pname)
                    param_type_annos[pname] = pty
                else:
                    params.append(p)
                    param_type_annos[p] = None
            if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
                raise ValueError("Procedure name must be an identifier")
            proc = ProcedureDef(name=name, params=params, return_type_annotation=ret_anno, param_type_annotations=param_type_annos)
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
                # Top-level python block accumulates into program.toplevel_body
                program.toplevel_body.append(stmt_block)
            else:
                current_body.append(stmt_block)
            continue

        # Inside a procedure body or at top-level
        # Assignment: [name] = expr OR [name: type] = expr
        if stripped.startswith("[") and "]" in stripped and "=" in stripped:
            # Use matching to find the correct closing bracket (support nested type brackets inside)
            close_br = _find_matching(stripped, 0, "[", "]")
            target_inner = stripped[1:close_br].strip()
            rest = stripped[close_br + 1 :].lstrip()
            if not rest.startswith("="):
                raise ValueError("Malformed assignment; expected '=' after [name]")
            rhs_src = rest[1:]
            type_anno: str | None = None
            if ":" in target_inner:
                name_part, type_part = target_inner.split(":", 1)
                name = name_part.strip()
                type_anno = type_part.strip()
            else:
                name = target_inner
            if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
                raise ValueError("Assignment target must be an identifier")
            # Special case: value-returning python block assignment: [name(: type)] = ``` ... ```
            if rhs_src.strip() == "```":
                # parse block similar to above
                # advance to next line to start collecting block contents
                i += 1
                block_lines: List[str] = []
                while i < len(lines):
                    cur = lines[i]
                    cur_stripped = cur.strip()
                    if cur_stripped == "```":
                        break
                    if cur_stripped in (BEGIN, END):
                        i += 1
                        continue
                    block_lines.append(cur)
                    i += 1
                if i >= len(lines) or lines[i].strip() != "```":
                    raise ValueError("Unterminated python block; expected closing ``` line")
                # consume closing fence
                i += 1

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

                stmt_assign_block = AssignBlockStmt(name=name, code="\n".join(dedented), type_annotation=type_anno)
                if current_body is None:
                    program.toplevel_body.append(stmt_assign_block)
                else:
                    current_body.append(stmt_assign_block)
                continue

            # Special case: backtick-only inline python expression (native value assignment)
            rhs_trim = rhs_src.strip()
            if rhs_trim.startswith("`") and rhs_trim.endswith("`") and len(rhs_trim) >= 2 and rhs_trim.count("`") == 2:
                code = rhs_trim[1:-1]
                stmt_assign = AssignStmt(name=name, expr=[PythonExprNativeSegment(code=code)], type_annotation=type_anno)
                if current_body is None:
                    program.toplevel_body.append(stmt_assign)
                else:
                    current_body.append(stmt_assign)
                i += 1
                continue

            rhs_segments = _parse_segments(rhs_src)
            stmt_assign = AssignStmt(name=name, expr=rhs_segments, type_annotation=type_anno)
            if current_body is None:
                program.toplevel_body.append(stmt_assign)
            else:
                current_body.append(stmt_assign)
            i += 1
            continue

        # Inside a procedure body or at top-level
        if stripped.startswith("="):
            expr_src = stripped[1:]
            # Inline python native return: = `expr`
            expr_trim = expr_src.strip()
            if expr_trim.startswith("`") and expr_trim.endswith("`") and len(expr_trim) >= 2 and expr_trim.count("`") == 2:
                code = expr_trim[1:-1]
                ret_block = ReturnBlockStmt(code=code)
                if current_body is None:
                    program.toplevel_return = ret_block
                else:
                    current_body.append(ret_block)
                i += 1
                continue
            # Value-returning python block return: = ``` ... ```
            if expr_src.strip() == "```":
                block_lines: List[str] = []
                i += 1
                while i < len(lines):
                    cur = lines[i]
                    cur_stripped = cur.strip()
                    if cur_stripped == "```":
                        break
                    if cur_stripped in (BEGIN, END):
                        i += 1
                        continue
                    block_lines.append(cur)
                    i += 1
                if i >= len(lines) or lines[i].strip() != "```":
                    raise ValueError("Unterminated python block; expected closing ``` line")
                # consume closing fence
                i += 1

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

                ret_block = ReturnBlockStmt(code="\n".join(dedented))
                if current_body is None:
                    program.toplevel_return = ret_block
                else:
                    current_body.append(ret_block)
                continue

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
            # Accumulate top-level template lines
            program.toplevel_body.append(stmt)
        else:
            current_body.append(stmt)
        i += 1

    # Close any open procedures
    while proc_stack:
        program.procedures.append(proc_stack.pop())

    return program


