"""Escape-unescape and dedent helpers ported from the pre-refactor parser.

These functions exist verbatim in the legacy ``segment_parser.py`` and
``block_parser.py`` and are reused by the new tree-sitter CST→AST walker
so behaviour stays bit-identical:

  - ``unescape_text_with_tracking`` decodes ``\\<``, ``\\@``, ``\\\\``,
    ``\\t``, ``\\n``, ``\\s``, etc. into their literal characters and
    records the positions of "escaped whitespace" characters so the
    runtime can preserve them at render time.

  - ``dedent_block`` strips the common leading horizontal whitespace
    from a list of fenced-Python-block lines (tabs = 4 spaces), matching
    the legacy ``_dedent_block`` behaviour.

  - ``unescape_call_arg_text`` is a lighter-weight unescape for the
    body of a call argument's text segments — same character set as
    ``unescape_text_with_tracking`` but discards the whitespace
    position tracking (call args are rendered to strings without
    preserving escaped-WS markers in the AST).
"""

from __future__ import annotations

from typing import List, Tuple


# Characters that may follow a backslash inside template / call-arg text.
_ESCAPABLE = frozenset(
    [
        "<",
        ">",
        "[",
        "]",
        "=",
        "@",
        ",",
        "\\",
        "#",
        "~",
        "`",
        "(",
        ")",
        "t",
        "n",
        "s",
    ]
)


def unescape_text_with_tracking(text: str) -> Tuple[str, set[int]]:
    """Decode template text escapes; track positions of escaped WS chars.

    Returns ``(unescaped_text, escaped_positions)`` where
    ``escaped_positions`` is the set of indices in ``unescaped_text``
    whose character came from a ``\\t``, ``\\n``, or ``\\s`` escape.

    A backslash immediately followed by a literal newline byte
    (``\\<newline>``) is line continuation: it collapses to a single
    space and consumes any leading horizontal whitespace of the next
    line, matching the pre-refactor parser's
    :func:`merge_line_continuations` behaviour. (The pre-refactor
    parser did this in a preprocessing pass before this function ran;
    the tree-sitter scanner instead absorbs `\\<newline>` into the
    text token, so we handle the collapse here.)
    """
    escaped_positions: set[int] = set()
    out: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "=":
            # Legacy ``segment_parser`` rejects a bare `=` inside a
            # template body; users must escape it as `\=` for a
            # literal. The grammar already routes line-start `=` to
            # ``return_stmt`` and bracketed `=` to ``assign_stmt``, so
            # anything reaching here is a mid-text raw `=`.
            raise ValueError("Literal '=' must be escaped as '\\='")

        if ch == "~":
            # Legacy ``_strip_inline_comment`` treats `~~` as a
            # backward-compat escape for a literal `~`, and rejects
            # bare `~` mid-text (a type-definition character must be
            # escaped via `\~` or `~~`). The tree-sitter grammar
            # already routes line-start `~Name(...)` to ``type_def``,
            # so anything we see here is mid-text by construction.
            if i + 1 < n and text[i + 1] == "~":
                out.append("~")
                i += 2
                continue
            raise ValueError("Literal '~' must be escaped as '\\~' or '~~'")

        if ch == "\\" and i + 1 < n:
            nxt = text[i + 1]
            if nxt == "\n":
                # Line continuation: collapse `<trailing WS>\<newline>
                # <leading WS>` to a single space. Matches the
                # pre-refactor `merge_line_continuations` semantics:
                # ``current.rstrip() + " " + s2.lstrip()``.
                while out and out[-1] in (" ", "\t"):
                    out.pop()
                out.append(" ")
                i += 2
                while i < n and (text[i] == " " or text[i] == "\t"):
                    i += 1
                continue
            if nxt == " " or nxt == "\t":
                # `\<WS>+<newline>` is also line continuation (legacy
                # `merge_line_continuations` rstrips before checking
                # for trailing `\`). Look past the WS run; if it
                # terminates at a newline, collapse the entire region
                # — including any trailing WS already accumulated in
                # the output — to a single joining space.
                j = i + 1
                while j < n and (text[j] == " " or text[j] == "\t"):
                    j += 1
                if j < n and text[j] == "\n":
                    while out and out[-1] in (" ", "\t"):
                        out.pop()
                    out.append(" ")
                    i = j + 1
                    while i < n and (text[i] == " " or text[i] == "\t"):
                        i += 1
                    continue
                # Fall through — `\<space>` mid-text is rejected below.
            if nxt in _ESCAPABLE:
                if nxt == "t":
                    out.append("\t")
                    escaped_positions.add(len(out) - 1)
                elif nxt == "n":
                    out.append("\n")
                    escaped_positions.add(len(out) - 1)
                elif nxt == "s":
                    out.append(" ")
                    escaped_positions.add(len(out) - 1)
                else:
                    out.append(nxt)
                i += 2
                continue
            raise ValueError(f"Invalid escape sequence in text: \\{nxt}")
        out.append(ch)
        i += 1
    return "".join(out), escaped_positions


def unescape_call_arg_text(text: str) -> str:
    """Decode call-argument text escapes without tracking WS positions."""
    unescaped, _ = unescape_text_with_tracking(text)
    return unescaped


def preserve_escaped_whitespace(
    text: str, escaped_positions: set[int] | None = None
) -> str:
    """Trim leading/trailing whitespace, but keep `\\s`-escaped chars in place.

    Mirrors the pre-refactor parser's ``preprocessing.preserve_escaped_whitespace``
    helper used by the runtime when rendering template segments.
    """
    if not text:
        return text
    if escaped_positions is None or not escaped_positions:
        return text.strip()
    start = 0
    while start < len(text):
        if text[start] == " " and start not in escaped_positions:
            start += 1
        else:
            break
    end = len(text)
    while end > start:
        if text[end - 1] == " " and (end - 1) not in escaped_positions:
            end -= 1
        else:
            break
    return text[start:end]


def _indent_width(line: str) -> int:
    """Compute the leading-whitespace column of a line, treating tabs as 4."""
    n = 0
    for ch in line:
        if ch == " ":
            n += 1
        elif ch == "\t":
            n += 4
        else:
            break
    return n


def dedent_block(lines: List[str]) -> List[str]:
    """Strip common leading whitespace from a fenced Python block.

    Non-empty lines determine the minimum common indent (tabs=4); each
    line is then stripped of that many leading whitespace columns
    (mixing tabs and spaces in the strip is supported, matching the
    legacy ``_dedent_block``). Empty lines are returned unchanged.
    """
    non_empty = [s for s in lines if s.strip() != ""]
    if not non_empty:
        return list(lines)

    min_indent = min(_indent_width(s) for s in non_empty)
    dedented: List[str] = []
    for s in lines:
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
    return dedented
