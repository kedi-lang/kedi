"""Text preprocessing for Marker parser.

Handles comment stripping, line continuations, indentation-based blocks,
and text unescaping.
"""
from __future__ import annotations

SPECIALS = {"<", ">", "[", "]", "=", "@", ",", "~"}

# Block markers for indentation-based structure
BEGIN = "__BEGIN__"
END = "__END__"


def strip_comments(source: str) -> str:
    """Remove inline and block comments."""
    lines = source.splitlines()
    result: list[str] = []
    block_stack: list[str] = []
    in_python = False
    
    for line in lines:
        stripped = line.strip()
        
        # Check for line comments first
        if stripped.startswith("#") and not stripped.startswith("##"):
            continue
        
        # Track Python blocks
        if stripped.endswith("```"):
            result.append(line)
            in_python = not in_python
            continue
        
        if in_python:
            result.append(line)
            continue
        
        # Handle block comments
        if stripped == "###":
            if block_stack and block_stack[-1] == "###":
                block_stack.pop()
            else:
                block_stack.append("###")
            continue
        
        if block_stack:
            if "###" in stripped:
                raise ValueError("Invalid use of '###' inside block comment")
            continue
        
        # Outside block comments: ### must occupy entire line
        if "###" in stripped:
            raise ValueError("'###' must occupy an entire line for block comments")
        
        # Process line, stripping inline comments
        result.append(_strip_inline_comment(line))
    
    if block_stack:
        raise ValueError("Unterminated block comment")
    
    return "\n".join(result)


def merge_line_continuations(source: str) -> str:
    """Merge lines ending with backslash."""
    lines = source.splitlines()
    merged: list[str] = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        s = line.rstrip()
        
        if s.endswith("\\") and not s.endswith("\\\\"):
            # Merge continuation
            current = s[:-1]
            j = i + 1
            while j < len(lines):
                nxt = lines[j]
                t = nxt.rstrip()
                s2 = nxt.lstrip()
                
                if t.endswith("\\") and not t.endswith("\\\\"):
                    current = current.rstrip() + " " + s2.rstrip()[:-1].lstrip()
                    j += 1
                else:
                    current = current.rstrip() + " " + s2.lstrip()
                    break
            merged.append(current)
            i = j + 1
        else:
            merged.append(line)
            i += 1
    
    return "\n".join(merged)


def unescape_text(text: str) -> str:
    """Unescape doubled special characters."""
    text = text.replace("<<", "<").replace(">>", ">")
    text = text.replace("[[", "[").replace("]]", "]")
    text = text.replace("==", "=").replace("@@", "@")
    text = text.replace("##", "#").replace("~~", "~")
    text = text.replace(",,", ",")
    text = text.replace("\\\\", "\\")
    return text


def _strip_inline_comment(line: str) -> str:
    """Strip inline comment from a line."""
    processed: list[str] = []
    i = 0
    
    while i < len(line):
        ch = line[i]
        nxt = line[i + 1] if i + 1 < len(line) else ""
        
        if ch == "#":
            if nxt == "#":
                processed.append("#")
                i += 2
                continue
            break  # Rest is comment
        
        if ch == "~":
            if nxt == "~":
                processed.append("~")
                i += 2
                continue
            # Check if it's a type definition start
            if line[:i].strip() == "" and line[i:].lstrip().startswith("~"):
                processed.append(ch)
                i += 1
                continue
            raise ValueError("Literal '~' must be escaped as '~~'")
        
        processed.append(ch)
        i += 1
    
    return "".join(processed)


def preprocess_blocks(source: str) -> str:
    """Insert explicit block markers based on indentation changes.

    - Emits BEGIN when indent increases, END when it decreases.
    - Preserves original lines unchanged.
    - Blank lines are preserved but do not affect indentation.
    """
    lines = source.splitlines()
    out: list[str] = []
    stack: list[int] = [0]

    for raw in lines:
        if raw.strip() == "":
            out.append(raw)
            continue

        width = _indent_width(raw)
        current = stack[-1]
        if width > current:
            stack.append(width)
            out.append(BEGIN)
        else:
            while width < current and len(stack) > 1:
                stack.pop()
                current = stack[-1]
                out.append(END)
        out.append(raw)

    # Close any remaining blocks
    while len(stack) > 1:
        stack.pop()
        out.append(END)

    return "\n".join(out)


def _indent_width(line: str) -> int:
    """Calculate the indentation width of a line (tabs count as 4 spaces)."""
    count = 0
    for ch in line:
        if ch == " ":
            count += 1
        elif ch == "\t":
            # Treat a tab as width 4 consistently
            count += 4
        else:
            break
    return count