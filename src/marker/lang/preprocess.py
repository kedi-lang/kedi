from __future__ import annotations

from typing import List

BEGIN = "__BEGIN__"
END = "__END__"


def _indent_width(line: str) -> int:
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


def preprocess_blocks(source: str) -> str:
    """Insert explicit block markers based on indentation changes.

    - Emits BEGIN when indent increases, END when it decreases.
    - Preserves original lines unchanged.
    - Blank lines are preserved but do not affect indentation.
    """
    lines = source.splitlines()
    out: List[str] = []
    stack: List[int] = [0]

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
