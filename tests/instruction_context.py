"""The guidance a model holds at a request, across Pydantic AI's two delivery styles.

Before 2.6x every instruction block was rendered into ``info.instructions`` on each request. Later
releases keep a block's first rendering there and deliver later changes in the conversation: the
first change as a full ``<context id="...">`` element, then small changes as a unified diff in a
``<context_diff id="...">`` element. The newest statement for an id is the one in effect, and a
withdrawn block says so instead of carrying text. On the older release nothing is appended, so this
returns ``info.instructions`` unchanged.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from pydantic_ai.messages import ModelMessage, UserPromptPart

_ELEMENT = re.compile(r'<(context|context_diff) id="([^"]*)">\n(.*?)\n</\1>', re.DOTALL)
_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@")
_WITHDRAWN = "This context has been withdrawn"


def _apply_diff(previous: str, diff: str) -> str:
    """Apply a ``difflib.unified_diff`` body to ``previous``; hunks come in ascending order."""
    old = previous.split("\n") if previous else []
    result: list[str] = []
    cursor = 0
    lines = diff.split("\n")
    index = 0
    while index < len(lines):
        header = _HUNK.match(lines[index])
        index += 1
        if header is None:
            continue  # the "--- before" / "+++ after" file headers
        start = int(header.group(1)) - (1 if header.group(2) != "0" else 0)
        result.extend(old[cursor:start])
        cursor = start
        while index < len(lines) and not lines[index].startswith("@@"):
            tag, text = lines[index][:1], lines[index][1:]
            if tag == " ":
                result.append(text)
                cursor += 1
            elif tag == "-":
                cursor += 1
            elif tag == "+":
                result.append(text)
            index += 1
    result.extend(old[cursor:])
    return "\n".join(result)


def context_blocks(messages: Iterable[ModelMessage]) -> dict[str, str]:
    """The effective, non-withdrawn appended block per id, with diffs applied in order."""
    effective: dict[str, str] = {}
    for message in messages:
        for part in message.parts:
            if not (isinstance(part, UserPromptPart) and isinstance(part.content, str)):
                continue
            for kind, identifier, text in _ELEMENT.findall(part.content):
                effective[identifier] = (
                    _apply_diff(effective.get(identifier, ""), text)
                    if kind == "context_diff"
                    else text
                )
    return {key: text for key, text in effective.items() if not text.startswith(_WITHDRAWN)}


def effective_instructions(info: Any, messages: Iterable[ModelMessage]) -> str:
    return "\n".join([info.instructions or "", *context_blocks(messages).values()])
