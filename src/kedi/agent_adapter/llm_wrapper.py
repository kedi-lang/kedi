"""LLM wrapper that strips model thinking blocks from DSPy output."""

from __future__ import annotations

import re
from typing import Any

import dspy

# Matches [[ ## field_name ## ]] with flexible whitespace.
# Multi-word field names (e.g. "first name") are supported.
_THINKING_MARKER_RE = re.compile(r"\[\[\s*##\s+\w+(?:\s+\w+)*\s*##\s*\]\]")

# XML-style thinking blocks: ``  thinking ...  ``  (DeepSeek, o-series, Anthropic).
# Also handles ``  thought ...  `` and ``  reasoning ...  `` variants.
# The closing tag may optionally have a ``/`` prefix.
# Trailing newlines only so the response-stripping pass keeps the ``  response`` prefix.
_XML_THINKING_RE = re.compile(
    r"  (?:think(?:ing)?|thought|reason(?:ing)?)"
    r".*?"
    r"  /?(?:think(?:ing)?|thought|reason(?:ing)?)\n*",
    re.DOTALL,
)

# ``  response`` / ``  output`` wrapper tags that may surround the real
# completion after the thinking block.
_XML_RESPONSE_TAG_RE = re.compile(
    r"  /?(?:response|output)\s*>?\s*",
    re.DOTALL,
)


class ThinkingStrippingLM(dspy.LM):
    """A dspy.LM subclass that strips thinking blocks from model output.

    Reasoning models emit ``  thinking ...  `` blocks
    that can contain text matching DSPy's structured output markers
    (e.g. ``[[ ## field_name ## ]]``).  This LM strips those blocks
    before DSPy's parser sees the output.

    Attributes:
        last_thinking_traces: Thinking traces from the most recent call
                              (one per completion string).  Empty list
                              if no traces were stripped.
        last_raw_outputs: Full raw outputs (pre-stripping) from the most
                          recent call, for debugging.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.last_thinking_traces: list[str] = []
        self.last_raw_outputs: list[str] = []

    def __call__(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any] | str]:
        completions = super().__call__(prompt=prompt, messages=messages, **kwargs)

        self.last_raw_outputs = []
        self.last_thinking_traces = []

        cleaned: list[dict[str, Any] | str] = []
        for item in completions:
            if isinstance(item, str):
                self.last_raw_outputs.append(item)
                stripped = _strip_thinking(item)
                if stripped.thinking:
                    self.last_thinking_traces.append(stripped.thinking)
                cleaned.append(stripped.output)
            else:
                # Dict completions — pass through unchanged
                self.last_raw_outputs.append(repr(item))
                cleaned.append(item)

        return cleaned


class _StrippedResult:
    __slots__ = ("thinking", "output")

    def __init__(self, thinking: str, output: str) -> None:
        self.thinking = thinking
        self.output = output


def _strip_thinking(raw: str) -> _StrippedResult:
    """Strip thinking blocks from raw model output.

    Strategy (tried in order):

    1. **XML-style thinking blocks** (``  thinking ...  ``):
       Strip the thinking tags and their content.  Everything removed is
       saved as the thinking trace.  Any ``  response`` / ``  output``
       wrapper tags are also stripped from the cleaned output.

    2. **Field-marker fallback**: When no XML thinking tags are detected,
       find the last ``[[ ## field ## ]]`` marker that is NOT
       ``[[ ## completed ## ]]`` and split on it — everything before is
       the thinking trace.

    If neither heuristic matches, the output is returned as-is with an
    empty thinking trace.
    """
    # -- Strategy 1: XML thinking block -----------------------------------
    stripped_parts: list[str] = []

    def _collect_removed(m: re.Match[str]) -> str:
        stripped_parts.append(m.group())
        return ""

    after_thinking = _XML_THINKING_RE.sub(_collect_removed, raw)

    if stripped_parts:
        # Strip `  response` / `  output` wrapper tags from the output
        after_thinking = _XML_RESPONSE_TAG_RE.sub("", after_thinking).strip()
        thinking = "".join(stripped_parts).strip()
        return _StrippedResult(thinking=thinking, output=after_thinking)

    # -- Strategy 2: field-marker split -----------------------------------
    matches = list(_THINKING_MARKER_RE.finditer(raw))
    if not matches:
        return _StrippedResult(thinking="", output=raw)

    # Walk backwards to find the last marker that is NOT [[ ## completed ## ]]
    pivot: re.Match[str] | None = None
    for m in reversed(matches):
        text = m.group()
        if "completed" not in text:
            pivot = m
            break

    if pivot is None:
        return _StrippedResult(thinking="", output=raw)

    thinking = raw[: pivot.start()]
    output = raw[pivot.start() :]
    return _StrippedResult(thinking=thinking, output=output)