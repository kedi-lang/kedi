"""LLM wrapper that strips model thinking blocks from DSPy output."""

from __future__ import annotations

import re
from typing import Any

# Matches [[ ## field_name ## ]] with flexible whitespace.
# Multi-word field names (e.g. "first name") are supported.
_THINKING_MARKER_RE = re.compile(r"\[\[\s*##\s+\w+(?:\s+\w+)*\s*##\s*\]\]")

# XML-style thinking blocks: ``  thinking ...  ``  (DeepSeek, o-series, Anthropic).
# Also handles ``  thought ...  `` and ``  reasoning ...  `` variants.
# The closing tag may optionally have a ``/`` prefix.
# Trailing whitespace stops BEFORE the ``  `` that starts a response/next tag
# so the response-stripping pass can see the ``  response`` prefix.
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


class ThinkingStrippingLM:
    """Wraps a dspy.LM to strip thinking blocks before DSPy parsing.

    Reasoning models emit ``  thinking ...  `` blocks
    that can contain text matching DSPy's structured output markers
    (e.g. ``[[ ## field_name ## ]]``).  This wrapper strips those blocks
    and returns only the cleaned output to DSPy.

    Attributes:
        wrapped: The underlying dspy.LM instance.
        last_thinking_traces: Thinking traces from the most recent call
                              (one per completion string).  Empty list
                              if no traces were stripped.
        last_raw_outputs: Full raw outputs (pre-stripping) from the most
                          recent call, for debugging.
    """

    def __init__(self, lm: Any) -> None:
        self._wrapped = lm
        self.last_thinking_traces: list[str] = []
        self.last_raw_outputs: list[str] = []

    @property
    def wrapped(self) -> Any:
        return self._wrapped

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)

    def __call__(self, prompt: Any, **kwargs: Any) -> list[str]:
        completions: list[str] = self._wrapped(prompt, **kwargs)

        self.last_raw_outputs = list(completions)
        self.last_thinking_traces = []

        cleaned: list[str] = []
        for text in completions:
            stripped = _strip_thinking(text)
            if stripped.thinking:
                self.last_thinking_traces.append(stripped.thinking)
            cleaned.append(stripped.output)

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