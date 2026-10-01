"""Prompt-toolkit input and syntax highlighting for Kedi IDLE."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text.base import StyleAndTextTuples
from prompt_toolkit.history import History
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.processors import TabsProcessor
from prompt_toolkit.lexers import Lexer
from prompt_toolkit.output import ColorDepth
from prompt_toolkit.styles import Style

MAX_HIGHLIGHT_CHARS = 8_000
IDLE_INDENT = "  "

_TOKEN_STYLES = {
    "namespace": "class:kedi.namespace",
    "type": "class:kedi.type",
    "class": "class:kedi.type",
    "enum": "class:kedi.type",
    "interface": "class:kedi.type",
    "struct": "class:kedi.type",
    "typeParameter": "class:kedi.type",
    "parameter": "class:kedi.parameter",
    "variable": "class:kedi.variable",
    "property": "class:kedi.property",
    "enumMember": "class:kedi.constant",
    "event": "class:kedi.function",
    "function": "class:kedi.function",
    "method": "class:kedi.function",
    "macro": "class:kedi.function",
    "keyword": "class:kedi.keyword",
    "modifier": "class:kedi.keyword",
    "comment": "class:kedi.comment",
    "string": "class:kedi.string",
    "number": "class:kedi.number",
    "regexp": "class:kedi.string",
    "operator": "class:kedi.operator",
    "decorator": "class:kedi.decorator",
}

KEDI_IDLE_STYLE = Style.from_dict(
    {
        "kedi.namespace": "#82aaff",
        "kedi.type": "#c792ea",
        "kedi.parameter": "#f6c177",
        "kedi.variable": "#d8dee9",
        "kedi.property": "#7fdbca",
        "kedi.constant": "#d0bf69",
        "kedi.function": "#82aaff",
        "kedi.keyword": "#ff7ab2 bold",
        "kedi.comment": "#6c7086 italic",
        "kedi.string": "#a8cc8c",
        "kedi.number": "#d0bf69",
        "kedi.operator": "#89ddff",
        "kedi.decorator": "#f78c6c",
    }
)


class KediLexer(Lexer):
    """Render Kedi and embedded Python using the canonical tree-sitter token pipeline."""

    def __init__(self, *, max_chars: int = MAX_HIGHLIGHT_CHARS) -> None:
        from .lsp.features import SEMANTIC_TOKEN_TYPES, compute_semantic_tokens_full

        self._max_chars = max_chars
        self._semantic_token_types = SEMANTIC_TOKEN_TYPES
        self._compute_semantic_tokens = compute_semantic_tokens_full
        self._cached_source: str | None = None
        self._cached_lines: list[StyleAndTextTuples] = []

    def lex_document(self, document: Document) -> Callable[[int], StyleAndTextTuples]:
        source = document.text
        if source != self._cached_source:
            self._cached_source = source
            self._cached_lines = self._highlight(source)
        lines = self._cached_lines

        def get_line(line_number: int) -> StyleAndTextTuples:
            if 0 <= line_number < len(lines):
                return lines[line_number]
            return [("", "")]

        return get_line

    def _highlight(self, source: str) -> list[StyleAndTextTuples]:
        lines = source.split("\n")
        if len(source) > self._max_chars:
            return [[("", line)] for line in lines]

        encoded = self._compute_semantic_tokens(source, None).data or []
        spans: dict[int, list[tuple[int, int, str]]] = {}
        line_number = 0
        column = 0
        for offset in range(0, len(encoded), 5):
            delta_line, delta_column, length, token_index, _modifiers = encoded[offset : offset + 5]
            line_number += delta_line
            column = column + delta_column if delta_line == 0 else delta_column
            if not (0 <= line_number < len(lines)):
                continue
            token_name = self._semantic_token_types[token_index]
            style = _TOKEN_STYLES.get(token_name, "")
            line = lines[line_number]
            start = _byte_column_to_character(line, column)
            end = _byte_column_to_character(line, column + length)
            if end > start:
                spans.setdefault(line_number, []).append((start, end, style))

        return [_line_fragments(line, spans.get(index, [])) for index, line in enumerate(lines)]


class PromptToolkitInput:
    """Read normal and explicit multiline Kedi input from one terminal session."""

    def __init__(
        self,
        *,
        highlight: bool,
        requires_continuation: Callable[[str], bool],
        continuation_marker: str,
    ) -> None:
        self._requires_continuation = requires_continuation
        self._continuation_marker = continuation_marker
        self._color_depth = ColorDepth.DEPTH_24_BIT if highlight else None
        self._line_bindings = _line_key_bindings()
        self._multiline_bindings = self._build_multiline_bindings()
        self._session: PromptSession[str] = PromptSession(
            history=ReadlineHistory(),
            lexer=KediLexer() if highlight else None,
            style=KEDI_IDLE_STYLE if highlight else None,
            input_processors=[
                TabsProcessor(
                    tabstop=len(IDLE_INDENT),
                    char1=" ",
                    char2=" ",
                    style="",
                )
            ],
        )

    def read_line(self, prompt: str) -> str:
        """Read one ordinary IDLE input line."""
        return self._session.prompt(
            prompt,
            multiline=False,
            key_bindings=self._line_bindings,
            enable_history_search=True,
            color_depth=self._color_depth,
        )

    def read_multiline(self, prompt: str) -> str:
        """Read one source fragment, submitting on a complete double Enter."""
        return self._session.prompt(
            prompt,
            multiline=True,
            key_bindings=self._multiline_bindings,
            prompt_continuation=self._continuation_prompt,
            enable_history_search=True,
            color_depth=self._color_depth,
        )

    def _continuation_prompt(
        self,
        width: int,
        _line_number: int,
        _wrap_count: int,
    ) -> str:
        return f"{self._continuation_marker} ".rjust(width)

    def _build_multiline_bindings(self) -> KeyBindings:
        bindings = KeyBindings()

        @bindings.add("tab")
        def insert_tab(event: Any) -> None:
            event.current_buffer.insert_text(IDLE_INDENT)

        @bindings.add("escape", "enter")
        def force_submit(event: Any) -> None:
            event.current_buffer.validate_and_handle()

        @bindings.add("enter")
        def newline_or_submit(event: Any) -> None:
            document = event.current_buffer.document
            if _should_submit_multiline(document, self._requires_continuation):
                event.current_buffer.validate_and_handle()
            else:
                event.current_buffer.insert_text("\n")

        _add_deletion_bindings(bindings)
        return bindings


class ReadlineHistory(History):
    """Share the existing Kedi readline history with prompt-toolkit input."""

    def __init__(self) -> None:
        try:
            import readline
        except ImportError:
            self._readline = None
        else:
            self._readline = readline
        super().__init__()

    def load_history_strings(self) -> list[str]:
        if self._readline is None:
            return []
        entries = [
            self._readline.get_history_item(index)
            for index in range(1, self._readline.get_current_history_length() + 1)
        ]
        return [entry for entry in reversed(entries) if entry is not None]

    def store_string(self, string: str) -> None:
        if self._readline is not None:
            self._readline.add_history(string)


def _line_key_bindings() -> KeyBindings:
    bindings = KeyBindings()

    @bindings.add("tab")
    def insert_tab(event: Any) -> None:
        event.current_buffer.insert_text(IDLE_INDENT)

    _add_deletion_bindings(bindings)
    return bindings


def _add_deletion_bindings(bindings: KeyBindings) -> None:
    @bindings.add("backspace")
    def delete_before_cursor(event: Any) -> None:
        event.current_buffer.delete_before_cursor(count=1)

    @bindings.add("delete")
    def delete_at_cursor(event: Any) -> None:
        event.current_buffer.delete(count=1)


def _should_submit_multiline(
    document: Document,
    requires_continuation: Callable[[str], bool],
) -> bool:
    """Accept the second Enter on an empty line once the fragment can be submitted."""
    if document.current_line or "\n" not in document.text_before_cursor:
        return False
    source = document.text.rstrip("\n")
    return not source.strip() or not requires_continuation(source)


def _byte_column_to_character(line: str, column: int) -> int:
    encoded = line.encode("utf-8")
    return len(encoded[:column].decode("utf-8", errors="ignore"))


def _line_fragments(
    line: str,
    spans: list[tuple[int, int, str]],
) -> StyleAndTextTuples:
    fragments: StyleAndTextTuples = []
    cursor = 0
    for start, end, style in sorted(spans):
        start = max(cursor, min(start, len(line)))
        end = max(start, min(end, len(line)))
        if start > cursor:
            fragments.append(("", line[cursor:start]))
        if end > start:
            fragments.append((style, line[start:end]))
        cursor = end
    if cursor < len(line):
        fragments.append(("", line[cursor:]))
    return fragments or [("", "")]


__all__ = [
    "IDLE_INDENT",
    "KEDI_IDLE_STYLE",
    "MAX_HIGHLIGHT_CHARS",
    "KediLexer",
    "PromptToolkitInput",
    "ReadlineHistory",
]
