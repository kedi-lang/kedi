"""Bounded lexical passage retrieval over artifact text projections."""

from __future__ import annotations

import re
import secrets
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import closing
from typing import Any

from .errors import ArtifactInputError

_WORDS = re.compile(r"[^\W_]+", re.UNICODE)
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_STOP_WORDS = frozenset(
    "a an and are as at be by for from how i in is it of on or that the this to was "
    "were what when where which who why with".split()
)


def _search_text(text: str) -> str:
    # Keep identifiers intact as well as their components in the lexical index.
    split = _CAMEL.sub(" ", text).replace("_", " ")
    return text + "\n" + split


def _terms(query: str) -> list[str]:
    if not isinstance(query, str) or not query.strip() or len(query) > 2_048:
        raise ArtifactInputError("artifact query must contain 1 to 2048 characters")
    words = list(dict.fromkeys(_WORDS.findall(_CAMEL.sub(" ", query).casefold())))
    terms = [word for word in words if word not in _STOP_WORDS] or words
    if not terms or len(terms) > 64:
        raise ArtifactInputError("artifact query must contain 1 to 64 searchable terms")
    return terms


def _passages(blocks: Iterable[str], size: int) -> Iterator[tuple[int, str]]:
    """Split sequential text blocks with overlap and exact character offsets."""
    buffer = ""
    offset = 0
    overlap = size // 4
    for block in blocks:
        buffer += block
        while len(buffer) > size:
            end = buffer.rfind("\n", size // 2, size)
            end = end + 1 if end >= 0 else size
            yield offset, buffer[:end]
            step = end - overlap
            buffer = buffer[step:]
            offset += step
    if buffer:
        yield offset, buffer


def _clip_start(content: str, highlighted: str, marker: str, terms: list[str], budget: int) -> int:
    """Keep a cluster of distinct hits, not the first repeated query word."""
    hits: dict[int, set[str]] = {}
    pattern = re.compile(re.escape(marker) + r"(.*?)" + re.escape(marker), re.DOTALL)
    for index, match in enumerate(pattern.finditer(highlighted)):
        offset = match.start() - index * 2 * len(marker)
        if offset >= len(content):
            break
        for word in _WORDS.finditer(match[1]):
            hits.setdefault(offset + word.start(), set()).add(word.group().casefold())
    # Component-only matches occur in the normalized index copy. Map them back
    # even if a common word also matched the unmodified text.
    term_set = set(terms)
    for word in _WORDS.finditer(content):
        components = set(_CAMEL.sub(" ", word.group()).casefold().split()) & term_set
        if components:
            hits.setdefault(word.start(), set()).update(components)
    if not hits:
        return 0
    starts = sorted({min(max(0, pos - budget // 3), len(content) - budget) for pos in hits})

    def score(start: int) -> tuple[int, int]:
        distances: dict[str, int] = {}
        for pos, words in hits.items():
            if start <= pos < start + budget:
                distance = abs(pos - (start + budget // 3))
                for word in words:
                    distances[word] = min(distance, distances.get(word, budget))
        # Repeated noise cannot win by frequency. Among equal-coverage windows,
        # keep the evidence away from the clipped edge, with room for its value.
        return len(distances), -sum(distances.values())

    return max(starts, key=score)


def query_passages(
    blocks: Iterable[str], *, query: str, max_chars: int, top_k: int
) -> dict[str, Any]:
    """Rank overlapping passages using SQLite's Porter/Unicode FTS5 BM25.

    The temporary index is disk-backed, private to this call, and deleted on
    close. It never outlives an artifact lease or retains released content.
    """
    terms = _terms(query)
    expression = " OR ".join(f'"{term}"' for term in terms)
    if len(terms) > 1:
        expression = f'"{" ".join(terms)}" OR {expression}'
    with closing(sqlite3.connect("")) as db:
        db.execute("PRAGMA secure_delete=ON")
        db.execute(
            "CREATE VIRTUAL TABLE passages USING fts5("
            "terms, content UNINDEXED, offset UNINDEXED, "
            "tokenize='porter unicode61 remove_diacritics 2')"
        )
        count = 0
        total_chars = 0
        for offset, content in _passages(blocks, 1_200):
            db.execute(
                "INSERT INTO passages(terms, content, offset) VALUES (?, ?, ?)",
                (_search_text(content), content, offset),
            )
            count += 1
            total_chars = offset + len(content)
        marker = secrets.token_hex(24)
        matches = db.execute(
            "SELECT content, offset, bm25(passages) AS rank, "
            "highlight(passages, 0, ?, ?) FROM passages "
            "WHERE passages MATCH ? ORDER BY rank, CAST(offset AS INTEGER)",
            (marker, marker, expression),
        )
        selected: list[dict[str, Any]] = []
        selected_ranges: list[tuple[int, int]] = []
        returned = 0
        for content, offset, rank, highlighted in matches:
            end = offset + len(content)
            if any(offset < right and end > left for left, right in selected_ranges):
                continue
            selected_ranges.append((offset, end))
            budget = max_chars - returned
            if len(content) > budget:
                start = _clip_start(content, highlighted, marker, terms, budget)
                content = content[start : start + budget]
                offset += start
                end = offset + len(content)
            selected.append(
                {"content": content, "offset": offset, "end_offset": end, "score": -rank}
            )
            returned += len(content)
            if len(selected) == top_k or returned == max_chars:
                break
    return {
        "matches": selected,
        "returned_chars": returned,
        "total_chars": total_chars,
        "searched_passages": count,
    }
