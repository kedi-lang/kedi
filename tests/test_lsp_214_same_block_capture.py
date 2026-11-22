"""Kedi #214 (6.13): a capture cannot be read as input in its own request.

The capture being defined does not exist yet while its template runs,
so ``<answer>`` inside ``[answer] << ... <answer> ...`` must not resolve
to the capture. This is editor behavior only: no language change is made
and the program stays legal. Definition, references, and rename already
exclude the self-read; these tests pin that behavior.
"""

from __future__ import annotations

from tests.lsp_helpers import definition_at, references_at, rename_at

URI = "file:///tmp/x.kedi"
PATH = "/tmp/x.kedi"

SRC = "@p():\n  [answer: str] << What is <answer>?\n  = <answer>\n"


def test_definition_inside_own_capture_template_is_none() -> None:
    # Occurrence 2 of "answer" is the `<answer>` inside its own template.
    assert definition_at(SRC, URI, PATH, "answer", occurrence=2) is None


def test_references_exclude_the_self_template_read() -> None:
    refs = references_at(SRC, URI, PATH, "answer", True, occurrence=3)
    assert sorted({r.range.start.line for r in refs}) == [1, 2]


def test_rename_excludes_the_self_template_read() -> None:
    edit = rename_at(SRC, URI, PATH, "answer", "reply", occurrence=3)
    assert edit is not None
    changes = edit.changes or {}
    assert sorted({e.range.start.line for e in changes[URI]}) == [1, 2]


def test_return_use_still_resolves_to_the_capture() -> None:
    loc = definition_at(SRC, URI, PATH, "answer", occurrence=3)
    assert loc is not None
    assert loc.range.start.line == 1
