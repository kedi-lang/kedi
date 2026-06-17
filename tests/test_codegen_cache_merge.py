"""codegen cache must merge by procedure identity, not clobber.

Writing the cache for a second procedure must preserve the block generated
for an earlier procedure in the same ``.cache.kedi`` file.
"""

from __future__ import annotations

from pathlib import Path

from kedi.codegen.manager import CodegenManager
from kedi.codegen.mock_agents import MockPassingAgent


def test_write_cache_merges_multiple_procedures(tmp_path: Path) -> None:
    src = tmp_path / "prog.kedi"
    src.write_text("= ok\n", encoding="utf-8")
    mgr = CodegenManager(MockPassingAgent())

    mgr.write_cache(src, "a", "", "@a():\n  = a-result\n")
    mgr.write_cache(src, "b", "", "@b():\n  = b-result\n")

    cache_text = mgr.cache_path_for(src).read_text(encoding="utf-8")

    # Both generated procedures survive the second write.
    assert "@a()" in cache_text
    assert "@b()" in cache_text
    assert "a-result" in cache_text
    assert "b-result" in cache_text
    assert mgr.is_cached(src, "a")
    assert mgr.is_cached(src, "b")

    # Markers are Kedi comments, so the merged cache still parses cleanly with
    # both procedures present.
    merged = mgr.parse_with_cache(src)
    assert {"a", "b"} <= {proc.name for proc in merged.procedures}


def test_write_cache_replaces_only_the_named_block(tmp_path: Path) -> None:
    src = tmp_path / "prog.kedi"
    src.write_text("= ok\n", encoding="utf-8")
    mgr = CodegenManager(MockPassingAgent())

    mgr.write_cache(src, "a", "", "@a():\n  = first\n")
    mgr.write_cache(src, "b", "", "@b():\n  = b-result\n")
    # Regenerate @a: only its block changes, @b stays intact.
    mgr.write_cache(src, "a", "", "@a():\n  = second\n")

    cache_text = mgr.cache_path_for(src).read_text(encoding="utf-8")
    assert "second" in cache_text
    assert "first" not in cache_text
    assert "b-result" in cache_text
    # No duplicate block for @a.
    assert cache_text.count("# >>> kedi-codegen: a\n") == 1


def test_write_cache_handles_prefix_named_procedures(tmp_path: Path) -> None:
    src = tmp_path / "prog.kedi"
    src.write_text("= ok\n", encoding="utf-8")
    mgr = CodegenManager(MockPassingAgent())

    # `a` is a prefix of `ab`: marker matching must not confuse the two blocks.
    mgr.write_cache(src, "a", "", "@a():\n  = a-result\n")
    mgr.write_cache(src, "ab", "", "@ab():\n  = ab-result\n")
    mgr.write_cache(src, "a", "", "@a():\n  = a-updated\n")

    cache_text = mgr.cache_path_for(src).read_text(encoding="utf-8")
    assert "a-updated" in cache_text
    assert "a-result" not in cache_text
    assert "ab-result" in cache_text
    assert mgr.is_cached(src, "a")
    assert mgr.is_cached(src, "ab")
