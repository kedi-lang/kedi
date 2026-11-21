"""6.24: the tree-sitter build script targets the notebook, not the playground."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_notebook_tree_sitter.sh"
LEGACY_SCRIPT = ROOT / "scripts" / "build_playground_tree_sitter.sh"
MAKEFILE = ROOT / "Makefile"


def test_tree_sitter_build_script_targets_the_notebook() -> None:
    assert SCRIPT.is_file()
    assert not LEGACY_SCRIPT.exists()
    text = SCRIPT.read_text(encoding="utf-8")
    assert 'STATIC_ROOT="$ROOT/notebook/src/kedi_notebook/static"' in text
    assert "playground/src/playground/static" not in text
    assert (ROOT / "notebook" / "src" / "kedi_notebook" / "static" / "grammars" / "kedi").is_dir()


def test_makefile_references_the_renamed_script() -> None:
    text = MAKEFILE.read_text(encoding="utf-8")
    assert "scripts/build_notebook_tree_sitter.sh" in text
    assert "build_playground_tree_sitter.sh" not in text
