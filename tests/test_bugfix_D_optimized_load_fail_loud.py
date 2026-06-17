"""BUG-013/014/024: optimized-artifact load errors must fail loud.

An invalid ``.optimized.json`` must raise a clear error from the loader, and
``run_tests`` / ``run_evals`` must propagate it instead of silently falling
back to the unoptimized base program (which would produce a false green).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kedi.optimization.services.prompt_loader import OptimizedPromptLoader
from kedi.tests_runner import run_evals, run_tests


def _write_invalid_optimized(source: Path) -> None:
    optimized = source.with_suffix(source.suffix + ".optimized.json")
    optimized.write_text("{not json", encoding="utf-8")


def test_load_optimized_prompts_missing_file_returns_empty(tmp_path: Path) -> None:
    source = tmp_path / "prog.kedi"
    source.write_text("= ok\n", encoding="utf-8")

    # Absent file is the only case that returns {} (not an error).
    assert OptimizedPromptLoader(source).load_optimized_prompts() == {}


def test_load_optimized_prompts_invalid_json_raises(tmp_path: Path) -> None:
    source = tmp_path / "prog.kedi"
    source.write_text("= ok\n", encoding="utf-8")
    _write_invalid_optimized(source)

    with pytest.raises(ValueError, match="Invalid optimized prompt file"):
        OptimizedPromptLoader(source).load_optimized_prompts()


def test_run_tests_propagates_invalid_optimized_json(tmp_path: Path) -> None:
    source = tmp_path / "tests.kedi"
    source.write_text(
        "@id(x):\n  = <x>\n\n@test: id:\n  > case: ok:\n    `assert id('x') == 'x'`\n",
        encoding="utf-8",
    )
    _write_invalid_optimized(source)

    with pytest.raises(ValueError, match="Invalid optimized prompt file"):
        run_tests(str(source), adapter=None)


def test_run_evals_propagates_invalid_optimized_json(tmp_path: Path) -> None:
    source = tmp_path / "evals.kedi"
    source.write_text(
        "@identity(x):\n"
        "  = `x`\n\n"
        "@eval: identity:\n"
        "  > data: mapping:\n"
        "    = `{'train': 1}`\n"
        "  > metric: mapping_score(mapping):\n"
        "    = `1.0`\n",
        encoding="utf-8",
    )
    _write_invalid_optimized(source)

    with pytest.raises(ValueError, match="Invalid optimized prompt file"):
        run_evals(str(source), adapter=None)
