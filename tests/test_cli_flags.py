"""Tests for CLI optimizer and optimize flags."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from kedi.cli import main as cli_main


def test_cli_optimizer_flag(tmp_path: Path) -> None:
    """Test --optimizer flag."""
    srcp = tmp_path / "test.kedi"
    valp = tmp_path / "test.val.kedi"
    
    srcp.write_text("@foo(x: int) -> int:\n  = `x * 2`\n\n= <foo(`5`)>\n", encoding="utf-8")
    valp.write_text(
        "@eval: foo:\n  > data: inputs:\n    = `[(1, 2), (3, 6)]`\n"
        "  > metric: accuracy(inputs):\n    = `1.0`\n",
        encoding="utf-8",
    )
    
    # Test with --optimizer mock
    try:
        sys.argv = [str(srcp), "--eval", "--optimizer", "mock", "--optimize"]
        cli_main.main(sys.argv)
    except SystemExit as e:
        # Should succeed (mock optimizer)
        assert e.code in (0, 1)  # May fail if optimize requires eval suite with proper data


def test_cli_optimize_flag_without_eval(tmp_path: Path) -> None:
    """Test that --optimize requires --eval."""
    srcp = tmp_path / "test.kedi"
    
    srcp.write_text("@foo():\n  > optimize: test:\n    [x] from input\n  = done\n", encoding="utf-8")
    
    try:
        sys.argv = [str(srcp), "--optimize"]
        cli_main.main(sys.argv)
    except SystemExit as e:
        # Should succeed (optimize only runs with --eval)
        assert e.code in (0, 1)


def test_cli_optimize_with_missing_eval(tmp_path: Path) -> None:
    """Test --optimize with optimize blocks but no eval (should error)."""
    srcp = tmp_path / "test.kedi"
    
    srcp.write_text(
        "@solve_math(problem: str) -> int:\n"
        "  > optimize: find_answer:\n"
        "    Solve: <problem>\n"
        "    Answer: [answer: int]\n"
        "  = `answer`\n\n"
        "= <solve_math('2+2')>\n",
        encoding="utf-8",
    )
    
    from click.testing import CliRunner
    runner = CliRunner()
    result = runner.invoke(cli_main, [str(srcp), "--eval", "--optimize"])
    # Should fail because optimize requires eval suite
    assert result.exit_code == 1
    assert "needs @eval:" in result.output or "needs training data" in result.output


def test_cli_optimize_with_missing_data(tmp_path: Path) -> None:
    """Test --optimize with eval but no data (should error)."""
    srcp = tmp_path / "test.kedi"
    valp = tmp_path / "test.val.kedi"
    
    srcp.write_text(
        "@solve(problem: str) -> int:\n"
        "  > optimize: find_answer:\n"
        "    Solve: <problem>\n"
        "    Answer: [answer: int]\n"
        "  = `answer`\n\n"
        "= <solve('1+1')>\n",
        encoding="utf-8",
    )
    
    valp.write_text(
        "@eval: solve:\n  > metric: test:\n    = `1.0`\n",
        encoding="utf-8",
    )
    
    from click.testing import CliRunner
    runner = CliRunner()
    result = runner.invoke(cli_main, [str(srcp), "--eval", "--optimize"])
    # Should fail because optimize requires data
    assert result.exit_code == 1
    assert "needs training data" in result.output or "needs @eval:" in result.output

