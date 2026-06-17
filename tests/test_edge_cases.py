"""Edge-case coverage the topical suites don't already exercise."""

from __future__ import annotations

import asyncio
import contextvars
import typing

import pytest
from click.testing import CliRunner

import kedi.cli as cli
from kedi.agent_adapter.utils import run_sync
from kedi.cli import main as cli_main
from kedi.codegen.manager import CodegenManager
from kedi.lang.compiler import compile_program
from kedi.lang.parser import parse_program
from kedi.optimization.services.prompt_loader import OptimizedPromptLoader
from kedi.resolved_types import resolved_type_name, runtime_isinstance


def test_resolved_type_name_on_union() -> None:
    assert resolved_type_name(typing.Union[int, str]) == "Union"


def test_unresolvable_return_type_raises() -> None:
    src = "@p() -> MissingType:\n  = `1`\n\n= <`str(p())`>\n"
    rt = compile_program(parse_program(src), adapter=None)
    with pytest.raises(Exception, match="Unknown type annotation MissingType"):
        rt.run_main()


def test_inject_optimized_prompts_noop_when_absent(tmp_path) -> None:
    src = tmp_path / "p.kedi"
    src.write_text("@p() -> str:\n  = ok\n", encoding="utf-8")
    program = parse_program(src.read_text(encoding="utf-8"), source_path=str(src))
    # No .optimized.json -> load returns {} -> inject returns early without changes.
    OptimizedPromptLoader(src).inject_optimized_prompts(program)


def test_remove_block_absent_returns_existing() -> None:
    mgr = CodegenManager(agent=None)
    existing = "# >>> kedi-codegen: a\nx\n# <<< kedi-codegen: a\n"
    assert mgr._remove_block(existing, "nonexistent") == existing


def test_cli_test_no_cache_cleans_cache(tmp_path, monkeypatch) -> None:
    src = tmp_path / "p.kedi"
    src.write_text("= ok\n", encoding="utf-8")
    cache = tmp_path / "p.cache.kedi"
    cache.write_text("stale\n", encoding="utf-8")
    monkeypatch.setattr(cli, "run_tests_cli", lambda *a, **k: (0, "tests ok"))

    result = CliRunner().invoke(cli_main, [str(src), "--test", "--no-cache"])
    assert result.exit_code == 0
    assert not cache.exists()


def test_cli_eval_no_cache_cleans_cache(tmp_path, monkeypatch) -> None:
    src = tmp_path / "p.kedi"
    src.write_text("= ok\n", encoding="utf-8")
    cache = tmp_path / "p.cache.kedi"
    cache.write_text("stale\n", encoding="utf-8")
    monkeypatch.setattr(cli, "run_evals_cli", lambda *a, **k: (0, "evals ok"))

    result = CliRunner().invoke(cli_main, [str(src), "--eval", "--no-cache"])
    assert result.exit_code == 0
    assert not cache.exists()


def test_runtime_isinstance_rejects_mistyped_generics() -> None:
    assert runtime_isinstance(["a", "b"], list[str]) is True
    # Strict: a tuple is not a list, and a float is not an int element.
    assert runtime_isinstance(("a", "b"), list[str]) is False
    assert runtime_isinstance([1, 2.0], list[int]) is False
    assert runtime_isinstance({"a": 1.0}, dict[str, int]) is False


def test_run_sync_preserves_contextvars_inside_running_loop() -> None:
    var: contextvars.ContextVar[str] = contextvars.ContextVar("t", default="default")

    async def _read() -> str:
        return var.get()

    async def _outer() -> str:
        var.set("inner")
        # In a running loop run_sync uses a worker thread; the context must carry over.
        return run_sync(_read())

    assert asyncio.run(_outer()) == "inner"


def test_write_cache_rejects_marker_in_generated_body(tmp_path) -> None:
    mgr = CodegenManager(agent=None)
    with pytest.raises(ValueError, match="reserved kedi-codegen"):
        mgr.write_cache(tmp_path / "p.kedi", "p", "# >>> kedi-codegen: evil", "")


def test_untyped_param_nonstr_default_raises() -> None:
    # Untyped param is str; a native non-str default violates that and raises when used.
    src = "@f(n = `0`):\n  = <`str(n)`>\n\n= <f()>\n"
    rt = compile_program(parse_program(src), adapter=None)
    with pytest.raises(Exception, match="Parameter 'n' does not match annotated type"):
        rt.run_main()
