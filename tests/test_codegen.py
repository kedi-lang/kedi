from __future__ import annotations

import os
import sys
from pathlib import Path

from kedi.cli import main as cli_main
from kedi.codegen.manager import CodegenManager
from kedi.codegen.mock_agents import MockPassingAgent
from kedi.tests_runner import run_tests


def test_codegen_cache_paths(tmp_path: Path) -> None:
    src = tmp_path / "prog.kedi"
    src.write_text("= ok\n", encoding="utf-8")
    mgr = CodegenManager(MockPassingAgent())
    cache = mgr.cache_path_for(src)
    assert cache.name.endswith(".cache.kedi")
    assert not mgr.is_cached(src, "foo")
    mgr.write_cache(src, "", "@foo():\n  = ok\n")
    assert mgr.is_cached(src, "foo")


def test_codegen_orchestrator_happy_path(tmp_path: Path) -> None:
    src = tmp_path / "g.kedi"
    # Base program contains a header we want to implement
    src.write_text(
        "@f():\n  > auto: \n    implement a function returning ok\n\n= <f()>\n",
        encoding="utf-8",
    )
    mgr = CodegenManager(MockPassingAgent())
    ok = mgr.orchestrate_generation(src, "@f():", "implement", "spec")
    assert ok
    # Now tests should pass using discovery (includes .cache)
    results = run_tests(str(src))
    assert results and all(c.success for s in results for c in s.cases)


def test_cli_like_codegen_runner(tmp_path: Path) -> None:
    src = tmp_path / "h.kedi"
    src.write_text(
        "@f():\n  > auto:\n    implement f returning ok\n\n= <f()>\n",
        encoding="utf-8",
    )

    os.environ["KEDI_CODEGEN_AGENT"] = "mock"
    sys.argv = [str(src)]
    try:
        cli_main.main(sys.argv)
    except SystemExit as e:
        assert e.code == 0
