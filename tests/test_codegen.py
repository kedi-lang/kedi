from __future__ import annotations

from pathlib import Path

from marker.codegen.manager import CodegenManager
from marker.codegen.mock_agents import MockPassingAgent
from marker.codegen.runner import run_codegen
from marker.cli import main as cli_main
from marker.tests_runner import run_tests


def test_codegen_cache_paths(tmp_path: Path) -> None:
    src = tmp_path / "prog.marker"
    src.write_text("= ok\n", encoding="utf-8")
    mgr = CodegenManager(MockPassingAgent())
    cache = mgr.cache_path_for(src)
    assert cache.name.endswith(".cache.marker")
    assert not mgr.is_cached(src, "foo")
    mgr.write_cache(src, "", "@foo():\n  = ok\n")
    assert mgr.is_cached(src, "foo")


def test_codegen_orchestrator_happy_path(tmp_path: Path) -> None:
    src = tmp_path / "g.marker"
    # Base program contains a header we want to implement
    src.write_text("@f():\n  > implement a function returning ok\n\n= <f()>\n", encoding="utf-8")
    mgr = CodegenManager(MockPassingAgent())
    ok = mgr.orchestrate_generation(src, "@f():", "implement", "spec")
    assert ok
    # Now tests should pass using discovery (includes .cache)
    results = run_tests(str(src))
    assert results and all(c.success for s in results for c in s.cases)


def test_cli_like_codegen_runner(tmp_path: Path) -> None:
    src = tmp_path / "h.marker"
    src.write_text("@f():\n  > implement f returning ok\n\n= <f()>\n", encoding="utf-8")
    import os, sys
    os.environ["MARKER_CODEGEN_AGENT"] = "mock"
    sys.argv = ["marker", str(src)]
    rc = cli_main()
    assert rc == 0


