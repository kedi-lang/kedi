"""Coverage for branches added by the feedback bugfixes that the cluster tests
don't already exercise (keeps the 100% `make prod` gate satisfied)."""

from __future__ import annotations

import asyncio
import contextvars
import typing
from contextlib import nullcontext

import pytest
from click.testing import CliRunner

import kedi.cli as cli
from kedi.agent_adapter.lazy import LazyAdapter
from kedi.cli import main as cli_main
from kedi.codegen.manager import CodegenManager
from kedi.lang.compiler import compile_program
from kedi.lang.parser import parse_program
from kedi.optimization.services.prompt_loader import OptimizedPromptLoader
from kedi.resolved_types import resolved_type_name


# --- LazyAdapter delegation (BUG-007) --------------------------------------


class _RecordingAdapter:
    def __init__(self) -> None:
        self.calls: list[object] = []
        self.scoped_tools = ("tool-a",)

    def default_model(self) -> str:
        self.calls.append("default_model")
        return "m"

    def produce_sync(self, *a: object, **k: object) -> str:
        self.calls.append("produce_sync")
        return "ps"

    async def produce(self, *a: object, **k: object) -> str:
        self.calls.append("produce")
        return "p"

    def profile_override(self, profile: object) -> object:
        self.calls.append("profile_override")
        return nullcontext()

    def tool_scope(self, tools: object = ()) -> object:
        self.calls.append("tool_scope")
        return nullcontext()

    def register_tool(self, tool: object) -> None:
        self.calls.append(("register_tool", tool))


def test_lazy_adapter_builds_once_and_delegates() -> None:
    rec = _RecordingAdapter()
    builds: list[int] = []

    def factory() -> _RecordingAdapter:
        builds.append(1)
        return rec

    lazy = LazyAdapter(factory)
    assert builds == []  # nothing built until first use

    assert lazy.default_model() == "m"
    assert lazy.produce_sync() == "ps"
    assert asyncio.run(lazy.produce()) == "p"
    with lazy.profile_override(object()):
        pass
    with lazy.tool_scope(["x"]):
        pass
    lazy.register_tool("tool")
    assert lazy.scoped_tools == ("tool-a",)

    assert builds == [1]  # built exactly once, then reused
    assert "produce_sync" in rec.calls
    assert "produce" in rec.calls
    assert "profile_override" in rec.calls
    assert "tool_scope" in rec.calls
    assert ("register_tool", "tool") in rec.calls


# --- resolved_type_name on a typing special form (no isinstance-type) -------


def test_resolved_type_name_on_union() -> None:
    assert resolved_type_name(typing.Union[int, str]) == "Union"


# --- unresolvable return type re-raises the precise TypeError (BUG-004/016) -


def test_unresolvable_return_type_reraises_typeerror() -> None:
    src = "@p() -> MissingType:\n  = `1`\n\n= <`str(p())`>\n"
    rt = compile_program(parse_program(src), adapter=None)
    with pytest.raises(Exception, match="Unknown type annotation MissingType"):
        rt.run_main()


# --- inject_optimized_prompts skips procedures absent from the artifact -----


def test_inject_optimized_prompts_noop_when_absent(tmp_path) -> None:
    src = tmp_path / "p.kedi"
    src.write_text("@p() -> str:\n  = ok\n", encoding="utf-8")
    program = parse_program(src.read_text(encoding="utf-8"), source_path=str(src))
    # No .optimized.json -> load returns {} -> inject returns early without changes.
    OptimizedPromptLoader(src).inject_optimized_prompts(program)


# --- codegen cache: removing an absent block leaves the cache unchanged -----


def test_remove_block_absent_returns_existing() -> None:
    mgr = CodegenManager(agent=None)
    existing = "# >>> kedi-codegen: a\nx\n# <<< kedi-codegen: a\n"
    assert mgr._remove_block(existing, "nonexistent") == existing


# --- CLI --no-cache cleanup on the --test / --eval branches -----------------


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


# --- review Finding 1: runtime_isinstance is strict for generic args -----------


def test_runtime_isinstance_strict_generic_rejections() -> None:
    from kedi.resolved_types import runtime_isinstance

    assert runtime_isinstance(["a", "b"], list[str]) is True
    # Strict: a tuple is not a list, and a float is not an int element.
    assert runtime_isinstance(("a", "b"), list[str]) is False
    assert runtime_isinstance([1, 2.0], list[int]) is False
    assert runtime_isinstance({"a": 1.0}, dict[str, int]) is False


# --- review Finding 2: run_sync carries contextvars into the worker thread -----


def test_run_sync_preserves_contextvars_inside_running_loop() -> None:
    from kedi.agent_adapter.utils import run_sync

    var: contextvars.ContextVar[str] = contextvars.ContextVar("t", default="default")

    async def _read() -> str:
        return var.get()

    async def _outer() -> str:
        var.set("inner")
        # Inside a running loop run_sync uses a worker thread; the context must
        # still carry ``var`` so the coroutine reads "inner", not the default.
        return run_sync(_read())

    assert asyncio.run(_outer()) == "inner"


# --- review Finding 7: codegen refuses to cache a body containing a marker -----


def test_write_cache_rejects_marker_in_generated_body(tmp_path) -> None:
    mgr = CodegenManager(agent=None)
    with pytest.raises(ValueError, match="reserved kedi-codegen"):
        mgr.write_cache(tmp_path / "p.kedi", "p", "# >>> kedi-codegen: evil", "")


# --- review Finding 3: untyped param's default must also be str ----------------


def test_untyped_param_nonstr_default_raises() -> None:
    # Untyped param means `str`; a native non-str default violates that and raises
    # when used. Annotate the param (e.g. `n: int = \`0\``) for a non-str default.
    src = "@f(n = `0`):\n  = <`str(n)`>\n\n= <f()>\n"
    rt = compile_program(parse_program(src), adapter=None)
    with pytest.raises(Exception, match="Parameter 'n' does not match annotated type"):
        rt.run_main()
