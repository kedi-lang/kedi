"""LazyAdapter: run the program once, build the adapter only on first LLM use.

Guards against the double-execution where run_program_cli ran offline first and,
when a template needed the LLM, re-ran the whole program (firing top-level side
effects twice).
"""

from __future__ import annotations

import asyncio
from contextlib import nullcontext

import kedi.app as app
from kedi.agent_adapter.lazy import LazyAdapter
from kedi.app import run_program_cli
from kedi.utils import AttrDict
from tests.mock_adapter import MockAdapter

_COUNTER_BLOCK = (
    "```\n"
    "from pathlib import Path\n"
    "p = Path(args.counter)\n"
    'p.write_text(str(int(p.read_text()) + 1) if p.exists() else "1")\n'
    "```\n"
)


def test_top_level_side_effects_run_once_with_llm_template(tmp_path, monkeypatch) -> None:
    counter = tmp_path / "count.txt"
    prog = tmp_path / "prog.kedi"
    prog.write_text(
        _COUNTER_BLOCK + ">> Produce a greeting. [greeting]\n= <`greeting`>\n",
        encoding="utf-8",
    )

    builds: list[tuple[str, str | None]] = []

    def fake_build(adapter: str, model: str | None) -> MockAdapter:
        builds.append((adapter, model))
        return MockAdapter(default_model=model)

    monkeypatch.setattr(app, "_build_runtime_adapter", fake_build)

    code, out = run_program_cli(
        str(prog),
        adapter="pydantic",
        model=None,
        runtime_args=AttrDict({"counter": str(counter)}, default=None),
    )

    assert code == 0
    assert counter.read_text() == "1"  # the side effect fires once, not twice
    assert len(builds) == 1  # adapter built once, lazily, because a template needs it
    assert "greeting-value" in out


def test_offline_program_never_builds_adapter(tmp_path, monkeypatch) -> None:
    counter = tmp_path / "count.txt"
    prog = tmp_path / "prog.kedi"
    prog.write_text(_COUNTER_BLOCK + "= done\n", encoding="utf-8")

    builds: list[int] = []

    def fake_build(adapter: str, model: str | None) -> MockAdapter:
        builds.append(1)
        return MockAdapter(default_model=model)

    monkeypatch.setattr(app, "_build_runtime_adapter", fake_build)

    code, out = run_program_cli(
        str(prog),
        adapter="pydantic",
        model=None,
        runtime_args=AttrDict({"counter": str(counter)}, default=None),
    )

    assert code == 0
    assert out == "done"
    assert counter.read_text() == "1"
    assert builds == []  # no template -> adapter never constructed


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
