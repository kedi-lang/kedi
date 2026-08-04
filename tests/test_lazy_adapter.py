"""LazyAdapter: run the program once, build the adapter only on first LLM use.

Guards against the double-execution where run_program_cli ran offline first and,
when a template needed the LLM, re-ran the whole program (firing top-level side
effects twice).
"""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

import pytest

import kedi.app as app
from kedi.agent_adapter.adapters.lazy import LazyAdapter
from kedi.agent_adapter.conversation import AdapterConversationContext
from kedi.agent_adapter.subagents import (
    SubagentConversationContext,
    SubagentExecutionRequest,
    SupportsSubagentExecution,
)
from kedi.agent_adapter.subagents.limits import SubagentUsageLimits
from kedi.agent_adapter.tool_override import ToolSpec
from kedi.agent_profile import AgentProfile
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
    shortname = "recording"
    supports_native_subagent_resume = True

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

    def invoke_sync(self, *a: object, **k: object) -> str:
        self.calls.append("invoke_sync")
        return "is"

    async def invoke(self, *a: object, **k: object) -> str:
        self.calls.append("invoke")
        return "i"

    def profile_override(self, profile: object) -> object:
        self.calls.append("profile_override")
        return nullcontext()

    def tool_scope(self, tools: object = ()) -> object:
        self.calls.append("tool_scope")
        return nullcontext()

    def register_tool(self, tool: object) -> None:
        self.calls.append(("register_tool", tool))

    def conversation_scope(self, conversation: object) -> object:
        self.calls.append(("conversation_scope", conversation))
        return nullcontext()

    def telemetry_model_name(self) -> str:
        self.calls.append("telemetry_model_name")
        return "telemetry-m"

    def subagent_scope(self, request: object) -> object:
        self.calls.append(("subagent_scope", request))
        return nullcontext()

    def subagent_run_kwargs(self, limits: object) -> dict[str, object]:
        self.calls.append(("subagent_run_kwargs", limits))
        return {"limits": limits}

    async def produce_subagent(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(("produce_subagent", kwargs))
        return {"task_summary": "done", "final_result": {"value": 1}}


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
    assert lazy.invoke_sync() == "is"
    assert asyncio.run(lazy.invoke()) == "i"
    profile = AgentProfile()
    tool = ToolSpec(fn=lambda: None, name="tool")
    with lazy.profile_override(profile):
        pass
    with lazy.tool_scope(["x"]):
        pass
    conversation = AdapterConversationContext()
    with lazy.conversation_scope(conversation):
        pass
    lazy.register_tool(tool)
    assert lazy.scoped_tools == ("tool-a",)

    assert builds == [1]  # built exactly once, then reused
    assert "produce_sync" in rec.calls
    assert "produce" in rec.calls
    assert "invoke_sync" in rec.calls
    assert "invoke" in rec.calls
    assert "profile_override" in rec.calls
    assert "tool_scope" in rec.calls
    assert ("conversation_scope", conversation) in rec.calls
    assert ("register_tool", tool) in rec.calls


def test_lazy_adapter_delegates_complete_subagent_protocol() -> None:
    rec = _RecordingAdapter()
    lazy = LazyAdapter(
        lambda: rec,
        shortname="pydantic",
    )
    limits = SubagentUsageLimits(request_limit=2)
    request = SubagentExecutionRequest(
        profile=AgentProfile(name="worker"),
        tools=(),
        instructions="Work carefully.",
        approval_ceiling=(),
        usage_limits=limits,
        conversation=SubagentConversationContext(conversation_id="conversation"),
    )

    assert isinstance(lazy, SupportsSubagentExecution)
    assert lazy.shortname == "pydantic"
    assert lazy.telemetry_model_name() == "telemetry-m"
    with lazy.subagent_scope(request):
        pass
    assert lazy.subagent_run_kwargs(limits) == {"limits": limits}
    assert asyncio.run(
        lazy.produce_subagent(
            template="Build a value",
            output_schema={"type": "object"},
        )
    ) == {
        "task_summary": "done",
        "final_result": {"value": 1},
    }
    assert lazy.supports_native_subagent_resume
    assert ("subagent_scope", request) in rec.calls
    assert ("subagent_run_kwargs", limits) in rec.calls


def test_lazy_adapter_rejects_subagent_execution_when_real_adapter_lacks_it() -> None:
    lazy = LazyAdapter(lambda: object(), shortname="unsupported")

    with pytest.raises(
        RuntimeError,
        match="Adapter 'unsupported' does not implement Kedi child execution",
    ):
        lazy.subagent_run_kwargs(SubagentUsageLimits())


def test_lazy_adapter_factory_is_called_once_for_concurrent_first_use() -> None:
    rec = _RecordingAdapter()
    builds: list[int] = []

    def factory() -> _RecordingAdapter:
        time.sleep(0.01)
        builds.append(1)
        return rec

    lazy = LazyAdapter(factory)
    with ThreadPoolExecutor(max_workers=8) as executor:
        assert list(executor.map(lambda _: lazy.produce_sync(), range(8))) == ["ps"] * 8

    assert builds == [1]
