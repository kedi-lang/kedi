from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from threading import Lock
from types import SimpleNamespace
from typing import Any, cast

import pytest

import kedi
import kedi.lang.parser as parser_module
from kedi.agent_adapter.adapters.lazy import LazyAdapter
from kedi.agent_adapter.adapters.pydantic import (
    PydanticAdapter,
    _KediTelemetryCapability,
)
from kedi.agent_adapter.approvals.policy import ApprovalDecision, ApprovalPolicy, ApprovalRequest
from kedi.agent_adapter.approvals.telemetry import (
    approval_mechanism,
    approval_span,
    record_immediate_decision,
)
from kedi.agent_adapter.invocation import InvocationUsage
from kedi.agent_adapter.meta import AdapterCapabilities
from kedi.agent_adapter.request_budget import ModelRequestBudget
from kedi.agent_adapter.skills import SkillStore
from kedi.agent_adapter.subagents.execution import _subagent_run_span
from kedi.agent_adapter.subagents.workflow import _dynamic_workflow_span
from kedi.agent_adapter.telemetry import (
    adapter_owns_native_span,
    agent_call_span,
    mcp_initialization_span,
    mcp_server_name,
    mcp_transport_name,
    model_call_span,
    native_agent_span_scope,
    native_tool_span_scope,
    reclassify_native_agent_span,
    record_conversation_assets,
    resolved_adapter_model,
    tool_call_span,
    tool_source,
)
from kedi.agent_adapter.tools.execution import with_approval, with_tool_telemetry
from kedi.agent_adapter.tools.specs import ToolSpec
from kedi.agent_profile import AgentProfile
from kedi.app import run_program_cli, run_source_cli
from kedi.artifacts.telemetry import artifact_span
from kedi.lang import KediDiagnostic, KediParseError, Span
from kedi.telemetry import (
    NoOpTelemetryBackend,
    get_backend,
    install_backend,
    reset_backend,
    restore_backend,
    restore_backend_if_current,
    runtime_span,
)
from tests.telemetry_helpers import RecordingTelemetryBackend, installed_recording_backend


def test_conversation_assets_cover_template_profile_tools_and_schemas() -> None:
    tool = ToolSpec(
        fn=lambda city: city,
        name="weather",
        description="Look up weather.",
        json_schema={"type": "object", "properties": {"city": {"type": "string"}}},
        return_schema={"type": "string"},
        risk="read_only",
    )
    profile = AgentProfile(
        name="researcher",
        model="test-model",
        system="Use evidence.",
        settings={"temperature": 0},
        tools=("weather",),
        skills=True,
    )
    backend = RecordingTelemetryBackend(
        capture=frozenset({"content", "source_path", "source_snippet"})
    )

    with installed_recording_backend(backend):
        record_conversation_assets(
            prompt="Find weather for Istanbul.",
            instructions="Use evidence.",
            profile=profile,
            tools=(tool,),
            output_schema={"temperature": "<class 'float'>"},
            call_kind="produce",
            source_path="weather.kedi",
            source_line=12,
            source_snippet="Weather: {{ city }}",
        )

    kinds = {asset.kind for asset in backend.assets}
    assert kinds == {
        "agent_profile",
        "prompt",
        "schema",
        "source_identity",
        "template",
        "tool",
    }
    assert {asset.source_locator for asset in backend.assets if asset.kind == "schema"} == {
        "kedi:schema:template:weather.kedi:12:produce:output",
        "kedi:schema:tool:weather:input",
        "kedi:schema:tool:weather:return",
    }
    assert all(asset.content is not None for asset in backend.assets)


def test_conversation_asset_content_is_hashed_when_capture_is_disabled() -> None:
    backend = RecordingTelemetryBackend()
    profile = AgentProfile(name="private")

    with installed_recording_backend(backend):
        for prompt in ("first secret", "second secret"):
            record_conversation_assets(
                prompt=prompt,
                instructions=None,
                profile=profile,
                tools=(),
                output_schema=None,
                call_kind="invoke",
                source_path=None,
                source_line=None,
                source_snippet=None,
            )

    templates = [asset for asset in backend.assets if asset.kind == "template"]
    assert len(templates) == 2
    assert templates[0].content is None
    assert templates[1].content is None
    assert templates[0].content_fingerprint != templates[1].content_fingerprint
    assert "first secret" not in repr(backend.assets)
    assert "second secret" not in repr(backend.assets)


def test_skill_content_capture_is_explicit_and_versionable(tmp_path: Path) -> None:
    skill_path = tmp_path / "skills" / "review" / "SKILL.md"
    skill_path.parent.mkdir(parents=True)
    skill_path.write_text("Review carefully.", encoding="utf-8")
    backend = RecordingTelemetryBackend(capture=frozenset({"content", "source_path"}))

    with installed_recording_backend(backend):
        assert SkillStore(tmp_path / "skills").read_skill(skill_name="review") == (
            "Review carefully."
        )

    assert len(backend.assets) == 1
    asset = backend.assets[0]
    assert asset.kind == "skill"
    assert asset.content == "Review carefully."
    assert asset.metadata["source_path"] == str(skill_path)


def test_cli_program_trace_has_stable_runtime_topology() -> None:
    with installed_recording_backend() as backend:
        assert run_source_cli("= `1`", adapter="pydantic", model=None) == (0, "1")

    spans = {span.operation: span for span in backend.finished_spans}
    root = spans["run_program"]
    assert root.name == "kedi run <command>"
    assert root.parent is None
    assert root.attributes["kedi.result.type"] == "int"
    assert spans["parse_program"].parent is root
    assert spans["parse_program"].attributes["kedi.parse.statement_count"] == 1
    assert spans["compile_program"].parent is root
    assert spans["call_procedure"].parent is root
    assert spans["run_python"].parent is spans["call_procedure"]
    assert all("code.file.path" not in span.attributes for span in backend.finished_spans)

    metric_names = {metric.name for metric in backend.metrics}
    assert {
        "kedi.run.count",
        "kedi.run.duration",
        "kedi.parse.count",
        "kedi.parse.duration",
        "kedi.compile.count",
        "kedi.compile.duration",
        "kedi.procedure.calls",
        "kedi.procedure.duration",
        "kedi.python.execution.count",
        "kedi.python.execution.duration",
    } <= metric_names


def test_cli_parse_failure_marks_parse_and_root_without_source_content() -> None:
    source = "@broken-name():\n"
    with installed_recording_backend() as backend:
        with pytest.raises(KediParseError):
            run_source_cli(source, adapter="pydantic", model=None)

    spans = {span.operation: span for span in backend.finished_spans}
    assert spans["parse_program"].attributes["error.type"] == "KediParseError"
    assert spans["run_program"].attributes["error.type"] == "KediParseError"
    assert source not in repr(backend.finished_spans)
    assert backend.finished_spans[-1].exceptions


def test_tool_and_dynamic_approval_have_one_nested_topology() -> None:
    def write_file(*, path: str) -> str:
        return path

    tool = ToolSpec(
        fn=write_file,
        name="write_file",
        risk="mutating",
    )
    tool = with_approval(
        tool,
        policy=ApprovalPolicy.dynamic(lambda _request: ApprovalDecision.allow()),
        adapter_shortname="test",
    )
    tool = with_tool_telemetry(tool, adapter_shortname="test")

    with installed_recording_backend() as backend:
        assert tool.fn(path="notes.txt") == "notes.txt"

    approval, call = backend.finished_spans
    assert call.name == "call write_file"
    assert call.operation == "call_tool"
    assert approval.name == "await approval write_file"
    assert approval.operation == "await_approval"
    assert approval.parent is call
    assert call.attributes["kedi.tool.approval.decision"] == "allow"
    assert "notes.txt" not in repr(backend.finished_spans)


@dataclass
class _LazyTarget:
    model: str = "test-model"

    def default_model(self) -> str:
        return self.model


def test_lazy_adapter_initialization_is_single_under_thread_contention() -> None:
    factory_lock = Lock()
    factory_calls = 0

    def factory() -> _LazyTarget:
        nonlocal factory_calls
        with factory_lock:
            factory_calls += 1
        return _LazyTarget()

    adapter = LazyAdapter(
        factory,
        kind="agent-framework",
        shortname="test",
        capabilities=AdapterCapabilities(),
    )
    assert adapter.is_initialized is False
    with installed_recording_backend() as backend:
        with ThreadPoolExecutor(max_workers=8) as pool:
            models = list(pool.map(lambda _index: adapter.default_model(), range(32)))

    assert models == ["test-model"] * 32
    assert adapter.is_initialized is True
    assert factory_calls == 1
    assert [span.name for span in backend.finished_spans] == ["initialize adapter test"]
    assert [
        metric.name for metric in backend.metrics if metric.name == "kedi.adapter.initializations"
    ] == ["kedi.adapter.initializations"]


def test_lazy_adapter_default_model_is_optional() -> None:
    adapter = LazyAdapter(
        object,
        kind="agent-framework",
        shortname="minimal",
        capabilities=AdapterCapabilities(),
    )

    assert adapter.default_model() is None


def test_lazy_adapter_records_failed_initialization_and_can_retry() -> None:
    calls = 0

    def factory() -> _LazyTarget:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("not ready")
        return _LazyTarget()

    adapter = LazyAdapter(
        factory,
        kind="agent-framework",
        shortname="test",
        capabilities=AdapterCapabilities(),
    )
    with installed_recording_backend() as backend:
        with pytest.raises(RuntimeError, match="not ready"):
            adapter.default_model()
        assert adapter.default_model() == "test-model"

    assert calls == 2
    assert [span.attributes.get("error.type") for span in backend.finished_spans] == [
        "RuntimeError",
        None,
    ]


def test_recording_backend_satisfies_telemetry_protocol() -> None:
    backend = RecordingTelemetryBackend()
    assert backend.enabled("runtime", "detailed") is True


def test_context_preflight_is_an_event_with_namespaced_attributes() -> None:
    with installed_recording_backend() as backend:
        with backend.start_span(
            scope="agent",
            name="agent test",
            operation="run_agent",
            kind="internal",
            level="lifecycle",
            attributes=None,
        ) as parent:
            ModelRequestBudget().reserve("hello")

    assert backend.finished_spans == [parent]
    event = parent.events[0]
    assert event.name == "context preflight"
    assert event.attributes["kedi.context.estimated_input_tokens"] > 0
    assert parent.attributes["kedi.context.decision"] == "admitted"
    assert parent.attributes["kedi.context.request_number"] == 1


def test_artifact_event_preserves_already_namespaced_attributes() -> None:
    with installed_recording_backend() as backend:
        with backend.start_span(
            scope="agent",
            name="agent test",
            operation="run_agent",
            kind="internal",
            level="lifecycle",
            attributes=None,
        ) as parent:
            with artifact_span("search", **{"kedi.artifact.search.result_count": 2}):
                pass

    assert parent.events[0].attributes["kedi.artifact.search.result_count"] == 2


@dataclass
class _TelemetryAdapter:
    shortname: str = "test"
    kind: str = "agent-framework"
    model: object = "test-model"

    def default_model(self) -> object:
        return self.model


def _agent_span(adapter: object):
    return agent_call_span(
        adapter,
        profile_name="planner",
        call_kind="produce",
        prompt_chars=42,
        output_field_count=2,
        tool_count=3,
        mcp_server_count=1,
        artifacts_enabled=True,
        configured_model=None,
        prompt_kind="template",
        instruction_count=2,
        history_count=4,
        cache_epoch=3,
        turn_sequence=5,
    )


def test_agent_and_model_telemetry_cover_native_ownership_usage_and_failures() -> None:
    adapter = _TelemetryAdapter()
    usage = InvocationUsage(
        requests=1,
        tool_calls=2,
        input_tokens=11,
        cache_read_tokens=7,
        cache_write_tokens=3,
        output_tokens=7,
        total_tokens=18,
        cost_usd=Decimal("0.0015"),
    )
    native_backend = RecordingTelemetryBackend(
        native_span_owners={"test": frozenset({"run_agent", "chat"})}
    )
    with installed_recording_backend(native_backend):
        reclassify_native_agent_span()
        with _agent_span(adapter) as state:
            reclassify_native_agent_span()
            state.observe(usage)
            state.complete("native", model="test-model")

        with native_backend.start_span(
            scope="agent",
            name="native run",
            operation="chat",
            kind="client",
            level="lifecycle",
            attributes=None,
        ) as native:
            with native_agent_span_scope(True):
                with _agent_span(adapter) as state:
                    reclassify_native_agent_span()
                    state.observe(usage)
                    state.complete("native", model="test-model")

        with model_call_span(adapter, call_kind="produce", configured_model=None) as model_state:
            model_state.observe(usage)
            model_state.complete("native", model="test-model")

        with pytest.raises(asyncio.CancelledError):
            with model_call_span(adapter, call_kind="invoke", configured_model=None):
                raise asyncio.CancelledError

    assert native.name == "agent planner"
    assert native.attributes["kedi.operation.name"] == "run_agent"
    assert native.events[0].name == "agent request prepared"

    with installed_recording_backend() as backend:
        with _agent_span(adapter) as state:
            state.observe(usage)
            state.complete({"ok": True}, model="resolved-model")
        with pytest.raises(asyncio.CancelledError):
            with _agent_span(adapter):
                raise asyncio.CancelledError
        with pytest.raises(RuntimeError, match="agent failed"):
            with _agent_span(adapter):
                raise RuntimeError("agent failed")
        with model_call_span(adapter, call_kind="invoke", configured_model=None) as state:
            state.observe(usage)
            state.complete("done", model=None)
        with pytest.raises(RuntimeError, match="model failed"):
            with model_call_span(adapter, call_kind="invoke", configured_model="m"):
                raise RuntimeError("model failed")

    agent_spans = [span for span in backend.finished_spans if span.operation == "run_agent"]
    assert agent_spans[0].attributes["gen_ai.usage.cache_read_tokens"] == 7
    assert agent_spans[0].attributes["gen_ai.usage.cache_write_tokens"] == 3
    assert agent_spans[0].attributes["kedi.usage.uncached_input_tokens"] == 4
    assert agent_spans[0].attributes["kedi.usage.cost_usd"] == 0.0015
    assert agent_spans[0].attributes["gen_ai.usage.output_tokens"] == 7
    assert agent_spans[0].attributes["kedi.prompt.kind"] == "template"
    assert agent_spans[0].attributes["kedi.instructions.count"] == 2
    assert agent_spans[0].attributes["kedi.history.message_count"] == 4
    assert agent_spans[0].attributes["kedi.cache.epoch"] == 3
    assert agent_spans[0].attributes["kedi.conversation.turn_sequence"] == 5
    assert agent_spans[1].attributes["error.type"] == "CancelledError"
    assert agent_spans[1].events[-1].name == "agent cancelled"
    assert agent_spans[2].attributes["error.type"] == "RuntimeError"
    chat_spans = [span for span in backend.finished_spans if span.operation == "chat"]
    assert chat_spans[0].name == "chat test"
    assert chat_spans[0].attributes["gen_ai.usage.cache_read_tokens"] == 7
    assert chat_spans[0].attributes["gen_ai.usage.cache_write_tokens"] == 3
    assert chat_spans[0].attributes["kedi.usage.uncached_input_tokens"] == 4
    assert chat_spans[0].attributes["kedi.usage.cost_usd"] == 0.0015
    assert chat_spans[1].attributes["error.type"] == "RuntimeError"

    native_backend = RecordingTelemetryBackend(
        native_span_owners={"test": frozenset({"run_agent"})}
    )
    with installed_recording_backend(native_backend):
        with pytest.raises(asyncio.CancelledError):
            with _agent_span(adapter):
                raise asyncio.CancelledError
        with pytest.raises(RuntimeError, match="native agent failed"):
            with _agent_span(adapter):
                raise RuntimeError("native agent failed")

    assert native_backend.finished_spans == []
    outcomes = {
        cast(str, metric.attributes["outcome"])
        for metric in native_backend.metrics
        if metric.name == "kedi.agent.invocations"
    }
    assert outcomes == {"cancelled", "error"}


def test_resolved_adapter_model_tolerates_unsupported_and_failing_adapters() -> None:
    class FailingAdapter:
        def telemetry_model_name(self) -> str:
            raise RuntimeError("not initialized")

    assert resolved_adapter_model(object(), None) is None
    assert resolved_adapter_model(_TelemetryAdapter(model=123), None) is None
    assert resolved_adapter_model(FailingAdapter(), None) is None
    assert resolved_adapter_model(FailingAdapter(), "configured") == "configured"


def test_tool_telemetry_covers_results_sources_native_spans_and_failures() -> None:
    byte_tool = ToolSpec(fn=lambda: b"ok", name="download", risk="read_only")
    artifact_tool = ToolSpec(fn=lambda: None, name="lookup", risk="mutating")

    with installed_recording_backend() as backend:
        with tool_call_span(byte_tool, adapter_shortname="test", argument_count=0) as state:
            state.complete(b"ok")
        with tool_call_span(artifact_tool, adapter_shortname="test", argument_count=1) as state:
            state.complete(
                {"ref_id": "tool_call_result_1", "logical_type": "dict", "size_bytes": 12}
            )
        with pytest.raises(asyncio.CancelledError):
            with tool_call_span(byte_tool, adapter_shortname="test", argument_count=0):
                raise asyncio.CancelledError
        with pytest.raises(RuntimeError, match="tool failed"):
            with tool_call_span(byte_tool, adapter_shortname="test", argument_count=0):
                raise RuntimeError("tool failed")

    calls = [span for span in backend.finished_spans if span.operation == "call_tool"]
    assert calls[0].attributes["kedi.tool.result.size_bytes"] == 2
    assert calls[1].attributes["kedi.tool.result.artifactized"] is True
    assert calls[2].attributes["error.type"] == "CancelledError"
    assert calls[2].events[-1].name == "tool cancelled"
    assert calls[3].attributes["error.type"] == "RuntimeError"

    native_backend = RecordingTelemetryBackend(
        native_span_owners={"test": frozenset({"call_tool"})}
    )
    with installed_recording_backend(native_backend):
        with native_backend.start_span(
            scope="agent",
            name="native tool",
            operation="chat",
            kind="client",
            level="lifecycle",
            attributes={"gen_ai.tool.name": "download"},
        ) as native:
            with native_agent_span_scope(True):
                with native_tool_span_scope("download"):
                    with tool_call_span(
                        byte_tool, adapter_shortname="test", argument_count=2
                    ) as state:
                        state.complete(object())
        with tool_call_span(byte_tool, adapter_shortname="test", argument_count=0) as state:
            state.complete("not attached")

    assert native.name == "call download"
    assert native.attributes["kedi.operation.name"] == "call_tool"
    assert native.attributes["kedi.tool.argument_count"] == 2
    assert (
        tool_source(ToolSpec(fn=lambda: None, name="mcp", metadata={"source": "external"})) == "mcp"
    )
    assert (
        tool_source(ToolSpec(fn=lambda: None, name="internal", metadata={"source": "kedi"}))
        == "internal"
    )


def test_native_tool_telemetry_without_a_recording_provider_span_is_safe() -> None:
    tool = ToolSpec(fn=lambda: "ok", name="download", risk="read_only")
    backend = RecordingTelemetryBackend(native_span_owners={"test": frozenset({"call_tool"})})

    with installed_recording_backend(backend):
        with native_agent_span_scope(True), native_tool_span_scope("download"):
            with tool_call_span(tool, adapter_shortname="test", argument_count=0) as state:
                state.complete("ok")
        with pytest.raises(RuntimeError, match="failed"):
            with native_agent_span_scope(True), native_tool_span_scope("download"):
                with tool_call_span(tool, adapter_shortname="test", argument_count=0):
                    raise RuntimeError("failed")

    assert backend.finished_spans == []


def test_sync_tool_telemetry_handles_awaitables_across_asyncio_contexts() -> None:
    def awaitable_tool() -> Any:
        async def resolve() -> str:
            await asyncio.sleep(0)
            return "resolved"

        return resolve()

    tool = with_tool_telemetry(
        ToolSpec(fn=awaitable_tool, name="awaitable_tool"),
        adapter_shortname="test",
    )
    assert with_tool_telemetry(tool, adapter_shortname="test") is tool

    with installed_recording_backend() as backend:
        result = tool.fn()
        assert asyncio.run(result) == "resolved"

    assert [span.name for span in backend.finished_spans] == ["call awaitable_tool"]
    assert backend.finished_spans[0].attributes["kedi.tool.result.size_bytes"] == 8


def test_sync_tool_telemetry_closes_awaitable_failure_in_source_context() -> None:
    def failing_tool() -> Any:
        async def fail() -> None:
            await asyncio.sleep(0)
            raise RuntimeError("await failed")

        return fail()

    tool = with_tool_telemetry(
        ToolSpec(fn=failing_tool, name="failing_tool"),
        adapter_shortname="test",
    )

    with installed_recording_backend() as backend:
        result = tool.fn()
        with pytest.raises(RuntimeError, match="await failed"):
            asyncio.run(result)

    assert [span.name for span in backend.finished_spans] == ["call failing_tool"]
    assert backend.finished_spans[0].attributes["error.type"] == "RuntimeError"


def test_mcp_telemetry_helpers_and_failure_are_credential_free() -> None:
    assert mcp_server_name(command="/usr/local/bin/docs-server", url=None, index=0) == "docs-server"
    assert (
        mcp_server_name(command=None, url="https://token@example.com/mcp", index=0) == "example.com"
    )
    assert mcp_server_name(command=None, url="not a host", index=2) == "server_3"
    assert mcp_server_name(command=None, url="relative/path", index=3) == "server_4"
    assert mcp_server_name(command=None, url=None, index=4) == "server_5"
    assert mcp_transport_name("http") == "streamable_http"
    assert mcp_transport_name("stdio") == "stdio"

    with installed_recording_backend() as backend:
        with pytest.raises(RuntimeError, match="cannot connect"):
            with mcp_initialization_span(
                adapter_shortname="test",
                server_name="example.com",
                transport="streamable_http",
            ):
                raise RuntimeError("cannot connect")

    span = backend.finished_spans[0]
    assert span.attributes["error.type"] == "RuntimeError"
    assert "token" not in repr(span)


def test_approval_telemetry_covers_native_mechanism_immediate_and_async_cancellation() -> None:
    handler = SimpleNamespace(telemetry_mechanism="native_adapter")
    assert approval_mechanism(handler) == "native_adapter"
    request = ApprovalRequest.create(
        tool_name="write_file",
        arguments={},
        risk="mutating",
        adapter_shortname="test",
        description=None,
        metadata=None,
    )

    with installed_recording_backend() as backend:
        tool = ToolSpec(fn=lambda: None, name="write_file", risk="mutating")
        with tool_call_span(tool, adapter_shortname="test", argument_count=0) as tool_state:
            assert tool_state.span is not None
            parent = tool_state.span
            record_immediate_decision(
                request,
                ApprovalDecision.allow(),
                mode="static_allow",
            )
            with pytest.raises(asyncio.CancelledError):
                with approval_span(request, mechanism="native_adapter"):
                    raise asyncio.CancelledError

    assert parent.attributes["kedi.tool.risk"] == "mutating"
    assert parent.attributes["kedi.tool.approval.required"] is True
    assert parent.attributes["kedi.tool.approval.decision"] == "allow"
    approval = next(span for span in backend.finished_spans if span.operation == "await_approval")
    assert approval.attributes["error.type"] == "CancelledError"
    metric = next(metric for metric in backend.metrics if metric.name == "kedi.approval.requests")
    assert metric.attributes["decision"] == "cancelled"


def test_subagent_and_workflow_native_tool_spans_skip_non_scalar_attributes() -> None:
    with installed_recording_backend():
        delegate_tool = ToolSpec(fn=lambda: None, name="delegate_task")
        with tool_call_span(
            delegate_tool, adapter_shortname="test", argument_count=0
        ) as tool_state:
            assert tool_state.span is not None
            subagent = tool_state.span
            with _subagent_run_span({"kedi.subagent.profile": "planner", "ignored": object()}):
                pass
        workflow_tool = ToolSpec(fn=lambda: None, name="run_workflow")
        with tool_call_span(
            workflow_tool, adapter_shortname="test", argument_count=0
        ) as tool_state:
            assert tool_state.span is not None
            workflow = tool_state.span
            with _dynamic_workflow_span(
                {"kedi.workflow.parent_profile": "builder", "ignored": object()}
            ):
                pass

    assert subagent.name == "planner run"
    assert subagent.attributes["kedi.operation.name"] == "run_agent"
    assert "ignored" not in subagent.attributes
    assert workflow.name == "builder workflow"
    assert workflow.attributes["kedi.operation.name"] == "run_workflow"
    assert "ignored" not in workflow.attributes


def test_source_capture_covers_cli_modules_procedures_and_python_api_bind(tmp_path: Path) -> None:
    module = tmp_path / "lib.kedi"
    module.write_text("@show():\n  = ok\n> export:\n  show\n", encoding="utf-8")
    main = tmp_path / "main.kedi"
    main.write_text("> import: lib\n= <show()>\n", encoding="utf-8")
    bound_source = tmp_path / "bound.kedi"
    bound_source.write_text("= done\n", encoding="utf-8")

    with installed_recording_backend(
        RecordingTelemetryBackend(capture=frozenset({"source_path"}))
    ) as backend:
        assert run_program_cli(str(main), adapter="pydantic", model=None) == (0, "ok")
        assert run_source_cli(
            "= done", adapter="pydantic", model=None, source_path=str(tmp_path / "source.kedi")
        ) == (0, "done")
        assert run_source_cli(
            "= ```\nreturn 1\n```\n",
            adapter="pydantic",
            model=None,
            source_path=str(tmp_path / "python.kedi"),
        ) == (0, "1")

        @kedi.bind(file=bound_source)
        def bound() -> str:
            raise AssertionError("the Kedi file is the implementation")

        assert bound() == "done"

    paths = {
        cast(str, span.attributes["code.file.path"])
        for span in backend.finished_spans
        if "code.file.path" in span.attributes
    }
    assert str(main) in paths
    assert str(module) in paths
    assert str(tmp_path / "source.kedi") in paths
    assert str(tmp_path / "python.kedi") in paths
    assert str(bound_source) in paths


def test_parser_telemetry_handles_bytes_multiple_diagnostics_and_internal_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    diagnostics = [
        KediDiagnostic(
            severity="error",
            message="primary",
            span=Span(0, 1, 1, 1, 1, 2, "broken.kedi"),
        ),
        KediDiagnostic(
            severity="error",
            message="secondary",
            span=Span(2, 3, 2, 1, 2, 2, "broken.kedi"),
            code="K002",
        ),
        KediDiagnostic(
            severity="error",
            message="without code",
            span=Span(4, 5, 3, 1, 3, 2, "broken.kedi"),
        ),
    ]

    def raise_diagnostics(*_args: object, **_kwargs: object) -> Any:
        raise KediParseError(diagnostics)

    monkeypatch.setattr(parser_module, "cst_to_ast", raise_diagnostics)
    with installed_recording_backend(
        RecordingTelemetryBackend(capture=frozenset({"source_path"}))
    ) as backend:
        with pytest.raises(KediParseError, match="primary"):
            parser_module.parse_program(b"broken", source_path="broken.kedi")  # type: ignore[arg-type]

    span = backend.finished_spans[0]
    assert span.attributes["kedi.parse.error_count"] == 3
    assert span.attributes["code.file.path"] == "broken.kedi"
    assert span.events[0].attributes["kedi.diagnostic.code"] == "K002"
    assert "kedi.diagnostic.code" not in span.events[1].attributes

    monkeypatch.setattr(parser_module, "cst_to_ast", raise_diagnostics)
    diagnostics.clear()
    with installed_recording_backend() as backend:
        with pytest.raises(KediParseError, match="kedi parse error"):
            parser_module.parse_program("broken")
    assert backend.finished_spans[0].attributes["kedi.parse.error_count"] == 0

    def raise_internal(_source: bytes) -> Any:
        raise RuntimeError("parser backend failed")

    monkeypatch.setattr(parser_module, "_parse", raise_internal)
    with installed_recording_backend() as backend:
        with pytest.raises(RuntimeError, match="parser backend failed"):
            parser_module.parse_program("broken")
    assert backend.finished_spans[0].attributes["error.type"] == "RuntimeError"


def test_runtime_cancellation_and_backend_lifecycle_are_observable() -> None:
    previous = get_backend()
    backend = RecordingTelemetryBackend()
    try:
        assert install_backend(backend) is previous
        assert get_backend() is backend
        with pytest.raises(asyncio.CancelledError):
            with runtime_span("kedi run cancelled", "run_program"):
                raise asyncio.CancelledError
        reset_backend()
        assert get_backend() is not backend
    finally:
        restore_backend(previous)

    span = backend.finished_spans[0]
    assert span.attributes["error.type"] == "CancelledError"
    assert span.events[-1].name == "run cancelled"


def test_noop_backend_span_supports_attribute_reads() -> None:
    assert NoOpTelemetryBackend().current_span().get_attribute("missing") is None


def test_backend_compare_and_restore_preserves_newer_owner() -> None:
    previous = get_backend()
    owned = RecordingTelemetryBackend()
    newer = RecordingTelemetryBackend()
    try:
        install_backend(owned)
        assert restore_backend_if_current(expected=owned, replacement=previous) is True
        assert get_backend() is previous

        install_backend(owned)
        install_backend(newer)
        assert restore_backend_if_current(expected=owned, replacement=previous) is False
        assert get_backend() is newer
    finally:
        restore_backend(previous)


def test_pydantic_telemetry_capability_is_owned_once_and_reclassifies() -> None:
    backend = RecordingTelemetryBackend(native_span_owners={"pydantic": frozenset({"run_agent"})})
    adapter = SimpleNamespace(native_telemetry_enabled=lambda operation: operation == "run_agent")
    with installed_recording_backend(backend):
        capabilities = PydanticAdapter._with_telemetry_capability(cast(Any, adapter), None)
        assert len(capabilities) == 1
        assert (
            PydanticAdapter._with_telemetry_capability(cast(Any, adapter), capabilities)
            == capabilities
        )
        with pytest.raises(TypeError, match="list or tuple"):
            PydanticAdapter._with_telemetry_capability(cast(Any, adapter), object())

        disabled = SimpleNamespace(native_telemetry_enabled=lambda operation: False)
        assert PydanticAdapter._with_telemetry_capability(cast(Any, disabled), None) == []

    async def handler() -> str:
        return "done"

    capability = _KediTelemetryCapability()
    ctx = SimpleNamespace()
    result = asyncio.run(capability.wrap_run(cast(Any, ctx), handler=cast(Any, handler)))
    assert result == "done"

    async def tool_handler(args: object) -> object:
        return args

    tool_args = object()
    result = asyncio.run(
        capability.wrap_tool_execute(
            cast(Any, ctx),
            call=cast(Any, SimpleNamespace(tool_name="lookup")),
            tool_def=cast(Any, SimpleNamespace()),
            args=cast(Any, tool_args),
            handler=cast(Any, tool_handler),
        )
    )
    assert result is tool_args


def test_native_agent_ownership_and_reclassification_require_an_active_span() -> None:
    class NativeAdapter:
        shortname = "native"

        def native_telemetry_enabled(self, operation: str) -> bool:
            return operation == "run_agent"

    adapter = NativeAdapter()
    backend = RecordingTelemetryBackend(native_span_owners={"native": frozenset({"run_agent"})})
    with installed_recording_backend(backend):
        assert adapter_owns_native_span(adapter, "run_agent") is True
        assert adapter_owns_native_span(adapter, "chat") is False
        with agent_call_span(
            adapter,
            profile_name=None,
            call_kind="invoke",
            prompt_chars=1,
            output_field_count=0,
            tool_count=0,
            mcp_server_count=0,
            artifacts_enabled=True,
            configured_model=None,
        ):
            with native_agent_span_scope(True):
                reclassify_native_agent_span()

    assert backend.finished_spans == []
