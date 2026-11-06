from __future__ import annotations

import asyncio
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, Literal, Protocol
from urllib.parse import urlsplit

from typing_extensions import runtime_checkable

from ..agent_profile.telemetry import capability_attributes, capability_trace_id
from ..telemetry import (
    TelemetryAsset,
    TelemetrySpan,
    add_counter,
    asset_fingerprint,
    capture_enabled,
    current_span,
    owns_native_spans,
    record_asset,
    record_histogram,
    span,
)
from ..telemetry.naming import qualified_type_name
from .invocation import InvocationUsage
from .meta import AgentAdapterKind
from .tools.specs import ToolSpec

if TYPE_CHECKING:
    from ..agent_profile import AgentProfile
    from ..agent_profile.capabilities import CapabilityToolPolicy

AgentCallKind = Literal["produce", "invoke", "subagent"]


def record_conversation_assets(
    *,
    prompt: str,
    instructions: str | None,
    profile: AgentProfile,
    tools: Sequence[ToolSpec],
    output_schema: Mapping[str, str] | None,
    call_kind: AgentCallKind,
    source_path: str | None,
    source_line: int | None,
    source_snippet: str | None,
) -> None:
    profile_name = profile.name or "kedi"
    template_key = (
        f"{source_path}:{source_line or 0}:{call_kind}"
        if source_path is not None
        else f"{profile_name}:{call_kind}"
    )
    profile_locator = f"kedi:profile:{profile_name}"
    profile_content = {
        "name": profile.name,
        "model": profile.model,
        "adapter_kind": profile.adapter_kind,
        "adapter_shortname": profile.adapter_shortname,
        "agent_command": profile.agent_command,
        "effort": profile.effort,
        "system": profile.system,
        "settings": dict(profile.settings),
        "tools": list(profile.tools),
        "tool_bindings": sorted(profile.tool_bindings),
        "mcp_servers": [repr(server) for server in profile.mcp_servers],
        "approval": repr(profile.approval),
        "skills": profile.skills,
        "subagents": list(profile.subagents),
        "subagent_bindings": sorted(profile.subagent_bindings),
        "max_agents": profile.max_agents,
        "workflow_mode": profile.workflow_mode,
        "workflow_tools": None if profile.workflow_tools is None else list(profile.workflow_tools),
        "history": profile.history,
        "compaction": repr(profile.compaction),
        "output_type": repr(profile.output_type),
        "artifacts": repr(profile.artifacts),
        "capabilities": [capability_attributes(capability) for capability in profile.capabilities],
    }
    _record_behavioral_asset(
        kind="agent_profile",
        local_id=profile_name,
        name=profile_name,
        source_locator=profile_locator,
        content=profile_content,
        metadata={"call_kind": call_kind},
    )
    source_locator: str | None = None
    if source_path is not None:
        source_locator = f"kedi:source:{source_path}:{source_line or 0}"
        source_content = {
            "path": source_path,
            "line": source_line,
            "snippet": source_snippet if capture_enabled("source_snippet") else None,
        }
        # Definitions go first so the rendered template below can link to its source.
        _record_behavioral_asset(
            kind="source_identity",
            local_id=f"{source_path}:{source_line or 0}",
            name=source_path,
            source_locator=source_locator,
            content=source_content,
            metadata={"path": source_path, "line": source_line},
            capture_content=source_snippet is not None and capture_enabled("source_snippet"),
        )
    _record_behavioral_asset(
        kind="template",
        local_id=template_key,
        name=f"{profile_name} {call_kind} template",
        source_locator=f"kedi:template:{template_key}",
        content=prompt,
        metadata={
            "call_kind": call_kind,
            "profile": profile_name,
            "source_path": source_path,
            "source_line": source_line,
        },
        sensitivity="sensitive",
        representation="effective",
        definition_locator=source_locator,
    )
    if instructions is not None:
        _record_behavioral_asset(
            kind="prompt",
            local_id=f"{profile_name}:instructions",
            name=f"{profile_name} instructions",
            source_locator=f"kedi:prompt:{profile_name}:instructions",
            content=instructions,
            metadata={"profile": profile_name},
            sensitivity="sensitive",
            representation="effective",
            definition_locator=profile_locator,
        )
    for tool in tools:
        source = tool_source(tool)
        _record_behavioral_asset(
            kind="tool",
            local_id=tool.name,
            name=tool.name,
            source_locator=f"kedi:tool:{tool.name}",
            content={
                "name": tool.name,
                "description": tool.description,
                "risk": tool.risk,
                "source": source,
                "artifact_result": tool.artifact_result,
                "sequential": tool.sequential,
                "metadata": dict(tool.metadata or {}),
            },
            metadata={"source": source, "risk": tool.risk},
        )
        for schema_role, schema in (
            ("input", tool.json_schema),
            ("return", tool.return_schema),
        ):
            if schema is not None:
                _record_behavioral_asset(
                    kind="schema",
                    local_id=f"tool:{tool.name}:{schema_role}",
                    name=f"{tool.name} {schema_role} schema",
                    source_locator=f"kedi:schema:tool:{tool.name}:{schema_role}",
                    content=schema,
                    metadata={"owner": tool.name, "role": schema_role},
                )
    if output_schema is not None:
        _record_behavioral_asset(
            kind="schema",
            local_id=f"template:{template_key}:output",
            name=f"{profile_name} output schema",
            source_locator=f"kedi:schema:template:{template_key}:output",
            content=dict(output_schema),
            metadata={"owner": template_key, "role": "output"},
        )


def _record_behavioral_asset(
    *,
    kind: Literal[
        "agent_profile",
        "prompt",
        "schema",
        "source_identity",
        "template",
        "tool",
    ],
    local_id: str,
    name: str,
    source_locator: str,
    content: Any,
    metadata: Mapping[str, Any],
    sensitivity: Literal["public", "internal", "sensitive"] = "internal",
    capture_content: bool | None = None,
    representation: Literal["definition", "effective"] = "definition",
    definition_locator: str | None = None,
) -> None:
    retain_content = capture_enabled("content") if capture_content is None else capture_content
    record_asset(
        TelemetryAsset(
            kind=kind,
            local_id=local_id,
            name=name,
            source_locator=source_locator,
            content_fingerprint=asset_fingerprint(content),
            content=content if retain_content else None,
            metadata=dict(metadata),
            sensitivity=sensitivity,
            representation=representation,
            definition_locator=definition_locator,
        )
    )


@dataclass(frozen=True, slots=True)
class NativeAgentSpanMetadata:
    """Kedi metadata applied inside an adapter-owned agent span."""

    name: str
    attributes: Mapping[str, bool | str | int]
    prompt_chars: int
    tool_count: int
    output_field_count: int


_native_agent_span_metadata: ContextVar[NativeAgentSpanMetadata | None] = ContextVar(
    "kedi_native_agent_span_metadata",
    default=None,
)
_native_agent_span_active: ContextVar[bool] = ContextVar(
    "kedi_native_agent_span_active",
    default=False,
)
_native_tool_call_name: ContextVar[str | None] = ContextVar(
    "kedi_native_tool_call_name",
    default=None,
)
_current_tool_call_name: ContextVar[str | None] = ContextVar(
    "kedi_current_tool_call_name",
    default=None,
)


@runtime_checkable
class _AdapterIdentity(Protocol):
    shortname: str


@runtime_checkable
class _AdapterKind(Protocol):
    kind: AgentAdapterKind


@runtime_checkable
class _AdapterDefaultModel(Protocol):
    def default_model(self) -> str | None: ...


@runtime_checkable
class _AdapterTelemetryModel(Protocol):
    def telemetry_model_name(self) -> str | None: ...


@runtime_checkable
class _AdapterNativeTelemetry(Protocol):
    def native_telemetry_enabled(self, operation: str) -> bool: ...


def adapter_shortname(adapter: object) -> str:
    """Return stable telemetry identity without imposing it on legacy adapters."""

    if isinstance(adapter, _AdapterIdentity) and adapter.shortname:
        return adapter.shortname
    return type(adapter).__name__


def adapter_kind(adapter: object) -> AgentAdapterKind:
    if isinstance(adapter, _AdapterKind) and adapter.kind in {
        "agent-framework",
        "agent-harness",
    }:
        return adapter.kind
    return "agent-framework"


def adapter_kind_name(kind: AgentAdapterKind) -> Literal["framework", "harness"]:
    return "harness" if kind == "agent-harness" else "framework"


def adapter_span_kind(adapter: object) -> Literal["internal", "client"]:
    if adapter_kind(adapter) == "agent-harness" or adapter_shortname(adapter) == "webgpu":
        return "client"
    return "internal"


def adapter_owns_native_span(adapter: object, operation: str) -> bool:
    """Return whether this exact adapter invocation is expected to own a native span."""

    adapter_name = adapter_shortname(adapter)
    if not owns_native_spans(adapter_name, operation):
        return False
    if isinstance(adapter, _AdapterNativeTelemetry):
        return adapter.native_telemetry_enabled(operation)
    return True


@contextmanager
def native_agent_span_scope(active: bool) -> Generator[None, None, None]:
    """Mark the concrete adapter invocation as natively instrumented."""

    token = _native_agent_span_active.set(active)
    try:
        yield
    finally:
        _native_agent_span_active.reset(token)


def native_agent_span_active() -> bool:
    return _native_agent_span_active.get()


@contextmanager
def native_tool_span_scope(tool_name: str) -> Generator[None, None, None]:
    """Identify the provider-owned tool span for one concrete execution."""

    token = _native_tool_call_name.set(tool_name)
    try:
        yield
    finally:
        _native_tool_call_name.reset(token)


def current_tool_call_name() -> str | None:
    """Return Kedi's current tool identity without reading provider span internals."""

    return _current_tool_call_name.get()


@dataclass(slots=True)
class AgentCallTelemetry:
    adapter_name: str
    adapter_kind: Literal["framework", "harness"]
    call_kind: AgentCallKind
    span: TelemetrySpan | None
    native_owned: bool
    usage: InvocationUsage = InvocationUsage()

    def observe(self, usage: InvocationUsage) -> None:
        self.usage = self.usage + usage

    def complete(self, result: Any, *, model: str | None) -> None:
        if self.span is None:
            return
        self.span.set_attribute("kedi.result.type", qualified_type_name(result))
        if model is not None:
            self.span.set_attribute("gen_ai.request.model", model)


@contextmanager
def agent_call_span(
    adapter: object,
    *,
    profile_name: str | None,
    call_kind: AgentCallKind,
    prompt_chars: int,
    output_field_count: int,
    tool_count: int,
    mcp_server_count: int,
    artifacts_enabled: bool,
    configured_model: str | None,
    prompt_kind: str | None = None,
    instruction_count: int = 0,
    history_count: int = 0,
    cache_epoch: int | None = None,
    turn_sequence: int | None = None,
) -> Generator[AgentCallTelemetry, None, None]:
    """Trace one semantic Kedi agent call unless its adapter owns a native span."""

    adapter_name = adapter_shortname(adapter)
    kind_name = adapter_kind_name(adapter_kind(adapter))
    native_owned = adapter_owns_native_span(adapter, "run_agent")
    attributes: dict[str, bool | str | int] = {
        "gen_ai.agent.name": profile_name or "kedi",
        "gen_ai.output.type": "json" if call_kind == "produce" else "text",
        "kedi.adapter.name": adapter_name,
        "kedi.adapter.kind": kind_name,
        "kedi.agent.call.kind": call_kind,
        "kedi.agent.profile": profile_name or "kedi",
        "kedi.template.kind": "structured" if call_kind == "produce" else "raw",
        "kedi.output.field_count": output_field_count,
        "kedi.tools.available_count": tool_count,
        "kedi.mcp.server_count": mcp_server_count,
        "kedi.artifacts.enabled": artifacts_enabled,
        "kedi.instructions.count": instruction_count,
        "kedi.history.message_count": history_count,
    }
    if prompt_kind is not None:
        attributes["kedi.prompt.kind"] = prompt_kind
    if cache_epoch is not None:
        attributes["kedi.cache.epoch"] = cache_epoch
    if turn_sequence is not None:
        attributes["kedi.conversation.turn_sequence"] = turn_sequence
    if configured_model is not None:
        attributes["gen_ai.request.model"] = configured_model

    span_context = (
        nullcontext(None)
        if native_owned
        else span(
            "agent",
            f"agent {profile_name or 'kedi'}",
            "run_agent",
            kind=adapter_span_kind(adapter),
            attributes=attributes,
        )
    )
    metric_attributes = {
        "adapter": adapter_name,
        "adapter_kind": kind_name,
        "call_kind": call_kind,
    }
    started_at = perf_counter()
    outcome = "success"
    metadata_token = _native_agent_span_metadata.set(
        NativeAgentSpanMetadata(
            name=f"agent {profile_name or 'kedi'}",
            attributes=attributes,
            prompt_chars=prompt_chars,
            tool_count=tool_count,
            output_field_count=output_field_count,
        )
        if native_owned
        else None
    )
    try:
        with span_context as active:
            state = AgentCallTelemetry(
                adapter_name=adapter_name,
                adapter_kind=kind_name,
                call_kind=call_kind,
                span=active,
                native_owned=native_owned,
            )
            if active is not None:
                _add_agent_request_event(
                    active,
                    prompt_chars=prompt_chars,
                    tool_count=tool_count,
                    output_field_count=output_field_count,
                )
            try:
                yield state
            except asyncio.CancelledError as exc:
                outcome = "cancelled"
                if active is not None:
                    active.set_attribute("error.type", type(exc).__name__)
                    active.add_event("agent cancelled")
                raise
            except BaseException as exc:
                outcome = "error"
                if active is not None:
                    active.set_attribute("error.type", type(exc).__name__)
                raise
            finally:
                duration = max(0.0, perf_counter() - started_at)
                dimensions = {**metric_attributes, "outcome": outcome}
                add_counter("agent", "kedi.agent.invocations", attributes=dimensions)
                record_histogram(
                    "agent",
                    "kedi.agent.duration",
                    duration,
                    unit="s",
                    attributes=dimensions,
                )
                _record_agent_usage(state, dimensions=dimensions)
    finally:
        _native_agent_span_metadata.reset(metadata_token)


def reclassify_native_agent_span() -> None:
    """Apply Kedi semantics to the current adapter-owned agent span."""

    metadata = _native_agent_span_metadata.get()
    if metadata is None or not native_agent_span_active():
        return
    active = current_span()
    if not active.is_recording():
        return
    active.update_name(metadata.name)
    active.set_attribute("kedi.operation.name", "run_agent")
    active.set_attribute("gen_ai.operation.name", "invoke_agent")
    for name, value in metadata.attributes.items():
        active.set_attribute(name, value)
    _add_agent_request_event(
        active,
        prompt_chars=metadata.prompt_chars,
        tool_count=metadata.tool_count,
        output_field_count=metadata.output_field_count,
    )


def _add_agent_request_event(
    active: TelemetrySpan,
    *,
    prompt_chars: int,
    tool_count: int,
    output_field_count: int,
) -> None:
    active.add_event(
        "agent request prepared",
        {
            "kedi.agent.prompt_chars": prompt_chars,
            "kedi.tools.available_count": tool_count,
            "kedi.output.field_count": output_field_count,
        },
    )


def resolved_adapter_model(adapter: object, configured_model: str | None) -> str | None:
    if configured_model is not None:
        return configured_model
    try:
        if isinstance(adapter, _AdapterTelemetryModel):
            model = adapter.telemetry_model_name()
        elif isinstance(adapter, _AdapterDefaultModel):
            model = adapter.default_model()
        else:
            return None
    except Exception:
        return None
    return model if isinstance(model, str) else None


def _record_agent_usage(
    state: AgentCallTelemetry,
    *,
    dimensions: Mapping[str, str],
) -> None:
    usage = state.usage
    if state.span is not None:
        state.span.set_attribute("kedi.agent.requests", usage.requests)
        state.span.set_attribute("kedi.agent.tool_calls", usage.tool_calls)
        state.span.set_attribute("gen_ai.usage.input_tokens", usage.input_tokens)
        state.span.set_attribute("gen_ai.usage.cache_read_tokens", usage.cache_read_tokens)
        state.span.set_attribute("gen_ai.usage.cache_write_tokens", usage.cache_write_tokens)
        state.span.set_attribute("kedi.usage.uncached_input_tokens", usage.uncached_input_tokens)
        state.span.set_attribute("gen_ai.usage.output_tokens", usage.output_tokens)
        if usage.cost_usd is not None:
            state.span.set_attribute("kedi.usage.cost_usd", float(usage.cost_usd))
    if state.native_owned:
        return
    for token_type, value in (
        ("input", usage.input_tokens),
        ("cache_read", usage.cache_read_tokens),
        ("cache_write", usage.cache_write_tokens),
        ("output", usage.output_tokens),
    ):
        if value:
            record_histogram(
                "agent",
                "gen_ai.client.token.usage",
                value,
                unit="token",
                attributes={"gen_ai.token.type": token_type, **dimensions},
            )


@dataclass(slots=True)
class ModelCallTelemetry:
    adapter_name: str
    call_kind: AgentCallKind
    span: TelemetrySpan | None
    native_owned: bool
    started_at: float
    usage: InvocationUsage = InvocationUsage()
    outcome: str = "success"
    error_type: str | None = None
    model: str | None = None

    def observe(self, usage: InvocationUsage) -> None:
        self.usage = self.usage + usage

    def complete(self, result: Any, *, model: str | None) -> None:
        self.model = model
        if self.span is None:
            return
        if model is not None:
            self.span.update_name(f"chat {model}")
            self.span.set_attribute("gen_ai.request.model", model)
        self.span.set_attribute("kedi.result.type", qualified_type_name(result))

    def fail(self, exc: BaseException) -> None:
        self.outcome = "cancelled" if isinstance(exc, asyncio.CancelledError) else "error"
        self.error_type = type(exc).__name__
        if self.span is not None:
            self.span.set_attribute("error.type", self.error_type)

    def finish(self, *, model: str | None) -> None:
        if self.span is not None:
            self.span.set_attribute("gen_ai.usage.input_tokens", self.usage.input_tokens)
            self.span.set_attribute("gen_ai.usage.cache_read_tokens", self.usage.cache_read_tokens)
            self.span.set_attribute(
                "gen_ai.usage.cache_write_tokens", self.usage.cache_write_tokens
            )
            self.span.set_attribute(
                "kedi.usage.uncached_input_tokens", self.usage.uncached_input_tokens
            )
            self.span.set_attribute("gen_ai.usage.output_tokens", self.usage.output_tokens)
            if self.usage.cost_usd is not None:
                self.span.set_attribute("kedi.usage.cost_usd", float(self.usage.cost_usd))
        if self.native_owned:
            return
        dimensions: dict[str, str] = {
            "gen_ai.operation.name": "chat",
            "kedi.adapter.name": self.adapter_name,
            "kedi.agent.call.kind": self.call_kind,
        }
        if self.error_type is not None:
            dimensions["error.type"] = self.error_type
        if model is not None:
            dimensions["gen_ai.request.model"] = model
        record_histogram(
            "agent",
            "gen_ai.client.operation.duration",
            max(0.0, perf_counter() - self.started_at),
            unit="s",
            attributes=dimensions,
        )
        for token_type, value in (
            ("input", self.usage.input_tokens),
            ("cache_read", self.usage.cache_read_tokens),
            ("cache_write", self.usage.cache_write_tokens),
            ("output", self.usage.output_tokens),
        ):
            if value:
                record_histogram(
                    "agent",
                    "gen_ai.client.token.usage",
                    value,
                    unit="token",
                    attributes={"gen_ai.token.type": token_type, **dimensions},
                )


@contextmanager
def model_call_span(
    adapter: object,
    *,
    call_kind: AgentCallKind,
    configured_model: str | None,
) -> Generator[ModelCallTelemetry, None, None]:
    """Trace one adapter model boundary when no native instrumentor owns it."""

    adapter_name = adapter_shortname(adapter)
    native_owned = adapter_owns_native_span(adapter, "chat")
    display_model = configured_model or adapter_name
    span_context = (
        nullcontext(None)
        if native_owned
        else span(
            "agent",
            f"chat {display_model}",
            "chat",
            kind="client",
            attributes={
                "kedi.adapter.name": adapter_name,
                "kedi.agent.call.kind": call_kind,
                **(
                    {"gen_ai.request.model": configured_model}
                    if configured_model is not None
                    else {}
                ),
            },
        )
    )
    with span_context as active:
        state = ModelCallTelemetry(
            adapter_name=adapter_name,
            call_kind=call_kind,
            span=active,
            native_owned=native_owned,
            started_at=perf_counter(),
        )
        state.model = configured_model
        try:
            yield state
        except BaseException as exc:
            state.fail(exc)
            raise
        finally:
            if state.model is None and state.outcome == "success":
                state.model = resolved_adapter_model(adapter, None)
            state.finish(model=state.model)


@dataclass(slots=True)
class ToolCallTelemetry:
    span: TelemetrySpan | None
    source: str
    risk: str
    adapter_name: str
    started_at: float
    outcome: str = "success"

    def complete(self, result: Any) -> None:
        if self.span is None:
            return
        self.span.set_attribute("kedi.tool.result.type", qualified_type_name(result))
        size = _result_size(result)
        if size is not None:
            self.span.set_attribute("kedi.tool.result.size_bytes", size)
        artifactized = _is_artifact_ref(result)
        self.span.set_attribute("kedi.tool.result.artifactized", artifactized)
        self.span.add_event(
            "tool output admitted",
            {"kedi.tool.output.admission": "artifact" if artifactized else "inline"},
        )

    def fail(self, exc: BaseException) -> None:
        self.outcome = "cancelled" if isinstance(exc, asyncio.CancelledError) else "error"
        if self.span is None:
            return
        self.span.set_attribute("error.type", type(exc).__name__)
        if isinstance(exc, asyncio.CancelledError):
            self.span.add_event("tool cancelled")

    def finish(self) -> None:
        dimensions = {
            "adapter": self.adapter_name,
            "source": self.source,
            "risk": self.risk,
            "outcome": self.outcome,
        }
        add_counter("agent", "kedi.tool.calls", attributes=dimensions)
        record_histogram(
            "agent",
            "kedi.tool.duration",
            max(0.0, perf_counter() - self.started_at),
            unit="s",
            attributes=dimensions,
        )


@contextmanager
def tool_call_span(
    tool: ToolSpec,
    *,
    adapter_shortname: str,
    argument_count: int,
) -> Generator[ToolCallTelemetry, None, None]:
    source = tool_source(tool)
    native_owned = (
        owns_native_spans(adapter_shortname, "call_tool")
        and native_agent_span_active()
        and _native_tool_call_name.get() == tool.name
    )
    active: TelemetrySpan | None = None
    span_context: Any
    if native_owned:
        current = current_span()
        if current.is_recording():
            active = current
            current.update_name(f"call {tool.name}")
            current.set_attribute("kedi.operation.name", "call_tool")
            current.set_attribute("gen_ai.operation.name", "execute_tool")
            current.set_attribute("gen_ai.tool.name", tool.name)
        span_context = nullcontext(active)
    else:
        span_context = span(
            "agent",
            f"call {tool.name}",
            "call_tool",
            attributes={
                "gen_ai.tool.name": tool.name,
                "kedi.adapter.name": adapter_shortname,
                "kedi.tool.source": source,
                "kedi.tool.risk": tool.risk,
                "kedi.tool.argument_count": argument_count,
            },
        )

    tool_token = _current_tool_call_name.set(tool.name)
    try:
        with span_context as active_span:
            if active_span is not None:
                from .tools.records import current_tool_call_tracker

                tracker = current_tool_call_tracker()
                if tracker is not None:
                    active_span.set_attribute("gen_ai.tool.call.id", tracker.tool_call_id)
                active_span.set_attribute("kedi.adapter.name", adapter_shortname)
                active_span.set_attribute("kedi.tool.source", source)
                active_span.set_attribute("kedi.tool.risk", tool.risk)
                active_span.set_attribute("kedi.tool.argument_count", argument_count)
                record_tool_ownership(tool, tool.capability_policy)
                active_span.add_event("tool execution started")
            state = ToolCallTelemetry(
                span=active_span,
                source=source,
                risk=tool.risk,
                adapter_name=adapter_shortname,
                started_at=perf_counter(),
            )
            try:
                yield state
            except BaseException as exc:
                state.fail(exc)
                raise
            finally:
                state.finish()
    finally:
        _current_tool_call_name.reset(tool_token)


def record_tool_ownership(tool: ToolSpec, policy: CapabilityToolPolicy | None) -> None:
    active = current_span()
    if not active.is_recording() or current_tool_call_name() != tool.name:
        return
    active.set_attribute(
        "kedi.tool.capability.owners",
        () if policy is None else tuple(capability_trace_id(owner) for owner in policy.owners),
    )
    active.set_attribute(
        "kedi.tool.approval.source",
        "tool"
        if tool.approval is not None
        else "capability"
        if policy is not None and policy.approval is not None
        else "agent",
    )


def tool_source(tool: ToolSpec) -> str:
    if tool.proc is not None:
        return "procedure"
    metadata = tool.metadata or {}
    kind = metadata.get("kedi_tool_kind") or metadata.get("kind")
    if kind == "skill":
        return "skill"
    if kind == "artifact_management":
        return "artifact"
    if kind in {"subagent", "subagent_lifecycle"}:
        return "subagent"
    if kind == "dynamic_workflow":
        return "workflow"
    source = metadata.get("source")
    if source in {"external", "langchain-external"}:
        return "mcp"
    if source in {"kedi", "claude-native"}:
        return "internal"
    return "python"


@contextmanager
def mcp_initialization_span(
    *,
    adapter_shortname: str,
    server_name: str,
    transport: str,
    server_count: int = 1,
) -> Generator[TelemetrySpan, None, None]:
    started_at = perf_counter()
    outcome = "success"
    attributes = {
        "kedi.adapter.name": adapter_shortname,
        "kedi.mcp.server.name": server_name,
        "kedi.mcp.transport": transport,
        "kedi.mcp.server_count": server_count,
        "kedi.mcp.reused": False,
    }
    with span(
        "agent",
        f"initialize mcp {server_name}",
        "initialize_mcp",
        kind="client",
        attributes=attributes,
    ) as active:
        try:
            yield active
        except BaseException as exc:
            outcome = "error"
            active.set_attribute("error.type", type(exc).__name__)
            raise
        else:
            active.add_event("mcp connected")
        finally:
            dimensions = {
                "adapter": adapter_shortname,
                "transport": transport,
                "outcome": outcome,
            }
            add_counter("agent", "kedi.mcp.initializations", attributes=dimensions)
            record_histogram(
                "agent",
                "kedi.mcp.initialization.duration",
                max(0.0, perf_counter() - started_at),
                unit="s",
                attributes=dimensions,
            )


def mcp_server_name(*, command: str | None, url: str | None, index: int) -> str:
    """Return a credential-free MCP server label suitable for a span name."""

    if command:
        return Path(command).name
    if url:
        parsed = urlsplit(url)
        if parsed.hostname:
            return parsed.hostname
    return f"server_{index + 1}"


def mcp_transport_name(transport: str) -> str:
    return "streamable_http" if transport == "http" else transport


def _result_size(result: Any) -> int | None:
    if isinstance(result, str):
        return len(result.encode())
    if isinstance(result, bytes):
        return len(result)
    return None


def _is_artifact_ref(result: Any) -> bool:
    return (
        isinstance(result, Mapping)
        and {
            "ref_id",
            "logical_type",
            "size_bytes",
        }
        <= result.keys()
    )
