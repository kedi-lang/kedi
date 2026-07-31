from __future__ import annotations

import asyncio
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from pydantic_ai.models.test import TestModel

import kedi
import kedi.api as kedi_api
import kedi.app as app_module
from kedi.agent_adapter import approval as approval_module
from kedi.agent_adapter import approval_telemetry
from kedi.agent_adapter import utils as adapter_utils
from kedi.agent_adapter.adapters import PydanticAdapter
from kedi.agent_adapter.adapters.lazy import LazyAdapter
from kedi.agent_adapter.approval import (
    ApprovalCancelled,
    ApprovalDecision,
    ApprovalPolicy,
    ApprovalRequest,
    ToolApprovalDenied,
    resolve_tool_approval,
    resolve_tool_approval_async,
)
from kedi.agent_adapter.cli_approval import CliApprovalHandler
from kedi.agent_adapter.tool_override import ToolSpec, with_approval
from kedi.agent_adapter.utils import (
    _procedure_tool_risk,
    kedi_procedure_tool_spec,
    python_callable_tool_spec,
)
from kedi.agent_profile import AgentProfile
from kedi.agent_profile.model import eval_approval_directive, eval_profile_definition
from kedi.agent_profile.registry import ProfileApprovalMember, ProfileDefinition
from kedi.core import KediRuntime
from kedi.errors import KediExecutionError
from kedi.lang import compile_program, parse_program
from kedi.lang.ast import ApprovalDirectiveStmt
from kedi.resolved_types import ResolvedKediType
from kedi.utils import AttrDict


@pytest.fixture(autouse=True)
def reset_python_api_config() -> Iterator[None]:
    kedi.reset_config()
    yield
    kedi.reset_config()


def _tool(*, risk: str = "mutating") -> ToolSpec:
    def write_file(*, path: str, content: str) -> str:
        return f"{path}:{content}"

    return ToolSpec(fn=write_file, name="write_file", risk=risk)  # type: ignore[arg-type]


def test_read_only_tools_auto_allow_without_policy() -> None:
    tool = with_approval(_tool(risk="read_only"), policy=None, adapter_shortname="test")

    assert tool.fn(path="notes.txt", content="ignored") == "notes.txt:ignored"


def test_approval_telemetry_records_states_without_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spans: list[object] = []

    class FakeSpan:
        def __init__(self, attributes: dict[str, object]) -> None:
            self.attributes = dict(attributes)
            self.events: list[str] = []

        def set_attribute(self, name: str, value: object) -> None:
            self.attributes[name] = value

        def add_event(self, name: str) -> None:
            self.events.append(name)

    @contextmanager
    def start_span(_name: str, *, attributes: dict[str, object]) -> Iterator[FakeSpan]:
        span = FakeSpan(attributes)
        spans.append(span)
        yield span

    monkeypatch.setattr(
        approval_telemetry._tracer,
        "start_as_current_span",
        start_span,
    )
    request = ApprovalRequest.create(
        tool_name="write_file",
        arguments={"path": ".env", "content": "secret"},
        risk="sensitive",
        adapter_shortname="pydantic",
        description=None,
        metadata=None,
    )

    decision = resolve_tool_approval(
        ApprovalPolicy.dynamic(lambda _request: ApprovalDecision.edit({"path": "safe.txt"})),
        request,
    )

    assert decision.decision == "edit"
    span = spans[0]
    assert isinstance(span, FakeSpan)
    assert span.attributes == {
        "approval.tool_name": "write_file",
        "approval.risk": "sensitive",
        "approval.state": "allowed",
        "approval.adapter": "pydantic",
        "approval.arguments_edited": True,
    }
    assert span.events == [
        "approval.requested",
        "approval.pending",
        "approval.allowed",
    ]
    assert "path" not in span.attributes
    assert "content" not in span.attributes


def test_approval_telemetry_records_cancellation(monkeypatch: pytest.MonkeyPatch) -> None:
    states: list[str] = []

    class FakeSpan:
        def set_attribute(self, name: str, value: object) -> None:
            if name == "approval.state":
                states.append(str(value))

        def add_event(self, _name: str) -> None:
            pass

    @contextmanager
    def start_span(_name: str, *, attributes: dict[str, object]) -> Iterator[FakeSpan]:
        states.append(str(attributes["approval.state"]))
        yield FakeSpan()

    monkeypatch.setattr(
        approval_telemetry._tracer,
        "start_as_current_span",
        start_span,
    )
    request = ApprovalRequest.create(
        tool_name="write_file",
        arguments={},
        risk="mutating",
        adapter_shortname=None,
        description=None,
        metadata=None,
    )

    def cancel(_request: ApprovalRequest) -> ApprovalDecision:
        raise ApprovalCancelled("cancelled")

    with pytest.raises(ApprovalCancelled, match="cancelled"):
        resolve_tool_approval(ApprovalPolicy.dynamic(cancel), request)

    assert states == ["requested", "pending", "cancelled"]


def test_mutating_tools_default_to_denied() -> None:
    tool = with_approval(_tool(), policy=None, adapter_shortname="test")

    with pytest.raises(ToolApprovalDenied, match="no approval policy configured"):
        tool.fn(path="notes.txt", content="blocked")


def test_cli_approval_handler_allows_once_and_aborts_cancelled_prompt() -> None:
    request = ApprovalRequest.create(
        tool_name="write_file",
        arguments={"path": "notes.txt"},
        risk="mutating",
        adapter_shortname="pydantic",
        description=None,
        metadata=None,
    )
    allowed = CliApprovalHandler(prompt=lambda _request: "allow", is_interactive=lambda: True)
    cancelled = CliApprovalHandler(prompt=lambda _request: None, is_interactive=lambda: True)

    assert allowed(request).decision == "allow"
    with pytest.raises(ApprovalCancelled, match="Approval cancelled"):
        cancelled(request)


def test_cli_approval_handler_allow_always_is_session_local_and_risk_scoped() -> None:
    calls: list[str] = []
    handler = CliApprovalHandler(
        prompt=lambda _request: calls.append("asked") or "always",
        is_interactive=lambda: True,
    )
    mutating = ApprovalRequest.create(
        tool_name="write_file",
        arguments={},
        risk="mutating",
        adapter_shortname="pydantic",
        description=None,
        metadata=None,
    )
    sensitive = ApprovalRequest.create(
        tool_name="write_file",
        arguments={},
        risk="sensitive",
        adapter_shortname="pydantic",
        description=None,
        metadata=None,
    )

    assert handler(mutating).decision == "allow"
    assert handler(mutating).decision == "allow"
    assert handler(sensitive).decision == "allow"
    assert calls == ["asked", "asked"]


def test_cli_approval_handler_accepts_async_prompt_callbacks() -> None:
    async def prompt(_request: ApprovalRequest) -> str:
        await asyncio.sleep(0)
        return "allow"

    handler = CliApprovalHandler(prompt=prompt, is_interactive=lambda: True)
    request = ApprovalRequest.create(
        tool_name="write_file",
        arguments={},
        risk="mutating",
        adapter_shortname="pydantic",
        description=None,
        metadata=None,
    )

    assert (
        asyncio.run(resolve_tool_approval_async(ApprovalPolicy.dynamic(handler), request)).decision
        == "allow"
    )
    with pytest.raises(RuntimeError, match="async CLI approval prompts require an async adapter"):
        handler(request)

    loop = asyncio.new_event_loop()
    try:
        future = loop.create_future()
        sync_handler = CliApprovalHandler(
            prompt=lambda _request: future,
            is_interactive=lambda: True,
        )
        with pytest.raises(
            RuntimeError, match="async CLI approval prompts require an async adapter"
        ):
            sync_handler(request)
    finally:
        loop.close()


def test_cli_approval_handler_refuses_noninteractive_requests() -> None:
    handler = CliApprovalHandler(is_interactive=lambda: False)
    request = ApprovalRequest.create(
        tool_name="write_file",
        arguments={},
        risk="mutating",
        adapter_shortname="pydantic",
        description=None,
        metadata=None,
    )

    decision = handler(request)

    assert decision.decision == "deny"
    assert decision.reason == "interactive CLI approval requires a TTY"


def test_cli_approval_handler_async_path_handles_always_and_noninteractive() -> None:
    request = ApprovalRequest.create(
        tool_name="write_file",
        arguments={},
        risk="mutating",
        adapter_shortname="pydantic",
        description=None,
        metadata=None,
    )
    handler = CliApprovalHandler(prompt=lambda _request: "always", is_interactive=lambda: True)

    assert asyncio.run(handler.resolve_async(request)).decision == "allow"
    assert asyncio.run(handler.resolve_async(request)).reason == "allowed for this CLI run"

    noninteractive = CliApprovalHandler(is_interactive=lambda: False)
    assert asyncio.run(noninteractive.resolve_async(request)).decision == "deny"


def test_cli_approval_handler_uses_questionary_choices(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    class FakePrompt:
        def ask(self) -> str:
            return "deny"

        async def ask_async(self) -> str:
            return "deny"

    class FakeQuestionary:
        @staticmethod
        def Choice(title: str, *, value: str) -> tuple[str, str]:
            return title, value

        @staticmethod
        def select(message: str, *, choices: list[tuple[str, str]]) -> FakePrompt:
            seen["message"] = message
            seen["choices"] = choices
            return FakePrompt()

    monkeypatch.setitem(sys.modules, "questionary", FakeQuestionary)
    handler = CliApprovalHandler(is_interactive=lambda: True)
    request = ApprovalRequest.create(
        tool_name="write_file",
        arguments={},
        risk="mutating",
        adapter_shortname="pydantic",
        description=None,
        metadata=None,
    )

    assert handler(request).decision == "deny"
    assert asyncio.run(handler.resolve_async(request)).decision == "deny"
    assert seen["message"] == "Allow mutating tool 'write_file'?"
    assert seen["choices"] == [
        ("Deny", "deny"),
        ("Allow once", "allow"),
        ("Allow always for this run", "always"),
    ]


def test_cli_approval_handler_maps_questionary_interrupts_to_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakePrompt:
        def ask(self) -> str:
            raise EOFError

        async def ask_async(self) -> str:
            raise KeyboardInterrupt

    class FakeQuestionary:
        @staticmethod
        def Choice(title: str, *, value: str) -> tuple[str, str]:
            return title, value

        @staticmethod
        def select(message: str, *, choices: object) -> FakePrompt:
            return FakePrompt()

    monkeypatch.setitem(sys.modules, "questionary", FakeQuestionary)
    handler = CliApprovalHandler(is_interactive=lambda: True)
    request = ApprovalRequest.create(
        tool_name="write_file",
        arguments={},
        risk="mutating",
        adapter_shortname="pydantic",
        description=None,
        metadata=None,
    )

    with pytest.raises(ApprovalCancelled, match="write_file"):
        handler(request)
    with pytest.raises(ApprovalCancelled, match="write_file"):
        asyncio.run(handler.resolve_async(request))


def test_async_approval_resolution_records_cancellation() -> None:
    async def cancel(_request: ApprovalRequest) -> ApprovalDecision:
        raise ApprovalCancelled("async cancelled")

    request = ApprovalRequest.create(
        tool_name="write_file",
        arguments={},
        risk="mutating",
        adapter_shortname="pydantic",
        description=None,
        metadata=None,
    )
    with pytest.raises(ApprovalCancelled, match="async cancelled"):
        asyncio.run(resolve_tool_approval_async(ApprovalPolicy.dynamic(cancel), request))


def test_run_source_cli_supplies_interactive_default_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    class Runtime:
        def set_initial_globals(self, _globals: object) -> None:
            return None

        def run_main(self) -> None:
            return None

    def fake_compile(_program: object, **kwargs: object) -> Runtime:
        seen.update(kwargs)
        return Runtime()

    monkeypatch.setattr(app_module, "compile_program", fake_compile)

    assert app_module.run_source_cli("= done", adapter="pydantic", model=None) == (0, "")
    profile = seen["default_agent_profile"]
    assert isinstance(profile, AgentProfile)
    assert isinstance(profile.approval, ApprovalPolicy)
    assert isinstance(profile.approval.handler, CliApprovalHandler)
    adapter = seen["adapter"]
    assert isinstance(adapter, LazyAdapter)
    assert adapter.capabilities.supports("subagents")


def test_run_source_cli_denies_risky_tools_without_an_explicit_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = """@write_file(path: str) -> str:
  = <path>

> use: write_file
>> Write [answer]
= <answer>
"""
    monkeypatch.setattr(
        app_module,
        "_build_runtime_adapter",
        lambda _adapter, _model: _ToolCallingAdapter(),
    )

    with pytest.raises(ToolApprovalDenied, match="interactive CLI approval requires a TTY"):
        app_module.run_source_cli(source, adapter="dspy", model=None)


def test_run_source_cli_explicit_allow_overrides_the_cli_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = """@write_file(path: str) -> str:
  = <path>

> use: write_file
> approval: allow
>> Write [answer]
= <answer>
"""
    monkeypatch.setattr(
        app_module,
        "_build_runtime_adapter",
        lambda _adapter, _model: _ToolCallingAdapter(),
    )

    assert app_module.run_source_cli(source, adapter="dspy", model=None) == (0, "safe.txt")


def test_compiler_default_profile_reaches_top_level_and_procedure_tool_approval() -> None:
    calls: list[str] = []

    def approve(request: ApprovalRequest) -> ApprovalDecision:
        calls.append(request.tool_name)
        return ApprovalDecision.allow()

    source = """@write_file(path: str) -> str:
  = <path>

@run() -> str:
  > use: write_file
  >> Write [answer]
  = <answer>

= <run()>
"""
    runtime = compile_program(
        parse_program(source),
        adapter=_ToolCallingAdapter(),
        default_agent_profile=AgentProfile(approval=ApprovalPolicy.dynamic(approve)),
    )

    assert runtime.run_main() == "safe.txt"
    assert calls == ["write_file"]


def test_static_allow_and_deny_policies() -> None:
    allowed = with_approval(_tool(), policy=ApprovalPolicy.allow(), adapter_shortname="test")
    denied = with_approval(_tool(), policy=ApprovalPolicy.deny(), adapter_shortname="test")

    assert allowed.fn(path="notes.txt", content="ok") == "notes.txt:ok"
    with pytest.raises(ToolApprovalDenied, match="static deny policy"):
        denied.fn(path="notes.txt", content="blocked")


def test_dynamic_handler_can_edit_arguments_and_receives_immutable_request() -> None:
    seen = []

    @kedi.approval
    def redirect(request):  # type: ignore[no-untyped-def]
        seen.append(request)
        with pytest.raises(TypeError):
            request.arguments["path"] = "other.txt"
        return ApprovalDecision.edit({"path": "safe.txt", "content": request.arguments["content"]})

    tool = with_approval(_tool(), policy=ApprovalPolicy.dynamic(redirect), adapter_shortname="test")

    assert tool.fn(path="unsafe.txt", content="ok") == "safe.txt:ok"
    assert seen[0].tool_name == "write_file"
    assert seen[0].risk == "mutating"


def test_dynamic_handler_may_be_async() -> None:
    async def scenario() -> None:
        @kedi.approval
        async def permit(_request: object) -> ApprovalDecision:
            await asyncio.sleep(0)
            return ApprovalDecision.allow()

        tool = with_approval(
            _tool(), policy=ApprovalPolicy.dynamic(permit), adapter_shortname="test"
        )

        assert await tool.fn(path="notes.txt", content="ok") == "notes.txt:ok"

    asyncio.run(scenario())


def test_async_handler_stays_on_the_callers_event_loop() -> None:
    async def scenario() -> None:
        lock = asyncio.Lock()
        await lock.acquire()

        async def permit(_request: object) -> ApprovalDecision:
            async with lock:
                return ApprovalDecision.allow()

        tool = with_approval(
            _tool(), policy=ApprovalPolicy.dynamic(permit), adapter_shortname="test"
        )
        release = asyncio.create_task(_release_lock(lock))
        try:
            assert await tool.fn(path="notes.txt", content="ok") == "notes.txt:ok"
        finally:
            await release

    asyncio.run(scenario())


async def _release_lock(lock: asyncio.Lock) -> None:
    await asyncio.sleep(0)
    lock.release()


def test_approval_decorator_registers_the_default_python_api_policy() -> None:
    @kedi.approval
    def permit(_request: object) -> ApprovalDecision:
        return ApprovalDecision.allow()

    policy = kedi_api._config_var.get().profile.approval

    assert policy is not None
    assert policy.handler is permit


def test_argument_aware_tool_risk_requires_approval_for_sensitive_access() -> None:
    tool = ToolSpec(
        fn=lambda *, secret_files=False: "secret",  # noqa: E731
        name="read_text_file",
        risk="read_only",
        risk_resolver=lambda arguments: "sensitive" if arguments["secret_files"] else "read_only",
    )
    guarded = with_approval(tool, policy=None, adapter_shortname="test")

    assert guarded.fn(secret_files=False) == "secret"
    with pytest.raises(ToolApprovalDenied, match="no approval policy configured"):
        guarded.fn(secret_files=True)


def test_risk_resolver_can_only_elevate_static_risk() -> None:
    tool = ToolSpec(
        fn=lambda: "secret",  # noqa: E731
        name="sensitive_tool",
        risk="sensitive",
        risk_resolver=lambda _arguments: "read_only",
    )

    guarded = with_approval(tool, policy=None, adapter_shortname="test")
    with pytest.raises(ToolApprovalDenied, match="no approval policy configured"):
        guarded.fn()


def test_secret_file_writes_require_approval_before_mutation() -> None:
    writes: list[str] = []

    def write_text_file(*, file_path: str, content: str) -> str:
        writes.append(file_path)
        return file_path

    def policy(request: object) -> ApprovalDecision:
        risk = getattr(request, "risk")
        return ApprovalDecision.deny() if risk == "sensitive" else ApprovalDecision.allow()

    tool = ToolSpec(
        fn=write_text_file,
        name="write_text_file",
        risk="mutating",
        risk_resolver=lambda arguments: (
            "sensitive" if Path(str(arguments["file_path"])).name.startswith(".env") else "mutating"
        ),
    )
    guarded = with_approval(tool, policy=ApprovalPolicy.dynamic(policy), adapter_shortname="test")

    assert guarded.fn(file_path="notes.txt", content="ok") == "notes.txt"
    with pytest.raises(ToolApprovalDenied):
        guarded.fn(file_path=".env.local", content="TOKEN=hidden")
    assert writes == ["notes.txt"]


def test_edited_python_tool_arguments_are_revalidated() -> None:
    calls: list[int] = []

    def typed_tool(*, count: int = 1) -> int:
        calls.append(count)
        return count

    def valid_edit(_request: object) -> ApprovalDecision:
        return ApprovalDecision.edit({"count": 2})

    guarded = with_approval(
        python_callable_tool_spec("typed_tool", typed_tool),
        policy=ApprovalPolicy.dynamic(valid_edit),
        adapter_shortname="test",
    )
    assert guarded.fn(count=0) == 2
    assert calls == [2]

    def invalid_edit(_request: object) -> ApprovalDecision:
        return ApprovalDecision.edit({"count": "not-an-int"})

    blocked = with_approval(
        python_callable_tool_spec("typed_tool", typed_tool),
        policy=ApprovalPolicy.dynamic(invalid_edit),
        adapter_shortname="test",
    )
    with pytest.raises(Exception, match="valid integer"):
        blocked.fn(count=0)
    assert calls == [2]


def test_edited_kedi_tool_arguments_are_revalidated_against_custom_types() -> None:
    program = parse_program("~Count(value: int)\n@take(count: Count) -> int:\n  = `count.value`\n")
    runtime = compile_program(program)
    spec = kedi_procedure_tool_spec(runtime, program.procedures[0])

    guarded = with_approval(
        spec,
        policy=ApprovalPolicy.dynamic(
            lambda _request: ApprovalDecision.edit({"count": {"value": "not-an-int"}})
        ),
        adapter_shortname="test",
    )

    with pytest.raises(TypeError, match="does not match annotated type"):
        guarded.fn(count={"value": 1})


def test_user_owned_filesystem_name_is_not_treated_as_builtin(tmp_path: Path) -> None:
    program = parse_program(
        "@read_text_file(file_path: str) -> str:\n  = user_owned\n",
        source_path=str(tmp_path / "filesystem.kedi"),
    )

    assert _procedure_tool_risk(program.procedures[0]) == "mutating"


def test_filesystem_secret_read_requires_approval(tmp_path: Path) -> None:
    filesystem_path = Path(kedi.__file__).parent / "modules" / "filesystem.kedi"
    source = filesystem_path.read_text(encoding="utf-8").replace(
        "_ROOT = Path.cwd().resolve()",
        f"_ROOT = Path({str(tmp_path)!r}).resolve()",
    )
    program = parse_program(
        source,
        source_path=str(filesystem_path),
    )
    runtime = compile_program(program)
    read_proc = next(proc for proc in program.procedures if proc.name == "read_text_file")
    patch_proc = next(proc for proc in program.procedures if proc.name == "apply_patch")
    spec = kedi_procedure_tool_spec(runtime, read_proc)
    secret_path = tmp_path / ".env"
    secret_path.write_text("TOKEN=hidden\n", encoding="utf-8")

    blocked = with_approval(spec, policy=None, adapter_shortname="test")
    with pytest.raises(ToolApprovalDenied, match="no approval policy configured"):
        blocked.fn(file_path=str(secret_path), secret_files=True)

    allowed = with_approval(spec, policy=ApprovalPolicy.allow(), adapter_shortname="test")
    streamed = allowed.fn(file_path=str(secret_path), secret_files=True)
    assert isinstance(streamed, kedi.ArtifactStream)
    assert list(streamed.iter_chunks()) == ["TOKEN=hidden\n"]
    with pytest.raises(KediExecutionError, match="secret_files=True"):
        allowed.fn(file_path=str(secret_path), secret_files=False)

    patch_spec = kedi_procedure_tool_spec(runtime, patch_proc)
    protected_patch = with_approval(patch_spec, policy=None, adapter_shortname="test")
    with pytest.raises(ToolApprovalDenied, match="no approval policy configured"):
        protected_patch.fn(
            operation="update",
            file_path=str(secret_path),
            old_text="TOKEN=hidden",
            new_text="TOKEN=changed",
        )
    assert secret_path.read_text(encoding="utf-8") == "TOKEN=hidden\n"


def test_non_edit_decisions_reject_edited_arguments() -> None:
    with pytest.raises(ValueError, match="only be provided"):
        ApprovalDecision(decision="allow", edit_arguments={"path": "notes.txt"})

    with pytest.raises(ValueError, match="requires edit_arguments"):
        ApprovalDecision(decision="edit")

    with pytest.raises(ValueError, match="must be 'allow', 'deny', or 'edit'"):
        ApprovalDecision(decision="unsupported")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="must be callable"):
        ApprovalPolicy.dynamic("not-a-handler")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="mutually exclusive"):
        ApprovalPolicy(mode="allow", handler=lambda _request: ApprovalDecision.allow())

    with pytest.raises(ValueError, match="must be 'allow' or 'deny'"):
        ApprovalPolicy(mode="edit")  # type: ignore[arg-type]


def test_approval_handler_must_return_a_decision() -> None:
    tool = with_approval(
        _tool(),
        policy=ApprovalPolicy.dynamic(lambda _request: "allow"),  # type: ignore[arg-type]
        adapter_shortname="test",
    )

    with pytest.raises(ValueError, match="must return ApprovalDecision"):
        tool.fn(path="notes.txt", content="blocked")


def test_tool_wrapper_rejects_positional_arguments_and_invalid_risk() -> None:
    guarded = with_approval(_tool(), policy=ApprovalPolicy.allow(), adapter_shortname="test")
    with pytest.raises(TypeError, match="only accepts keyword"):
        guarded.fn("notes.txt", "blocked")

    invalid_risk = ToolSpec(
        fn=lambda **_kwargs: None,
        name="invalid_risk",
        risk_resolver=lambda _arguments: "unknown",  # type: ignore[return-value]
    )
    with pytest.raises(ValueError, match="invalid Kedi tool risk"):
        with_approval(invalid_risk, policy=ApprovalPolicy.allow(), adapter_shortname="test").fn()


def test_sync_and_async_approval_resolvers_await_dynamic_handlers() -> None:
    request = ApprovalRequest.create(
        tool_name="write_file",
        arguments={"path": "notes.txt"},
        risk="mutating",
        adapter_shortname="test",
        description=None,
        metadata=None,
    )

    async def permit(_request: ApprovalRequest) -> ApprovalDecision:
        await asyncio.sleep(0)
        return ApprovalDecision.allow()

    policy = ApprovalPolicy.dynamic(permit)
    assert resolve_tool_approval(policy, request).decision == "allow"
    assert asyncio.run(resolve_tool_approval_async(policy, request)).decision == "allow"
    assert approval_module._resolve_awaitable is not None

    async def reject_sync_resolution_inside_an_event_loop() -> None:
        with pytest.raises(
            RuntimeError, match="async approval handlers require an async tool wrapper"
        ):
            resolve_tool_approval(policy, request)

    asyncio.run(reject_sync_resolution_inside_an_event_loop())

    class _CustomAwaitable:
        def __await__(self):
            if False:
                yield None
            return ApprovalDecision.allow()

    def non_coroutine_awaitable(_request: ApprovalRequest) -> _CustomAwaitable:
        return _CustomAwaitable()

    async def reject_non_coroutine_awaitable_inside_an_event_loop() -> None:
        with pytest.raises(
            RuntimeError, match="async approval handlers require an async tool wrapper"
        ):
            resolve_tool_approval(ApprovalPolicy.dynamic(non_coroutine_awaitable), request)

    asyncio.run(reject_non_coroutine_awaitable_inside_an_event_loop())


def test_async_approval_wrapper_rejects_positional_arguments() -> None:
    async def write_file(*, path: str, content: str) -> str:
        return f"{path}:{content}"

    guarded = with_approval(
        ToolSpec(fn=write_file, name="write_file"),
        policy=ApprovalPolicy.allow(),
        adapter_shortname="test",
    )

    with pytest.raises(TypeError, match="only accepts keyword"):
        asyncio.run(guarded.fn("notes.txt", "blocked"))


def test_python_callable_validator_preserves_untyped_variadic_keyword_arguments() -> None:
    def collect(value, *extra, **metadata):  # type: ignore[no-untyped-def]
        return value, extra, metadata

    spec = python_callable_tool_spec("collect", collect)
    assert spec.argument_validator is not None
    assert spec.argument_validator({"value": "ok", "source": "test"}) == {
        "value": "ok",
        "source": "test",
    }


def test_builtin_filesystem_identity_fails_closed_when_paths_cannot_resolve(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    program = parse_program(
        "@read_text_file(file_path: str) -> str:\n  = user_owned\n",
        source_path=str(tmp_path / "filesystem.kedi"),
    )

    def raise_os_error(_self: Path, *args: object, **kwargs: object) -> Path:
        raise OSError("cannot resolve")

    monkeypatch.setattr(adapter_utils.Path, "resolve", raise_os_error)
    assert not adapter_utils._is_builtin_filesystem_procedure(program.procedures[0])


def test_python_tool_risk_and_api_approval_inputs_are_validated() -> None:
    def unsafe() -> None:
        return None

    unsafe.__kedi_tool_risk__ = "unknown"  # type: ignore[attr-defined]
    with pytest.raises(ValueError, match="invalid Kedi tool risk"):
        python_callable_tool_spec("unsafe", unsafe)

    assert kedi_api._coerce_approval("allow") == ApprovalPolicy.allow()
    assert kedi_api._coerce_approval("deny") == ApprovalPolicy.deny()
    policy = ApprovalPolicy.allow()
    assert kedi_api._coerce_approval(policy) is policy
    assert kedi_api._approval_identity(policy) == ("allow", None)
    assert kedi_api._approval_identity(None) is None

    def permit(_request: object) -> ApprovalDecision:
        return ApprovalDecision.allow()

    dynamic_policy = kedi_api._coerce_approval(permit)
    assert dynamic_policy is not None
    assert dynamic_policy.handler is permit
    assert kedi_api._approval_identity(dynamic_policy)[:3] == (
        None,
        permit.__module__,
        permit.__qualname__,
    )
    with pytest.raises(TypeError, match="approval must be"):
        kedi_api._coerce_approval("edit")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="approval must be"):
        kedi_api._coerce_approval(1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="tool risk"):
        kedi.tool(risk="unknown")  # type: ignore[arg-type]


def test_approval_directive_and_profile_validation_errors() -> None:
    runtime = KediRuntime()
    with pytest.raises(TypeError, match="must evaluate to a callable"):
        eval_approval_directive(
            runtime,
            ApprovalDirectiveStmt(backtick_code="handler"),
            {"handler": "not callable"},
        )
    with pytest.raises(ValueError, match="has no value"):
        eval_approval_directive(runtime, ApprovalDirectiveStmt(), {})
    with pytest.raises(ValueError, match="must be 'allow', 'deny'"):
        eval_approval_directive(runtime, ApprovalDirectiveStmt(plain_value="edit"), {})

    def handler(_request: object) -> ApprovalDecision:
        return ApprovalDecision.allow()

    dynamic = eval_profile_definition(
        runtime,
        ProfileDefinition(
            name="dynamic",
            members=[ProfileApprovalMember(backtick_code="handler")],
        ),
        {"handler": handler},
    )
    assert dynamic.approval == ApprovalPolicy.dynamic(handler)
    denied = eval_profile_definition(
        runtime,
        ProfileDefinition(name="denied", members=[ProfileApprovalMember(plain_value="deny")]),
        {},
    )
    assert denied.approval == ApprovalPolicy.deny()
    with pytest.raises(TypeError, match="must evaluate to a callable"):
        eval_profile_definition(
            runtime,
            ProfileDefinition(
                name="not_callable",
                members=[ProfileApprovalMember(backtick_code="handler")],
            ),
            {"handler": 1},
        )
    with pytest.raises(ValueError, match="must be 'allow' or 'deny'"):
        eval_profile_definition(
            runtime,
            ProfileDefinition(name="invalid", members=[ProfileApprovalMember()]),
            {},
        )


class _ToolCallingAdapter:
    kind = "agent-framework"
    shortname = "test"

    def __init__(self) -> None:
        self._tools: list[ToolSpec] = []

    @contextmanager
    def profile_override(self, _profile: AgentProfile) -> Iterator[None]:
        yield

    def tool_scope(self, _tools=()):  # type: ignore[no-untyped-def]
        adapter = self

        class Scope:
            def __enter__(self):  # type: ignore[no-untyped-def]
                adapter._tools = []

            def __exit__(self, *_args):  # type: ignore[no-untyped-def]
                return None

        return Scope()

    def register_tool(self, tool: ToolSpec) -> None:
        self._tools.append(tool)

    async def produce(self, **kwargs):  # type: ignore[no-untyped-def]
        return self.produce_sync(**kwargs)

    def produce_sync(
        self,
        *,
        template: str,
        output_schema: dict[str, ResolvedKediType],
        **_kwargs: object,
    ) -> AttrDict[str, str]:
        assert template
        assert output_schema == {"answer": str}
        return AttrDict({"answer": self._tools[0].fn(path="safe.txt")})


def test_dsl_static_approval_reaches_the_registered_tool() -> None:
    source = """@write_file(path: str) -> str:
  = <path>

> artifacts:
  enabled: false
> use: write_file
> approval: allow
>> Write [answer]
= <answer>
"""
    runtime = compile_program(parse_program(source), adapter=_ToolCallingAdapter())

    assert runtime.run_main() == "safe.txt"


@pytest.mark.parametrize(
    "source",
    [
        "> model: same-model\n> approval: allow\n>> Return [answer].\n= <answer>\n",
        "> model: same-model\n> approval: allow\n[answer] << Return ok.\n= <answer>\n",
    ],
)
def test_runtime_enters_approval_profile_without_a_model_override(source: str) -> None:
    class ProfileAwareAdapter:
        def __init__(self) -> None:
            self.active_profile = AgentProfile.empty()
            self.entered_profiles: list[AgentProfile] = []

        def default_model(self) -> str:
            return "same-model"

        @contextmanager
        def profile_override(self, profile: AgentProfile) -> Iterator[None]:
            previous = self.active_profile
            self.active_profile = profile
            self.entered_profiles.append(profile)
            try:
                yield
            finally:
                self.active_profile = previous

        def produce_sync(self, **_kwargs: object) -> AttrDict[str, str]:
            assert self.active_profile.approval == ApprovalPolicy.allow()
            return AttrDict({"answer": "ok"})

        def invoke_sync(self, **_kwargs: object) -> str:
            assert self.active_profile.approval == ApprovalPolicy.allow()
            return "ok"

    adapter = ProfileAwareAdapter()
    runtime = compile_program(parse_program(source), adapter=adapter)

    assert runtime.run_main() == "ok"
    assert len(adapter.entered_profiles) == 1
    assert adapter.active_profile == AgentProfile.empty()


def test_static_approval_reaches_pydantic_native_handler_without_an_override() -> None:
    source = """@write_status(name: str) -> str:
  = saved-<name>

> use: write_status
> approval: allow

[response] << Call write_status.
= <response>
"""
    runtime = compile_program(
        parse_program(source),
        adapter=PydanticAdapter(TestModel(call_tools=["write_status"])),
    )

    assert runtime.run_main() == '{"write_status":"saved-a"}'


def test_dsl_static_deny_prevents_the_registered_tool() -> None:
    source = """@write_file(path: str) -> str:
  = <path>

> artifacts:
  enabled: false
> use: write_file
> approval: deny
>> Write [answer]
= <answer>
"""
    runtime = compile_program(parse_program(source), adapter=_ToolCallingAdapter())

    with pytest.raises(ToolApprovalDenied, match="static deny policy"):
        runtime.run_main()


def test_dsl_dynamic_approval_handler_is_evaluated_from_python_environment() -> None:
    seen: list[str] = []

    @kedi.approval
    def approve(request):  # type: ignore[no-untyped-def]
        seen.append(request.tool_name)
        return ApprovalDecision.allow(reason="test handler")

    source = """@write_file(path: str) -> str:
  = <path>

> artifacts:
  enabled: false
> use: write_file
> approval: `approve`
>> Write [answer]
= <answer>
"""
    runtime = compile_program(
        parse_program(source),
        adapter=_ToolCallingAdapter(),
        runtime_globals={"approve": approve},
    )

    assert runtime.run_main() == "safe.txt"
    assert seen == ["write_file"]


def test_dsl_dynamic_approval_handler_is_available_from_the_prelude() -> None:
    source = """```
from kedi import ApprovalDecision

def approval_handler(_request):
    return ApprovalDecision.allow()
```

@write_file(path: str) -> str:
  = <path>

> artifacts:
  enabled: false
> use: write_file
> approval: `approval_handler`
>> Write [answer]
= <answer>
"""
    runtime = compile_program(parse_program(source), adapter=_ToolCallingAdapter())

    assert runtime.run_main() == "safe.txt"


def test_procedure_approval_directive_is_applied_in_its_call_scope() -> None:
    source = """@write_file(path: str) -> str:
  = <path>

@run() -> str:
  > artifacts:
    enabled: false
  > use: write_file
  > approval: allow
  >> Write [answer]
  = <answer>

= <run()>
"""
    runtime = compile_program(parse_program(source), adapter=_ToolCallingAdapter())

    assert runtime.run_main() == "safe.txt"


def test_approval_edit_is_not_a_valid_static_dsl_mode() -> None:
    with pytest.raises(Exception, match="only valid as ApprovalDecision.edit"):
        parse_program("> approval: edit\n")


def test_profile_approval_is_applied_when_profile_is_used() -> None:
    source = """@write_file(path: str) -> str:
  = <path>

> profile: guarded:
  > artifacts:
    enabled: false
  > approval: allow
  > use: write_file

> use: guarded
>> Write [answer]
= <answer>
"""
    runtime = compile_program(parse_program(source), adapter=_ToolCallingAdapter())

    assert runtime.run_main() == "safe.txt"
