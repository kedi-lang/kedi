from __future__ import annotations

import asyncio
from collections.abc import Iterator, Mapping
from contextlib import AsyncExitStack
from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic_ai import ModelRetry
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.tools import ToolDefinition

import kedi
import kedi.agent_profile.directives as profile_model
import kedi.api.cache as api_cache_module
from kedi import (
    HookDeniedError,
    HookExecutionError,
    HookTimeoutError,
    PostToolUseEvent,
    PostToolUseFailureEvent,
    PreToolUseDecision,
    PreToolUseRequest,
    UserPromptSubmitDecision,
    UserPromptSubmitRequest,
)
from kedi.agent_adapter.adapters import PydanticAdapter
from kedi.agent_adapter.adapters.acp import ACPAdapter
from kedi.agent_adapter.adapters.claude.adapter import ClaudeAdapter
from kedi.agent_adapter.adapters.claude.hooks import (
    _claude_post_tool_use_failure_hook,
    _claude_post_tool_use_hook,
    _claude_pre_tool_use_hook,
    _with_kedi_approval_hook,
)
from kedi.agent_adapter.adapters.claude.messages import (
    _active_claude_hook_identities,
    _claude_session_fingerprint,
)
from kedi.agent_adapter.adapters.claude.tools import _claude_tool_call_input
from kedi.agent_adapter.adapters.codex.adapter import CodexAdapter
from kedi.agent_adapter.adapters.langchain.adapter import LangChainAdapter
from kedi.agent_adapter.adapters.langchain.middleware import (
    _langchain_approval_middleware,
    _langchain_hook_origin,
)
from kedi.agent_adapter.adapters.langchain.tools import _validated_tool_arguments
from kedi.agent_adapter.adapters.lazy import LazyAdapter
from kedi.agent_adapter.adapters.pydantic.hooks import _KediHookCapability
from kedi.agent_adapter.adapters.pydantic.tools import pydantic_toolset_from_specs
from kedi.agent_adapter.adapters.webgpu import WebGPUAdapter
from kedi.agent_adapter.approvals.policy import (
    ApprovalDecision,
    ApprovalPolicy,
    ApprovalRequest,
    ToolApprovalDenied,
)
from kedi.agent_adapter.conversation import AdapterConversationContext
from kedi.agent_adapter.events import agent_event_run, bind_agent_run, observe_agent_events
from kedi.agent_adapter.hooks import (
    AdapterHookSnapshot,
    HookDispatcher,
    HookDispatcherSnapshot,
    HookRegistrationMixin,
    bind_hook_dispatcher,
    dispatch_post_tool_use,
    dispatch_post_tool_use_failure,
    dispatch_post_tool_use_failure_sync,
    dispatch_post_tool_use_sync,
    dispatch_pre_tool_use,
    dispatch_pre_tool_use_sync,
    hooked_agent_run,
    is_tool_execution_interrupted,
    native_agent_hook_run,
    with_hook_lifecycle,
)
from kedi.agent_adapter.hooks import boundary as hook_boundary
from kedi.agent_adapter.hooks import dispatcher as hook_dispatcher
from kedi.agent_adapter.hooks import tool_wrapper as hook_tool_wrapper
from kedi.agent_adapter.hooks.telemetry import HookChainTelemetry
from kedi.agent_adapter.tools.execution import with_artifact_result
from kedi.agent_adapter.tools.specs import ToolSpec
from kedi.agent_profile import (
    AgentProfile,
    AgentProfileOverrideUnsupported,
    HookSettings,
    eval_hooks_directive,
    normalize_hook_settings,
)
from kedi.agent_profile.compile import build_capture_env, profile_stmt_to_definition
from kedi.agent_profile.evaluation import eval_profile_definition
from kedi.artifacts import ArtifactManager, ArtifactPolicy
from kedi.core.runtime import KediRuntime
from kedi.lang import compile_program
from kedi.lang.ast import HooksDirectiveStmt, ProfileDefStmt, SettingsField
from kedi.lang.parser import KediParseError, parse_program
from tests.telemetry_helpers import installed_recording_backend


@pytest.fixture(autouse=True)
def _reset_python_api_config() -> Iterator[None]:
    kedi.reset_config()
    yield
    kedi.reset_config()


class _Adapter(HookRegistrationMixin):
    def __init__(self, hook_handler: Any = None) -> None:
        self._init_hook_registration(hook_handler)


def _prompt(content: str = "original") -> UserPromptSubmitRequest:
    return UserPromptSubmitRequest(
        run_id="run",
        sequence=0,
        adapter_shortname="fake",
        content=content,
    )


def _pre(arguments: dict[str, Any] | None = None) -> PreToolUseRequest:
    return PreToolUseRequest(
        run_id="run",
        sequence=0,
        adapter_shortname="fake",
        tool_call_id="call",
        tool_name="write_file",
        origin="kedi",
        arguments=arguments or {"path": "before"},
    )


def test_adapter_on_accepts_one_or_multiple_event_names() -> None:
    observed: list[str] = []
    adapter = _Adapter()

    @adapter.on(("user_prompt_submit", "post_tool_use_failure"))
    def observe(event: Any) -> None:
        observed.append(event.event)

    snapshot = adapter.hook_snapshot()
    assert snapshot.handlers_for("user_prompt_submit") == (observe,)
    assert snapshot.handlers_for("post_tool_use_failure") == (observe,)
    assert snapshot.handlers_for("pre_tool_use") == ()

    with pytest.raises(ValueError, match="Unknown Kedi hook event"):
        adapter.on("PreToolUse")
    with pytest.raises(ValueError, match="At least one"):
        adapter.on(())


def test_hook_registration_validates_handlers_and_transfers_snapshots() -> None:
    with pytest.raises(TypeError, match="hook_handler must be callable"):
        _Adapter(42)

    source = _Adapter()

    @source.on(["user_prompt_submit", "pre_tool_use"])
    def first(_event: Any) -> None:
        return None

    target = _Adapter()
    with pytest.raises(TypeError, match="handler must be callable"):
        target.on("pre_tool_use")(cast(Any, 42))

    target._replace_hook_snapshot(source.hook_snapshot())
    assert target.hook_snapshot().handlers_for("user_prompt_submit") == (first,)
    assert target.hook_snapshot().handlers_for("pre_tool_use") == (first,)


def test_hook_settings_validate_mapping_inputs_and_enforcement_edges() -> None:
    settings = HookSettings()
    with pytest.raises(TypeError, match="must be callable"):
        settings.append("pre_tool_use", cast(Any, 42))
    assert settings.compose_enforced(None) is settings

    with pytest.raises(TypeError, match="event-to-handler mapping"):
        normalize_hook_settings(cast(Any, 42))
    with pytest.raises(ValueError, match="Unknown Kedi hook event"):
        normalize_hook_settings({"PreToolUse": lambda _event: None})
    with pytest.raises(TypeError, match="Every hook handler"):
        normalize_hook_settings({"pre_tool_use": [lambda _event: None, 42]})
    with pytest.raises(TypeError, match="callable or a sequence"):
        normalize_hook_settings({"pre_tool_use": "handler"})

    cleared = normalize_hook_settings({"pre_tool_use": []})
    assert cleared is not None
    assert cleared.pre_tool_use == ()


def test_adapter_hook_capability_matrix_matches_declared_bridges() -> None:
    for adapter_type in (
        PydanticAdapter,
        LangChainAdapter,
        ClaudeAdapter,
        CodexAdapter,
        WebGPUAdapter,
    ):
        hooks = adapter_type.capabilities.hooks
        assert hooks is not None
        assert hooks.user_prompt_submit is not None
        assert hooks.user_prompt_submit.supports_mutation is True
        assert hooks.pre_tool_use is not None
        assert hooks.pre_tool_use.supports_mutation is True
        assert hooks.post_tool_use is not None
        assert hooks.post_tool_use.supports_mutation is False

    assert ACPAdapter.capabilities.hooks is not None
    assert ACPAdapter.capabilities.hooks.user_prompt_submit is not None
    assert ACPAdapter.capabilities.hooks.pre_tool_use is None
    codex_hooks = CodexAdapter.capabilities.hooks
    assert codex_hooks is not None
    codex_pre = codex_hooks.pre_tool_use
    assert codex_pre is not None
    assert "provider_builtin" not in codex_pre.origins


def test_common_tool_wrapper_distinguishes_denial_from_interrupted_execution() -> None:
    observed: list[Any] = []

    def observe(event: Any) -> Any:
        observed.append(event)
        if isinstance(event, PreToolUseRequest) and event.arguments.get("deny") is True:
            return PreToolUseDecision.deny("blocked")
        return None

    async def cancel(*, deny: bool) -> None:
        del deny
        raise asyncio.CancelledError

    tool = with_hook_lifecycle(
        ToolSpec(fn=cancel, name="cancel", risk="read_only"),
        adapter_shortname="fake",
    )
    dispatcher = HookDispatcher(
        HookDispatcherSnapshot.compose(
            AdapterHookSnapshot.empty(),
            HookSettings(
                pre_tool_use=(observe,),
                post_tool_use_failure=(observe,),
            ),
        )
    )

    async def exercise() -> None:
        with bind_agent_run("run"), bind_hook_dispatcher(dispatcher):
            with pytest.raises(HookDeniedError, match="blocked"):
                await tool.fn(deny=True)
            with pytest.raises(asyncio.CancelledError):
                await tool.fn(deny=False)

    asyncio.run(exercise())
    assert [event.event for event in observed] == [
        "pre_tool_use",
        "pre_tool_use",
        "post_tool_use_failure",
    ]
    failure = observed[-1]
    assert isinstance(failure, PostToolUseFailureEvent)
    assert failure.duration_seconds is not None
    assert failure.duration_seconds >= 0
    assert failure.interrupted is True


def test_common_tool_wrapper_observes_artifact_reference_after_admission() -> None:
    observed: list[Any] = []
    manager = ArtifactManager()
    artifact_tool = with_artifact_result(
        ToolSpec(fn=lambda: "large-result", name="lookup", risk="read_only"),
        manager=manager,
        policy=ArtifactPolicy(threshold_bytes=0),
    )
    tool = with_hook_lifecycle(artifact_tool, adapter_shortname="fake")
    dispatcher = HookDispatcher(
        HookDispatcherSnapshot.compose(
            AdapterHookSnapshot.empty(),
            HookSettings(post_tool_use=(observed.append,)),
        )
    )

    try:
        with bind_agent_run("run"), bind_hook_dispatcher(dispatcher):
            result = tool.fn()
        assert result["ref_id"] == "tool_call_result_1"
        assert len(observed) == 1
        assert isinstance(observed[0], PostToolUseEvent)
        assert observed[0].result == result
        assert manager.get_value(result["ref_id"]) == "large-result"
    finally:
        manager.close()


def test_claude_session_fingerprint_changes_with_stable_hook_identity() -> None:
    def first(_event: Any) -> None:
        return None

    def second(_event: Any) -> None:
        return None

    base = _claude_session_fingerprint({}, tools=())
    first_fingerprint = _claude_session_fingerprint(
        {},
        tools=(),
        hook_identities={"pre_tool_use": (f"{first.__module__}.{first.__qualname__}",)},
    )
    same_first = _claude_session_fingerprint(
        {},
        tools=(),
        hook_identities={"pre_tool_use": (f"{first.__module__}.{first.__qualname__}",)},
    )
    second_fingerprint = _claude_session_fingerprint(
        {},
        tools=(),
        hook_identities={"pre_tool_use": (f"{second.__module__}.{second.__qualname__}",)},
    )

    assert base != first_fingerprint
    assert first_fingerprint == same_first
    assert first_fingerprint != second_fingerprint


def test_runtime_rejects_hook_profile_for_unsupported_adapter() -> None:
    adapter = SimpleNamespace(shortname="custom", capabilities=None)
    profile = AgentProfile(
        hooks=HookSettings(pre_tool_use=(lambda _event: None,)),
    )

    with pytest.raises(AgentProfileOverrideUnsupported, match="not supported by adapter 'custom'"):
        KediRuntime._assert_profile_override_supported(profile, adapter)


def test_runtime_validates_hook_support_per_event() -> None:
    adapter = ACPAdapter.__new__(ACPAdapter)

    KediRuntime._assert_profile_override_supported(
        AgentProfile(hooks=HookSettings(user_prompt_submit=(lambda _event: None,))),
        adapter,
    )

    with pytest.raises(
        AgentProfileOverrideUnsupported,
        match="pre_tool_use.*not supported by adapter 'acp'",
    ):
        KediRuntime._assert_profile_override_supported(
            AgentProfile(hooks=HookSettings(pre_tool_use=(lambda _event: None,))),
            adapter,
        )


def test_kedi_on_registers_handlers_in_context_local_profile() -> None:
    @kedi.on(("user_prompt_submit", "pre_tool_use"))
    def outer(event: Any) -> None:
        return None

    from kedi.api.state import _config_var

    outer_settings = _config_var.get().profile.hooks
    assert outer_settings is not None
    assert outer_settings.user_prompt_submit == (outer,)
    assert outer_settings.pre_tool_use == (outer,)

    with kedi.context(hooks={"pre_tool_use": []}):
        inner = _config_var.get().profile.hooks
        assert inner is not None
        assert inner.user_prompt_submit == (outer,)
        assert inner.pre_tool_use == ()

    assert _config_var.get().profile.hooks == outer_settings


def test_kedi_on_accepts_lists_and_rejects_non_callable_decorator_targets() -> None:
    @kedi.on(["user_prompt_submit", "post_tool_use"])
    def observe(_event: Any) -> None:
        return None

    from kedi.api.state import _config_var

    hooks = _config_var.get().profile.hooks
    assert hooks is not None
    assert hooks.user_prompt_submit == (observe,)
    assert hooks.post_tool_use == (observe,)
    with pytest.raises(TypeError, match="handler must be callable"):
        kedi.on("pre_tool_use")(cast(Any, 42))


def test_hook_profile_helpers_cover_empty_and_invalid_internal_values() -> None:
    empty = AgentProfile.empty()
    assert empty.with_hook_enforcement() is empty
    assert api_cache_module._hooks_identity(None) is None
    identity = api_cache_module._hooks_identity(HookSettings(enabled=True, pre_tool_use=(len,)))
    assert identity is not None
    assert identity[0] is True
    assert identity[1][1][1][0][0] == "builtins"

    with pytest.raises(TypeError, match="requires an inline Python value"):
        profile_model._eval_hooks_settings(
            KediRuntime(),
            enabled=True,
            fields=[SettingsField(name="pre_tool_use")],
            env={},
        )


def test_hooks_directive_parses_in_all_agent_scopes() -> None:
    program = parse_program(
        "> hooks:\n"
        "    user_prompt_submit: `redact`\n"
        "    pre_tool_use: `[validate, audit]`\n\n"
        "> profile: checked:\n"
        "    > hooks: disabled\n\n"
        "@run():\n"
        "    > hooks: enabled\n"
        "    = done\n"
    )

    top = program.ordered_toplevel[0]
    assert isinstance(top, HooksDirectiveStmt)
    assert top.enabled is True
    assert [(field.name, field.backtick_code) for field in top.fields] == [
        ("user_prompt_submit", "redact"),
        ("pre_tool_use", "[validate, audit]"),
    ]
    profile = program.ordered_toplevel[1]
    assert isinstance(profile, ProfileDefStmt)
    assert profile.hooks_members[0].enabled is False
    assert isinstance(program.procedures[0].body[0], HooksDirectiveStmt)
    assert program.procedures[0].body[0].enabled is True


def test_hooks_directive_evaluates_ordered_handlers_and_clear() -> None:
    def first(_event: Any) -> None:
        return None

    def second(_event: Any) -> None:
        return None

    def third(_event: Any) -> None:
        return None

    directive = parse_program(
        "> hooks:\n"
        "    pre_tool_use: `[first, second]`\n"
        "    pre_tool_use: `[]`\n"
        "    pre_tool_use: `third`\n"
    ).ordered_toplevel[0]
    assert isinstance(directive, HooksDirectiveStmt)
    profile = eval_hooks_directive(
        KediRuntime(),
        directive,
        {"first": first, "second": second, "third": third},
    )

    assert profile.hooks is not None
    assert profile.hooks.enabled is True
    assert profile.hooks.handlers_for("pre_tool_use") == (third,)


def test_top_level_hooks_are_captured_as_procedure_enforcement() -> None:
    def enforce(_event: Any) -> None:
        return None

    program = parse_program("> hooks:\n    user_prompt_submit: `enforce`\n\n@run():\n    = done\n")
    compile_program(program, runtime_globals={"enforce": enforce})

    captured = program.procedures[0].captured_agent_profile
    assert isinstance(captured, AgentProfile)
    assert captured.hooks is not None
    assert captured._hook_enforcement is not None
    assert captured.hooks.handlers_for("user_prompt_submit") == (enforce,)
    assert captured._hook_enforcement.handlers_for("user_prompt_submit") == (enforce,)


def test_hooks_directives_execute_at_top_level_and_in_procedure_scope() -> None:
    def observe(_event: Any) -> None:
        return None

    source = """
> hooks:
    user_prompt_submit: `observe`

@local() -> int:
    > hooks:
        pre_tool_use: `observe`
    = `1`

= `local()`
"""
    runtime = compile_program(
        parse_program(source),
        runtime_globals={"observe": observe},
    )
    assert runtime.run_main() == 1


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("> hooks: active\n", "must be 'enabled' or 'disabled'"),
        ("> hooks:\n    PreToolUse: `handler`\n", "Unknown hook event"),
        ("> hooks:\n    pre_tool_use: handler\n", "unexpected syntax"),
    ],
)
def test_hooks_directive_rejects_noncanonical_syntax(source: str, message: str) -> None:
    with pytest.raises(KediParseError, match=message):
        parse_program(source)


def test_hooks_directive_rejects_non_callable_values() -> None:
    directive = parse_program(
        "> hooks:\n    pre_tool_use: `[lambda event: None, 1]`\n"
    ).ordered_toplevel[0]
    assert isinstance(directive, HooksDirectiveStmt)

    with pytest.raises(TypeError, match="Every hook handler"):
        eval_hooks_directive(KediRuntime(), directive, {})


def test_hooks_directive_body_ignores_comments() -> None:
    directive = parse_program(
        "> hooks:\n    # handler order is intentional\n    pre_tool_use: `handler`\n"
    ).ordered_toplevel[0]
    assert isinstance(directive, HooksDirectiveStmt)
    assert [field.name for field in directive.fields] == ["pre_tool_use"]


def test_profile_hooks_merge_in_source_order_and_can_be_reenabled() -> None:
    def inherited(_event: Any) -> None:
        return None

    def local(_event: Any) -> None:
        return None

    profile = parse_program(
        "> profile: checked:\n"
        "    > hooks:\n"
        "        pre_tool_use: `inherited`\n"
        "    > hooks: disabled\n"
        "    > hooks: enabled\n"
        "    > hooks:\n"
        "        pre_tool_use: `local`\n"
    ).profile_defs["checked"]
    evaluated = eval_profile_definition(
        KediRuntime(),
        profile_stmt_to_definition(profile),
        {"inherited": inherited, "local": local},
    )

    assert evaluated.hooks is not None
    assert evaluated.hooks.enabled is True
    assert evaluated.hooks.handlers_for("pre_tool_use") == (inherited, local)


def test_dispatcher_orders_surfaces_and_applies_edits_cumulatively() -> None:
    observed: list[tuple[str, str]] = []

    def constructor(event: Any) -> None:
        observed.append(("constructor", event.arguments["path"]))

    def adapter_edit(event: Any) -> PreToolUseDecision:
        observed.append(("adapter", event.arguments["path"]))
        return PreToolUseDecision.edit({"path": "adapter"})

    async def profile_edit(event: Any) -> PreToolUseDecision:
        await asyncio.sleep(0)
        observed.append(("profile", event.arguments["path"]))
        return PreToolUseDecision.edit({"path": "profile"})

    adapter = _Adapter(constructor)
    adapter.on("pre_tool_use")(adapter_edit)
    dispatcher = HookDispatcher(
        HookDispatcherSnapshot.compose(
            adapter.hook_snapshot(),
            HookSettings(pre_tool_use=(profile_edit,)),
        )
    )

    result = asyncio.run(dispatcher.dispatch(_pre()))
    assert result == PreToolUseDecision.edit({"path": "profile"})
    assert observed == [
        ("constructor", "before"),
        ("adapter", "before"),
        ("profile", "adapter"),
    ]


def test_dispatcher_suspends_recursive_hook_dispatch() -> None:
    calls = 0
    dispatcher: HookDispatcher

    async def recurse(_event: Any) -> UserPromptSubmitDecision:
        nonlocal calls
        calls += 1
        nested = await dispatcher.dispatch(_prompt("nested"))
        assert nested == UserPromptSubmitDecision.continue_()
        return UserPromptSubmitDecision.continue_()

    dispatcher = HookDispatcher(
        HookDispatcherSnapshot.compose(
            AdapterHookSnapshot.empty(),
            HookSettings(user_prompt_submit=(recurse,)),
        )
    )
    assert asyncio.run(dispatcher.dispatch(_prompt())) == UserPromptSubmitDecision.continue_()
    assert calls == 1


def test_dispatcher_keeps_concurrent_tool_calls_independent() -> None:
    observed: list[tuple[str, int, int]] = []

    async def edit(event: Any) -> PreToolUseDecision:
        index = int(event.arguments["index"])
        await asyncio.sleep((5 - index) * 0.001)
        observed.append((event.tool_call_id, event.sequence, index))
        return PreToolUseDecision.edit({"index": index + 10})

    dispatcher = HookDispatcher(
        HookDispatcherSnapshot.compose(
            AdapterHookSnapshot.empty(),
            HookSettings(pre_tool_use=(edit,)),
        )
    )

    async def exercise() -> list[Any]:
        return await asyncio.gather(
            *(
                dispatcher.dispatch(
                    replace(
                        _pre({"index": index}),
                        tool_call_id=f"call-{index}",
                    )
                )
                for index in range(6)
            )
        )

    results = asyncio.run(exercise())
    assert sorted(event[1] for event in observed) == list(range(1, 7))
    assert {event[0] for event in observed} == {f"call-{index}" for index in range(6)}
    assert {
        int(result.edit_arguments["index"])
        for result in results
        if isinstance(result, PreToolUseDecision) and result.edit_arguments is not None
    } == set(range(10, 16))


def test_dispatcher_cancellation_drains_active_hook() -> None:
    entered = asyncio.Event()
    exited = asyncio.Event()

    async def block(_event: Any) -> None:
        entered.set()
        try:
            await asyncio.Future()
        finally:
            exited.set()

    dispatcher = HookDispatcher(
        HookDispatcherSnapshot.compose(
            AdapterHookSnapshot.empty(),
            HookSettings(pre_tool_use=(block,)),
        )
    )

    async def exercise() -> None:
        task = asyncio.create_task(dispatcher.dispatch(_pre()))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert exited.is_set()

    asyncio.run(exercise())


def test_dispatcher_validates_timeout_and_default_results() -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        HookDispatcherSnapshot.compose(None, None, timeout=0)

    dispatcher = HookDispatcher(HookDispatcherSnapshot.compose(None, None))
    assert asyncio.run(dispatcher.dispatch(_prompt())) == UserPromptSubmitDecision.continue_()
    assert dispatcher.dispatch_sync(_pre()) == PreToolUseDecision.continue_()

    post = PostToolUseEvent(
        run_id="run",
        sequence=0,
        adapter_shortname="fake",
        tool_call_id="call",
        tool_name="tool",
        origin="kedi",
        arguments={},
        result="done",
    )
    assert asyncio.run(dispatcher.dispatch(post)) is None


def test_dispatcher_sync_recursion_is_suspended_and_denial_stops_chain() -> None:
    calls: list[str] = []
    dispatcher: HookDispatcher

    def deny(_event: Any) -> PreToolUseDecision:
        calls.append("deny")
        assert dispatcher.dispatch_sync(_pre()) == PreToolUseDecision.continue_()
        return PreToolUseDecision.deny("policy")

    def unreachable(_event: Any) -> None:
        calls.append("unreachable")

    dispatcher = HookDispatcher(
        HookDispatcherSnapshot.compose(
            AdapterHookSnapshot.empty(),
            HookSettings(pre_tool_use=(deny, unreachable)),
        )
    )
    assert dispatcher.dispatch_sync(_pre()) == PreToolUseDecision.deny("policy")
    assert calls == ["deny"]


def test_dispatcher_wraps_sync_and_async_handler_failures() -> None:
    failure = HookExecutionError(
        event="pre_tool_use",
        handler_name="already-wrapped",
        cause=ValueError("failure"),
    )

    def wrapped(_event: Any) -> None:
        raise failure

    def broken(_event: Any) -> None:
        raise ValueError("broken")

    async def broken_async(_event: Any) -> None:
        await asyncio.sleep(0)
        raise ValueError("async broken")

    for handler, use_async in ((wrapped, False), (broken, False), (broken_async, True)):
        dispatcher = HookDispatcher(
            HookDispatcherSnapshot.compose(
                AdapterHookSnapshot.empty(),
                HookSettings(pre_tool_use=(handler,)),
            )
        )
        with pytest.raises(HookExecutionError) as info:
            if use_async:
                asyncio.run(dispatcher.dispatch(_pre()))
            else:
                dispatcher.dispatch_sync(_pre())
        if handler is wrapped:
            assert info.value is failure
        else:
            assert isinstance(info.value.original, ValueError)


def test_dispatcher_timeout_paths_are_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def wait_forever(_event: Any) -> None:
        await asyncio.Future()

    async_dispatcher = HookDispatcher(
        HookDispatcherSnapshot.compose(
            AdapterHookSnapshot.empty(),
            HookSettings(pre_tool_use=(wait_forever,)),
            timeout=0.001,
        )
    )
    with pytest.raises(HookTimeoutError):
        asyncio.run(async_dispatcher.dispatch(_pre()))
    with pytest.raises(HookTimeoutError):
        async_dispatcher.dispatch_sync(_pre())

    ticks = iter((0.0, 2.0))
    monkeypatch.setattr(hook_dispatcher, "monotonic", lambda: next(ticks))
    sync_dispatcher = HookDispatcher(
        HookDispatcherSnapshot.compose(
            AdapterHookSnapshot.empty(),
            HookSettings(pre_tool_use=(lambda _event: None,)),
            timeout=1.0,
        )
    )
    with pytest.raises(HookTimeoutError):
        sync_dispatcher.dispatch_sync(_pre())


def test_sync_dispatch_rejects_async_handler_inside_active_loop() -> None:
    async def handler(_event: Any) -> None:
        return None

    dispatcher = HookDispatcher(
        HookDispatcherSnapshot.compose(
            AdapterHookSnapshot.empty(),
            HookSettings(pre_tool_use=(handler,)),
        )
    )

    async def exercise() -> None:
        with pytest.raises(HookExecutionError, match="cannot await") as info:
            dispatcher.dispatch_sync(_pre())
        assert isinstance(info.value.original, RuntimeError)

    asyncio.run(exercise())


def test_sync_dispatch_awaits_async_handler_without_timeout() -> None:
    async def handler(_event: Any) -> PreToolUseDecision:
        await asyncio.sleep(0)
        return PreToolUseDecision.edit({"path": "edited"})

    dispatcher = HookDispatcher(
        HookDispatcherSnapshot.compose(
            AdapterHookSnapshot.empty(),
            HookSettings(pre_tool_use=(handler,)),
        )
    )
    assert dispatcher.dispatch_sync(_pre()) == PreToolUseDecision.edit({"path": "edited"})


def test_sync_dispatch_rejects_noncoroutine_awaitable_on_active_loop() -> None:
    class DecisionAwaitable:
        def __await__(self) -> Any:
            async def resolve() -> PreToolUseDecision:
                return PreToolUseDecision.continue_()

            return resolve().__await__()

    def handler(_event: Any) -> Any:
        return DecisionAwaitable()

    dispatcher = HookDispatcher(
        HookDispatcherSnapshot.compose(
            AdapterHookSnapshot.empty(),
            HookSettings(pre_tool_use=(handler,)),
        )
    )

    async def exercise() -> None:
        with pytest.raises(HookExecutionError, match="cannot await"):
            dispatcher.dispatch_sync(_pre())

    asyncio.run(exercise())


def test_dispatcher_rejects_wrong_handler_results_and_unknown_events() -> None:
    cases = (
        (_prompt(), lambda _event: PreToolUseDecision.continue_(), "user_prompt_submit"),
        (_pre(), lambda _event: UserPromptSubmitDecision.continue_(), "pre_tool_use"),
        (
            PostToolUseEvent(
                run_id="run",
                sequence=0,
                adapter_shortname="fake",
                tool_call_id="call",
                tool_name="tool",
                origin="kedi",
                arguments={},
                result="done",
            ),
            lambda _event: PreToolUseDecision.continue_(),
            "post_tool_use",
        ),
    )
    for event, handler, message in cases:
        dispatcher = HookDispatcher(
            HookDispatcherSnapshot.compose(
                AdapterHookSnapshot.empty(),
                HookSettings(**{event.event: (handler,)}),
            )
        )
        with pytest.raises(TypeError, match=message):
            dispatcher.dispatch_sync(event)

    with pytest.raises(TypeError, match="Unsupported Kedi hook event"):
        hook_dispatcher._apply_result(cast(Any, object()), None)


def test_hook_telemetry_records_chain_outcome_without_payloads() -> None:
    canary = "private-hook-canary"

    def edit(event: Any) -> PreToolUseDecision:
        assert event.arguments["path"] == canary
        return PreToolUseDecision.edit({"path": "redacted"})

    dispatcher = HookDispatcher(
        HookDispatcherSnapshot.compose(
            AdapterHookSnapshot.empty(),
            HookSettings(pre_tool_use=(edit,)),
        )
    )

    with installed_recording_backend() as backend:
        dispatcher.dispatch_sync(_pre({"path": canary}))

    assert [span.name for span in backend.finished_spans] == ["run pre tool use hooks"]
    span = backend.finished_spans[0]
    assert span.attributes["kedi.hook.outcome"] == "edited"
    assert span.attributes["kedi.hook.handler_count"] == 1
    assert canary not in repr(backend.finished_spans)
    assert canary not in repr(backend.metrics)
    assert {metric.name for metric in backend.metrics} == {
        "kedi.hook.calls",
        "kedi.hook.duration",
        "kedi.hook.edits",
    }


def test_hook_telemetry_without_entered_span_is_a_safe_noop() -> None:
    telemetry = HookChainTelemetry(_prompt(), 1)
    assert telemetry.__exit__(None, None, None) is None


class _BoundaryAdapter(_Adapter):
    active_profile = AgentProfile.empty()

    @hooked_agent_run("prompt")
    def invoke_sync(self, *, prompt: str) -> str:
        return prompt

    @hooked_agent_run("prompt")
    async def invoke(self, *, prompt: str) -> str:
        await asyncio.sleep(0)
        return prompt

    @hooked_agent_run("prompt")
    def nested(self, *, prompt: str) -> str:
        return self.invoke_sync(prompt=prompt)


def test_prompt_boundaries_cover_sync_async_native_and_nested_runs() -> None:
    plain = _BoundaryAdapter()
    assert plain.invoke_sync(prompt="plain") == "plain"
    assert asyncio.run(plain.invoke(prompt="plain")) == "plain"

    observed: list[UserPromptSubmitRequest] = []

    def edit(event: Any) -> UserPromptSubmitDecision:
        assert isinstance(event, UserPromptSubmitRequest)
        observed.append(event)
        return UserPromptSubmitDecision.edit(f"edited:{event.content}")

    adapter = _BoundaryAdapter(edit)
    assert adapter.invoke_sync(prompt="sync") == "edited:sync"
    assert asyncio.run(adapter.invoke(prompt="async")) == "edited:async"
    assert adapter.nested(prompt="nested") == "edited:nested"
    assert [event.content for event in observed] == ["sync", "async", "nested"]

    async def native() -> tuple[Any, ...]:
        async with native_agent_hook_run(adapter, "native") as edited:
            nested_dispatcher = hook_dispatcher.current_hook_dispatcher()
            async with native_agent_hook_run(adapter, "already-bound") as untouched:
                pass
        async with native_agent_hook_run(adapter, 42) as non_string:
            pass
        async with native_agent_hook_run(plain, "no-hooks") as no_hooks:
            pass
        return edited, untouched, non_string, no_hooks, nested_dispatcher

    edited, untouched, non_string, no_hooks, nested_dispatcher = asyncio.run(native())
    assert edited == "edited:native"
    assert untouched == "already-bound"
    assert non_string == 42
    assert no_hooks == "no-hooks"
    assert nested_dispatcher is not None


def test_prompt_boundary_validates_decorator_receiver_prompt_and_decision() -> None:
    with pytest.raises(TypeError, match="has no 'missing' argument"):

        @hooked_agent_run("missing")
        def invalid(self: Any, *, prompt: str) -> str:
            del self
            return prompt

    with pytest.raises(TypeError, match="HookRegistrationMixin"):
        _BoundaryAdapter.invoke_sync(cast(Any, object()), prompt="value")

    adapter = _BoundaryAdapter(lambda _event: None)
    with pytest.raises(TypeError, match="must be a string"):
        adapter.invoke_sync(prompt=cast(Any, 42))
    with pytest.raises(TypeError, match="invalid decision"):
        hook_boundary._resolve_prompt_decision(None, "prompt")


def test_hook_decision_and_failure_models_validate_all_modes() -> None:
    with pytest.raises(ValueError, match="continue.*reason or edit"):
        UserPromptSubmitDecision(reason="unexpected")
    with pytest.raises(ValueError, match="denial reason"):
        UserPromptSubmitDecision(
            decision="edit",
            reason="unexpected",
            edited_content="edited",
        )
    with pytest.raises(ValueError, match="Unknown hook decision"):
        UserPromptSubmitDecision(decision=cast(Any, "later"))
    with pytest.raises(ValueError, match="cannot be negative"):
        PostToolUseFailureEvent(
            run_id="run",
            sequence=1,
            adapter_shortname="fake",
            tool_call_id="call",
            tool_name="tool",
            origin="kedi",
            arguments={},
            error_type="ValueError",
            error_message="failure",
            duration_seconds=-1,
        )

    timeout = HookTimeoutError(
        event="pre_tool_use",
        handler_name="handler",
        timeout=1.5,
    )
    assert timeout.timeout == 1.5
    assert timeout.handler_name == "handler"
    assert "timed out" in timeout.message


def test_tool_dispatch_helpers_noop_without_matching_handlers() -> None:
    tool = ToolSpec(fn=lambda *, value: value, name="identity", risk="read_only")

    async def exercise() -> None:
        assert await dispatch_pre_tool_use(
            tool,
            {"value": 1},
            adapter_shortname="fake",
            tool_call_id="call",
        ) == {"value": 1}
        await dispatch_post_tool_use(
            tool,
            {"value": 1},
            1,
            adapter_shortname="fake",
            tool_call_id="call",
        )
        await dispatch_post_tool_use_failure(
            tool,
            {"value": 1},
            ValueError("failure"),
            adapter_shortname="fake",
            tool_call_id="call",
        )

    asyncio.run(exercise())
    assert dispatch_pre_tool_use_sync(
        tool,
        {"value": 2},
        adapter_shortname="fake",
        tool_call_id="call",
    ) == {"value": 2}
    dispatch_post_tool_use_sync(
        tool,
        {"value": 2},
        2,
        adapter_shortname="fake",
        tool_call_id="call",
    )
    dispatch_post_tool_use_failure_sync(
        tool,
        {"value": 2},
        ValueError("failure"),
        adapter_shortname="fake",
        tool_call_id="call",
    )


def test_common_tool_wrapper_covers_sync_async_and_awaitable_lifecycles() -> None:
    observed: list[Any] = []

    def observe(event: Any) -> Any:
        observed.append(event)
        if isinstance(event, PreToolUseRequest):
            return PreToolUseDecision.edit({"value": int(event.arguments["value"]) + 1})
        return None

    dispatcher = HookDispatcher(
        HookDispatcherSnapshot.compose(
            AdapterHookSnapshot.empty(),
            HookSettings(
                pre_tool_use=(observe,),
                post_tool_use=(observe,),
                post_tool_use_failure=(observe,),
            ),
        )
    )

    async def async_success(*, value: int) -> int:
        await asyncio.sleep(0)
        return value * 2

    async def async_failure(*, value: int) -> int:
        del value
        await asyncio.sleep(0)
        raise ValueError("async failure")

    def awaitable_success(*, value: int) -> Any:
        async def resolve() -> int:
            return value * 3

        return resolve()

    def awaitable_failure(*, value: int) -> Any:
        del value

        async def fail() -> int:
            raise ValueError("awaitable failure")

        return fail()

    def sync_failure(*, value: int) -> int:
        del value
        raise ValueError("sync failure")

    wrapped_async = with_hook_lifecycle(
        ToolSpec(fn=async_success, name="async_success", risk="read_only"),
        adapter_shortname="fake",
    )
    wrapped_async_failure = with_hook_lifecycle(
        ToolSpec(fn=async_failure, name="async_failure", risk="read_only"),
        adapter_shortname="fake",
    )
    wrapped_awaitable = with_hook_lifecycle(
        ToolSpec(fn=awaitable_success, name="awaitable_success", risk="read_only"),
        adapter_shortname="fake",
    )
    wrapped_awaitable_failure = with_hook_lifecycle(
        ToolSpec(fn=awaitable_failure, name="awaitable_failure", risk="read_only"),
        adapter_shortname="fake",
    )
    wrapped_sync_failure = with_hook_lifecycle(
        ToolSpec(fn=sync_failure, name="sync_failure", risk="read_only"),
        adapter_shortname="fake",
    )
    assert with_hook_lifecycle(wrapped_async, adapter_shortname="fake") is wrapped_async

    async def exercise() -> None:
        with bind_agent_run("run"), bind_hook_dispatcher(dispatcher):
            assert await wrapped_async.fn(value=1) == 4
            assert await wrapped_awaitable.fn(value=1) == 6
            with pytest.raises(ValueError, match="async failure"):
                await wrapped_async_failure.fn(value=1)
            with pytest.raises(ValueError, match="awaitable failure"):
                await wrapped_awaitable_failure.fn(value=1)
            with pytest.raises(ValueError, match="sync failure"):
                wrapped_sync_failure.fn(value=1)
            with pytest.raises(TypeError, match="keyword arguments"):
                wrapped_sync_failure.fn(1)

    asyncio.run(exercise())
    assert [event.event for event in observed].count("post_tool_use") == 2
    assert [event.event for event in observed].count("post_tool_use_failure") == 3


def test_common_tool_wrapper_does_not_report_policy_denials_as_execution_failures() -> None:
    observed: list[Any] = []

    def observe(event: Any) -> None:
        observed.append(event)

    denied = ToolApprovalDenied(message="denied")

    async def async_denied(*, value: int) -> int:
        del value
        raise denied

    def sync_denied(*, value: int) -> int:
        del value
        raise denied

    def awaitable_denied(*, value: int) -> Any:
        del value

        async def fail() -> int:
            raise denied

        return fail()

    tools = (
        with_hook_lifecycle(
            ToolSpec(fn=async_denied, name="async_denied", risk="read_only"),
            adapter_shortname="fake",
        ),
        with_hook_lifecycle(
            ToolSpec(fn=sync_denied, name="sync_denied", risk="read_only"),
            adapter_shortname="fake",
        ),
        with_hook_lifecycle(
            ToolSpec(fn=awaitable_denied, name="awaitable_denied", risk="read_only"),
            adapter_shortname="fake",
        ),
    )
    dispatcher = HookDispatcher(
        HookDispatcherSnapshot.compose(
            AdapterHookSnapshot.empty(),
            HookSettings(post_tool_use_failure=(observe,)),
        )
    )

    async def exercise() -> None:
        with bind_agent_run("run"), bind_hook_dispatcher(dispatcher):
            with pytest.raises(ToolApprovalDenied):
                await tools[0].fn(value=1)
            with pytest.raises(ToolApprovalDenied):
                tools[1].fn(value=1)
            with pytest.raises(ToolApprovalDenied):
                await tools[2].fn(value=1)

    asyncio.run(exercise())
    assert observed == []


def test_tool_wrapper_decision_and_interruption_helpers_validate_edges() -> None:
    tool = ToolSpec(fn=lambda **_kwargs: None, name="tool", risk="read_only")
    with pytest.raises(TypeError, match="invalid decision"):
        hook_tool_wrapper._resolve_pre_decision(tool, None, {})
    assert (
        hook_tool_wrapper._is_execution_failure(HookDeniedError(event="pre_tool_use", reason="x"))
        is False
    )
    assert is_tool_execution_interrupted(asyncio.CancelledError()) is True
    assert is_tool_execution_interrupted(KeyboardInterrupt()) is True
    assert is_tool_execution_interrupted(SystemExit()) is True
    assert is_tool_execution_interrupted(ValueError()) is False


def test_subagent_inherits_enforcement_hooks_but_not_parent_profile_hooks() -> None:
    observed: list[tuple[str, UserPromptSubmitRequest]] = []

    def enforcement(event: Any) -> None:
        assert isinstance(event, UserPromptSubmitRequest)
        observed.append(("enforcement", event))

    def parent_local(event: Any) -> None:
        assert isinstance(event, UserPromptSubmitRequest)
        observed.append(("parent", event))

    def child_local(event: Any) -> None:
        assert isinstance(event, UserPromptSubmitRequest)
        observed.append(("child", event))

    async def respond(_messages: list[Any], _info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart("child complete")])

    source = """
> profile: worker:
    > adapter: pydantic
    > hooks:
        user_prompt_submit: `child_local`

> profile: worker_disabled:
    > adapter: pydantic
    > hooks:
        user_prompt_submit: `child_local`
    > hooks: disabled

> profile: manager:
    > adapter: pydantic
    > subagent: worker
    > subagent: worker_disabled
"""
    adapter = PydanticAdapter(FunctionModel(respond))
    runtime = compile_program(
        parse_program(source),
        adapter=adapter,
        runtime_globals={"child_local": child_local},
    )
    manager = eval_profile_definition(
        runtime,
        runtime._profile_registry["manager"],
        build_capture_env(runtime),
    )
    enforcement_settings = HookSettings(user_prompt_submit=(enforcement,))
    manager = replace(
        manager,
        hooks=enforcement_settings.merge(HookSettings(user_prompt_submit=(parent_local,))),
        _hook_enforcement=enforcement_settings,
    )
    delegate_task = runtime._resolve_agent_tool_specs(manager, adapter)[0]

    events: list[Any] = []
    with observe_agent_events(events.append):
        with bind_agent_run("parent-run", agent_name="manager"), agent_event_run("pydantic"):
            result = asyncio.run(
                delegate_task.fn(
                    subagent="worker",
                    task="Inspect the parser",
                    final_schema=None,
                )
            )

    assert result["task_summary"] == "child complete"
    assert [name for name, _event in observed] == ["enforcement", "child"]
    for _name, event in observed:
        assert event.run_id == result["run_id"]
        assert event.parent_run_id == "parent-run"
        assert event.agent_name == "worker"
        assert event.profile_name == "worker"

    observed.clear()
    with bind_agent_run("parent-run", agent_name="manager"), agent_event_run("pydantic"):
        disabled_result = asyncio.run(
            delegate_task.fn(
                subagent="worker_disabled",
                task="Inspect the parser",
                final_schema=None,
            )
        )

    assert disabled_result["task_summary"] == "child complete"
    assert [name for name, _event in observed] == ["enforcement"]
    assert observed[0][1].profile_name == "worker_disabled"


def test_decisions_reject_fields_for_the_wrong_mode() -> None:
    with pytest.raises(ValueError, match="cannot include edit_arguments"):
        PreToolUseDecision(decision="deny", reason="no", edit_arguments={"x": 1})
    with pytest.raises(ValueError, match="requires edited_content"):
        UserPromptSubmitDecision(decision="edit")
    with pytest.raises(ValueError, match="requires a non-empty reason"):
        UserPromptSubmitDecision(decision="deny", reason="")


def test_pydantic_prompt_hook_edits_transport_input_once() -> None:
    transported: list[str] = []
    hook_events: list[UserPromptSubmitRequest] = []

    async def respond(messages: list[Any], _info: AgentInfo) -> ModelResponse:
        request = next(message for message in messages if isinstance(message, ModelRequest))
        prompt_part = next(part for part in request.parts if isinstance(part, UserPromptPart))
        transported.append(str(prompt_part.content))
        return ModelResponse(parts=[TextPart("ok")])

    def catch_all(event: Any) -> None:
        if isinstance(event, UserPromptSubmitRequest):
            hook_events.append(event)

    adapter = PydanticAdapter(FunctionModel(respond), hook_handler=catch_all)
    conversation = AdapterConversationContext()

    @adapter.on("user_prompt_submit")
    def redact(event: Any) -> UserPromptSubmitDecision:
        assert isinstance(event, UserPromptSubmitRequest)
        return UserPromptSubmitDecision.edit(event.content.replace("secret", "[redacted]"))

    with installed_recording_backend() as backend, adapter.conversation_scope(conversation):
        assert adapter.invoke_sync(prompt="secret") == "ok"
    assert transported == ["[redacted]"]
    assert len(hook_events) == 1
    assert hook_events[0].content == "secret"
    assert "secret" not in repr(conversation.result_state)
    assert "secret" not in repr(backend.finished_spans)
    assert "secret" not in repr(backend.metrics)


def test_runtime_assembles_history_from_hook_edited_prompt() -> None:
    hook_calls = 0

    async def respond(_messages: list[Any], _info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart("ok")])

    adapter = PydanticAdapter(FunctionModel(respond))

    @adapter.on("user_prompt_submit")
    def redact(event: Any) -> UserPromptSubmitDecision:
        nonlocal hook_calls
        hook_calls += 1
        assert isinstance(event, UserPromptSubmitRequest)
        return UserPromptSubmitDecision.edit(event.content.replace("secret", "[redacted]"))

    runtime = compile_program(
        parse_program("> history: enabled\n[answer] << secret\n= <answer>\n"),
        adapter=adapter,
    )

    assert runtime.run_main() == "ok"
    assert hook_calls == 1
    assert runtime._implicit_conversation_state is not None
    turns = runtime._implicit_conversation_state.snapshot()
    assert turns[0].content == "[redacted]"
    assert "secret" not in repr(turns)


def test_pydantic_prompt_hook_denial_prevents_model_transport() -> None:
    calls = 0

    async def respond(_messages: list[Any], _info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        return ModelResponse(parts=[TextPart("unexpected")])

    adapter = PydanticAdapter(FunctionModel(respond))
    adapter.on("user_prompt_submit")(
        lambda _event: UserPromptSubmitDecision.deny("blocked by policy")
    )

    with pytest.raises(HookDeniedError, match="blocked by policy"):
        adapter.invoke_sync(prompt="secret")
    assert calls == 0


def test_nested_pydantic_subagent_produce_dispatches_prompt_hook_once() -> None:
    hook_calls = 0

    async def respond(_messages: list[Any], info: AgentInfo) -> ModelResponse:
        return ModelResponse(
            parts=[
                ToolCallPart(
                    info.output_tools[0].name,
                    {"task_summary": "done"},
                )
            ]
        )

    def observe(_event: Any) -> None:
        nonlocal hook_calls
        hook_calls += 1

    adapter = PydanticAdapter(FunctionModel(respond), hook_handler=observe)
    result = asyncio.run(
        adapter.produce_subagent(
            template="perform task",
            output_schema={
                "type": "object",
                "properties": {"task_summary": {"type": "string"}},
                "required": ["task_summary"],
                "additionalProperties": False,
            },
        )
    )

    assert result == {"task_summary": "done"}
    assert hook_calls == 1


def test_lazy_adapter_forwards_hooks_registered_before_and_after_resolution() -> None:
    real = _Adapter()
    lazy = LazyAdapter(lambda: real, shortname="fake")

    def before(_event: Any) -> None:
        return None

    def after(_event: Any) -> None:
        return None

    lazy.on("pre_tool_use")(before)
    assert lazy._ensure() is real
    lazy.on("pre_tool_use")(after)

    assert lazy.hook_snapshot().handlers_for("pre_tool_use") == (before, after)
    assert real.hook_snapshot().handlers_for("pre_tool_use") == (before, after)


def test_runtime_resolves_lazy_adapter_before_prompt_hook_dispatch() -> None:
    hook_calls = 0

    async def respond(_messages: list[Any], _info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart("ok")])

    lazy = LazyAdapter(
        lambda: PydanticAdapter(FunctionModel(respond)),
        shortname="pydantic",
    )

    @lazy.on("user_prompt_submit")
    def observe(_event: Any) -> None:
        nonlocal hook_calls
        hook_calls += 1

    runtime = compile_program(
        parse_program("[answer] << hello\n= <answer>\n"),
        adapter=lazy,
    )

    assert runtime.run_main() == "ok"
    assert lazy.is_initialized is True
    assert hook_calls == 1


def test_post_hook_event_repr_does_not_include_payload() -> None:
    event = PostToolUseFailureEvent(
        run_id="run",
        sequence=1,
        adapter_shortname="fake",
        tool_call_id="call",
        tool_name="tool",
        origin="kedi",
        arguments={"secret": "argument-canary"},
        metadata={"secret": "metadata-canary"},
        error_type="RuntimeError",
        error_message="error-canary",
    )
    rendered = repr(event)
    assert "argument-canary" not in rendered
    assert "metadata-canary" not in rendered
    assert "error-canary" not in rendered


def test_pydantic_native_tool_hooks_edit_before_approval_and_observe_result() -> None:
    transported: list[str] = []
    executed: list[str] = []
    approved: list[str] = []
    events: list[Any] = []
    model_calls = 0

    async def respond(messages: list[Any], _info: AgentInfo) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        request = next(message for message in messages if isinstance(message, ModelRequest))
        prompt = next(part for part in request.parts if isinstance(part, UserPromptPart))
        transported.append(str(prompt.content))
        if model_calls == 1:
            return ModelResponse(parts=[ToolCallPart("read_file", {"path": " raw "}, "call-read")])
        assert any(
            isinstance(part, ToolReturnPart) and part.content == "content:edited"
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
        )
        return ModelResponse(parts=[TextPart("done")])

    def read_file(*, path: str) -> str:
        executed.append(path)
        return f"content:{path}"

    def approve(request: ApprovalRequest) -> ApprovalDecision:
        approved.append(str(request.arguments["path"]))
        return ApprovalDecision.allow()

    spec = ToolSpec(
        fn=read_file,
        name="read_file",
        argument_validator=lambda args: {"path": str(args["path"]).strip()},
        risk="mutating",
    )
    adapter = PydanticAdapter(FunctionModel(respond))

    @adapter.on("user_prompt_submit")
    def edit_prompt(event: Any) -> UserPromptSubmitDecision:
        return UserPromptSubmitDecision.edit(f"edited:{event.content}")

    @adapter.on(("pre_tool_use", "post_tool_use"))
    def tool_hook(event: Any) -> Any:
        events.append(event)
        if isinstance(event, PreToolUseRequest):
            assert event.arguments == {"path": "raw"}
            return PreToolUseDecision.edit({"path": " edited "})
        return None

    with adapter.profile_override(AgentProfile(approval=ApprovalPolicy.dynamic(approve))):
        result = adapter.run_sync(
            "prompt",
            toolsets=[pydantic_toolset_from_specs([spec])],
        )

    assert result.output == "done"
    assert transported[0] == "edited:prompt"
    assert executed == ["edited"]
    assert approved == ["edited"]
    assert [event.event for event in events] == ["pre_tool_use", "post_tool_use"]
    assert events[0].tool_call_id == events[1].tool_call_id == "call-read"
    assert events[1].result == "content:edited"


def test_pydantic_constructor_tool_hooks_edit_before_native_approval() -> None:
    executed: list[int] = []
    approved: list[int] = []
    observed: list[Any] = []
    model_calls = 0

    def native_tool(value: int) -> int:
        """Double a native value."""

        executed.append(value)
        return value * 2

    async def respond(_messages: list[Any], _info: AgentInfo) -> ModelResponse:
        nonlocal model_calls
        model_calls += 1
        if model_calls == 1:
            return ModelResponse(parts=[ToolCallPart("native_tool", {"value": 1}, "native-call")])
        return ModelResponse(parts=[TextPart("done")])

    def approve(request: ApprovalRequest) -> ApprovalDecision:
        assert request.description == "Double a native value."
        approved.append(cast(int, request.arguments["value"]))
        return ApprovalDecision.allow()

    adapter = PydanticAdapter(FunctionModel(respond), tools=[native_tool])

    @adapter.on(("pre_tool_use", "post_tool_use"))
    def hook(event: Any) -> Any:
        observed.append(event)
        if isinstance(event, PreToolUseRequest):
            assert event.description == "Double a native value."
            return PreToolUseDecision.edit({"value": 3})
        return None

    with adapter.profile_override(AgentProfile(approval=ApprovalPolicy.dynamic(approve))):
        result = adapter.run_sync("call the native tool")

    assert result.output == "done"
    assert executed == [3]
    assert approved == [3]
    assert [event.event for event in observed] == ["pre_tool_use", "post_tool_use"]
    assert all(event.tool_call_id == "native-call" for event in observed)
    assert all(event.origin == "adapter_native" for event in observed)


def test_pydantic_hook_capability_maps_denial_failure_and_cancellation() -> None:
    observed: list[Any] = []

    async def respond(_messages: list[Any], _info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart("unused")])

    adapter = PydanticAdapter(FunctionModel(respond))
    capability = _KediHookCapability(adapter)
    tool_def = ToolDefinition(
        name="native_tool",
        description="Native tool",
        parameters_json_schema={"type": "object", "properties": {}},
        toolset_id="native",
    )

    deny_dispatcher = HookDispatcher(
        HookDispatcherSnapshot.compose(
            AdapterHookSnapshot.empty(),
            HookSettings(pre_tool_use=(lambda _event: PreToolUseDecision.deny("blocked"),)),
        )
    )

    async def deny() -> None:
        with bind_agent_run("run"), bind_hook_dispatcher(deny_dispatcher):
            with pytest.raises(ModelRetry, match="blocked"):
                await capability.before_tool_execute(
                    cast(Any, SimpleNamespace(tool_call_approved=False)),
                    call=ToolCallPart("native_tool", {"value": 1}, "deny-call"),
                    tool_def=tool_def,
                    args={"value": 1},
                )

    asyncio.run(deny())

    def observe(event: Any) -> None:
        observed.append(event)

    failure_dispatcher = HookDispatcher(
        HookDispatcherSnapshot.compose(
            AdapterHookSnapshot.empty(),
            HookSettings(post_tool_use_failure=(observe,)),
        )
    )

    async def fail_and_cancel() -> None:
        with bind_agent_run("run"), bind_hook_dispatcher(failure_dispatcher):
            cancel_call = ToolCallPart("native_tool", {"value": 2}, "cancel-call")
            await capability.before_tool_execute(
                cast(Any, SimpleNamespace(tool_call_approved=False)),
                call=cancel_call,
                tool_def=tool_def,
                args={"value": 2},
            )

            async def cancel(_args: Any) -> Any:
                raise asyncio.CancelledError

            with pytest.raises(asyncio.CancelledError):
                await capability.wrap_tool_execute(
                    cast(Any, SimpleNamespace()),
                    call=cancel_call,
                    tool_def=tool_def,
                    args={"value": 2},
                    handler=cast(Any, cancel),
                )

            error_call = ToolCallPart("native_tool", {"value": 3}, "error-call")
            await capability.before_tool_execute(
                cast(Any, SimpleNamespace(tool_call_approved=False)),
                call=error_call,
                tool_def=tool_def,
                args={"value": 3},
            )
            adapter._remember_hook_started("error-call")
            with pytest.raises(ValueError, match="tool failed"):
                await capability.on_tool_execute_error(
                    cast(Any, SimpleNamespace()),
                    call=error_call,
                    tool_def=tool_def,
                    args={"value": 3},
                    error=ValueError("tool failed"),
                )

            retry_call = ToolCallPart("native_tool", {"value": 4}, "retry-call")
            await capability.before_tool_execute(
                cast(Any, SimpleNamespace(tool_call_approved=False)),
                call=retry_call,
                tool_def=tool_def,
                args={"value": 4},
            )

            async def retry(_args: Any) -> Any:
                raise ModelRetry("correct the arguments")

            with pytest.raises(ModelRetry, match="correct the arguments"):
                await capability.wrap_tool_execute(
                    cast(Any, SimpleNamespace()),
                    call=retry_call,
                    tool_def=tool_def,
                    args={"value": 4},
                    handler=cast(Any, retry),
                )

    asyncio.run(fail_and_cancel())
    assert [event.event for event in observed] == [
        "post_tool_use_failure",
        "post_tool_use_failure",
        "post_tool_use_failure",
    ]
    assert observed[0].interrupted is True
    assert observed[1].interrupted is False
    assert observed[2].error_type == "ModelRetry"

    async def pass_through_framework_errors() -> None:
        async def fail(_args: Any) -> Any:
            raise ValueError("framework error")

        with pytest.raises(ValueError, match="framework error"):
            await capability.wrap_tool_execute(
                cast(Any, SimpleNamespace()),
                call=ToolCallPart("native_tool", {}, "framework-call"),
                tool_def=tool_def,
                args={},
                handler=cast(Any, fail),
            )
        with pytest.raises(HookDeniedError, match="blocked"):
            await capability.on_tool_execute_error(
                cast(Any, SimpleNamespace()),
                call=ToolCallPart("native_tool", {}, "denied-error"),
                tool_def=tool_def,
                args={},
                error=HookDeniedError(event="pre_tool_use", reason="blocked"),
            )

    asyncio.run(pass_through_framework_errors())

    uninitialized = PydanticAdapter.__new__(PydanticAdapter)
    started = uninitialized._hook_started_for_current_adapter()
    assert started.get() == {}
    assert uninitialized._hook_started_for_current_adapter() is started


def test_langchain_tool_hooks_edit_before_approval_and_observe_result() -> None:
    from langchain.messages import ToolMessage

    observed: list[Any] = []
    approved: list[str] = []
    executed: list[str] = []

    def hook(event: Any) -> Any:
        observed.append(event)
        if isinstance(event, PreToolUseRequest):
            return PreToolUseDecision.edit({"path": " edited "})
        return None

    def approve(request: ApprovalRequest) -> ApprovalDecision:
        approved.append(str(request.arguments["path"]))
        return ApprovalDecision.allow()

    tool = ToolSpec(
        fn=lambda **_kwargs: None,
        name="read_file",
        argument_validator=lambda args: {"path": str(args["path"]).strip()},
    )
    middleware = _langchain_approval_middleware(
        profile=AgentProfile(approval=ApprovalPolicy.dynamic(approve)),
        tools=(tool,),
    )
    dispatcher = HookDispatcher(
        HookDispatcherSnapshot.compose(
            AdapterHookSnapshot.empty(),
            HookSettings(
                pre_tool_use=(hook,),
                post_tool_use=(hook,),
                post_tool_use_failure=(hook,),
            ),
        )
    )

    @dataclass(frozen=True)
    class Request:
        tool_call: dict[str, Any]
        tool: Any = None

        def override(self, **changes: Any) -> Request:
            return replace(self, **changes)

    async def exercise() -> Any:
        async def handler(request: Request) -> ToolMessage:
            path = request.tool_call["args"]["path"]
            executed.append(path)
            return ToolMessage(
                content=f"content:{path}",
                name="read_file",
                tool_call_id="call-read",
            )

        with bind_agent_run("run"), bind_hook_dispatcher(dispatcher):
            return await middleware.awrap_tool_call(
                Request(
                    tool_call={
                        "name": "read_file",
                        "args": {"path": " raw "},
                        "id": "call-read",
                    }
                ),
                handler,
            )

    result = asyncio.run(exercise())
    assert result.content == "content:edited"
    assert approved == ["edited"]
    assert executed == ["edited"]
    assert [event.event for event in observed] == ["pre_tool_use", "post_tool_use"]
    assert observed[1].result is result


def test_langchain_hook_middleware_reports_error_results_and_cancellation() -> None:
    from langchain.messages import ToolMessage

    observed: list[Any] = []
    dispatcher = HookDispatcher(
        HookDispatcherSnapshot.compose(
            AdapterHookSnapshot.empty(),
            HookSettings(post_tool_use_failure=(observed.append,)),
        )
    )
    middleware = _langchain_approval_middleware(
        profile=AgentProfile.empty(),
        tools=(),
        approval_enabled=False,
    )

    @dataclass(frozen=True)
    class Request:
        tool_call: dict[str, Any]
        tool: Any = None

        def override(self, **changes: Any) -> Request:
            return replace(self, **changes)

    request = Request(
        tool_call={"name": "external", "args": {"value": 1}, "id": "external-call"},
        tool=SimpleNamespace(metadata={"kedi_origin": "mcp"}),
    )

    async def exercise_error() -> Any:
        async def handler(_request: Request) -> ToolMessage:
            return ToolMessage(
                content="tool failed",
                name="external",
                tool_call_id="external-call",
                status="error",
            )

        with bind_agent_run("run"), bind_hook_dispatcher(dispatcher):
            return await middleware.awrap_tool_call(request, handler)

    result = asyncio.run(exercise_error())
    assert result.status == "error"
    assert observed[-1].origin == "mcp"
    assert observed[-1].error_message == "tool failed"

    async def exercise_cancel() -> None:
        async def handler(_request: Request) -> Any:
            raise asyncio.CancelledError

        with bind_agent_run("run"), bind_hook_dispatcher(dispatcher):
            with pytest.raises(asyncio.CancelledError):
                await middleware.awrap_tool_call(request, handler)

    asyncio.run(exercise_cancel())
    assert observed[-1].interrupted is True


def test_langchain_native_tool_projection_and_argument_validation() -> None:
    native = SimpleNamespace(
        name="native_tool",
        description="Native tool",
        args_schema={"type": "object", "properties": {}},
        metadata={},
    )
    adapter = LangChainAdapter(model=object())
    middleware = adapter._approval_middleware_for_run([native])
    assert len(middleware) == 1

    external = ToolSpec(
        fn=lambda **_kwargs: None,
        name="external",
        metadata={"source": "langchain-external"},
        argument_validator=lambda arguments: {"value": int(arguments["value"]) + 1},
    )
    assert (
        _langchain_hook_origin(
            external,
            SimpleNamespace(metadata={"kedi_origin": "mcp"}),
        )
        == "mcp"
    )
    assert _validated_tool_arguments(external, {"value": 1}) == {"value": 2}


def test_claude_native_hook_bridge_preserves_call_identity_and_terminal_event() -> None:
    observed: list[Any] = []

    def hook(event: Any) -> Any:
        observed.append(event)
        if isinstance(event, PreToolUseRequest):
            return PreToolUseDecision.edit({"path": "edited"})
        return None

    tool = ToolSpec(
        fn=lambda **_kwargs: None,
        name="read_file",
        risk="read_only",
    )
    dispatcher = HookDispatcher(
        HookDispatcherSnapshot.compose(
            AdapterHookSnapshot.empty(),
            HookSettings(
                pre_tool_use=(hook,),
                post_tool_use=(hook,),
                post_tool_use_failure=(hook,),
            ),
        )
    )

    async def exercise_success() -> dict[str, Any]:
        started_at: dict[str, float] = {}
        with bind_agent_run("run"), bind_hook_dispatcher(dispatcher):
            decision = await _claude_pre_tool_use_hook(
                profile=AgentProfile.empty(),
                tools=(tool,),
                started_at=started_at,
            )(
                {"tool_name": "read_file", "tool_input": {"path": "raw"}},
                "claude-call",
                None,
            )
            await _claude_post_tool_use_hook(tools=(tool,), started_at=started_at)(
                {
                    "tool_name": "read_file",
                    "tool_input": {"path": "edited"},
                    "tool_response": "content",
                },
                "claude-call",
                None,
            )
            return decision

    decision = asyncio.run(exercise_success())
    assert decision["hookSpecificOutput"]["updatedInput"] == {"path": "edited"}
    assert [event.event for event in observed] == ["pre_tool_use", "post_tool_use"]
    assert all(event.tool_call_id == "claude-call" for event in observed)
    assert isinstance(observed[-1], PostToolUseEvent)
    assert observed[-1].result == "content"

    observed.clear()

    async def exercise_failure() -> None:
        started_at: dict[str, float] = {}
        with bind_agent_run("run"), bind_hook_dispatcher(dispatcher):
            await _claude_pre_tool_use_hook(
                profile=AgentProfile.empty(),
                tools=(tool,),
                started_at=started_at,
            )(
                {"tool_name": "read_file", "tool_input": {"path": "edited"}},
                "claude-failure",
                None,
            )
            await _claude_post_tool_use_failure_hook(
                tools=(tool,),
                started_at=started_at,
            )(
                {
                    "tool_name": "read_file",
                    "tool_input": {"path": "edited"},
                    "error": "unavailable",
                    "is_interrupt": True,
                },
                "claude-failure",
                None,
            )

    asyncio.run(exercise_failure())
    assert [event.event for event in observed] == ["pre_tool_use", "post_tool_use_failure"]
    failure = observed[-1]
    assert failure.tool_call_id == "claude-failure"
    assert failure.duration_seconds is not None
    assert failure.interrupted is True


def test_claude_hook_configuration_tracks_active_handlers_and_terminal_bridges() -> None:
    @dataclass
    class Matcher:
        matcher: Any
        hooks: list[Any]

    sdk = SimpleNamespace(HookMatcher=Matcher)
    failure_dispatcher = HookDispatcher(
        HookDispatcherSnapshot.compose(
            AdapterHookSnapshot.empty(),
            HookSettings(post_tool_use_failure=(lambda _event: None,)),
        )
    )
    with bind_hook_dispatcher(failure_dispatcher):
        identities = _active_claude_hook_identities()
        hooks = _with_kedi_approval_hook(
            sdk,
            None,
            profile=AgentProfile.empty(),
            tools=(),
        )
    assert tuple(identities) == ("post_tool_use_failure",)
    assert set(hooks) == {"PreToolUse", "PostToolUse", "PostToolUseFailure"}

    post_dispatcher = HookDispatcher(
        HookDispatcherSnapshot.compose(
            AdapterHookSnapshot.empty(),
            HookSettings(post_tool_use=(lambda _event: None,)),
        )
    )
    with bind_hook_dispatcher(post_dispatcher):
        post_hooks = _with_kedi_approval_hook(
            sdk,
            {},
            profile=AgentProfile.empty(),
            tools=(),
        )
    assert set(post_hooks) == {"PreToolUse", "PostToolUse"}


def test_claude_hook_bridge_maps_denial_and_validates_sdk_payloads() -> None:
    dispatcher = HookDispatcher(
        HookDispatcherSnapshot.compose(
            AdapterHookSnapshot.empty(),
            HookSettings(pre_tool_use=(lambda _event: PreToolUseDecision.deny("blocked"),)),
        )
    )

    async def deny() -> dict[str, Any]:
        with bind_agent_run("run"), bind_hook_dispatcher(dispatcher):
            return await _claude_pre_tool_use_hook(
                profile=AgentProfile.empty(),
                tools=(),
            )(
                {"tool_name": "Write", "tool_input": {"path": "file"}},
                None,
                None,
            )

    decision = asyncio.run(deny())
    assert decision["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "blocked" in decision["hookSpecificOutput"]["permissionDecisionReason"]

    async def post_without_duration() -> None:
        with bind_agent_run("run"):
            await _claude_post_tool_use_hook(tools=())(
                {"tool_name": "Read", "tool_input": {}, "tool_response": "done"},
                None,
                None,
            )

    asyncio.run(post_without_duration())

    with pytest.raises(TypeError, match="invalid PreToolUse"):
        _claude_tool_call_input(42, "PreToolUse")
    with pytest.raises(TypeError, match="incomplete PreToolUse"):
        _claude_tool_call_input({"tool_name": "Read"}, "PreToolUse")


def test_codex_projected_tools_run_one_hook_lifecycle() -> None:
    observed: list[Any] = []
    executed: list[str] = []

    def hook(event: Any) -> Any:
        observed.append(event)
        if isinstance(event, PreToolUseRequest):
            return PreToolUseDecision.edit({"path": "edited"})
        return None

    adapter = CodexAdapter(client=cast(Any, object()))
    tool = ToolSpec(
        fn=lambda *, path: executed.append(path) or f"content:{path}",
        name="read_file",
        risk="read_only",
    )
    dispatcher = HookDispatcher(
        HookDispatcherSnapshot.compose(
            AdapterHookSnapshot.empty(),
            HookSettings(pre_tool_use=(hook,), post_tool_use=(hook,)),
        )
    )

    with adapter.tool_scope((tool,)), bind_agent_run("run"), bind_hook_dispatcher(dispatcher):
        run = adapter._prepare_invoke("inspect", {})
        result = run.tools["read_file"](path="raw")

    assert result == "content:edited"
    assert executed == ["edited"]
    assert [event.event for event in observed] == ["pre_tool_use", "post_tool_use"]
    assert observed[0].tool_call_id == observed[1].tool_call_id


def test_webgpu_tool_loop_emits_exactly_one_terminal_hook_event() -> None:
    observed: list[Any] = []

    class Bridge:
        def request(
            self,
            payload: Mapping[str, Any],
            *,
            timeout: float,
        ) -> dict[str, Any]:
            del payload
            return {"text": str(timeout)}

    def hook(event: Any) -> Any:
        observed.append(event)
        if isinstance(event, PreToolUseRequest):
            return PreToolUseDecision.edit({"value": 2})
        return None

    adapter = WebGPUAdapter(Bridge(), model="browser")
    tool = ToolSpec(
        fn=lambda *, value: value * 3,
        name="multiply",
        risk="read_only",
    )
    dispatcher = HookDispatcher(
        HookDispatcherSnapshot.compose(
            AdapterHookSnapshot.empty(),
            HookSettings(
                pre_tool_use=(hook,),
                post_tool_use=(hook,),
                post_tool_use_failure=(hook,),
            ),
        )
    )

    async def exercise() -> tuple[str, bool, list[dict[str, Any]]]:
        runtime_tool = adapter._runtime_tools
        with adapter.tool_scope((tool,)):
            async with AsyncExitStack() as stack:
                tools = await runtime_tool(stack, AgentProfile.empty())
                with bind_agent_run("run"), bind_hook_dispatcher(dispatcher):
                    return await adapter._execute_tool_call(
                        {
                            "name": "multiply",
                            "arguments": {"value": 1},
                            "callId": "browser-call",
                        },
                        tools,
                    )

    _name, ok, messages = asyncio.run(exercise())
    assert ok is True
    assert '"result": 6' in messages[-1]["content"]
    assert [event.event for event in observed] == ["pre_tool_use", "post_tool_use"]
    assert all(event.tool_call_id == "browser-call" for event in observed)


def test_webgpu_hook_bridge_handles_mcp_denial_validation_and_cancellation() -> None:
    class Bridge:
        def request(self, payload: Mapping[str, Any], *, timeout: float) -> dict[str, Any]:
            del payload, timeout
            return {"text": "unused"}

    class RuntimeTool:
        name = "external"
        description = "External tool"
        input_schema = {"type": "object", "properties": {}}

        async def call(self, arguments: dict[str, Any]) -> Any:
            del arguments
            raise asyncio.CancelledError

    adapter = WebGPUAdapter(Bridge(), model="browser")
    runtime_tool = cast(Any, RuntimeTool())

    deny_dispatcher = HookDispatcher(
        HookDispatcherSnapshot.compose(
            AdapterHookSnapshot.empty(),
            HookSettings(pre_tool_use=(lambda _event: PreToolUseDecision.deny("blocked"),)),
        )
    )

    async def deny() -> Any:
        with bind_agent_run("run"), bind_hook_dispatcher(deny_dispatcher):
            return await adapter._execute_tool_call(
                {"name": "external", "arguments": {}, "callId": "denied"},
                [runtime_tool],
            )

    _name, ok, messages = asyncio.run(deny())
    assert ok is False
    assert "blocked" in messages[-1]["content"]

    observed: list[Any] = []
    failure_dispatcher = HookDispatcher(
        HookDispatcherSnapshot.compose(
            AdapterHookSnapshot.empty(),
            HookSettings(post_tool_use_failure=(observed.append,)),
        )
    )

    async def cancel() -> None:
        with bind_agent_run("run"), bind_hook_dispatcher(failure_dispatcher):
            with pytest.raises(asyncio.CancelledError):
                await adapter._execute_tool_call(
                    {"name": "external", "arguments": {}, "callId": "cancelled"},
                    [runtime_tool],
                )

    asyncio.run(cancel())
    assert observed[-1].origin == "mcp"
    assert observed[-1].interrupted is True

    invalid = ToolSpec(
        fn=lambda **_kwargs: None,
        name="invalid",
        risk="read_only",
        argument_validator=lambda _arguments: (_ for _ in ()).throw(ValueError("invalid args")),
    )

    async def invalid_arguments() -> Any:
        with adapter.tool_scope((invalid,)):
            async with AsyncExitStack() as stack:
                tools = await adapter._runtime_tools(stack, AgentProfile.empty())
                return await adapter._execute_tool_call(
                    {"name": "invalid", "arguments": {}, "callId": "invalid"},
                    tools,
                )

    _name, ok, messages = asyncio.run(invalid_arguments())
    assert ok is False
    assert "invalid args" in messages[-1]["content"]

    broken_dispatcher = HookDispatcher(
        HookDispatcherSnapshot.compose(
            AdapterHookSnapshot.empty(),
            HookSettings(
                pre_tool_use=(lambda _event: (_ for _ in ()).throw(ValueError("broken hook")),)
            ),
        )
    )

    async def broken_hook() -> None:
        with bind_agent_run("run"), bind_hook_dispatcher(broken_dispatcher):
            with pytest.raises(HookExecutionError, match="broken hook"):
                await adapter._execute_tool_call(
                    {"name": "external", "arguments": {}, "callId": "broken"},
                    [runtime_tool],
                )

    asyncio.run(broken_hook())
