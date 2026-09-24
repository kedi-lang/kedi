from __future__ import annotations

import asyncio
from typing import Annotated, Any

import pytest
from pydantic import AfterValidator

import kedi.agent_adapter.tools.conversion as conversion_module
from kedi import HookExecutionError, query, tool
from kedi.agent_adapter.events import bind_agent_run
from kedi.agent_adapter.hooks import (
    AdapterHookSnapshot,
    HookDispatcher,
    HookDispatcherSnapshot,
    bind_hook_dispatcher,
    with_hook_lifecycle,
)
from kedi.agent_adapter.tools.conversion import (
    kedi_procedure_tool_spec,
    python_callable_tool_spec,
)
from kedi.agent_adapter.tools.retry import ToolRetryPolicy, tool_retry_policy
from kedi.agent_profile import HookSettings
from kedi.errors import KediExecutionError, KediLoopIterationLimitError, KediTypeExecutionError
from kedi.lang import KediParseError, compile_program, parse_program
from tests.mock_adapter import MockAdapter


def _native_spec(source: str, **runtime_globals: Any):
    program = parse_program(source)
    runtime = compile_program(program, runtime_globals=runtime_globals)
    return runtime, program.procedures[0], kedi_procedure_tool_spec(runtime, program.procedures[0])


def test_native_tool_metadata_preserves_source_identity_and_docstring() -> None:
    source = """
@read_release(version: str) -> str:
    ###
    Native procedure documentation.
    ###
    > tool:
        name: lookup_release
        description: Read release notes.
        risk: read_only
        retries: 2
        retry_on:
            TimeoutError
            ConnectionError
    = `version`
"""
    runtime, procedure, spec = _native_spec(source)

    assert procedure.tool is not None
    assert procedure.tool.name == "lookup_release"
    assert procedure.tool.description == "Read release notes."
    assert procedure.tool.retry_on == ("TimeoutError", "ConnectionError")
    assert procedure.docstring == "Native procedure documentation."
    assert runtime._procedures["read_release"].__doc__ == "Native procedure documentation."
    assert spec.name == "lookup_release"
    assert spec.description is not None
    assert spec.description.startswith("Read release notes.")
    assert spec.risk == "read_only"
    assert spec.retry_policy.retries == 2
    assert spec.retry_policy.retry_on == (TimeoutError, ConnectionError)
    assert spec.metadata is not None
    assert spec.metadata["kedi_source_name"] == "read_release"
    assert spec.metadata["kedi_schema"]["name"] == "lookup_release"
    assert runtime._procedures["read_release"](version="v1") == "v1"


def test_native_tool_alias_accepts_plain_exposed_names_like_python_tools() -> None:
    _, _, spec = _native_spec(
        """
@read_release() -> str:
    > tool:
        name: lookup-release
    = ok
"""
    )

    assert spec.name == "lookup-release"


def test_native_tool_and_requirement_lists_ignore_comments_and_deduplicate() -> None:
    program = parse_program(
        """
> requires:
    stream_events
    # Keep the declaration stable when generated configuration repeats it.
    stream_events

@lookup() -> str:
    > tool:
        # Retry only transient transport failures.
        retry_on:
            TimeoutError
            # Repeated names do not change policy order.
            TimeoutError
    = ok
"""
    )

    requirement = program.toplevel_body[0]
    assert requirement.requirements == ["stream_events"]
    assert program.procedures[0].tool is not None
    assert program.procedures[0].tool.retry_on == ("TimeoutError",)


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (
            """
@lookup() -> str:
    > tool:
        retries: 1
        retry_on: TimeoutError
    = ok
""",
            "Expected source_file",
        ),
        (
            """
@lookup() -> str:
    = ok
    > tool:
        risk: read_only
""",
            "before executable procedure statements",
        ),
        (
            """
@lookup() -> str:
    > tool:
        risk: read_only
    > tool:
        retries: 1
    = ok
""",
            "only one '> tool:'",
        ),
        (
            """
@lookup() -> str:
    > tool:
        mystery: value
    = ok
""",
            "Unknown tool field",
        ),
        (
            """
@lookup() -> str:
    > tool:
        retries: true
    = ok
""",
            "nonnegative integer",
        ),
        (
            """
@lookup() -> str:
    > tool:
        retries: -1
    = ok
""",
            "nonnegative integer",
        ),
        (
            """
@lookup() -> str:
    > tool:
        risk: harmless
    = ok
""",
            "tool risk must be",
        ),
        (
            """
@lookup() -> str:
    > tool:
        risk: read_only
        risk: sensitive
    = ok
""",
            "Duplicate tool field",
        ),
        (
            """
@lookup() -> str:
    > tool:
        retry_on:
            TimeoutError
        retry_on:
            ConnectionError
    = ok
""",
            "Duplicate tool field",
        ),
        (
            """
@lookup() -> str:
    > if: `is_safe`:
        > tool:
            risk: read_only
    = ok
""",
            "only valid in a procedure metadata header",
        ),
    ],
)
def test_native_tool_metadata_rejects_invalid_declarations(source: str, message: str) -> None:
    with pytest.raises(KediParseError, match=message):
        parse_program(source)


def test_native_tool_retry_matches_body_failure_only() -> None:
    class ReleaseClient:
        def __init__(self) -> None:
            self.calls = 0

        def read(self, version: str) -> str:
            self.calls += 1
            if self.calls == 1:
                raise TimeoutError("temporary")
            return f"release:{version}"

    client = ReleaseClient()
    source = """
@read_release(version: str) -> str:
    > tool:
        retries: 2
        retry_on:
            TimeoutError
    = `release_client.read(version)`
"""
    _, _, spec = _native_spec(source, release_client=client)

    assert spec.fn(version="v1") == "release:v1"
    assert client.calls == 2


def test_native_tool_argument_validation_happens_once_before_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validations = 0
    original = conversion_module._validated_procedure_kwargs

    def count_validation(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal validations
        validations += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        conversion_module,
        "_validated_procedure_kwargs",
        count_validation,
    )
    _, _, spec = _native_spec(
        """
@lookup(value: int) -> str:
    > tool:
        retries: 2
    = `str(value)`
"""
    )

    with pytest.raises(TypeError):
        spec.fn()
    assert validations == 1


def test_native_tool_retry_does_not_repeat_nonmatching_or_return_validation_errors() -> None:
    class Fixture:
        def __init__(self) -> None:
            self.calls = 0
            self.mode = "value-error"

        def run(self) -> object:
            self.calls += 1
            if self.mode == "value-error":
                raise ValueError("permanent")
            return 42

    fixture = Fixture()
    source = """
@lookup() -> str:
    > tool:
        retries: 2
        retry_on:
            TimeoutError
    = `fixture.run()`
"""
    _, _, spec = _native_spec(source, fixture=fixture)

    with pytest.raises(KediExecutionError, match="permanent"):
        spec.fn()
    assert fixture.calls == 1

    fixture.mode = "invalid-return"
    with pytest.raises(KediTypeExecutionError, match="Return value"):
        spec.fn()
    assert fixture.calls == 2


def test_native_tool_retry_resolves_custom_exceptions_before_body_execution() -> None:
    class TemporaryLookupError(Exception):
        pass

    source = """
@lookup() -> str:
    > tool:
        retries: 1
        retry_on:
            TemporaryLookupError
    = ok
"""
    _, _, spec = _native_spec(source, TemporaryLookupError=TemporaryLookupError)
    assert spec.retry_policy.retry_on == (TemporaryLookupError,)

    missing = source.replace("TemporaryLookupError", "MissingLookupError")
    program = parse_program(missing)
    with pytest.raises(KediExecutionError, match="Unknown retry exception"):
        compile_program(program)

    invalid = source.replace("TemporaryLookupError", "NotAnException")
    program = parse_program(invalid)
    with pytest.raises(KediExecutionError, match="must resolve to an Exception class"):
        compile_program(program, runtime_globals={"NotAnException": object()})


def test_native_retry_validates_annotated_argument_once_before_body_attempts() -> None:
    validations: list[int] = []
    attempts: list[int] = []

    def validate(value: int) -> int:
        validations.append(value)
        return value + 1

    def execute(value: int) -> str:
        attempts.append(value)
        if len(attempts) == 1:
            raise TimeoutError("temporary")
        return "ok"

    runtime, _, spec = _native_spec(
        """
@lookup(value: CheckedInt) -> str:
    > tool:
        retries: 1
        retry_on:
            TimeoutError
    = `execute(value)`
""",
        CheckedInt=Annotated[int, AfterValidator(validate)],
        execute=execute,
    )

    assert spec.fn(value=1) == "ok"
    assert validations == [1]
    assert attempts == [2, 2]
    assert runtime._procedures["lookup"](value=3) == "ok"
    assert validations == [1, 3]
    assert attempts == [2, 2, 4]


def test_native_retry_policy_is_frozen_at_definition_before_first_exposure() -> None:
    program = parse_program("""
@lookup() -> str:
    > tool:
        retries: 1
        retry_on:
            TransientError
    = ok
""")
    runtime = compile_program(program, runtime_globals={"TransientError": TimeoutError})
    runtime._globals_env["TransientError"] = ValueError

    spec = kedi_procedure_tool_spec(runtime, program.procedures[0])
    assert spec.retry_policy.retry_on == (TimeoutError,)


def test_untyped_native_tool_arguments_remain_strings_with_prevalidated_execution() -> None:
    _, _, spec = _native_spec("@lookup(value) -> str:\n    = <value>\n")
    with pytest.raises(TypeError, match="Parameter 'value'"):
        spec.fn(value=123)
    assert spec.fn(value="valid") == "valid"


def test_python_selective_retry_supports_sync_async_subclasses_and_empty_filter() -> None:
    class TemporaryError(TimeoutError):
        pass

    sync_calls = 0

    @tool(retries=1, retry_on=(TimeoutError,))
    def sync_lookup() -> str:
        nonlocal sync_calls
        sync_calls += 1
        if sync_calls == 1:
            raise TemporaryError("temporary")
        return "ok"

    async_calls = 0

    @tool(retries=1, retry_on=(ConnectionError,))
    async def async_lookup() -> str:
        nonlocal async_calls
        async_calls += 1
        if async_calls == 1:
            raise ConnectionError("temporary")
        return "ok"

    no_retry_calls = 0

    @tool(retries=3, retry_on=())
    def no_retry() -> str:
        nonlocal no_retry_calls
        no_retry_calls += 1
        raise RuntimeError("stop")

    assert sync_lookup() == "ok"
    assert asyncio.run(async_lookup()) == "ok"
    with pytest.raises(RuntimeError, match="stop"):
        no_retry()
    assert (sync_calls, async_calls, no_retry_calls) == (2, 2, 1)


def test_retry_filter_deduplicates_and_never_retries_control_flow_errors() -> None:
    @tool(retries=1, retry_on=(TimeoutError, TimeoutError))
    def lookup() -> str:
        return "ok"

    assert tool_retry_policy(lookup).retry_on == (TimeoutError,)
    policy = ToolRetryPolicy(retries=1)
    assert policy.matches(asyncio.CancelledError()) is False
    assert policy.matches(KediLoopIterationLimitError("loop limit")) is False
    assert (
        policy.matches(
            HookExecutionError(
                event="post_tool_use",
                handler_name="audit",
                cause=RuntimeError("broken hook"),
            )
        )
        is False
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"retries": True},
        {"retries": 1.5},
        {"retry_on": "TimeoutError"},
        {"retry_on": (TimeoutError("instance"),)},
        {"retry_on": (KeyboardInterrupt,)},
    ],
)
def test_python_retry_policy_rejects_invalid_inputs(kwargs: dict[str, Any]) -> None:
    with pytest.raises(TypeError):
        tool(**kwargs)


def test_managed_python_tool_has_one_retry_owner_and_fresh_arguments() -> None:
    calls = 0
    seen: list[list[int]] = []

    @tool(retries=1, retry_on=(TimeoutError,))
    def mutate_then_retry(values: list[int]) -> str:
        nonlocal calls
        calls += 1
        seen.append(list(values))
        values.append(99)
        if calls == 1:
            raise TimeoutError("temporary")
        return "ok"

    spec = python_callable_tool_spec("mutate_then_retry", mutate_then_retry)
    original = [1]

    assert spec.fn(values=original) == "ok"
    assert calls == 2
    assert seen == [[1], [1]]
    assert original == [1]


def test_python_managed_retry_does_not_stack_with_decorator_retry() -> None:
    calls = 0

    @tool(retries=1)
    def fail() -> str:
        nonlocal calls
        calls += 1
        raise RuntimeError("broken")

    spec = python_callable_tool_spec("fail", fail)
    with pytest.raises(RuntimeError, match="broken"):
        spec.fn()
    assert calls == 2


def test_post_success_hook_failure_does_not_retry_tool_body() -> None:
    calls = 0

    @tool(retries=2)
    def lookup() -> str:
        nonlocal calls
        calls += 1
        return "ok"

    def fail_after_success(_event: Any) -> None:
        raise RuntimeError("post hook failed")

    spec = with_hook_lifecycle(
        python_callable_tool_spec("lookup", lookup),
        adapter_shortname="test",
    )
    dispatcher = HookDispatcher(
        HookDispatcherSnapshot.compose(
            AdapterHookSnapshot.empty(),
            HookSettings(post_tool_use=(fail_after_success,)),
        )
    )

    with (
        bind_agent_run("run"),
        bind_hook_dispatcher(dispatcher),
        pytest.raises(HookExecutionError, match="post hook failed"),
    ):
        spec.fn()

    assert calls == 1


def test_native_alias_collision_fails_before_model_request() -> None:
    adapter = MockAdapter()
    runtime = compile_program(
        parse_program(
            """
@first() -> str:
    > tool:
        name: lookup
    = first

@second() -> str:
    > tool:
        name: lookup
    = second

> use:
    first
    second

>> Result is [answer].
"""
        ),
        adapter=adapter,
    )

    with pytest.raises(KediExecutionError, match="Tool name collision"):
        runtime.run_main()
    assert adapter.request_calls == []


def test_python_tool_alias_collision_fails_during_query_registration() -> None:
    @tool(name="lookup")
    def first() -> str:
        return "first"

    @tool(name="lookup")
    def second() -> str:
        return "second"

    with pytest.raises(ValueError, match="duplicate exposed tool name: 'lookup'"):
        query(tools=(first, second))
