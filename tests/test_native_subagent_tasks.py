from __future__ import annotations

import ast
import asyncio
import time
import warnings
from dataclasses import replace
from threading import Event, Thread
from typing import Any

import pytest
from pydantic import BaseModel

import kedi.agent_adapter.execution as execution_module
import kedi.lang.compiler.subagent_tasks as subagent_tasks_module
from kedi.agent_adapter import SubagentResult, SubagentTask
from kedi.agent_adapter.adapters import LangChainAdapter
from kedi.agent_adapter.execution import OwnedAsyncLoop, use_run_deadline
from kedi.agent_adapter.subagents.budget import SubagentBudgetExceeded
from kedi.agent_adapter.subagents.native import NativeSubagentOwner
from kedi.agent_profile import AgentProfile, block_agent_profile_scope
from kedi.errors import KediExecutionError
from kedi.lang import KediParseError, compile_program, parse_program
from kedi.lang.ast import AwaitStmt, TaskStmt
from kedi.lang.compiler.statement_compiler import StatementCompiler
from kedi.lang.compiler.subagent_tasks import _inline_pydantic_refs
from kedi.lsp.features.completion import compute_completion
from kedi.lsp.features.navigation import (
    _collect_value_reference_candidates,
    _find_all_references,
)
from kedi.lsp.features.scope import _direct_statement_declarations
from kedi.lsp.features.symbols import SymbolRef
from kedi.lsp.python_virtual.document import compute_python_virtual_document
from kedi.run_budget import RunBudget, current_run_budget
from tests.lsp_helpers import definition_at, diagnostics_for, position_of, references_at
from tests.test_subagents import RecordingPydanticAdapter

PROFILES = """
> profile: worker:
    > adapter: pydantic
> profile: manager:
    > adapter: pydantic
    > subagent: worker
    > max_agents: 2
> use: manager
"""


def _runtime(
    body: str, adapter: RecordingPydanticAdapter | None = None
) -> tuple[Any, RecordingPydanticAdapter]:
    adapter = adapter or RecordingPydanticAdapter()
    return compile_program(parse_program(PROFILES + body), adapter=adapter), adapter


def test_task_capture_uses_one_child_call_and_no_parent_call() -> None:
    runtime, adapter = _runtime(
        "> task [job]: worker:\n"
        "    >> The value of <input> is [value: int].\n"
        "> await [answer]: job\n"
        "= `answer.output.value`\n"
    )
    runtime._initial_globals_env["input"] = "3 + 4"

    assert runtime.run_main() == 7
    assert len(adapter.invocations) == 1
    assert adapter.invocations[0][0] == "The value of 3 + 4 is [value: int]."
    assert "value" not in runtime._globals_env
    assert isinstance(runtime._globals_env["answer"].output, BaseModel)


def test_multiple_task_captures_are_attributes_on_one_model() -> None:
    class ReviewAdapter(RecordingPydanticAdapter):
        async def produce(self, template: str, **kwargs: Any) -> Any:
            self.invocations.append((template, None, (), ()))
            return {
                "task_summary": "Reviewed the change.",
                "final_result": {
                    "issue": "Missing test",
                    "recommendation": "Add integration coverage",
                },
            }

    runtime, adapter = _runtime(
        "> task [job]: worker:\n"
        "    >> The issue in <change> is [issue: str].\n"
        "    The recommended fix is [recommendation: str].\n"
        "> await [review]: job\n"
        "= `review.output.recommendation`\n",
        ReviewAdapter(),
    )
    runtime._initial_globals_env["change"] = "new feature"
    assert runtime.run_main() == "Add integration coverage"
    assert isinstance(runtime._globals_env["review"].output, BaseModel)
    assert runtime._globals_env["review"].output.issue == "Missing test"
    assert len(adapter.invocations) == 1


def test_child_declared_output_type_is_materialized() -> None:
    source = (
        "~Review(value: int)\n"
        + PROFILES.replace(
            "> profile: worker:\n    > adapter: pydantic",
            "> profile: worker:\n    > adapter: pydantic\n    > output: Review",
        )
        + "> task [job]: worker:\n    >> Compute a value.\n"
        "> await [review]: job\n= `review.output.value`\n"
    )
    runtime = compile_program(parse_program(source), adapter=RecordingPydanticAdapter())
    assert runtime.run_main() == 7
    assert isinstance(runtime._globals_env["review"].output, BaseModel)
    assert type(runtime._globals_env["review"].output).__name__ == "Review"


def test_bare_await_blocks_and_later_wait_reuses_result() -> None:
    runtime, adapter = _runtime(
        "> task [job]: worker:\n"
        "    >> Explain the result.\n"
        "> await: job\n"
        "> await [answer]: job\n"
        "= `answer.task_summary`\n"
    )
    assert runtime.run_main() == "child:Explain the result."
    assert len(adapter.invocations) == 1


def test_unawaited_task_fails_closed() -> None:
    runtime, _ = _runtime("> task [job]: worker:\n    >> Explain the result.\n")
    with pytest.raises(KediExecutionError, match="was not awaited"):
        runtime.run_main()


def test_non_child_is_rejected_before_start() -> None:
    runtime, adapter = _runtime("> task [job]: stranger:\n    >> Explain the result.\n")
    with pytest.raises(KediExecutionError, match="is not allowed"):
        runtime.run_main()
    assert not adapter.invocations


def test_task_captures_do_not_override_child_output_type() -> None:
    source = PROFILES.replace(
        "> profile: worker:\n    > adapter: pydantic",
        "> profile: worker:\n    > adapter: pydantic\n    > output: dict[str, int]",
    )
    adapter = RecordingPydanticAdapter()
    runtime = compile_program(
        parse_program(source + "> task [job]: worker:\n    >> Give [value: int].\n"),
        adapter=adapter,
    )
    with pytest.raises(KediExecutionError, match="cannot combine task captures"):
        runtime.run_main()
    assert not adapter.invocations


def test_child_error_propagates_from_bare_await() -> None:
    class FailingAdapter(RecordingPydanticAdapter):
        async def invoke(self, *, prompt: str, **kwargs: Any) -> str:
            raise RuntimeError("child execution failed")

    runtime, _ = _runtime(
        "> task [job]: worker:\n    >> Explain the result.\n> await: job\n",
        FailingAdapter(),
    )
    with pytest.raises(RuntimeError, match="child execution failed"):
        runtime.run_main()


def test_two_children_start_before_first_await() -> None:
    started = Event()
    release = Event()

    class PausedAdapter(RecordingPydanticAdapter):
        async def invoke(self, *, prompt: str, **kwargs: Any) -> str:
            self.invocations.append((prompt, None, (), ()))
            if len(self.invocations) == 2:
                started.set()
            await asyncio.to_thread(release.wait)
            return f"child:{prompt}"

    runtime, adapter = _runtime(
        "> task [first]: worker:\n    >> First.\n"
        "> task [second]: worker:\n    >> Second.\n"
        "> await: first\n> await: second\n",
        PausedAdapter(),
    )
    outcome: list[Any] = []

    def run() -> None:
        try:
            outcome.append(runtime.run_main())
        except BaseException as exc:
            outcome.append(exc)

    thread = Thread(target=run)
    thread.start()
    try:
        assert started.wait(10), "Second child did not start before the first await"
    finally:
        release.set()
        thread.join(timeout=10)
    assert not thread.is_alive()
    assert outcome == [None]
    assert len(adapter.invocations) == 2


def test_python_scope_returns_public_result_and_rejects_stale_handle() -> None:
    runtime, _ = _runtime("")

    async def run() -> tuple[SubagentTask, SubagentResult]:
        async with runtime.subagents(parent="manager") as agents:
            job = await agents.start("worker", task="Explain the result.")
            result = await job.wait()
        return job, result

    job, result = asyncio.run(run())
    assert result.subagent == "worker"
    assert result.output is None
    assert result.task_summary == "child:Explain the result."
    with pytest.raises(KediExecutionError, match="closed invocation"):
        asyncio.run(job.wait())


def test_python_scope_cancels_unawaited_child() -> None:
    class PausedAdapter(RecordingPydanticAdapter):
        async def invoke(self, *, prompt: str, **kwargs: Any) -> str:
            await asyncio.Event().wait()
            return prompt

    runtime, _ = _runtime("", PausedAdapter())

    async def run() -> None:
        async with runtime.subagents(parent="manager") as agents:
            await agents.start("worker", task="Do work.")

    with pytest.raises(KediExecutionError, match="was not awaited"):
        asyncio.run(run())


def test_unobserved_child_failure_is_not_silently_discarded() -> None:
    class FailingAdapter(RecordingPydanticAdapter):
        async def invoke(self, *, prompt: str, **kwargs: Any) -> str:
            raise RuntimeError("child failed")

    runtime, _ = _runtime("", FailingAdapter())

    async def run() -> None:
        async with runtime.subagents(parent="manager") as agents:
            job = await agents.start("worker", task="Review.")
            while runtime._subagent_status(job.run_id) == "running":
                await asyncio.sleep(0)

    with pytest.raises(RuntimeError, match="child failed"):
        asyncio.run(run())


def test_unawaited_result_survives_terminal_record_retention() -> None:
    runtime, _ = _runtime("")
    runtime._subagent_coordinator.max_retained_runs = 1

    async def completed(run_id: str) -> None:
        async def poll() -> None:
            while runtime._subagent_status(run_id) == "running":
                await asyncio.sleep(0)

        await asyncio.wait_for(poll(), timeout=2)

    async def run() -> None:
        async with runtime.subagents(parent="manager") as agents:
            first = await agents.start("worker", task="First.")
            await completed(first.run_id)
            second = await agents.start("worker", task="Second.")
            await completed(second.run_id)
            assert (await first.wait()).task_summary == "child:First."
            assert (await second.wait()).task_summary == "child:Second."
            assert (await first.wait()).task_summary == "child:First."

    asyncio.run(run())


def test_python_handle_rejects_another_event_loop() -> None:
    runtime, _ = _runtime("")

    async def run() -> None:
        async with runtime.subagents(parent="manager") as agents:
            job = await agents.start("worker", task="Review.")
            with pytest.raises(KediExecutionError, match="event loop"):
                await asyncio.to_thread(lambda: asyncio.run(job.wait()))
            assert (await job.wait()).task_summary == "child:Review."

    asyncio.run(run())


def test_cancelled_waiter_does_not_cancel_owned_child() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    class PausedAdapter(RecordingPydanticAdapter):
        async def invoke(self, *, prompt: str, **kwargs: Any) -> str:
            entered.set()
            await release.wait()
            return f"child:{prompt}"

    runtime, _ = _runtime("", PausedAdapter())

    async def run() -> None:
        async with runtime.subagents(parent="manager") as agents:
            job = await agents.start("worker", task="Review.")
            first_waiter = asyncio.create_task(job.wait())
            second_waiter = asyncio.create_task(job.wait())
            await asyncio.wait_for(entered.wait(), timeout=2)
            first_waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first_waiter
            release.set()
            assert (await second_waiter).task_summary == "child:Review."
            assert (await job.wait()).task_summary == "child:Review."

    asyncio.run(run())


def test_cancelled_parent_drains_only_its_child() -> None:
    entered = asyncio.Event()

    class PausedAdapter(RecordingPydanticAdapter):
        async def invoke(self, *, prompt: str, **kwargs: Any) -> str:
            entered.set()
            await asyncio.Event().wait()
            return prompt

    runtime, _ = _runtime("", PausedAdapter())

    async def run() -> None:
        async def parent() -> None:
            async with runtime.subagents(parent="manager") as agents:
                await agents.start("worker", task="Review.")
                await asyncio.Event().wait()

        parent_task = asyncio.create_task(parent())
        await asyncio.wait_for(entered.wait(), timeout=2)
        parent_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await parent_task
        assert runtime._subagent_coordinator.active_runs == 0
        assert not runtime._subagent_coordinator._retained_native_runs

    asyncio.run(run())


def test_repeated_parent_cancellation_waits_for_child_cleanup() -> None:
    entered = asyncio.Event()
    cleanup_entered = asyncio.Event()
    finish_cleanup = asyncio.Event()

    class SlowCancellationAdapter(RecordingPydanticAdapter):
        async def invoke(self, *, prompt: str, **kwargs: Any) -> str:
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cleanup_entered.set()
                await finish_cleanup.wait()
                raise

    runtime, _ = _runtime("", SlowCancellationAdapter())

    async def run() -> None:
        async def parent() -> None:
            async with runtime.subagents(parent="manager") as agents:
                await agents.start("worker", task="Review.")
                await asyncio.Event().wait()

        parent_task = asyncio.create_task(parent())
        await asyncio.wait_for(entered.wait(), timeout=2)
        parent_task.cancel()
        await asyncio.wait_for(cleanup_entered.wait(), timeout=2)
        parent_task.cancel()
        finish_cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await parent_task
        await asyncio.sleep(0)
        assert runtime._subagent_coordinator.active_runs == 0
        assert not runtime._subagent_coordinator._retained_native_runs

    asyncio.run(run())


def test_nested_capture_type_materializes_nested_model() -> None:
    class NestedAdapter(RecordingPydanticAdapter):
        async def produce(self, template: str, **kwargs: Any) -> Any:
            return {
                "task_summary": "Reviewed.",
                "final_result": {"findings": [{"message": "Missing test"}]},
            }

    source = (
        "~Finding(message: str)\n" + PROFILES + "> task [job]: worker:\n"
        "    >> The findings in <change> are [findings: list[Finding]].\n"
        "> await [review]: job\n"
        "= `review.output.findings[0].message`\n"
    )
    runtime = compile_program(parse_program(source), adapter=NestedAdapter())
    runtime._initial_globals_env["change"] = "new feature"
    assert runtime.run_main() == "Missing test"


def test_nested_capture_description_reaches_child_schema() -> None:
    seen: list[dict[str, Any]] = []

    class SchemaAdapter(RecordingPydanticAdapter):
        async def produce_subagent(
            self, *, template: str, output_schema: dict[str, Any], **kwargs: Any
        ) -> Any:
            seen.append(output_schema)
            return {"task_summary": "Reviewed.", "final_result": {"findings": ["Missing test"]}}

    runtime, _ = _runtime(
        "> task [job]: worker:\n"
        '    >> The findings are [findings: list[Annotated[str, "Finding label"]]].\n'
        "> await [review]: job\n"
        "= `review.output.findings[0]`\n",
        SchemaAdapter(),
    )
    assert runtime.run_main() == "Missing test"
    item = seen[0]["properties"]["final_result"]["properties"]["findings"]["items"]
    assert item["description"] == "Finding label"


def test_duplicate_task_capture_is_rejected_during_parsing() -> None:
    with pytest.raises(KediParseError, match="output field 'issue' more than once"):
        parse_program(
            PROFILES + "> task [job]: worker:\n"
            "    >> The issue is [issue: str] and the fix is [issue: str].\n"
        )


def test_invalid_task_capture_schema_is_rejected_before_child_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, adapter = _runtime("> task [job]: worker:\n    >> The value is [value: int].\n")

    def invalid_schema(schema: dict[str, Any]) -> None:
        raise ValueError("schema rejected")

    monkeypatch.setattr(subagent_tasks_module, "validate_final_schema", invalid_schema)
    with pytest.raises(KediExecutionError, match="schema rejected"):
        runtime.run_main()
    assert not adapter.invocations


@pytest.mark.parametrize(
    ("schema", "message"),
    [
        ({"$ref": "https://example.invalid/schema"}, "Unsupported"),
        ({"$ref": "#/$defs/Missing"}, "Unknown"),
        (
            {
                "$defs": {"Node": {"properties": {"child": {"$ref": "#/$defs/Node"}}}},
                "$ref": "#/$defs/Node",
            },
            "Recursive",
        ),
    ],
)
def test_task_schema_reference_errors_are_explicit(schema: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _inline_pydantic_refs(schema)


def test_nested_python_scopes_share_budget_but_new_roots_do_not() -> None:
    runtime, adapter = _runtime("")

    async def run() -> None:
        async with runtime.subagents(parent="manager") as outer:
            first = await outer.start("worker", task="First.")
            await first.wait()
            async with runtime.subagents(parent="manager") as inner:
                second = await inner.start("worker", task="Second.")
                await second.wait()
            with pytest.raises(SubagentBudgetExceeded):
                await outer.start("worker", task="Third.")
        async with runtime.subagents(parent="manager") as fresh:
            third = await fresh.start("worker", task="Third.")
            await third.wait()

    asyncio.run(run())
    assert [entry[0] for entry in adapter.invocations] == ["First.", "Second.", "Third."]


def test_owner_budget_tightens_when_profile_limit_drops() -> None:
    runtime, _ = _runtime("")
    owner = NativeSubagentOwner(runtime)
    first = owner.budget_for(AgentProfile(name="manager", max_agents=4))
    second = owner.budget_for(AgentProfile(name="manager", max_agents=2))
    assert first is second
    assert second.limit == 2
    assert owner.budget_for(AgentProfile(name="manager", max_agents=None)) is first
    assert first.limit == 2


def test_invalid_owner_and_scope_calls_fail_without_starting_child() -> None:
    runtime, adapter = _runtime("")
    owner = NativeSubagentOwner(runtime)
    other = NativeSubagentOwner(runtime)
    token = runtime._native_subagent_owner.set(owner)
    try:
        with pytest.raises(KediExecutionError, match="another invocation"):
            owner.wait_sync(SubagentTask(other, "foreign"))
        with pytest.raises(KediExecutionError, match="no active invocation loop"):
            owner.wait_sync(SubagentTask(owner, "missing"))
    finally:
        runtime._native_subagent_owner.reset(token)

    async def run() -> None:
        scope = runtime.subagents(parent="manager")
        with pytest.raises(KediExecutionError, match="not active"):
            await scope.start("worker", task="Review.")
        async with scope:
            with pytest.raises(KediExecutionError, match="already active"):
                await scope.__aenter__()
            assert scope._owner is not None
            scope._owner._closing = True
            with pytest.raises(KediExecutionError, match="closing"):
                await scope.start("worker", task="Review.")
            scope._owner._closing = False
        await owner.close()
        await owner.close()
        unknown_scope = runtime.subagents(parent="unknown")
        async with unknown_scope:
            with pytest.raises(KediExecutionError, match="Unknown parent"):
                await unknown_scope.start("worker", task="Review.")

    asyncio.run(run())
    assert not adapter.invocations


def test_allowed_but_missing_child_definition_fails_before_io() -> None:
    runtime, adapter = _runtime("")
    owner = NativeSubagentOwner(runtime)
    manager = runtime._evaluate_profile_definition(runtime._profile_registry["manager"])
    manager = replace(manager, subagents=("missing",), subagent_bindings={})

    async def run() -> None:
        token = runtime._native_subagent_owner.set(owner)
        try:
            with pytest.raises(KediExecutionError, match="Unknown subagent profile"):
                await owner.start("missing", task="Review.", parent_profile=manager)
            await owner.close()
        finally:
            runtime._native_subagent_owner.reset(token)

    asyncio.run(run())
    assert not adapter.invocations


def test_child_definition_can_resolve_from_registry_when_binding_is_absent() -> None:
    runtime, adapter = _runtime("")
    manager = runtime._evaluate_profile_definition(runtime._profile_registry["manager"])
    manager = replace(manager, subagent_bindings={})

    async def run() -> None:
        async with runtime.subagents(parent="manager") as scope:
            assert scope._owner is not None
            job = await scope._owner.start("worker", task="Review.", parent_profile=manager)
            assert (await job.wait()).task_summary == "child:Review."

    asyncio.run(run())
    assert len(adapter.invocations) == 1


def test_child_definition_prefers_lexical_binding_over_registry() -> None:
    runtime, adapter = _runtime("")
    manager = runtime._evaluate_profile_definition(runtime._profile_registry["manager"])

    def worker_definition(instructions: str) -> Any:
        source = f"> profile: worker:\n    > adapter: pydantic\n    > system: {instructions}\n"
        return compile_program(parse_program(source), adapter=adapter)._profile_registry["worker"]

    bound_worker = worker_definition("Bound instructions.")
    runtime._profile_registry["worker"] = worker_definition("Registry instructions.")
    manager = replace(manager, subagent_bindings={"worker": bound_worker})

    async def run() -> None:
        async with runtime.subagents(parent="manager") as scope:
            assert scope._owner is not None
            job = await scope._owner.start("worker", task="Review.", parent_profile=manager)
            assert (await job.wait()).task_summary == "child:Review."

    asyncio.run(run())
    assert len(adapter.invocations) == 1
    assert "Bound instructions" in (adapter.invocations[0][1] or "")
    assert "Registry instructions" not in (adapter.invocations[0][1] or "")


def test_completion_race_does_not_report_a_cancelled_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _ = _runtime("")
    cancelled: list[str] = []

    async def run() -> None:
        async with runtime.subagents(parent="manager") as agents:
            job = await agents.start("worker", task="Review.")
            await job.wait()
            monkeypatch.setattr(runtime, "_subagent_status", lambda run_id: "running")
            monkeypatch.setattr(
                runtime,
                "_cancel_subagent",
                lambda run_id: cancelled.append(run_id) or False,
            )

    asyncio.run(run())
    assert len(cancelled) == 1


def test_closed_async_owner_loop_rejects_sync_submission() -> None:
    runtime, _ = _runtime("")
    owner = NativeSubagentOwner(runtime)
    loop = asyncio.new_event_loop()
    loop.close()
    owner._execution_loop = loop
    with pytest.raises(KediExecutionError, match="event loop is closed"):
        owner._run_sync(lambda: asyncio.sleep(0))


def test_cancel_during_normal_scope_exit_still_drains_child() -> None:
    entered = asyncio.Event()
    cleanup_entered = asyncio.Event()
    finish_cleanup = asyncio.Event()

    class SlowCancellationAdapter(RecordingPydanticAdapter):
        async def invoke(self, *, prompt: str, **kwargs: Any) -> str:
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cleanup_entered.set()
                await finish_cleanup.wait()
                raise

    runtime, _ = _runtime("", SlowCancellationAdapter())

    async def run() -> None:
        async def parent() -> None:
            async with runtime.subagents(parent="manager") as agents:
                await agents.start("worker", task="Review.")
                await entered.wait()

        parent_task = asyncio.create_task(parent())
        await asyncio.wait_for(cleanup_entered.wait(), timeout=2)
        parent_task.cancel()
        finish_cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await parent_task
        assert runtime._subagent_coordinator.active_runs == 0
        assert not runtime._subagent_coordinator._retained_native_runs

    asyncio.run(run())


def test_python_scope_rejects_parent_impersonation() -> None:
    runtime, _ = _runtime("")

    async def run() -> None:
        async with runtime.subagents(parent="manager") as agents:
            with block_agent_profile_scope(AgentProfile(name="stranger")):
                with pytest.raises(KediExecutionError, match="effective parent"):
                    await agents.start("worker", task="Review.")
            token = runtime._subagent_stack.set((("run-1", "worker"),))
            try:
                with pytest.raises(KediExecutionError, match="impersonate"):
                    await agents.start("worker", task="Review.")
            finally:
                runtime._subagent_stack.reset(token)

    asyncio.run(run())


def test_handle_cannot_be_waited_from_another_root_with_same_profile() -> None:
    runtime, _ = _runtime("")

    async def run() -> None:
        delivered: asyncio.Future[SubagentTask] = asyncio.get_running_loop().create_future()

        async def other_root() -> None:
            async with runtime.subagents(parent="manager"):
                foreign = await delivered
                with pytest.raises(KediExecutionError, match="another or closed invocation"):
                    await foreign.wait()

        other = asyncio.create_task(other_root())
        async with runtime.subagents(parent="manager") as agents:
            own = await agents.start("worker", task="Review.")
            delivered.set_result(own)
            await other
            assert (await own.wait()).task_summary == "child:Review."

    asyncio.run(run())


@pytest.mark.parametrize(
    ("output_type", "value"),
    [
        ("int", 0),
        ("bool", False),
        ("str", ""),
        ("list[int]", [1, 2]),
        ("dict[str, int]", {"count": 2}),
        ("int | None", None),
    ],
)
def test_explicit_child_output_preserves_valid_values(output_type: str, value: Any) -> None:
    class ValueAdapter(RecordingPydanticAdapter):
        async def produce(self, template: str, **kwargs: Any) -> Any:
            return {"task_summary": "Done.", "final_result": value}

    profiles = PROFILES.replace(
        "> profile: worker:\n    > adapter: pydantic",
        f"> profile: worker:\n    > adapter: pydantic\n    > output: {output_type}",
    )
    runtime = compile_program(
        parse_program(
            profiles + "> task [job]: worker:\n    >> Compute the value.\n"
            "> await [answer]: job\n= `answer.output`\n"
        ),
        adapter=ValueAdapter(),
    )
    assert runtime.run_main() == value
    assert type(runtime._globals_env["answer"].output) is type(value)


def test_task_input_expression_is_rendered_once_before_child_start() -> None:
    runtime, adapter = _runtime(
        "> task [job]: worker:\n    >> Review <`source()`>.\n> await: job\n"
    )
    calls: list[str] = []

    def source() -> str:
        calls.append("called")
        return "original"

    runtime._initial_globals_env["source"] = source
    runtime.run_main()
    assert calls == ["called"]
    assert adapter.invocations[0][0] == "Review original."


def test_primary_error_survives_cleanup_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime, _ = _runtime("")
    original_release = runtime._subagent_coordinator.release_native_runs

    def failed_release(run_ids: tuple[str, ...]) -> None:
        original_release(run_ids)
        raise RuntimeError("cleanup failed")

    monkeypatch.setattr(runtime._subagent_coordinator, "release_native_runs", failed_release)

    async def run() -> None:
        async with runtime.subagents(parent="manager") as agents:
            job = await agents.start("worker", task="Review.")
            await job.wait()
            raise ValueError("primary failure")

    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="primary failure") as error:
            asyncio.run(run())
    if hasattr(error.value, "add_note"):
        assert "cleanup failed" in str(error.value.__notes__)
        assert not seen
    else:
        assert any(
            item.category is RuntimeWarning and "cleanup failed" in str(item.message)
            for item in seen
        )
    assert runtime._subagent_coordinator.active_runs == 0
    assert runtime._native_subagent_owner.get() is None


def test_cleanup_warns_when_primary_error_cannot_accept_notes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class LegacyError(Exception):
        add_note = None

    runtime, _ = _runtime("")

    def failed_release(run_ids: tuple[str, ...]) -> None:
        raise RuntimeError("cleanup failed")

    monkeypatch.setattr(runtime._subagent_coordinator, "release_native_runs", failed_release)

    async def run() -> None:
        async with runtime.subagents(parent="manager"):
            raise LegacyError("primary failure")

    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        with pytest.raises(LegacyError, match="primary failure"):
            asyncio.run(run())
    assert any("cleanup failed" in str(item.message) for item in seen)


def test_program_primary_error_survives_owner_cleanup_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _ = _runtime("= `fail()`\n")

    def fail() -> None:
        raise ValueError("primary failure")

    def failed_close(self: NativeSubagentOwner, error: BaseException | None = None) -> None:
        raise RuntimeError("cleanup failed")

    runtime._initial_globals_env["fail"] = fail
    monkeypatch.setattr(NativeSubagentOwner, "close_sync", failed_close)
    with pytest.raises(ValueError, match="primary failure"):
        runtime.run_main()


def test_task_and_await_executor_guards_reject_missing_owner_or_wrong_handle() -> None:
    runtime, _ = _runtime("")
    compiler = StatementCompiler(runtime)
    with pytest.raises(RuntimeError, match="active Kedi invocation"):
        compiler.execute_task(TaskStmt(binder="job", subagent="worker"), {}, AgentProfile())
    with pytest.raises(TypeError, match="expects a subagent task"):
        compiler.execute_await(AwaitStmt(handle="job"), {"job": 1})
    with pytest.raises(RuntimeError, match="active Kedi invocation"):
        compiler.execute_await(
            AwaitStmt(handle="job"),
            {"job": SubagentTask(NativeSubagentOwner(runtime), "fake")},
        )


def test_langchain_native_task_materializes_capture_model() -> None:
    class RecordingLangChainAdapter(LangChainAdapter):
        def __init__(self) -> None:
            super().__init__(model=object())
            self.tasks: list[str] = []

        async def produce_subagent(
            self, *, template: str, output_schema: dict[str, Any], **kwargs: Any
        ) -> Any:
            self.tasks.append(template)
            return {"task_summary": "Reviewed.", "final_result": {"recommendation": "Add tests"}}

    adapter = RecordingLangChainAdapter()
    source = (
        PROFILES.replace("> adapter: pydantic", "> adapter: langchain") + "> task [job]: worker:\n"
        "    >> The recommendation for <change> is [recommendation: str].\n"
        "> await [review]: job\n"
        "= `review.output.recommendation`\n"
    )
    runtime = compile_program(parse_program(source), adapter=adapter)
    runtime._initial_globals_env["change"] = "the change"
    assert runtime.run_main() == "Add tests"
    assert adapter.tasks == ["The recommendation for the change is [recommendation: str]."]


def test_unsupported_parent_adapter_rejects_task_before_child_io() -> None:
    class NoBackgroundAdapter(RecordingPydanticAdapter):
        capabilities = replace(
            RecordingPydanticAdapter.capabilities,
            supports_background_subagents=False,
        )

    adapter = NoBackgroundAdapter()
    runtime, _ = _runtime(
        "> task [job]: worker:\n    >> Review the change.\n",
        adapter,
    )
    with pytest.raises(KediExecutionError, match="background subagents"):
        runtime.run_main()
    assert not adapter.invocations


def test_owned_loop_reports_worker_startup_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken_worker(loop: OwnedAsyncLoop) -> None:
        loop._startup_error = RuntimeError("worker failed")
        loop._ready.set()

    monkeypatch.setattr(OwnedAsyncLoop, "_serve", broken_worker)
    with pytest.raises(RuntimeError, match="worker failed"):
        OwnedAsyncLoop()


def test_owned_loop_reports_real_worker_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    def failed_run(coroutine: Any) -> None:
        coroutine.close()
        raise RuntimeError("event loop unavailable")

    monkeypatch.setattr(execution_module.asyncio, "run", failed_run)
    with pytest.raises(RuntimeError, match="event loop unavailable"):
        OwnedAsyncLoop()


def test_owned_loop_reports_startup_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(OwnedAsyncLoop, "_serve", lambda self: None)
    monkeypatch.setattr(execution_module.Event, "wait", lambda self, timeout=None: False)
    with pytest.raises(RuntimeError, match="did not start"):
        OwnedAsyncLoop()


def test_owned_loop_rejects_same_loop_and_closed_loop_calls() -> None:
    loop = OwnedAsyncLoop()

    async def same_loop() -> None:
        coroutine = asyncio.sleep(0)
        try:
            with pytest.raises(RuntimeError, match="Cannot synchronously wait"):
                loop.run(coroutine)
        finally:
            coroutine.close()

    loop.run(same_loop())
    loop.close()
    loop.close()
    coroutine = asyncio.sleep(0)
    try:
        with pytest.raises(RuntimeError, match="closed"):
            loop.run(coroutine)
    finally:
        coroutine.close()


def test_expired_request_deadline_does_not_abort_owner_cleanup() -> None:
    runtime, _ = _runtime("")
    owner = NativeSubagentOwner(runtime)
    owner._loop = OwnedAsyncLoop()
    token = runtime._native_subagent_owner.set(owner)
    try:
        with use_run_deadline(time.monotonic() - 1):
            owner.close_sync()
        assert owner._closed
        assert not owner._loop._thread.is_alive()
    finally:
        runtime._native_subagent_owner.reset(token)


def test_expired_deadline_still_cancels_and_drains_a_running_child() -> None:
    started = Event()
    stopped = Event()

    class PausedAdapter(RecordingPydanticAdapter):
        async def invoke(self, *, prompt: str, **kwargs: Any) -> str:
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

    runtime, _ = _runtime("", PausedAdapter())
    manager = runtime._evaluate_profile_definition(runtime._profile_registry["manager"])
    owner = NativeSubagentOwner(runtime)
    token = runtime._native_subagent_owner.set(owner)
    try:
        job = owner.start_sync("worker", task="Review.", parent_profile=manager)
        assert started.wait(timeout=3)
        with use_run_deadline(time.monotonic() - 1):
            owner.close_sync(RuntimeError("parent deadline elapsed"))
        assert stopped.wait(timeout=3)
        assert owner._closed
        assert job.run_id not in runtime._subagent_coordinator._retained_native_runs
    finally:
        runtime._native_subagent_owner.reset(token)


def test_lsp_tracks_task_handle_without_leaking_capture_into_parent_scope() -> None:
    source = (
        PROFILES + "> task [job]: worker:\n"
        "    >> The issue in <change> is [issue: str].\n"
        "> await [review]: job\n"
        "= `review.output.issue`\n"
    )
    virtual = compute_python_virtual_document(source, "/tmp/task.kedi")
    ast.parse(virtual["text"])
    assert "job: SubagentTask" in virtual["text"]
    assert "review: SubagentResult" in virtual["text"]
    assert "issue: str" not in virtual["text"]
    uri = "file:///tmp/task.kedi"
    target = definition_at(source, uri, "/tmp/task.kedi", "job", occurrence=2)
    assert target is not None
    assert target.range.start.line == 8
    refs = references_at(source, uri, "/tmp/task.kedi", "job", True, occurrence=2)
    assert {ref.range.start.line for ref in refs} == {8, 10}
    profile = definition_at(source, uri, "/tmp/task.kedi", "worker", occurrence=3)
    assert profile is not None
    assert profile.range.start.line == 1
    profile_refs = references_at(source, uri, "/tmp/task.kedi", "worker", True, occurrence=3)
    assert {ref.range.start.line for ref in profile_refs} >= {1, 5, 8}
    assert not any(
        diag.code == "K-TASK-SUBAGENT" for diag in diagnostics_for(source, "/tmp/task.kedi")
    )


def test_native_task_completions_and_non_child_diagnostic() -> None:
    path = "/tmp/native-completion.kedi"
    directives = compute_completion("> ta", path, position_of("> ta", "ta", offset=2))
    assert next(item for item in directives if item.label == "task").insert_text == "task [job]: "
    assert next(item for item in directives if item.label == "await").insert_text == "await: "

    partial_task = PROFILES + "> task [job]: wo"
    child_items = compute_completion(
        partial_task, path, position_of(partial_task, "wo", occurrence=3, offset=2)
    )
    assert {item.label for item in child_items} == {"worker"}

    partial_await = PROFILES + "> task [job]: worker:\n    >> Review.\n> await [result]: jo"
    handle_items = compute_completion(
        partial_await, path, position_of(partial_await, "jo", occurrence=2, offset=2)
    )
    assert {item.label for item in handle_items} == {"job"}

    source = (
        PROFILES
        + "> profile: outsider:\n    > adapter: pydantic\n"
        + "> task [job]: outsider:\n    >> Review.\n> await: job\n"
    )
    assert any(diag.code == "K-TASK-NOT-CHILD" for diag in diagnostics_for(source, path))


def test_procedure_task_completion_uses_lexical_profile_and_local_handles() -> None:
    profiles = (
        PROFILES + "> profile: outsider:\n    > adapter: pydantic\n" + "> profile: alternate:\n"
        "    > adapter: pydantic\n"
        "    > subagent: outsider\n"
    )
    task_source = (
        profiles + "@review():\n    > use: alternate\n    > show: setup\n    > task [local_job]: ou"
    )
    path = "/tmp/proc-task-completion.kedi"
    task_line = "    > task [local_job]: ou"
    task_pos = position_of(task_source, task_line, offset=len(task_line))
    assert {item.label for item in compute_completion(task_source, path, task_pos)} == {"outsider"}

    await_source = (
        profiles + "@review():\n"
        "    > use: alternate\n"
        "    > task [local_job]: outsider:\n"
        "        >> Review.\n"
        "    > await: loc"
    )
    await_line = "    > await: loc"
    await_pos = position_of(await_source, await_line, offset=len(await_line))
    assert {item.label for item in compute_completion(await_source, path, await_pos)} == {
        "local_job"
    }
    invalid_source = "invalid source ???\n> task [job]: wo"
    invalid_pos = position_of(invalid_source, "wo", offset=2)
    assert compute_completion(invalid_source, path, invalid_pos) == []

    first_statement = PROFILES + "@review():\n    > task [job]: wo"
    first_line = "    > task [job]: wo"
    first_pos = position_of(first_statement, first_line, offset=len(first_line))
    assert {item.label for item in compute_completion(first_statement, path, first_pos)} == {
        "worker"
    }


def test_missing_source_spans_do_not_create_phantom_task_references() -> None:
    source = (
        PROFILES
        + "> task [top_job]: worker:\n    >> Review.\n"
        + "> await: top_job\n"
        + "@review():\n"
        "    > task [local_job]: worker:\n"
        "        >> Review.\n"
        "    > await: local_job\n"
    )
    program = parse_program(source)
    top_task = next(stmt for stmt in program.toplevel_body if isinstance(stmt, TaskStmt))
    top_await = next(stmt for stmt in program.toplevel_body if isinstance(stmt, AwaitStmt))
    proc_task = next(stmt for stmt in program.procedures[0].body if isinstance(stmt, TaskStmt))
    assert top_task.loc is not None and proc_task.loc is not None
    top_task.subagent_loc = None
    top_task.binder_loc = None
    proc_task.subagent_loc = None
    top_await.handle_loc = None
    profile_span = program.profile_defs["worker"].loc
    assert profile_span is not None
    symbol = SymbolRef("agent_profile", "worker", profile_span, None, profile_span)
    spans = _find_all_references(program, symbol, include_declaration=False)
    assert {top_task.loc.start_line, proc_task.loc.start_line}.isdisjoint(
        span.start_line for span in spans
    )
    references: list[Any] = []
    _collect_value_reference_candidates([top_await], "top_job", references)
    assert references == []
    assert _direct_statement_declarations(top_task, "top_job", None) == []
    virtual = compute_python_virtual_document(
        PROFILES + "> task [job]: worker:\n    >> Review.\n> await: job\n",
        "/tmp/await.kedi",
    )
    assert "__kedi_await(job)" in virtual["text"]
    assert "SubagentResult =" not in virtual["text"]


def test_lsp_ignores_task_and_await_keywords_for_definition() -> None:
    source = PROFILES + "> task [job]: worker:\n    >> Review.\n> await [result]: job\n"
    uri = "file:///tmp/task-keywords.kedi"
    path = "/tmp/task-keywords.kedi"
    assert definition_at(source, uri, path, "task", occurrence=1) is None
    assert definition_at(source, uri, path, "await", occurrence=1) is None


def test_task_and_await_inside_procedure_use_one_child() -> None:
    runtime, adapter = _runtime(
        "@review_change(change: str) -> int:\n"
        "    > task [job]: worker:\n"
        "        >> The value for <change> is [value: int].\n"
        "    > await [answer]: job\n"
        "    = `answer.output.value`\n"
        '= `review_change("test")`\n'
    )
    assert runtime.run_main() == 7
    assert len(adapter.invocations) == 1


def test_bare_await_inside_procedure_creates_no_result_binding() -> None:
    runtime, adapter = _runtime(
        "@review_change():\n"
        "    > task [job]: worker:\n"
        "        >> Review the change.\n"
        "    > await: job\n"
        "    = done\n"
        "= `review_change()`\n"
    )
    assert runtime.run_main() == "done"
    assert len(adapter.invocations) == 1


def test_sync_program_in_async_scope_uses_the_existing_owner_loop() -> None:
    runtime, adapter = _runtime(
        "> task [job]: worker:\n"
        "    >> The value is [value: int].\n"
        "> await [answer]: job\n"
        "= `answer.output.value`\n"
    )

    async def run() -> int:
        async with runtime.subagents(parent="manager"):
            return await asyncio.to_thread(runtime.run_main)

    assert asyncio.run(run()) == 7
    assert len(adapter.invocations) == 1


def test_sync_program_inside_python_scope_keeps_root_budget_and_identity() -> None:
    runtime, _ = _runtime("= `observe()`\n")
    runtime.run_budget = RunBudget(request_limit=3)
    observed: list[tuple[object, object, object]] = []

    def observe() -> str:
        observed.append(
            (
                runtime._request_budget.get(),
                runtime._invocation_id.get(),
                current_run_budget(),
            )
        )
        return "ok"

    runtime._initial_globals_env["observe"] = observe

    async def run() -> None:
        async with runtime.subagents(parent="manager"):
            request_budget = runtime._request_budget.get()
            invocation_id = runtime._invocation_id.get()
            ledger = current_run_budget()
            assert ledger is not None
            assert ledger.budget.request_limit == 3
            assert await asyncio.to_thread(runtime.run_main) == "ok"
            observed_request, observed_invocation, observed_ledger = observed[0]
            assert observed_request is request_budget
            assert observed_invocation is invocation_id
            assert observed_ledger is not None
            assert observed_ledger.parent.parent is ledger
            assert runtime._request_budget.get() is request_budget
            assert runtime._invocation_id.get() is invocation_id
            assert current_run_budget() is ledger
        assert current_run_budget() is None

    asyncio.run(run())


def test_sync_program_cannot_block_its_async_owner_loop() -> None:
    runtime, _ = _runtime("> task [job]: worker:\n    >> Review the change.\n> await: job\n")

    async def run() -> None:
        async with runtime.subagents(parent="manager"):
            with pytest.raises(KediExecutionError, match="event loop"):
                runtime.run_main()

    asyncio.run(run())


def test_lsp_resolves_procedure_task_symbols_and_reports_unknown_child() -> None:
    source = (
        PROFILES + "@review_change(change: str) -> int:\n"
        "    > task [job]: worker:\n"
        "        >> The value for <change> is [value: int].\n"
        "    > await [answer]: job\n"
        "    = `answer.output.value`\n"
    )
    path = "/tmp/native-task.kedi"
    uri = "file:///tmp/native-task.kedi"
    virtual = compute_python_virtual_document(source, path)
    ast.parse(virtual["text"])
    assert "job: SubagentTask" in virtual["text"]
    assert "answer: SubagentResult" in virtual["text"]
    assert "value: int" not in virtual["text"]
    handle = definition_at(source, uri, path, "job", occurrence=2)
    assert handle is not None
    assert handle.range.start.line == 9
    handle_declaration = definition_at(source, uri, path, "job", occurrence=1)
    assert handle_declaration is not None
    assert handle_declaration.range.start.line == 9
    result_declaration = definition_at(source, uri, path, "answer", occurrence=1)
    assert result_declaration is not None
    assert result_declaration.range.start.line == 11
    profile = definition_at(source, uri, path, "worker", occurrence=3)
    assert profile is not None
    assert profile.range.start.line == 1
    refs = references_at(source, uri, path, "worker", True, occurrence=3)
    assert {ref.range.start.line for ref in refs} >= {1, 5, 9}
    labels = {
        item.label
        for item in compute_completion(source, path, position_of(source, "= `answer.output"))
    }
    assert "job" in labels
    assert "answer" in labels
    unknown = source.replace("> task [job]: worker:", "> task [job]: stranger:")
    assert any(diag.code == "K-TASK-SUBAGENT" for diag in diagnostics_for(unknown, path))


@pytest.mark.parametrize(
    "source",
    [
        "> task [job]: worker:\n    >> First.\n    >> Second.\n",
        "> task [job]: worker:\n",
        "> task [job]: worker\n    >> First.\n",
        "> task [job]: worker:\n    `print(1)`\n",
        "> await: job:\n",
        "> await [answer]: job:\n",
        "> task [job]: worker:\n    task: Review the change.\n",
    ],
)
def test_invalid_task_or_await_syntax_is_rejected(source: str) -> None:
    with pytest.raises(KediParseError):
        parse_program(source)
