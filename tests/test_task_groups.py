from __future__ import annotations

import asyncio
import ast
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event
from types import SimpleNamespace
from typing import Any

import pytest

from kedi.engine import SequentialEngine, ThreadEngine
from kedi.agent_adapter import LangChainAdapter
from kedi.agent_adapter.execution import RunDeadlineExceeded, use_run_deadline
from kedi.errors import KediExecutionError
from kedi.lang import KediParseError, compile_program, parse_program
from kedi.lang.ast import TaskGroupStmt
from kedi.lang.compiler.task_groups import execute_task_group
from kedi.lsp.features.completion import compute_completion
from kedi.lsp.python_virtual.document import compute_python_virtual_document
from tests.lsp_helpers import definition_at, diagnostics_for, position_of
from tests.test_native_subagent_tasks import PROFILES
from tests.test_subagents import RecordingPydanticAdapter
from kedi.run_budget import RunBudget, RunBudgetExceeded, current_run_budget


TWO_TASKS = """
> task [first]: worker:
    >> First.
> task [second]: worker:
    >> Second.
"""


class GroupAdapter(RecordingPydanticAdapter):
    async def produce(self, template: str, **kwargs: Any) -> Any:
        if kwargs.get("output_schema") is not None:
            self.invocations.append((template, None, (), self.active_profile.subagents))
            return SimpleNamespace(value=7)
        return await super().produce(template, **kwargs)


def runtime_for(body: str, *, adapter: Any = None, engine: Any = None) -> Any:
    return compile_program(
        parse_program(PROFILES.replace("max_agents: 2", "max_agents: 8") + body),
        adapter=adapter or GroupAdapter(),
        engine=engine or SequentialEngine(),
    )


@pytest.mark.parametrize("workers", [0, 1, 4])
def test_ready_arm_runs_before_earlier_slow_child(workers: int) -> None:
    released = Event()
    seen: list[str] = []

    class Adapter(RecordingPydanticAdapter):
        async def invoke(self, *, prompt: str, **kwargs: Any) -> str:
            if prompt == "First.":
                while not released.is_set():
                    await asyncio.sleep(0.001)
            return prompt

    engine = ThreadEngine(workers) if workers else SequentialEngine()
    rt = runtime_for(
        TWO_TASKS
        + """
> task_group:
    > await [a]: first
    > process:
        `seen.append(a.task_summary)`
    > await [b]: second
    > process:
        `seen.append(b.task_summary)`
        `released.set()`
= done
""",
        adapter=Adapter(),
        engine=engine,
    )
    rt._initial_globals_env.update(seen=seen, released=released)
    try:
        with ThreadPoolExecutor(1) as executor:
            future = executor.submit(rt.run_main)
            try:
                assert future.result(timeout=5) == "done"
            finally:
                released.set()
        assert seen == ["Second.", "First."]
        assert not {"a", "b"}.intersection(rt._globals_env)
        assert rt._subagent_coordinator.active_runs == 0
    finally:
        engine.shutdown()


@pytest.mark.parametrize("from_worker", [False, True])
def test_ready_processing_runs_concurrently_without_local_declaration_leaks(
    from_worker: bool,
) -> None:
    barrier = Barrier(2)
    seen: list[tuple[str, str]] = []
    engine = ThreadEngine(2)
    rt = runtime_for(
        TWO_TASKS
        + """
> task_group:
    > await [answer]: first
    > process:
        [label] = first
        @local() -> str:
            = <label>
        ~Local(value: str)
        `barrier.wait(timeout=5)`
        [result] = <local()>
        `seen.append((label, result))`
    > await [answer]: second
    > process:
        [label] = second
        @local() -> str:
            = <label>
        ~Local(value: int)
        `barrier.wait(timeout=5)`
        [result] = <local()>
        `seen.append((label, result))`
= done
""",
        engine=engine,
    )
    rt._initial_globals_env.update(barrier=barrier, seen=seen)
    try:
        result = (
            engine.schedule((), rt.run_main).result(timeout=10) if from_worker else rt.run_main()
        )
        assert result == "done"
        assert sorted(seen) == [("first", "first"), ("second", "second")]
        assert "local" not in rt._procedures
        assert all("Local" not in scope for scope in rt._type_env_stack)
        assert "label" not in rt._globals_env
    finally:
        engine.shutdown()


def test_nested_groups_and_models_finish_with_one_worker() -> None:
    engine = ThreadEngine(1)
    adapter = GroupAdapter()
    rt = runtime_for(
        """
> task [first]: worker:
    >> First.
> task_group:
    > await [a]: first
    > process:
        > task [nested]: worker:
            >> Nested.
        > task_group:
            > await [b]: nested
            > process:
                >> The answer is [value: int].

                > show: <`seen.append(value)`>
= done
""",
        engine=engine,
        adapter=adapter,
    )
    seen: list[int] = []
    rt._initial_globals_env["seen"] = seen
    try:
        with ThreadPoolExecutor(1) as executor:
            assert executor.submit(rt.run_main).result(timeout=5) == "done"
        assert seen == [7]
        assert not rt._outstanding
    finally:
        engine.shutdown()


def test_worker_dispatches_late_ready_sibling_while_an_arm_is_waiting() -> None:
    entered, released = Event(), Event()
    engine = ThreadEngine(2)

    class Adapter(GroupAdapter):
        async def invoke(self, *, prompt: str, **kwargs: Any) -> str:
            if prompt == "Second.":
                while not entered.is_set():
                    await asyncio.sleep(0.001)
            return prompt

    def wait_for_sibling() -> None:
        entered.set()
        assert released.wait(5), "ready sibling did not start"

    rt = runtime_for(
        TWO_TASKS
        + """
> task_group:
    > await: first
    > process:
        `wait_for_sibling()`
    > await: second
    > process:
        `released.set()`
""",
        adapter=Adapter(),
        engine=engine,
    )
    rt._initial_globals_env.update(wait_for_sibling=wait_for_sibling, released=released)
    try:
        engine.schedule((), rt.run_main).result(timeout=10)
        assert rt._subagent_coordinator.active_runs == 0
    finally:
        released.set()
        engine.shutdown()


def test_child_failure_cancels_processing_even_while_dispatcher_is_on_a_worker() -> None:
    entered, released = Event(), Event()
    engine = ThreadEngine(1)
    unexpected: list[str] = []

    class Adapter(GroupAdapter):
        async def invoke(self, *, prompt: str, **kwargs: Any) -> str:
            if prompt == "Second.":
                while not entered.is_set():
                    await asyncio.sleep(0.001)
                asyncio.get_running_loop().call_later(0.05, released.set)
                raise ValueError("late child failed")
            return prompt

    def block() -> None:
        entered.set()
        assert released.wait(5)

    rt = runtime_for(
        TWO_TASKS
        + """
> task_group:
    > await: first
    > process:
        `block()`
        `unexpected.append("continued after sibling failure")`
    > await: second
""",
        adapter=Adapter(),
        engine=engine,
    )
    rt._initial_globals_env.update(block=block, unexpected=unexpected)
    try:
        with pytest.raises(ValueError, match="late child failed"):
            engine.schedule((), rt.run_main).result(timeout=10)
        assert unexpected == []
        assert rt._subagent_coordinator.active_runs == 0
    finally:
        released.set()
        engine.shutdown()


def test_group_without_process_observes_children() -> None:
    rt = runtime_for(
        TWO_TASKS
        + """
> task_group:
    > await: first
    > await [unused]: second
= done
"""
    )
    assert rt.run_main() == "done"
    assert "unused" not in rt._globals_env


@pytest.mark.parametrize("inside_procedure", [False, True])
def test_group_can_live_in_procedure_and_inherits_parent_profile(inside_procedure: bool) -> None:
    body = """> task [first]: worker:
    >> First.
> task_group:
    > await [answer]: first
    > process:
        >> The value is [value: int].
= done
"""
    if inside_procedure:
        body = (
            "@run():\n"
            + "".join("    " + line + "\n" for line in body.splitlines())
            + "= <run()>\n"
        )
    rt = runtime_for(body)
    assert rt.run_main() == "done"


@pytest.mark.parametrize(
    "statement",
    [
        "= no",
        "> if: `True`:\n    = no",
        "> loop [n]: `[1]`:\n    = no",
        "> loop [n]: `[1]`:\n    > show: <n>\n> map:\n    = no",
    ],
)
def test_direct_process_returns_are_parse_errors(statement: str) -> None:
    body = "\n".join("        " + line for line in statement.splitlines())
    with pytest.raises(KediParseError, match="Direct returns"):
        parse_program("> task_group:\n    > await: job\n    > process:\n" + body + "\n")


@pytest.mark.parametrize(
    "source",
    [
        "> process:\n    > show: no\n",
        "> task_group:\n    > process:\n        > show: no\n",
        "> task_group:\n    > await: job\n    > process:\n        > show: a\n    > process:\n        > show: b\n",
        "> task_group:\n    > show: no\n",
    ],
)
def test_invalid_group_structure_rejected(source: str) -> None:
    with pytest.raises(KediParseError):
        parse_program(source)


def test_all_handles_validated_before_any_process_side_effect() -> None:
    seen: list[str] = []
    rt = runtime_for("""
> task [first]: worker:
    >> First.
[bad] = not-a-task
> task_group:
    > await: first
    > process:
        `seen.append("unexpected")`
    > await: bad
""")
    rt._initial_globals_env["seen"] = seen
    with pytest.raises(TypeError, match="expects a subagent task"):
        rt.run_main()
    assert seen == []


def test_process_error_cancels_slow_sibling_and_keeps_original_error() -> None:
    cancelled = Event()

    class Adapter(RecordingPydanticAdapter):
        async def invoke(self, *, prompt: str, **kwargs: Any) -> str:
            if prompt == "First.":
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()
            return prompt

    rt = runtime_for(
        TWO_TASKS
        + """
> task_group:
    > await: first
    > process:
        > show: never
    > await: second
    > process:
        `1 / 0`
""",
        adapter=Adapter(),
    )
    with pytest.raises(KediExecutionError, match="division by zero"):
        rt.run_main()
    assert cancelled.is_set()
    assert rt._subagent_coordinator.active_runs == 0


def test_unawaited_process_descendant_fails_at_group_boundary() -> None:
    rt = runtime_for("""
> task [first]: worker:
    >> First.
> task_group:
    > await: first
    > process:
        > task [leaked]: worker:
            >> Leaked.
`seen.append("after group")`
""")
    seen: list[str] = []
    rt._initial_globals_env["seen"] = seen
    with pytest.raises(KediExecutionError, match="was not awaited"):
        rt.run_main()
    assert not seen
    assert rt._subagent_coordinator.active_runs == 0


def test_child_failure_cancels_sibling_without_running_process() -> None:
    cancelled = Event()

    class Adapter(GroupAdapter):
        async def invoke(self, *, prompt: str, **kwargs: Any) -> str:
            if prompt == "Second.":
                raise ValueError("child failed")
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
            return prompt

    rt = runtime_for(
        TWO_TASKS
        + """
> task_group:
    > await: first
    > process:
        `seen.append("unexpected")`
    > await: second
""",
        adapter=Adapter(),
    )
    seen: list[str] = []
    rt._initial_globals_env["seen"] = seen
    with pytest.raises(ValueError, match="child failed"):
        rt.run_main()
    assert cancelled.is_set()
    assert seen == []


def test_deadline_expiry_still_drains_children() -> None:
    cleaned = Event()

    class Adapter(GroupAdapter):
        async def invoke(self, **kwargs: Any) -> str:
            try:
                await asyncio.Event().wait()
            finally:
                await asyncio.sleep(0.01)
                cleaned.set()
            return "unreachable"

    rt = runtime_for(
        """
> task [job]: worker:
    >> Wait.
> task_group:
    > await: job
""",
        adapter=Adapter(),
    )
    with use_run_deadline(time.monotonic() + 0.15):
        with pytest.raises(KediExecutionError) as error:
            rt.run_main()
    assert isinstance(error.value.original, RunDeadlineExceeded)
    assert cleaned.is_set()
    assert rt._subagent_coordinator.active_runs == 0


def test_group_joins_unconsumed_model_work_and_uses_parent_config() -> None:
    seen: list[str | None] = []

    class Adapter(GroupAdapter):
        async def produce(self, template: str, **kwargs: Any) -> Any:
            await asyncio.sleep(0.01)
            seen.append(self.active_profile.system)
            return await super().produce(template, **kwargs)

    engine = ThreadEngine(2)
    rt = runtime_for(
        """
> system: Parent configuration.
> task [job]: worker:
    >> First.
> task_group:
    > await: job
    > process:
        >> The value is [value: int].
`assert len(seen) == 1`
""",
        adapter=Adapter(),
        engine=engine,
    )
    rt._initial_globals_env["seen"] = seen
    try:
        rt.run_main()
        assert seen == ["Parent configuration."]
        assert not rt._outstanding
    finally:
        engine.shutdown()


def test_multiple_process_waiters_share_one_result() -> None:
    results: list[Any] = []
    engine = ThreadEngine(2)
    rt = runtime_for(
        """
> task [job]: worker:
    >> The answer is [value: int].
> task_group:
    > await [a]: job
    > process:
        `results.append(a)`
    > await [b]: job
    > process:
        `results.append(b)`
""",
        engine=engine,
    )
    rt._initial_globals_env["results"] = results
    try:
        rt.run_main()
        assert results[0] is results[1]
        assert results[0].output.value == 7
    finally:
        engine.shutdown()


def test_group_lsp_scope_and_virtual_python() -> None:
    source = (
        PROFILES
        + TWO_TASKS
        + """
> task_group:
    > await [alpha]: first
    > process:
        [local] = <alpha>
        > show: <alpha>
    > await [beta]: second
    > process:
        > show: <beta>
> show: done
"""
    )
    path, uri = "/tmp/group.kedi", "file:///tmp/group.kedi"
    definition = definition_at(source, uri, path, "alpha", occurrence=2)
    assert definition is not None
    assert definition.range.start.line == position_of(source, "[alpha]").line
    first = position_of(source, "<alpha>", occurrence=2, offset=1)
    second = position_of(source, "<beta>", offset=1)
    outside = position_of(source, "done", offset=1)
    assert "alpha" in {item.label for item in compute_completion(source, path, first)}
    assert "alpha" not in {item.label for item in compute_completion(source, path, second)}
    assert "beta" not in {item.label for item in compute_completion(source, path, outside)}
    virtual = compute_python_virtual_document(source, path)
    tree = ast.parse(virtual["text"])
    processes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name.startswith("__kedi_process_")
    ]
    assert len(processes) == 2
    assert "__kedi_await(first)" in virtual["text"]
    assert not [d for d in diagnostics_for(source, path) if d.code == "K-TASK-NOT-CHILD"]


def test_process_local_return_allowed_but_nested_process_return_rejected() -> None:
    rt = runtime_for("""
> task [job]: worker:
    >> First.
> task_group:
    > await: job
    > process:
        @local() -> int:
            = `4`
        > show: <local()>
""")
    rt.run_main()
    with pytest.raises(KediParseError, match="Direct returns"):
        parse_program("""
> task_group:
    > await: job
    > process:
        @local():
            > task_group:
                > await: nested
                > process:
                    = invalid
""")


def test_process_budget_is_a_child_of_the_existing_ledger() -> None:
    ledgers: list[Any] = []
    rt = runtime_for("""
> task [job]: worker:
    >> First.
> task_group:
    > await: job
    > process:
        `capture_budget()`
        > budget:
            tool_attempt_limit: 1
        `spend()`
        `spend()`
""")
    rt.run_budget = RunBudget(tool_attempt_limit=2)

    def capture_budget() -> None:
        ledgers.append(current_run_budget())

    def spend() -> None:
        ledger = current_run_budget()
        assert ledger is not None
        ledger.admit("tool_attempt")

    rt._initial_globals_env.update(capture_budget=capture_budget, spend=spend)
    with pytest.raises(RunBudgetExceeded):
        rt.run_main()
    assert len(ledgers) == 1
    assert ledgers[0].spent("tool_attempt") == 1
    assert ledgers[0].parent.spent("tool_attempt") == 1
    assert current_run_budget() is None


def test_async_owner_cancellation_joins_running_sync_process() -> None:
    entered, release, finished = Event(), Event(), Event()
    engine = ThreadEngine(1)
    rt = runtime_for("", engine=engine)
    group = parse_program("""
> task_group:
    > await: job
    > process:
        `block()`
        `unexpected.append("continued after cancellation")`
""").toplevel_body[0]
    assert isinstance(group, TaskGroupStmt)
    unexpected: list[str] = []

    def block() -> None:
        entered.set()
        assert release.wait(timeout=5)
        finished.set()

    async def scenario() -> None:
        work = None

        async def release_later() -> None:
            await asyncio.sleep(0.05)
            assert not finished.is_set()
            release.set()

        with pytest.raises(ValueError, match="parent failed"):
            async with rt.subagents(parent="manager") as agents:
                job = await agents.start("worker", task="First.")
                profile = rt._evaluate_profile_definition(rt._profile_registry["manager"])
                work = asyncio.create_task(
                    asyncio.to_thread(
                        execute_task_group,
                        rt,
                        group,
                        {"job": job, "block": block, "unexpected": unexpected},
                        profile,
                    )
                )
                assert await asyncio.to_thread(entered.wait, 5)
                releaser = asyncio.create_task(release_later())
                raise ValueError("parent failed")
        await releaser
        assert finished.is_set()
        assert unexpected == []
        assert work is not None
        with pytest.raises(KediExecutionError, match="cancelled"):
            await work
        assert rt._subagent_coordinator.active_runs == 0

    try:
        asyncio.run(scenario())
    finally:
        release.set()
        engine.shutdown()


def test_processes_are_not_retried_by_model_job_manager() -> None:
    from kedi.engine import JobManagerEngine

    engine = JobManagerEngine(max_workers=1)
    attempts: list[int] = []

    def fail() -> None:
        attempts.append(1)
        raise ConnectionError("side effect failed")

    try:
        with pytest.raises(ConnectionError):
            engine.schedule_control(fail).result(timeout=5)
        assert attempts == [1]
    finally:
        engine.shutdown()


@pytest.mark.parametrize("adapter_name", ["pydantic", "langchain"])
def test_both_adapters_use_identical_group_lifecycle(adapter_name: str) -> None:
    class Adapter(LangChainAdapter):
        def __init__(self) -> None:
            super().__init__(model=object())

        async def invoke(self, *, prompt: str, **kwargs: Any) -> str:
            return prompt

        async def produce(self, template: str, **kwargs: Any) -> Any:
            return SimpleNamespace(value=7)

    engine = ThreadEngine(2)
    source = (
        PROFILES
        + TWO_TASKS
        + """
> task_group:
    > await [a]: first
    > process:
        >> The result is [value: int].
        > show: <`seen.append(value)`>
    > await [b]: second
    > process:
        >> The result is [value: int].
        > show: <`seen.append(value)`>
"""
    )
    rt = compile_program(
        parse_program(source.replace("adapter: pydantic", f"adapter: {adapter_name}")),
        adapter=Adapter() if adapter_name == "langchain" else GroupAdapter(),
        engine=engine,
    )
    seen: list[int] = []
    rt._initial_globals_env["seen"] = seen
    try:
        rt.run_main()
        assert seen == [7, 7]
        assert rt._subagent_coordinator.active_runs == 0
        assert not rt._outstanding
    finally:
        engine.shutdown()


@pytest.mark.parametrize("grouped", [False, True])
def test_nested_child_can_schedule_work_back_on_a_saturated_engine(grouped: bool) -> None:
    engine = ThreadEngine(1)

    class Adapter(GroupAdapter):
        async def invoke(self, *, prompt: str, **kwargs: Any) -> str:
            if prompt == "Nested.":
                # Child tools may re-enter the engine from the async owner's thread.
                future = engine.schedule((), lambda: "nested tool completed")
                return await asyncio.to_thread(future.result, 2)
            return prompt

    source = """
> task [first]: worker:
    >> First.
> task_group:
    > await: first
    > process:
        > task [nested]: worker:
            >> Nested.
        > task_group:
            > await [answer]: nested
            > process:
                > show: <`answer.task_summary`>
"""
    if not grouped:
        source = source.replace(
            "        > task_group:\n            > await [answer]: nested\n            > process:\n                > show: <`answer.task_summary`>",
            "        > await [answer]: nested\n        > show: <`answer.task_summary`>",
        )
    rt = runtime_for(
        source,
        adapter=Adapter(),
        engine=engine,
    )
    try:
        rt.run_main()
        assert rt._subagent_coordinator.active_runs == 0
    finally:
        engine.shutdown()
