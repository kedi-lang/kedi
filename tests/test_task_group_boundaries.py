from __future__ import annotations

from concurrent.futures import CancelledError, Future
from typing import Any

import pytest

from kedi.agent_adapter.subagents.groups import ProcessingRegion, check_process_cancelled
from kedi.agent_adapter.subagents.native import NativeSubagentOwner, SubagentTask
from kedi.agent_profile import AgentProfile
from kedi.core.scopes import ProcessingScope
from kedi.engine import ThreadEngine
from kedi.errors import KediExecutionError
from kedi.explain import explain_program
from kedi.lang import KediParseError, compile_program, parse_program
from kedi.lang.ast import AwaitStmt, Program, TaskGroupArm, TaskGroupStmt
from kedi.lang.compiler.task_groups import execute_task_group
from kedi.lsp.features.completion import compute_completion
from kedi.lsp.features.navigation import _collect_loop_binder_refs
from kedi.lsp.features.scope import _declaration_in_scope_path
from kedi.lsp.features.symbols import SymbolRef
from kedi.lsp.python_virtual.document import compute_python_virtual_document
from kedi.lsp.python_virtual.scope import _all_procedures as virtual_procedures
from tests.lsp_helpers import definition_at, hover_text_at, position_of, references_at
from tests.test_task_groups import TWO_TASKS, GroupAdapter, runtime_for


def test_regions_cancel_late_registrations_and_ignore_foreign_children(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rt = runtime_for("")
    owner = NativeSubagentOwner(rt)
    region = ProcessingRegion(owner)
    cancelled: list[str] = []
    monkeypatch.setattr(rt, "_cancel_subagent", lambda run_id: cancelled.append(run_id))
    region.add_child(SubagentTask(NativeSubagentOwner(rt), "foreign"))
    region.cancel()
    region.add_child(SubagentTask(owner, "late"))
    future: Future[Any] = Future()
    region.add_future(future)
    assert future.cancelled()
    assert cancelled == ["late"]
    with pytest.raises(CancelledError):
        with region.activate():
            pytest.fail("cancelled scope entered")
    check_process_cancelled()


def test_owner_rejects_new_group_during_close() -> None:
    rt = runtime_for("")
    owner = NativeSubagentOwner(rt)
    token = rt._native_subagent_owner.set(owner)
    try:
        owner._closing = True
        with pytest.raises(KediExecutionError, match="closing"):
            owner.register_group(Future(), lambda: None)
    finally:
        rt._native_subagent_owner.reset(token)


def test_processing_scope_setters_do_not_replace_parent_declarations() -> None:
    rt = runtime_for("")
    original_procs, original_types = rt._procedures, rt._type_env_stack
    token = rt._processing_scope.set(ProcessingScope({}, []))
    try:
        rt._procedures = {"local": lambda: None}
        rt._type_env_stack = [{"Local": int}]
        assert "local" in rt._procedures
        assert rt._type_env_stack == [{"Local": int}]
    finally:
        rt._processing_scope.reset(token)
    assert rt._procedures is original_procs
    assert rt._type_env_stack is original_types


def test_group_requires_an_owner_and_handles_from_that_owner() -> None:
    rt = runtime_for("")
    group = TaskGroupStmt(arms=[TaskGroupArm(wait=AwaitStmt(handle="job"))])
    owner = NativeSubagentOwner(rt)
    job = SubagentTask(NativeSubagentOwner(rt), "foreign")
    with pytest.raises(RuntimeError, match="active Kedi invocation"):
        execute_task_group(rt, group, {"job": job}, AgentProfile())
    token = rt._native_subagent_owner.set(owner)
    try:
        with pytest.raises(KediExecutionError, match="another invocation"):
            execute_task_group(rt, group, {"job": job}, AgentProfile())
    finally:
        rt._native_subagent_owner.reset(token)


@pytest.mark.parametrize("child_fails", [False, True])
def test_registration_failure_joins_already_registered_waiters(
    monkeypatch: pytest.MonkeyPatch, child_fails: bool
) -> None:
    original = NativeSubagentOwner.submit
    waits = 0

    def submit(self, factory, **kwargs):
        nonlocal waits
        if getattr(factory, "__name__", None) == "wait":
            waits += 1
            if waits == 2:
                raise RuntimeError("wait registration failed")
        return original(self, factory, **kwargs)

    class Adapter(GroupAdapter):
        async def invoke(self, **kwargs: Any) -> str:
            if child_fails:
                raise ValueError("child failed too")
            return "done"

    monkeypatch.setattr(NativeSubagentOwner, "submit", submit)
    rt = runtime_for(
        TWO_TASKS
        + """
> task_group:
    > await: first
    > await: second
""",
        adapter=Adapter(),
    )
    with pytest.raises(RuntimeError, match="wait registration failed"):
        rt.run_main()
    assert rt._subagent_coordinator.active_runs == 0


def test_ready_success_does_not_mask_sibling_failure_with_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = NativeSubagentOwner.submit

    def submit(self, factory, **kwargs):
        future = original(self, factory, **kwargs)
        if getattr(factory, "__name__", None) == "wait":
            future.exception(timeout=5)
        return future

    class Adapter(GroupAdapter):
        async def invoke(self, *, prompt: str, **kwargs: Any) -> str:
            if prompt == "Second.":
                raise ValueError("second child failed")
            return prompt

    monkeypatch.setattr(NativeSubagentOwner, "submit", submit)
    seen: list[str] = []
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
    rt._initial_globals_env["seen"] = seen
    with pytest.raises(ValueError, match="second child failed"):
        rt.run_main()
    assert not seen
    assert rt._subagent_coordinator.active_runs == 0


def test_unconsumed_process_model_failure_surfaces_at_join() -> None:
    class Adapter(GroupAdapter):
        async def produce(self, template: str, **kwargs: Any) -> Any:
            raise ValueError("unconsumed model failed")

    engine = ThreadEngine(1)
    rt = runtime_for(
        """
> task [job]: worker:
    >> Review.
> task_group:
    > await: job
    > process:
        >> The result is [value: int].
""",
        adapter=Adapter(),
        engine=engine,
    )
    try:
        with pytest.raises(ValueError, match="unconsumed model failed"):
            rt.run_main()
        assert not rt._outstanding
    finally:
        engine.shutdown()


def test_empty_ast_group_is_rejected_before_execution() -> None:
    with pytest.raises(KediParseError, match="at least one await"):
        compile_program(Program(toplevel_body=[TaskGroupStmt()]))


def test_comments_hover_navigation_and_virtual_types_in_processing() -> None:
    source = """
> loop [item]: `[1]`:
    > task_group:
        # Attached to the following arm.
        > await [answer]: job
        > process:
            @local(value: list[int]) -> int:
                = `len(value)`
            ~Result(value: int)
            [record: Result] = `Result(value=1)`
            > show: <local(`[item]`)>
            > show: <item>
        > await: second
"""
    path, uri = "/tmp/group-tools.kedi", "file:///tmp/group-tools.kedi"
    assert "Task group" in hover_text_at(source, path, "task_group")
    assert definition_at(source, uri, path, "process") is None
    definition = definition_at(source, uri, path, "local", occurrence=2)
    assert definition is not None
    assert definition.range.start.line == 6
    assert definition_at(source, uri, path, "Result", occurrence=2) is not None
    inside = {
        item.label
        for item in compute_completion(source, path, position_of(source, "<local", offset=1))
    }
    outside = {
        item.label for item in compute_completion(source, path, position_of(source, "second"))
    }
    assert {"local", "Result"}.issubset(inside)
    assert {"local", "Result"}.isdisjoint(outside)
    refs = references_at(source, uri, path, "item", include_declaration=False, occurrence=1)
    assert refs
    loop = parse_program(source).toplevel_body[0]
    collected = []
    _collect_loop_binder_refs(
        parse_program(source),
        SymbolRef("loop_binder", "item", loop.binder_loc, None, loop.binder_loc),
        collected,
    )
    assert collected
    virtual = compute_python_virtual_document(source, path)
    assert "class Result" in virtual["text"]
    assert "def local" in virtual["text"]
    # A cursor on the group header or a spanless synthetic arm must not see its locals.
    group = parse_program(source).toplevel_body[0].body[0]
    assert isinstance(group, TaskGroupStmt) and group.loc is not None
    assert _declaration_in_scope_path([group], "answer", group.loc.start_byte + 1, None) is None
    group.arms[0].loc = None
    assert _declaration_in_scope_path([group], "answer", group.loc.end_byte - 1, None) is None


def test_explain_and_nested_procedure_virtual_scope_include_process_models() -> None:
    source = """
> adapter: pydantic
> model: parent-model
@run():
    > task_group:
        > await: job
        > process:
            @inner() -> list[int]:
                = `[1]`
            >> The value is [value: list[int]].
"""
    program = parse_program(source)
    report = explain_program(program)
    assert len(report.calls) == 1
    assert report.calls[0].scope == "procedure run (process job)"
    assert report.calls[0].model.value == "parent-model"
    virtual = compute_python_virtual_document(source, "/tmp/group-proc.kedi")
    assert "def inner" in virtual["text"]
    assert [proc.name for proc in virtual_procedures(program.procedures)] == ["run", "inner"]
