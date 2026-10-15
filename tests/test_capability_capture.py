from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier, Lock

import pytest

from kedi.lang import compile_program, parse_program
from kedi.lang.procedure_introspection import get_procedure_def
from tests.mock_adapter import MockAdapter


class CapabilityAdapter(MockAdapter):
    capabilities = replace(
        MockAdapter.capabilities,
        supports_capabilities=True,
        supports_deferred_tools=True,
    )


@pytest.mark.parametrize("wrapper", ["search", "bundle"])
def test_main_evaluates_capability_at_use_not_during_procedure_preparation(wrapper):
    evaluations = []

    def defer_search():
        evaluations.append("defer")
        return False

    runtime = compile_program(
        parse_program(
            "> capability: search:\n"
            "  > defer: `defer_search()`\n"
            "  > instructions: Search guidance.\n"
            "> profile: bundle:\n  > use: search\n"
            f"> use: {wrapper}\n"
            "@review() -> str:\n"
            "  >> The review is [review: str].\n  = <review>\n"
            "= <review()>\n"
        ),
        adapter=CapabilityAdapter(),
        runtime_globals={"defer_search": defer_search},
    )
    try:
        assert evaluations == []
        runtime.run_main()
        assert evaluations == ["defer"]
        runtime.run_main()
        assert evaluations == ["defer", "defer"]
    finally:
        runtime.artifacts.close()


def test_capture_execution_contexts_do_not_share_resolved_values():
    from kedi.agent_profile import AgentProfile
    from kedi.agent_profile.capture import CapturedAgentProfile, profile_capture_scope

    barrier = Barrier(2)
    calls = []
    lock = Lock()

    def evaluate(env):
        with lock:
            calls.append(env["instruction"])
        return AgentProfile(system=env["instruction"])

    capture = CapturedAgentProfile(
        AgentProfile.empty(), evaluate, {"instruction": "Definition rules."}
    )

    def execute(instruction):
        with profile_capture_scope((capture,)):
            capture.apply(AgentProfile.empty(), {"instruction": instruction})
            barrier.wait(timeout=5)
            return capture.resolve().system

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(execute, "First rules.")
        second = pool.submit(execute, "Second rules.")
        assert first.result(timeout=10) == "First rules."
        assert second.result(timeout=10) == "Second rules."
    assert sorted(calls) == ["First rules.", "Second rules."]
    assert capture.resolve().system == "Definition rules."
    assert len(calls) == 3


def test_failed_capture_scope_does_not_poison_next_run():
    from kedi.agent_profile import AgentProfile
    from kedi.agent_profile.capture import CapturedAgentProfile, profile_capture_scope

    attempts = []

    def evaluate(env):
        attempts.append(env["valid"])
        if not env["valid"]:
            raise ValueError("Invalid capability configuration")
        return AgentProfile(system="Valid rules.")

    capture = CapturedAgentProfile(AgentProfile.empty(), evaluate, {"valid": True})
    with pytest.raises(ValueError, match="Invalid capability configuration"):
        with profile_capture_scope((capture,)):
            capture.apply(AgentProfile.empty(), {"valid": False})
    with profile_capture_scope((capture,)):
        assert capture.resolve().system == "Valid rules."
    assert attempts == [False, True]


def test_capability_after_last_procedure_is_not_evaluated_during_capture():
    evaluations = []
    runtime = compile_program(
        parse_program(
            "> capability: search:\n"
            "  > defer: `record() is None`\n"
            "@unrelated() -> str:\n  = done\n"
            "> use: search\n= done\n"
        ),
        adapter=CapabilityAdapter(),
        runtime_globals={"record": lambda: evaluations.append("defer")},
    )
    try:
        assert evaluations == []
        runtime.run_main()
        assert evaluations == ["defer"]
    finally:
        runtime.artifacts.close()


def test_exported_siblings_share_one_definition_site_evaluation():
    evaluations = []
    lock = Lock()

    def record():
        with lock:
            evaluations.append("defer")
        return False

    adapter = CapabilityAdapter()
    runtime = compile_program(
        parse_program(
            "> capability: search:\n"
            "  > defer: `record()`\n  > instructions: Search guidance.\n"
            "> instructions: Definition rules.\n> use: search\n"
            "@first() -> str:\n  >> The result is [result: str].\n  = <result>\n"
            "@second() -> str:\n  >> The result is [result: str].\n  = <result>\n"
            "> instructions: Later caller rules.\n"
        ),
        adapter=adapter,
        runtime_globals={"record": record},
    )
    try:
        assert evaluations == []
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [
                pool.submit(runtime._invoke_procedure, runtime._get_proc(name), [])
                for name in ["first", "second"] * 4
            ]
            assert all(future.result(timeout=10) for future in futures)
        assert evaluations == ["defer"]
        assert len(adapter.instruction_calls) == 8
        assert all("Definition rules." in call for call in adapter.instruction_calls)
        assert all("Search guidance." in call for call in adapter.instruction_calls)
        assert all("Later caller rules." not in call for call in adapter.instruction_calls)
    finally:
        runtime.artifacts.close()


def test_program_reentry_resolves_new_defer_value_without_mutating_old_callable():
    selected = {"defer": False}
    evaluations = []

    def mode():
        evaluations.append(selected["defer"])
        return selected["defer"]

    runtime = compile_program(
        parse_program(
            "> capability: search:\n  > defer: `mode()`\n"
            "  > instructions: Search guidance.\n"
            "> use: search\n@review() -> str:\n  = done\n= done\n"
        ),
        adapter=CapabilityAdapter(),
        runtime_globals={"mode": mode},
    )
    try:
        assert evaluations == []
        runtime.run_main()
        old = get_procedure_def(runtime._get_proc("review"))
        assert old.captured_agent_profile.capabilities[0].deferred is False
        selected["defer"] = True
        runtime.run_main()
        new = get_procedure_def(runtime._get_proc("review"))
        assert new.captured_agent_profile.capabilities[0].deferred is True
        assert old.captured_agent_profile.capabilities[0].deferred is False
        assert evaluations == [False, True]
    finally:
        runtime.artifacts.close()


def test_pending_capture_preserves_order_and_instruction_reset():
    evaluations = []

    def instruction():
        evaluations.append("instructions")
        return "Final rules."

    adapter = CapabilityAdapter()
    runtime = compile_program(
        parse_program(
            "> capability: search:\n  > defer: `record() is not None`\n"
            "  > instructions: Suppressed guidance.\n"
            "> use: search\n> instructions: <`instruction()`>\n"
            "@review() -> str:\n  >> The review is [review: str].\n  = <review>\n"
        ),
        adapter=adapter,
        runtime_globals={"record": lambda: evaluations.append("defer"), "instruction": instruction},
    )
    try:
        assert evaluations == []
        runtime._invoke_procedure(runtime._get_proc("review"), [])
        assert evaluations == ["defer", "instructions"]
        assert "Final rules." in adapter.instruction_calls[0]
        assert "Suppressed guidance." not in adapter.instruction_calls[0]
    finally:
        runtime.artifacts.close()


@pytest.mark.parametrize("call_before_use", [False, True])
def test_forward_call_and_declaration_share_one_use_evaluation(call_before_use):
    evaluations = []
    adapter = CapabilityAdapter()
    call = "> show: <review()>\n"
    runtime = compile_program(
        parse_program(
            "> capability: search:\n  > defer: `record() is not None`\n"
            "  > instructions: Search guidance.\n"
            + (call if call_before_use else "")
            + "> use: search\n"
            + ("" if call_before_use else call)
            + "@review() -> str:\n  >> The review is [review: str].\n  = <review>\n"
            "= <review()>\n"
        ),
        adapter=adapter,
        runtime_globals={"record": lambda: evaluations.append("defer")},
    )
    try:
        assert evaluations == []
        runtime.run_main()
        assert evaluations == ["defer"]
        runtime.run_main()
        assert evaluations == ["defer", "defer"]
        assert all("Search guidance." in call for call in adapter.instruction_calls)
    finally:
        runtime.artifacts.close()


@pytest.mark.parametrize(
    "use",
    ["> use: combined\n", "> use:\n  left\n  right\n"],
)
def test_defer_expression_runs_once_per_use_graph_not_once_per_diamond_path(use):
    evaluations = []
    runtime = compile_program(
        parse_program(
            "> capability: shared:\n  > defer: `record() is not None`\n"
            "> capability: left:\n  > use: shared\n"
            "> capability: right:\n  > use: shared\n"
            "> capability: combined:\n  > use: left\n  > use: right\n" + use + use + "= done\n"
        ),
        adapter=CapabilityAdapter(),
        runtime_globals={"record": lambda: evaluations.append("defer")},
    )
    try:
        runtime.run_main()
        assert evaluations == ["defer", "defer"]
    finally:
        runtime.artifacts.close()


def test_runtime_capability_expression_can_read_prior_top_level_binding():
    runtime = compile_program(
        parse_program(
            "> capability: search:\n  > defer: `mode`\n"
            "[mode: bool] = `True`\n> use: search\n"
            "@review() -> str:\n  = done\n= done\n"
        ),
        adapter=CapabilityAdapter(),
    )
    try:
        runtime.run_main()
        proc = get_procedure_def(runtime._get_proc("review"))
        assert proc.captured_agent_profile.capabilities[0].deferred is True
    finally:
        runtime.artifacts.close()


def test_direct_export_uses_definition_environment_not_arguments():
    adapter = CapabilityAdapter()
    runtime = compile_program(
        parse_program(
            "> capability: search:\n  > defer: `mode`\n"
            "  > instructions: Search guidance.\n> use: search\n"
            "@review(mode: bool) -> str:\n"
            "  >> The review is [review: str].\n  = <review>\n"
        ),
        adapter=adapter,
        runtime_globals={"mode": False},
    )
    try:
        runtime._invoke_procedure(runtime._get_proc("review"), [True])
        assert "Search guidance." in adapter.instruction_calls[0]
    finally:
        runtime.artifacts.close()


def test_import_diamond_evaluates_shared_capability_once(tmp_path):
    (tmp_path / "shared.kedi").write_text(
        "> capability: shared:\n  > defer: `record() is not None`\n> export:\n  shared\n"
    )
    for name in ("left", "right"):
        (tmp_path / f"{name}.kedi").write_text(
            f"> import: shared\n> capability: {name}:\n  > use: shared\n> export:\n  {name}\n"
        )
    evaluations = []
    runtime = compile_program(
        parse_program(
            "> import: left\n> import: right\n> use:\n  left\n  right\n= done\n",
            source_path=str(tmp_path / "main.kedi"),
        ),
        adapter=CapabilityAdapter(),
        runtime_globals={"record": lambda: evaluations.append("defer")},
    )
    try:
        assert evaluations == []
        runtime.run_main()
        assert evaluations == ["defer"]
    finally:
        runtime.artifacts.close()


def test_failed_direct_export_does_not_leak_a_type_scope():
    from kedi.errors import KediExecutionError

    selected = {"mode": "invalid"}
    runtime = compile_program(
        parse_program(
            "> capability: search:\n  > defer: `selected['mode']`\n"
            "> use: search\n@review() -> str:\n  = done\n",
            source_path="capture.kedi",
        ),
        adapter=CapabilityAdapter(),
        runtime_globals={"selected": selected},
    )
    try:
        before = list(runtime._type_env_stack)
        review = runtime._get_proc("review")
        with pytest.raises(KediExecutionError, match="defer must evaluate to a bool") as error:
            review()
        assert "capture.kedi" in str(error.value)
        assert "line 2" in str(error.value)
        assert runtime._type_env_stack == before
        selected["mode"] = False
        assert review() == "done"
        assert runtime._type_env_stack == before
    finally:
        runtime.artifacts.close()
