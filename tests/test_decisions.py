from __future__ import annotations

import gc
import json
import time
import weakref
from collections.abc import Sequence
from concurrent.futures import CancelledError, Future
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from kedi import InteractiveSession, SessionDumpError, SessionLoadError, decision_info
from kedi._interactive import persistence
from kedi.agent_adapter.adapters.langchain.messages import (
    _capture_langchain_typesafe_decision_evidence,
)
from kedi.agent_adapter.adapters.pydantic.messages import (
    _capture_typesafe_decision_evidence,
)
from kedi.decisions import (
    CapturedDecisionEvidence,
    DecisionInfo,
    DecisionOutput,
    DecisionRecord,
    DecisionReference,
    DecisionSource,
    capture_decision_evidence,
    collect_decision_evidence,
    decision_info_for_env,
    fingerprint_text,
    fingerprint_value,
)
from kedi.engine import Engine, Job, JobManagerEngine, SequentialEngine, ThreadEngine
from kedi.errors import KediExecutionError
from kedi.lang import compile_program, parse_program
from kedi.lang.compiler.environment import BindingMetadata
from tests.mock_adapter import MockAdapter


class DecisionAdapter(MockAdapter):
    def produce_sync(
        self,
        *,
        template: str,
        output_schema: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        result = super().produce_sync(
            template=template,
            output_schema=output_schema,
            **kwargs,
        )
        answers = {
            key: {
                "type": "noul",
                "probability": 0.92,
                "output_kind": "boolean",
                "path": [key],
            }
            for key in (output_schema or {})
        }
        capture_decision_evidence(
            provider="typesafe",
            model="jev-test",
            metadata={
                "schema_version": 1,
                "answers": answers,
                "boolean_threshold": 0.85,
                "boolean_comparator": ">",
                "request_fingerprint": "request-test",
                "state_fingerprint": "state-test",
                "questions_fingerprint": "questions-test",
                "config_fingerprint": "config-test",
                "usage": {"input_tokens": 4, "output_tokens": 1},
            },
        )
        return result


class RacingDecisionAdapter(DecisionAdapter):
    def produce_sync(
        self,
        *,
        template: str,
        output_schema: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        if "slow" in template:
            time.sleep(0.05)
        result = MockAdapter.produce_sync(
            self,
            template=template,
            output_schema=output_schema,
            **kwargs,
        )
        probability = 0.2 if "slow" in template else 0.8
        capture_decision_evidence(
            provider="typesafe",
            model="jev-race",
            metadata={
                "schema_version": 1,
                "answers": {
                    key: {
                        "type": "noul",
                        "probability": probability,
                        "output_kind": "boolean",
                        "path": [key],
                    }
                    for key in (output_schema or {})
                },
                "boolean_threshold": 0.5,
                "boolean_comparator": ">",
            },
        )
        return result


class CancelledEngine(Engine):
    parallel = True

    def schedule(self, deps: Sequence[Future], job: Job) -> Future:
        del deps, job
        future: Future[Any] = Future()
        future.cancel()
        return future


class FailingDecisionAdapter(DecisionAdapter):
    def produce_sync(
        self,
        *,
        template: str,
        output_schema: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        del template, output_schema, kwargs
        raise RuntimeError("provider failed")


def _run(
    source: str,
    *,
    parallel: bool = False,
    adapter: MockAdapter | None = None,
):
    engine = ThreadEngine(max_workers=2) if parallel else SequentialEngine()
    runtime = compile_program(
        parse_program(source),
        adapter=adapter or DecisionAdapter(),
        engine=engine,
    )
    try:
        result = runtime.run_main()
        return runtime, result
    finally:
        engine.shutdown()


def _info_from_metadata(metadata: dict[str, Any], field: str) -> DecisionInfo:
    output_fingerprint = fingerprint_value(True)
    assert output_fingerprint is not None
    record = DecisionRecord(
        call_id="direct-test",
        source=DecisionSource(path="test.kedi", line=1, column=1),
        prompt_fingerprint=fingerprint_text("test"),
        inputs=(),
        inputs_complete=True,
        outputs=(DecisionOutput((field,), output_fingerprint, True),),
        evidence=CapturedDecisionEvidence.create(
            provider="typesafe",
            model="jev-test",
            metadata=metadata,
        ),
    )
    return DecisionInfo(record, (field,))


@pytest.mark.parametrize("parallel", [False, True])
def test_template_output_exposes_immutable_decision_evidence(parallel: bool) -> None:
    runtime, result = _run(
        '[ticket: str] = `"charged twice"`\n'
        ">> Does <ticket> describe a duplicate charge? [duplicate_charge: bool]\n"
        "= <duplicate_charge>\n",
        parallel=parallel,
    )

    assert result == "True"
    info = decision_info_for_env(runtime._globals_env, "duplicate_charge")
    assert info is not None
    assert info.model == "jev-test"
    assert info.provider == "typesafe"
    assert info.field_path == ("duplicate_charge",)
    assert info.probability == 0.92
    assert info.threshold == 0.85
    assert info.comparator == ">"
    assert info.confidence is None
    assert info.request_fingerprint == "request-test"
    assert info.state_fingerprint == "state-test"
    assert info.criteria_fingerprint == "questions-test"
    assert info.config_fingerprint == "config-test"
    assert info.matches_inputs(ticket="charged twice") is True
    assert info.matches_inputs(ticket="charged once") is False
    with pytest.raises(TypeError):
        cast(Any, info.answers)[0]["probability"] = 0.1


def test_multiple_outputs_share_call_record_but_keep_field_evidence() -> None:
    runtime, _ = _run(
        ">> Evaluate both [approved: bool] and [urgent: bool]\n= `approved and urgent`\n"
    )

    approved = decision_info_for_env(runtime._globals_env, "approved")
    urgent = decision_info_for_env(runtime._globals_env, "urgent")
    assert approved is not None and urgent is not None
    assert approved.call_id == urgent.call_id
    assert approved.field_path == ("approved",)
    assert urgent.field_path == ("urgent",)
    assert approved.answers[0]["path"] == ("approved",)
    assert urgent.answers[0]["path"] == ("urgent",)


def test_input_reassignment_preserves_history_and_output_overwrite_clears_reference() -> None:
    session = InteractiveSession(adapter=DecisionAdapter(), engine=SequentialEngine())
    session.execute('[ticket: str] = `"charged twice"`')
    session.execute(">> Does <ticket> describe a duplicate charge? [duplicate_charge: bool]")
    historical = session.decision_info("duplicate_charge")
    assert historical is not None

    session.execute('[ticket] := `"charged once"`')
    assert historical.matches_inputs(ticket="charged twice") is True
    assert historical.matches_inputs(ticket="charged once") is False
    current = session.decision_info("duplicate_charge")
    assert current is not None
    assert current is historical or current.call_id == historical.call_id

    session.execute("[duplicate_charge] := `False`")
    assert session.decision_info("duplicate_charge") is None


def test_failed_overwrite_preserves_decision_and_python_rebind_clears_it() -> None:
    session = InteractiveSession(adapter=DecisionAdapter(), engine=SequentialEngine())
    session.execute(">> Decide [approved: bool]")
    original = session.decision_info("approved")
    assert original is not None

    with pytest.raises(TypeError):
        session.execute('[approved] := `"invalid"`')
    assert session.decision_info("approved") is not None

    session.execute("`print(approved)`")
    assert session.decision_info("approved") is not None
    session.execute("`approved = False`")
    assert session.decision_info("approved") is None


def test_in_place_output_mutation_invalidates_current_binding_view() -> None:
    session = InteractiveSession(adapter=DecisionAdapter(), engine=SequentialEngine())
    session.execute(">> Produce [values_list: list[str]]")
    assert session.decision_info("values_list") is not None
    session.execute('`values_list.append("changed")`')
    assert session.decision_info("values_list") is None


def test_parallel_overwrite_does_not_reattach_completed_record() -> None:
    runtime, result = _run(
        ">> Decide [approved: bool]\n[approved] := `False`\n= `approved`\n",
        parallel=True,
    )
    assert result is False
    assert decision_info_for_env(runtime._globals_env, "approved") is None


def test_parallel_calls_keep_call_local_evidence_when_completion_order_reverses() -> None:
    adapter = RacingDecisionAdapter()
    runtime, _ = _run(
        ">> slow decision [slow_result: bool]\n"
        ">> fast decision [fast_result: bool]\n"
        "= `slow_result and fast_result`\n",
        parallel=True,
        adapter=adapter,
    )

    slow = decision_info_for_env(runtime._globals_env, "slow_result")
    fast = decision_info_for_env(runtime._globals_env, "fast_result")
    assert slow is not None and fast is not None
    assert slow.call_id != fast.call_id
    assert slow.probability == 0.2
    assert fast.probability == 0.8


def test_cancelled_parallel_job_settles_decision_reference() -> None:
    runtime = compile_program(
        parse_program(">> Is this approved? [approved: bool]\n"),
        adapter=DecisionAdapter(),
        engine=CancelledEngine(),
    )

    with pytest.raises(CancelledError):
        runtime.run_main()

    assert decision_info_for_env(runtime._globals_env, "approved") is None


def test_failed_parallel_job_settles_decision_reference() -> None:
    engine = ThreadEngine(max_workers=1)
    runtime = compile_program(
        parse_program(">> Is this approved? [approved: bool]\n"),
        adapter=FailingDecisionAdapter(),
        engine=engine,
    )
    try:
        with pytest.raises(KediExecutionError, match="provider failed"):
            runtime.run_main()
        assert decision_info_for_env(runtime._globals_env, "approved") is None
    finally:
        engine.shutdown()


def test_procedure_shadow_keeps_decision_records_lexical() -> None:
    runtime, result = _run(
        "[infos: list] = `[]`\n"
        ">> Root decision [approved: bool]\n"
        '[root_info: Any] = `__import__("kedi").decision_info("approved")`\n'
        "@judge(label: str) -> str:\n"
        "  >> Decide for <label> [approved: bool]\n"
        '  [saved: Any] = `infos.append(__import__("kedi").decision_info("approved"))`\n'
        "  = done\n\n"
        "[ignored: str] = <judge(local)>\n"
        "= `len(infos)`\n"
    )

    assert result == 1
    root = decision_info_for_env(runtime._globals_env, "approved")
    local = runtime._globals_env["infos"][0]
    assert isinstance(root, DecisionInfo)
    assert isinstance(local, DecisionInfo)
    assert runtime._globals_env["root_info"].call_id == root.call_id
    assert local.call_id != root.call_id


def test_nested_loop_map_keeps_iteration_decisions_distinct() -> None:
    runtime, result = _run(
        "[infos: list] = `[]`\n"
        "> loop [outer]: `[1, 2]`:\n"
        "  > loop [inner]: `[10, 20]`:\n"
        "    >> Decide <outer>-<inner> [approved: bool]\n"
        "  > map:\n"
        '    [saved: Any] = `infos.append(__import__("kedi").decision_info("approved"))`\n'
        "= `len(infos)`\n"
    )

    assert result == 4
    infos = runtime._globals_env["infos"]
    assert all(isinstance(info, DecisionInfo) for info in infos)
    assert len({info.call_id for info in infos}) == 4
    assert {info.probability for info in infos} == {0.92}


def test_lexical_decision_helper_reads_the_inner_control_flow_binding() -> None:
    _, result = _run(
        "[seen: list[float]] = `[]`\n"
        "> if: `True`:\n"
        "  >> Decide [approved: bool]\n"
        '  [seen] := `seen + [__import__("kedi").decision_info("approved").probability]`\n'
        "= `seen`\n"
    )
    assert result == [0.92]


def test_active_python_helper_uses_lexical_environment() -> None:
    source = (
        "```\nfrom kedi import decision_info\n```\n"
        ">> Decide [approved: bool]\n"
        '= `decision_info("approved").probability`\n'
    )
    _, result = _run(source)
    assert result == 0.92
    with pytest.raises(RuntimeError, match="active Kedi Python execution"):
        decision_info("approved")


def test_non_typesafe_output_has_no_decision_info() -> None:
    runtime = compile_program(
        parse_program(">> Fill [answer]\n= <answer>\n"),
        adapter=MockAdapter(),
        engine=SequentialEngine(),
    )
    runtime.run_main()
    assert decision_info_for_env(runtime._globals_env, "answer") is None
    with pytest.raises(NameError, match="missing"):
        decision_info_for_env(runtime._globals_env, "missing")


def test_historical_handle_does_not_retain_runtime_or_adapter() -> None:
    adapter = DecisionAdapter()
    runtime, _ = _run(">> Decide [approved: bool]\n= <approved>\n", adapter=adapter)
    info = decision_info_for_env(runtime._globals_env, "approved")
    assert info is not None
    runtime_ref = weakref.ref(runtime)
    adapter_ref = weakref.ref(adapter)

    del runtime
    del adapter
    gc.collect()

    assert runtime_ref() is None
    assert adapter_ref() is None
    assert info.probability == 0.92


def test_session_dump_roundtrips_binding_and_saved_handle(tmp_path: Path) -> None:
    snapshot = tmp_path / "decision.kedi-state"
    session = InteractiveSession(adapter=DecisionAdapter(), engine=SequentialEngine())
    session.execute('[ticket: str] = `"charged twice"`')
    session.execute(">> Does <ticket> describe a duplicate charge? [duplicate_charge: bool]")
    session.execute('[saved: Any] = `__import__("kedi").decision_info("duplicate_charge")`')
    session.dump(snapshot)

    payload = json.loads(snapshot.read_text())["payload"]
    assert len(payload["decision_records"]) == 1

    restored_adapter = DecisionAdapter()
    restored = InteractiveSession.load(
        snapshot,
        adapter=restored_adapter,
        engine=SequentialEngine(),
    )
    assert restored_adapter.request_calls == []
    info = restored.decision_info("duplicate_charge")
    assert info is not None
    assert info.probability == 0.92
    assert info.matches_inputs(ticket="charged twice") is True
    saved = restored._env["saved"]
    assert saved.call_id == info.call_id


def test_session_load_rejects_dangling_decision_reference(tmp_path: Path) -> None:
    snapshot = tmp_path / "decision.kedi-state"
    session = InteractiveSession(adapter=DecisionAdapter(), engine=SequentialEngine())
    session.execute(">> Decide [approved: bool]")
    session.dump(snapshot)

    payload = persistence.read_snapshot(snapshot)
    payload["binding_metadata"]["approved"]["decision"]["record_id"] = "missing"
    persistence.write_snapshot(snapshot, payload)

    with pytest.raises(SessionLoadError, match="unknown decision output"):
        InteractiveSession.load(
            snapshot,
            adapter=DecisionAdapter(),
            engine=SequentialEngine(),
        )


def test_session_dump_rejects_pending_decision_without_writing(tmp_path: Path) -> None:
    snapshot = tmp_path / "pending.kedi-state"
    session = InteractiveSession(adapter=DecisionAdapter(), engine=SequentialEngine())
    pending: Future[DecisionRecord | None] = Future()
    cast(Any, session._env).declare(
        "pending",
        True,
        BindingMetadata(decision=DecisionReference(pending, ("pending",))),
    )

    with pytest.raises(SessionDumpError, match="pending decision evidence"):
        session.dump(snapshot)
    assert not snapshot.exists()


def test_session_load_accepts_snapshot_without_decision_schema(tmp_path: Path) -> None:
    snapshot = tmp_path / "legacy.kedi-state"
    session = InteractiveSession(adapter=DecisionAdapter(), engine=SequentialEngine())
    session.execute("[value: int] = `42`")
    session.dump(snapshot)

    payload = persistence.read_snapshot(snapshot)
    payload.pop("decision_records")
    for metadata in payload["binding_metadata"].values():
        metadata.pop("decision", None)
    persistence.write_snapshot(snapshot, payload)

    restored = InteractiveSession.load(
        snapshot,
        adapter=DecisionAdapter(),
        engine=SequentialEngine(),
    )
    assert restored._env["value"] == 42
    assert restored.decision_info("value") is None


def test_matches_inputs_rejects_wrong_names_and_unsupported_values() -> None:
    runtime, _ = _run(
        "[ticket: list] = `[object()]`\n>> Evaluate <ticket> [approved: bool]\n= <approved>\n"
    )
    info = decision_info_for_env(runtime._globals_env, "approved")
    assert info is not None
    with pytest.raises(ValueError, match="does not support stable matching"):
        info.matches_inputs(ticket=[object()])
    with pytest.raises(ValueError, match="do not match"):
        info.matches_inputs(other="value")


def test_matches_inputs_preserves_types_and_sequence_order_without_requests() -> None:
    adapter = DecisionAdapter()
    session = InteractiveSession(adapter=adapter, engine=SequentialEngine())
    session.execute("[value: Any] = `True`")
    session.execute(">> Evaluate <value> [approved: bool]")
    info = session.decision_info("approved")
    assert info is not None
    request_count = len(adapter.request_calls)

    assert info.matches_inputs(value=True) is True
    assert info.matches_inputs(value=1) is False
    assert info.matches_inputs(value=1.0) is False
    assert session.decision_info("approved") is not None
    assert len(adapter.request_calls) == request_count

    session.execute("[ordered: list[int]] = `[1, 2]`")
    session.execute(">> Evaluate <ordered> [ordered_result: bool]")
    ordered = session.decision_info("ordered_result")
    assert ordered is not None
    assert ordered.matches_inputs(ordered=[1, 2]) is True
    assert ordered.matches_inputs(ordered=[2, 1]) is False


def test_computed_substitution_refuses_partial_input_match() -> None:
    session = InteractiveSession(adapter=DecisionAdapter(), engine=SequentialEngine())
    session.execute(">> Evaluate <`'computed'`> [approved: bool]")
    info = session.decision_info("approved")
    assert info is not None
    with pytest.raises(ValueError, match="computed substitutions"):
        info.matches_inputs()


def test_unsupported_input_matching_does_not_call_repr_again() -> None:
    class ReprCounter:
        calls = 0

        def __repr__(self) -> str:
            type(self).calls += 1
            return "counter"

    value = ReprCounter()
    session = InteractiveSession(adapter=DecisionAdapter(), engine=SequentialEngine())
    cast(Any, session._env).declare("value", [value])
    session.execute(">> Evaluate <value> [approved: bool]")
    info = session.decision_info("approved")
    assert info is not None
    calls_after_request = ReprCounter.calls

    with pytest.raises(ValueError, match="does not support stable matching"):
        info.matches_inputs(value=[value])
    assert ReprCounter.calls == calls_after_request


def test_nested_and_multilabel_answers_remain_separate() -> None:
    info = _info_from_metadata(
        {
            "answers": {
                "risk": {
                    "type": "noul",
                    "probability": 0.7,
                    "output_kind": "probability",
                    "path": ["assessment", "risk"],
                },
                "label_a": {
                    "type": "noul",
                    "probability": 0.8,
                    "output_kind": "boolean",
                    "path": ["assessment", "labels"],
                    "label": "a",
                },
                "label_b": {
                    "type": "noul",
                    "probability": 0.3,
                    "output_kind": "boolean",
                    "path": ["assessment", "labels"],
                    "label": "b",
                },
            },
            "boolean_threshold": 0.85,
            "boolean_comparator": ">",
        },
        "assessment",
    )

    assert len(info.answers) == 3
    assert {answer.get("label") for answer in info.answers} == {None, "a", "b"}
    assert info.probability is None
    assert info.threshold is None


def test_probability_choice_and_score_evidence_keep_distinct_semantics() -> None:
    probability = _info_from_metadata(
        {
            "answers": {
                "probability": {
                    "type": "noul",
                    "probability": 0.72,
                    "output_kind": "probability",
                    "path": ["probability"],
                }
            },
            "boolean_threshold": 0.85,
            "boolean_comparator": ">",
        },
        "probability",
    )
    choice = _info_from_metadata(
        {
            "answers": {
                "route": {
                    "type": "choice",
                    "choice": "allow",
                    "confidence": 0.8,
                    "probabilities": {"allow": 0.9, "deny": 0.1},
                    "path": ["route"],
                }
            }
        },
        "route",
    )
    score = _info_from_metadata(
        {
            "answers": {
                "quality": {
                    "type": "score",
                    "score": 1.4,
                    "confidence": 0.75,
                    "probabilities": {0: 0.2, 1: 0.8},
                    "legend": {0: "bad", 1: "good"},
                    "path": ["quality"],
                }
            }
        },
        "quality",
    )

    assert probability.probability == 0.72
    assert probability.threshold is None
    assert probability.confidence is None
    assert choice.probability is None
    assert choice.confidence == 0.8
    assert choice.distribution == {"allow": 0.9, "deny": 0.1}
    assert score.confidence == 0.75
    assert score.distribution == {0: 0.2, 1: 0.8}
    assert score.rubric == {0: "bad", 1: "good"}


def test_pydantic_capture_uses_final_typesafe_response_only() -> None:
    result = SimpleNamespace(
        new_messages=lambda: [
            SimpleNamespace(provider_details={"typesafe": {"answers": {"old": {}}}}),
            SimpleNamespace(
                provider_details={"typesafe": {"answers": {"final": {}}}},
                model_name="jev-final",
            ),
        ]
    )
    with collect_decision_evidence() as evidence:
        _capture_typesafe_decision_evidence(result)
    assert len(evidence) == 1
    assert evidence[0].model == "jev-final"
    assert "final" in evidence[0].payload()["answers"]


def test_langchain_capture_uses_final_typesafe_response_only() -> None:
    result = {
        "messages": [
            SimpleNamespace(response_metadata={"typesafe": {"answers": {"old": {}}}}),
            SimpleNamespace(
                response_metadata={
                    "model_name": "jev-final",
                    "typesafe": {"answers": {"final": {}}},
                }
            ),
        ]
    }
    with collect_decision_evidence() as evidence:
        _capture_langchain_typesafe_decision_evidence(result)
    assert len(evidence) == 1
    assert evidence[0].model == "jev-final"
    assert "final" in evidence[0].payload()["answers"]


def test_langchain_capture_ignores_typesafe_evidence_from_input_history() -> None:
    historical = SimpleNamespace(
        type="ai",
        content="old result",
        response_metadata={
            "model_name": "jev-old",
            "typesafe": {"answers": {"old": {}}},
        },
    )
    prompt = SimpleNamespace(type="human", content="new request")
    current = SimpleNamespace(
        type="ai",
        content="ordinary response",
        response_metadata={},
    )
    result = {"messages": [historical, prompt, current]}

    with collect_decision_evidence() as evidence:
        _capture_langchain_typesafe_decision_evidence(
            result,
            input_messages=[historical, prompt],
        )

    assert evidence == []


def test_langchain_capture_reads_typesafe_evidence_from_current_turn() -> None:
    historical = SimpleNamespace(
        type="ai",
        content="old result",
        response_metadata={
            "model_name": "jev-old",
            "typesafe": {"answers": {"old": {}}},
        },
    )
    prompt = SimpleNamespace(type="human", content="new request")
    current = SimpleNamespace(
        type="ai",
        content="new result",
        response_metadata={
            "model_name": "jev-current",
            "typesafe": {"answers": {"current": {}}},
        },
    )
    result = {"messages": [historical, prompt, current]}

    with collect_decision_evidence() as evidence:
        _capture_langchain_typesafe_decision_evidence(
            result,
            input_messages=[historical, prompt],
        )

    assert len(evidence) == 1
    assert evidence[0].model == "jev-current"
    assert "current" in evidence[0].payload()["answers"]


def test_evidence_codec_preserves_numeric_distribution_keys() -> None:
    evidence = CapturedDecisionEvidence.create(
        provider="typesafe",
        model="jev-test",
        metadata={"answers": {"quality": {"probabilities": {0: 0.25, 1: 0.75}}}},
    )
    probabilities = evidence.payload()["answers"]["quality"]["probabilities"]
    assert probabilities == {0: 0.25, 1: 0.75}
    with pytest.raises(TypeError):
        probabilities[0] = 1.0  # type: ignore[index]


def test_retryable_model_failure_keeps_decision_pending_until_success() -> None:
    class RetryAdapter(DecisionAdapter):
        attempts = 0

        def produce_sync(self, **kwargs: Any) -> Any:
            self.attempts += 1
            if self.attempts == 1:
                raise TimeoutError("temporary provider timeout")
            return super().produce_sync(**kwargs)

    adapter = RetryAdapter()
    engine = JobManagerEngine(max_workers=1, max_attempts=2, base_delay=0)
    runtime = compile_program(
        parse_program(">> Decide [approved: bool]\n= <approved>\n"),
        adapter=adapter,
        engine=engine,
    )
    try:
        assert runtime.run_main() == "True"
        info = decision_info_for_env(runtime._globals_env, "approved")
        assert info is not None
        assert info.probability == 0.92
        assert adapter.attempts == 2
    finally:
        engine.shutdown()


def test_repeated_input_with_changed_identity_disables_stable_matching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kedi.core import model_calls

    original = model_calls._snapshot_prompt_input
    count = 0

    def changing_snapshot(name: str, value: Any) -> Any:
        nonlocal count
        count += 1
        return original(name, value + str(count))

    monkeypatch.setattr(model_calls, "_snapshot_prompt_input", changing_snapshot)
    runtime, _ = _run(
        '[value: str] = `"input"`\n>> Compare <value> with <value> [approved: bool]\n= <approved>\n'
    )
    info = decision_info_for_env(runtime._globals_env, "approved")
    assert info is not None
    with pytest.raises(ValueError, match="computed substitutions"):
        info.matches_inputs(value="input")
