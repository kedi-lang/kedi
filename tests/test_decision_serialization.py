from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass, replace
from typing import Any

import pytest
from pydantic import BaseModel

from kedi import InteractiveSession, SessionLoadError
from kedi._interactive import persistence, state
from kedi.agent_adapter.adapters.langchain.messages import (
    _capture_langchain_typesafe_decision_evidence,
)
from kedi.decisions import (
    CapturedDecisionEvidence,
    DecisionInfo,
    DecisionInput,
    DecisionOutput,
    DecisionRecord,
    DecisionReference,
    collect_decision_evidence,
    decision_info_for_env,
    fingerprint_value,
)
from kedi.engine import SequentialEngine
from kedi.lang.compiler.environment import BindingMetadata, ScopeFrame
from tests.test_decisions import DecisionAdapter, _info_from_metadata


def _record() -> DecisionRecord:
    return _info_from_metadata({}, "approved")._record


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("version",), 2),
        (("source",), None),
        (("evidence",), []),
        (("inputs",), {}),
        (("outputs",), None),
        (("evidence", "payload_encoding"), None),
        (("evidence", "provider"), 1),
        (("evidence", "model"), False),
        (("source", "path"), 1),
        (("source", "line"), True),
        (("source", "line"), "1"),
        (("source", "column"), True),
        (("source", "column"), "1"),
        (("call_id",), None),
        (("prompt_fingerprint",), None),
        (("inputs_complete",), 1),
        (("inputs",), [None]),
        (("inputs",), [{"name": 1}]),
        (("inputs",), [{"name": "x", "fingerprint": 1}]),
        (("inputs",), [{"name": "x", "supported": 1}]),
        (("outputs",), [None]),
        (("outputs",), [{"field_path": "x"}]),
        (("outputs",), [{"field_path": [1]}]),
        (("outputs",), [{"field_path": ["x"], "fingerprint": 1}]),
        (("outputs",), [{"field_path": ["x"], "supported": 1}]),
    ],
)
def test_record_rejects_corrupt_fields(path: tuple[str, ...], value: Any) -> None:
    data = _record().to_data()
    target = data
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        DecisionRecord.from_data(data)


@pytest.mark.parametrize(
    "encoded",
    [
        None,
        [],
        [0],
        ["unknown"],
        ["bool", 1],
        ["int", "oops"],
        ["float", "inf"],
        ["map", [None]],
        ["map", [[1]]],
        ["map", [[["str", "x"], ["none"]], [["str", "x"], ["none"]]]],
        ["map", [[["list", []], ["none"]]]],
    ],
)
def test_evidence_rejects_corrupt_encoding(encoded: Any) -> None:
    with pytest.raises(ValueError):
        CapturedDecisionEvidence.from_encoded(provider="typesafe", model=None, encoded=encoded)


@pytest.mark.parametrize("value", [float("inf"), float("nan"), object()])
def test_evidence_rejects_unsupported_values(value: Any) -> None:
    with pytest.raises((ValueError, TypeError)):
        CapturedDecisionEvidence.create(provider="typesafe", model=None, metadata={"x": value})


def test_evidence_rejects_cycles_and_roundtrips_nested_values() -> None:
    cyclic: list[Any] = []
    cyclic.append(cyclic)
    with pytest.raises(ValueError, match="cycles"):
        CapturedDecisionEvidence.create(provider="typesafe", model=None, metadata={"x": cyclic})
    metadata = {"x": (None, True, 1, 0.5, "text", [False])}
    evidence = CapturedDecisionEvidence.create(provider="typesafe", model=None, metadata=metadata)
    assert evidence.payload()["x"] == (None, True, 1, 0.5, "text", (False,))
    assert (
        DecisionRecord.from_data(replace(_record(), evidence=evidence).to_data()).evidence
        == evidence
    )


def test_fingerprints_cover_value_types_and_reject_cycles() -> None:
    class Model(BaseModel):
        x: int

    @dataclass
    class Data:
        x: int

    values = [None, b"bytes", (1, "x"), float("inf"), Model(x=1), Data(1)]
    for value in values:
        assert fingerprint_value(value) is not None
        assert fingerprint_value(value) == fingerprint_value(value)
    assert fingerprint_value(Model(x=1)) != fingerprint_value(Model(x=2))
    cyclic: list[Any] = []
    cyclic.append(cyclic)
    assert fingerprint_value(cyclic) is None


def test_sparse_evidence_and_public_properties() -> None:
    info = _info_from_metadata({"answers": {"bad": 1, "other": {"path": ["other"]}}}, "approved")
    assert info.answers == ()
    assert info.source.path == "test.kedi"
    assert info.prompt_fingerprint
    assert info.inputs == {}
    assert info.comparator is None
    assert info.distribution is None
    assert info.rubric is None
    assert _info_from_metadata({}, "approved").answers == ()
    record = replace(_record(), inputs=(DecisionInput("x", fingerprint_value(True), True),))
    with pytest.raises(ValueError, match="does not support stable matching"):
        DecisionInfo(record, ("approved",)).matches_inputs(x=[object()])


@pytest.mark.parametrize("name", ["", None, 1])
def test_decision_lookup_rejects_invalid_names(name: Any) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        decision_info_for_env({}, name)


def test_decision_lookup_handles_absent_or_unsupported_outputs() -> None:
    env = ScopeFrame()
    for outputs in [(), (DecisionOutput(("approved",), None, False),)]:
        record = replace(_record(), outputs=outputs)
        env.declare(
            "approved",
            object(),
            BindingMetadata(decision=DecisionReference.completed(record, ("approved",))),
        )
        assert decision_info_for_env(env, "approved") is not None
    assert decision_info_for_env({"plain": 1}, "plain") is None


@pytest.mark.parametrize("result", [None, {"messages": [object()]}])
def test_langchain_capture_ignores_nonresponse_values(result: Any) -> None:
    with collect_decision_evidence() as evidence:
        _capture_langchain_typesafe_decision_evidence(result)
    assert evidence == []


def test_session_inspection_rejects_busy_and_closed_session() -> None:
    session = InteractiveSession(adapter=DecisionAdapter(), engine=SequentialEngine())
    session._execute_lock.acquire()
    try:
        with pytest.raises(RuntimeError, match="executing"):
            session.decision_info("x")
    finally:
        session._execute_lock.release()
    session.close()
    with pytest.raises(RuntimeError, match="closed"):
        session.decision_info("x")


@pytest.mark.parametrize(
    "table", [[], {1: {}}, {"x": None}, {"x": {}}, {"wrong": _record().to_data()}]
)
def test_session_rejects_invalid_record_tables(table: Any) -> None:
    with pytest.raises(SessionLoadError):
        state._decode_decision_records(table)


@pytest.mark.parametrize(
    "reference", [1, {"record_id": 1, "field_path": []}, {"record_id": "x", "field_path": [1]}]
)
def test_session_rejects_invalid_binding_references(reference: Any) -> None:
    metadata = {
        "approved": {
            "kind": "value",
            "type_contract": None,
            "assignable": True,
            "synthetic": False,
            "decision": reference,
        }
    }
    with pytest.raises(SessionLoadError):
        state._decode_binding_metadata(metadata, {"approved": True})


def test_session_dump_rejects_failed_and_conflicting_decisions() -> None:
    env = ScopeFrame()
    failed: Future[DecisionRecord | None] = Future()
    failed.set_exception(RuntimeError("failed"))
    empty: Future[DecisionRecord | None] = Future()
    empty.set_result(None)
    env.declare("invalid", True, BindingMetadata(decision=object()))
    env.declare("failed", True, BindingMetadata(decision=DecisionReference(failed, ("failed",))))
    env.declare("empty", True, BindingMetadata(decision=DecisionReference(empty, ("empty",))))
    record = _record()
    env.declare(
        "approved",
        True,
        BindingMetadata(decision=DecisionReference.completed(record, ("approved",))),
    )
    issues: list[str] = []
    encoded = state._encode_binding_metadata(env, issues, decision_records={record.call_id: {}})
    assert encoded["empty"]["decision"] is None
    assert len(issues) == 3
    assert "invalid decision reference" in issues[0]
    assert "failed decision evidence" in issues[1]
    assert "conflicting payloads" in issues[2]


def test_handle_encoder_requires_consistent_record_table() -> None:
    info = _info_from_metadata({}, "approved")
    for table in [None, {info.call_id: {}}]:
        encoder = persistence.SessionValueEncoder(kedi_models={}, decision_records=table)
        with pytest.raises(persistence.UnsupportedSessionValue):
            encoder.encode(info, path="saved")


@pytest.mark.parametrize(
    "record_id,field_path", [(1, []), ("x", "approved"), ("x", [1]), ("missing", ["approved"])]
)
def test_handle_decoder_rejects_invalid_references(record_id: Any, field_path: Any) -> None:
    decoder = persistence.SessionValueDecoder(kedi_models={})
    with pytest.raises(SessionLoadError):
        decoder.decode(
            {"$kedi": "decision_info", "record_id": record_id, "field_path": field_path},
            path="saved",
        )
