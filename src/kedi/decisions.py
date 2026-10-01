"""Shared decision criteria and immutable provenance for Kedi output bindings."""

from __future__ import annotations

import base64
import hashlib
import json
import math
from collections.abc import Iterator, Mapping
from concurrent.futures import Future
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, fields, is_dataclass
from threading import Lock
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

from pydantic import BaseModel

if TYPE_CHECKING:
    from kedi_decisions import BooleanCriteria, ChoiceCriteria, Probability, Rubric


def __getattr__(name: str) -> Any:
    # Evidence is core functionality; criteria remain an optional dependency.
    if name not in {"BooleanCriteria", "ChoiceCriteria", "Probability", "Rubric"}:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        import kedi_decisions
    except ModuleNotFoundError as exc:
        if exc.name != "kedi_decisions":
            raise
        raise ModuleNotFoundError(
            "Decision criteria require the optional kedi-decisions package. "
            "Install it with: uv add kedi-decisions",
            name="kedi_decisions",
        ) from exc
    return getattr(kedi_decisions, name)


_FINGERPRINT_CODEC = "kedi-decision-value-v1"
_current_evidence_collector: ContextVar[list[CapturedDecisionEvidence] | None] = ContextVar(
    "kedi_decision_evidence_collector",
    default=None,
)
_current_decision_environment: ContextVar[dict[str, Any] | None] = ContextVar(
    "kedi_decision_environment",
    default=None,
)


@dataclass(frozen=True, slots=True)
class DecisionSource:
    """Source location of the template that produced a decision."""

    path: str | None
    line: int | None
    column: int | None


@dataclass(frozen=True, slots=True)
class DecisionInput:
    """A named, model-facing input identity captured at request construction."""

    name: str
    fingerprint: str | None
    supported: bool


@dataclass(frozen=True, slots=True)
class DecisionOutput:
    """A typed output identity used to detect later in-place mutation."""

    field_path: tuple[str, ...]
    fingerprint: str | None
    supported: bool


@dataclass(frozen=True, slots=True)
class CapturedDecisionEvidence:
    """Adapter-neutral provider evidence emitted for one successful model call."""

    provider: str
    model: str | None
    payload_json: str

    @classmethod
    def create(
        cls,
        *,
        provider: str,
        model: str | None,
        metadata: Mapping[str, Any],
    ) -> CapturedDecisionEvidence:
        payload = json.dumps(
            _encode_evidence_value(metadata, seen=set()),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return cls(provider=provider, model=model, payload_json=payload)

    @classmethod
    def from_encoded(
        cls,
        *,
        provider: str,
        model: str | None,
        encoded: Any,
    ) -> CapturedDecisionEvidence:
        payload_json = json.dumps(
            encoded,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        evidence = cls(provider=provider, model=model, payload_json=payload_json)
        evidence.payload()
        return evidence

    def payload(self) -> Mapping[str, Any]:
        value = _decode_evidence_value(json.loads(self.payload_json))
        if not isinstance(value, dict):  # pragma: no cover - constructor guarantees this
            raise ValueError("Decision evidence payload must be an object")
        return _readonly_json(value)


@dataclass(frozen=True, slots=True)
class DecisionRecord:
    """Immutable evidence shared by every output field from one model call."""

    call_id: str
    source: DecisionSource
    prompt_fingerprint: str
    inputs: tuple[DecisionInput, ...]
    inputs_complete: bool
    outputs: tuple[DecisionOutput, ...]
    evidence: CapturedDecisionEvidence

    def output(self, field_path: tuple[str, ...]) -> DecisionOutput | None:
        return next((item for item in self.outputs if item.field_path == field_path), None)

    def to_data(self) -> dict[str, Any]:
        return {
            "version": 1,
            "call_id": self.call_id,
            "source": {
                "path": self.source.path,
                "line": self.source.line,
                "column": self.source.column,
            },
            "prompt_fingerprint": self.prompt_fingerprint,
            "inputs": [
                {
                    "name": item.name,
                    "fingerprint": item.fingerprint,
                    "supported": item.supported,
                }
                for item in self.inputs
            ],
            "inputs_complete": self.inputs_complete,
            "outputs": [
                {
                    "field_path": list(item.field_path),
                    "fingerprint": item.fingerprint,
                    "supported": item.supported,
                }
                for item in self.outputs
            ],
            "evidence": {
                "provider": self.evidence.provider,
                "model": self.evidence.model,
                "payload_encoding": json.loads(self.evidence.payload_json),
            },
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> DecisionRecord:
        if data.get("version") != 1:
            raise ValueError("Unsupported decision record version")
        source = _required_mapping(data, "source")
        evidence = _required_mapping(data, "evidence")
        raw_inputs = data.get("inputs")
        raw_outputs = data.get("outputs")
        if not isinstance(raw_inputs, list) or not isinstance(raw_outputs, list):
            raise ValueError("Decision record inputs and outputs must be arrays")
        inputs = tuple(_decode_input(item) for item in raw_inputs)
        outputs = tuple(_decode_output(item) for item in raw_outputs)
        payload_encoding = evidence.get("payload_encoding")
        if payload_encoding is None:
            raise ValueError("Decision evidence payload encoding is missing")
        provider = evidence.get("provider")
        model = evidence.get("model")
        if not isinstance(provider, str) or (model is not None and not isinstance(model, str)):
            raise ValueError("Decision evidence provider/model is invalid")
        path = source.get("path")
        line = source.get("line")
        column = source.get("column")
        if path is not None and not isinstance(path, str):
            raise ValueError("Decision source path is invalid")
        if line is not None and (not isinstance(line, int) or isinstance(line, bool)):
            raise ValueError("Decision source line is invalid")
        if column is not None and (not isinstance(column, int) or isinstance(column, bool)):
            raise ValueError("Decision source column is invalid")
        call_id = data.get("call_id")
        prompt_fingerprint = data.get("prompt_fingerprint")
        inputs_complete = data.get("inputs_complete")
        if not isinstance(call_id, str) or not isinstance(prompt_fingerprint, str):
            raise ValueError("Decision record identifiers are invalid")
        if not isinstance(inputs_complete, bool):
            raise ValueError("Decision record input coverage flag is invalid")
        return cls(
            call_id=call_id,
            source=DecisionSource(path=path, line=line, column=column),
            prompt_fingerprint=prompt_fingerprint,
            inputs=inputs,
            inputs_complete=inputs_complete,
            outputs=outputs,
            evidence=CapturedDecisionEvidence.from_encoded(
                provider=provider,
                model=model,
                encoded=payload_encoding,
            ),
        )


@dataclass(frozen=True, slots=True)
class DecisionReference:
    """Binding-owned reference to a shared pending or completed decision record."""

    record_future: Future[DecisionRecord | None]
    field_path: tuple[str, ...]

    @classmethod
    def completed(
        cls,
        record: DecisionRecord,
        field_path: tuple[str, ...],
    ) -> DecisionReference:
        future: Future[DecisionRecord | None] = Future()
        future.set_result(record)
        return cls(future, field_path)


@dataclass(frozen=True, slots=True)
class DecisionInfo:
    """Read-only field view over one historical provider decision."""

    _record: DecisionRecord
    field_path: tuple[str, ...]

    @property
    def call_id(self) -> str:
        return self._record.call_id

    @property
    def model(self) -> str | None:
        return self._record.evidence.model

    @property
    def provider(self) -> str:
        return self._record.evidence.provider

    @property
    def source(self) -> DecisionSource:
        return self._record.source

    @property
    def prompt_fingerprint(self) -> str:
        return self._record.prompt_fingerprint

    @property
    def inputs(self) -> Mapping[str, str | None]:
        return MappingProxyType({item.name: item.fingerprint for item in self._record.inputs})

    @property
    def request_fingerprint(self) -> str | None:
        value = self._record.evidence.payload().get("request_fingerprint")
        return value if isinstance(value, str) else None

    @property
    def state_fingerprint(self) -> str | None:
        value = self._record.evidence.payload().get("state_fingerprint")
        return value if isinstance(value, str) else None

    @property
    def criteria_fingerprint(self) -> str | None:
        value = self._record.evidence.payload().get("questions_fingerprint")
        return value if isinstance(value, str) else None

    @property
    def config_fingerprint(self) -> str | None:
        value = self._record.evidence.payload().get("config_fingerprint")
        return value if isinstance(value, str) else None

    @property
    def answers(self) -> tuple[Mapping[str, Any], ...]:
        raw_answers = self._record.evidence.payload().get("answers")
        if not isinstance(raw_answers, Mapping):
            return ()
        matched: list[Mapping[str, Any]] = []
        for answer in raw_answers.values():
            if not isinstance(answer, Mapping):
                continue
            path = answer.get("path")
            if (
                isinstance(path, (list, tuple))
                and tuple(path[: len(self.field_path)]) == self.field_path
            ):
                matched.append(answer)
        return tuple(matched)

    @property
    def probability(self) -> float | None:
        answer = self._single_answer("noul")
        value = None if answer is None else answer.get("probability")
        return (
            float(value)
            if isinstance(value, (int, float)) and not isinstance(value, bool)
            else None
        )

    @property
    def threshold(self) -> float | None:
        answer = self._single_answer("noul")
        if answer is None or answer.get("output_kind") != "boolean":
            return None
        value = self._record.evidence.payload().get("boolean_threshold")
        return (
            float(value)
            if isinstance(value, (int, float)) and not isinstance(value, bool)
            else None
        )

    @property
    def comparator(self) -> str | None:
        if self.threshold is None:
            return None
        value = self._record.evidence.payload().get("boolean_comparator")
        return value if isinstance(value, str) else None

    @property
    def confidence(self) -> float | None:
        answer = self._single_answer()
        value = None if answer is None else answer.get("confidence")
        return (
            float(value)
            if isinstance(value, (int, float)) and not isinstance(value, bool)
            else None
        )

    @property
    def distribution(self) -> Mapping[Any, float] | None:
        answer = self._single_answer()
        value = None if answer is None else answer.get("probabilities")
        return value if isinstance(value, Mapping) else None

    @property
    def rubric(self) -> Mapping[Any, Any] | None:
        answer = self._single_answer("score")
        value = None if answer is None else answer.get("legend")
        return value if isinstance(value, Mapping) else None

    def matches_inputs(self, **inputs: Any) -> bool:
        """Compare explicit inputs only; this never calls a model or checks external state."""
        if not self._record.inputs_complete:
            raise ValueError("Decision input matching is unavailable for computed substitutions")
        expected = {item.name: item for item in self._record.inputs}
        if set(inputs) != set(expected):
            missing = sorted(set(expected) - set(inputs))
            extra = sorted(set(inputs) - set(expected))
            raise ValueError(f"Decision input names do not match; missing={missing}, extra={extra}")
        for name, value in inputs.items():
            identity = expected[name]
            if not identity.supported or identity.fingerprint is None:
                raise ValueError(f"Decision input {name!r} does not support stable matching")
            current = fingerprint_model_input(value)
            if current is None:
                raise ValueError(f"Input {name!r} does not support stable matching")
            if current != identity.fingerprint:
                return False
        return True

    def _single_answer(self, expected_type: str | None = None) -> Mapping[str, Any] | None:
        answers = self.answers
        if len(answers) != 1:
            return None
        answer = answers[0]
        if expected_type is not None and answer.get("type") != expected_type:
            return None
        return answer


class DecisionCapture:
    """Completed decision snapshots collected during one explicit capture scope."""

    def __init__(self) -> None:
        self._records: list[DecisionRecord] = []
        self._lock = Lock()
        self._closed = False

    @property
    def decisions(self) -> tuple[DecisionInfo, ...]:
        """One immutable view per output binding, in call completion order."""
        with self._lock:
            records = tuple(self._records)
        return tuple(
            DecisionInfo(record, output.field_path)
            for record in records
            for output in record.outputs
        )

    def _append(self, record: DecisionRecord) -> None:
        with self._lock:
            if not self._closed:
                self._records.append(record)

    def _close(self) -> None:
        with self._lock:
            self._closed = True


_decision_captures: ContextVar[tuple[DecisionCapture, ...]] = ContextVar(
    "kedi_decision_captures", default=()
)


@contextmanager
def capture_decisions() -> Iterator[DecisionCapture]:
    """Capture newly completed template decisions without changing return values.

    Nested scopes also contribute to their enclosing capture. Use a regular
    ``with`` around awaited calls; await all desired work before leaving the
    scope. Exit does not wait for or trigger model calls. Cached returns do not
    generate new decisions. Successful records survive later call failures.
    """
    capture = DecisionCapture()
    token = _decision_captures.set((*_decision_captures.get(), capture))
    try:
        yield capture
    finally:
        capture._close()
        _decision_captures.reset(token)


def publish_decision_record(record: DecisionRecord) -> None:
    """Deliver a validated runtime record to active call-local captures."""
    for capture in _decision_captures.get():
        capture._append(record)


@contextmanager
def collect_decision_evidence() -> Iterator[list[CapturedDecisionEvidence]]:
    """Collect provider evidence for exactly one adapter call in the current context."""
    collected: list[CapturedDecisionEvidence] = []
    token = _current_evidence_collector.set(collected)
    try:
        yield collected
    finally:
        _current_evidence_collector.reset(token)


def capture_decision_evidence(
    *,
    provider: str,
    model: str | None,
    metadata: Mapping[str, Any],
) -> None:
    """Publish final response metadata when a core call-local collector is active."""
    collector = _current_evidence_collector.get()
    if collector is None:
        return
    collector.append(
        CapturedDecisionEvidence.create(provider=provider, model=model, metadata=metadata)
    )


@contextmanager
def decision_environment(env: dict[str, Any]) -> Iterator[None]:
    token = _current_decision_environment.set(env)
    try:
        yield
    finally:
        _current_decision_environment.reset(token)


def decision_info(name: str) -> DecisionInfo | None:
    """Inspect a binding in the active Kedi lexical environment."""
    env = _current_decision_environment.get()
    if env is None:
        raise RuntimeError(
            "decision_info() requires active Kedi Python execution; "
            "use InteractiveSession.decision_info() outside a fragment"
        )
    return decision_info_for_env(env, name)


def decision_info_for_env(env: dict[str, Any], name: str) -> DecisionInfo | None:
    from kedi.lang.compiler.environment import ScopeFrame

    if not isinstance(name, str) or not name:
        raise ValueError("Decision binding name must be a non-empty string")
    if name not in env:
        raise NameError(f"Value not found in environment: {name}")
    metadata = env.binding_metadata(name) if isinstance(env, ScopeFrame) else None
    reference = metadata.decision if metadata is not None else None
    if not isinstance(reference, DecisionReference):
        return None
    record = reference.record_future.result()
    if record is None:
        return None
    output = record.output(reference.field_path)
    if output is not None and output.supported and output.fingerprint is not None:
        current = env.raw_value(name) if isinstance(env, ScopeFrame) else env[name]
        from kedi.promise import force

        current_fingerprint = fingerprint_value(force(current))
        if current_fingerprint != output.fingerprint:
            return None
    return DecisionInfo(record, reference.field_path)


def new_call_id() -> str:
    return uuid4().hex


def fingerprint_text(value: str) -> str:
    return _digest(value.encode("utf-8"))


def fingerprint_value(value: Any) -> str | None:
    """Return a type-preserving deterministic fingerprint, or ``None`` if unsupported."""
    try:
        canonical = _canonical_value(value, seen=set())
        payload = json.dumps(
            canonical,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError):
        return None
    return _digest(payload)


def fingerprint_model_input(value: Any) -> str | None:
    from kedi.lang.compiler.environment import model_value

    return fingerprint_value(model_value(value))


def _digest(payload: bytes) -> str:
    return f"{_FINGERPRINT_CODEC}:sha256:{hashlib.sha256(payload).hexdigest()}"


def _canonical_value(value: Any, *, seen: set[int]) -> Any:
    if value is None:
        return ["none"]
    if type(value) is bool:
        return ["bool", value]
    if type(value) is int:
        return ["int", str(value)]
    if type(value) is float:
        return ["float", value.hex()]
    if type(value) is str:
        return ["str", value]
    if type(value) is bytes:
        return ["bytes", base64.b64encode(value).decode("ascii")]
    if isinstance(value, BaseModel):
        values = object.__getattribute__(value, "__dict__")
        return [
            "pydantic",
            value.__class__.__module__,
            value.__class__.__qualname__,
            _canonical_value(values, seen=seen),
        ]
    if is_dataclass(value) and not isinstance(value, type):
        return [
            "dataclass",
            value.__class__.__module__,
            value.__class__.__qualname__,
            _canonical_value(
                {item.name: object.__getattribute__(value, item.name) for item in fields(value)},
                seen=seen,
            ),
        ]
    if type(value) in {list, tuple, dict}:
        container = cast(Any, value)
        identity = id(value)
        if identity in seen:
            raise ValueError("Cyclic values do not support decision fingerprints")
        seen.add(identity)
        try:
            if type(value) is list:
                return ["list", [_canonical_value(item, seen=seen) for item in container]]
            if type(value) is tuple:
                return ["tuple", [_canonical_value(item, seen=seen) for item in container]]
            entries = [
                [_canonical_value(key, seen=seen), _canonical_value(item, seen=seen)]
                for key, item in container.items()
            ]
            entries.sort(
                key=lambda item: json.dumps(item[0], sort_keys=True, separators=(",", ":"))
            )
            return ["mapping", entries]
        finally:
            seen.remove(identity)
    raise TypeError(f"Unsupported decision fingerprint value: {type(value).__name__}")


def _readonly_json(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _readonly_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_readonly_json(item) for item in value)
    return value


def _encode_evidence_value(value: Any, *, seen: set[int]) -> Any:
    if value is None:
        return ["none"]
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int):
        return ["int", str(value)]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Decision evidence cannot contain non-finite floats")
        return ["float", value.hex()]
    if isinstance(value, str):
        return ["str", value]
    if isinstance(value, (list, tuple, Mapping)):
        identity = id(value)
        if identity in seen:
            raise ValueError("Decision evidence cannot contain cycles")
        seen.add(identity)
        try:
            if isinstance(value, list):
                return ["list", [_encode_evidence_value(item, seen=seen) for item in value]]
            if isinstance(value, tuple):
                return ["tuple", [_encode_evidence_value(item, seen=seen) for item in value]]
            return [
                "map",
                [
                    [
                        _encode_evidence_value(key, seen=seen),
                        _encode_evidence_value(item, seen=seen),
                    ]
                    for key, item in value.items()
                ],
            ]
        finally:
            seen.remove(identity)
    raise TypeError(f"Unsupported decision evidence value: {type(value).__name__}")


def _decode_evidence_value(value: Any) -> Any:
    if not isinstance(value, list) or not value or not isinstance(value[0], str):
        raise ValueError("Invalid decision evidence encoding")
    tag = value[0]
    if tag == "none" and len(value) == 1:
        return None
    if tag == "bool" and len(value) == 2 and isinstance(value[1], bool):
        return value[1]
    if tag == "int" and len(value) == 2 and isinstance(value[1], str):
        return int(value[1])
    if tag == "float" and len(value) == 2 and isinstance(value[1], str):
        decoded_float = float.fromhex(value[1])
        if not math.isfinite(decoded_float):
            raise ValueError("Decision evidence cannot contain non-finite floats")
        return decoded_float
    if tag == "str" and len(value) == 2 and isinstance(value[1], str):
        return value[1]
    if tag in {"list", "tuple"} and len(value) == 2 and isinstance(value[1], list):
        items = [_decode_evidence_value(item) for item in value[1]]
        return items if tag == "list" else tuple(items)
    if tag == "map" and len(value) == 2 and isinstance(value[1], list):
        decoded: dict[Any, Any] = {}
        for pair in value[1]:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError("Invalid decision evidence mapping entry")
            key = _decode_evidence_value(pair[0])
            try:
                hash(key)
            except TypeError as exc:
                raise ValueError("Invalid decision evidence mapping key") from exc
            if key in decoded:
                raise ValueError("Duplicate decision evidence mapping key")
            decoded[key] = _decode_evidence_value(pair[1])
        return decoded
    raise ValueError(f"Unknown or invalid decision evidence tag {tag!r}")


def _required_mapping(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = data.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Decision record {key!r} must be an object")
    return value


def _decode_input(value: Any) -> DecisionInput:
    if not isinstance(value, Mapping):
        raise ValueError("Decision input must be an object")
    name = value.get("name")
    fingerprint = value.get("fingerprint")
    supported = value.get("supported")
    if not isinstance(name, str) or (fingerprint is not None and not isinstance(fingerprint, str)):
        raise ValueError("Decision input identity is invalid")
    if not isinstance(supported, bool):
        raise ValueError("Decision input support flag is invalid")
    return DecisionInput(name=name, fingerprint=fingerprint, supported=supported)


def _decode_output(value: Any) -> DecisionOutput:
    if not isinstance(value, Mapping):
        raise ValueError("Decision output must be an object")
    field_path = value.get("field_path")
    fingerprint = value.get("fingerprint")
    supported = value.get("supported")
    if not isinstance(field_path, list) or not all(isinstance(item, str) for item in field_path):
        raise ValueError("Decision output field path is invalid")
    if fingerprint is not None and not isinstance(fingerprint, str):
        raise ValueError("Decision output fingerprint is invalid")
    if not isinstance(supported, bool):
        raise ValueError("Decision output support flag is invalid")
    return DecisionOutput(tuple(field_path), fingerprint, supported)


__all__ = [
    "BooleanCriteria",
    "ChoiceCriteria",
    "DecisionCapture",
    "DecisionInfo",
    "DecisionInput",
    "DecisionOutput",
    "DecisionRecord",
    "DecisionReference",
    "DecisionSource",
    "Probability",
    "Rubric",
    "capture_decision_evidence",
    "capture_decisions",
    "decision_info",
]
