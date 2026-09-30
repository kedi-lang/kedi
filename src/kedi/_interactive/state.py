"""Logical-state capture and restoration for interactive Kedi sessions."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from ..agent_adapter.conversation import ConversationState
from ..decisions import DecisionRecord, DecisionReference, decision_info_for_env
from ..lang.compiler.environment import BindingMetadata, ScopeFrame
from ..lang.compiler.program_compiler import ProgramCompiler
from ..lang.compiler.type_system import KediType
from ..lang.parser import parse_program
from ..promise import KediPromise, force
from .persistence import (
    SessionDumpError,
    SessionLoadError,
    SessionValueDecoder,
    SessionValueEncoder,
    UnsupportedSessionValue,
    decode_profile,
    encode_profile,
    read_snapshot,
    write_snapshot,
)

if TYPE_CHECKING:
    from ..agent_adapter.meta import AgentAdapter
    from ..executors.base import Executor
    from ..interactive_session import InteractiveSession


@dataclass(frozen=True)
class ExecutedFragment:
    source: str
    source_name: str


def dump_session(session: InteractiveSession, path: str | Path) -> Path:
    return write_snapshot(path, _build_snapshot_payload(session))


def load_session(
    session_type: type[InteractiveSession],
    path: str | Path,
    *,
    adapter: AgentAdapter[Any] | None,
    executor: Executor | None,
    engine: Any | None,
    conversation_state: ConversationState | None,
    close_conversation_state: bool | None,
) -> InteractiveSession:
    payload = read_snapshot(path)
    try:
        default_profile = decode_profile(
            payload["default_profile"],
            decoder=SessionValueDecoder(kedi_models={}),
            kedi_models={},
        )
        conversation_state, owns_conversation = _load_conversation(
            payload["conversation_session_id"],
            conversation_state,
        )
        if close_conversation_state is not None:
            owns_conversation = close_conversation_state

        limits_data = payload.get("subagent_usage_limits")
        limits = None
        if limits_data is not None:
            from ..agent_adapter.subagents.limits import SubagentUsageLimits

            limits = SubagentUsageLimits(**limits_data)
        session = session_type(
            adapter=adapter,
            executor=executor,
            engine=engine,
            default_agent_profile=default_profile,
            cwd=payload["cwd"],
            subagent_max_depth=payload["subagent_max_depth"],
            subagent_max_concurrency=payload["subagent_max_concurrency"],
            subagent_timeout_seconds=payload["subagent_timeout_seconds"],
            subagent_usage_limits=limits,
            subagent_state_path=payload["subagent_state_path"],
            conversation_state=conversation_state,
            close_conversation_state=owns_conversation,
            loop_iteration_limit=payload["loop_iteration_limit"],
        )
    except SessionLoadError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise SessionLoadError(f"Invalid interactive session metadata: {exc}") from exc

    try:
        _restore_payload(session, payload)
    except SessionLoadError:
        session.close()
        raise
    except (KeyError, TypeError, ValueError) as exc:
        session.close()
        raise SessionLoadError(f"Invalid interactive session payload: {exc}") from exc
    except BaseException:
        session.close()
        raise
    return session


def _load_conversation(
    conversation_id: str | None,
    supplied: ConversationState | None,
) -> tuple[ConversationState | None, bool]:
    if conversation_id is None:
        if supplied is not None:
            raise SessionLoadError(
                "The snapshot did not own conversation state, but one was supplied"
            )
        return None, False
    if supplied is None:
        return ConversationState(session_id=conversation_id), True
    issues = _conversation_persistence_issues(supplied)
    if issues:
        raise SessionLoadError(
            "The supplied conversation state is not empty:\n"
            + "\n".join(f"- {issue}" for issue in issues)
        )
    if supplied.session_id != conversation_id:
        raise SessionLoadError("The supplied conversation state has a different session_id")
    return supplied, False


def _build_snapshot_payload(session: InteractiveSession) -> dict[str, Any]:
    issues = list(session._persistence_blockers)
    programs: list[Any] = []
    for fragment in session._fragments:
        try:
            program = parse_program(fragment.source, source_path=fragment.source_name)
        except BaseException as exc:
            issues.append(f"fragment {fragment.source_name!r} no longer parses: {exc}")
            continue
        issues.extend(_fragment_persistence_issues(program, fragment.source_name))
        programs.append(program)

    issues.extend(_conversation_persistence_issues(session._conversation_state))
    if session._conversation_state is None:
        issues.extend(
            _artifact_manager_persistence_issues(
                session._compiler.rt.artifacts,
                prefix="runtime",
            )
        )
    if any(session._compiler.rt._outstanding.values()):
        issues.append("runtime has unfinished model work")
    if session._subagent_state_path is not None:
        issues.append("session uses external persistent subagent state")
    issues.extend(_declaration_persistence_issues(session._compiler, programs))

    model_names = _kedi_model_names(session._compiler)
    raw_environment = _materialize_environment(session._env, issues)
    decision_records: dict[str, dict[str, Any]] = {}
    binding_metadata = _encode_binding_metadata(
        session._env,
        issues,
        decision_records=decision_records,
    )
    preflight_encoder = SessionValueEncoder(kedi_models=model_names)
    preflight_encoder.bind_decision_records(decision_records)
    for name, value in raw_environment.items():
        try:
            preflight_encoder.encode(value, path=f"environment[{name!r}]")
        except UnsupportedSessionValue as exc:
            issues.append(str(exc))
    for label, profile in (
        ("default_profile", session._compiler.rt.default_agent_profile),
        ("active_profile", session._top_level_state.active_profile),
    ):
        try:
            encode_profile(
                profile,
                encoder=preflight_encoder,
                model_names=model_names,
                path=label,
            )
        except UnsupportedSessionValue as exc:
            issues.append(str(exc))
    if issues:
        raise SessionDumpError(issues)

    encoder = SessionValueEncoder(kedi_models=model_names)
    encoder.bind_decision_records(decision_records)
    try:
        encoded_environment = encoder.encode(raw_environment, path="environment")
        default_profile = encode_profile(
            session._compiler.rt.default_agent_profile,
            encoder=encoder,
            model_names=model_names,
            path="default_profile",
        )
        active_profile = None
        if (
            session._top_level_state.active_profile
            is not session._compiler.rt.default_agent_profile
        ):
            active_profile = encode_profile(
                session._top_level_state.active_profile,
                encoder=encoder,
                model_names=model_names,
                path="active_profile",
            )
    except UnsupportedSessionValue as exc:
        raise SessionDumpError([str(exc)]) from exc

    limits = session._subagent_usage_limits
    return {
        "cwd": str(session._cwd),
        "cell_number": session._cell_number,
        "source_names": sorted(session._source_names),
        "fragments": [
            {
                "source": fragment.source,
                "source_name": fragment.source_name,
                "digest": hashlib.sha256(fragment.source.encode("utf-8")).hexdigest(),
            }
            for fragment in session._fragments
        ],
        "environment": encoded_environment,
        "binding_metadata": binding_metadata,
        "decision_records": decision_records,
        "default_profile": default_profile,
        "active_profile": active_profile,
        "conversation_session_id": (
            None if session._conversation_state is None else session._conversation_state.session_id
        ),
        "subagent_max_depth": session._subagent_max_depth,
        "subagent_max_concurrency": session._subagent_max_concurrency,
        "subagent_timeout_seconds": session._subagent_timeout_seconds,
        "subagent_usage_limits": (
            None
            if limits is None
            else {item.name: getattr(limits, item.name) for item in fields(limits)}
        ),
        "subagent_state_path": (
            None if session._subagent_state_path is None else str(session._subagent_state_path)
        ),
        "loop_iteration_limit": session._loop_iteration_limit,
    }


def _materialize_environment(env: Mapping[Any, Any], issues: list[str]) -> dict[str, Any]:
    materialized: dict[str, Any] = {}
    for name, value in env.items():
        if not isinstance(name, str):
            issues.append(f"environment key {name!r} is not a string")
            continue
        try:
            materialized[name] = force(value) if isinstance(value, KediPromise) else value
        except BaseException as exc:
            issues.append(f"environment[{name!r}] could not be resolved: {exc}")
    return materialized


def _encode_binding_metadata(
    env: Mapping[str, Any],
    issues: list[str],
    *,
    decision_records: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    if not isinstance(env, ScopeFrame):
        issues.append("interactive environment does not expose lexical binding metadata")
        return {}

    encoded: dict[str, dict[str, Any]] = {}
    record_table = {} if decision_records is None else decision_records
    for name, metadata in env.local_binding_metadata().items():
        contract = metadata.type_contract
        if contract is None:
            type_contract = None
        elif isinstance(contract, KediType):
            type_contract = None if contract.is_none() else str(contract)
        else:
            issues.append(
                f"binding metadata for {name!r} has unsupported type contract "
                f"{type(contract).__module__}.{type(contract).__qualname__}"
            )
            continue
        decision = None
        reference = metadata.decision
        if reference is not None:
            if not isinstance(reference, DecisionReference):
                issues.append(f"binding metadata for {name!r} has an invalid decision reference")
                continue
            if not reference.record_future.done():
                issues.append(f"binding metadata for {name!r} has pending decision evidence")
                continue
            try:
                record = reference.record_future.result()
            except BaseException as exc:
                issues.append(f"binding metadata for {name!r} has failed decision evidence: {exc}")
                continue
            if record is not None and decision_info_for_env(env, name) is not None:
                encoded_record = record.to_data()
                previous = record_table.setdefault(record.call_id, encoded_record)
                if previous != encoded_record:
                    issues.append(f"decision record {record.call_id!r} has conflicting payloads")
                    continue
                decision = {
                    "record_id": record.call_id,
                    "field_path": list(reference.field_path),
                }
        encoded[name] = {
            "kind": metadata.kind,
            "type_contract": type_contract,
            "assignable": metadata.assignable,
            "synthetic": metadata.synthetic,
            "decision": decision,
        }
    return encoded


def _decode_binding_metadata(
    encoded: Any,
    environment: dict[str, Any],
    decision_records: Mapping[str, DecisionRecord] | None = None,
) -> dict[str, BindingMetadata]:
    if not isinstance(encoded, dict) or set(encoded) != set(environment):
        raise SessionLoadError("Restored binding metadata must match the environment exactly")

    decoded: dict[str, BindingMetadata] = {}
    record_table = {} if decision_records is None else decision_records
    for name, item in encoded.items():
        if not isinstance(name, str) or not isinstance(item, dict):
            raise SessionLoadError("Restored binding metadata entries must be objects")
        kind = item.get("kind")
        type_contract = item.get("type_contract")
        assignable = item.get("assignable")
        synthetic = item.get("synthetic")
        decision_data = item.get("decision")
        if not isinstance(kind, str):
            raise SessionLoadError(f"Binding metadata for {name!r} has an invalid kind")
        if type_contract is not None and not isinstance(type_contract, str):
            raise SessionLoadError(f"Binding metadata for {name!r} has an invalid type contract")
        if not isinstance(assignable, bool) or not isinstance(synthetic, bool):
            raise SessionLoadError(f"Binding metadata for {name!r} has invalid flags")
        decision = None
        if decision_data is not None:
            if not isinstance(decision_data, dict):
                raise SessionLoadError(f"Binding metadata for {name!r} has invalid decision data")
            record_id = decision_data.get("record_id")
            field_path = decision_data.get("field_path")
            if not isinstance(record_id, str) or not isinstance(field_path, list):
                raise SessionLoadError(
                    f"Binding metadata for {name!r} has an invalid decision reference"
                )
            if not all(isinstance(segment, str) for segment in field_path):
                raise SessionLoadError(
                    f"Binding metadata for {name!r} has an invalid decision field path"
                )
            record = record_table.get(record_id)
            if record is None or record.output(tuple(field_path)) is None:
                raise SessionLoadError(
                    f"Binding metadata for {name!r} references an unknown decision output"
                )
            decision = DecisionReference.completed(record, tuple(field_path))
        decoded[name] = BindingMetadata(
            kind=kind,
            type_contract=None if type_contract is None else KediType(type_contract),
            assignable=assignable,
            synthetic=synthetic,
            decision=decision,
        )
    return decoded


def _declaration_persistence_issues(
    compiler: ProgramCompiler,
    programs: list[Any],
) -> list[str]:
    issues: list[str] = []
    expected_procedures = {
        procedure.name
        for program in programs
        for procedure in program.procedures
        if not procedure.auto_spec and not procedure.ai_prompt
    }
    extra_procedures = set(compiler.rt._procedures) - expected_procedures
    if extra_procedures:
        issues.append(
            "runtime contains procedures without restorable fragment source: "
            + ", ".join(sorted(extra_procedures))
        )
    expected_types = {typedef.name for program in programs for typedef in program.type_defs}
    extra_types = set(_kedi_models(compiler)) - expected_types
    if extra_types:
        issues.append(
            "runtime contains types without restorable fragment source: "
            + ", ".join(sorted(extra_types))
        )
    expected_profiles = {name for program in programs for name in program.profile_defs}
    extra_profiles = set(compiler.rt._profile_registry) - expected_profiles
    if extra_profiles:
        issues.append(
            "runtime contains profiles without restorable fragment source: "
            + ", ".join(sorted(extra_profiles))
        )
    return issues


def _restore_payload(session: InteractiveSession, payload: dict[str, Any]) -> None:
    fragments: list[ExecutedFragment] = []
    for entry in payload["fragments"]:
        source = entry["source"]
        source_name = entry["source_name"]
        digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
        if digest != entry["digest"]:
            raise SessionLoadError(f"Fragment {source_name!r} failed its integrity check")
        program = parse_program(source, source_path=source_name)
        issues = _fragment_persistence_issues(program, source_name)
        if issues:
            raise SessionLoadError("Unsafe fragment in snapshot:\n" + "\n".join(issues))
        session._compiler.prepare_fragment(
            program,
            import_source_path=session._import_source_path(source_name),
        )
        fragments.append(ExecutedFragment(source, source_name))

    models = _kedi_models(session._compiler)
    decision_records = _decode_decision_records(payload.get("decision_records", {}))
    decoder = SessionValueDecoder(kedi_models=models, decision_records=decision_records)
    environment = decoder.decode(payload["environment"], path="environment")
    if not isinstance(environment, dict) or not all(isinstance(name, str) for name in environment):
        raise SessionLoadError("Restored environment must be a string-keyed mapping")
    metadata = _decode_binding_metadata(
        payload["binding_metadata"],
        environment,
        decision_records,
    )
    if not isinstance(session._env, ScopeFrame):
        raise SessionLoadError("Interactive environment is not a lexical scope frame")
    session._env.restore_locals(environment, metadata)
    session._compiler.rt._initial_globals_env = dict(environment)
    session._compiler.rt._globals_env = session._env

    active_data = payload.get("active_profile")
    session._top_level_state.active_profile = (
        session._compiler.rt.default_agent_profile
        if active_data is None
        else decode_profile(active_data, decoder=decoder, kedi_models=models)
    )
    session._cell_number = payload["cell_number"]
    session._source_names = set(payload["source_names"])
    session._fragments = fragments


def _decode_decision_records(encoded: Any) -> dict[str, DecisionRecord]:
    if not isinstance(encoded, dict):
        raise SessionLoadError("Restored decision records must be an object")
    records: dict[str, DecisionRecord] = {}
    for record_id, value in encoded.items():
        if not isinstance(record_id, str) or not isinstance(value, dict):
            raise SessionLoadError("Restored decision record entries must be objects")
        try:
            record = DecisionRecord.from_data(value)
        except ValueError as exc:
            raise SessionLoadError(f"Invalid decision record {record_id!r}: {exc}") from exc
        if record.call_id != record_id:
            raise SessionLoadError(f"Decision record {record_id!r} has a mismatched call_id")
        records[record_id] = record
    return records


def _kedi_model_names(compiler: ProgramCompiler) -> dict[type[BaseModel], str]:
    return {model: name for name, model in _kedi_models(compiler).items()}


def _kedi_models(compiler: ProgramCompiler) -> dict[str, type[BaseModel]]:
    type_env = compiler.rt._type_env_stack[0]
    models: dict[str, type[BaseModel]] = {}
    for name, value in type_env.items():
        if name.startswith("__") or not isinstance(value, type):
            continue
        if issubclass(value, BaseModel):
            models[name] = value
    return models


def _conversation_persistence_issues(state: ConversationState | None) -> list[str]:
    if state is None:
        return []
    issues: list[str] = []
    with state._lock:
        if state._closed:
            issues.append("conversation state is closed")
        if state._turns:
            issues.append("conversation contains portable turns")
        if state._adapter_states:
            issues.append("conversation contains adapter-native continuation state")
        if state._adapter_cleanups:
            issues.append("conversation owns process-local adapter cleanup callbacks")
        if state._cache_epoch:
            issues.append("conversation cache epoch is not initial")
    if state.history.snapshot():
        issues.append("conversation contains artifact history")
    issues.extend(_artifact_manager_persistence_issues(state.artifacts, prefix="conversation"))
    return issues


def _artifact_manager_persistence_issues(manager: Any, *, prefix: str) -> list[str]:
    with manager._lock:
        if manager._ref_stores:
            return [f"{prefix} contains active or retained artifacts"]
    return []


def _fragment_persistence_issues(program: Any, source_name: str) -> list[str]:
    from ..lang.ast import ProcDefStmt, TypeDefStmt

    issues: list[str] = []
    if program.imports:
        issues.append(
            f"fragment {source_name!r} imports modules whose initialization cannot be replayed"
        )
    if program.prelude is not None:
        issues.append(
            f"fragment {source_name!r} has an inline Python prelude that cannot be restored "
            "without execution"
        )
    for typedef in program.type_defs:
        if any(field.default_expr is not None for field in typedef.fields):
            issues.append(
                f"fragment {source_name!r} defines type {typedef.name!r} with executable defaults"
            )
    for item in _walk_statements(program.ordered_toplevel or program.toplevel_body):
        if isinstance(item, ProcDefStmt):
            issues.append(
                f"fragment {source_name!r} contains a runtime-scoped procedure definition"
            )
        elif isinstance(item, TypeDefStmt):
            issues.append(f"fragment {source_name!r} contains a runtime-scoped type definition")
    return issues


def _walk_statements(statements: Any) -> Any:
    for statement in statements:
        yield statement
        for attribute in ("body", "else_body"):
            nested = getattr(statement, attribute, None)
            if nested:
                yield from _walk_statements(nested)
