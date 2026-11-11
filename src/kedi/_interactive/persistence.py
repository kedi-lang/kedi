"""Strict, pickle-free persistence for :class:`kedi.InteractiveSession`."""

from __future__ import annotations

import base64
import hashlib
import importlib
import inspect
import json
import math
import os
import re
import tempfile
from dataclasses import fields
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping
from uuid import UUID

from pydantic import BaseModel

from .._version import __version__
from ..agent_adapter.approvals.policy import ApprovalPolicy
from ..agent_profile import AgentProfile
from ..agent_profile.codemode import CodeModeSettings
from ..agent_profile.compaction import CompactionSettings
from ..agent_profile.connection import A2AAuthConfig, A2AConnection
from ..agent_profile.hooks import HookSettings
from ..agent_profile.mcp import McpServerSpec
from ..agent_profile.skills import SkillsSettings
from ..artifacts import ArtifactPolicy
from ..decisions import DecisionInfo, DecisionRecord
from ..media import Audio, Document, Image, Media, Video

SNAPSHOT_FORMAT = "kedi.interactive-session"
SNAPSHOT_FORMAT_VERSION = 1
MAX_SNAPSHOT_BYTES = 64 * 1024 * 1024
MAX_VALUE_DEPTH = 100
_TAG = "$kedi"
_PATTERN_TYPE = type(re.compile(""))
_SURROGATES = re.compile(r"[\ud800-\udfff]")


class SessionPersistenceError(RuntimeError):
    """Base class for strict interactive-session persistence failures."""


class SessionDumpError(SessionPersistenceError):
    """The complete logical session state cannot be persisted."""

    def __init__(self, issues: list[str] | tuple[str, ...]) -> None:
        self.issues = tuple(issues)
        rendered = "\n".join(f"- {issue}" for issue in self.issues)
        super().__init__(f"Interactive session cannot be dumped:\n{rendered}")


class SessionLoadError(SessionPersistenceError):
    """A session snapshot is invalid, incompatible, or cannot be restored."""


class SnapshotTrustRequired(SessionLoadError):
    """An executable Python snapshot was loaded without explicit trust."""


class UnsupportedSessionValue(TypeError):
    """One value cannot be represented without changing its semantics."""

    def __init__(self, path: str, value: object, reason: str | None = None) -> None:
        logical_type = f"{type(value).__module__}.{type(value).__qualname__}"
        detail = f" ({reason})" if reason else ""
        super().__init__(f"{path}: unsupported {logical_type}{detail}")


class SessionValueEncoder:
    """Encode a conservative, type-preserving Python value subset as JSON data."""

    def __init__(
        self,
        *,
        kedi_models: Mapping[type[BaseModel], str],
        decision_records: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self._kedi_models = dict(kedi_models)
        self._decision_records = decision_records
        self._seen: dict[int, tuple[object, str]] = {}

    def bind_decision_records(self, records: dict[str, dict[str, Any]]) -> None:
        """Attach the snapshot-owned decision table without changing encoder subclasses."""
        self._decision_records = records

    def encode(self, value: Any, *, path: str = "value", depth: int = 0) -> Any:
        if depth > MAX_VALUE_DEPTH:
            raise UnsupportedSessionValue(path, value, "maximum nesting depth exceeded")
        if isinstance(value, str) and _SURROGATES.search(value):
            return {
                _TAG: "surrogate_str",
                "data": base64.b64encode(value.encode("utf-8", "surrogatepass")).decode("ascii"),
            }
        if value is None or isinstance(value, (bool, int, str)):
            return value
        if isinstance(value, float):
            if math.isfinite(value):
                return value
            return {_TAG: "float", "value": repr(value)}
        if isinstance(value, bytes):
            return {_TAG: "bytes", "data": base64.b64encode(value).decode("ascii")}
        if isinstance(value, complex):
            return {_TAG: "complex", "real": repr(value.real), "imag": repr(value.imag)}
        if isinstance(value, Decimal):
            return {_TAG: "decimal", "value": str(value)}
        if isinstance(value, UUID):
            return {_TAG: "uuid", "value": str(value)}
        if isinstance(value, datetime):
            return {_TAG: "datetime", "value": value.isoformat(), "fold": value.fold}
        if isinstance(value, date):
            return {_TAG: "date", "value": value.isoformat()}
        if isinstance(value, time):
            return {_TAG: "time", "value": value.isoformat(), "fold": value.fold}
        if isinstance(value, timedelta):
            return {
                _TAG: "timedelta",
                "days": value.days,
                "seconds": value.seconds,
                "microseconds": value.microseconds,
            }
        if isinstance(value, Path):
            return {_TAG: "path", "value": str(value)}
        if isinstance(value, Media):
            # The existing bytes encoding carries the content; MAX_SNAPSHOT_BYTES bounds it.
            return {
                _TAG: "media",
                "kind": value.kind,
                "media_type": value.media_type,
                "data": None if value.data is None else self.encode(value.data, path=path),
                "url": value.url,
                "file_id": value.file_id,
                "provider": value.provider,
                "name": value.name,
                "filename": value.filename,
            }
        if isinstance(value, _PATTERN_TYPE):
            return {
                _TAG: "regex",
                "pattern": self.encode(value.pattern, path=f"{path}.pattern", depth=depth + 1),
                "flags": value.flags,
            }
        if isinstance(value, DecisionInfo):
            if self._decision_records is None:
                raise UnsupportedSessionValue(path, value, "decision record table is unavailable")
            record = value._record
            encoded_record = record.to_data()
            previous = self._decision_records.setdefault(record.call_id, encoded_record)
            if previous != encoded_record:
                raise UnsupportedSessionValue(path, value, "decision record id is conflicting")
            return {
                _TAG: "decision_info",
                "record_id": record.call_id,
                "field_path": list(value.field_path),
            }

        next_depth = depth + 1
        if type(value) is list:
            self._claim_identity(value, path)
            return {
                _TAG: "list",
                "items": [
                    self.encode(item, path=f"{path}[{index}]", depth=next_depth)
                    for index, item in enumerate(value)
                ],
            }
        if type(value) is tuple:
            return {
                _TAG: "tuple",
                "items": [
                    self.encode(item, path=f"{path}[{index}]", depth=next_depth)
                    for index, item in enumerate(value)
                ],
            }
        if type(value) in (set, frozenset):
            if type(value) is set:
                self._claim_identity(value, path)
            items = [
                self.encode(item, path=f"{path}[{index}]", depth=next_depth)
                for index, item in enumerate(value)
            ]
            items.sort(key=_canonical_json)
            return {
                _TAG: "frozenset" if type(value) is frozenset else "set",
                "items": items,
            }
        if type(value) is dict:
            self._claim_identity(value, path)
            return {
                _TAG: "dict",
                "items": [
                    [
                        self.encode(key, path=f"{path}.<key>", depth=next_depth),
                        self.encode(item, path=f"{path}[{key!r}]", depth=next_depth),
                    ]
                    for key, item in value.items()
                ],
            }
        if isinstance(value, BaseModel):
            self._claim_identity(value, path)
            model_name = self._kedi_models.get(type(value))
            if model_name is None:
                raise UnsupportedSessionValue(
                    path,
                    value,
                    "only source-backed Kedi models can be restored",
                )
            model_fields = type(value).model_fields
            return {
                _TAG: "kedi_model",
                "name": model_name,
                "fields": {
                    name: self.encode(
                        getattr(value, name),
                        path=f"{path}.{name}",
                        depth=next_depth,
                    )
                    for name in model_fields
                },
            }
        raise UnsupportedSessionValue(path, value)

    def _claim_identity(self, value: object, path: str) -> None:
        value_id = id(value)
        previous = self._seen.get(value_id)
        if previous is not None:
            raise UnsupportedSessionValue(
                path,
                value,
                f"shared or cyclic object also appears at {previous[1]}",
            )
        # Retain temporary values so Python cannot recycle an already-seen ID.
        self._seen[value_id] = (value, path)


class SessionValueDecoder:
    """Decode only the fixed value tags owned by Kedi."""

    def __init__(
        self,
        *,
        kedi_models: Mapping[str, type[BaseModel]],
        decision_records: Mapping[str, DecisionRecord] | None = None,
    ) -> None:
        self._kedi_models = dict(kedi_models)
        self._decision_records = dict(decision_records or {})

    def decode(self, encoded: Any, *, path: str = "value", depth: int = 0) -> Any:
        if depth > MAX_VALUE_DEPTH:
            raise SessionLoadError(f"{path}: maximum nesting depth exceeded")
        if encoded is None or isinstance(encoded, (bool, int, float, str)):
            return encoded
        if not isinstance(encoded, dict) or not isinstance(encoded.get(_TAG), str):
            raise SessionLoadError(f"{path}: invalid encoded session value")
        tag = encoded[_TAG]
        next_depth = depth + 1
        try:
            if tag == "float":
                return float(encoded["value"])
            if tag == "bytes":
                return base64.b64decode(encoded["data"], validate=True)
            if tag == "media":
                media_class = {
                    "image": Image,
                    "audio": Audio,
                    "video": Video,
                    "document": Document,
                }[encoded["kind"]]
                data = encoded["data"]
                return media_class(
                    media_type=encoded["media_type"],
                    data=None if data is None else self.decode(data, path=path, depth=next_depth),
                    url=encoded["url"],
                    file_id=encoded["file_id"],
                    provider=encoded["provider"],
                    name=encoded["name"],
                    filename=encoded["filename"],
                )
            if tag == "surrogate_str":
                return base64.b64decode(encoded["data"], validate=True).decode(
                    "utf-8", "surrogatepass"
                )
            if tag == "complex":
                return complex(float(encoded["real"]), float(encoded["imag"]))
            if tag == "decimal":
                return Decimal(encoded["value"])
            if tag == "uuid":
                return UUID(encoded["value"])
            if tag == "datetime":
                return datetime.fromisoformat(encoded["value"]).replace(fold=encoded["fold"])
            if tag == "date":
                return date.fromisoformat(encoded["value"])
            if tag == "time":
                return time.fromisoformat(encoded["value"]).replace(fold=encoded["fold"])
            if tag == "timedelta":
                return timedelta(
                    days=encoded["days"],
                    seconds=encoded["seconds"],
                    microseconds=encoded["microseconds"],
                )
            if tag == "path":
                return Path(encoded["value"])
            if tag == "regex":
                pattern = self.decode(encoded["pattern"], path=f"{path}.pattern", depth=next_depth)
                return re.compile(pattern, encoded["flags"])
            if tag == "decision_info":
                record_id = encoded["record_id"]
                field_path = encoded["field_path"]
                if not isinstance(record_id, str) or not isinstance(field_path, list):
                    raise SessionLoadError(f"{path}: invalid decision info reference")
                if not all(isinstance(segment, str) for segment in field_path):
                    raise SessionLoadError(f"{path}: invalid decision info field path")
                record = self._decision_records.get(record_id)
                resolved_path = tuple(field_path)
                if record is None or record.output(resolved_path) is None:
                    raise SessionLoadError(f"{path}: unknown decision info reference")
                return DecisionInfo(record, resolved_path)
            if tag in {"list", "tuple", "set", "frozenset"}:
                items = [
                    self.decode(item, path=f"{path}[{index}]", depth=next_depth)
                    for index, item in enumerate(encoded["items"])
                ]
                if tag == "list":
                    return items
                if tag == "tuple":
                    return tuple(items)
                if tag == "set":
                    return set(items)
                return frozenset(items)
            if tag == "dict":
                result: dict[Any, Any] = {}
                for index, pair in enumerate(encoded["items"]):
                    if not isinstance(pair, list) or len(pair) != 2:
                        raise SessionLoadError(f"{path}: invalid encoded dict entry")
                    key = self.decode(pair[0], path=f"{path}.<key:{index}>", depth=next_depth)
                    if key in result:
                        raise SessionLoadError(f"{path}: duplicate decoded dict key {key!r}")
                    result[key] = self.decode(
                        pair[1],
                        path=f"{path}[{key!r}]",
                        depth=next_depth,
                    )
                return result
            if tag == "kedi_model":
                name = encoded["name"]
                model = self._kedi_models.get(name)
                if model is None:
                    raise SessionLoadError(f"{path}: unknown source-backed Kedi model {name!r}")
                values = {
                    name: self.decode(value, path=f"{path}.{name}", depth=next_depth)
                    for name, value in encoded["fields"].items()
                }
                return model.model_validate(values)
        except SessionLoadError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise SessionLoadError(f"{path}: invalid {tag!r} session value: {exc}") from exc
        raise SessionLoadError(f"{path}: unknown session value tag {tag!r}")


def encode_profile(
    profile: AgentProfile,
    *,
    encoder: SessionValueEncoder,
    model_names: Mapping[type[BaseModel], str],
    path: str,
) -> dict[str, Any]:
    if profile.tool_bindings:
        raise UnsupportedSessionValue(path + ".tool_bindings", profile.tool_bindings)
    if profile.subagent_bindings:
        raise UnsupportedSessionValue(path + ".subagent_bindings", profile.subagent_bindings)
    if profile.approval is not None and profile.approval.handler is not None:
        raise UnsupportedSessionValue(
            path + ".approval",
            profile.approval.handler,
            "dynamic approval handlers are process-local",
        )
    output_type: dict[str, str] | None = None
    if profile.output_type is not None:
        builtin_types = {
            bool: "bool",
            bytes: "bytes",
            complex: "complex",
            dict: "dict",
            float: "float",
            frozenset: "frozenset",
            int: "int",
            list: "list",
            set: "set",
            str: "str",
        }
        if profile.output_type in builtin_types:
            output_type = {"kind": "builtin", "name": builtin_types[profile.output_type]}
        elif profile.output_type in model_names:
            output_type = {"kind": "kedi", "name": model_names[profile.output_type]}
        else:
            raise UnsupportedSessionValue(path + ".output_type", profile.output_type)

    return {
        "name": profile.name,
        "model": profile.model,
        "adapter_kind": profile.adapter_kind,
        "adapter_shortname": profile.adapter_shortname,
        "agent_command": list(profile.agent_command)
        if isinstance(profile.agent_command, tuple)
        else profile.agent_command,
        "a2a_connection": _encode_a2a_connection(
            profile.a2a_connection,
            path=path + ".a2a_connection",
        ),
        "effort": profile.effort,
        "system": profile.system,
        "settings": encoder.encode(dict(profile.settings), path=path + ".settings"),
        "tools": list(profile.tools),
        "mcp_servers": [
            {
                "transport": server.transport,
                "command": server.command,
                "args": list(server.args),
                "env": server.env,
                "url": server.url,
                "headers": server.headers,
            }
            for server in profile.mcp_servers
        ],
        "approval": None if profile.approval is None else {"mode": profile.approval.mode},
        "hooks": _encode_hooks(profile.hooks, path=path + ".hooks"),
        "hook_enforcement": _encode_hooks(
            profile._hook_enforcement,
            path=path + ".hook_enforcement",
        ),
        "skills": _encode_skills(profile.skills),
        "subagents": list(profile.subagents),
        "max_agents": profile.max_agents,
        "workflow_mode": profile.workflow_mode,
        "workflow_tools": None if profile.workflow_tools is None else list(profile.workflow_tools),
        "history": profile.history,
        "compaction": _dataclass_values(profile.compaction),
        "codemode": _dataclass_values(profile.codemode),
        "output_type": output_type,
        "artifacts": _encode_artifact_policy(profile.artifacts),
    }


def decode_profile(
    encoded: Mapping[str, Any],
    *,
    decoder: SessionValueDecoder,
    kedi_models: Mapping[str, type[BaseModel]],
) -> AgentProfile:
    output_type_data = encoded.get("output_type")
    output_type: Any | None = None
    if output_type_data is not None:
        kind = output_type_data.get("kind")
        name = output_type_data.get("name")
        if kind == "builtin":
            try:
                output_type = {
                    "bool": bool,
                    "bytes": bytes,
                    "complex": complex,
                    "dict": dict,
                    "float": float,
                    "frozenset": frozenset,
                    "int": int,
                    "list": list,
                    "set": set,
                    "str": str,
                }[name]
            except KeyError as exc:
                raise SessionLoadError(f"Unknown builtin output type {name!r}") from exc
        elif kind == "kedi":
            output_type = kedi_models.get(name)
            if output_type is None:
                raise SessionLoadError(f"Unknown Kedi output type {name!r}")
        else:
            raise SessionLoadError("Invalid profile output type")

    command = encoded.get("agent_command")
    if isinstance(command, list):
        command = tuple(command)
    approval_data = encoded.get("approval")
    approval = None
    if approval_data is not None:
        mode = approval_data.get("mode")
        approval = ApprovalPolicy(mode=mode)
    return AgentProfile(
        name=encoded.get("name"),
        model=encoded.get("model"),
        adapter_kind=encoded.get("adapter_kind"),
        adapter_shortname=encoded.get("adapter_shortname"),
        agent_command=command,
        a2a_connection=_decode_a2a_connection(encoded.get("a2a_connection")),
        effort=encoded.get("effort"),
        system=encoded.get("system"),
        settings=decoder.decode(encoded["settings"], path="profile.settings"),
        tools=tuple(encoded.get("tools", ())),
        mcp_servers=tuple(_decode_mcp_server(item) for item in encoded.get("mcp_servers", ())),
        approval=approval,
        hooks=_decode_hooks(encoded.get("hooks")),
        _hook_enforcement=_decode_hooks(encoded.get("hook_enforcement")),
        skills=_decode_skills(encoded.get("skills")),
        subagents=tuple(encoded.get("subagents", ())),
        max_agents=encoded.get("max_agents"),
        workflow_mode=encoded.get("workflow_mode"),
        workflow_tools=(
            None if encoded.get("workflow_tools") is None else tuple(encoded["workflow_tools"])
        ),
        history=encoded.get("history"),
        compaction=_decode_dataclass(CompactionSettings, encoded.get("compaction")),
        codemode=_decode_dataclass(CodeModeSettings, encoded.get("codemode")),
        output_type=output_type,
        artifacts=_decode_artifact_policy(encoded.get("artifacts")),
    )


def _encode_a2a_connection(
    connection: A2AConnection | None,
    *,
    path: str,
) -> dict[str, Any] | None:
    if connection is None:
        return None
    auth = connection.auth
    if auth is not None and auth.provider is not None:
        raise UnsupportedSessionValue(
            path + ".auth.provider",
            auth.provider,
            "callable A2A credential providers are not persisted",
        )
    return {
        "endpoint": connection.endpoint,
        "request_timeout": connection.request_timeout,
        "task_timeout": connection.task_timeout,
        "poll_interval": connection.poll_interval,
        "auth": (
            None
            if auth is None
            else {
                "scheme": auth.scheme,
                "username": auth.username,
                "secret_env": auth.secret_env,
                "secret_file": auth.secret_file,
                "header": auth.header,
            }
        ),
    }


def _decode_a2a_connection(value: Any) -> A2AConnection | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise SessionLoadError("Invalid A2A connection in profile snapshot")
    auth_value = value.get("auth")
    auth = None
    if auth_value is not None:
        if not isinstance(auth_value, Mapping):
            raise SessionLoadError("Invalid A2A auth in profile snapshot")
        try:
            auth = A2AAuthConfig(
                scheme=auth_value["scheme"],
                username=auth_value.get("username"),
                secret_env=auth_value.get("secret_env"),
                secret_file=auth_value.get("secret_file"),
                header=auth_value.get("header"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise SessionLoadError("Invalid A2A auth in profile snapshot") from exc
    try:
        return A2AConnection(
            endpoint=value["endpoint"],
            auth=auth,
            request_timeout=value.get("request_timeout", 30.0),
            task_timeout=value.get("task_timeout", 600.0),
            poll_interval=value.get("poll_interval", 0.5),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SessionLoadError("Invalid A2A connection in profile snapshot") from exc


def _encode_hooks(value: HookSettings | None, *, path: str) -> dict[str, Any] | None:
    if value is None:
        return None
    encoded: dict[str, Any] = {"enabled": value.enabled}
    for event in (
        "user_prompt_submit",
        "pre_tool_use",
        "post_tool_use",
        "post_tool_use_failure",
    ):
        handlers = getattr(value, event)
        encoded[event] = (
            None
            if handlers is None
            else [
                _encode_importable_callable(handler, path=f"{path}.{event}[{index}]")
                for index, handler in enumerate(handlers)
            ]
        )
    return encoded


def _decode_hooks(value: Any) -> HookSettings | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise SessionLoadError("profile.hooks must be an object")
    enabled = value.get("enabled")
    if enabled is not None and type(enabled) is not bool:
        raise SessionLoadError("profile.hooks.enabled must be a boolean or null")
    decoded: dict[str, tuple[Any, ...] | None] = {}
    for event in (
        "user_prompt_submit",
        "pre_tool_use",
        "post_tool_use",
        "post_tool_use_failure",
    ):
        handlers = value.get(event)
        if handlers is None:
            decoded[event] = None
            continue
        if not isinstance(handlers, list):
            raise SessionLoadError(f"profile.hooks.{event} must be a list or null")
        decoded[event] = tuple(
            _decode_importable_callable(
                descriptor,
                path=f"profile.hooks.{event}[{index}]",
            )
            for index, descriptor in enumerate(handlers)
        )
    return HookSettings(enabled=enabled, **decoded)


def _encode_importable_callable(value: Any, *, path: str) -> dict[str, str]:
    module = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None)
    if inspect.ismethod(value) and value.__self__ is not None:
        raise UnsupportedSessionValue(path, value, "bound callables are not restorable")
    if not isinstance(module, str) or not isinstance(qualname, str):
        raise UnsupportedSessionValue(path, value, "callable has no import address")
    if module == "__main__":
        raise UnsupportedSessionValue(path, value, "__main__ callables are not restorable")
    if "<locals>" in qualname or "<lambda>" in qualname:
        raise UnsupportedSessionValue(path, value, "local functions and lambdas are not restorable")
    try:
        resolved = _resolve_importable_callable(module, qualname, path=path)
    except SessionLoadError as exc:
        raise UnsupportedSessionValue(path, value, str(exc)) from exc
    if resolved is not value:
        raise UnsupportedSessionValue(
            path,
            value,
            "import address does not resolve to the same callable",
        )
    return {"module": module, "qualname": qualname}


def _decode_importable_callable(value: Any, *, path: str) -> Any:
    if not isinstance(value, Mapping):
        raise SessionLoadError(f"{path}: callable descriptor must be an object")
    module = value.get("module")
    qualname = value.get("qualname")
    if not isinstance(module, str) or not isinstance(qualname, str):
        raise SessionLoadError(f"{path}: callable descriptor requires module and qualname")
    if module == "__main__" or "<locals>" in qualname or "<lambda>" in qualname:
        raise SessionLoadError(f"{path}: unsafe callable import address")
    return _resolve_importable_callable(module, qualname, path=path)


def _resolve_importable_callable(module: str, qualname: str, *, path: str) -> Any:
    try:
        resolved: Any = importlib.import_module(module)
        for part in qualname.split("."):
            if not part or part.startswith("<"):
                raise AttributeError(part)
            resolved = getattr(resolved, part)
    except (ImportError, AttributeError) as exc:
        raise SessionLoadError(f"{path}: cannot resolve callable {module}.{qualname}") from exc
    if not callable(resolved):
        raise SessionLoadError(f"{path}: resolved object is not callable")
    return resolved


def write_snapshot(path: str | Path, payload: Mapping[str, Any]) -> Path:
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    document = {
        "format": SNAPSHOT_FORMAT,
        "format_version": SNAPSHOT_FORMAT_VERSION,
        "kedi_version": __version__,
        "digest": digest,
        "payload": payload,
    }
    rendered = (_canonical_json(document) + "\n").encode("utf-8")
    if len(rendered) > MAX_SNAPSHOT_BYTES:
        raise SessionDumpError(
            [f"snapshot is {len(rendered)} bytes; limit is {MAX_SNAPSHOT_BYTES} bytes"]
        )
    _atomic_write(destination, rendered)
    return destination


def read_snapshot(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser()
    try:
        size = source.stat().st_size
        if size > MAX_SNAPSHOT_BYTES:
            raise SessionLoadError(
                f"Interactive session snapshot exceeds {MAX_SNAPSHOT_BYTES} bytes"
            )
        document = json.loads(source.read_text(encoding="utf-8"))
    except SessionLoadError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SessionLoadError(f"Cannot read interactive session snapshot: {exc}") from exc
    if not isinstance(document, dict):
        raise SessionLoadError("Interactive session snapshot root must be an object")
    if document.get("format") != SNAPSHOT_FORMAT:
        raise SessionLoadError("Not a Kedi interactive session snapshot")
    if document.get("format_version") != SNAPSHOT_FORMAT_VERSION:
        raise SessionLoadError(
            f"Unsupported interactive session snapshot version {document.get('format_version')!r}"
        )
    if document.get("kedi_version") != __version__:
        raise SessionLoadError(
            "Interactive session snapshot was created by Kedi "
            f"{document.get('kedi_version')!r}; current version is {__version__!r}"
        )
    payload = document.get("payload")
    if not isinstance(payload, dict):
        raise SessionLoadError("Interactive session snapshot payload must be an object")
    digest = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    if document.get("digest") != digest:
        raise SessionLoadError("Interactive session snapshot integrity check failed")
    return payload


def _encode_skills(value: SkillsSettings | bool | None) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, bool):
        value = SkillsSettings(enabled=value)
    return {
        "enabled": value.enabled,
        "cwd": None if value.cwd is None else str(value.cwd),
        "max_skills": value.max_skills,
        "include_registry": value.include_registry,
        "include_all": value.include_all,
        "exclude_paths": [str(path) for path in value.exclude_paths],
    }


def _decode_skills(value: Mapping[str, Any] | None) -> SkillsSettings | None:
    if value is None:
        return None
    return SkillsSettings(
        enabled=value["enabled"],
        cwd=None if value.get("cwd") is None else Path(value["cwd"]),
        max_skills=value["max_skills"],
        include_registry=value["include_registry"],
        include_all=value["include_all"],
        exclude_paths=tuple(Path(path) for path in value["exclude_paths"]),
    )


def _decode_mcp_server(value: Mapping[str, Any]) -> McpServerSpec:
    values = dict(value)
    values["args"] = tuple(values.get("args", ()))
    return McpServerSpec(**values)


def _encode_artifact_policy(value: ArtifactPolicy | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        item.name: (
            sorted(value._specified_fields)
            if item.name == "_specified_fields"
            else str(getattr(value, item.name))
            if item.name == "path"
            else getattr(value, item.name)
        )
        for item in fields(value)
    }


def _decode_artifact_policy(value: Mapping[str, Any] | None) -> ArtifactPolicy | None:
    if value is None:
        return None
    values = dict(value)
    values["path"] = Path(values["path"])
    values["_specified_fields"] = frozenset(values["_specified_fields"])
    return ArtifactPolicy(**values)


def _dataclass_values(
    value: CompactionSettings | CodeModeSettings | bool | None,
) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, bool):
        value = CodeModeSettings(enabled=value)
    return {item.name: getattr(value, item.name) for item in fields(type(value))}


def _decode_dataclass(cls: Any, value: Mapping[str, Any] | None) -> Any:
    return None if value is None else cls(**value)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _atomic_write(path: Path, payload: bytes) -> None:
    temporary: Path | None = None
    fd: int | None = None
    try:
        fd, raw_path = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        temporary = Path(raw_path)
        # mkstemp already creates owner-only files on POSIX; Windows uses ACLs.
        with os.fdopen(fd, "wb") as stream:
            fd = None
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
        if os.name != "nt":
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if fd is not None:
            os.close(fd)
        if temporary is not None:
            temporary.unlink(missing_ok=True)
