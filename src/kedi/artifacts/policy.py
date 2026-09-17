from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from datetime import timedelta
from pathlib import Path
from typing import Any, Literal, TypeAlias

from .errors import ArtifactPolicyError

ArtifactStoreKind: TypeAlias = Literal["memory", "file"]

_ALL_POLICY_FIELDS = frozenset(
    {
        "enabled",
        "query_artifacts",
        "store",
        "path",
        "threshold_bytes",
        "ttl_seconds",
        "idle_ttl_seconds",
        "preview_chars",
        "read_max_chars",
        "session_quota_bytes",
        "max_artifacts",
        "cleanup_interval_seconds",
    }
)
_INPUT_ALIASES = {
    "threshold": "threshold_bytes",
    "ttl": "ttl_seconds",
    "idle_ttl": "idle_ttl_seconds",
    "session_quota": "session_quota_bytes",
    "cleanup_interval": "cleanup_interval_seconds",
}
ARTIFACT_POLICY_INPUT_FIELDS = frozenset(
    {
        "enabled",
        "query_artifacts",
        "store",
        "path",
        "threshold",
        "ttl",
        "idle_ttl",
        "preview_chars",
        "read_max_chars",
        "session_quota",
        "max_artifacts",
        "cleanup_interval",
    }
)
_BYTE_UNITS = {
    "b": 1,
    "kb": 1_000,
    "mb": 1_000_000,
    "gb": 1_000_000_000,
    "kib": 1_024,
    "mib": 1_048_576,
    "gib": 1_073_741_824,
}
_DURATION_UNITS = {
    "ms": 0.001,
    "s": 1.0,
    "m": 60.0,
    "h": 3_600.0,
    "d": 86_400.0,
}
_NUMBER_WITH_UNIT = re.compile(r"^\s*(?P<number>(?:\d+(?:\.\d*)?|\.\d+))\s*(?P<unit>[a-zA-Z]+)\s*$")


@dataclass(frozen=True, kw_only=True)
class ArtifactPolicy:
    """Normalized artifact policy with field-aware overlay semantics."""

    enabled: bool = True
    # Unset inherits the adapter default; a resolved standalone policy defaults to disabled.
    query_artifacts: Literal["enabled", "disabled"] | None = None
    store: ArtifactStoreKind = "memory"
    path: Path = Path(".kedi/artifacts")
    threshold_bytes: int = 100_000
    ttl_seconds: float = 3_600.0
    idle_ttl_seconds: float | None = None
    preview_chars: int = 1_200
    read_max_chars: int = 4_000
    session_quota_bytes: int = 256_000_000
    max_artifacts: int = 128
    cleanup_interval_seconds: float = 60.0
    _specified_fields: frozenset[str] = field(
        default=_ALL_POLICY_FIELDS,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        _validate_policy(self)

    @classmethod
    def patch(cls, values: Mapping[str, object]) -> ArtifactPolicy:
        normalized: dict[str, Any] = {}
        specified: set[str] = set()
        for input_name, value in values.items():
            name = _INPUT_ALIASES.get(input_name, input_name)
            if name not in _ALL_POLICY_FIELDS:
                raise ArtifactPolicyError(f"unknown artifact policy field {input_name!r}")
            normalized[name] = _normalize_policy_value(name, value)
            specified.add(name)
        return cls(**normalized, _specified_fields=frozenset(specified))

    def overlay(self, other: ArtifactPolicy) -> ArtifactPolicy:
        updates = {
            item.name: getattr(other, item.name)
            for item in fields(self)
            if item.name in other._specified_fields
        }
        specified = self._specified_fields | other._specified_fields
        return replace(self, **updates, _specified_fields=specified)

    def resolved(self) -> ArtifactPolicy:
        return replace(
            self,
            query_artifacts=self.query_artifacts or "disabled",
            _specified_fields=_ALL_POLICY_FIELDS,
        )

    def queries_enabled(self, *, default: bool = False) -> bool:
        return default if self.query_artifacts is None else self.query_artifacts == "enabled"

    @property
    def is_explicit(self) -> bool:
        return bool(self._specified_fields)

    def as_cache_key(self) -> tuple[object, ...]:
        return (
            self.enabled,
            self.query_artifacts,
            self.store,
            str(self.path),
            self.threshold_bytes,
            self.ttl_seconds,
            self.idle_ttl_seconds,
            self.preview_chars,
            self.read_max_chars,
            self.session_quota_bytes,
            self.max_artifacts,
            self.cleanup_interval_seconds,
        )


ArtifactPolicyInput: TypeAlias = ArtifactPolicy | Mapping[str, object] | bool | None


def normalize_artifact_policy(
    value: ArtifactPolicyInput,
    *,
    base: ArtifactPolicy | None = None,
) -> ArtifactPolicy | None:
    if value is None:
        return base
    if isinstance(value, bool):
        policy = ArtifactPolicy(enabled=value)
    elif isinstance(value, ArtifactPolicy):
        policy = value
    elif isinstance(value, Mapping):
        policy = ArtifactPolicy.patch(value)
    else:
        raise ArtifactPolicyError("artifacts must be an ArtifactPolicy, mapping, boolean, or None")
    return base.overlay(policy) if base is not None else policy


def parse_byte_size(value: object, *, field_name: str = "size") -> int:
    if isinstance(value, bool):
        raise ArtifactPolicyError(f"{field_name} must be a byte size, not bool")
    if isinstance(value, int):
        if value < 0:
            raise ArtifactPolicyError(f"{field_name} must not be negative")
        return value
    if isinstance(value, float):
        if value < 0 or not value.is_integer():
            raise ArtifactPolicyError(f"{field_name} numeric values must be whole bytes")
        return int(value)
    if not isinstance(value, str):
        raise ArtifactPolicyError(f"{field_name} must be bytes or a value such as '100kb'")
    match = _NUMBER_WITH_UNIT.match(value)
    if match is None:
        raise ArtifactPolicyError(f"{field_name} must include a supported byte unit")
    unit = match.group("unit").lower()
    multiplier = _BYTE_UNITS.get(unit)
    if multiplier is None:
        raise ArtifactPolicyError(f"unsupported {field_name} unit {match.group('unit')!r}")
    result = float(match.group("number")) * float(multiplier)
    if result != int(result):
        raise ArtifactPolicyError(f"{field_name} must resolve to whole bytes")
    return int(result)


def parse_duration(
    value: object,
    *,
    field_name: str = "duration",
    allow_none: bool = False,
) -> float | None:
    if value is None and allow_none:
        return None
    if isinstance(value, str) and value.strip().lower() in {"none", "null"}:
        if allow_none:
            return None
        raise ArtifactPolicyError(f"{field_name} cannot be none")
    if isinstance(value, bool):
        raise ArtifactPolicyError(f"{field_name} must be a duration, not bool")
    if isinstance(value, timedelta):
        seconds = value.total_seconds()
    elif isinstance(value, (int, float)):
        seconds = float(value)
    elif isinstance(value, str):
        match = _NUMBER_WITH_UNIT.match(value)
        if match is None:
            raise ArtifactPolicyError(f"{field_name} must be seconds or a value such as '1h'")
        unit = match.group("unit").lower()
        multiplier = _DURATION_UNITS.get(unit)
        if multiplier is None:
            raise ArtifactPolicyError(f"unsupported {field_name} unit {match.group('unit')!r}")
        seconds = float(match.group("number")) * multiplier
    else:
        raise ArtifactPolicyError(f"{field_name} must be seconds, timedelta, or a duration string")
    if seconds <= 0:
        raise ArtifactPolicyError(f"{field_name} must be greater than zero")
    return seconds


def _normalize_policy_value(name: str, value: object) -> object:
    if name == "query_artifacts":
        if not isinstance(value, str) or value not in {"enabled", "disabled"}:
            raise ArtifactPolicyError("query_artifacts must be 'enabled' or 'disabled'")
        return value
    if name in {"threshold_bytes", "session_quota_bytes"}:
        return parse_byte_size(value, field_name=name)
    if name in {"ttl_seconds", "cleanup_interval_seconds"}:
        return parse_duration(value, field_name=name)
    if name == "idle_ttl_seconds":
        return parse_duration(value, field_name=name, allow_none=True)
    if name in {"preview_chars", "read_max_chars", "max_artifacts"}:
        return _positive_int(value, field_name=name)
    if name == "enabled":
        if not isinstance(value, bool):
            raise ArtifactPolicyError("enabled must be a boolean")
        return value
    if name == "store":
        if value not in {"memory", "file"}:
            raise ArtifactPolicyError("store must be 'memory' or 'file'")
        return value
    if name == "path":
        if not isinstance(value, (str, Path)):
            raise ArtifactPolicyError("path must be a string or pathlib.Path")
        return Path(value)
    raise ArtifactPolicyError(
        f"unsupported normalized artifact policy field {name!r}"
    )  # pragma: no cover


def _positive_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ArtifactPolicyError(f"{field_name} must be a positive integer")
    return value


def _validate_policy(policy: ArtifactPolicy) -> None:
    if policy.query_artifacts is not None:
        _normalize_policy_value("query_artifacts", policy.query_artifacts)
    if policy.store not in {"memory", "file"}:
        raise ArtifactPolicyError("store must be 'memory' or 'file'")
    if policy.threshold_bytes < 0:
        raise ArtifactPolicyError("threshold_bytes must not be negative")
    if policy.ttl_seconds <= 0:
        raise ArtifactPolicyError("ttl_seconds must be greater than zero")
    if policy.idle_ttl_seconds is not None and policy.idle_ttl_seconds <= 0:
        raise ArtifactPolicyError("idle_ttl_seconds must be greater than zero")
    for name in ("preview_chars", "read_max_chars", "max_artifacts"):
        if getattr(policy, name) <= 0:
            raise ArtifactPolicyError(f"{name} must be a positive integer")
    if policy.session_quota_bytes <= 0:
        raise ArtifactPolicyError("session_quota_bytes must be greater than zero")
    if policy.cleanup_interval_seconds <= 0:
        raise ArtifactPolicyError("cleanup_interval_seconds must be greater than zero")
    if not policy._specified_fields <= _ALL_POLICY_FIELDS:
        raise ArtifactPolicyError("artifact policy contains unknown specified fields")
