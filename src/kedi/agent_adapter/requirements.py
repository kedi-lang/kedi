from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

from kedi.errors import KediExecutionError

from .meta import (
    AdapterCapabilities,
    AdapterCapability,
    adapter_capability_names,
)

RequirementStatus = Literal["satisfied", "missing", "unresolved"]


@dataclass(frozen=True, slots=True)
class RequirementCheck:
    name: AdapterCapability
    origin: Literal["explicit", "inferred"]
    status: RequirementStatus
    source_location: Any | None = None
    detail: str | None = None


class MissingAdapterCapabilityError(KediExecutionError):
    def __init__(
        self,
        *,
        adapter_name: str,
        requirement: AdapterCapability,
        source_location: Any | None = None,
    ) -> None:
        self.adapter_name = adapter_name
        self.requirement = requirement
        self.source_location = source_location
        super().__init__(
            f"Adapter {adapter_name!r} does not satisfy required capability {requirement!r}"
        )


def normalize_adapter_requirements(
    values: Sequence[str] | None,
) -> tuple[AdapterCapability, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes)):
        raise TypeError("adapter requirements must be a sequence of capability names")
    known = adapter_capability_names()
    normalized: list[AdapterCapability] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str):
            raise TypeError("adapter requirements must contain only strings")
        if value not in known:
            expected = ", ".join(sorted(known))
            raise ValueError(f"unknown adapter requirement {value!r}; expected one of: {expected}")
        if value not in seen:
            seen.add(value)
            normalized.append(cast(AdapterCapability, value))
    return tuple(normalized)


def check_adapter_requirements(
    *,
    capabilities: AdapterCapabilities | None,
    requirements: tuple[AdapterCapability, ...],
    origins: Mapping[str, Sequence[Any]] | None = None,
) -> tuple[RequirementCheck, ...]:
    checks: list[RequirementCheck] = []
    for requirement in requirements:
        source_locations = (origins or {}).get(requirement, ())
        source_location = source_locations[-1] if source_locations else None
        supported = (
            capabilities.capability_status(requirement) if capabilities is not None else None
        )
        status: RequirementStatus
        if supported is True:
            status = "satisfied"
        elif supported is None:
            status = "unresolved"
        else:
            status = "missing"
        checks.append(
            RequirementCheck(
                name=requirement,
                origin="explicit",
                status=status,
                source_location=source_location,
            )
        )
    return tuple(checks)


def validate_adapter_requirements(
    *,
    adapter: Any,
    requirements: tuple[AdapterCapability, ...],
    origins: Mapping[str, Sequence[Any]] | None = None,
) -> tuple[RequirementCheck, ...]:
    capabilities = getattr(adapter, "capabilities", None)
    checks = check_adapter_requirements(
        capabilities=capabilities if isinstance(capabilities, AdapterCapabilities) else None,
        requirements=requirements,
        origins=origins,
    )
    adapter_name = getattr(adapter, "shortname", type(adapter).__name__)
    for check in checks:
        if (
            check.status == "unresolved"
            and isinstance(capabilities, AdapterCapabilities)
            and check.name in capabilities.negotiated_capabilities
        ):
            continue
        if check.status != "satisfied":
            raise MissingAdapterCapabilityError(
                adapter_name=adapter_name,
                requirement=check.name,
                source_location=check.source_location,
            )
    return checks


__all__ = [
    "MissingAdapterCapabilityError",
    "RequirementCheck",
    "check_adapter_requirements",
    "normalize_adapter_requirements",
    "validate_adapter_requirements",
]
