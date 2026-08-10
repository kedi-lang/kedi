from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, TypeAlias, cast

CompactionMode: TypeAlias = Literal["native", "disabled"]


class CompactionSettingsError(ValueError):
    """Raised when a Kedi compaction setting is invalid or unsupported."""


@dataclass(frozen=True, kw_only=True)
class CompactionSettings:
    """Provider compaction policy selected through an agent profile."""

    mode: CompactionMode = "native"
    threshold: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"native", "disabled"}:
            raise CompactionSettingsError("compaction mode must be 'native' or 'disabled'")
        if isinstance(self.threshold, bool) or (
            self.threshold is not None
            and (not isinstance(self.threshold, int) or self.threshold <= 0)
        ):
            raise CompactionSettingsError("compaction threshold must be a positive integer or none")
        if self.mode == "disabled" and self.threshold is not None:
            raise CompactionSettingsError(
                "compaction threshold cannot be set when mode is 'disabled'"
            )


CompactionSettingsInput: TypeAlias = CompactionSettings | Mapping[str, object] | None


def normalize_compaction_settings(
    value: CompactionSettingsInput,
) -> CompactionSettings | None:
    """Normalize the public settings mapping into a validated policy."""

    if value is None:
        return None
    if isinstance(value, CompactionSettings):
        return value
    if not isinstance(value, Mapping):
        raise CompactionSettingsError(
            "compaction must be a CompactionSettings value, mapping, or none"
        )
    unknown = set(value) - {"mode", "threshold"}
    if unknown:
        names = ", ".join(sorted(repr(name) for name in unknown))
        raise CompactionSettingsError(f"unknown compaction setting: {names}")
    mode = value.get("mode", "native")
    if not isinstance(mode, str):
        raise CompactionSettingsError("compaction mode must be a string")
    return CompactionSettings(
        mode=cast(CompactionMode, mode),
        threshold=cast(int | None, value.get("threshold")),
    )


def compaction_settings_from_profile(
    settings: Mapping[str, object],
) -> CompactionSettings | None:
    """Read the reserved compaction entry from an agent settings mapping."""

    value = cast(CompactionSettingsInput, settings.get("compaction"))
    return normalize_compaction_settings(value)
