"""Configuration for scoped Kedi skill discovery."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Mapping

DEFAULT_MAX_SKILLS = 20
MAX_SKILLS = 100
MAX_SKILL_BYTES = 256 * 1024
SKILL_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")


class SkillsSettingsError(ValueError):
    """Raised when a skills directive produces an invalid configuration."""


@dataclass(frozen=True, kw_only=True)
class SkillsSettings:
    """Resolved skill discovery policy for one agent scope."""

    enabled: bool
    cwd: Path | None = None
    max_skills: int = DEFAULT_MAX_SKILLS
    include_registry: bool = True
    include_all: bool = False
    exclude_paths: tuple[Path, ...] = ()


def skills_settings(value: SkillsSettings | bool | None) -> SkillsSettings | None:
    """Normalize the public boolean shorthand to the full settings model."""

    if value is None:
        return value
    if isinstance(value, SkillsSettings):
        return normalize_skills_settings(
            {
                "cwd": value.cwd,
                "max_skills": value.max_skills,
                "include_registry": value.include_registry,
                "include_all": value.include_all,
                "exclude_paths": value.exclude_paths,
            },
            enabled=value.enabled,
        )
    if isinstance(value, bool):
        return SkillsSettings(enabled=value)
    raise SkillsSettingsError("skills must be true, false, or SkillsSettings")


def normalize_skills_settings(
    values: Mapping[str, Any],
    *,
    enabled: bool,
) -> SkillsSettings:
    """Validate evaluated directive fields and build an immutable policy."""

    cwd = _optional_path(values.get("cwd"), field_name="cwd")
    max_skills = values.get("max_skills", DEFAULT_MAX_SKILLS)
    if not isinstance(max_skills, int) or isinstance(max_skills, bool) or max_skills <= 0:
        raise SkillsSettingsError("skills max_skills must be a positive integer")
    if max_skills > MAX_SKILLS:
        raise SkillsSettingsError(f"skills max_skills cannot exceed {MAX_SKILLS}")

    include_registry = _boolean(
        values.get("include_registry", True),
        field_name="include_registry",
    )
    include_all = _boolean(values.get("include_all", False), field_name="include_all")
    exclude_paths = _path_list(values.get("exclude_paths", ()), field_name="exclude_paths")
    return SkillsSettings(
        enabled=enabled,
        cwd=cwd,
        max_skills=max_skills,
        include_registry=include_registry,
        include_all=include_all,
        exclude_paths=exclude_paths,
    )


def _boolean(value: Any, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise SkillsSettingsError(f"skills {field_name} must be true or false")
    return value


def _optional_path(value: Any, *, field_name: str) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, (str, Path)):
        raise SkillsSettingsError(f"skills {field_name} must be a path string")
    return Path(value).expanduser()


def _path_list(value: Any, *, field_name: str) -> tuple[Path, ...]:
    if not isinstance(value, (list, tuple)):
        raise SkillsSettingsError(f"skills {field_name} must be a list of path strings")
    paths: list[Path] = []
    for item in value:
        path = _optional_path(item, field_name=field_name)
        if path is None:
            raise SkillsSettingsError(f"skills {field_name} entries must be path strings")
        paths.append(path)
    return tuple(paths)
