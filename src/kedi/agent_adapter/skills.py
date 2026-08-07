"""Project-local, one-file skill discovery for agent tool calls."""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pydantic_ai import format_as_xml

from kedi.artifacts.stream import ArtifactStream
from kedi.telemetry import (
    TelemetryAsset,
    asset_fingerprint,
    capture_enabled,
    record_asset,
)

from .tool_override import ToolSpec

_SKILL_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")
_SKILL_FILE_NAME = "SKILL.md"
MAX_SKILL_LIST_LIMIT = 100
MAX_SKILL_BYTES = 256 * 1024
_SKILL_READ_CHARS = 64 * 1024


class SkillStore:
    """Read plain ``SKILL.md`` files rooted at one project-local directory."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    @classmethod
    def for_current_project(cls) -> SkillStore:
        return cls(Path.cwd() / ".agents" / "skills")

    def list_skills(self, *, all: bool = False, limit: int = 20) -> list[str]:
        """Return deterministic skill identifiers without exposing paths or contents."""
        del all  # Reserved for future source/scope selection.
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            raise ValueError("skill list limit must be a positive integer")
        if limit > MAX_SKILL_LIST_LIMIT:
            raise ValueError(f"skill list limit cannot exceed {MAX_SKILL_LIST_LIMIT}")
        if not self.root.is_dir():
            return []

        names: list[str] = []
        for entry in self.root.iterdir():
            if not entry.is_dir() or _SKILL_NAME_RE.fullmatch(entry.name) is None:
                continue
            try:
                self._validate_skill_text(self._bounded_skill_file(entry.name))
            except (FileNotFoundError, PermissionError, ValueError):
                continue
            names.append(entry.name)
        return sorted(names)[:limit]

    def read_skill(self, *, skill_name: str) -> str:
        """Read a single valid, project-local plain ``SKILL.md`` file."""
        path = self._bounded_skill_file(skill_name)
        content = "".join(self._skill_chunks(path, skill_name=skill_name))
        self._record_skill_asset(skill_name, path, content)
        return content

    def stream_skill(self, *, skill_name: str) -> ArtifactStream[str]:
        """Return a lazy text transport for one validated skill file."""

        path = self._bounded_skill_file(skill_name)
        self._record_skill_asset(skill_name, path)

        return ArtifactStream.text(self._skill_chunks(path, skill_name=skill_name))

    def _record_skill_asset(
        self,
        skill_name: str,
        path: Path,
        content: str | None = None,
    ) -> None:
        active_content = (
            "".join(self._skill_chunks(path, skill_name=skill_name)) if content is None else content
        )
        source_path = str(path) if capture_enabled("source_path") else None
        snippet = active_content[:2048] if capture_enabled("source_snippet") else None
        record_asset(
            TelemetryAsset(
                kind="skill",
                local_id=skill_name,
                name=skill_name,
                source_locator=f"kedi:skill:{skill_name}",
                content_fingerprint=asset_fingerprint(active_content),
                content=active_content if capture_enabled("content") else None,
                metadata={
                    "source_path": source_path,
                    "source_snippet": snippet,
                    "byte_count": path.stat().st_size,
                },
                sensitivity="sensitive",
            )
        )

    def _skill_chunks(self, path: Path, *, skill_name: str) -> Iterator[str]:
        try:
            with path.open("r", encoding="utf-8") as handle:
                while chunk := handle.read(_SKILL_READ_CHARS):
                    yield chunk
        except UnicodeDecodeError as exc:
            raise ValueError(f"Skill {skill_name!r} must be UTF-8 text") from exc

    def _validate_skill_text(self, path: Path) -> None:
        for _ in self._skill_chunks(path, skill_name=path.parent.name):
            pass

    def _bounded_skill_file(self, skill_name: str) -> Path:
        path = self._skill_file(skill_name)
        if path.stat().st_size > MAX_SKILL_BYTES:
            raise ValueError(
                f"Skill {skill_name!r} exceeds the {MAX_SKILL_BYTES} byte content limit"
            )
        return path

    def _skill_file(self, skill_name: str) -> Path:
        if not isinstance(skill_name, str) or _SKILL_NAME_RE.fullmatch(skill_name) is None:
            raise ValueError(f"Invalid skill name: {skill_name!r}")
        path = self.root / skill_name / _SKILL_FILE_NAME
        try:
            resolved = path.resolve(strict=True)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Skill {skill_name!r} was not found") from exc
        if not resolved.is_relative_to(self.root):
            raise ValueError(f"Skill {skill_name!r} resolves outside the project skill root")
        if not resolved.is_file():
            raise FileNotFoundError(f"Skill {skill_name!r} was not found")
        return resolved


def skill_tool_specs(root: Path | None = None) -> tuple[ToolSpec, ToolSpec]:
    """Build the two read-only tool specs for one agent call scope."""
    store = SkillStore(root) if root is not None else SkillStore.for_current_project()

    def list_skills(*, all: bool = False, limit: int = 20) -> list[str]:
        return store.list_skills(all=all, limit=limit)

    def read_skill(*, skill_name: str) -> ArtifactStream[str]:
        return store.stream_skill(skill_name=skill_name)

    metadata: dict[str, Any] = {"kedi_tool_kind": "skill"}
    return (
        ToolSpec(
            fn=list_skills,
            name="list_skills",
            description="List available project-local SKILL.md instruction names.",
            json_schema={
                "type": "object",
                "properties": {
                    "all": {"type": "boolean", "default": False},
                    "limit": {"type": "integer", "default": 20, "minimum": 1},
                },
                "additionalProperties": False,
            },
            metadata=metadata,
            risk="read_only",
        ),
        ToolSpec(
            fn=read_skill,
            name="read_skill",
            description="Read the plain SKILL.md instructions for one listed skill name.",
            json_schema={
                "type": "object",
                "properties": {"skill_name": {"type": "string"}},
                "required": ["skill_name"],
                "additionalProperties": False,
            },
            metadata=metadata,
            risk="read_only",
            return_schema={"type": "string"},
        ),
    )


def skill_instructions() -> str:
    """Return the compact system instruction accompanying the skill tools."""
    return format_as_xml(
        {
            "availability": "Skills are focused task instructions available on demand.",
            "discovery": ("Use list_skills when a task may benefit from a reusable procedure."),
            "reading": ("Use read_skill with a listed skill name before following that skill."),
            "scope": (
                "A skill applies only to the current task. Do not claim to have read one unless "
                "read_skill returned its contents."
            ),
        },
        root_tag="SKILL_INSTRUCTIONS",
    )
