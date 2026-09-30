"""Scoped one-file skill discovery for agent tool calls."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from pydantic_ai import format_as_xml

from kedi.agent_profile.skills import (
    DEFAULT_MAX_SKILLS,
    MAX_SKILL_BYTES,
    MAX_SKILLS,
    SKILL_NAME_RE,
    SkillsSettings,
)
from kedi.artifacts.stream import ArtifactStream
from kedi.package_registry import kedi_home
from kedi.telemetry import (
    TelemetryAsset,
    asset_fingerprint,
    capture_enabled,
    record_asset,
)

from .tools.specs import ToolSpec

_SKILL_FILE_NAME = "SKILL.md"
MAX_SKILL_LIST_LIMIT = 100
_SKILL_READ_CHARS = 64 * 1024


class SkillStore:
    """Read plain ``SKILL.md`` files from ordered, bounded roots."""

    def __init__(
        self,
        root: Path | Sequence[Path],
        *,
        include_all: bool = True,
        exclude_paths: Sequence[Path] = (),
        max_skills: int = DEFAULT_MAX_SKILLS,
    ) -> None:
        roots = (root,) if isinstance(root, Path) else tuple(root)
        if (
            not isinstance(max_skills, int)
            or isinstance(max_skills, bool)
            or not 1 <= max_skills <= MAX_SKILLS
        ):
            raise ValueError(f"max_skills must be an integer from 1 through {MAX_SKILLS}")
        self.roots = tuple(dict.fromkeys(item.expanduser().resolve() for item in roots))
        self.root = self.roots[0] if self.roots else Path.cwd().resolve()
        self.include_all = include_all
        self.exclude_paths = tuple(
            dict.fromkeys(item.expanduser().resolve() for item in exclude_paths)
        )
        self.max_skills = max_skills

    @classmethod
    def for_current_project(cls) -> SkillStore:
        return cls(Path.cwd() / ".agents" / "skills")

    def list_skills(self, *, all: bool = False, limit: int = 20) -> list[str]:
        """Return deterministic skill identifiers without exposing paths or contents."""
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            raise ValueError("skill list limit must be a positive integer")
        if limit > MAX_SKILL_LIST_LIMIT:
            raise ValueError(f"skill list limit cannot exceed {MAX_SKILL_LIST_LIMIT}")
        del all  # Source merging is configured by the scoped skills policy.
        return self._visible_names()[:limit]

    def _visible_names(self) -> list[str]:
        names: dict[str, None] = {}
        for root in self._active_roots():
            for name in self._valid_names(root):
                names.setdefault(name, None)
            if names and not self.include_all:
                break
        return sorted(names)[: self.max_skills]

    def read_skill(self, *, skill_name: str) -> str:
        """Read one valid plain ``SKILL.md`` from the configured sources."""
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
        self._validate_skill_text(path)
        if skill_name not in self._visible_names():
            raise FileNotFoundError(f"Skill {skill_name!r} was not found")
        return path

    def _skill_file(self, skill_name: str) -> Path:
        if not isinstance(skill_name, str) or SKILL_NAME_RE.fullmatch(skill_name) is None:
            raise ValueError(f"Invalid skill name: {skill_name!r}")
        for root in self._active_roots():
            path = root / skill_name / _SKILL_FILE_NAME
            if self._is_excluded(path):
                continue
            try:
                resolved = path.resolve(strict=True)
            except FileNotFoundError:
                continue
            if not resolved.is_relative_to(root):
                raise ValueError(f"Skill {skill_name!r} resolves outside its skill root")
            if resolved.is_file():
                return resolved
        raise FileNotFoundError(f"Skill {skill_name!r} was not found")

    def _active_roots(self) -> tuple[Path, ...]:
        roots = tuple(root for root in self.roots if root.is_dir() and not self._is_excluded(root))
        if self.include_all:
            return roots
        for root in roots:
            if self._valid_names(root):
                return (root,)
        return roots[:1]

    def _valid_names(self, root: Path) -> list[str]:
        names: list[str] = []
        for entry in root.iterdir():
            if (
                not entry.is_dir()
                or SKILL_NAME_RE.fullmatch(entry.name) is None
                or self._is_excluded(entry)
            ):
                continue
            try:
                path = self._skill_file_in_root(root, entry.name)
                self._validate_skill_text(path)
            except (FileNotFoundError, PermissionError, ValueError):
                continue
            names.append(entry.name)
        return names

    def _skill_file_in_root(self, root: Path, skill_name: str) -> Path:
        path = root / skill_name / _SKILL_FILE_NAME
        resolved = path.resolve(strict=True)
        if self._is_excluded(resolved) or not resolved.is_relative_to(root):
            raise ValueError(f"Skill {skill_name!r} resolves outside its skill root")
        if not resolved.is_file() or resolved.stat().st_size > MAX_SKILL_BYTES:
            raise FileNotFoundError(f"Skill {skill_name!r} was not found")
        return resolved

    def _is_excluded(self, path: Path) -> bool:
        resolved = path.expanduser().resolve()
        return any(
            resolved == excluded or excluded in resolved.parents for excluded in self.exclude_paths
        )


def resolve_skill_store(
    settings: SkillsSettings,
    *,
    base_dir: Path | None = None,
) -> SkillStore:
    """Resolve registry, project, and user skill roots in precedence order."""

    base = (base_dir or Path.cwd()).expanduser().resolve()
    configured_cwd = settings.cwd
    if configured_cwd is not None:
        base = (
            configured_cwd.resolve()
            if configured_cwd.is_absolute()
            else (base / configured_cwd).resolve()
        )
    roots: list[Path] = []
    if settings.include_registry:
        roots.append(kedi_home() / "registry" / "skills")
    roots.extend((base / ".agents" / "skills", Path.home() / ".agents" / "skills"))
    excludes = tuple(
        path.resolve() if path.is_absolute() else (base / path).resolve()
        for path in settings.exclude_paths
    )
    return SkillStore(
        roots,
        include_all=settings.include_all,
        exclude_paths=excludes,
        max_skills=settings.max_skills,
    )


def skill_tool_specs(
    root: Path | None = None,
    *,
    store: SkillStore | None = None,
    stream_results: bool = True,
) -> tuple[ToolSpec, ToolSpec]:
    """Build the two read-only tool specs for one agent call scope."""
    active_store = store or (
        SkillStore(root) if root is not None else SkillStore.for_current_project()
    )

    def list_skills(*, all: bool = False, limit: int = 20) -> list[str]:
        return active_store.list_skills(all=all, limit=limit)

    def read_skill(*, skill_name: str) -> str | ArtifactStream[str]:
        if stream_results:
            return active_store.stream_skill(skill_name=skill_name)
        return active_store.read_skill(skill_name=skill_name)

    metadata: dict[str, Any] = {"kedi_tool_kind": "skill"}
    return (
        ToolSpec(
            fn=list_skills,
            name="list_skills",
            description="List available scoped SKILL.md instruction names.",
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


def skill_instructions(*, store: SkillStore | None = None) -> str:
    """Return the compact system instruction accompanying the skill tools."""
    names = store.list_skills(limit=MAX_SKILL_LIST_LIMIT) if store is not None else None
    instructions: dict[str, Any] = {
        "availability": "Skills are focused task instructions available on demand.",
        "discovery": (
            "The available names below are an inventory, not the skill instructions. "
            "Read a relevant skill before answering a project-specific task it covers. "
            "Use list_skills only to refresh discovery; no listing or reading is needed "
            "for an unrelated task."
            if names is not None
            else "Use list_skills when a task may benefit from a reusable procedure."
        ),
        "reading": ("Use read_skill with a listed skill name before following that skill."),
        "scope": (
            "A skill applies only to the current task. Do not claim to have read one unless "
            "read_skill returned its contents."
        ),
    }
    if names is not None:
        instructions["available_skills"] = names
    return format_as_xml(
        instructions,
        root_tag="SKILL_INSTRUCTIONS",
    )
