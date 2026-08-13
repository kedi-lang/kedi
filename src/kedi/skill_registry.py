"""Installation surface for user-scoped one-file Kedi skills."""

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from .agent_profile.skills import MAX_SKILL_BYTES, SKILL_NAME_RE
from .package_registry import _run_git, kedi_home

_GITHUB_REPOSITORY_RE = re.compile(
    r"(?P<owner>[A-Za-z0-9](?:[A-Za-z0-9-]{0,38}))/(?P<repo>[A-Za-z0-9_.-]+)\Z"
)


@dataclass(frozen=True)
class InstalledSkill:
    name: str
    root: Path
    source_revision: str | None = None


def skill_registry_root() -> Path:
    return kedi_home() / "registry" / "skills"


def install_skill_path(source: Path) -> InstalledSkill:
    """Install a root ``SKILL.md`` from a local directory or file."""

    expanded = source.expanduser()
    if expanded.is_symlink():
        raise ValueError("Skill source must not be a symlink")
    skill_file = expanded if expanded.name == "SKILL.md" else expanded / "SKILL.md"
    content = _read_validated_skill_source(skill_file)
    resolved = skill_file.resolve(strict=True)
    name = resolved.parent.name
    if SKILL_NAME_RE.fullmatch(name) is None:
        raise ValueError(f"Invalid skill name derived from source directory: {name!r}")
    return _install_skill(name, content)


def install_skill_repo(repository: str) -> InstalledSkill:
    """Clone ``OWNER/REPOSITORY`` and install its root ``SKILL.md``."""

    match = _GITHUB_REPOSITORY_RE.fullmatch(repository.strip())
    if match is None:
        raise ValueError("Skill repository must use the OWNER/REPOSITORY form")
    repo = match.group("repo").removesuffix(".git")
    url = f"https://github.com/{match.group('owner')}/{repo}.git"
    with tempfile.TemporaryDirectory(prefix="kedi-skill-") as temporary:
        checkout = Path(temporary) / "source"
        _run_git(
            ["clone", "--depth", "1", "--filter=blob:none", "--no-tags", url, str(checkout)],
            error_context=f"Unable to clone skill repository {repository!r}",
        )
        revision = _run_git(
            ["-C", str(checkout), "rev-parse", "HEAD"],
            error_context=f"Unable to inspect skill repository {repository!r}",
        )
        skill_file = checkout / "SKILL.md"
        content = _read_validated_skill_source(skill_file)
        installed = _install_skill(repo, content)
        return InstalledSkill(installed.name, installed.root, revision)


def _read_validated_skill_source(path: Path) -> str:
    if path.is_symlink():
        raise ValueError("Skill SKILL.md must not be a symlink")
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Skill source must contain a root SKILL.md: {path.parent}"
        ) from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"Skill source must contain a root SKILL.md: {path.parent}")
    return _read_skill_source(resolved)


def _read_skill_source(path: Path) -> str:
    size = path.stat().st_size
    if size > MAX_SKILL_BYTES:
        raise ValueError(f"Skill exceeds the {MAX_SKILL_BYTES} byte content limit")
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Skill SKILL.md must be UTF-8 text") from exc


def _install_skill(name: str, content: str) -> InstalledSkill:
    if SKILL_NAME_RE.fullmatch(name) is None:
        raise ValueError(f"Invalid skill name: {name!r}")
    registry = skill_registry_root()
    registry_parent = registry.parent
    if registry_parent.is_symlink():
        raise ValueError("Kedi registry directory must not be a symlink")
    registry_parent.mkdir(parents=True, exist_ok=True)
    if registry.is_symlink():
        raise ValueError("Kedi skill registry must not be a symlink")
    registry.mkdir(exist_ok=True)
    target = registry / name
    if target.is_symlink():
        raise ValueError(f"Installed skill directory must not be a symlink: {target}")
    target.mkdir(exist_ok=True)
    destination = target / "SKILL.md"
    if destination.is_symlink():
        raise ValueError(f"Installed skill file must not be a symlink: {destination}")
    temporary = target / f".SKILL.md.{uuid4().hex}.tmp"
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return InstalledSkill(name=name, root=target.resolve())
