"""What an installed package created, and removing it again.

A package leaves no stale files. Its install receipt (``.kedi-install.json``) carries a
``lifecycle`` record: what the installation created, and how and when each thing is removed.

- **Inside the package directory** (the source tree, the runtime directory its adapters write
  to): removed with the package, by ``kedi remove`` or by a reinstall that replaces it.
- **Python distributions added to Kedi's interpreter** by ``python_dependencies``: recorded,
  and removed only by ``kedi remove --with-python-dependencies``. Even then, Kedi keeps a
  distribution that another installed package recorded, that another installed distribution
  requires, or whose version changed since; it says which and why.
- **Distributions the installation upgraded** were there before it: Kedi records the change and
  never reverts it, since whatever installed them may depend on them.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from packaging.markers import default_environment
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

from .package_dependencies import installed_distributions, uninstall_python_distributions
from .package_environment import ENVIRONMENT_DIRECTORY, UV_SYNC
from .package_graph import describe, installed_dependents, installed_metadata
from .package_manifest import parse_kedi_requirement, validate_package_name

RUNTIME_DIRECTORY = ".kedi-runtime"
"""Inside an installed package: where its adapters keep run-time files (caches, homes)."""

REMOVED_WITH_PACKAGE = "with the package"
REMOVED_WITH_FLAG = (
    "by kedi remove --with-python-dependencies, unless another installed package recorded it or "
    "another installed distribution requires it"
)
NEVER_REVERTED = (
    "never: Kedi does not revert an upgrade of a distribution that was already installed"
)


def lifecycle_record(
    *,
    previous: dict[str, Any] | None,
    before: dict[str, str] | None,
    after: dict[str, str] | None,
    environment: bool = False,
    installed_as: str = "explicit",
) -> dict[str, Any]:
    """The ``lifecycle`` record of one installation.

    ``before`` and ``after`` are the interpreter's distributions around installing the
    ``python_dependencies`` (both ``None`` when there were none). A reinstall finds the
    distributions its first install added already present, so the record carries forward
    those still installed. ``installed_as`` is ``explicit`` for a package the user asked for and
    ``dependency`` for one installed because another needs it; ``kedi remove`` takes an orphaned
    dependency along."""
    added: dict[str, str] = {}
    current = after if after is not None else installed_distributions() if previous else {}
    for entry in python_additions(previous):
        if entry["name"] in current:
            added[entry["name"]] = current[entry["name"]]
    changed: list[dict[str, Any]] = list((previous or {}).get("changed") or [])
    if before is not None and after is not None:
        for name, version in after.items():
            if name not in before:
                added[name] = version
            elif before[name] != version and name not in added:
                changed.append(
                    {
                        "kind": "python_distribution",
                        "name": name,
                        "from": before[name],
                        "to": version,
                        "target": sys.executable,
                        "removed": NEVER_REVERTED,
                    }
                )
    created: list[dict[str, Any]] = [
        {"kind": "package_directory", "path": ".", "removed": REMOVED_WITH_PACKAGE},
        {"kind": "runtime_directory", "path": RUNTIME_DIRECTORY, "removed": REMOVED_WITH_PACKAGE},
    ]
    if environment:
        created.append(
            {
                "kind": "environment",
                "path": ENVIRONMENT_DIRECTORY,
                "command": "uv " + " ".join(UV_SYNC),
                "contains": "the environment (venv/) and its Python (python/, python-bin/)",
                "removed": REMOVED_WITH_PACKAGE,
            }
        )
    created += [
        {
            "kind": "python_distribution",
            "name": name,
            "version": version,
            "target": sys.executable,
            "removed": REMOVED_WITH_FLAG,
        }
        for name, version in sorted(added.items())
    ]
    return {"installed_as": installed_as, "created": created, "changed": changed}


def python_additions(record: dict[str, Any] | None) -> list[dict[str, Any]]:
    """The Python distributions a lifecycle record says the installation added."""
    return [
        entry
        for entry in (record or {}).get("created") or []
        if entry.get("kind") == "python_distribution" and entry.get("name") and entry.get("version")
    ]


def read_lifecycle(package_root: Path) -> dict[str, Any] | None:
    """An installed package's lifecycle record, or None for a receipt written before it existed."""
    from .package_registry import INSTALL_RECEIPT_NAME

    receipt = package_root / INSTALL_RECEIPT_NAME
    try:
        payload = json.loads(receipt.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    record = payload.get("lifecycle") if isinstance(payload, dict) else None
    return record if isinstance(record, dict) else None


@dataclass
class PackageRemoval:
    name: str
    paths: list[Path]
    """The package directory and an interrupted installation's leftovers."""
    lock: Path | None
    """The package's install lock file, deleted last."""


@dataclass
class RemovalPlan:
    packages: list[PackageRemoval]
    """The package asked for, then the dependencies it leaves orphaned."""
    uninstall: list[tuple[str, str]] = field(default_factory=list)
    kept: list[tuple[str, str]] = field(default_factory=list)
    """Recorded distributions left in place, and why."""
    notes: list[str] = field(default_factory=list)

    def lines(self, *, dry_run: bool) -> list[str]:
        verb = "Would remove" if dry_run else "Removed"
        lines = [
            f"{verb} {path}"
            for package in self.packages
            for path in [*package.paths, *([package.lock] if package.lock else [])]
        ]
        uninstall = "Would uninstall" if dry_run else "Uninstalled"
        lines += [
            f"{uninstall} Python distribution {name} {version}" for name, version in self.uninstall
        ]
        lines += [f"Kept Python distribution {name}: {reason}" for name, reason in self.kept]
        return lines + self.notes


def _package_removal(registry_root: Path, name: str) -> PackageRemoval:
    target = registry_root / name
    paths = [target] if target.exists() or target.is_symlink() else []
    paths += sorted(registry_root.glob(f".{name}.staging-*"))
    paths += sorted(registry_root.glob(f".{name}.backup-*"))
    lock = registry_root / f".{name}.lock"
    return PackageRemoval(name, paths, lock if lock.exists() else None)


def _kedi_dependencies(registry_root: Path, name: str) -> list[str]:
    metadata = installed_metadata(registry_root, name)
    return (
        [parse_kedi_requirement(line).name for line in metadata.kedi_dependencies]
        if metadata
        else []
    )


def plan_removal(
    package_name: str, *, registry_root: Path, with_python_dependencies: bool
) -> RemovalPlan:
    """What ``kedi remove`` deletes, without deleting anything.

    A package another installed package needs is not removed. The dependencies it was the
    last to need, installed only because something needed them, go with it."""
    name = validate_package_name(package_name)
    requested = _package_removal(registry_root, name)
    if not requested.paths and requested.lock is None:
        raise ValueError(f"Kedi package {name!r} is not installed in {registry_root}")
    dependents = installed_dependents(registry_root, name)
    if dependents:
        needs = "; ".join(describe(d, r) for d, r in dependents.items())
        raise ValueError(
            f"Kedi package {name!r} is needed by installed packages ({needs}); remove them first"
        )
    plan = RemovalPlan([requested])
    removing = [name]
    queue = _kedi_dependencies(registry_root, name)
    while queue:
        dependency = queue.pop(0)
        record = read_lifecycle(registry_root / dependency)
        if dependency in removing or (record or {}).get("installed_as") != "dependency":
            continue
        if any(d not in removing for d in installed_dependents(registry_root, dependency)):
            continue
        removing.append(dependency)
        plan.packages.append(_package_removal(registry_root, dependency))
        plan.notes.append(
            f"{dependency} goes too: it was installed as a dependency, and no installed package "
            "needs it any more"
        )
        queue += _kedi_dependencies(registry_root, dependency)

    recorded: list[dict[str, Any]] = []
    for package in removing:
        target = registry_root / package
        record = read_lifecycle(target) if target.is_dir() else None
        if record is None and target.is_dir():
            plan.notes.append(
                f"{package} was installed before Kedi recorded what packages add to Python; "
                "its Python dependencies, if any, are not known and stay in place"
            )
        for entry in (record or {}).get("changed") or []:
            plan.notes.append(
                f"Python distribution {entry['name']} was upgraded from {entry['from']} to "
                f"{entry['to']} by {package}; Kedi does not revert that"
            )
        recorded += python_additions(record)
    # A distribution is removed once, whichever of the packages recorded it.
    recorded = list({entry["name"]: entry for entry in recorded}.values())
    if not recorded:
        return plan
    if not with_python_dependencies:
        names = ", ".join(sorted({entry["name"] for entry in recorded}))
        plan.notes.append(
            f"Python distributions these packages added stay installed: {names}. "
            f"Run kedi remove {name} --with-python-dependencies to remove them as well"
        )
        return plan
    installed = installed_distributions()
    claimed = _claimed_by_other_packages(registry_root, set(removing))
    candidates: dict[str, str] = {}
    for entry in recorded:
        dist, version = entry["name"], entry["version"]
        if entry.get("target") != sys.executable:
            plan.kept.append(
                (dist, f"it was installed into another interpreter, {entry.get('target')}")
            )
        elif dist not in installed:
            plan.notes.append(f"Python distribution {dist} is no longer installed")
        elif installed[dist] != version:
            plan.kept.append(
                (
                    dist,
                    f"its version changed since this package installed it ({version} → {installed[dist]})",
                )
            )
        elif dist in claimed:
            plan.kept.append((dist, f"installed package {claimed[dist]} needs it"))
        else:
            candidates[dist] = version
    for dist, reason in _required_by_others(candidates, installed).items():
        plan.kept.append((dist, reason))
        candidates.pop(dist)
    plan.uninstall = sorted(candidates.items())
    return plan


def _claimed_by_other_packages(registry_root: Path, removing: set[str]) -> dict[str, str]:
    """The distributions another installed package recorded or requires. A package whose
    ``python_dependencies`` were already installed when it came records none of them, yet
    needs them."""
    from .package_manifest import PACKAGE_MANIFEST_NAME, load_package_manifest

    claimed: dict[str, str] = {}
    for root in sorted(registry_root.iterdir()) if registry_root.is_dir() else []:
        if root.name in removing or root.name.startswith(".") or not root.is_dir():
            continue
        for entry in python_additions(read_lifecycle(root)):
            claimed.setdefault(entry["name"], root.name)
        try:
            requirements = load_package_manifest(root / PACKAGE_MANIFEST_NAME).python_dependencies
        except (OSError, ValueError):
            continue  # not a package: nothing it needs
        # An installed manifest passed validation, so its requirements parse.
        for line in requirements:
            claimed.setdefault(canonicalize_name(Requirement(line).name), root.name)
    return claimed


def _required_by_others(candidates: dict[str, str], installed: dict[str, str]) -> dict[str, str]:
    """The candidates another installed distribution still requires, directly or through a
    candidate it keeps. Extras count whatever their condition: Kedi cannot know which extras
    were asked for, so it keeps more rather than break something."""
    import importlib.metadata

    environment: dict[str, str] = {key: str(value) for key, value in default_environment().items()}
    requires: dict[str, set[str]] = {}
    for distribution in importlib.metadata.distributions():
        owner = canonicalize_name(distribution.metadata["Name"] or "")
        if not owner:
            continue
        needed = requires.setdefault(owner, set())
        extras = [""] + list(distribution.metadata.get_all("Provides-Extra") or [])
        for line in distribution.requires or []:
            try:
                requirement = Requirement(line)
            except InvalidRequirement:
                continue
            if requirement.marker is None or any(
                requirement.marker.evaluate({**environment, "extra": extra}) for extra in extras
            ):
                needed.add(canonicalize_name(requirement.name))
    kept: dict[str, str] = {}
    changed = True
    while changed:
        changed = False
        for owner, needed in requires.items():
            if owner in candidates and owner not in kept:
                continue  # a candidate that goes away needs nothing
            for dist in needed:
                if dist in candidates and dist not in kept and dist != owner:
                    kept[dist] = f"installed distribution {owner} requires it"
                    changed = True
    return {dist: reason for dist, reason in kept.items() if dist in installed}


def remove(plan: RemovalPlan, *, registry_root: Path) -> None:
    """Carries out ``plan``: Python distributions first, so a failed uninstall leaves the package
    in place to try again, then the package and its leftovers."""
    from .package_registry import _clear_installed_module_cache, _package_install_lock, _remove_tree

    uninstall_python_distributions([name for name, _ in plan.uninstall])
    for package in plan.packages:
        with _package_install_lock(registry_root, package.name):
            for path in package.paths:
                if path.exists() or path.is_symlink():
                    _remove_tree(path)
        # The lock file is the last trace of the package. A concurrent install of the same name
        # recreates its own; nothing else holds this one once the removal is done.
        if package.lock is not None:
            package.lock.unlink(missing_ok=True)
    _clear_installed_module_cache()
