"""Safe local installation and lookup seams for Kedi packages."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Protocol
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

from .lang import PackageMetadata
from .package_manifest import (
    PACKAGE_MANIFEST_NAME,
    load_package_manifest,
    validate_package_metadata,
    validate_package_name,
)
from .registry_api import HTTPRegistryAPIClient, RegistryPackageRecord

KEDI_HOME_ENV = "KEDI_HOME"
MOCK_REGISTRY_ROOT_ENV = "KEDI_REGISTRY_MOCK_ROOT"
INSTALL_RECEIPT_NAME = ".kedi-install.json"
GIT_TIMEOUT_SECONDS = 60
MAX_GIT_ERROR_CHARS = 8_192
MAX_PACKAGE_SOURCE_FILES = 10_000
MAX_PACKAGE_SOURCE_BYTES = 50 * 1024 * 1024
MAX_REGISTRY_MANIFEST_BYTES = 1024 * 1024
_INSTALL_SOURCE_KINDS = frozenset({"git", "local", "registry"})
_GIT_CREDENTIAL_URL_RE = re.compile(r"(https?://)[^\s/@]+(?::[^\s/@]*)?@")

_THREAD_LOCKS: dict[tuple[str, str], threading.Lock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


class PackageRegistryError(RuntimeError):
    """Base error for an invalid or unsafe package-registry entry."""


class CorruptInstalledPackageError(PackageRegistryError):
    """An installed registry package violates the package integrity contract."""


@dataclass(frozen=True)
class InstalledPackage:
    name: str
    root: Path
    source: Path
    source_revision: str | None = None


@dataclass(frozen=True)
class ResolvedPackageSource:
    """A manifest source that has been proven to remain inside its package root."""

    package_root: Path
    relative_path: Path
    path: Path


@dataclass(frozen=True)
class PreparedPackage:
    """Validated immutable installation inputs, prepared before registry mutation."""

    manifest_path: Path
    metadata: PackageMetadata
    source: ResolvedPackageSource


@dataclass
class _SourceCopyBudget:
    """Bound registry writes from one package source tree."""

    files: int = 0
    bytes: int = 0

    def consume_file(self, path: Path, size: int) -> None:
        if self.files >= MAX_PACKAGE_SOURCE_FILES:
            raise ValueError(
                "Kedi package source exceeds the maximum "
                f"of {MAX_PACKAGE_SOURCE_FILES} regular files"
            )
        if self.bytes + size > MAX_PACKAGE_SOURCE_BYTES:
            raise ValueError(
                f"Kedi package source exceeds the maximum of {MAX_PACKAGE_SOURCE_BYTES} bytes"
            )
        self.files += 1
        self.bytes += size


class PackageRegistryClient(Protocol):
    """Filesystem lookup seam retained for explicit test registries."""

    def package_root(self, package_name: str) -> Path:
        """Return a directory containing the package's ``package.kedi``."""
        ...


class MockPackageRegistryClient:
    """Filesystem-backed stand-in until the public package registry exists."""

    def __init__(self, root: Path):
        self.root = root.expanduser().resolve()

    def package_root(self, package_name: str) -> Path:
        name = validate_package_name(package_name)
        candidate = self.root / name
        _reject_symlink(candidate, "mock registry package directory")
        package_root = candidate.resolve()
        _require_contained(package_root, self.root, "mock registry package directory")
        if not (package_root / PACKAGE_MANIFEST_NAME).is_file():
            raise ModuleNotFoundError(
                f"Kedi package {package_name!r} is not available in the mock registry"
            )
        return package_root


def kedi_home() -> Path:
    """Return the configured absolute Kedi home.

    A registry is process-global state.  Accepting a relative value would make
    the selected registry change after ``os.chdir()``, so the override is
    intentionally absolute-only.
    """
    configured = os.getenv(KEDI_HOME_ENV)
    if configured:
        path = Path(configured).expanduser()
        if not path.is_absolute():
            raise ValueError(f"{KEDI_HOME_ENV} must be an absolute path")
    else:
        path = Path.home() / ".kedi"
    return path.resolve()


def package_registry_root() -> Path:
    return (kedi_home() / "registry").resolve()


def installed_package_root(package_name: str, *, registry_root: Path | None = None) -> Path:
    """Return the canonical location for one validated installed package."""
    name = validate_package_name(package_name)
    root = _resolved_registry_root(registry_root)
    return root / name


def resolve_package_source(
    package_root: Path,
    declared_source: str | None,
    *,
    error_type: type[Exception] = ValueError,
) -> ResolvedPackageSource:
    """Validate a package source directory without following package symlinks.

    The installer and installed-package resolver use this same boundary so a
    manifest cannot become more permissive after it has reached the registry.
    """
    root = _validated_package_root(package_root, error_type=error_type)
    relative_path = _package_source_relative_path(declared_source, error_type=error_type)

    _reject_symlinked_components(root, relative_path, "package source")
    source = (root / relative_path).resolve()
    _require_contained(source, root, "package source")
    _reject_symlink(root / relative_path, "package source")
    if not source.is_dir():
        raise FileNotFoundError(f"Kedi package source directory not found: {source}")

    main = source / "main.kedi"
    if not main.exists():
        raise FileNotFoundError(f"Kedi package source must contain main.kedi: {source}")
    _require_regular_file(main, "Kedi package source main.kedi", error_type=error_type)
    return ResolvedPackageSource(root, relative_path, source)


def _package_source_relative_path(
    declared_source: str | None,
    *,
    error_type: type[Exception],
) -> Path:
    """Validate a manifest source before a Git sparse checkout needs it."""
    if declared_source is None:
        raise error_type("package metadata requires a source field")

    # Git sparse-checkout interprets its input as patterns. A manifest source
    # is a directory name, not a sparse-checkout expression, so rejecting
    # pattern/control characters prevents a package from widening the
    # temporary checkout beyond its declared source tree.
    if any(character in declared_source for character in "*?[]!\\\r\n"):
        raise error_type("package source must not contain Git sparse-checkout pattern characters")

    relative_path = Path(declared_source)
    if (
        PurePosixPath(declared_source).is_absolute()
        or PureWindowsPath(declared_source).drive
        or relative_path == Path(".")
        or not relative_path.parts
        or ".." in relative_path.parts
    ):
        raise error_type(
            "package source must be a relative path naming a directory below the package directory"
        )
    return relative_path


def resolve_installed_package_file(
    source: ResolvedPackageSource,
    relative_path: Path,
) -> Path | None:
    """Resolve one installed module file while rejecting package-tree symlinks."""
    if relative_path.is_absolute() or not relative_path.parts or ".." in relative_path.parts:
        raise CorruptInstalledPackageError("installed package module path escapes its source tree")

    _reject_symlinked_components(source.path, relative_path, "installed package module")
    candidate = source.path / relative_path
    if not candidate.exists():
        return None
    _require_regular_file(
        candidate,
        "installed package module",
        error_type=CorruptInstalledPackageError,
    )
    resolved = candidate.resolve()
    _require_contained(resolved, source.path, "installed package module")
    return resolved


def prepare_package(
    manifest_path: Path,
    *,
    expected_name: str | None = None,
) -> PreparedPackage:
    """Parse and validate all package inputs before any registry state changes."""
    if manifest_path.is_symlink():
        raise ValueError("Kedi package manifest must not be a symlink")
    resolved_manifest = manifest_path.expanduser().resolve()
    metadata = load_package_manifest(resolved_manifest)
    validate_package_metadata(metadata, enforce_python_compatibility=True)
    name = validate_package_name(metadata.name)
    if expected_name is not None and name != validate_package_name(expected_name):
        raise ValueError(
            "Registry package name mismatch: "
            f"requested {expected_name!r}, manifest defines {metadata.name!r}"
        )
    source = resolve_package_source(resolved_manifest.parent, metadata.source)
    return PreparedPackage(resolved_manifest, metadata, source)


def install_package(
    manifest_path: Path,
    *,
    registry_root: Path | None = None,
    expected_name: str | None = None,
    source_kind: str = "local",
    source_url: str | None = None,
    source_revision: str | None = None,
    registry_record: RegistryPackageRecord | None = None,
) -> InstalledPackage:
    """Install a package with an atomic, per-package registry transaction."""
    if source_kind not in _INSTALL_SOURCE_KINDS:
        raise ValueError(f"Unknown Kedi package installation source kind: {source_kind!r}")
    prepared = prepare_package(manifest_path, expected_name=expected_name)
    if registry_record is not None:
        if source_kind != "registry":
            raise ValueError("Registry provenance may only be attached to registry installations")
        if prepared.metadata.name != registry_record.name:
            raise ValueError("Registry package name differs from the checked-out package manifest")
        if prepared.metadata.version != registry_record.version:
            raise ValueError(
                "Registry package version differs from the checked-out package manifest"
            )
    root = _resolved_registry_root(registry_root)
    root.mkdir(parents=True, exist_ok=True)
    target = installed_package_root(prepared.metadata.name, registry_root=root)
    staging = root / f".{prepared.metadata.name}.staging-{uuid4().hex}"

    try:
        staging.mkdir()
        _copy_regular_file(prepared.manifest_path, staging / PACKAGE_MANIFEST_NAME)
        destination_source = staging / prepared.source.relative_path
        _copy_source_tree(prepared.source.path, destination_source)
        _write_install_receipt(
            staging,
            prepared,
            source_kind=source_kind,
            source_url=source_url,
            source_revision=source_revision,
            registry_record=registry_record,
        )
        _commit_staged_package(root, prepared.metadata.name, staging, target)
    except Exception:
        if staging.exists() or staging.is_symlink():
            _remove_tree(staging)
        raise

    _clear_installed_module_cache()
    return InstalledPackage(
        prepared.metadata.name,
        target,
        target / prepared.source.relative_path,
        source_revision=source_revision,
    )


def _commit_staged_package(root: Path, package_name: str, staging: Path, target: Path) -> None:
    """Atomically replace one package while serialising same-package installers."""
    with _package_install_lock(root, package_name):
        backup = root / f".{package_name}.backup-{uuid4().hex}"
        moved_previous = False
        try:
            if target.exists() or target.is_symlink():
                target.replace(backup)
                moved_previous = True
            staging.replace(target)
        except Exception:
            if moved_previous and backup.exists() and not target.exists():
                backup.replace(target)
            raise
        else:
            # The new package is committed. A failed cleanup must not turn a
            # successful installation into an ambiguous error for the caller.
            if backup.exists() or backup.is_symlink():
                _remove_tree(backup, ignore_errors=True)


@contextmanager
def _package_install_lock(root: Path, package_name: str) -> Iterator[None]:
    key = (str(root), package_name)
    with _THREAD_LOCKS_GUARD:
        thread_lock = _THREAD_LOCKS.setdefault(key, threading.Lock())

    with thread_lock:
        lock_path = root / f".{package_name}.lock"
        with lock_path.open("a+", encoding="utf-8") as lock_file:
            _acquire_file_lock(lock_file.fileno())
            try:
                yield
            finally:
                _release_file_lock(lock_file.fileno())


def _acquire_file_lock(file_descriptor: int) -> None:
    """Use an advisory POSIX lock when the host exposes ``fcntl``."""
    try:
        from fcntl import LOCK_EX, flock
    except ImportError:  # pragma: no cover - exercised only on unsupported Windows installs.
        return
    flock(file_descriptor, LOCK_EX)


def _release_file_lock(file_descriptor: int) -> None:
    try:
        from fcntl import LOCK_UN, flock
    except ImportError:  # pragma: no cover - exercised only on unsupported Windows installs.
        return
    flock(file_descriptor, LOCK_UN)


def _write_install_receipt(
    staging: Path,
    prepared: PreparedPackage,
    *,
    source_kind: str,
    source_url: str | None,
    source_revision: str | None,
    registry_record: RegistryPackageRecord | None,
) -> None:
    receipt: dict[str, bool | int | str] = {
        "format_version": 2 if registry_record is not None else 1,
        "install_source_kind": source_kind,
        "package": prepared.metadata.name,
        "source": prepared.source.relative_path.as_posix(),
        "manifest_sha256": _sha256_file(prepared.manifest_path),
    }
    if source_url is not None:
        receipt["source_url"] = source_url
    if source_revision is not None:
        receipt["source_revision"] = source_revision
    if registry_record is not None:
        receipt.update(
            {
                "package_version": prepared.metadata.version or "0",
                "verified": True,
                "repository": registry_record.repository,
                "verified_commit": registry_record.verified_commit,
                "package_manifest_path": registry_record.package_manifest_path,
                "registry_url": registry_record.registry_url,
                "registry_manifest_digest": registry_record.registry_manifest_digest,
                "registry_revision": registry_record.registry_revision,
                "installed_manifest_digest": f"sha256:{_sha256_file(prepared.manifest_path)}",
            }
        )
    (staging / INSTALL_RECEIPT_NAME).write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_source_tree(
    source: Path,
    destination: Path,
    *,
    budget: _SourceCopyBudget | None = None,
) -> None:
    """Copy only regular files/directories and reject symlinks and special files."""
    budget = budget or _SourceCopyBudget()
    _reject_symlink(source, "package source")
    source_stat = source.lstat()
    if not stat.S_ISDIR(source_stat.st_mode):
        raise ValueError(f"package source is not a directory: {source}")
    destination.mkdir(parents=True, exist_ok=False)
    with os.scandir(source) as entries:
        for entry in sorted(entries, key=lambda item: item.name):
            entry_path = Path(entry.path)
            mode = entry.stat(follow_symlinks=False).st_mode
            if stat.S_ISLNK(mode):
                raise ValueError(f"package source must not contain symlinks: {entry_path}")
            destination_path = destination / entry.name
            if stat.S_ISDIR(mode):
                _copy_source_tree(entry_path, destination_path, budget=budget)
            elif stat.S_ISREG(mode):
                _copy_regular_file(entry_path, destination_path, budget=budget)
            else:
                raise ValueError(
                    f"package source must contain only regular files and directories: {entry_path}"
                )
    shutil.copystat(source, destination, follow_symlinks=False)


def _copy_regular_file(
    source: Path,
    destination: Path,
    *,
    budget: _SourceCopyBudget | None = None,
) -> None:
    _require_regular_file(source, "package file", error_type=ValueError)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(source, flags)
    try:
        source_stat = os.fstat(descriptor)
        if not stat.S_ISREG(source_stat.st_mode):
            raise ValueError(f"package file is not a regular file: {source}")
        if budget is not None:
            budget.consume_file(source, source_stat.st_size)
        with os.fdopen(descriptor, "rb", closefd=False) as input_file:
            with destination.open("xb") as output_file:
                shutil.copyfileobj(input_file, output_file)
    finally:
        os.close(descriptor)
    shutil.copystat(source, destination, follow_symlinks=False)


def _validated_package_root(
    package_root: Path,
    *,
    error_type: type[Exception],
) -> Path:
    _reject_symlink(package_root, "package directory", error_type=error_type)
    root = package_root.expanduser().resolve()
    if not root.is_dir():
        raise error_type(f"Kedi package directory not found: {root}")
    return root


def _resolved_registry_root(registry_root: Path | None) -> Path:
    root = registry_root.expanduser() if registry_root is not None else package_registry_root()
    return root.resolve()


def _require_contained(candidate: Path, root: Path, label: str) -> None:
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} must remain inside {root}") from exc


def _reject_symlinked_components(
    root: Path,
    relative_path: Path,
    label: str,
    *,
    error_type: type[Exception] = ValueError,
) -> None:
    current = root
    for part in relative_path.parts:
        current = current / part
        _reject_symlink(current, label, error_type=error_type)


def _reject_symlink(
    path: Path,
    label: str,
    *,
    error_type: type[Exception] = ValueError,
) -> None:
    if path.is_symlink():
        raise error_type(f"{label} must not be a symlink: {path}")


def _require_regular_file(
    path: Path,
    label: str,
    *,
    error_type: type[Exception],
) -> None:
    _reject_symlink(path, label, error_type=error_type)
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        raise FileNotFoundError(f"{label} not found: {path}") from None
    if not stat.S_ISREG(mode):
        raise error_type(f"{label} must be a regular file: {path}")


def _remove_tree(path: Path, *, ignore_errors: bool = False) -> None:
    if path.is_symlink() or path.is_file():
        try:
            path.unlink(missing_ok=True)
        except OSError:
            if not ignore_errors:
                raise
        return
    try:
        shutil.rmtree(path)
    except OSError:
        if not ignore_errors:
            raise


def _clear_installed_module_cache() -> None:
    # Kept local to avoid a package-registry/module-registry import cycle.
    from .module_registry import clear_installed_module_cache

    clear_installed_module_cache()


def _git_https_url(source: str) -> str:
    if not source.startswith("git+"):
        raise ValueError("Git package sources must start with 'git+'")
    parsed = urlsplit(source.removeprefix("git+"))
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.hostname.lower() != "github.com"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "Git package sources must use a credential-free https://github.com URL without query parameters"
        )
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def _run_git(args: list[str], *, error_context: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_SECONDS,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
    except FileNotFoundError as exc:
        raise ModuleNotFoundError("Git is required to install a git+ Kedi package") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"{error_context}: git did not finish within {GIT_TIMEOUT_SECONDS} seconds"
        ) from exc
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown git error"
        if len(detail) > MAX_GIT_ERROR_CHARS:
            detail = detail[-MAX_GIT_ERROR_CHARS:]
        raise RuntimeError(f"{error_context}: {_redact_git_diagnostic(detail)}")
    return result.stdout.strip()


def _redact_git_diagnostic(detail: str) -> str:
    return _GIT_CREDENTIAL_URL_RE.sub(r"\1<redacted>@", detail)


def install_git_package(source: str, *, registry_root: Path | None = None) -> InstalledPackage:
    """Clone an explicit git source and install its root ``package.kedi``."""
    url = _git_https_url(source)
    with tempfile.TemporaryDirectory(prefix="kedi-package-") as temporary:
        checkout = Path(temporary) / "checkout"
        _run_git(
            [
                "clone",
                "--depth",
                "1",
                "--no-tags",
                "--filter=blob:none",
                "--no-checkout",
                url,
                str(checkout),
            ],
            error_context=f"Could not clone Kedi package from {url}",
        )
        _run_git(
            [
                "-C",
                str(checkout),
                "sparse-checkout",
                "set",
                "--no-cone",
                f"/{PACKAGE_MANIFEST_NAME}",
            ],
            error_context=f"Could not read Kedi package metadata from {url}",
        )
        _run_git(
            ["-C", str(checkout), "checkout", "--quiet"],
            error_context=f"Could not check out Kedi package metadata from {url}",
        )
        manifest_path = checkout / PACKAGE_MANIFEST_NAME
        _require_regular_file(manifest_path, "Kedi package manifest", error_type=ValueError)
        metadata = load_package_manifest(manifest_path)
        validate_package_metadata(metadata, enforce_python_compatibility=True)
        source_path = _package_source_relative_path(metadata.source, error_type=ValueError)
        _run_git(
            [
                "-C",
                str(checkout),
                "sparse-checkout",
                "set",
                "--no-cone",
                f"/{PACKAGE_MANIFEST_NAME}",
                f"/{source_path.as_posix()}/**",
            ],
            error_context=f"Could not check out Kedi package source from {url}",
        )
        revision = _run_git(
            ["-C", str(checkout), "rev-parse", "HEAD"],
            error_context=f"Could not determine the commit for Kedi package from {url}",
        )
        return install_package(
            checkout / PACKAGE_MANIFEST_NAME,
            registry_root=registry_root,
            source_kind="git",
            source_url=url,
            source_revision=revision,
        )


def install_registry_package(
    package_name: str,
    *,
    registry_root: Path | None = None,
    api_client: HTTPRegistryAPIClient | None = None,
) -> InstalledPackage:
    """Install one active package at the registry's exact verified commit."""
    name = validate_package_name(package_name)
    selected_client = api_client or HTTPRegistryAPIClient()
    record = selected_client.package_record(name)
    if record.name != name:
        raise PackageRegistryError(
            f"Registry returned package {record.name!r} for requested package {name!r}"
        )
    if record.status != "active":
        raise PackageRegistryError(
            f"Kedi package {name!r} is {record.status} and cannot be installed"
        )
    repository = _registry_repository_url(
        record.repository,
        allow_local=selected_client.is_local,
    )
    with tempfile.TemporaryDirectory(prefix="kedi-registry-package-") as temporary:
        checkout = Path(temporary) / "checkout"
        manifest_path = _checkout_registry_commit(checkout, repository, record)
        return install_package(
            manifest_path,
            registry_root=registry_root,
            expected_name=name,
            source_kind="registry",
            source_url=repository,
            source_revision=record.verified_commit,
            registry_record=record,
        )


def _checkout_registry_commit(
    checkout: Path,
    repository: str,
    record: RegistryPackageRecord,
) -> Path:
    _run_git(["init", "--quiet", str(checkout)], error_context="Could not initialize checkout")
    _run_git(
        ["-C", str(checkout), "remote", "add", "origin", repository],
        error_context=f"Could not configure Kedi package repository {repository}",
    )
    _run_git(
        [
            "-C",
            str(checkout),
            "-c",
            "protocol.file.allow=always",
            "fetch",
            "--quiet",
            "--depth",
            "1",
            "--no-tags",
            "--filter=blob:none",
            "origin",
            record.verified_commit,
        ],
        error_context=(
            f"Could not fetch verified commit {record.verified_commit} from {repository}"
        ),
    )
    fetched_commit = _run_git(
        ["-C", str(checkout), "rev-parse", "FETCH_HEAD"],
        error_context="Could not verify the fetched Kedi package commit",
    )
    if fetched_commit != record.verified_commit:
        raise PackageRegistryError(
            "Fetched Kedi package commit differs from the registry's verified commit"
        )

    manifest_object = f"{record.verified_commit}:{record.package_manifest_path}"
    manifest_size_text = _run_git(
        ["-C", str(checkout), "cat-file", "-s", manifest_object],
        error_context="Could not inspect the verified package manifest",
    )
    try:
        manifest_size = int(manifest_size_text)
    except ValueError as exc:
        raise PackageRegistryError("Git returned an invalid package manifest size") from exc
    if manifest_size > MAX_REGISTRY_MANIFEST_BYTES:
        raise PackageRegistryError(
            f"Registry package manifest exceeds {MAX_REGISTRY_MANIFEST_BYTES} bytes"
        )
    manifest_text = _run_git(
        ["-C", str(checkout), "show", manifest_object],
        error_context="Could not read the verified package manifest",
    )
    preview_manifest = Path(checkout.parent) / PACKAGE_MANIFEST_NAME
    preview_manifest.write_text(f"{manifest_text}\n", encoding="utf-8")
    metadata = load_package_manifest(preview_manifest)
    validate_package_metadata(metadata, enforce_python_compatibility=True)
    if metadata.name != record.name or metadata.version != record.version:
        raise PackageRegistryError(
            "Registry package metadata differs from the verified package manifest"
        )
    package_source = _package_source_relative_path(metadata.source, error_type=PackageRegistryError)
    manifest_relative = PurePosixPath(record.package_manifest_path)
    repository_source = manifest_relative.parent / PurePosixPath(package_source.as_posix())
    if any(character in repository_source.as_posix() for character in "*?["):
        raise PackageRegistryError("Registry package source path must not contain glob syntax")

    _run_git(
        ["-C", str(checkout), "sparse-checkout", "init", "--no-cone"],
        error_context="Could not initialize sparse checkout for the Kedi package",
    )
    _run_git(
        [
            "-C",
            str(checkout),
            "sparse-checkout",
            "set",
            "--no-cone",
            f"/{manifest_relative.as_posix()}",
            f"/{repository_source.as_posix()}/**",
        ],
        error_context="Could not select the verified Kedi package source",
    )
    _run_git(
        ["-C", str(checkout), "checkout", "--detach", "--quiet", record.verified_commit],
        error_context="Could not check out the verified Kedi package commit",
    )
    checked_out_commit = _run_git(
        ["-C", str(checkout), "rev-parse", "HEAD"],
        error_context="Could not verify the checked-out Kedi package commit",
    )
    if checked_out_commit != record.verified_commit:
        raise PackageRegistryError(
            "Checked-out Kedi package commit differs from the registry's verified commit"
        )
    manifest_path = checkout.joinpath(*manifest_relative.parts)
    _reject_symlinked_components(
        checkout,
        Path(*manifest_relative.parts),
        "registry package manifest",
        error_type=PackageRegistryError,
    )
    _require_regular_file(
        manifest_path,
        "Registry package manifest",
        error_type=PackageRegistryError,
    )
    return manifest_path


def _registry_repository_url(value: str, *, allow_local: bool) -> str:
    parsed = urlsplit(value)
    if (
        parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise PackageRegistryError(
            "Registry repository URL must not contain credentials, query, or fragment"
        )
    if parsed.scheme == "https" and parsed.hostname:
        return value
    if allow_local and parsed.scheme == "file" and parsed.hostname in {None, "", "localhost"}:
        # URL paths use forward slashes even for Windows file:///C:/... URLs.
        if PurePosixPath(parsed.path).is_absolute():
            return value
    if (
        allow_local
        and parsed.scheme == "http"
        and parsed.hostname
        in {
            "localhost",
            "127.0.0.1",
            "::1",
        }
    ):
        return value
    raise PackageRegistryError(
        "Registry repositories must use credential-free HTTPS; local registries may use "
        "loopback HTTP or absolute file URLs"
    )


def mock_registry_client_from_environment() -> PackageRegistryClient:
    configured = os.getenv(MOCK_REGISTRY_ROOT_ENV)
    if not configured:
        raise RuntimeError(
            f"Set {MOCK_REGISTRY_ROOT_ENV} to a directory of package sources "
            "before requesting a mock registry client."
        )
    return MockPackageRegistryClient(Path(configured))


def add_package(
    package_name_or_source: str,
    *,
    registry_root: Path | None = None,
    client: PackageRegistryClient | None = None,
) -> InstalledPackage:
    """Install a package from the registry or an explicit ``git+https`` source."""
    if package_name_or_source.startswith("git+"):
        return install_git_package(package_name_or_source, registry_root=registry_root)
    package_name = validate_package_name(package_name_or_source)
    selected_client = client
    if selected_client is None and os.getenv(MOCK_REGISTRY_ROOT_ENV):
        selected_client = mock_registry_client_from_environment()
    if selected_client is not None:
        package_root = selected_client.package_root(package_name)
        return install_package(
            package_root / PACKAGE_MANIFEST_NAME,
            registry_root=registry_root,
            expected_name=package_name,
            source_kind="registry",
        )
    return install_registry_package(package_name, registry_root=registry_root)
