"""Audit installed package receipts against exact registry commit statuses."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .package_manifest import PACKAGE_MANIFEST_NAME, validate_package_name
from .package_registry import INSTALL_RECEIPT_NAME, package_registry_root
from .registry_api import HTTPRegistryAPIClient, RegistryAuditRecord

MAX_INSTALL_RECEIPT_BYTES = 64 * 1024
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_ACTIONABLE_STATUSES = frozenset({"yanked", "revoked", "unknown", "malformed"})


@dataclass(frozen=True)
class PackageAuditResult:
    package: str
    status: str
    version: str | None = None
    commit: str | None = None
    source_kind: str | None = None
    detail: str | None = None
    advisory_url: str | None = None


@dataclass(frozen=True)
class PackageAuditReport:
    results: tuple[PackageAuditResult, ...]

    @property
    def counts(self) -> Counter[str]:
        return Counter(result.status for result in self.results)

    @property
    def exit_code(self) -> int:
        return 1 if any(result.status in _ACTIONABLE_STATUSES for result in self.results) else 0


@dataclass(frozen=True)
class _InstalledRegistryReceipt:
    package: str
    version: str
    repository: str
    commit: str

    @property
    def identity(self) -> tuple[str, str, str]:
        return (self.package, self.repository, self.commit)


def audit_installed_packages(
    *,
    registry_root: Path | None = None,
    api_client: HTTPRegistryAPIClient | None = None,
) -> PackageAuditReport:
    """Inspect receipts without importing packages and check verified commits online."""
    root = (registry_root or package_registry_root()).expanduser().resolve()
    if not root.exists():
        return PackageAuditReport(())
    if not root.is_dir():
        return PackageAuditReport(
            (
                PackageAuditResult(
                    str(root), "malformed", detail="package registry is not a directory"
                ),
            )
        )

    immediate: list[PackageAuditResult] = []
    verified: list[_InstalledRegistryReceipt] = []
    for package_path in sorted(root.iterdir(), key=lambda path: path.name):
        if package_path.name.startswith("."):
            continue
        if package_path.is_symlink():
            immediate.append(
                PackageAuditResult(
                    package_path.name,
                    "malformed",
                    detail="installed package directory is a symlink",
                )
            )
            continue
        if not package_path.is_dir():
            continue
        parsed = _read_installed_receipt(package_path)
        if isinstance(parsed, PackageAuditResult):
            immediate.append(parsed)
        else:
            verified.append(parsed)

    if not verified:
        return PackageAuditReport(tuple(immediate))

    records = (api_client or HTTPRegistryAPIClient()).audit_records()
    record_by_identity = {record.identity: record for record in records}
    for receipt in verified:
        record = record_by_identity.get(receipt.identity)
        if record is None:
            immediate.append(
                PackageAuditResult(
                    receipt.package,
                    "unknown",
                    version=receipt.version,
                    commit=receipt.commit,
                    source_kind="registry",
                    detail="verified commit is absent from the registry audit index",
                )
            )
            continue
        immediate.append(_result_from_registry_record(receipt, record))
    immediate.sort(key=lambda result: result.package)
    return PackageAuditReport(tuple(immediate))


def _read_installed_receipt(
    package_path: Path,
) -> PackageAuditResult | _InstalledRegistryReceipt:
    package_name = package_path.name
    try:
        validate_package_name(package_name)
    except ValueError as exc:
        return PackageAuditResult(package_name, "malformed", detail=str(exc))

    receipt_path = package_path / INSTALL_RECEIPT_NAME
    if receipt_path.is_symlink() or not receipt_path.is_file():
        return PackageAuditResult(
            package_name,
            "malformed",
            detail="installed package receipt is missing or not a regular file",
        )
    try:
        if receipt_path.stat().st_size > MAX_INSTALL_RECEIPT_BYTES:
            raise ValueError("installed package receipt exceeds the maximum size")
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        return PackageAuditResult(package_name, "malformed", detail=str(exc))
    if not isinstance(payload, dict):
        return PackageAuditResult(package_name, "malformed", detail="receipt must be an object")
    try:
        receipt_package = _required_string(payload, "package")
        if receipt_package != package_name:
            raise ValueError("receipt package name differs from its installation directory")
        source_kind = _required_string(payload, "install_source_kind")
        _verify_installed_manifest(package_path, payload)
        if source_kind != "registry" or payload.get("verified") is not True:
            return PackageAuditResult(
                package_name,
                "unverified",
                version=_optional_string(payload.get("package_version")),
                commit=_optional_string(payload.get("source_revision")),
                source_kind=source_kind,
                detail="local and direct-Git installs have no registry attestation",
            )
        if payload.get("format_version") != 2:
            raise ValueError("verified registry receipt must use format version 2")
        version = _required_string(payload, "package_version")
        repository = _required_string(payload, "repository")
        commit = _required_string(payload, "verified_commit")
        if _COMMIT_RE.fullmatch(commit) is None:
            raise ValueError("receipt verified_commit is not a full lowercase Git commit ID")
        if payload.get("source_revision") != commit or payload.get("source_url") != repository:
            raise ValueError("receipt source provenance differs from its verified provenance")
        _required_string(payload, "package_manifest_path")
        _required_string(payload, "registry_url")
        _required_string(payload, "registry_revision")
        registry_digest = _required_string(payload, "registry_manifest_digest")
        installed_digest = _required_string(payload, "installed_manifest_digest")
        if re.fullmatch(r"sha256:[0-9a-f]{64}", registry_digest) is None:
            raise ValueError("receipt registry_manifest_digest is invalid")
        expected_installed_digest = f"sha256:{_required_string(payload, 'manifest_sha256')}"
        if installed_digest != expected_installed_digest:
            raise ValueError("receipt installed manifest digest is inconsistent")
    except (OSError, ValueError) as exc:
        return PackageAuditResult(package_name, "malformed", detail=str(exc))
    return _InstalledRegistryReceipt(package_name, version, repository, commit)


def _verify_installed_manifest(package_path: Path, payload: dict[str, Any]) -> None:
    manifest = package_path / PACKAGE_MANIFEST_NAME
    if manifest.is_symlink() or not manifest.is_file():
        raise ValueError("installed package manifest is missing or not a regular file")
    expected = _required_string(payload, "manifest_sha256")
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("receipt manifest_sha256 is invalid")
    if _sha256_file(manifest) != expected:
        raise ValueError("installed package manifest differs from its installation receipt")


def _result_from_registry_record(
    receipt: _InstalledRegistryReceipt,
    record: RegistryAuditRecord,
) -> PackageAuditResult:
    if record.version != receipt.version:
        return PackageAuditResult(
            receipt.package,
            "malformed",
            version=receipt.version,
            commit=receipt.commit,
            source_kind="registry",
            detail="registry audit version differs from the installed receipt",
        )
    return PackageAuditResult(
        receipt.package,
        record.status,
        version=receipt.version,
        commit=receipt.commit,
        source_kind="registry",
        detail=record.reason,
        advisory_url=record.advisory_url,
    )


def _required_string(payload: dict[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"receipt field {field!r} must be a non-empty string")
    return value


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
