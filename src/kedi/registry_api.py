"""Strict HTTP models and transport for the Kedi package registry."""

from __future__ import annotations

import ipaddress
import json
import os
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urljoin, urlsplit
from urllib.request import Request, urlopen

from .package_manifest import validate_package_name

REGISTRY_URL_ENV = "KEDI_REGISTRY_URL"
DEFAULT_REGISTRY_URL = "https://registry.kedi-lang.org"
REGISTRY_REQUEST_TIMEOUT_SECONDS = 10
MAX_REGISTRY_RESPONSE_BYTES = 2 * 1024 * 1024

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_PACKAGE_STATUSES = frozenset({"active", "yanked", "revoked"})
_AUDIT_STATUSES = frozenset({"active", "superseded", "yanked", "revoked"})


class RegistryAPIError(RuntimeError):
    """A registry response could not be fetched or trusted."""


@dataclass(frozen=True)
class RegistryPackageRecord:
    schema_version: int
    name: str
    version: str
    description: str
    author_name: str
    author_contact: str
    repository: str
    verified_commit: str
    package_manifest_path: str
    status: str
    registry_manifest_digest: str
    registry_revision: str
    registry_url: str


@dataclass(frozen=True)
class RegistryAuditRecord:
    name: str
    version: str
    repository: str
    verified_commit: str
    status: str
    reason: str | None = None
    advisory_url: str | None = None

    @property
    def identity(self) -> tuple[str, str, str]:
        return (self.name, self.repository, self.verified_commit)


class HTTPRegistryAPIClient:
    """Fetch strict, bounded static registry records over HTTP(S)."""

    def __init__(self, base_url: str | None = None):
        self.base_url = _validate_registry_base_url(
            base_url or os.getenv(REGISTRY_URL_ENV) or DEFAULT_REGISTRY_URL
        )
        self.is_local = _is_loopback_hostname(urlsplit(self.base_url).hostname)

    def package_record(self, package_name: str) -> RegistryPackageRecord:
        name = validate_package_name(package_name)
        url = self._url(f"v1/package/{quote(name, safe='')}.json")
        try:
            payload = self._get_json(url)
        except RegistryAPIError as exc:
            if exc.__cause__ is not None and isinstance(exc.__cause__, HTTPError):
                if exc.__cause__.code == 404:
                    raise ModuleNotFoundError(
                        f"Kedi package {name!r} is not available in the registry"
                    ) from exc
            raise
        return _parse_package_record(payload, registry_url=url)

    def audit_records(self) -> tuple[RegistryAuditRecord, ...]:
        payload = self._get_json(self._url("v1/audit.json"))
        _require_exact_keys(payload, required={"schema_version", "registry_revision", "entries"})
        if payload["schema_version"] != 1:
            raise RegistryAPIError("Unsupported registry audit schema version")
        _require_nonempty_string(payload["registry_revision"], "registry_revision")
        entries = payload["entries"]
        if not isinstance(entries, list):
            raise RegistryAPIError("Registry audit entries must be a list")
        parsed = tuple(_parse_audit_record(entry) for entry in entries)
        identities = [entry.identity for entry in parsed]
        if len(identities) != len(set(identities)):
            raise RegistryAPIError("Registry audit index contains duplicate commit identities")
        return parsed

    def _url(self, relative_path: str) -> str:
        return urljoin(f"{self.base_url}/", relative_path)

    def _get_json(self, url: str) -> dict[str, Any]:
        request = Request(
            url,
            headers={"Accept": "application/json", "User-Agent": "kedi-package-registry/1"},
        )
        try:
            with urlopen(request, timeout=REGISTRY_REQUEST_TIMEOUT_SECONDS) as response:
                _require_same_origin(self.base_url, response.geturl())
                declared_size = response.headers.get("Content-Length")
                if declared_size is not None and int(declared_size) > MAX_REGISTRY_RESPONSE_BYTES:
                    raise RegistryAPIError("Registry response exceeds the maximum size")
                content = response.read(MAX_REGISTRY_RESPONSE_BYTES + 1)
        except RegistryAPIError:
            raise
        except (HTTPError, URLError, OSError, ValueError) as exc:
            raise RegistryAPIError(f"Could not fetch registry record from {url}: {exc}") from exc
        if len(content) > MAX_REGISTRY_RESPONSE_BYTES:
            raise RegistryAPIError("Registry response exceeds the maximum size")
        try:
            payload = json.loads(content)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RegistryAPIError("Registry returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise RegistryAPIError("Registry response must be a JSON object")
        return payload


def _parse_package_record(payload: dict[str, Any], *, registry_url: str) -> RegistryPackageRecord:
    required = {
        "schema_version",
        "name",
        "version",
        "description",
        "author",
        "repository",
        "verified_commit",
        "package_manifest_path",
        "status",
        "registry_manifest_digest",
        "registry_revision",
    }
    _require_exact_keys(payload, required=required)
    if payload["schema_version"] != 1:
        raise RegistryAPIError("Unsupported package registry schema version")
    name = validate_package_name(_require_nonempty_string(payload["name"], "name"))
    author = payload["author"]
    if not isinstance(author, dict):
        raise RegistryAPIError("Registry package author must be an object")
    _require_exact_keys(author, required={"name", "contact"})
    commit = _require_nonempty_string(payload["verified_commit"], "verified_commit")
    if _COMMIT_RE.fullmatch(commit) is None:
        raise RegistryAPIError("Registry verified_commit must be a full lowercase Git commit ID")
    digest = _require_nonempty_string(
        payload["registry_manifest_digest"], "registry_manifest_digest"
    )
    if _DIGEST_RE.fullmatch(digest) is None:
        raise RegistryAPIError("Registry manifest digest must be a sha256 digest")
    status = _require_nonempty_string(payload["status"], "status")
    if status not in _PACKAGE_STATUSES:
        raise RegistryAPIError(f"Unknown registry package status: {status!r}")
    manifest_path = _validate_manifest_path(
        _require_nonempty_string(payload["package_manifest_path"], "package_manifest_path")
    )
    return RegistryPackageRecord(
        schema_version=1,
        name=name,
        version=_require_nonempty_string(payload["version"], "version"),
        description=_require_nonempty_string(payload["description"], "description"),
        author_name=_require_nonempty_string(author["name"], "author.name"),
        author_contact=_require_nonempty_string(author["contact"], "author.contact"),
        repository=_require_nonempty_string(payload["repository"], "repository"),
        verified_commit=commit,
        package_manifest_path=manifest_path,
        status=status,
        registry_manifest_digest=digest,
        registry_revision=_require_nonempty_string(
            payload["registry_revision"], "registry_revision"
        ),
        registry_url=registry_url,
    )


def _parse_audit_record(value: Any) -> RegistryAuditRecord:
    if not isinstance(value, dict):
        raise RegistryAPIError("Registry audit entry must be an object")
    required = {"name", "version", "repository", "verified_commit", "status"}
    optional = {"reason", "advisory_url"}
    _require_exact_keys(value, required=required, optional=optional)
    commit = _require_nonempty_string(value["verified_commit"], "verified_commit")
    if _COMMIT_RE.fullmatch(commit) is None:
        raise RegistryAPIError("Registry audit commit must be a full lowercase Git commit ID")
    status = _require_nonempty_string(value["status"], "status")
    if status not in _AUDIT_STATUSES:
        raise RegistryAPIError(f"Unknown registry audit status: {status!r}")
    return RegistryAuditRecord(
        name=validate_package_name(_require_nonempty_string(value["name"], "name")),
        version=_require_nonempty_string(value["version"], "version"),
        repository=_require_nonempty_string(value["repository"], "repository"),
        verified_commit=commit,
        status=status,
        reason=_optional_string(value.get("reason"), "reason"),
        advisory_url=_optional_string(value.get("advisory_url"), "advisory_url"),
    )


def _require_exact_keys(
    value: dict[str, Any], *, required: set[str], optional: set[str] | None = None
) -> None:
    allowed = required | (optional or set())
    missing = required - value.keys()
    unknown = value.keys() - allowed
    if missing:
        raise RegistryAPIError(f"Registry record is missing fields: {', '.join(sorted(missing))}")
    if unknown:
        raise RegistryAPIError(f"Registry record has unknown fields: {', '.join(sorted(unknown))}")


def _require_nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RegistryAPIError(f"Registry field {field!r} must be a non-empty string")
    return value


def _optional_string(value: Any, field: str) -> str | None:
    if value is None:
        return None
    return _require_nonempty_string(value, field)


def _validate_manifest_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or value.startswith(("/", "\\"))
        or "\\" in value
        or any(character in value for character in "*?[")
        or path.name != "package.kedi"
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise RegistryAPIError(
            "Registry package_manifest_path must be a normalized package.kedi path"
        )
    return path.as_posix()


def _validate_registry_base_url(value: str) -> str:
    parsed = urlsplit(value.rstrip("/"))
    if (
        parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Kedi registry URL must not contain credentials, query, or fragment")
    if parsed.scheme == "https" and parsed.hostname:
        return value.rstrip("/")
    if parsed.scheme == "http" and _is_loopback_hostname(parsed.hostname):
        return value.rstrip("/")
    raise ValueError("Kedi registry URL must use HTTPS, except for a loopback development server")


def _is_loopback_hostname(hostname: str | None) -> bool:
    if hostname is None:
        return False
    if hostname.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _require_same_origin(base_url: str, response_url: str) -> None:
    base = urlsplit(base_url)
    response = urlsplit(response_url)
    if (base.scheme, base.hostname, base.port) != (
        response.scheme,
        response.hostname,
        response.port,
    ):
        raise RegistryAPIError("Registry response redirected to a different origin")
