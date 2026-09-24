from __future__ import annotations

import hashlib
import io
import json
from dataclasses import replace
from pathlib import Path, PureWindowsPath
from unittest.mock import Mock
from urllib.error import HTTPError

import pytest
from click.testing import CliRunner

import kedi.cli as cli
import kedi.package_audit as audit
import kedi.package_registry as packages
import kedi.registry_api as api


def package_payload():
    return {
        "schema_version": 1,
        "name": "demo",
        "version": "1.0",
        "description": "A test package",
        "author": {"name": "Author", "contact": "author@example.test"},
        "repository": "https://example.test/demo.git",
        "verified_commit": "a" * 40,
        "package_manifest_path": "package.kedi",
        "status": "active",
        "registry_manifest_digest": "sha256:" + "b" * 64,
        "registry_revision": "revision-1",
    }


def package_record():
    return api._parse_package_record(package_payload(), registry_url="https://example.test")


@pytest.mark.parametrize(
    "update",
    [
        {"schema_version": 2},
        {"name": ""},
        {"author": []},
        {"author": {"name": "A"}},
        {"verified_commit": "main"},
        {"registry_manifest_digest": "bad"},
        {"status": "unknown"},
        {"unexpected": True},
        {"package_manifest_path": "../package.kedi"},
    ],
)
def test_invalid_package_records_are_rejected(update):
    with pytest.raises((api.RegistryAPIError, ValueError)):
        api._parse_package_record(package_payload() | update, registry_url="https://example.test")


@pytest.mark.parametrize(
    "entry",
    [[], {"verified_commit": "main"}, {"status": "unknown"}, {"reason": ""}],
)
def test_invalid_audit_entries_are_rejected(entry):
    valid = {
        key: package_payload()[key]
        for key in ("name", "version", "repository", "verified_commit", "status")
    }
    with pytest.raises(api.RegistryAPIError):
        api._parse_audit_record(valid | entry if isinstance(entry, dict) else entry)


@pytest.mark.parametrize("mutation", ["schema", "entries", "duplicate"])
def test_invalid_audit_indexes_are_rejected(monkeypatch, mutation):
    entry = {
        key: package_payload()[key]
        for key in ("name", "version", "repository", "verified_commit", "status")
    }
    payload = {"schema_version": 1, "registry_revision": "r", "entries": [entry]}
    if mutation == "schema":
        payload["schema_version"] = 2
    elif mutation == "entries":
        payload["entries"] = {}
    else:
        payload["entries"] = [entry, entry]
    client = api.HTTPRegistryAPIClient()
    monkeypatch.setattr(client, "_get_json", lambda _: payload)
    with pytest.raises(api.RegistryAPIError):
        client.audit_records()


@pytest.mark.parametrize("code", [404, 503])
def test_registry_http_errors_keep_their_meaning(monkeypatch, code):
    def fail(*args, **kwargs):
        raise HTTPError("https://example.test", code, "failure", {}, None)

    monkeypatch.setattr(api, "urlopen", fail)
    with pytest.raises(ModuleNotFoundError if code == 404 else api.RegistryAPIError):
        api.HTTPRegistryAPIClient().package_record("demo")


@pytest.mark.parametrize(
    ("body", "headers", "url", "message"),
    [
        pytest.param(
            b"{}",
            {"Content-Length": "999999999"},
            "http://127.0.0.1:8767/v1/audit.json",
            "exceeds the maximum size",
            id="oversized-content-length",
        ),
        pytest.param(
            b"x" * (api.MAX_REGISTRY_RESPONSE_BYTES + 1),
            {},
            "http://127.0.0.1:8767/v1/audit.json",
            "exceeds the maximum size",
            id="oversized-body",
        ),
        pytest.param(
            b"not json",
            {},
            "http://127.0.0.1:8767/v1/audit.json",
            "invalid JSON",
            id="invalid-json",
        ),
        pytest.param(
            b"[]",
            {},
            "http://127.0.0.1:8767/v1/audit.json",
            "must be a JSON object",
            id="non-object-json",
        ),
        pytest.param(
            b"{}",
            {},
            "https://other.example.test/v1/audit.json",
            "different origin",
            id="cross-origin-response",
        ),
        pytest.param(
            b"{}",
            {"Content-Length": "invalid"},
            "http://127.0.0.1:8767/v1/audit.json",
            "Could not fetch registry record",
            id="invalid-content-length",
        ),
    ],
)
def test_registry_rejects_invalid_http_responses(monkeypatch, body, headers, url, message):
    response = io.BytesIO(body)
    response.headers = headers
    response.geturl = lambda: url
    monkeypatch.setattr(api, "urlopen", lambda *args, **kwargs: response)
    with pytest.raises(api.RegistryAPIError, match=message):
        api.HTTPRegistryAPIClient("http://127.0.0.1:8767").audit_records()


@pytest.mark.parametrize("url", ["https://host.test", "http://localhost:80", "http://[::1]:80"])
def test_valid_registry_origins(url):
    assert api.HTTPRegistryAPIClient(url).base_url == url


@pytest.mark.parametrize(
    "url",
    ["https://user:pass@host.test", "https://host.test/?x", "http://host.test", "file:///tmp"],
)
def test_registry_rejects_unsafe_origins(url):
    with pytest.raises(ValueError):
        api.HTTPRegistryAPIClient(url)
    assert not api._is_loopback_hostname(None)


@pytest.fixture
def installed_receipt(tmp_path):
    root = tmp_path / "installed"
    package = root / "demo"
    package.mkdir(parents=True)
    manifest = package / "package.kedi"
    manifest.write_text("> package: demo:\n  version: 1.0\n  source: src/demo\n")
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    receipt = {
        "package": "demo",
        "package_version": "1.0",
        "format_version": 2,
        "install_source_kind": "registry",
        "verified": True,
        "repository": "https://example.test/demo.git",
        "source_url": "https://example.test/demo.git",
        "verified_commit": "a" * 40,
        "source_revision": "a" * 40,
        "package_manifest_path": "package.kedi",
        "registry_url": "https://example.test",
        "registry_revision": "r",
        "registry_manifest_digest": "sha256:" + "b" * 64,
        "manifest_sha256": digest,
        "installed_manifest_digest": "sha256:" + digest,
    }
    (package / packages.INSTALL_RECEIPT_NAME).write_text(json.dumps(receipt))
    return root, package, receipt


@pytest.mark.parametrize(
    "update",
    [
        {"package": "other"},
        {"format_version": 1},
        {"verified_commit": "main"},
        {"source_revision": "b" * 40},
        {"registry_manifest_digest": "bad"},
        {"installed_manifest_digest": "bad"},
        {"manifest_sha256": "bad"},
        {"manifest_sha256": "c" * 64},
        {"registry_revision": ""},
    ],
)
def test_audit_reports_tampered_receipts(installed_receipt, update):
    root, package, receipt = installed_receipt
    (package / packages.INSTALL_RECEIPT_NAME).write_text(json.dumps(receipt | update))
    report = audit.audit_installed_packages(registry_root=root)
    assert report.exit_code == 1
    assert report.results[0].status == "malformed"


@pytest.mark.parametrize(
    "kind",
    ["missing", "oversized", "invalid_json", "not_object", "missing_manifest", "invalid_name"],
)
def test_audit_handles_broken_installations(installed_receipt, kind):
    root, package, _ = installed_receipt
    receipt = package / packages.INSTALL_RECEIPT_NAME
    if kind == "missing":
        receipt.unlink()
    elif kind == "oversized":
        receipt.write_text("x" * (audit.MAX_INSTALL_RECEIPT_BYTES + 1))
    elif kind == "invalid_json":
        receipt.write_text("{")
    elif kind == "not_object":
        receipt.write_text("[]")
    elif kind == "missing_manifest":
        (package / "package.kedi").unlink()
    else:
        package.rename(root / "bad name")
    assert audit.audit_installed_packages(registry_root=root).results[0].status == "malformed"


def test_audit_missing_root_and_non_directory(tmp_path):
    root = tmp_path / "absent"
    assert audit.audit_installed_packages(registry_root=root).results == ()
    root.write_text("not a directory")
    assert audit.audit_installed_packages(registry_root=root).exit_code == 1


def test_audit_ignores_internal_entries_and_flags_symlinks(tmp_path):
    (tmp_path / ".staging").mkdir()
    (tmp_path / "unrelated.txt").write_text("ignored")
    (tmp_path / "linked").symlink_to(tmp_path, target_is_directory=True)
    report = audit.audit_installed_packages(registry_root=tmp_path)
    assert len(report.results) == 1
    assert report.results[0].status == "malformed"


@pytest.mark.parametrize("version", [None, "2.0"])
def test_audit_unknown_commit_and_mismatched_version(installed_receipt, version):
    root, _, _ = installed_receipt
    records = (
        []
        if version is None
        else [
            api.RegistryAuditRecord(
                "demo", version, "https://example.test/demo.git", "a" * 40, "active"
            )
        ]
    )
    client = Mock(audit_records=lambda: records)
    report = audit.audit_installed_packages(registry_root=root, api_client=client)
    assert report.results[0].status == ("unknown" if version is None else "malformed")


def test_audit_cli_help_arguments_and_advisory(monkeypatch):
    runner = CliRunner()
    assert "Check installed" in runner.invoke(cli.main, ["audit", "--help"]).output
    assert runner.invoke(cli.main, ["audit", "extra"]).exit_code == 2
    report = audit.PackageAuditReport(
        (
            audit.PackageAuditResult("active_demo", "active"),
            audit.PackageAuditResult(
                "demo",
                "unverified",
                source_kind="local",
                advisory_url="https://example.test/advisory",
            ),
        )
    )
    monkeypatch.setattr(cli, "audit_installed_packages", lambda: report)
    result = runner.invoke(cli.main, ["audit"])
    assert result.exit_code == 0
    assert "(local)" in result.output
    assert "https://example.test/advisory" in result.output


@pytest.mark.parametrize(
    "url,local,valid",
    [
        ("https://example.test/demo.git", False, True),
        ("https://user:pass@example.test/demo.git", False, False),
        ("file:relative", True, False),
        ("file:///tmp/demo", False, False),
        ("file:///tmp/demo", True, True),
        ("file:///C:/packages/demo", True, True),
        ("file:///C:/packages/demo", False, False),
        ("file://localhost/C:/packages/demo%20repo", True, True),
        ("file://remote-host/share/demo", True, False),
        ("http://localhost/demo", True, True),
        ("http://example.test/demo", True, False),
    ],
)
@pytest.mark.parametrize("path_type", [Path, PureWindowsPath], ids=["host-path", "windows-path"])
def test_registry_repository_transport_policy(url, local, valid, path_type, monkeypatch):
    monkeypatch.setattr(packages, "Path", path_type)
    if valid:
        assert packages._registry_repository_url(url, allow_local=local) == url
    else:
        with pytest.raises(packages.PackageRegistryError):
            packages._registry_repository_url(url, allow_local=local)


@pytest.mark.parametrize("path_type", [Path, PureWindowsPath], ids=["host-path", "windows-path"])
@pytest.mark.parametrize("source", ["/outside", "C:/outside", "C:outside", "../outside"])
def test_package_source_rejects_rooted_paths_on_every_host(source, path_type, monkeypatch):
    monkeypatch.setattr(packages, "Path", path_type)
    with pytest.raises(ValueError, match="relative path"):
        packages._package_source_relative_path(source, error_type=ValueError)
    assert packages._package_source_relative_path("src/demo", error_type=ValueError) == path_type(
        "src/demo"
    )


def test_registry_cannot_substitute_another_package(tmp_path):
    client = Mock(package_record=lambda _: replace(package_record(), name="other"))
    with pytest.raises(packages.PackageRegistryError, match="requested package"):
        packages.install_registry_package("demo", registry_root=tmp_path, api_client=client)


@pytest.mark.parametrize("kind", ["source", "name", "version"])
def test_install_rejects_mismatched_attestation(tmp_path, kind):
    manifest = tmp_path / "package.kedi"
    manifest.write_text("> package: demo:\n  version: 1.0\n  source: src/demo\n")
    (tmp_path / "src/demo").mkdir(parents=True)
    (tmp_path / "src/demo/main.kedi").write_text("= ok\n")
    record = package_record()
    if kind == "name":
        record = replace(record, name="other")
    elif kind == "version":
        record = replace(record, version="2.0")
    with pytest.raises(ValueError):
        packages.install_package(
            manifest,
            registry_root=tmp_path / "installed",
            registry_record=record,
            source_kind="local" if kind == "source" else "registry",
        )


@pytest.mark.parametrize("failure", ["fetch", "size", "oversized", "metadata", "glob", "checkout"])
def test_verified_checkout_rejects_invalid_git_results(tmp_path, monkeypatch, failure):
    record = package_record()
    if failure == "glob":
        record = replace(record, package_manifest_path="*/package.kedi")

    def git(args, **kwargs):
        if "rev-parse" in args:
            bad = (failure == "fetch" and "FETCH_HEAD" in args) or (
                failure == "checkout" and "HEAD" in args
            )
            return "b" * 40 if bad else record.verified_commit
        if "cat-file" in args:
            return "invalid" if failure == "size" else str(2**30 if failure == "oversized" else 60)
        if "show" in args:
            name = "other" if failure == "metadata" else "demo"
            source = "src/demo"
            return f"> package: {name}:\n  version: 1.0\n  source: {source}\n"
        return ""

    monkeypatch.setattr(packages, "_run_git", git)
    with pytest.raises(packages.PackageRegistryError):
        packages._checkout_registry_commit(tmp_path / "checkout", record.repository, record)


def test_explicit_mock_client_requires_configuration(monkeypatch):
    monkeypatch.delenv(packages.MOCK_REGISTRY_ROOT_ENV, raising=False)
    with pytest.raises(RuntimeError, match="mock registry client"):
        packages.mock_registry_client_from_environment()
