from __future__ import annotations

import json
import subprocess
import threading
from contextlib import contextmanager
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterator

import pytest
from click.testing import CliRunner

import kedi.cli as cli_module
from kedi.package_audit import audit_installed_packages
from kedi.package_registry import INSTALL_RECEIPT_NAME, add_package, install_package
from kedi.registry_api import HTTPRegistryAPIClient


class _QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, _format: str, *args: object) -> None:
        pass


@contextmanager
def _serve_registry(root: Path) -> Iterator[str]:
    handler = partial(_QuietHandler, directory=str(root))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _git(repository: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _package_repository(tmp_path: Path) -> tuple[Path, str]:
    repository = tmp_path / "package-repository"
    repository.mkdir()
    subprocess.run(
        ["git", "init", "--quiet", "--initial-branch=main", str(repository)],
        check=True,
    )
    _git(repository, "config", "user.email", "registry-tests@kedi-lang.org")
    _git(repository, "config", "user.name", "Kedi Registry Tests")
    source = repository / "src" / "kedi_http"
    source.mkdir(parents=True)
    (repository / "package.kedi").write_text(
        "> package: kedi_http:\n  version: 1.0\n  source: src/kedi_http\n",
        encoding="utf-8",
    )
    (source / "main.kedi").write_text("= verified\n", encoding="utf-8")
    _git(repository, "add", ".")
    _git(repository, "commit", "--quiet", "-m", "verified package")
    verified_commit = _git(repository, "rev-parse", "HEAD")

    (source / "main.kedi").write_text("= repository head\n", encoding="utf-8")
    _git(repository, "add", ".")
    _git(repository, "commit", "--quiet", "-m", "unreviewed head")
    return repository, verified_commit


def _write_registry_api(
    root: Path,
    *,
    repository: Path,
    commit: str,
    package_status: str = "active",
    audit_status: str = "active",
    reason: str | None = None,
) -> None:
    package_api = root / "v1" / "package"
    package_api.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": 1,
        "name": "kedi_http",
        "version": "1.0",
        "description": "HTTP registry integration fixture.",
        "author": {"name": "Kedi", "contact": "registry@kedi-lang.org"},
        "repository": repository.resolve().as_uri(),
        "verified_commit": commit,
        "package_manifest_path": "package.kedi",
        "status": package_status,
        "registry_manifest_digest": f"sha256:{'0' * 64}",
        "registry_revision": "local-test-1",
    }
    (package_api / "kedi_http.json").write_text(
        json.dumps(record, sort_keys=True), encoding="utf-8"
    )
    audit_entry: dict[str, object] = {
        "name": "kedi_http",
        "version": "1.0",
        "repository": repository.resolve().as_uri(),
        "verified_commit": commit,
        "status": audit_status,
    }
    if reason is not None:
        audit_entry["reason"] = reason
    (root / "v1" / "audit.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "registry_revision": "local-test-1",
                "entries": [audit_entry],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def test_http_registry_installs_exact_commit_instead_of_repository_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, verified_commit = _package_repository(tmp_path)
    api_root = tmp_path / "registry-api"
    _write_registry_api(api_root, repository=repository, commit=verified_commit)
    install_root = tmp_path / "installed"

    with _serve_registry(api_root) as registry_url:
        monkeypatch.setenv("KEDI_REGISTRY_URL", registry_url)
        installed = add_package("kedi_http", registry_root=install_root)

    assert installed.source.joinpath("main.kedi").read_text(encoding="utf-8") == "= verified\n"
    receipt = json.loads((installed.root / INSTALL_RECEIPT_NAME).read_text(encoding="utf-8"))
    assert receipt["verified"] is True
    assert receipt["verified_commit"] == verified_commit
    assert receipt["source_revision"] == verified_commit


@pytest.mark.parametrize("status", ["yanked", "revoked"])
def test_audit_reports_exact_withdrawn_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    repository, verified_commit = _package_repository(tmp_path)
    api_root = tmp_path / "registry-api"
    _write_registry_api(api_root, repository=repository, commit=verified_commit)
    install_root = tmp_path / "kedi-home" / "registry"

    with _serve_registry(api_root) as registry_url:
        monkeypatch.setenv("KEDI_REGISTRY_URL", registry_url)
        add_package("kedi_http", registry_root=install_root)
        _write_registry_api(
            api_root,
            repository=repository,
            commit=verified_commit,
            audit_status=status,
            reason="withdrawn by registry maintainers",
        )
        report = audit_installed_packages(registry_root=install_root)

    assert report.exit_code == 1
    assert report.results[0].status == status
    assert report.results[0].commit == verified_commit
    assert report.results[0].detail == "withdrawn by registry maintainers"


def test_cli_audit_prints_findings_and_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, verified_commit = _package_repository(tmp_path)
    api_root = tmp_path / "registry-api"
    _write_registry_api(api_root, repository=repository, commit=verified_commit)
    kedi_home = tmp_path / "kedi-home"

    with _serve_registry(api_root) as registry_url:
        monkeypatch.setenv("KEDI_HOME", str(kedi_home))
        monkeypatch.setenv("KEDI_REGISTRY_URL", registry_url)
        add_package("kedi_http")
        _write_registry_api(
            api_root,
            repository=repository,
            commit=verified_commit,
            audit_status="revoked",
            reason="security withdrawal",
        )
        result = CliRunner().invoke(cli_module.main, ["audit"])

    assert result.exit_code == 1
    assert f"REVOKED kedi_http 1.0 {verified_commit[:12]}" in result.output
    assert "security withdrawal" in result.output
    assert "revoked=1" in result.output


def test_superseded_commit_is_auditable_but_not_actionable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, verified_commit = _package_repository(tmp_path)
    api_root = tmp_path / "registry-api"
    _write_registry_api(api_root, repository=repository, commit=verified_commit)
    install_root = tmp_path / "installed"

    with _serve_registry(api_root) as registry_url:
        monkeypatch.setenv("KEDI_REGISTRY_URL", registry_url)
        add_package("kedi_http", registry_root=install_root)
        _write_registry_api(
            api_root,
            repository=repository,
            commit=verified_commit,
            audit_status="superseded",
        )
        report = audit_installed_packages(registry_root=install_root)

    assert report.exit_code == 0
    assert report.results[0].status == "superseded"


def test_local_package_is_reported_unverified_without_registry_request(tmp_path: Path) -> None:
    package = tmp_path / "local-package"
    source = package / "src" / "local_package"
    source.mkdir(parents=True)
    (package / "package.kedi").write_text(
        "> package: local_package:\n  version: 1.0\n  source: src/local_package\n",
        encoding="utf-8",
    )
    (source / "main.kedi").write_text("= local\n", encoding="utf-8")
    install_root = tmp_path / "installed"

    install_package(package / "package.kedi", registry_root=install_root)
    report = audit_installed_packages(registry_root=install_root)

    assert report.exit_code == 0
    assert report.results[0].status == "unverified"
    assert report.results[0].source_kind == "local"


def test_cli_audit_uses_exit_two_when_registry_is_unreachable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, verified_commit = _package_repository(tmp_path)
    api_root = tmp_path / "registry-api"
    _write_registry_api(api_root, repository=repository, commit=verified_commit)
    kedi_home = tmp_path / "kedi-home"

    with _serve_registry(api_root) as registry_url:
        monkeypatch.setenv("KEDI_HOME", str(kedi_home))
        monkeypatch.setenv("KEDI_REGISTRY_URL", registry_url)
        add_package("kedi_http")

    result = CliRunner().invoke(cli_module.main, ["audit"])

    assert result.exit_code == 2
    assert "Registry audit could not complete" in result.output


def test_registry_rejects_yanked_package_before_git_fetch(tmp_path: Path) -> None:
    repository, verified_commit = _package_repository(tmp_path)
    api_root = tmp_path / "registry-api"
    _write_registry_api(
        api_root,
        repository=repository,
        commit=verified_commit,
        package_status="yanked",
        audit_status="yanked",
    )

    with _serve_registry(api_root) as registry_url:
        with pytest.raises(RuntimeError, match="is yanked"):
            from kedi.package_registry import install_registry_package

            install_registry_package(
                "kedi_http",
                registry_root=tmp_path / "installed",
                api_client=HTTPRegistryAPIClient(registry_url),
            )
