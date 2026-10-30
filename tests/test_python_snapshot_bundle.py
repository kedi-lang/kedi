"""Security and ownership regressions for executable session snapshot storage."""

from __future__ import annotations

import hashlib
import io
import json
import os
import pickle
import stat
import struct
import warnings
import zipfile
from pathlib import Path

import pytest

import kedi
from kedi._interactive import bundle
from kedi._interactive.persistence import SessionDumpError, SessionLoadError


class _MarkerReducer:
    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self):
        return (Path.write_text, (self.marker, "executed"))


def _read_archive(path: Path) -> tuple[bytes, dict[str, bytes]]:
    content = path.read_bytes()
    prefix = content[: content.index(b"PK\x03\x04")]
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        return prefix, {name: archive.read(name) for name in archive.namelist()}


def _rewrite(path: Path, mutate=None, *, extra=None, compression=zipfile.ZIP_STORED) -> None:
    prefix, members = _read_archive(path)
    manifest = json.loads(members["manifest.json"])
    if mutate:
        mutate(manifest, members)
    members["manifest.json"] = json.dumps(manifest).encode()
    with path.open("wb") as stream:
        stream.write(prefix)
        with zipfile.ZipFile(stream, "w", compression=compression) as archive:
            for name, data in members.items():
                archive.writestr(name, data)
            if extra:
                info, data = extra
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    archive.writestr(info, data)


def _write_bundle(tmp_path: Path, *, files=None, limit=0) -> Path:
    path = tmp_path / "snapshot.kedi-state"
    bundle.write_python_bundle(path, b"graph", {"resource_limit_bytes": limit}, files or {})
    return path


@pytest.fixture
def executable_snapshot(tmp_path: Path) -> tuple[Path, Path]:
    path = tmp_path / "session.kedi-state"
    marker = tmp_path / "external-marker"
    with kedi.InteractiveSession() as session:
        session.execute("[value: int] = `7`")
        session.dump(path, format="python")
    import struct

    from kedi._interactive.python_graph import _MAGIC

    _, original_members = _read_archive(path)
    graph = original_members["graph.pkl"]
    header_start = len(_MAGIC) + 8
    header_size = struct.unpack(">Q", graph[len(_MAGIC) : header_start])[0]
    payload = graph[: header_start + header_size] + pickle.dumps(_MarkerReducer(marker))

    def inject(manifest, members):
        members["graph.pkl"] = payload
        manifest["members"]["graph.pkl"] = {
            "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

    _rewrite(path, inject)
    return path, marker


@pytest.mark.parametrize(
    "rejection",
    ["trust", "python", "implementation", "kedi", "cloudpickle", "package", "corruption"],
)
def test_rejection_precedes_executable_graph(executable_snapshot, rejection: str) -> None:
    path, marker = executable_snapshot

    def invalidate(manifest, members):
        environment = manifest["environment"]
        if rejection == "python":
            environment["python_version"][2] += 1
        elif rejection == "implementation":
            environment["python_implementation"] = "incompatible-implementation"
        elif rejection == "kedi":
            environment["kedi_version"] = "incompatible-version"
        elif rejection == "cloudpickle":
            environment["cloudpickle_version"] = "incompatible-version"
        elif rejection == "package":
            environment["packages"]["kedi-snapshot-nonexistent-package"] = "999"
        elif rejection == "corruption":
            members["graph.pkl"] = b"!" + members["graph.pkl"][1:]

    _rewrite(path, invalidate)
    with pytest.raises(SessionLoadError):
        kedi.InteractiveSession.load(path, trusted=rejection != "trust")
    assert not marker.exists()


def test_resource_extraction_is_private_and_owned(tmp_path: Path) -> None:
    files = {"resources/owner/payload.bin": b"secret payload", "resources/owner/index.json": b"{}"}
    path = _write_bundle(tmp_path, files=files, limit=sum(map(len, files.values())))
    loaded = bundle.read_python_bundle(path, trusted=True)
    root = loaded.root
    try:
        assert loaded.graph == b"graph"
        for name, data in files.items():
            assert (root / name).read_bytes() == data
        if os.name != "nt":
            assert stat.S_IMODE(root.stat().st_mode) == 0o700
            assert stat.S_IMODE((root / "resources/owner").stat().st_mode) == 0o700
            assert stat.S_IMODE((root / "resources/owner/payload.bin").stat().st_mode) == 0o600
    finally:
        loaded.cleanup()
    assert not root.exists()
    loaded.cleanup()


@pytest.mark.parametrize(
    "name",
    [
        "../escaped",
        "/absolute",
        "resources/../escaped",
        "resources//payload",
        "resources\\payload",
        "unknown.bin",
    ],
)
def test_unsafe_or_unknown_member_is_rejected(tmp_path: Path, name: str) -> None:
    path = _write_bundle(tmp_path)
    _rewrite(path, extra=(name, b"payload"))
    with pytest.raises(SessionLoadError, match="unsafe|unsupported"):
        bundle.read_python_bundle(path, trusted=True)
    assert not (tmp_path / "escaped").exists()


def test_duplicate_member_is_rejected(tmp_path: Path) -> None:
    path = _write_bundle(tmp_path)
    _rewrite(path, extra=("graph.pkl", b"replacement"))
    with pytest.raises(SessionLoadError, match="duplicate"):
        bundle.read_python_bundle(path, trusted=True)


def test_symlink_member_is_rejected(tmp_path: Path) -> None:
    path = _write_bundle(tmp_path)
    link = zipfile.ZipInfo("resources/link")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    _rewrite(path, extra=(link, b"/outside"))
    with pytest.raises(SessionLoadError, match="regular files"):
        bundle.read_python_bundle(path, trusted=True)


def test_compressed_bomb_is_rejected(tmp_path: Path) -> None:
    path = _write_bundle(
        tmp_path, files={"resources/bomb": b"0" * (1024 * 1024)}, limit=1024 * 1024
    )
    _rewrite(path, compression=zipfile.ZIP_DEFLATED)
    with pytest.raises(SessionLoadError, match="compressed"):
        bundle.read_python_bundle(path, trusted=True)


def test_plain_zip_is_routed_to_safe_python_reader(tmp_path: Path) -> None:
    path = tmp_path / "not-a-session"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("graph.pkl", pickle.dumps(_MarkerReducer(tmp_path / "marker")))
    assert bundle.is_python_snapshot(path)
    with pytest.raises(SessionLoadError):
        kedi.InteractiveSession.load(path, trusted=True)
    assert not (tmp_path / "marker").exists()


@pytest.mark.parametrize("kind", ["logical", "resources", "count", "hard-resources"])
def test_declared_bounds_are_enforced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    path = _write_bundle(tmp_path, files={"resources/payload": b"0123456789"}, limit=10)
    if kind == "logical":
        monkeypatch.setattr(bundle, "MAX_SNAPSHOT_BYTES", 1)
    elif kind == "count":
        monkeypatch.setattr(bundle, "_MAX_MEMBERS", 2)
    elif kind == "hard-resources":
        monkeypatch.setattr(bundle, "_MAX_ARTIFACT_BYTES", 9)
    else:
        _rewrite(path, lambda manifest, _: manifest["metadata"].update(resource_limit_bytes=9))
    with pytest.raises(SessionLoadError):
        bundle.read_python_bundle(path, trusted=True)


def test_digest_failure_removes_extracted_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_bundle(tmp_path, files={"resources/payload": b"secret"}, limit=6)
    _rewrite(
        path, lambda manifest, _: manifest["members"]["resources/payload"].update(sha256="0" * 64)
    )
    allocated = []
    real_mkdtemp = bundle.tempfile.mkdtemp

    def capture_root(*args, **kwargs):
        root = real_mkdtemp(*args, **kwargs)
        allocated.append(Path(root))
        return root

    monkeypatch.setattr(bundle.tempfile, "mkdtemp", capture_root)
    with pytest.raises(SessionLoadError, match="digest"):
        bundle.read_python_bundle(path, trusted=True)
    assert allocated
    assert all(not root.exists() for root in allocated)


@pytest.mark.parametrize("failure", ["replace", "write", "fsync"])
def test_atomic_failure_preserves_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    path = tmp_path / "existing.kedi-state"
    path.write_bytes(b"previous snapshot")

    def fail(*args, **kwargs):
        raise OSError("injected storage failure")

    if failure == "replace":
        monkeypatch.setattr(bundle.os, "replace", fail)
    elif failure == "fsync":
        monkeypatch.setattr(bundle.os, "fsync", fail)
    else:
        monkeypatch.setattr(bundle.zipfile.ZipFile, "writestr", fail)
    with pytest.raises(SessionDumpError, match="injected storage failure"):
        bundle.write_python_bundle(path, b"new graph", {"resource_limit_bytes": 0}, {})
    assert path.read_bytes() == b"previous snapshot"
    assert not list(tmp_path.glob(".existing.kedi-state.*.tmp"))


def test_artifact_writer_limit_preserves_destination(tmp_path: Path) -> None:
    path = tmp_path / "existing.kedi-state"
    path.write_bytes(b"previous snapshot")
    with pytest.raises(SessionDumpError, match="allowance"):
        bundle.write_python_bundle(
            path, b"graph", {"resource_limit_bytes": 1}, {"resources/payload": b"oversize"}
        )
    assert path.read_bytes() == b"previous snapshot"


@pytest.mark.skipif(os.name == "nt", reason="POSIX directory fsync")
@pytest.mark.parametrize("existing", [True, False])
def test_directory_fsync_failure_rolls_back_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing: bool
) -> None:
    path = tmp_path / "session.kedi-state"
    if existing:
        path.write_bytes(b"previous snapshot")
    real_fsync = bundle.os.fsync
    calls = 0

    def fail_second_fsync(fd):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("directory sync failed")
        return real_fsync(fd)

    monkeypatch.setattr(bundle.os, "fsync", fail_second_fsync)
    with pytest.raises(SessionDumpError, match="directory sync failed"):
        bundle.write_python_bundle(path, b"new graph", {"resource_limit_bytes": 0}, {})
    if existing:
        assert path.read_bytes() == b"previous snapshot"
    else:
        assert not path.exists()
    assert not list(tmp_path.glob(".session.kedi-state.*"))


@pytest.mark.skipif(os.name == "nt", reason="POSIX directory fsync")
def test_failed_rollback_retains_recovery_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "session.kedi-state"
    path.write_bytes(b"previous snapshot")
    real_fsync = bundle.os.fsync
    real_replace = bundle.os.replace
    sync_calls = 0
    replace_calls = 0

    def fail_directory_sync(fd):
        nonlocal sync_calls
        sync_calls += 1
        if sync_calls == 2:
            raise OSError("directory sync failed")
        return real_fsync(fd)

    def fail_rollback(source, destination):
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 2:
            raise OSError("rollback replace failed")
        return real_replace(source, destination)

    monkeypatch.setattr(bundle.os, "fsync", fail_directory_sync)
    monkeypatch.setattr(bundle.os, "replace", fail_rollback)
    with pytest.raises(
        SessionDumpError,
        match="rollback failed.*rollback replace failed.*previous snapshot retained",
    ):
        bundle.write_python_bundle(path, b"new graph", {"resource_limit_bytes": 0}, {})
    recovery = list(tmp_path.glob(".session.kedi-state.*.rollback"))
    assert len(recovery) == 1
    assert recovery[0].read_bytes() == b"previous snapshot"
    assert not list(tmp_path.glob(".session.kedi-state.*.tmp"))


def test_zip64_override_is_rejected_before_zip_parser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_bundle(tmp_path)
    content = path.read_bytes()
    end_offset = content.rfind(b"PK\x05\x06")
    # A small classic directory passes the initial bounds but Python's ZIP
    # parser would replace its values with this oversized ZIP64 directory.
    zip64_end = struct.pack("<4sQ2H2L4Q", b"PK\x06\x06", 44, 45, 45, 0, 0, 2, 2, 1 << 30, 0)
    locator = struct.pack("<4sLQL", b"PK\x06\x07", 0, end_offset, 1)
    path.write_bytes(content[:end_offset] + zip64_end + locator + content[end_offset:])

    def forbidden_parser(*args, **kwargs):
        pytest.fail("ZIP parser must not inspect a ZIP64 directory")

    monkeypatch.setattr(bundle.zipfile, "ZipFile", forbidden_parser)
    with pytest.raises(SessionLoadError, match="ZIP64"):
        bundle.read_python_bundle(path, trusted=True)


@pytest.mark.parametrize(
    "corruption",
    ["under-count", "over-count", "invalid-signature", "oversized-entry", "invalid-offset"],
)
def test_directory_contents_are_bounded_before_zip_parser(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corruption: str
) -> None:
    path = _write_bundle(tmp_path, files={"resources/payload": b"data"}, limit=4)
    content = bytearray(path.read_bytes())
    end_offset = content.rfind(b"PK\x05\x06")
    directory_offset = struct.unpack_from("<L", content, end_offset + 16)[0]
    if corruption in ("under-count", "over-count"):
        count = 2 if corruption == "under-count" else 4
        struct.pack_into("<2H", content, end_offset + 8, count, count)
    elif corruption == "invalid-signature":
        content[directory_offset : directory_offset + 4] = b"BAD!"
    elif corruption == "oversized-entry":
        struct.pack_into("<H", content, directory_offset + 28, 65535)
    else:
        struct.pack_into("<L", content, end_offset + 16, directory_offset - 1)
    path.write_bytes(content)

    def forbidden_parser(*args, **kwargs):
        pytest.fail("ZIP parser must not allocate entries from an invalid directory")

    monkeypatch.setattr(bundle.zipfile, "ZipFile", forbidden_parser)
    with pytest.raises(SessionLoadError, match="ZIP directory"):
        bundle.read_python_bundle(path, trusted=True)


def test_exact_member_limit_accepts_valid_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_bundle(tmp_path, files={"resources/payload": b"data"}, limit=4)
    monkeypatch.setattr(bundle, "_MAX_MEMBERS", 3)
    loaded = bundle.read_python_bundle(path, trusted=True)
    try:
        assert loaded.graph == b"graph"
        assert (loaded.root / "resources/payload").read_bytes() == b"data"
    finally:
        loaded.cleanup()


def test_nul_member_name_is_rejected_before_extraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_name = "resources/payload!ignored"
    path = _write_bundle(tmp_path, files={original_name: b"secret"}, limit=6)

    def allow_truncated_name(manifest, members):
        manifest["members"]["resources/payload"] = manifest["members"].pop(original_name)

    _rewrite(path, allow_truncated_name)
    # ZipInfo truncates filename at NUL but keeps orig_filename for comparison
    # with the local header, so matching NUL names in both headers can load.
    content = path.read_bytes()
    assert content.count(original_name.encode()) == 2
    path.write_bytes(content.replace(original_name.encode(), b"resources/payload\x00ignored"))

    def forbidden_extraction(*args, **kwargs):
        pytest.fail("An unsafe archive name must fail before extraction")

    monkeypatch.setattr(bundle.tempfile, "mkdtemp", forbidden_extraction)
    with pytest.raises(SessionLoadError, match="unsafe"):
        bundle.read_python_bundle(path, trusted=True)


@pytest.mark.parametrize("trusted", [None, 1, "true"])
def test_truthy_trust_is_not_executable_permission(executable_snapshot, trusted) -> None:
    path, marker = executable_snapshot
    with pytest.raises(SessionLoadError, match="trusted=True"):
        kedi.InteractiveSession.load(path, trusted=trusted)
    assert not marker.exists()


def test_all_resource_digests_are_checked_before_executable_graph(executable_snapshot) -> None:
    path, marker = executable_snapshot

    def append_corrupt_resource(manifest, members):
        members["resources/payload"] = b"secret"
        manifest["metadata"]["resource_limit_bytes"] = 6
        manifest["members"]["resources/payload"] = {"size": 6, "sha256": "0" * 64}

    _rewrite(path, append_corrupt_resource)
    with pytest.raises(SessionLoadError, match="digest"):
        kedi.InteractiveSession.load(path, trusted=True)
    assert not marker.exists()


def _pinned_packages(path: Path) -> dict[str, str]:
    _, members = _read_archive(path)
    return json.loads(members["manifest.json"])["environment"]["packages"]


def _idna_snapshot(tmp_path: Path) -> Path:
    pytest.importorskip("idna")
    path = tmp_path / "idna.kedi-state"
    with kedi.InteractiveSession() as session:
        session.execute("[encode: Any] = `__import__('idna').encode`")
        session.dump(path, format="python")
    return path


def test_manifest_pins_only_graph_distributions_and_their_dependencies(tmp_path: Path) -> None:
    pinned = _pinned_packages(_idna_snapshot(tmp_path))
    assert "idna" in pinned
    assert {"pydantic", "pydantic-core"} <= set(pinned)
    # Development tooling installed beside Kedi is not part of the graph.
    assert "pytest" not in pinned
    assert "kedi" not in pinned


@pytest.mark.parametrize(("drifted", "loads"), [("pytest", True), ("idna", False)])
def test_only_referenced_distribution_drift_rejects_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, drifted: str, loads: bool
) -> None:
    path = _idna_snapshot(tmp_path)
    installed = bundle._installed_version
    monkeypatch.setattr(
        bundle,
        "_installed_version",
        lambda name: "0.0.0-drift" if name == drifted else installed(name),
    )
    if loads:
        kedi.InteractiveSession.load(path, trusted=True).close()
    else:
        with pytest.raises(SessionLoadError, match=f"package version mismatch: {drifted}"):
            kedi.InteractiveSession.load(path, trusted=True)
