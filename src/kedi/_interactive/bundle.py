"""Bounded, explicitly trusted storage for interactive Python snapshots."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import shutil
import stat
import struct
import sys
import tempfile
import zipfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from importlib import metadata as package_metadata
from pathlib import Path, PurePosixPath

from packaging.requirements import Requirement

from kedi._version import __version__

from .persistence import (
    MAX_SNAPSHOT_BYTES,
    SessionDumpError,
    SessionLoadError,
    SnapshotTrustRequired,
)

_MAGIC = b"KEDI-PYTHON-SNAPSHOT\x00\x01\n"
_FORMAT = "kedi.interactive-session.python"
_VERSION = 1
_MAX_MEMBERS = 10000
_MAX_ARTIFACT_BYTES = 1024 * 1024 * 1024
_MAX_ARCHIVE_BYTES = MAX_SNAPSHOT_BYTES + _MAX_ARTIFACT_BYTES + 16 * 1024 * 1024
_CHUNK = 1024 * 1024


@dataclass
class LoadedBundle:
    graph: bytes
    metadata: dict
    root: Path

    def cleanup(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)


def is_python_snapshot(path: str | Path) -> bool:
    """Recognize our magic and route all ZIP input through the safe reader."""
    try:
        with Path(path).expanduser().open("rb") as stream:
            prefix = stream.read(len(_MAGIC))
    except OSError as exc:
        raise SessionLoadError("Cannot read interactive session snapshot") from exc
    return (
        prefix.startswith(_MAGIC)
        or prefix.startswith(b"PK")
        or prefix.startswith(b"KEDI-PYTHON-SNAPSHOT")
    )


def _json_bytes(value: object) -> bytes:
    _primitives(value)
    return json.dumps(
        value, allow_nan=False, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _primitives(value: object, depth: int = 0) -> None:
    if depth > 100:
        raise ValueError("manifest nesting limit exceeded")
    if value is None or type(value) in (str, int, bool):
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is list:
        for item in value:
            _primitives(item, depth + 1)
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for item in value.values():
            _primitives(item, depth + 1)
        return
    raise ValueError("manifest must contain only finite JSON primitives")


def _canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _required_packages(modules: Iterable[str]) -> dict[str, str]:
    """Pin distributions owning by-reference graph modules plus their dependencies.

    Unrelated environment changes stay compatible. Kedi itself is pinned by
    ``kedi_version``; local or dynamic modules without a distribution are not.
    """
    owners = package_metadata.packages_distributions()
    pending = [
        distribution
        for module in modules
        for distribution in owners.get(module.partition(".")[0], ())
    ]
    # Kedi types are rebuilt by pydantic even when no pydantic object is referenced.
    pending.append("pydantic")
    result: dict[str, str] = {}
    while pending:
        name = _canonical(pending.pop())
        if name in result or name == "kedi":
            continue
        try:
            distribution = package_metadata.distribution(name)
        except package_metadata.PackageNotFoundError:
            continue
        result[name] = distribution.version
        for line in distribution.requires or ():
            requirement = Requirement(line)
            if requirement.marker is None or requirement.marker.evaluate({"extra": ""}):
                pending.append(requirement.name)
    return result


def _runtime_identity() -> dict:
    try:
        cloudpickle_version = package_metadata.version("cloudpickle")
    except package_metadata.PackageNotFoundError as exc:
        raise ValueError(
            "Python snapshots require the optional kedi[snapshots] dependency"
        ) from exc
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": list(sys.version_info),
        "kedi_version": __version__,
        "cloudpickle_version": cloudpickle_version,
    }


def _installed_version(name: str) -> str | None:
    try:
        return package_metadata.version(name)
    except package_metadata.PackageNotFoundError:
        return None


def _resource_limit(metadata: dict) -> int:
    limit = metadata.get("resource_limit_bytes", 0)
    if type(limit) is not int or not 0 <= limit <= _MAX_ARTIFACT_BYTES:
        raise ValueError(f"resource_limit_bytes must be between 0 and {_MAX_ARTIFACT_BYTES}")
    return limit


def _safe_name(name: str) -> bool:
    if not isinstance(name, str) or not name or "\\" in name or "\x00" in name:
        return False
    parts = name.split("/")
    return (
        not PurePosixPath(name).is_absolute()
        and all(part not in ("", ".", "..") for part in parts)
        and ":" not in name
        and len(name.encode("utf-8")) <= 1024
    )


def _resource_name(name: str) -> bool:
    return _safe_name(name) and name.startswith("resources/")


def _descriptor(data: bytes) -> dict:
    return {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def write_python_bundle(
    path: str | Path,
    graph: bytes,
    metadata: dict,
    files: Mapping[str, bytes],
    modules: Iterable[str] = (),
) -> Path:
    temporary: Path | None = None
    try:
        if type(graph) is not bytes or type(metadata) is not dict:
            raise ValueError("graph must be bytes and metadata must be a JSON object")
        _primitives(metadata)
        resource_limit = _resource_limit(metadata)
        if len(files) + 2 > _MAX_MEMBERS:
            raise ValueError("snapshot member count exceeds limit")
        artifact_size = 0
        members = {"graph.pkl": _descriptor(graph)}
        for name, data in files.items():
            if not _resource_name(name) or type(data) is not bytes:
                raise ValueError("artifact members must be bytes under resources/")
            artifact_size += len(data)
            if artifact_size > resource_limit:
                raise ValueError("artifact payloads exceed saved resource allowance")
            members[name] = _descriptor(data)
        manifest = {
            "format": _FORMAT,
            "format_version": _VERSION,
            "environment": {**_runtime_identity(), "packages": _required_packages(modules)},
            "metadata": metadata,
            "members": members,
        }
        rendered = _json_bytes(manifest)
        if len(rendered) + len(graph) > MAX_SNAPSHOT_BYTES:
            raise ValueError("snapshot metadata and graph exceed 64 MiB")
        destination = Path(path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, raw_path = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        temporary = Path(raw_path)
        with os.fdopen(fd, "w+b") as stream:
            stream.write(_MAGIC)
            with zipfile.ZipFile(
                stream, "w", compression=zipfile.ZIP_STORED, allowZip64=False
            ) as archive:
                for name, data in [("manifest.json", rendered), ("graph.pkl", graph)]:
                    _write_member(archive, name, data)
                for name, data in files.items():
                    _write_member(archive, name, data)
            stream.flush()
            os.fsync(stream.fileno())
        _publish(temporary, destination)
        temporary = None
        return destination
    except Exception as exc:
        raise SessionDumpError([f"Cannot write Python snapshot: {exc}"]) from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _sync_directory(path: Path) -> None:
    if os.name != "nt":  # pragma: no branch - directories cannot be fsynced on Windows
        directory_fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


def _publish(temporary: Path, destination: Path) -> None:
    """Keep the previous inode recoverable until publication is durable."""
    backup: Path | None = None
    published = False
    fd, raw_path = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".rollback", dir=destination.parent
    )
    os.close(fd)
    candidate = Path(raw_path)
    candidate.unlink()
    try:
        try:
            os.link(destination, candidate, follow_symlinks=False)
            backup = candidate
        except FileNotFoundError:
            pass
        os.replace(temporary, destination)
        published = True
        _sync_directory(destination.parent)
        if backup is not None:
            backup.unlink()
            backup = None
    except Exception as publication_error:
        if published:
            try:
                if backup is None:
                    destination.unlink()
                else:
                    os.replace(backup, destination)
                    backup = None
                _sync_directory(destination.parent)
            except Exception as rollback_error:
                recovery = f"; previous snapshot retained at {backup}" if backup is not None else ""
                # A failed rollback must retain its recovery copy, not erase it.
                backup = None
                raise OSError(
                    f"snapshot publication failed ({publication_error}); rollback failed "
                    f"({rollback_error}){recovery}"
                ) from publication_error
        raise
    finally:
        if backup is not None:
            backup.unlink(missing_ok=True)


def _write_member(archive: zipfile.ZipFile, name: str, data: bytes) -> None:
    info = zipfile.ZipInfo(name)
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o600) << 16
    info.compress_type = zipfile.ZIP_STORED
    archive.writestr(info, data)


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate manifest key")
        result[key] = value
    return result


def _check_directory(stream) -> None:
    """Bound ZIP's central-directory allocation before ZipFile parses it."""
    size = os.fstat(stream.fileno()).st_size
    stream.seek(max(0, size - 65557))
    ending = stream.read(65557)
    offset = ending.rfind(b"PK\x05\x06")
    if offset < 0 or len(ending) - offset < 22:
        raise ValueError("missing ZIP end record")
    # ZipFile consults a ZIP64 locator immediately before the classic EOCD,
    # even when the classic fields are small. Our writer never emits ZIP64.
    # The magic prefix precedes every record, so a locator position always exists.
    end_position = max(0, size - 65557) + offset
    stream.seek(end_position - 20)
    if stream.read(4) == b"PK\x06\x07":
        raise ValueError("ZIP64 snapshot archives are unsupported")
    _, disk, directory_disk, disk_count, count, directory_size, directory_offset, comment_size = (
        struct.unpack("<4s4H2LH", ending[offset : offset + 22])
    )
    if (
        disk
        or directory_disk
        or disk_count != count
        or not 2 <= count <= _MAX_MEMBERS
        or not count * 46 <= directory_size <= 16 * 1024 * 1024
        or directory_offset < len(_MAGIC)
        or directory_offset + directory_size != end_position
        or offset + 22 + comment_size != len(ending)
    ):
        raise ValueError("unsupported or excessive ZIP directory")
    # ZipFile ignores EOCD's entry count and parses the whole byte range. Check
    # actual entry boundaries before it can allocate an unbounded ZipInfo list.
    stream.seek(directory_offset)
    for _ in range(count):
        if end_position - stream.tell() < 46:
            raise ValueError("ZIP directory entry count does not match its contents")
        header = stream.read(46)
        if len(header) != 46 or header[:4] != b"PK\x01\x02":
            raise ValueError("invalid ZIP directory entry")
        name_size, extra_size, entry_comment_size = struct.unpack_from("<3H", header, 28)
        entry_end = stream.tell() + name_size + extra_size + entry_comment_size
        if entry_end > end_position:
            raise ValueError("ZIP directory entry exceeds its declared bounds")
        stream.seek(entry_end)
    if stream.tell() != end_position:
        raise ValueError("ZIP directory entry count does not match its contents")
    stream.seek(len(_MAGIC))


def read_python_bundle(path: str | Path, *, trusted: bool) -> LoadedBundle:
    # This gate precedes even manifest parsing and optional dependency inspection.
    if trusted is not True:
        raise SnapshotTrustRequired(
            "Python snapshots can execute arbitrary Python code and contain credentials; pass trusted=True only for a trusted snapshot"
        )
    root: Path | None = None
    try:
        source = Path(path).expanduser()
        with source.open("rb") as stream:
            if os.fstat(stream.fileno()).st_size > _MAX_ARCHIVE_BYTES:
                raise ValueError("snapshot archive exceeds hard size limit")
            if stream.read(len(_MAGIC)) != _MAGIC:
                raise ValueError("not a supported Python snapshot archive")
            _check_directory(stream)
            with zipfile.ZipFile(stream, "r") as archive:
                # _check_directory already bounded the entry count.
                infos = archive.infolist()
                by_name = {}
                total = 0
                for info in infos:
                    name = info.filename
                    if (
                        info.orig_filename != name
                        or not _safe_name(name)
                        or (name not in ("manifest.json", "graph.pkl") and not _resource_name(name))
                    ):
                        raise ValueError("unsupported or unsafe archive member")
                    if name in by_name:
                        raise ValueError("duplicate archive member")
                    mode = info.external_attr >> 16
                    if stat.S_IFMT(mode) not in (0, stat.S_IFREG) or info.is_dir():
                        raise ValueError("archive members must be regular files, not links")
                    if (
                        info.compress_type != zipfile.ZIP_STORED
                        or info.flag_bits & 1
                        or info.compress_size != info.file_size
                    ):
                        raise ValueError("compressed or encrypted members are unsupported")
                    total += info.file_size
                    if total > MAX_SNAPSHOT_BYTES + _MAX_ARTIFACT_BYTES:
                        raise ValueError("expanded snapshot exceeds hard size limit")
                    by_name[name] = info
                if "manifest.json" not in by_name or "graph.pkl" not in by_name:
                    raise ValueError("missing required archive members")
                logical_size = by_name["manifest.json"].file_size + by_name["graph.pkl"].file_size
                if logical_size > MAX_SNAPSHOT_BYTES:
                    raise ValueError("snapshot metadata and graph exceed 64 MiB")
                manifest = json.loads(
                    archive.read("manifest.json"), object_pairs_hook=_unique_object
                )
                _primitives(manifest)
                if type(manifest) is not dict or set(manifest) != {
                    "format",
                    "format_version",
                    "environment",
                    "metadata",
                    "members",
                }:
                    raise ValueError("invalid snapshot manifest")
                if (
                    manifest["format"] != _FORMAT
                    or type(manifest["format_version"]) is not int
                    or manifest["format_version"] != _VERSION
                ):
                    raise ValueError("unsupported Python snapshot format/version")
                metadata = manifest["metadata"]
                if type(metadata) is not dict:
                    raise ValueError("snapshot metadata must be an object")
                resource_limit = _resource_limit(metadata)
                if total - logical_size > resource_limit:
                    raise ValueError("artifact payloads exceed saved resource allowance")
                environment = manifest["environment"]
                current = _runtime_identity()
                if type(environment) is not dict or set(environment) != {*current, "packages"}:
                    raise ValueError("invalid compatibility metadata")
                for key in (
                    "python_implementation",
                    "python_version",
                    "kedi_version",
                    "cloudpickle_version",
                ):
                    if environment[key] != current[key]:
                        raise ValueError(f"Python snapshot compatibility mismatch: {key}")
                packages = environment["packages"]
                if type(packages) is not dict or any(
                    type(name) is not str or type(version) is not str
                    for name, version in packages.items()
                ):
                    raise ValueError("invalid installed package requirements")
                for name, version in packages.items():
                    if _installed_version(name) != version:
                        raise ValueError(f"Python snapshot package version mismatch: {name}")
                members = manifest["members"]
                if type(members) is not dict or set(members) != set(by_name) - {"manifest.json"}:
                    raise ValueError("manifest member allowlist does not match archive")
                for name, descriptor in members.items():
                    if (
                        type(descriptor) is not dict
                        or set(descriptor) != {"size", "sha256"}
                        or type(descriptor["size"]) is not int
                        or descriptor["size"] != by_name[name].file_size
                        or type(descriptor["sha256"]) is not str
                        or re.fullmatch(r"[0-9a-f]{64}", descriptor["sha256"]) is None
                    ):
                        raise ValueError("invalid archive member descriptor")
                root = Path(tempfile.mkdtemp(prefix="kedi-python-snapshot-"))
                os.chmod(root, 0o700)
                graph = b""
                for name, descriptor in members.items():
                    digest = hashlib.sha256()
                    received = 0
                    is_graph = name == "graph.pkl"
                    chunks: list[bytes] = []
                    output = None
                    try:
                        if not is_graph:
                            destination = root
                            # Every ancestor is allocated beneath the private root.
                            for part in PurePosixPath(name).parts[:-1]:
                                destination = destination / part
                                destination.mkdir(exist_ok=True)
                                os.chmod(destination, 0o700)
                            destination = destination / PurePosixPath(name).name
                            output = destination.open("xb")
                            os.chmod(destination, 0o600)
                        with archive.open(name) as member:
                            while chunk := member.read(_CHUNK):
                                received += len(chunk)
                                digest.update(chunk)
                                if output is None:
                                    chunks.append(chunk)
                                else:
                                    output.write(chunk)
                    finally:
                        if output is not None:
                            output.close()
                    if received != descriptor["size"] or digest.hexdigest() != descriptor["sha256"]:
                        raise ValueError("archive member digest or size mismatch")
                    if is_graph:
                        graph = b"".join(chunks)
                result = LoadedBundle(graph=graph, metadata=metadata, root=root)
                root = None
                return result
    except Exception as exc:
        raise SessionLoadError(f"Cannot load Python snapshot: {exc}") from exc
    finally:
        if root is not None:
            shutil.rmtree(root, ignore_errors=True)
