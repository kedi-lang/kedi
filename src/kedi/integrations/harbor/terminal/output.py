from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

OutputStream = Literal["stdout", "stderr", "combined"]


ProcessLifetime = Literal["session", "execution"]


_WRITE_TOOLS = frozenset(
    {
        "write_text_file",
        "apply_patch",
        "create_directory",
        "remove_file",
        "remove_directory",
    }
)


_VERIFICATION_COMMANDS = frozenset(
    {
        "cargo test",
        "go test",
        "make check",
        "make prod",
        "make test",
        "make tests",
        "mypy",
        "npm test",
        "pnpm test",
        "pre-commit",
        "python -m pytest",
        "python3 -m pytest",
        "pytest",
        "ruff check",
        "tox",
        "ty check",
        "uv run pytest",
        "yarn test",
    }
)


_NON_EXECUTING_VERIFICATION_FLAGS = frozenset(
    {"--collect-only", "--help", "--list", "--version", "-h"}
)


_READ_ONLY_COMMANDS = frozenset({"cat", "file", "head", "ls", "pwd", "stat", "tail", "wc"})


_READ_ONLY_GIT_COMMANDS = frozenset(
    {"diff", "grep", "log", "ls-files", "rev-parse", "show", "status"}
)


_TOOL_AUDIT_PREVIEW_CHARS = 4_096


_TOOL_AUDIT_KEY_CHARS = 256


_TOOL_AUDIT_MAX_ITEMS = 32


_TOOL_AUDIT_MAX_DEPTH = 4


@dataclass(slots=True)
class _BytePreview:
    limit: int
    total_bytes: int = 0
    head: bytearray = field(default_factory=bytearray)
    tail: bytearray = field(default_factory=bytearray)

    def append(self, chunk: bytes) -> None:
        self.total_bytes += len(chunk)
        if len(self.head) < self.limit:
            needed = self.limit - len(self.head)
            self.head.extend(chunk[:needed])
        self.tail.extend(chunk)
        if len(self.tail) > self.limit:
            del self.tail[: len(self.tail) - self.limit]

    def render(self) -> tuple[str, bool, bool]:
        has_tail = self.total_bytes > len(self.head)
        truncated = self.total_bytes > len(self.head) + len(self.tail)
        payload = bytes(self.head)
        if truncated:
            omitted = self.total_bytes - len(self.head) - len(self.tail)
            payload += f"\n... {omitted} bytes omitted ...\n".encode()
            payload += bytes(self.tail)
        elif has_tail:
            overlap = max(0, len(self.head) + len(self.tail) - self.total_bytes)
            payload += bytes(self.tail[overlap:])
        invalid_utf8 = False
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError:
            text = payload.decode("utf-8", errors="replace")
            invalid_utf8 = True
        return text, truncated, invalid_utf8


def _foreground_result(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    """Project a completed command into the compact result shown to the model."""

    result = {
        "process_id": snapshot["process_id"],
        "exit_code": snapshot["exit_code"],
        "duration_seconds": round(float(snapshot["duration_seconds"]), 3),
        "stdout": snapshot["stdout"],
        "stderr": snapshot["stderr"],
    }
    if "output_continuations" in snapshot:
        result["output_continuations"] = snapshot["output_continuations"]
        result["output_excerpt"] = snapshot["output_excerpt"]
    for stream in ("stdout", "stderr"):
        if snapshot[f"{stream}_truncated"] or snapshot["output_limited"]:
            result[f"{stream}_bytes"] = snapshot[f"{stream}_bytes"]
        if snapshot[f"{stream}_truncated"]:
            result[f"{stream}_truncated"] = True
        if snapshot[f"{stream}_invalid_utf8"]:
            result[f"{stream}_invalid_utf8"] = True
    for flag in ("timed_out", "output_limited"):
        if snapshot[flag]:
            result[flag] = True
    if snapshot.get("status") == "failed_to_start":
        result["status"] = "failed_to_start"
        result["start_error_type"] = snapshot.get("start_error_type")
    return result


def _preview_file(path: Path, limit: int) -> tuple[str, bool, bool, int]:
    try:
        total = path.stat().st_size
    except OSError:
        return "", False, False, 0
    if total == 0:
        return "", False, False, 0
    with path.open("rb") as handle:
        if total <= limit * 2:
            payload = handle.read()
        else:
            head = handle.read(limit)
            handle.seek(max(0, total - limit))
            tail = handle.read(limit)
            omitted = total - len(head) - len(tail)
            payload = head + f"\n... {omitted} bytes omitted ...\n".encode() + tail
    invalid_utf8 = False
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        text = payload.decode("utf-8", errors="replace")
        invalid_utf8 = True
    return text, total > limit * 2, invalid_utf8, total


class _empty_binary_file:
    def __enter__(self) -> _empty_binary_file:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def seek(self, _: int) -> None:
        return None

    def read(self, _: int) -> bytes:
        return b""
