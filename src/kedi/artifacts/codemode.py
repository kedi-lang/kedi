from __future__ import annotations

import atexit
import base64
import hashlib
import json
import math
from dataclasses import dataclass
from threading import RLock
from typing import TYPE_CHECKING, Any, Literal

from .errors import ArtifactCodeAccessError, ArtifactCodeBudgetError, ArtifactCodeError
from .models import ArtifactCodeResult

if TYPE_CHECKING:
    from pydantic_monty import Monty

    from .manager import ArtifactManager

OffsetFrom = Literal["start", "end"]


@dataclass(frozen=True, slots=True)
class ArtifactCodeLimits:
    max_duration_seconds: float = 5.0
    max_memory_bytes: int = 64_000_000
    max_recursion_depth: int = 100
    max_artifacts: int = 32
    max_host_calls: int = 256
    max_read_chars: int = 64_000
    max_total_read_bytes: int = 32_000_000
    max_materialize_bytes: int = 256_000
    max_stdout_bytes: int = 16_000
    max_result_bytes: int = 8_000_000
    max_result_depth: int = 64
    max_result_nodes: int = 100_000


_SANDBOX_HELPERS = """
def artifact_metadata(ref_id):
    return __kedi_artifact_metadata(ref_id)

def read_artifact(ref_id, offset=0, offset_from="start", max_chars=-1):
    return __kedi_read_artifact(ref_id, offset, offset_from, max_chars)

def find_artifact(ref_id, pattern, max_matches=20, context_chars=120):
    return __kedi_find_artifact(ref_id, pattern, max_matches, context_chars)

def get_artifact(ref_id):
    return __kedi_get_artifact(ref_id)

def iter_artifact(ref_id, chunk_chars=4000):
    chunks = []
    offset = 0
    while True:
        page = __kedi_read_artifact(ref_id, offset, "start", chunk_chars)
        if page["content"]:
            chunks.append(page["content"])
        if page["complete"]:
            return chunks
        offset = page["next_offset"]
"""


class ArtifactCodeRuntime:
    """Lazy reusable Monty pool for bounded artifact computation."""

    def __init__(self, *, limits: ArtifactCodeLimits | None = None) -> None:
        self.limits = limits or ArtifactCodeLimits()
        self._pool: Monty | None = None
        self._pool_lock = RLock()
        self._closed = False

    def execute(
        self,
        *,
        manager: ArtifactManager,
        code: str,
        artifact_refs: list[str] | tuple[str, ...],
    ) -> ArtifactCodeResult:
        if not code.strip():
            raise ArtifactCodeError("artifact code must not be empty")
        refs = tuple(dict.fromkeys(artifact_refs))
        if len(refs) > self.limits.max_artifacts:
            raise ArtifactCodeBudgetError(
                f"artifact code accepts at most {self.limits.max_artifacts} references"
            )
        bridge = _ArtifactHostBridge(manager, refs, self.limits)
        capture = self._collect_string()
        pool = self._get_pool()
        from pydantic_monty import ResourceLimits

        limits = ResourceLimits(
            max_duration_secs=self.limits.max_duration_seconds,
            max_memory=self.limits.max_memory_bytes,
            max_recursion_depth=self.limits.max_recursion_depth,
        )
        try:
            with pool.checkout(script_name="artifact_code.py", limits=limits) as session:
                result = session.feed_run(
                    f"{_SANDBOX_HELPERS}\n{code}",
                    external_lookup=bridge.external_lookup(),
                    print_callback=capture,
                    skip_type_check=True,
                )
        except ArtifactCodeError:
            raise
        except Exception as exc:
            raise ArtifactCodeError(_compact_monty_error(exc)) from exc
        safe_result = _validate_result(result, self.limits)
        return ArtifactCodeResult(
            result=safe_result,
            stdout=capture.output,
            source_refs=tuple(bridge.accessed_refs),
        )

    def close(self) -> None:
        with self._pool_lock:
            if self._closed:
                return
            self._closed = True
            pool, self._pool = self._pool, None
        if pool is not None:
            pool.__exit__(None, None, None)

    def _get_pool(self) -> Monty:
        with self._pool_lock:
            if self._closed:
                raise ArtifactCodeError("artifact code runtime is closed")
            if self._pool is None:
                from pydantic_monty import Monty

                pool = Monty(
                    min_processes=1,
                    max_processes=4,
                    request_timeout=self.limits.max_duration_seconds + 2.0,
                    max_checkouts_per_worker=256,
                )
                pool.__enter__()
                self._pool = pool
            return self._pool

    def _collect_string(self) -> Any:
        from pydantic_monty import CollectString

        return CollectString(max_bytes=self.limits.max_stdout_bytes)


class _ArtifactHostBridge:
    def __init__(
        self,
        manager: ArtifactManager,
        refs: tuple[str, ...],
        limits: ArtifactCodeLimits,
    ) -> None:
        self.manager = manager
        self.allowed_refs = frozenset(refs)
        self.limits = limits
        self.host_calls = 0
        self.read_bytes = 0
        self.accessed_refs: list[str] = []
        for ref_id in refs:
            manager.metadata(ref_id)

    def external_lookup(self) -> dict[str, Any]:
        return {
            "__kedi_artifact_metadata": self.metadata,
            "__kedi_read_artifact": self.read,
            "__kedi_find_artifact": self.find,
            "__kedi_get_artifact": self.get,
        }

    def metadata(self, ref_id: str) -> dict[str, Any]:
        self._begin_call(ref_id)
        return self.manager.metadata(ref_id)

    def read(
        self,
        ref_id: str,
        offset: int = 0,
        offset_from: OffsetFrom = "start",
        max_chars: int = -1,
    ) -> dict[str, Any]:
        self._begin_call(ref_id)
        limit = self._read_limit(max_chars)
        page = self.manager.read(
            ref_id,
            offset=offset,
            offset_from=offset_from,
            max_chars=limit,
        ).model_dump(mode="json")
        self._charge_read(page["content"])
        return page

    def find(
        self,
        ref_id: str,
        pattern: str,
        max_matches: int = 20,
        context_chars: int = 120,
    ) -> list[dict[str, Any]]:
        self._begin_call(ref_id)
        if not pattern:
            raise ArtifactCodeError("artifact search pattern cannot be empty")
        if len(pattern) > 4_096:
            raise ArtifactCodeBudgetError("artifact search pattern exceeds 4096 characters")
        if max_matches <= 0:
            raise ArtifactCodeError("artifact max_matches must be positive")
        if context_chars < 0:
            raise ArtifactCodeError("artifact context_chars must not be negative")
        max_matches = min(max_matches, 100)
        context_chars = min(context_chars, 2_000)
        chunk_chars = min(
            self.limits.max_read_chars,
            max(4_000, len(pattern) + context_chars * 2),
        )
        offset = 0
        overlap = ""
        matches: list[dict[str, Any]] = []
        seen: set[int] = set()
        while len(matches) < max_matches:  # pragma: no branch - validated positive limit
            page = self.manager.read(ref_id, offset=offset, max_chars=chunk_chars)
            self._charge_read(page.content)
            window = overlap + page.content
            base = offset - len(overlap)
            cursor = 0
            while len(matches) < max_matches:
                index = window.find(pattern, cursor)
                if index < 0:
                    break
                absolute = base + index
                if absolute not in seen:
                    seen.add(absolute)
                    left = max(0, index - context_chars)
                    right = min(len(window), index + len(pattern) + context_chars)
                    matches.append(
                        {
                            "offset": absolute,
                            "match": pattern,
                            "context": window[left:right],
                        }
                    )
                cursor = index + len(pattern)
            if page.complete or page.next_offset is None:
                break
            overlap = window[-max(0, len(pattern) - 1) :]
            offset = page.next_offset
        return matches

    def get(self, ref_id: str) -> Any:
        self._begin_call(ref_id)
        metadata = self.manager.metadata(ref_id)
        size_bytes = int(metadata["size_bytes"])
        if size_bytes > self.limits.max_materialize_bytes:
            raise ArtifactCodeBudgetError(
                f"artifact {ref_id!r} is {size_bytes} bytes; use iter_artifact, "
                "read_artifact, or find_artifact instead"
            )
        if self.read_bytes + size_bytes > self.limits.max_total_read_bytes:
            raise ArtifactCodeBudgetError("artifact code cumulative read budget exceeded")
        value = _validate_result(self.manager.get_value(ref_id), self.limits)
        self.read_bytes += size_bytes
        return value

    def _begin_call(self, ref_id: str) -> None:
        if ref_id not in self.allowed_refs:
            raise ArtifactCodeAccessError(
                f"artifact {ref_id!r} is not in this execution's artifact_refs allowlist"
            )
        self.host_calls += 1
        if self.host_calls > self.limits.max_host_calls:
            raise ArtifactCodeBudgetError("artifact code host-call budget exceeded")
        if ref_id not in self.accessed_refs:
            self.accessed_refs.append(ref_id)

    def _read_limit(self, requested: int) -> int:
        if requested == -1:
            return min(4_000, self.limits.max_read_chars)
        if requested <= 0:
            raise ArtifactCodeError("max_chars must be -1 or positive")
        return min(requested, self.limits.max_read_chars)

    def _charge_read(self, content: str) -> None:
        self.read_bytes += len(content.encode("utf-8"))
        if self.read_bytes > self.limits.max_total_read_bytes:
            raise ArtifactCodeBudgetError("artifact code cumulative read budget exceeded")


def artifact_code_provenance(code: str, source_refs: tuple[str, ...]) -> dict[str, Any]:
    return {
        "operation": "run_artifact_code",
        "source_refs": list(source_refs),
        "code_digest": hashlib.sha256(code.encode("utf-8")).hexdigest(),
        "derivation_depth": 1,
    }


def _validate_result(value: Any, limits: ArtifactCodeLimits) -> Any:
    nodes = 0
    active: set[int] = set()

    def visit(item: Any, depth: int) -> Any:
        nonlocal nodes
        nodes += 1
        if nodes > limits.max_result_nodes:
            raise ArtifactCodeBudgetError("artifact code result contains too many values")
        if depth > limits.max_result_depth:
            raise ArtifactCodeBudgetError("artifact code result is nested too deeply")
        if item is None or isinstance(item, (bool, int, str)):
            return item
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ArtifactCodeError("artifact code result contains a non-finite float")
            return item
        if isinstance(item, bytes):
            return item
        item_id = id(item)
        if item_id in active:
            raise ArtifactCodeError("artifact code result contains a cycle")
        active.add(item_id)
        try:
            if isinstance(item, (list, tuple)):
                return [visit(child, depth + 1) for child in item]
            if isinstance(item, dict):
                if not all(isinstance(key, str) for key in item):
                    raise ArtifactCodeError("artifact code result object keys must be strings")
                return {key: visit(child, depth + 1) for key, child in item.items()}
        finally:
            active.remove(item_id)
        raise ArtifactCodeError(
            f"artifact code result type {type(item).__name__!r} is not JSON-compatible"
        )

    validated = visit(value, 0)
    encoded = json.dumps(
        validated,
        allow_nan=False,
        ensure_ascii=False,
        default=_json_bytes,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(encoded) > limits.max_result_bytes:
        raise ArtifactCodeBudgetError(
            f"artifact code result exceeds {limits.max_result_bytes} bytes"
        )
    return validated


def _json_bytes(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"type": "bytes", "base64": base64.b64encode(value).decode("ascii")}
    raise TypeError


def _compact_monty_error(exc: Exception) -> str:
    message = str(exc).strip() or type(exc).__name__
    first_line = message.splitlines()[0]
    return f"artifact code failed: {first_line[:500]}"


_default_runtime = ArtifactCodeRuntime()
atexit.register(_default_runtime.close)


def run_artifact_code(
    *,
    manager: ArtifactManager,
    code: str,
    artifact_refs: list[str] | tuple[str, ...],
) -> ArtifactCodeResult:
    return _default_runtime.execute(manager=manager, code=code, artifact_refs=artifact_refs)
