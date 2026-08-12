from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from hashlib import sha256
from typing import Protocol, runtime_checkable

from .models import CompactionCheckpoint, HistoryRange, HistorySnapshot

TokenCounter = Callable[[bytes], int]
ESTIMATOR_VERSION = "utf8-div3-v1"
_ARTIFACT_REF_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])(?:tool_call_result|template_output|artifact_code_result|artifact)_\d+"
    r"(?![A-Za-z0-9_])"
)


@runtime_checkable
class AdapterHistoryCodec(Protocol):
    """Inspect and splice one adapter's canonical replay representation."""

    adapter_shortname: str
    state_key: str

    def inspect(self, state: Mapping[str, object]) -> HistorySnapshot:
        """Inspect replay state without mutating or normalizing retained entries."""
        ...

    def replace_range(
        self,
        state: Mapping[str, object],
        *,
        snapshot: HistorySnapshot,
        window: HistoryRange,
        checkpoint: CompactionCheckpoint,
    ) -> dict[str, object]:
        """Replace one native interval while retaining all other values exactly."""
        ...


class HistoryCodecRegistry:
    """Explicit adapter codec registry; no implicit cross-format conversion."""

    def __init__(self) -> None:
        self._codecs: dict[str, AdapterHistoryCodec] = {}

    def register(self, codec: AdapterHistoryCodec) -> None:
        if codec.adapter_shortname in self._codecs:
            raise ValueError(f"history codec already registered for {codec.adapter_shortname!r}")
        self._codecs[codec.adapter_shortname] = codec

    def get(self, adapter_shortname: str) -> AdapterHistoryCodec:
        try:
            return self._codecs[adapter_shortname]
        except KeyError as exc:
            raise KeyError(f"no history codec registered for {adapter_shortname!r}") from exc


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        default=str,
    ).encode("utf-8")


def fingerprint_value(value: object) -> str:
    return sha256(canonical_json_bytes(value)).hexdigest()


def fingerprint_sequence(values: tuple[object, ...] | list[object]) -> str:
    digest = sha256()
    for value in values:
        encoded = canonical_json_bytes(value)
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def conservative_token_estimate(encoded: bytes) -> int:
    """Versioned, deterministic fallback that intentionally overestimates text tokens."""

    return max(1, (len(encoded) + 2) // 3 + 4)


def extract_artifact_refs(value: object) -> tuple[str, ...]:
    """Find Kedi artifact IDs in canonical provider payloads without materializing content."""

    refs: set[str] = set()
    pending = [value]
    while pending:
        current = pending.pop()
        if isinstance(current, str):
            refs.update(_ARTIFACT_REF_PATTERN.findall(current))
        elif isinstance(current, Mapping):
            pending.extend(current.keys())
            pending.extend(current.values())
        elif isinstance(current, (list, tuple)):
            pending.extend(current)
    return tuple(sorted(refs))
