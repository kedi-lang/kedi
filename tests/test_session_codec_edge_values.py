from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

import kedi
from kedi._interactive.persistence import (
    SessionLoadError,
    SessionValueDecoder,
    SessionValueEncoder,
    read_snapshot,
    write_snapshot,
)


@pytest.mark.parametrize(
    "pattern",
    [
        re.compile(b"ab+"),
        re.compile(b"\x00\xff", re.IGNORECASE),
        re.compile("ab+", re.IGNORECASE),
        re.compile("\ud800\udc00"),
    ],
)
def test_compiled_regex_survives_json_and_session_round_trip(tmp_path: Path, pattern) -> None:
    encoded = SessionValueEncoder(kedi_models={}).encode(pattern)
    restored = SessionValueDecoder(kedi_models={}).decode(json.loads(json.dumps(encoded)))
    assert type(restored.pattern) is type(pattern.pattern)
    assert restored.pattern == pattern.pattern
    assert restored.flags == pattern.flags

    path = tmp_path / "regex.kedi-state"
    with kedi.interactive() as session:
        session.execute(
            f"[pattern: Any] = `__import__('re').compile({pattern.pattern!r}, {int(pattern.flags)})`"
        )
        session.dump(path)
    with kedi.load_session(path) as restored_session:
        restored_pattern = restored_session._env["pattern"]
        assert restored_pattern.pattern == pattern.pattern
        assert type(restored_pattern.pattern) is type(pattern.pattern)
        assert restored_pattern.flags == pattern.flags


@pytest.mark.parametrize(
    "value",
    [
        "caf\udce9",
        "\ud800",
        "\udfff",
        "\ud800\udc00",
        "\udc00\ud800",
        "caf\u00e9",
        "\U0001f408",
        r"\udce9",
    ],
)
def test_python_strings_round_trip_without_losing_codepoints(tmp_path: Path, value: str) -> None:
    path = tmp_path / "string.kedi-state"
    with kedi.interactive() as session:
        session.execute(f"[value: str] = `{value!r}`")
        session.dump(path)
    path.read_text(encoding="utf-8")
    with kedi.load_session(path) as restored:
        assert restored._env["value"] == value


def test_surrogate_strings_also_work_as_nested_mapping_keys(tmp_path: Path) -> None:
    value = {"\udce9": ["\ud800\udc00", "\udfff"]}
    encoded = SessionValueEncoder(kedi_models={}).encode(value)
    path = tmp_path / "nested.json"
    write_snapshot(path, {"value": encoded})
    restored = SessionValueDecoder(kedi_models={}).decode(read_snapshot(path)["value"])
    assert restored == value


def test_ordinary_unicode_snapshot_digest_is_unchanged(tmp_path: Path) -> None:
    payload = {"value": "caf\u00e9 \U0001f408"}
    old_canonical = json.dumps(
        payload, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    path = tmp_path / "compatible.json"
    write_snapshot(path, payload)
    document = json.loads(path.read_text())
    assert document["digest"] == hashlib.sha256(old_canonical.encode()).hexdigest()
    assert read_snapshot(path) == payload


@pytest.mark.parametrize("data", ["invalid base64!", "/w==", [], None])
def test_invalid_surrogate_encoding_has_a_structured_load_error(data) -> None:
    with pytest.raises(SessionLoadError, match="environment.value"):
        SessionValueDecoder(kedi_models={}).decode(
            {"$kedi": "surrogate_str", "data": data}, path="environment.value"
        )


def test_legacy_string_regex_encoding_remains_readable() -> None:
    result = SessionValueDecoder(kedi_models={}).decode(
        {"$kedi": "regex", "pattern": "ab+", "flags": int(re.IGNORECASE)}
    )
    assert result.fullmatch("ABBB")
