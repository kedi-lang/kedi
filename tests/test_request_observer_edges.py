from __future__ import annotations

from datetime import date, time
from pathlib import Path
from types import SimpleNamespace

import pytest

from kedi.integrations.harbor import request_observer as observer


def test_observer_bounded_records_and_capture_failure(tmp_path: Path, monkeypatch) -> None:
    capability = observer.PydanticRequestEvidenceCapability(records_dir=tmp_path, trial_id="test")
    assert capability.get_ordering().position == "innermost"
    captured = capability._capture_request(None, None, sequence=1, since_previous_seconds=None)
    assert captured["request"]["capture_error"] == "AttributeError"
    assert capability.summary().observer_failures == 1
    assert capability.summary().to_dict()["usable"] is False
    monkeypatch.setattr(observer, "_MAX_EVENT_BYTES", 100)
    capability._record({"event": "test", "value": "x" * 200})
    assert '"event":"observation_dropped"' in capability.event_path.read_text()
    assert capability.summary().dropped_events == 1
    monkeypatch.setattr(observer, "_MAX_EVENTS", 1)
    capability._record({"event": "test"})
    assert capability.summary().events == 1
    assert capability.summary().dropped_events == 2


def test_history_delta_partial_parts_and_bounded_digest_list(monkeypatch) -> None:
    part = observer._PartFingerprint("text", "a", 1)
    before = observer._MessageFingerprint("request", "before", 2, (part,))
    after = observer._MessageFingerprint("request", "after", 3, (part,))
    delta = observer._history_delta([before], [after])
    assert delta["common_prefix_parts_in_changed_message"] == 1
    assert delta["first_changed"]["part"] is None
    changed = observer._MessageFingerprint(
        "request", "changed", 3, (observer._PartFingerprint("text", "b", 1),)
    )
    assert observer._history_delta([before], [changed])["first_changed"]["part"] == 0
    monkeypatch.setattr(observer, "_MAX_RECORDED_MESSAGE_DIGESTS", 2)
    assert observer._history_delta([], [before, after, changed])["omitted_message_digests"] == 1


def test_cache_settings_continuation_and_complex_values() -> None:
    result = observer._cache_setting_summary(
        {"temperature": 1, "previous_response_id": "private", "cache_control": {"ttl": 5}}
    )
    assert "temperature" not in result
    assert result["previous_response_id_digest"] != "private"
    assert len(result["cache_control_digest"]) == 64
    assert observer._cache_mode({"previous_response_id": "private"}) == "provider_continuation"


def test_model_name_fallbacks() -> None:
    assert observer._model_name(SimpleNamespace(model_id="explicit")) == "explicit"
    assert observer._model_name(SimpleNamespace(model_id=None, model=object())) == "object"
    assert (
        observer._model_name(
            SimpleNamespace(model_id=None, model=SimpleNamespace(system="fallback"))
        )
        == "fallback"
    )


def test_serialization_fallbacks_remain_metadata_only(monkeypatch) -> None:
    def broken(*args, **kwargs):
        raise ValueError("serialization failed")

    monkeypatch.setattr(observer, "to_json", broken)
    fingerprint = observer._fingerprint(object())
    assert fingerprint.byte_count > 0
    assert observer._canonical_message_value_fingerprint(object()) == fingerprint


def test_canonicalization_accepts_nonlist_parts_without_mutating_input() -> None:
    assert observer._without_message_runtime_fields([1, None]) == [1, None]
    source = {"kind": "request", "parts": None, "timestamp": "runtime"}
    assert observer._without_message_runtime_fields(source) == {"kind": "request", "parts": None}
    assert source["timestamp"] == "runtime"
    assert observer._without_message_runtime_fields({"kind": "other", "timestamp": "data"}) == {
        "kind": "other",
        "timestamp": "data",
    }


def test_revision_tokens_cover_dates_depth_and_opaque_values() -> None:
    assert observer._revision_token(date(2026, 1, 1)) == (date, date(2026, 1, 1))
    assert observer._revision_token(time(1, 2)) == (time, time(1, 2))
    value = object()
    assert observer._revision_token(value) == (object, id(value))
    assert observer._revision_token(value, depth=20) == (object, id(value))


@pytest.mark.parametrize(
    "url,expected",
    [
        (None, None),
        ("https://host:bad", "configured"),
        ("relative/path", "configured"),
        ("https://api.openai.com:443/v1", "https://api.openai.com:443/v1"),
    ],
)
def test_endpoint_categories(url, expected) -> None:
    assert observer._endpoint_category(url) == expected


def test_unknown_endpoint_is_hashed() -> None:
    assert observer._endpoint_category("https://private.example/v1").startswith("custom:")
