from __future__ import annotations

import json
from datetime import date, time
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from kedi.integrations.harbor import request_observer as observer


def test_completed_request_usage_can_be_recovered_from_partial_evidence(tmp_path: Path) -> None:
    path = tmp_path / "model-requests.jsonl"
    completed = [
        {
            "event": "model_request_completed",
            "response": {
                "parts": [{"kind": "tool-call"}, {"kind": "thinking"}],
                "usage": {
                    "input_tokens": 100,
                    "cache_read_tokens": 60,
                    "cache_write_tokens": 5,
                    "output_tokens": 10,
                    "total_tokens": 110,
                    "cost_usd": 0.01,
                },
            },
        },
        {
            "event": "model_request_completed",
            "response": {
                "parts": [{"kind": "tool-call"}, {"kind": "tool-call"}],
                "usage": {
                    "input_tokens": 200,
                    "cache_read_tokens": 150,
                    "cache_write_tokens": 0,
                    "output_tokens": 20,
                    "total_tokens": 220,
                    "cost_usd": "0.02",
                },
            },
        },
    ]
    path.write_text(
        "\n".join(json.dumps(value) for value in [{"event": "model_request_started"}, *completed])
        + "\n",
        encoding="utf-8",
    )

    recovered = observer.recover_request_usage(path)

    assert recovered is not None
    assert recovered.usage.requests == 2
    assert recovered.usage.tool_calls == 3
    assert recovered.usage.input_tokens == 300
    assert recovered.usage.cache_read_tokens == 210
    assert recovered.usage.cache_write_tokens == 5
    assert recovered.usage.output_tokens == 30
    assert recovered.usage.total_tokens == 330
    assert recovered.usage.cost_usd == Decimal("0.03")
    assert recovered.invalid_events == 0
    assert recovered.events_without_cost == 0

    with path.open("a", encoding="utf-8") as stream:
        stream.write("{truncated\n")
        value = completed[0] | {
            "response": completed[0]["response"]
            | {"usage": completed[0]["response"]["usage"] | {"cost_usd": None}}
        }
        stream.write(json.dumps(value) + "\n")
        stream.write(
            json.dumps(
                completed[0]
                | {
                    "response": completed[0]["response"]
                    | {"usage": completed[0]["response"]["usage"] | {"input_tokens": True}}
                }
            )
            + "\n"
        )

    partial = observer.recover_request_usage(path)
    assert partial is not None
    assert partial.usage.requests == 3
    assert partial.usage.cost_usd is None
    assert partial.invalid_events == 2
    assert partial.events_without_cost == 1
    assert observer.recover_request_usage(tmp_path / "missing.jsonl") is None


def test_request_usage_recovery_rejects_unusable_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "model-requests.jsonl"
    path.write_text(
        json.dumps({"event": "model_request_completed", "response": None}) + "\n",
        encoding="utf-8",
    )
    assert observer.recover_request_usage(path) is None

    usage = {
        "input_tokens": 10,
        "cache_read_tokens": 5,
        "cache_write_tokens": 0,
        "output_tokens": 2,
        "total_tokens": 12,
    }
    events = [
        {
            "event": "model_request_completed",
            "response": {"parts": "not-a-part-list", "usage": usage | {"cost_usd": "bad"}},
        },
        {
            "event": "model_request_completed",
            "response": {"parts": None, "usage": usage},
        },
    ]
    path.write_text(
        "\n".join(json.dumps(event) for event in events) + "\n",
        encoding="utf-8",
    )

    recovered = observer.recover_request_usage(path)

    assert recovered is not None
    assert recovered.usage.requests == 2
    assert recovered.usage.tool_calls == 0
    assert recovered.usage.cost_usd is None
    assert recovered.events_without_cost == 2
    assert observer._normalized_usage_values(None) is None
    assert observer._nonnegative_decimal("NaN") is None

    original_open = Path.open

    def fail_for_evidence(candidate: Path, *args, **kwargs):
        if candidate == path:
            raise OSError("unreadable evidence")
        return original_open(candidate, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_for_evidence)
    assert observer.recover_request_usage(path) is None


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
