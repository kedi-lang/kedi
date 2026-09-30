from __future__ import annotations

import gc
import importlib
import json
import os
import threading
import weakref
from collections.abc import Mapping
from concurrent.futures import Future
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, computed_field, model_serializer

from kedi.lang.compiler.environment import KediEnv, ScopeFrame
from kedi.promise import KediPromise


@pytest.fixture
def inspection(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "debugger" / "src"))
    return importlib.import_module("kedi_debugger.inspection")


def _children(node: dict) -> dict[str, dict]:
    return {child["name"]: child for child in node.get("children", [])}


def _nodes(node: dict) -> list[dict]:
    return [node, *(item for child in node.get("children", []) for item in _nodes(child))]


def _hostile(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("Inspector invoked user code")


class Hostile:
    __repr__ = __str__ = __iter__ = __bool__ = __reduce__ = __deepcopy__ = _hostile
    __getattribute__ = _hostile

    @property
    def secret(self):
        return _hostile()


def test_unknown_objects_and_mapping_iterators_are_opaque(inspection):
    class HostileMapping(Mapping):
        __getitem__ = __iter__ = __len__ = __repr__ = _hostile

    class HostileDict(dict):
        items = __getitem__ = __iter__ = __len__ = __repr__ = _hostile

    class HostileList(list):
        __getitem__ = __iter__ = __len__ = __repr__ = _hostile

    values = [Hostile(), HostileMapping(), HostileDict(), HostileList(), iter([1, 2])]
    for value in values:
        node = inspection.Snapshot().capture("value", value)
        assert "opaque" in node["value"]
        assert "children" not in node
        json.dumps(node)
    assert next(values[-1]) == 1


def test_hostile_metaclass_and_class_spoofing_are_not_invoked(inspection):
    class Meta(type):
        __getattribute__ = __eq__ = __hash__ = _hostile

        @property
        def __name__(self):
            return _hostile()

        @property
        def __mro__(self):
            return _hostile()

    class Value(metaclass=Meta):
        __getattribute__ = _hostile

    node = inspection.Snapshot().capture("value", Value())
    assert node["type"] == "Value"
    assert "opaque" in node["value"]


def test_builtin_scalars_and_containers_are_json_safe(inspection):
    value = {"list": [None, True, 42, 1.25, float("nan")], "tuple": ("a",), "set": {2, 3}}
    value["frozen"] = frozenset([4])
    value["bytes"] = b"hello"
    value["bytearray"] = bytearray(b"world")
    node = inspection.Snapshot().capture("root", value)
    assert json.loads(json.dumps(node, allow_nan=False)) == node
    assert [child["value"] for child in _children(node)["list"]["children"]] == [
        "None",
        "True",
        "42",
        "1.25",
        "nan",
    ]
    assert _children(node)["bytes"]["value"] == "hello"
    assert all(
        type(item[field]) is str for item in _nodes(node) for field in ("name", "type", "value")
    )


def test_scope_reads_only_own_raw_storage(inspection, monkeypatch):
    future = Future()
    promise = KediPromise(future, "result")
    parent = ScopeFrame({"inherited": 3})
    scope = ScopeFrame({"result": promise, "local": 2}, parent=parent)
    for method in ("__getitem__", "get", "items", "__iter__", "raw_value", "visible_bindings"):
        monkeypatch.setattr(ScopeFrame, method, _hostile)
    node = inspection.Snapshot().capture("locals", scope)
    assert set(_children(node)) == {"result", "local"}
    assert _children(node)["result"]["value"] == "<promise: pending>"
    assert promise._resolved is False
    assert not future.done()


def test_python_environment_does_not_resolve_or_expose_builtins(inspection, monkeypatch):
    promise = KediPromise(Future(), "answer")
    environment = KediEnv({"answer": promise, "__builtins__": Hostile()})
    for method in ("__getitem__", "get", "items", "__iter__", "raw_value", "native_value"):
        monkeypatch.setattr(KediEnv, method, _hostile)
    node = inspection.Snapshot().capture("prelude", environment)
    assert list(_children(node)) == ["answer"]
    assert _children(node)["answer"]["value"] == "<promise: pending>"
    assert promise._resolved is False
    assert environment._dirty_bindings == set()


@pytest.mark.parametrize("state", ["pending", "running", "complete", "failed", "cancelled"])
def test_promises_are_observed_without_waiting_or_memoizing(inspection, state):
    future = Future()
    promise = KediPromise(future, "result")
    if state == "running":
        future.set_running_or_notify_cancel()
    elif state == "complete":
        future.set_result({"result": Hostile()})
    elif state == "failed":

        class Failure(Exception):
            __str__ = __repr__ = _hostile

        future.set_exception(Failure())
    elif state == "cancelled":
        future.cancel()
    for method in ("result", "exception", "done", "running", "cancelled"):
        setattr(future, method, _hostile)
    node = inspection.Snapshot().capture("result", promise)
    expected = "pending" if state == "running" else state
    if state == "complete":
        assert node["value"].startswith("<opaque Hostile")
    else:
        assert node["value"] == f"<promise: {expected}>"
    assert "children" not in node
    assert promise._resolved is False
    assert promise._value is None


def test_unknown_promise_future_and_subclasses_are_opaque(inspection):
    class UnsafeFuture(Future):
        __getattribute__ = _hostile

    class UnsafePromise(KediPromise):
        __getattribute__ = _hostile

    unknown = inspection.Snapshot().capture("result", KediPromise(UnsafeFuture(), None))
    assert unknown["value"] == "<promise: unknown>"
    opaque = inspection.Snapshot().capture("result", UnsafePromise(Future(), None))
    assert "opaque" in opaque["value"]
    uninitialized = object.__new__(KediPromise)
    assert inspection.Snapshot().capture("result", uninitialized)["value"] == "<promise: unknown>"


@pytest.mark.parametrize(
    "state",
    ["hostile", "unexpected", "CANCELLED_AND_NOTIFIED"],
    ids=["hostile", "unknown", "cancelled"],
)
def test_promise_state_storage_does_not_use_value_hooks(inspection, state):
    future = Future()
    future._state = Hostile() if state == "hostile" else state
    key = object()
    future.__dict__[key] = Hostile()
    node = inspection.Snapshot().capture("result", KediPromise(future, None))
    expected = (
        "cancelled" if type(state) is str and state == "CANCELLED_AND_NOTIFIED" else "unknown"
    )
    assert node["value"] == f"<promise: {expected}>"


def test_promise_with_enormous_storage_is_bounded(inspection):
    future = Future()
    future.__dict__.update({f"extra_{index}": Hostile() for index in range(1000)})
    node = inspection.Snapshot().capture("result", KediPromise(future, None))
    assert node["value"] == "<promise: unknown>"


def test_promise_storage_mutation_is_reported_without_failing_capture(inspection, monkeypatch):
    future = Future()
    snapshot = inspection.Snapshot()

    def changing_items(items):
        for entry in enumerate(items):
            yield entry
            future.__dict__["concurrent_change"] = True

    monkeypatch.setattr(inspection, "enumerate", changing_items, raising=False)
    node = snapshot.capture("result", KediPromise(future, None))
    assert node["value"] == "<promise: unknown>"


def test_model_reads_raw_dict_not_properties_serializers_or_getattribute(inspection):
    class Model(BaseModel):
        count: int
        password: str

        @computed_field
        @property
        def computed(self) -> str:
            return _hostile()

        @model_serializer
        def serialize(self):
            return _hostile()

        __repr__ = __str__ = model_dump = _hostile

    model = Model(count=7, password="private")
    Model.__getattribute__ = _hostile
    node = inspection.Snapshot().capture("model", model)
    assert set(_children(node)) == {"count", "password"}
    assert _children(node)["count"]["value"] == "7"
    assert _children(node)["password"]["value"] == "<redacted>"


def test_model_with_dict_property_is_not_read(inspection):
    class Model(BaseModel):
        count: int

        @property
        def __dict__(self):
            return _hostile()

    model = object.__new__(Model)
    node = inspection.Snapshot().capture("model", model)
    assert "children" not in node
    assert "opaque" in node["value"]


@pytest.mark.parametrize(
    "key",
    [
        "password",
        "PASSWORD",
        "api_key",
        "apiKey",
        "OPENAI_API_KEY",
        "access_token",
        "Authorization",
        "proxy-authorization",
        "Cookie",
        "set-cookie",
        "client_secret",
        "AWS_SECRET_ACCESS_KEY",
        "private_key",
        "connection_string",
        "headers",
        "environment",
        b"password",
        "x" * 1000 + "password",
    ],
)
def test_sensitive_fields_are_redacted_without_inspecting_values(inspection, key):
    node = inspection.Snapshot().capture("root", {key: Hostile()})
    assert node["children"][0]["value"] == "<redacted>"
    assert "children" not in node["children"][0]


def test_environment_secrets_redact_values_keys_types_and_bytes(inspection, monkeypatch):
    secret = "secret-credential-12345"
    monkeypatch.setenv("EXAMPLE_API_KEY", secret)
    cls = type(secret, (), {})
    node = inspection.Snapshot().capture(
        secret,
        {secret: ["prefix " + secret + " suffix", secret.encode(), cls(), {"nested": secret}]},
    )
    assert secret not in json.dumps(node)
    assert "<redacted>" in json.dumps(node)


@pytest.mark.parametrize(
    "name",
    [
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cache_read_tokens",
        "cache_write_tokens",
        "cache_creation_input_tokens",
        "cached_tokens",
        "reasoning_tokens",
        "prompt_tokens",
        "completion_tokens",
        "max_tokens",
        "max_output_tokens",
        "max_completion_tokens",
        "token_count",
        "token_budget",
        "token_limit",
        "tokens_used",
        "tokens_remaining",
        "promptTokenCount",
        "candidatesTokenCount",
        "thoughtsTokenCount",
        "cachedContentTokenCount",
        "inputTokenCount",
        "totalTokenCount",
        b"input_tokens",
    ],
)
def test_token_counters_and_limits_are_visible(inspection, name):
    node = inspection.Snapshot().capture("usage", {name: 1234})
    assert node["children"][0]["value"] == "1234"
    if type(name) is str:
        assert inspection.Snapshot().capture(name, 1234)["value"] == "1234"


def test_usage_containers_expose_metrics_but_still_hide_credentials(inspection):
    node = inspection.Snapshot().capture(
        "token_usage",
        {
            "input_tokens": 1200,
            "input_tokens_details": {"cached_tokens": 1000},
            "outputTokensDetails": {"reasoning_tokens": 30},
            "access_token": "private-access-credential",
        },
    )
    children = _children(node)
    assert children["input_tokens"]["value"] == "1200"
    assert _children(children["input_tokens_details"])["cached_tokens"]["value"] == "1000"
    assert _children(children["outputTokensDetails"])["reasoning_tokens"]["value"] == "30"
    assert children["access_token"]["value"] == "<redacted>"


@pytest.mark.parametrize(
    "name",
    [
        "token",
        "tokens",
        "access_token",
        "refreshToken",
        "API_TOKEN",
        "input_tokens_api_key",
        "token_usage_secret",
        "password_max_tokens",
    ],
)
def test_token_metric_exceptions_do_not_unmask_credentials(inspection, name):
    node = inspection.Snapshot().capture(name, Hostile())
    assert node["value"] == "<redacted>"


def test_metric_environment_values_are_not_secrets_but_known_secrets_still_are(
    inspection,
    monkeypatch,
):
    with monkeypatch.context() as patch:
        _environment(patch, {"MAX_TOKENS": "4096", "API_TOKEN": "private-credential"})
        snapshot = inspection.Snapshot()
    assert snapshot.redact_text("max_tokens=4096") == "max_tokens=4096"
    assert snapshot.capture("input_tokens", 4096)["value"] == "4096"
    assert snapshot.capture("token_usage", "private-credential")["value"] == "<redacted>"


def test_overlapping_environment_secrets_are_fully_redacted(inspection, monkeypatch):
    with monkeypatch.context() as patch:
        _environment(patch, {"FIRST_TOKEN": "abc", "SECOND_TOKEN": "cdef"})
        snapshot = inspection.Snapshot()
    assert snapshot.redact_text("abcdef") == "<redacted>"
    assert snapshot.capture("value", b"abcdef")["value"] == "<redacted>"
    assert snapshot.capture("abcdef", type("abcdef", (), {})())["type"] == "<redacted>"


def test_secret_crossing_string_boundary_is_redacted_before_truncating(inspection, monkeypatch):
    secret = "credential-with-a-long-distinctive-tail"
    monkeypatch.setenv("EXAMPLE_TOKEN", secret)
    node = inspection.Snapshot(max_string=32).capture("text", "x" * 15 + secret + "z" * 100)
    assert "credential" not in node["value"]
    assert "truncated" in node["value"]
    assert len(node["value"]) <= 32


def test_multibyte_secret_in_bytes_is_redacted_before_truncating(inspection, monkeypatch):
    secret = "\U0001f642" * 20
    monkeypatch.setenv("EXAMPLE_TOKEN", secret)
    text = ("x" * 15 + secret + "z" * 100).encode()
    node = inspection.Snapshot(max_string=32).capture("text", text)
    assert "\U0001f642" not in node["value"]
    assert "truncated" in node["value"]


def test_lone_surrogates_and_invalid_bytes_survive_utf8_json_ipc(inspection):
    node = inspection.Snapshot(max_string=32).capture(
        "\ud800", {"\udfff": ["\ud800" * 100, b"\xff"]}
    )
    encoded = json.dumps(node, ensure_ascii=False).encode("utf-8")
    assert json.loads(encoded) == node
    assert all(
        len(item[field]) <= 32 for item in _nodes(node) for field in ("name", "value", "type")
    )


def test_non_utf8_environment_secret_in_bytes_is_redacted(inspection, monkeypatch):
    with monkeypatch.context() as patch:
        _environment(patch, {b"EXAMPLE_TOKEN": b"\xff-private-value"})
        node = inspection.Snapshot().capture("text", b"\xff-private-value")
    assert node["value"] == "<redacted>"


def test_public_text_redaction_is_bounded_and_does_not_spend_node_budget(inspection, monkeypatch):
    monkeypatch.setenv("EXAMPLE_TOKEN", "never-disclose-this")
    snapshot = inspection.Snapshot(max_string=64, max_items=1)
    assert snapshot.redact_text("code = 'never-disclose-this'") == "code = '<redacted>'"
    assert len(snapshot.redact_text("x" * 10000)) <= 64
    assert snapshot.capture("answer", 42)["value"] == "42"
    with pytest.raises(TypeError):
        snapshot.redact_text(Hostile())


def _environment(monkeypatch: pytest.MonkeyPatch, storage: dict) -> None:
    # Hostile codecs ensure the inspector never uses the resolving Mapping API.
    environment = os._Environ(storage, _hostile, _hostile, _hostile, _hostile)
    monkeypatch.setattr(os, "environ", environment)


def test_source_masking_preserves_unicode_positions_and_line_endings(inspection, monkeypatch):
    secret = "private\r\n\U0001f642-next\nlast"
    with monkeypatch.context() as patch:
        _environment(patch, {"EXAMPLE_TOKEN": secret})
        snapshot = inspection.Snapshot(max_string=32)
    text = "prefix '" + secret + "'\r\n" + secret + "\n= done\r"
    expected_secret = "*******\r\n******\n****"
    masked = snapshot.mask_source(text)
    assert masked == text.replace(secret, expected_secret)
    assert len(masked) == len(text)
    assert [(i, c) for i, c in enumerate(masked) if c in "\r\n"] == [
        (i, c) for i, c in enumerate(text) if c in "\r\n"
    ]


def test_source_masking_covers_full_two_mebibytes_without_spending_node_budget(
    inspection, monkeypatch
):
    secret = "fixture-credential"
    with monkeypatch.context() as patch:
        _environment(patch, {"EXAMPLE_TOKEN": secret})
        snapshot = inspection.Snapshot(max_string=32, max_bytes=128, max_items=1)
    text = secret + "\n" + "x" * (2 * 1024 * 1024 - 2 * len(secret) - 2) + "\n" + secret
    masked = snapshot.mask_source(text)
    assert len(masked) == len(text) == 2 * 1024 * 1024
    assert masked == text.replace(secret, "*" * len(secret))
    assert snapshot.capture("answer", 42)["value"] == "42"
    assert snapshot.mask_source(secret) == "*" * len(secret)


def test_source_masking_merges_overlapping_secret_matches(inspection, monkeypatch):
    with monkeypatch.context() as patch:
        _environment(patch, {"FIRST_TOKEN": "abc", "SECOND_TOKEN": "cdef", "THIRD_TOKEN": "b"})
        snapshot = inspection.Snapshot()
    assert snapshot.mask_source("abcdef abc cdef") == "****** *** ****"


def test_source_masking_coalesces_adjacent_matches_before_allocating_fragments(
    inspection, monkeypatch
):
    with monkeypatch.context() as patch:
        _environment(patch, {"EXAMPLE_TOKEN": "x"})
        snapshot = inspection.Snapshot()
    pattern = inspection._SOURCE_CHARACTERS
    sizes = []

    class RecordingMask:
        def sub(self, replacement, text):
            sizes.append(len(text))
            return pattern.sub(replacement, text)

    monkeypatch.setattr(inspection, "_SOURCE_CHARACTERS", RecordingMask())
    assert snapshot.mask_source("x" * 65536) == "*" * 65536
    assert sizes == [65536]


def test_source_masking_is_literal_and_case_sensitive(inspection, monkeypatch):
    with monkeypatch.context() as patch:
        _environment(patch, {"EXAMPLE_TOKEN": "a.b[1]\\x"})
        snapshot = inspection.Snapshot()
    text = "a.b[1]\\x A.B[1]\\x azb1x"
    assert snapshot.mask_source(text) == "******** A.B[1]\\x azb1x"


@pytest.mark.parametrize("secrets", [{}, {"EXAMPLE_TOKEN": "absent-credential"}])
def test_source_masking_leaves_unmatched_source_unchanged(inspection, monkeypatch, secrets):
    with monkeypatch.context() as patch:
        _environment(patch, secrets)
        snapshot = inspection.Snapshot()
    text = "\U0001f642\r\n[value] = normal\n= <value>\r"
    assert snapshot.mask_source(text) == text
    assert snapshot.mask_source("") == ""


def test_source_masking_fails_closed_without_changing_positions(inspection, monkeypatch):
    with monkeypatch.context() as patch:
        _environment(patch, {"EXAMPLE_TOKEN": "x" * 8193})
        snapshot = inspection.Snapshot()
    text = "secret\r\n\U0001f642\t\nend\r"
    assert snapshot.mask_source(text) == "******\r\n**\n***\r"
    assert len(snapshot.mask_source(text)) == len(text)
    assert snapshot.mask_source("") == ""


def test_source_masking_rejects_nonexact_strings_without_hooks(inspection):
    class UnsafeText(str):
        __getattribute__ = __str__ = __len__ = _hostile

    snapshot = inspection.Snapshot()
    for value in (Hostile(), UnsafeText("source"), b"source", None):
        with pytest.raises(TypeError, match="exact string"):
            snapshot.mask_source(value)


def test_environment_scan_uses_bounded_raw_storage(inspection, monkeypatch):
    with monkeypatch.context() as patch:
        _environment(patch, {b"EXAMPLE_TOKEN": b"private-value", b"EMPTY_TOKEN": b""})
        snapshot = inspection.Snapshot()
    assert snapshot.redact_text("private-value") == "<redacted>"


def test_short_secrets_do_not_recursively_replace_redaction_markers(inspection, monkeypatch):
    with monkeypatch.context() as patch:
        _environment(patch, {"FIRST_TOKEN": "a", "SECOND_TOKEN": "redacted"})
        snapshot = inspection.Snapshot()
    assert snapshot.redact_text("aaaa") == "<redacted>" * 4


def test_snapshot_with_no_environment_secrets_does_not_redact(inspection, monkeypatch):
    with monkeypatch.context() as patch:
        _environment(patch, {"NORMAL": "not-a-secret"})
        snapshot = inspection.Snapshot()
    assert snapshot.redact_text("hello") == "hello"


@pytest.mark.parametrize(
    "storage",
    [
        {"TOO_BIG_TOKEN": "x" * 32769},
        {"TOO_BIG_TOKEN": "x" * 8193},
        {f"TOKEN_{index}": "x" * 8192 for index in range(9)},
        {f"TOKEN_{index}": str(index) for index in range(257)},
        {f"PUBLIC_{index}": "value" for index in range(4097)},
        {"TOKEN": Hostile()},
        {object(): "secret"},
    ],
    ids=["raw-size", "decoded-size", "total-size", "count", "scan-limit", "value", "key"],
)
def test_environment_overflow_and_unsafe_storage_fail_closed(inspection, monkeypatch, storage):
    with monkeypatch.context() as patch:
        _environment(patch, storage)
        snapshot = inspection.Snapshot()
    assert snapshot.capture("root", "do-not-expose")["value"] == "<redacted>"


def test_replaced_environment_is_never_iterated(inspection, monkeypatch):
    with monkeypatch.context() as patch:
        patch.setitem(vars(os), "environ", Hostile())
        snapshot = inspection.Snapshot()
    assert snapshot.capture("root", "do-not-expose")["value"] == "<redacted>"


def test_environment_with_replaced_backing_mapping_is_never_iterated(inspection, monkeypatch):
    with monkeypatch.context() as patch:
        _environment(patch, {})
        patch.setitem(vars(os.environ), "_data", Hostile())
        snapshot = inspection.Snapshot()
    assert snapshot.capture("root", "do-not-expose")["value"] == "<redacted>"


def test_environment_changed_during_scan_fails_closed(inspection, monkeypatch):
    storage = {"EXAMPLE_TOKEN": "private-value", "NORMAL": "value"}
    decode = os.fsdecode

    def changing_decode(value):
        storage.clear()
        return decode(value)

    with monkeypatch.context() as patch:
        _environment(patch, storage)
        patch.setattr(os, "fsdecode", changing_decode)
        snapshot = inspection.Snapshot()
    assert snapshot.redact_text("private-value") == "<redacted>"


def test_cycles_are_detected_but_shared_values_are_not_cycles(inspection):
    shared = [7]
    value = {"a": shared, "b": shared}
    value["self"] = value
    node = inspection.Snapshot().capture("root", value)
    assert _children(node)["self"]["value"] == "<cycle>"
    assert _children(_children(node)["a"])["0"]["value"] == "7"
    assert _children(_children(node)["b"])["0"]["value"] == "7"


def test_snapshot_is_detached_and_does_not_retain_objects(inspection):
    class Value:
        pass

    value = Value()
    reference = weakref.ref(value)
    container = [value, {"count": 1}]
    snapshot = inspection.Snapshot()
    node = snapshot.capture("root", container)
    original = json.dumps(node)
    container[1]["count"] = 9
    container.clear()
    del value
    gc.collect()
    assert reference() is None
    assert json.dumps(node) == original


def test_depth_and_item_budgets_are_shared_by_all_branches(inspection):
    node = inspection.Snapshot(max_depth=1).capture("root", {"a": [1], "b": [2]})
    for child in node["children"]:
        assert "truncated" in child["value"]
        assert "children" not in child
    snapshot = inspection.Snapshot(max_items=5)
    node = snapshot.capture("root", [[1] * 100, [2] * 100])
    assert len(_nodes(node)) == 5
    assert len(node["children"]) == 2
    assert all("truncated" in child["value"] for child in node["children"])
    with pytest.raises(inspection.SnapshotLimitError):
        snapshot.capture("other", 1)


@pytest.mark.parametrize("value", [{}, [], (), set(), frozenset()])
def test_empty_containers_need_no_children_even_at_depth_limit(inspection, value):
    node = inspection.Snapshot(max_depth=0).capture("root", value)
    assert node["children"] == []
    assert "truncated" not in node["value"]


@pytest.mark.parametrize("value", [[1, 2], {"first": 1, "second": 2}])
def test_concurrent_container_mutation_returns_explicit_truncation(inspection, monkeypatch, value):
    snapshot = inspection.Snapshot()
    scalar = snapshot._scalar
    capturing = threading.Event()
    changed = threading.Event()

    def format_scalar(item):
        capturing.set()
        assert changed.wait(5)
        return scalar(item)

    def mutate():
        if capturing.wait(5):
            value.clear()
        changed.set()

    monkeypatch.setattr(snapshot, "_scalar", format_scalar)
    thread = threading.Thread(target=mutate, daemon=True)
    thread.start()
    try:
        node = snapshot.capture("root", value)
    finally:
        capturing.set()
        thread.join(5)
    assert not thread.is_alive()
    assert "truncated: changed" in node["value"]
    assert len(node["children"]) == 1


@pytest.mark.parametrize(
    "value",
    ["x" * 1_000_000, b"x" * 1_000_000, 1 << 1000000, list(range(10000))],
    ids=["string", "bytes", "integer", "list"],
)
def test_enormous_values_are_bounded(inspection, value):
    node = inspection.Snapshot(max_items=8, max_string=64, max_bytes=512).capture("root", value)
    assert len(json.dumps(node)) <= 512
    assert len(_nodes(node)) <= 8
    assert all(
        len(item[field]) <= 64 for item in _nodes(node) for field in ("name", "value", "type")
    )
    assert "truncated" in json.dumps(node)


def test_large_container_name_uses_bounded_budget_marker(inspection):
    node = inspection.Snapshot(max_bytes=128).capture("n" * 512, [])
    assert node == {"name": "", "value": "<truncated: budget>", "type": ""}
    assert len(json.dumps(node)) <= 128


def test_invalid_capture_name_never_uses_string_conversion(inspection):
    with pytest.raises(TypeError):
        inspection.Snapshot().capture(Hostile(), 1)


def test_json_escaped_byte_budget_is_shared_across_roots(inspection):
    snapshot = inspection.Snapshot(max_bytes=1024, max_items=1000)
    roots = []
    for _ in range(100):
        try:
            roots.append(snapshot.capture("unicode", ["\U0001f642" * 50, "\x00" * 50]))
        except inspection.SnapshotLimitError:
            break
    assert len(roots) < 100
    assert sum(len(json.dumps(root)) for root in roots) <= 1024


def test_hostile_mapping_keys_never_use_repr_hash_or_equality_during_capture(inspection):
    class Key:
        __repr__ = __str__ = _hostile

    key = Key()
    value = {key: 1, (key,): 2}
    Key.__eq__ = Key.__hash__ = _hostile
    node = inspection.Snapshot().capture("root", value)
    assert len(node["children"]) == 2
    assert all("opaque" in child["name"] for child in node["children"])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_depth": -1},
        {"max_depth": 33},
        {"max_items": 0},
        {"max_items": True},
        {"max_string": 0},
        {"max_bytes": 1},
        {"max_bytes": 1 << 100},
    ],
)
def test_invalid_limits_fail_before_capture(inspection, kwargs):
    with pytest.raises(ValueError):
        inspection.Snapshot(**kwargs)


def test_completed_capture_exposes_value_without_resolving_or_leaking_secrets(inspection):
    future = Future()
    future.set_result({"answer": {"city": "Stockholm", "api_key": "private-value"}})
    promise = KediPromise(future, "answer")
    future.result = _hostile
    node = inspection.Snapshot().capture("answer", promise)
    assert _children(node)["city"]["value"] == "Stockholm"
    assert _children(node)["api_key"]["value"] == "<redacted>"
    assert promise._resolved is False


def test_large_value_does_not_hide_later_sibling_bindings(inspection):
    node = inspection.Snapshot(max_items=30).capture(
        "Locals", {"large": list(range(1000)), "answer": 42, "ready": True}
    )
    assert _children(node)["answer"]["value"] == "42"
    assert _children(node)["ready"]["value"] == "True"


def test_completed_promise_cycles_remain_bounded(inspection):
    future = Future()
    promise = KediPromise(future, None)
    future.set_result(promise)
    assert inspection.Snapshot().capture("answer", promise)["value"] == "<cycle>"


def test_completed_sensitive_capture_stays_redacted_under_an_alias(inspection):
    future = Future()
    future.set_result({"access_token": "private-value"})
    promise = KediPromise(future, "access_token")
    assert inspection.Snapshot().capture("alias", promise)["value"] == "<redacted>"


@pytest.mark.parametrize("key", ["answer", None])
def test_resolved_promise_shows_memoized_value_without_reading_source(inspection, key):
    future = Future()
    result = {"answer": 41}
    future.set_result(result if key is not None else 41)
    promise = KediPromise(future, key)
    assert promise.resolve() == 41
    if key is not None:
        result["answer"] = 99
    for method in ("result", "exception", "done", "running", "cancelled"):
        setattr(future, method, _hostile)
    assert inspection.Snapshot().capture("alias", promise)["value"] == "41"
    assert promise._value == 41


def test_resolved_sensitive_promise_keeps_capture_redaction(inspection):
    future = Future()
    future.set_result({"access_token": "private-value"})
    promise = KediPromise(future, "access_token")
    promise.resolve()
    assert inspection.Snapshot().capture("alias", promise)["value"] == "<redacted>"
