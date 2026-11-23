"""API response/parse cache identity and bounds (audit #214, item 4.11).

_stable_repr used to scrub memory addresses out of repr(), so distinct
objects whose reprs differed only by address shared a cache key and could
return each other's cached response. Plain values still key by value;
opaque objects now key by a process-local identity token, and both caches
are bounded with oldest-first eviction. In-flight dedup and clear-cache
generations must keep working.
"""

from __future__ import annotations

import threading
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any

import pytest

from kedi.agent_profile import AgentProfile
from kedi.api import cache as cache_module
from kedi.api.cache import (
    _MAX_PARSE_CACHE_ENTRIES,
    _MAX_RESPONSE_CACHE_ENTRIES,
    _claim_response_cache_key,
    _finish_response_flight,
    _parse_cached,
    _response_cache_key,
    _stable_repr,
    clear_cache,
)


class ConstantRepr:
    def __init__(self, payload: str) -> None:
        self.payload = payload

    def __repr__(self) -> str:  # noqa: D105
        return "ConstantRepr()"


class Color(Enum):
    RED = 1
    BLUE = 2


def _key_for(args: tuple[object, ...]) -> str:
    return _response_cache_key(
        kind="query",
        inner=lambda: None,
        source_id="s",
        source_hash="h",
        args=args,
        kwargs={},
        config_env={},
        env={},
        adapter_identity=("pydantic", "pydantic"),
        adapter_kwargs={},
        profile=AgentProfile.empty(),
        tools={},
    )


def test_stable_repr_distinguishes_distinct_objects_with_same_repr() -> None:
    assert _stable_repr(object()) != _stable_repr(object())
    assert _stable_repr(ConstantRepr("one")) != _stable_repr(ConstantRepr("two"))
    same = ConstantRepr("one")
    assert _stable_repr(same) == _stable_repr(same)
    # Nested opaque objects cannot collide through a container repr either.
    assert _stable_repr([object()]) != _stable_repr([object()])
    assert _stable_repr({"key": object()}) != _stable_repr({"key": object()})


def test_stable_repr_still_keys_plain_values_by_value() -> None:
    assert _stable_repr("hello") == _stable_repr("hello")
    assert _stable_repr(1) == _stable_repr(1)
    assert _stable_repr(1) != _stable_repr(True)
    assert _stable_repr([1, "x"]) == _stable_repr([1, "x"])
    assert _stable_repr({"a": 1, "b": [2]}) == _stable_repr({"b": [2], "a": 1})
    assert _stable_repr({1, 2}) == _stable_repr({2, 1})
    assert _stable_repr((1, 2)) != _stable_repr([1, 2])


def test_stable_repr_covers_enums_dates_decimals_and_paths() -> None:
    assert _stable_repr(Color.RED) == _stable_repr(Color.RED)
    assert _stable_repr(Color.RED) != _stable_repr(Color.BLUE)
    assert _stable_repr(datetime(2026, 1, 1)) == _stable_repr(datetime(2026, 1, 1))
    assert _stable_repr(date(2026, 1, 1)) != _stable_repr(datetime(2026, 1, 1))
    assert _stable_repr(Decimal("1.5")) == _stable_repr(Decimal("1.5"))
    assert _stable_repr(Path("/tmp/x")) == _stable_repr(Path("/tmp/x"))
    assert _stable_repr(bytearray(b"ab")) != _stable_repr(b"ab")
    assert _stable_repr(bytearray(b"ab")) != _stable_repr(b"ac")


def test_stable_repr_falls_back_when_repr_is_broken() -> None:
    class BrokenRepr:
        def __init__(self) -> None:
            self.label = "kept"

        def __repr__(self) -> str:  # noqa: D105
            raise RuntimeError("no repr")

    rendered = _stable_repr(BrokenRepr())
    assert "<unrepresentable>" in rendered
    assert "label='kept'" in rendered


def test_stable_repr_covers_string_slots_unset_slots_and_weird_slots() -> None:
    class StringSlot:
        __slots__ = "value"

        def __init__(self, value: str) -> None:
            self.value = value

    assert _stable_repr(StringSlot("x")) == _stable_repr(StringSlot("x"))
    assert _stable_repr(StringSlot("x")) != _stable_repr(StringSlot("y"))

    class TwoSlots:
        __slots__ = ("a", "b")

        def __init__(self, a: str) -> None:
            self.a = a

    assert _stable_repr(TwoSlots("x")) == _stable_repr(TwoSlots("x"))

    class IntSlots:
        def __init__(self) -> None:
            self.x = 1

    # A non-iterable __slots__ cannot be declared (class creation rejects
    # it), so install one after creation to cover the defensive branch.
    IntSlots.__slots__ = 123  # type: ignore[attr-defined]
    assert _stable_repr(IntSlots()) == _stable_repr(IntSlots())

    class MixedSlots:
        __slots__ = ("value",)

        def __init__(self, value: str) -> None:
            self.value = value

    # Likewise, a non-string slot name cannot be declared.
    MixedSlots.__slots__ = (1, "value")  # type: ignore[attr-defined]
    assert _stable_repr(MixedSlots("x")) == _stable_repr(MixedSlots("x"))


def test_stable_repr_hashes_long_state_and_names_container_subclasses() -> None:
    class BigState:
        def __init__(self) -> None:
            self.payload = "x" * 600

    assert "sha256:" in _stable_repr(BigState())

    class MyList(list):  # type: ignore[type-arg]
        pass

    rendered = _stable_repr(MyList([1]))
    assert "MyList" in rendered
    assert rendered != _stable_repr([1])


def test_stable_repr_shares_keys_for_same_state_despite_address() -> None:
    """A repr that carries the distinguishing state (only the address
    varies) still shares a key - the address normalization the old tests
    pin is preserved; only state-free reprs stop colliding."""

    class AddressReprValue:
        def __init__(self, label: str) -> None:
            self.label = label

        def __repr__(self) -> str:  # noqa: D105
            return f"<AddressReprValue label={self.label!r} at {hex(id(self))}>"

    assert _stable_repr(AddressReprValue("same")) == _stable_repr(AddressReprValue("same"))
    assert _stable_repr(AddressReprValue("same")) != _stable_repr(AddressReprValue("other"))


def test_stable_repr_handles_slots_and_reference_cycles() -> None:
    class Slotted:
        __slots__ = ("value",)

        def __init__(self, value: str) -> None:
            self.value = value

    assert _stable_repr(Slotted("x")) == _stable_repr(Slotted("x"))
    assert _stable_repr(Slotted("x")) != _stable_repr(Slotted("y"))
    cyclic: list[object] = []
    cyclic.append(cyclic)
    other: list[object] = []
    other.append(other)
    # No RecursionError, and two distinct cyclic lists do not collide.
    assert _stable_repr(cyclic) != _stable_repr(other)
    assert _stable_repr(cyclic) == _stable_repr(cyclic)


def test_response_cache_key_distinguishes_distinct_arg_objects() -> None:
    assert _key_for((object(),)) != _key_for((object(),))
    assert _key_for((ConstantRepr("one"),)) != _key_for((ConstantRepr("two"),))
    assert _key_for(("same",)) == _key_for(("same",))


def test_parse_cache_is_bounded_and_keeps_newest() -> None:
    clear_cache()
    try:
        for i in range(_MAX_PARSE_CACHE_ENTRIES + 100):
            _parse_cached(kind="query", source=f"# program {i}\n", source_path=None)
        assert len(cache_module._PARSE_CACHE) <= _MAX_PARSE_CACHE_ENTRIES
        from kedi.api.cache import _source_hash

        first_key = ("query", None, _source_hash("# program 0\n"))
        last_key = (
            "query",
            None,
            _source_hash(f"# program {_MAX_PARSE_CACHE_ENTRIES + 99}\n"),
        )
        assert first_key not in cache_module._PARSE_CACHE
        assert last_key in cache_module._PARSE_CACHE
    finally:
        clear_cache()


def test_response_cache_is_bounded_and_keeps_newest() -> None:
    clear_cache()
    try:
        for i in range(_MAX_RESPONSE_CACHE_ENTRIES + 100):
            key = f"key-{i}"
            _, flight, is_owner = _claim_response_cache_key(key)
            assert is_owner and flight is not None
            _finish_response_flight(key, flight, result=f"value-{i}")
        assert len(cache_module._RESPONSE_CACHE) <= _MAX_RESPONSE_CACHE_ENTRIES
        assert "key-0" not in cache_module._RESPONSE_CACHE
        assert f"key-{_MAX_RESPONSE_CACHE_ENTRIES + 99}" in cache_module._RESPONSE_CACHE
    finally:
        clear_cache()


def test_response_cache_dedup_and_clear_generation_survive_bounds() -> None:
    clear_cache()
    try:
        cached, flight, is_owner = _claim_response_cache_key("dedup")
        assert cached is cache_module._CACHE_MISS and is_owner and flight is not None
        # A second claim must come from another thread: the owner re-claiming
        # its own key is the recursive-request guard, not dedup.
        second_claim: dict[str, Any] = {}

        def claim_in_thread() -> None:
            second_claim["result"] = _claim_response_cache_key("dedup")

        worker = threading.Thread(target=claim_in_thread)
        worker.start()
        worker.join(timeout=5)
        assert not worker.is_alive()
        cached, same_flight, is_owner = second_claim["result"]
        assert cached is cache_module._CACHE_MISS and not is_owner
        assert same_flight is flight
        _finish_response_flight("dedup", flight, result="done")
        cached, _, is_owner = _claim_response_cache_key("dedup")
        assert cached == "done" and not is_owner

        # A flight finished after clear_cache must not store into the new
        # generation, and the key becomes claimable again.
        _, stale_flight, is_owner = _claim_response_cache_key("stale")
        assert is_owner and stale_flight is not None
        clear_cache()
        _finish_response_flight("stale", stale_flight, result="old")
        assert "stale" not in cache_module._RESPONSE_CACHE
        _, fresh_flight, is_owner = _claim_response_cache_key("stale")
        assert is_owner and fresh_flight is not None
        _finish_response_flight("stale", fresh_flight, result="new")
        assert cache_module._RESPONSE_CACHE["stale"] == "new"

        # A failed flight stores nothing and unblocks waiters with the error.
        _, error_flight, is_owner = _claim_response_cache_key("broken")
        assert is_owner and error_flight is not None
        _finish_response_flight("broken", error_flight, error=RuntimeError("boom"))
        assert "broken" not in cache_module._RESPONSE_CACHE
        with pytest.raises(RuntimeError, match="boom"):
            error_flight.future.result()
    finally:
        clear_cache()
