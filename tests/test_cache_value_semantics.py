"""Response cache keys preserve argument types and inherited instance state."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

from kedi.api import cache


def test_bytes_and_bytearray_do_not_share_response_cache_keys():
    assert cache._stable_repr(b"payload") != cache._stable_repr(bytearray(b"payload"))


def test_scalar_subclass_state_contributes_to_response_cache_keys():
    class TaggedNumber(int):
        pass

    first, second = TaggedNumber(3), TaggedNumber(3)
    first.label, second.label = "first", "second"
    assert cache._stable_repr(first) != cache._stable_repr(second)
    assert cache._stable_repr(first) != cache._stable_repr(3)


def test_inherited_and_private_slots_contribute_to_response_cache_keys():
    class Parent:
        __slots__ = ("__payload",)

        def __init__(self, payload):
            self.__payload = payload

    class Child(Parent):
        __slots__ = ("label",)

        def __init__(self, payload):
            super().__init__(payload)
            self.label = "same"

        def __repr__(self):
            return "Child()"

    first, second = Child("first"), Child("second")
    assert cache._stable_repr(first) != cache._stable_repr(second)


def test_shadowed_parent_slots_contribute_to_response_cache_keys():
    class Parent:
        __slots__ = ("payload",)

    class Child(Parent):
        __slots__ = ("payload",)

        def __repr__(self):
            return "Child()"

    first, second = Child(), Child()
    first.payload = second.payload = "same"
    Parent.payload.__set__(first, "first")
    Parent.payload.__set__(second, "second")
    assert cache._stable_repr(first) != cache._stable_repr(second)


def test_parse_cache_bounds_and_clear_are_safe_under_concurrent_access(monkeypatch):
    monkeypatch.setattr(cache, "_MAX_PARSE_CACHE_ENTRIES", 2)
    barrier = Barrier(8)

    def work(index):
        barrier.wait(timeout=5)
        for iteration in range(40):
            if index == 0:
                cache.clear_cache()
            else:
                cache._parse_cached(
                    kind="query", source=f"# {index} {iteration}\n", source_path=None
                )
            assert cache.cache_info().parse_entries <= 2

    cache.clear_cache()
    try:
        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(work, range(8)))
    finally:
        cache.clear_cache()


def test_non_weak_referenceable_cache_identities_are_bounded_and_never_recycled():
    cache.clear_cache()
    first = object()
    token = cache._object_identity(first)
    try:
        for _ in range(cache._MAX_RESPONSE_CACHE_ENTRIES + 100):
            cache._object_identity(object())
        assert len(cache._IDENTITIES) <= cache._MAX_RESPONSE_CACHE_ENTRIES
        assert cache._object_identity(first) != token
        assert cache._object_identity(first) == cache._object_identity(first)
    finally:
        cache.clear_cache()


def test_eviction_releases_cached_results_outside_the_response_lock(monkeypatch):
    monkeypatch.setattr(cache, "_MAX_RESPONSE_CACHE_ENTRIES", 1)
    observed = []

    class Result:
        def __del__(self):
            observed.append(cache._RESPONSE_CACHE_LOCK.locked())

    cache.clear_cache()
    try:
        _, flight, _ = cache._claim_response_cache_key("first")
        cache._finish_response_flight("first", flight, result=Result())
        del flight
        _, second, _ = cache._claim_response_cache_key("second")
        cache._finish_response_flight("second", second, result="second")
        assert observed and not any(observed)
    finally:
        cache.clear_cache()


def test_clear_releases_cached_results_and_identities_outside_registry_locks():
    observed = []

    class Result:
        def __del__(self):
            observed.append(cache._RESPONSE_CACHE_LOCK.locked())

    class Identity:
        __slots__ = ()

        def __del__(self):
            observed.append(cache._IDENTITY_LOCK._is_owned())

    cache.clear_cache()
    try:
        _, flight, _ = cache._claim_response_cache_key("result")
        cache._finish_response_flight("result", flight, result=Result())
        del flight
        cache._object_identity(Identity())
        cache.clear_cache()
        assert observed and not any(observed)
    finally:
        cache.clear_cache()
