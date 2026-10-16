from __future__ import annotations

import gc
import weakref
from dataclasses import replace

import pytest

from kedi.agent_adapter.adapters.lazy import LazyAdapter
from kedi.agent_profile import AgentProfile, HookSettings
from kedi.agent_profile.capabilities import AppliedCapability
from kedi.agent_profile.history import HistoryProcessingSettings
from kedi.agent_profile.instructions import CapabilityInstruction
from kedi.api import cache as cache_module
from kedi.api.cache import (
    _adapter_identity,
    _hooks_identity,
    _object_identity,
    _profile_identity,
    _stable_repr,
)


def test_cache_identity_tokens_do_not_retain_weak_referenceable_callables() -> None:
    def callback() -> None:
        pass

    token = _object_identity(callback)
    assert _object_identity(callback) == token
    reference = weakref.ref(callback)
    del callback
    gc.collect()
    assert reference() is None
    assert _object_identity(lambda: None) != token


def test_lazy_adapter_cache_identity_does_not_construct_the_backend() -> None:
    adapter = LazyAdapter(
        lambda: pytest.fail("cache identity initialized a model"),
        kind="agent-framework",
        shortname="pydantic",
    )
    assert _adapter_identity(adapter, kind="agent-framework", shortname="pydantic") == (
        "agent-framework",
        "pydantic",
        "registry",
    )


def test_cache_identity_supports_slot_only_unhashable_callables() -> None:
    class Callback:
        __slots__ = ()
        __hash__ = None

        def __call__(self) -> None:
            pass

    first, second = Callback(), Callback()
    assert _object_identity(first) == _object_identity(first)
    assert _object_identity(first) != _object_identity(second)
    assert _stable_repr(first) != _stable_repr(second)


def test_clearing_cache_does_not_break_pending_weakref_cleanup() -> None:
    def callback() -> None:
        pass

    _object_identity(callback)
    reference = cache_module._IDENTITIES[id(callback)][0]
    cache_module.clear_cache()
    del callback
    gc.collect()
    assert reference() is None


@pytest.mark.parametrize(
    "event",
    [
        "user_prompt_submit",
        "pre_tool_use",
        "post_tool_use",
        "post_tool_use_failure",
        "post_tool_result_failure",
    ],
)
def test_each_hook_event_contributes_callable_identity(event: str) -> None:
    def make_hook():
        return lambda event: None

    first, second = make_hook(), make_hook()
    assert first.__qualname__ == second.__qualname__
    assert _hooks_identity(HookSettings(**{event: (first,)})) != _hooks_identity(
        HookSettings(**{event: (second,)})
    )


def test_cache_profile_tracks_capability_instruction_and_history_callbacks() -> None:
    profile = AgentProfile()
    instruction = CapabilityInstruction(identity="review", content="Review carefully.")
    contribution = AppliedCapability(
        identity="review",
        name="review",
        description="Review",
        profile=AgentProfile(system="Review."),
    )
    changed = replace(profile, capability_instructions=(instruction,), capabilities=(contribution,))
    assert _profile_identity(profile) != _profile_identity(changed)
    assert _profile_identity(changed) != _profile_identity(
        replace(changed, capabilities=(replace(contribution, deferred=True),))
    )

    def make_processor():
        return lambda context: context.messages

    first, second = make_processor(), make_processor()
    assert _profile_identity(
        replace(
            profile,
            history_processing=HistoryProcessingSettings(
                processor=first,
            ),
        )
    ) != _profile_identity(
        replace(
            profile,
            history_processing=HistoryProcessingSettings(
                processor=second,
            ),
        )
    )
