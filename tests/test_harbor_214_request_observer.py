"""Harbor request-observer message cache identity (audit #214, item 6.11).

The message cache keyed revisions by id() for str/bytes, so a replaced
same-length string that reused the freed id returned the stale fingerprint
of the old content. The cache was also unbounded. Both are fixed: revision
tokens now carry the immutable value (or a bytes snapshot for bytearray),
and the cache evicts oldest entries past a bound.
"""
from __future__ import annotations

from pathlib import Path

from pydantic_ai.messages import ModelRequest, UserPromptPart

from kedi.integrations.harbor.request_observer import (
    _MAX_MESSAGE_CACHE_ENTRIES,
    PydanticRequestEvidenceCapability,
    _revision_token,
)


def _capability(tmp_path: Path) -> PydanticRequestEvidenceCapability:
    return PydanticRequestEvidenceCapability(
        records_dir=tmp_path,
        trial_id="test",
        manifest_digest=None,
    )


def _fresh_text(n: int, ch: str) -> str:
    # Build dynamically so the string is not interned or constant-folded.
    return "".join([ch for _ in range(n)])


def test_revision_tokens_carry_string_and_bytes_values() -> None:
    assert _revision_token("aaa") == (str, "aaa")
    assert _revision_token("aaa") != _revision_token("bbb")
    assert _revision_token(b"aaa") == (bytes, b"aaa")
    assert _revision_token(b"aaa") != _revision_token(b"bbb")
    buffer = bytearray(b"aaa")
    before = _revision_token(buffer)
    buffer[:] = b"bbb"
    assert _revision_token(buffer) != before


def test_message_fingerprint_detects_replaced_text(tmp_path: Path) -> None:
    """Replacing a part's text with a different same-length string must change
    the fingerprint, even when the new string reuses the freed id of the old
    one. Before the fix the revision token was (type, id, len), so a recycled
    id returned the stale digest; the value-based token changes the revision
    regardless of allocator reuse (and, post-fix, the token itself keeps the
    old string alive so its id cannot be recycled at all)."""
    capability = _capability(tmp_path)
    message = ModelRequest(parts=[UserPromptPart(content=_fresh_text(64, "a"))])
    first = capability._message_fingerprint(message)
    old_id = id(message.parts[0].content)
    message.parts[0].content = ""  # type: ignore[assignment]
    replacement = None
    for _ in range(10_000):
        candidate = _fresh_text(64, "b")
        if id(candidate) == old_id:
            replacement = candidate
            break
        del candidate
    if replacement is None:
        # The allocator did not recycle the id in this run (post-fix the
        # cached token keeps the old string alive, so this is expected);
        # a plain same-length replacement still exercises the comparison.
        replacement = _fresh_text(64, "b")
    message.parts[0].content = replacement  # type: ignore[assignment]
    second = capability._message_fingerprint(message)
    assert second.digest != first.digest


def test_message_cache_is_bounded_and_evicted_entries_recompute(
    tmp_path: Path,
) -> None:
    capability = _capability(tmp_path)
    first_message = ModelRequest(parts=[UserPromptPart(content="message 0")])
    first_digest = capability._message_fingerprint(first_message).digest
    for i in range(1, _MAX_MESSAGE_CACHE_ENTRIES + 200):
        message = ModelRequest(parts=[UserPromptPart(content=f"message {i}")])
        capability._message_fingerprint(message)
    assert len(capability._message_cache) <= _MAX_MESSAGE_CACHE_ENTRIES
    # An evicted entry is re-fingerprinted on next use with the same digest.
    assert capability._message_fingerprint(first_message).digest == first_digest
