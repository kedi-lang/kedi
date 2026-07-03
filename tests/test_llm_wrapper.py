"""Tests for ThinkingStrippingLM and _strip_thinking."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
from kedi.agent_adapter.llm_wrapper import ThinkingStrippingLM, _strip_thinking


class TestStripThinking:
    # -- XML thinking block stripping (primary strategy) -------------------

    def test_xml_thinking_block_stripped(self) -> None:
        raw = (
            "  thinking\n"
            "Some reasoning here...\n"
            "  thinking\n\n"
            "  response\n"
            "[[ ## capital ## ]]\n"
            "Stockholm\n"
            "[[ ## completed ## ]]\n"
            "  response\n"
        )
        result = _strip_thinking(raw)
        assert "thinking" in result.thinking.lower()
        assert "Some reasoning" in result.thinking
        assert "[[ ## capital ## ]]" in result.output
        assert "Stockholm" in result.output
        assert "[[ ## completed ## ]]" in result.output

    def test_xml_thinking_strips_thought_tag(self) -> None:
        raw = (
            "  thought\n"
            "reasoning...\n"
            "  thought\n\n"
            "  output\n"
            "result: 42\n"
            "  output\n"
        )
        result = _strip_thinking(raw)
        assert "reasoning" in result.thinking
        assert result.output == "result: 42"

    def test_xml_thinking_strips_reasoning_tag(self) -> None:
        raw = (
            "  reasoning\n"
            "deep thought\n"
            "  reasoning\n\n"
            "  response\n"
            "answer: yes\n"
            "  response\n"
        )
        result = _strip_thinking(raw)
        assert "deep thought" in result.thinking
        assert result.output == "answer: yes"

    def test_xml_thinking_with_markers_in_thinking(self) -> None:
        """Simulates the actual DeepSeek bug: markers IN the thinking block."""
        raw = (
            "  thinking\n"
            "Thinking process:\n"
            "The output should be [[ ## capital ## ]]\n"
            "{capital}\n"
            "[[ ## completed ## ]]\n"
            "  thinking\n\n"
            "  response\n"
            "[[ ## capital ## ]]\n"
            "Stockholm\n"
            "[[ ## completed ## ]]\n"
            "  response\n"
        )
        result = _strip_thinking(raw)
        assert "[[ ## capital ## ]]" in result.thinking
        assert "[[ ## completed ## ]]" in result.thinking
        assert "[[ ## capital ## ]]" in result.output
        assert "Stockholm" in result.output
        assert "[[ ## completed ## ]]" in result.output
        assert "Thinking process" not in result.output

    # -- Field-marker fallback (when no XML thinking tags present) --------

    def test_no_marker_returns_as_is(self) -> None:
        result = _strip_thinking("Stockholm is the capital.")
        assert result.thinking == ""
        assert result.output == "Stockholm is the capital."

    def test_field_marker_fallback_skips_completed(self) -> None:
        raw = (
            "garbage preamble\n"
            "[[ ## capital ## ]]\n"
            "more garbage\n"
            "[[ ## completed ## ]]\n"
            "even more\n"
            "[[ ## capital ## ]]\n"
            "Stockholm\n"
            "[[ ## completed ## ]]\n"
        )
        result = _strip_thinking(raw)
        assert "garbage" in result.thinking
        assert result.output.startswith("[[ ## capital ## ]]")
        assert "Stockholm" in result.output

    def test_field_marker_fallback_multi_word_field(self) -> None:
        raw = "garbage text\n[[ ## first name ## ]]\nAlice\n"
        result = _strip_thinking(raw)
        assert "garbage text" in result.thinking
        assert result.output.startswith("[[ ## first name ## ]]")

    def test_empty_string(self) -> None:
        result = _strip_thinking("")
        assert result.thinking == ""
        assert result.output == ""


class TestThinkingStrippingLM:
    """Tests for ThinkingStrippingLM.__call__ stripping behavior.

    We mock out dspy.LM.__init__ and dspy.LM.__call__ so we don't need
    a real API key or network.  The subclass still goes through the
    stripping logic in its overridden __call__.
    """

    def test_call_strips_thinking(self) -> None:
        with _mock_lm_init(), _mock_lm_call(
            ["  thinking\nreasoning\n  thinking\n\n"
             "  response\n[[ ## result ## ]]\n42\n  response\n"]
        ):
            lm = ThinkingStrippingLM(model="test/model")
            result = lm("hello")

        assert result == ["[[ ## result ## ]]\n42"]
        assert len(lm.last_thinking_traces) == 1
        assert "reasoning" in lm.last_thinking_traces[0]
        assert len(lm.last_raw_outputs) == 1
        assert "  thinking" in lm.last_raw_outputs[0]

    def test_call_no_thinking_passthrough(self) -> None:
        with _mock_lm_init(), _mock_lm_call(["Clean output with no thinking."]):
            lm = ThinkingStrippingLM(model="test/model")
            result = lm("hello")

        assert result == ["Clean output with no thinking."]
        assert lm.last_thinking_traces == []
        assert lm.last_raw_outputs == ["Clean output with no thinking."]

    def test_call_multiple_completions(self) -> None:
        with _mock_lm_init(), _mock_lm_call([
            "  thinking\nt1\n  thinking\n\n  response\n[[ ## a ## ]]\n1\n  response\n",
            "[[ ## a ## ]]\n2\n",
        ]):
            lm = ThinkingStrippingLM(model="test/model")
            result = lm("hello")

        assert result == ["[[ ## a ## ]]\n1", "[[ ## a ## ]]\n2\n"]
        assert len(lm.last_thinking_traces) == 1
        assert "t1" in lm.last_thinking_traces[0]

    def test_responds_to_lm_interface(self) -> None:
        """ThinkingStrippingLM is a dspy.LM subclass."""
        with _mock_lm_init():
            lm = ThinkingStrippingLM(model="test/model")
            # init is mocked — set attributes manually to verify interface
            lm.model = "test/model"
            lm.kwargs = {}
            assert lm.model == "test/model"
            assert lm.kwargs == {}


# -- Helpers ----------------------------------------------------------------


def _mock_lm_init() -> Any:
    """Mock dspy.LM.__init__ to be a no-op."""
    return patch.object(
        ThinkingStrippingLM.__bases__[0], "__init__", lambda self, **kw: None
    )


def _mock_lm_call(return_value: list[str]) -> Any:
    """Mock the parent __call__ to return canned completions."""
    return patch.object(
        ThinkingStrippingLM.__bases__[0],
        "__call__",
        lambda self, prompt=None, messages=None, **kw: return_value,
    )