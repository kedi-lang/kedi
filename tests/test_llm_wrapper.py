"""Tests for ThinkingStrippingLM and _strip_thinking."""

from __future__ import annotations

from typing import Any

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

    def test_xml_thinking_preserves_output_without_response_wrapper(self) -> None:
        # No ``  response`` wrapper — everything after ``  thinking`` is output
        raw = "  thinking\nreasoning\n  thinking\n\nplain output here"
        result = _strip_thinking(raw)
        assert "reasoning" in result.thinking
        assert "plain output here" in result.output

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
        # The thinking block markers should be in the thinking trace
        assert "[[ ## capital ## ]]" in result.thinking
        assert "[[ ## completed ## ]]" in result.thinking
        # The real output should be clean
        assert "[[ ## capital ## ]]" in result.output
        assert "Stockholm" in result.output
        assert "[[ ## completed ## ]]" in result.output
        # The real output should NOT contain "Thinking process"
        assert "Thinking process" not in result.output
        assert "reasoning" not in result.output.lower()

    # -- Field-marker fallback (when no XML thinking tags present) --------

    def test_no_marker_returns_as_is(self) -> None:
        result = _strip_thinking("Stockholm is the capital.")
        assert result.thinking == ""
        assert result.output == "Stockholm is the capital."

    def test_field_marker_fallback_skips_completed(self) -> None:
        """When no XML tags, use last non-completed marker as split point."""
        raw = (
            "garbage preamble\n"
            "[[ ## capital ## ]]\n"  # fake in thinking
            "more garbage\n"
            "[[ ## completed ## ]]\n"  # also fake
            "even more\n"
            "[[ ## capital ## ]]\n"  # real first field
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
    def test_delegates_attributes_to_wrapped(self) -> None:
        class FakeLM:
            model = "test/model"
            kwargs: dict[str, Any] = {"max_tokens": 100}

        wrapper: Any = ThinkingStrippingLM(FakeLM())
        assert wrapper.model == "test/model"
        assert wrapper.kwargs == {"max_tokens": 100}

    def test_call_strips_thinking(self) -> None:
        class FakeLM:
            model = "test/model"
            kwargs: dict[str, Any] = {}

            def __call__(self, prompt: Any, **kw: Any) -> list[str]:
                return [
                    "  thinking\nreasoning\n  thinking\n\n"
                    "  response\n"
                    "[[ ## result ## ]]\n42\n"
                    "  response\n"
                ]

        wrapper = ThinkingStrippingLM(FakeLM())
        result = wrapper("hello")

        assert result == ["[[ ## result ## ]]\n42"]
        assert len(wrapper.last_thinking_traces) == 1
        assert "reasoning" in wrapper.last_thinking_traces[0]
        assert len(wrapper.last_raw_outputs) == 1
        assert "  thinking" in wrapper.last_raw_outputs[0]

    def test_call_no_thinking_passthrough(self) -> None:
        class FakeLM:
            model = "test/model"
            kwargs: dict[str, Any] = {}

            def __call__(self, prompt: Any, **kw: Any) -> list[str]:
                return ["Clean output with no thinking."]

        wrapper = ThinkingStrippingLM(FakeLM())
        result = wrapper("hello")

        assert result == ["Clean output with no thinking."]
        assert wrapper.last_thinking_traces == []
        assert wrapper.last_raw_outputs == ["Clean output with no thinking."]

    def test_call_multiple_completions(self) -> None:
        class FakeLM:
            model = "test/model"
            kwargs: dict[str, Any] = {}

            def __call__(self, prompt: Any, **kw: Any) -> list[str]:
                return [
                    "  thinking\nt1\n  thinking\n\n  response\n[[ ## a ## ]]\n1\n  response\n",
                    "[[ ## a ## ]]\n2\n",  # no thinking
                ]

        wrapper = ThinkingStrippingLM(FakeLM())
        result = wrapper("hello")

        assert result == ["[[ ## a ## ]]\n1", "[[ ## a ## ]]\n2\n"]
        assert len(wrapper.last_thinking_traces) == 1
        assert "t1" in wrapper.last_thinking_traces[0]