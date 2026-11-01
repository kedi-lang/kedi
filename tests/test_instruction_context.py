from types import SimpleNamespace

from pydantic_ai.messages import ModelRequest, UserPromptPart

from tests.instruction_context import context_blocks, effective_instructions


def _request(text: str) -> ModelRequest:
    return ModelRequest(parts=[UserPromptPart(text)])


def test_older_releases_have_only_the_instructions():
    info = SimpleNamespace(instructions="Base guidance.")
    assert effective_instructions(info, [_request("hello")]) == "Base guidance."


def test_newest_full_block_replaces_the_previous_one():
    messages = [
        _request('<system><context id="a">\nfirst\n</context></system>'),
        _request('<system><context id="a">\nsecond\n</context></system>'),
    ]
    assert context_blocks(messages) == {"a": "second"}


def test_diffs_add_and_remove_lines_of_the_previous_block():
    messages = [
        _request('<system><context id="catalog">\n- one\n- two\n- three\n</context></system>'),
        _request(
            '<system><context_diff id="catalog">\n--- before\n+++ after\n@@ -1,3 +1,3 @@\n'
            " - one\n- - two\n+- four\n - three\n</context_diff></system>"
        ),
    ]
    assert context_blocks(messages) == {"catalog": "- one\n- four\n- three"}


def test_withdrawn_blocks_are_no_longer_in_effect():
    messages = [
        _request('<system><context id="a">\nguidance\n</context></system>'),
        _request('<system><context id="a">\nThis context has been withdrawn.\n</context></system>'),
    ]
    assert context_blocks(messages) == {}
