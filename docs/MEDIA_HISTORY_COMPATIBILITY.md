# Media and Structured History Compatibility

## Downloaded URLs in LangChain Conversations

Some models accept an image URL directly; others need Kedi to download it and
send bytes. LangChain history retains the original source identity in message
metadata, alongside the identity of the lowered block. Reusing the same reference
can then avoid another download and attachment on a later turn.

Reuse requires the matching media block to remain in replayed history. Removing
the block, replacing its bytes, changing its name/source, or starting a different
conversation makes Kedi attach the media again. Message metadata is serialized
with native history and does not enter the model's content or OpenAI payload.

Chat Completions still replays prior messages, including their media. Deduplication
avoids adding another copy to the new message; it is not an upload-once transport
or a promise that the provider bills the image only once.

## Completed Structured Responses

A successful structured response can end with an output tool result rather than
assistant text. Pydantic and LangChain adapters mark that terminal result in the
saved native history. The history classifier can then group the initiating user
prompt with the completed response, allowing `keep_recent_groups(0)` to remove
the old cycle when a new request starts.

Completion is recorded after a successful adapter run. Ordinary tool exchanges
inside a running loop remain distinct, and unfinished calls/results remain
protected. System messages, checkpoints, signed reasoning and provider-required
items retain their existing protections. A completed cycle containing those items
can therefore still be retained. Legacy histories without completion metadata
are treated conservatively; a tool name alone is not proof of a completed run.

Pydantic AI assigns plain Chat Completions reasoning the synthetic IDs `reasoning`,
`reasoning_content` or `content`. Those field names alone do not require replay.
Signed thinking, opaque provider details and other reasoning item IDs remain
protected.

## OpenCode PDF Annotation Limitation

The HY4/OpenCode gateway can return a file annotation with `type: file` and a
`file` object. Pydantic AI's `OpenAIChatModel` validates OpenAI Chat Completions,
whose annotation contract accepts URL citations. The file extension therefore
fails in a direct Pydantic AI call as well as through Kedi.

Pydantic AI's `OpenRouterModel` supports file annotations, but its response
envelope requires a top-level `provider` string. The recorded OpenCode response
omits that field. Selecting `OpenRouterModel` alone does not fix that gateway.
The existing upstream implementation is documented by
[Pydantic AI issue #3657](https://github.com/pydantic/pydantic-ai/issues/3657).

Use the model implementation matching the gateway's complete response dialect.
For this OpenCode path, a gateway correction or a separately justified integration
is still required. Kedi does not discard file annotations, fabricate provider
metadata, or silently retry through another adapter.

Deterministic tests in `test_gateway_file_annotations.py` distinguish the actual
OpenAI-compatible failure from a fully formed OpenRouter response and verify
annotation preservation and rejection of malformed annotations.
