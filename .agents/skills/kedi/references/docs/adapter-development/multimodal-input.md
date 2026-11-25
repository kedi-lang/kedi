# Multimodal Input

A prompt can carry images, audio, video, and documents. The program
references a value with `<<name>>`; the prompt text gets the name, and the
value travels beside the text as an attachment. Kedi resolves, checks, and
names the attachments. The adapter places each one in the user message in its
backend's native form.

## How Attachments Arrive

Attachments never arrive as keyword arguments; the entry points keep their
`prompt: str` signatures. Take them from the active request with
`kedi.agent_adapter.request.take_prompt_media()`:

```python
from kedi.agent_adapter.request import take_prompt_media

attachments = take_prompt_media()
```

The first native run inside a Kedi call takes the attachments; runs nested in
it, such as a tool's own model call or a child, get an empty tuple. That keeps
one image from being sent twice. The same attachments are also readable as
`current_agent_run_request().prompt.media`.

Each item is a `MediaAttachment`:

| Field | Content |
|---|---|
| `name` | the name the prompt text uses |
| `media` | the `Media` value: `kind` (`image`, `audio`, `video`, `document`), `media_type`, and one of `data` (bytes), `url`, or `file_id`; `size`; `text` for text documents |
| `artifact_ref` | set when the value is already an artifact and the model can read artifacts: send the reference instead of the content |
| `is_text`, `inline_text()` | text documents and artifact references travel as text elements |
| `summary` | a byte-free description for logs and hooks |

## Laying Out the Message

All attachments go in the same user message as the prompt, before the text,
each labeled with its name. `kedi.agent_adapter.request.media.attachment_content(prompt,
attachments)` produces that layout as a list of text pieces and the binary
attachments in their places:

```text
<attachments>
<image id="chart" type="image/png">
  [the image itself, as the backend's image part]
</image>
</attachments>
Describe chart in one sentence.
```

Turn each `str` piece into a text part and each `MediaAttachment` into the
backend's native part for its kind: inline bytes (`data`), a URL, or a
provider file id. Text documents and artifact references are already inlined
in the text pieces.

## Checks

Kedi checks before the call; the program sees each failure as a
`KediExecutionError` whose `original` is the `MediaError`:

| Condition | Error |
|---|---|
| a prompt has attachments and the adapter does not advertise `supports_media_input` | `MediaError: Adapter '<shortname>' does not accept media input; Pydantic AI, LangChain, the Claude Agent SDK, Codex, ACP and A2A take prompts with media` |
| an attachment is larger than the `media_limit` setting | `MediaError: <name>: ... is larger than the media_limit setting` |
| a known provider cannot take the attachment's kind, type, or source | `MediaError` naming the attachment |

The adapter adds the checks Kedi cannot make: refuse, with `MediaError` from
`kedi.media`, a kind or source the backend cannot send, before the request.
Never drop an attachment silently or describe it in text instead.

In a conversation, history replay must not resend an attachment the backend
already holds; send each attachment with the turn that introduced it.

## Running It

```python title="multimodal_adapter.py"
import base64
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from kedi.agent_adapter.request import take_prompt_media
from kedi.agent_adapter.request.media import attachment_content
from kedi.media import MediaError

from backend import Reply
from chat_adapter import ChatAdapter


class MultimodalAdapter(ChatAdapter):
    capabilities = replace(ChatAdapter.capabilities, supports_media_input=True)

    async def _run(
        self,
        *,
        prompt: str,
        instructions: str | None,
        output_schema: dict[str, Any] | None,
        history: Sequence[dict[str, Any]] = (),
        max_requests: int | None = None,
    ) -> tuple[Reply, list[dict[str, Any]]]:
        attachments = take_prompt_media()
        if not attachments:
            return await super()._run(
                prompt=prompt,
                instructions=instructions,
                output_schema=output_schema,
                history=history,
                max_requests=max_requests,
            )
        parts: list[dict[str, Any]] = []
        for piece in attachment_content(prompt, attachments):
            if isinstance(piece, str):
                parts.append({"type": "text", "text": piece})
                continue
            media = piece.media
            if media.kind != "image" or media.data is None:
                raise MediaError(f"{piece.name}: {self.shortname} sends only inline images")
            data = base64.b64encode(media.data).decode("ascii")
            parts.append({"type": "image", "media_type": media.media_type, "data": data})
        return await super()._run(
            prompt=parts,  # type: ignore[arg-type]  # a structured user message
            instructions=instructions,
            output_schema=output_schema,
            history=history,
            max_requests=max_requests,
        )
```

```python title="run_multimodal_input.py"
import asyncio

from kedi.lang import compile_program, parse_program
from kedi.media import MediaError

from backend import BackendRequest, Reply
from chat_adapter import ChatAdapter
from multimodal_adapter import MultimodalAdapter

PIXEL = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
    "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)
SOURCE = f"""
> import: media

[chart: Image] = `image("{PIXEL}")`
[answer] << Describe <<chart>> in one sentence.
= <answer>
"""


class Recorder:
    def __init__(self) -> None:
        self.requests: list[BackendRequest] = []

    async def complete(self, request: BackendRequest) -> Reply:
        self.requests.append(request)
        return Reply(text="A single pixel.")


def run(adapter: ChatAdapter) -> object:
    runtime = compile_program(parse_program(SOURCE), adapter=adapter)
    try:
        return runtime.run_main()
    finally:
        asyncio.run(runtime.aclose())


backend = Recorder()
assert run(MultimodalAdapter(backend, model="m1")) == "A single pixel."
parts = backend.requests[0].messages[0]["content"]
kinds = [part["type"] for part in parts]
assert kinds == ["text", "image", "text"], kinds
assert parts[0]["text"].startswith('<attachments>\n<image id="chart" type="image/png">')
assert parts[1]["media_type"] == "image/png"
assert parts[2]["text"].endswith("Describe chart in one sentence.")

try:
    run(ChatAdapter(Recorder(), model="m1"))
except Exception as error:
    assert isinstance(error.original, MediaError), error
    assert "does not accept media input" in str(error)
else:
    raise AssertionError("an adapter without supports_media_input must be refused")
```

The image reached the backend as its own part, labeled by name, before the
prompt text; the adapter without the capability was refused before any
request.
