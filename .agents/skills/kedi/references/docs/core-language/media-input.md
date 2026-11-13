# Media Input

Images, audio, video and documents reach a model by reference. A prompt writes
`<<name>>`: the text gets the name, and the value is attached to the model's
context under it.

```kedi
> import: media

[cat_image: Image] = `image("photos/cat.png")`
[dog_image: Image] = `image("photos/dog.png")`

>> Compare <<cat_image>> and <<dog_image>>. Is <<cat_image>> the happier one? [answer: bool]
>> Extract the total of the receipt in <<image(photos/receipt.png)>> as [total: float].
```

`<cat_image>` would be a type error: `<...>` always renders text, and an image
has no text form. See [Substitutions and Calls](substitutions-and-calls.md#references)
for the rules every reference follows.

## Loading Media

`> import: media` exports four loaders and five types:

| Name | Kind |
| --- | --- |
| `image(source)` | loads an `Image` |
| `audio(source)` | loads an `Audio` value |
| `video(source)` | loads a `Video` |
| `document(source)` | loads a `Document` |
| `Media`, `Image`, `Audio`, `Video`, `Document` | types for annotations |

A loader takes a path relative to the working directory, an `http(s)` URL or
a `data:` URI. Paths follow the [`filesystem`](../modules-and-packaging/filesystem.md) module's
rules: with `KEDI_WORKSPACE_POLICY=strict`, the default, a path cannot leave
the working directory, and `.env` files are refused. A media value is limited
to 20 MB.

Loading is always explicit. Keep a loaded value with single-backtick Python,
``[photo: Image] = `image("cat.png")` ``, or reference a loader call directly
in a prompt, `<<image(cat.png)>>`. A binding, parameter or capture typed
`Image` accepts media values only: a string is never read as a path or URL, so
`[photo: Image] = cat.png` and `<caption(cat.png)>` for a parameter typed
`Image` are type errors.

| Kind | Formats | Reaches the model as |
| --- | --- | --- |
| `Image` | PNG, JPEG, GIF, WebP | `<image>` with the image |
| `Audio` | WAV, MP3, OGG, FLAC, AIFF, AAC | `<audio>` with the audio |
| `Video` | MP4, MOV, WebM, MKV, FLV, MPEG, WMV, 3GP; YouTube URLs | `<video>` with the video |
| `Document` | PDF | `<document>` with the PDF |
| `Document` | UTF-8 text, Markdown, CSV, HTML, JSON, XML, YAML, TOML | `<document>` with the text |

Office files (docx, xlsx, pptx) are rejected: convert them to PDF or text.

## What the Model Receives

Every referenced value becomes an attachment. The attachments come first,
inside one `<attachments>` element, each tagged with its ID and media type; the
prompt text follows and shows only the IDs. The first template above sends one
user message:

```text
<attachments>
<image id="cat_image" type="image/png">
[the image]
</image>
<image id="dog_image" type="image/png">
[the image]
</image>
</attachments>
... Compare cat_image and dog_image. Is cat_image the happier one? [answer]
```

Coding agents and providers use the same pattern: Codex tags each image with
its name, and Anthropic recommends XML-tagged documents placed before the
question. The prompt text stays a single string for hooks, telemetry and
prefix caching. The same rules apply to `>>` templates, `<<` raw invokes and
typed captures, and `> if:` template conditions.

The ID is, in order:

1. a name given in Python: `Image.from_path(path, name="cat")`;
2. the binding name; a `list[Image]` binding `photos` gives `photos_1`,
   `photos_2`, ...;
3. `<proc>_result_1`, `<proc>_result_2`, ... for a call such as
   `<<image(photos/receipt.png)>>`.

File names never reach the model, because a name such as
`invoice_1250USD.png` can leak an answer. Within one prompt each content is
attached once, and a second name for the same content renders the first ID.

A text document reaches every model as text inside its element:

```text
<document id="notes" type="text/markdown">
...
</document>
```

The content is not escaped. A referenced document is attached in full whatever
its size: the [artifact](../runtime/artifact-policy.md) threshold never turns
referenced content into an artifact.

## Provider Support

With the Pydantic AI and LangChain adapters:

| Provider | Image | Text document | PDF | Audio | Video |
| --- | --- | --- | --- | --- | --- |
| OpenAI | yes | yes | yes | audio models; WAV and MP3 | no |
| Anthropic | yes | yes | yes | no | no |
| Google Gemini | yes | yes | yes | yes | yes, including YouTube URLs |
| AWS Bedrock | yes | yes | yes | no | Nova models |

Kedi fails before the model call when a known provider cannot take an
attachment, for example video on OpenAI or an OGG file on OpenAI. With
LangChain it also honors the model profile flags `image_inputs`, `pdf_inputs`,
`audio_inputs` and `video_inputs`. Kedi never transcribes or describes media
for a model that cannot read it.

The other adapters (Claude Agent SDK, Codex, WebGPU, ACP and A2A) do not
accept media: a prompt with media fails before model I/O, and the editor reports
it statically. See the [capability matrix](../reference/capability-matrix.md).

Decision models read less. Jev (`typesafe/...`) reads images and text documents:
images reach TypeSafe as image assets, and Kedi downloads image URLs first.
TypeSafe image input is experimental and takes PNG or JPEG images of 16 MiB at
most, each side 280 to 1568 pixels ([TypeSafe image limits](https://docs.typesafe.ai/images));
Kedi checks these before the call. Laya reads text documents only. Kedi rejects any other attachment for them before the
call.

URLs go to the provider as URLs when the model accepts them. Otherwise Kedi
downloads them over `http(s)` only, with private and cloud-metadata addresses
blocked. YouTube URLs are never downloaded, and only Gemini reads them.
`s3://` and `gs://` URIs pass through unchanged. Do not build media URLs from
untrusted input.

## Media Outside Prompts

A media value is a native value. Single backticks keep it:

- bindings: ``[copy: Image] = `cat_image` ``;
- parameters typed `Image`, `Audio`, `Video`, `Document` or `Media`;
- call arguments: ``<caption(`cat_image`)>``.

```kedi
> import: media

@caption(photo: Image) -> str:
  [text] << Describe <<photo>> in one sentence.
  = <text>

[invoice: Image] = `image(args.invoice)`
= <caption(`invoice`)>
```

`<cat_image>` anywhere is a type error, and the editor reports it as
`K-SUBSTITUTION-TYPE`. `<<cat_image>>` outside a model prompt, for example in
an assignment, a return, `> show:`, `> instructions:` or a subagent task, is a
parse error (`K-REFERENCE-CONTEXT`). ``> show: `cat_image` `` and Python
`str()` display a byte-free label such as `<image cat.png · image/png · 48 KB>`.
Media inside a dict, a model or a mixed list cannot be referenced; reference
each media value directly.

## History

With `> history: enabled`, an attachment that the replayed history already
holds under the same ID and content is not sent again; the prompt text still
writes the ID. When an ID now holds different content, for example a loop that
rebinds `photo`, the new content is attached again.

With a history processor, native compaction or history archiving, Kedi cannot
tell what the model still sees, so it attaches references on every turn.

## Python

`kedi.media` provides the same types:

```python
from kedi.media import Document, Image, Media

Image.from_path("photos/cat.png")
Image.from_url("https://example.com/cat.png")
Image.from_bytes(data, "image/png")
Document.from_path("notes.txt", encoding="cp1254")  # re-encoded to UTF-8
Media.from_path("clip.mp4")  # picks Video from the media type
```

References also accept Pydantic AI `BinaryContent`, `ImageUrl`, `AudioUrl`,
`VideoUrl`, `DocumentUrl` and `UploadedFile` objects and PIL images.

[Hooks](../agentic-engineering/hooks.md) see the attachments read-only in
`UserPromptSubmitRequest.media`: name, kind, media type, size and SHA-256. An
`edit` decision replaces only the prompt text. Telemetry and debugger events
record the same metadata and never the bytes.
