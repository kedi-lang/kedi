# Media Input

Images, audio, video and documents can be substituted into model prompts. A
media value reaches the model as an attachment, and the prompt text refers to
it by name.

```kedi
> import: media

[cat_image] = <image(photos/cat.png)>
[dog_image] = <image(photos/dog.png)>

>> Compare <cat_image> and <dog_image>. Is <cat_image> the happier one? [answer: bool]
>> Extract the total of the receipt in <image(photos/receipt.png)> as [total: float].
```

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

| Kind | Formats | Reaches the model as |
| --- | --- | --- |
| `Image` | PNG, JPEG, GIF, WebP | label and image |
| `Audio` | WAV, MP3, OGG, FLAC, AIFF, AAC | label and audio |
| `Video` | MP4, MOV, WebM, MKV, FLV, MPEG, WMV, 3GP; YouTube URLs | label and video |
| `Document` | PDF | label and document |
| `Document` | UTF-8 text, Markdown, CSV, HTML, JSON, XML, YAML, TOML | delimited text |

Office files (docx, xlsx, pptx) are rejected: convert them to PDF or text.

## What the Model Receives

Every media value in a prompt becomes an attachment. Attachments come before
the prompt text, each introduced by a `name:` label, and the text shows only
the name. The first template above sends one user message:

1. the text `cat_image:`, then the image;
2. the text `dog_image:`, then the image;
3. the template text, with `cat_image` and `dog_image` in place of the
   substitutions.

This is the structure Anthropic recommends for images, and it keeps the prompt
text a single string for hooks, telemetry and prefix caching. The same rules
apply to `>>` templates, `<<` raw invokes and typed captures, and `> if:`
template conditions.

The name is, in order:

1. a name given in Python: `Image.from_path(path, name="cat")`;
2. the binding name; a `list[Image]` binding `photos` gives `photos_1`,
   `photos_2`, ...;
3. `image_1`, `audio_1`, ... for an unnamed value such as
   `<image(photos/receipt.png)>`.

File names never reach the model, because a name such as
`invoice_1250USD.png` can leak an answer. Within one prompt each content is
attached once, and a second name for the same content renders the first name.

A text document reaches every model as text, in the format Pydantic AI uses for
providers without text-file input:

```text
-----BEGIN FILE id="notes" type="text/markdown"-----
...
-----END FILE id="notes"-----
```

With [artifacts](../runtime/artifact-policy.md) enabled, a text document at or above
the artifact threshold is stored as an artifact instead, and the model reads it
with the artifact tools.

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

URLs go to the provider as URLs when the model accepts them. Otherwise Kedi
downloads them over `http(s)` only, with private and cloud-metadata addresses
blocked. YouTube URLs are never downloaded, and only Gemini reads them.
`s3://` and `gs://` URIs pass through unchanged. Do not build media URLs from
untrusted input.

## Media Outside Prompts

A media value is a native value. These keep it:

- a sole loader call: `[cat_image] = <image(photos/cat.png)>`;
- backticks: ``[copy: Image] = `cat_image` ``;
- parameters typed `Image`, `Audio`, `Video`, `Document` or `Media`;
- backtick call arguments: ``<caption(`cat_image`)>``.

```kedi
> import: media

@caption(photo: Image) -> str:
  [text] << Describe <photo> in one sentence.
  = <text>

[invoice: Image] = <image(`args.invoice`)>
= <caption(`invoice`)>
```

A parameter typed `Image` also accepts a path, URL or data URI string and loads
it with the same rules.

Rendering media into text is an error, because the content would be lost.
These fail, and the editor reports them as `K-MEDIA-TEXT`:

- text assignments and returns: `[m] = See <cat_image>`, `[m] = <cat_image>`,
  `= <cat_image>`;
- string arguments: `<caption(<cat_image>)>`;
- `> instructions:`;
- subagent tasks and agent input.

`> show:` and Python `str()` display a byte-free label such as
`<image cat.png · image/png · 48 KB>`. Media inside a dict, a model or a mixed
list cannot be used in a prompt yet; substitute each media value directly.

## History

With `> history: enabled`, an attachment that the replayed history already
holds under the same name and content is not sent again; the prompt text still
names it. When a name now holds different content, for example a loop that
rebinds `photo`, the new content is attached again.

With a history processor, native compaction or history archiving, Kedi cannot
tell what the model still sees, so it attaches media on every turn.

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

Prompt substitutions also accept Pydantic AI `BinaryContent`, `ImageUrl`,
`AudioUrl`, `VideoUrl`, `DocumentUrl` and `UploadedFile` objects and PIL
images.

[Hooks](../agentic-engineering/hooks.md) see the attachments read-only in
`UserPromptSubmitRequest.media`: name, kind, media type, size and SHA-256. An
`edit` decision replaces only the prompt text. Telemetry and debugger events
record the same metadata and never the bytes.
