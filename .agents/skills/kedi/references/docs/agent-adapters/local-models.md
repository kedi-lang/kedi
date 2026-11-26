# Local Models

Kedi programs can run on small models on your own machine, offline, with no API
key. Kedi needs no special support for this. A local model joins through an
adapter Kedi already has:

- **A LangChain chat model.** Its profile declares what it reads and produces:
  `image_inputs`, `audio_inputs` or `audio_outputs`.
- **A Pydantic AI model.** It receives Kedi's typed output as an output tool,
  separate from Kedi's own tools.
- **A [JSON-RPC bridge](../adapter-development/json-rpc-bridges.md).** Use this
  for a runtime outside Python.

The program is the same as for a cloud model. Only the adapter's model changes.

## Tested Setups

Each row was measured end to end through Kedi on 2026-11-26, on an Apple M1
with 8 GB of RAM and Python 3.12, with outbound networking denied. Every
example has a README with setup, commands, evaluation sets and results.

| Task | Model | Kedi surface | Joins as | Download | Measured |
| --- | --- | --- | --- | --- | --- |
| Typed fields from text | Cactus Needle 3 | `[refund: Refund] << ...` | Pydantic AI model | 34 MB | schema-valid 24/24, every field right 12/24, 0.16 s, 195 MB peak |
| Reading text in images | Cactus Mirage | `[writing] << Read the writing in <<photo>>.` | LangChain, `image_inputs` | 21 MB | character error rate 0.002, 0.15 s, 267 MB peak |
| Turkish speech | EMA Lightning | `[speech: Audio] << <sentence>` | LangChain, `audio_outputs` | 34 MB, plus PyTorch | real-time factor 0.064, 481 MB peak |
| English speech | Phonon 31m | `[speech: Audio] << <sentence>` | LangChain, `audio_outputs` | 65 MB | real-time factor 0.081, 347 MB peak |
| English transcription | Cactus Whistle | `[text] << Transcribe <<clip>>.` | LangChain, `audio_inputs` | 16 MB | word error rate 0.067 on Phonon's speech, 0.06–0.11 s per clip |
| Turkish transcription | Whisper small | `[text] << Transcribe <<clip>>.` | LangChain, `audio_inputs` | 484 MB | 2 real errors in 43 words on EMA's speech, real-time factor about 0.3 |

The examples are
[local_extraction](https://github.com/kedi-lang/kedi/tree/stable/examples/local_extraction),
[local_vision](https://github.com/kedi-lang/kedi/tree/stable/examples/local_vision)
and [local_speech](https://github.com/kedi-lang/kedi/tree/stable/examples/local_speech).

Typed decisions are covered separately. For booleans, choices, probabilities,
rubrics and constrained extraction, with decision evidence, use
[Local Laya](laya.md). The setups above cover what Laya does not: open fields
such as names and amounts, text in images, and speech.

## What These Models Do Not Do

Each wrapper does only what its model does and refuses the rest with a clear
error:

- **Needle 3** answers only with structured output. It does not call Kedi's
  tools, and an untyped `<<` prompt fails with a clear error.
- **Mirage** is used only to read text, and it reads Latin script only. Its
  form fields did not read receipts.
- **The speech models and transcribers** take no tools. EMA Lightning has a
  single voice, so a `voice` setting for it is an error.

## Writing for a Small Model

Small models are sensitive to the program's shape. These rules come from the
measurements:

- **Use a typed raw invoke.** `[x: T] << message` sends the message alone.
  A `>>` template capture adds template instructions written for capable
  models. On the refund set, the raw invoke scored 25/32 fields and the
  template capture 16/32.
- **Describe the allowed values.** A `Literal` whose description explains each
  value raised the reason field from 10/24 to 15/24.
- **Let the program decide.** The model reads; Kedi or Python code checks and
  routes. In the examples, a refund goes to a person when its amount does not
  appear in the message, and a receipt fails when its items do not add up.
  These checks catch misreads such as "1,299.00" read as 1.299.
- **Measure before relying on it.** A consistent answer can still be wrong:
  Needle gave the same wrong reason on every run.

## Offline Runs

- **Download once, then run offline.** Provisioning needs the network; running
  does not. Set `HF_HUB_OFFLINE=1` for models from Hugging Face.
- **Turn off Cactus telemetry.** Cactus sends a usage event on each call unless
  `CACTUS_TELEMETRY=0` or `DO_NOT_TRACK` is set. The example wrappers set both
  before Cactus loads.
- **Check it with the network denied.** On macOS:
  `sandbox-exec -p '(version 1)(allow default)(deny network*)' python ...`.
- **Keep model packages out of Kedi.** The model packages are not Kedi
  dependencies. Each example pins its own in `requirements.txt`, for a separate
  environment.
