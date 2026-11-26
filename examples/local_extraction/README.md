# Local extraction

Read refund requests into typed records with a 34 MB model that runs on your machine, then
route them with ordinary Kedi and Python logic. After a one-time download, nothing leaves the
machine.

[`refunds.kedi`](refunds.kedi) has three parts:

- **A `~Refund` type.** Each field is described, and the reason is one of four values.
- **`read_request`.** A typed raw invoke, `[refund: Refund] << Refund request: <message>`.
- **`route`.** It approves small refunds for damaged or wrong items automatically and sends
  the rest to a person. It also sends a request to a person when the extracted amount does
  not appear in the message.

The model only reads; the program decides.

[`needle_model.py`](needle_model.py) wraps
[Cactus Needle 3](https://huggingface.co/Cactus-Compute/needle3) as a Pydantic AI model:

- **What Needle gets.** Kedi asks for the `Refund` through an output tool. The wrapper hands
  that tool's schema to Needle's extraction, with Kedi's prompt and system text unchanged,
  and answers with the tool call.
- **Validation.** Kedi and Pydantic AI validate the result, so Cactus's own repair is off.
- **What Needle does not do.** It answers only with structured output: it does not call
  Kedi's other tools, and an untyped `<<` prompt fails with a clear error.

| Model | Package | License | Download |
|---|---|---|---|
| Cactus Needle 3 | `cactus-compute==4.2.7` | Apache 2.0 | 34 MB of weights, plus the Cactus engine |

## Setup

```bash
uv venv .venv-local && source .venv-local/bin/activate
uv pip install -e . -r examples/local_extraction/requirements.txt
python -c "import os; os.environ['CACTUS_TELEMETRY']='0'; from cactus import Needle; Needle()"
```

The last line downloads the engine and the weights once, into `~/.cache/cactus`.

`needle_model.py` sets `CACTUS_TELEMETRY=0` and `DO_NOT_TRACK=1` before Cactus loads.
Without them, Cactus sends a usage event to its servers on each extraction.

## Run

```bash
python examples/local_extraction/run.py              # Needle 3, offline, 3 runs over 8 requests
python examples/local_extraction/run.py --cloud google:gemini-3.8-flash
```

The second command runs the same program, inputs and scoring on a cloud model. It is a
separate, explicit comparison, never a fallback.

Results go to `examples/local_extraction/out/local.json`, or `cloud.json` for the cloud run.

[`requests.jsonl`](requests.jsonl) holds the fixed evaluation set:

- eight requests with explicit expected fields and routes;
- varied wording and name positions;
- currency written as `€34.50`, `1,299.00 euros` and `120 euros`;
- numbers that are not the amount, such as dates, sizes and item counts.

## Results

Measured on 2026-11-26, on an Apple M1 with 8 GB of RAM and Python 3.12, over 3 runs of the
8 requests. Outbound networking was denied with
`sandbox-exec -p '(version 1)(allow default)(deny network*)'`.

| Measure | Result |
|---|---|
| Schema-valid answers | 24/24 |
| Field accuracy | order id 24/24, customer 21/24, amount 21/24, reason 15/24 |
| Task completion (every field right) | 12/24 |
| Correct route | 18/24 |
| Identical answers across the 3 runs | 8/8 requests |
| Warm latency (median per request) | 0.16 s |
| Load from cache | 0.1 s; the first load, with its download, took 10.6 s |
| Peak process memory | 195 MB |

The errors, which repeat on every run:

- **"1,299.00 euros" read as 1.299.** The route check catches this: 1.299 does not appear
  in the message, so the request goes to a person instead of being approved.
- **A name at the end of a message.** "... I paid 120 euros. Lena Fischer" gives "Fischer".
- **Reasons.** "you sent a red toaster" reads as damaged instead of wrong_item, and "I changed
  my mind" reads as damaged instead of other. The second one is approved automatically.
  The amount check cannot catch a wrong reason, so a reason this model reads needs a person
  or a stronger model before it triggers an action.

**How the program shape changed the scores.** These numbers come from an iteration on the
program, and the earlier shapes did worse:

| Program | Field accuracy (first run) | Route |
|---|---|---|
| A `>>` template capture, with Kedi's template instructions | 16/32 | 3/8 |
| A typed raw invoke | 25/32 | 5/8 |
| The same, with a description for each reason and the amount check | 27/32 | 6/8 |

The template instructions are written for capable models, and they confuse a model this
small. A typed raw invoke sends just the message. Describing each reason value also helped.
The wrapper still sends Kedi's text unchanged: the program, not the transport, was changed.

**Cloud comparison.** The cloud leg could not be measured on 2026-11-26. The free-tier Gemini
quota answered 429 after one request.

## Files

| File | Purpose |
|---|---|
| `refunds.kedi` | the program: type, extraction, routing |
| `needle_model.py` | Needle 3 as a Pydantic AI model |
| `run.py` | score the fixed set, locally or explicitly on a cloud model |
| `requests.jsonl` | the fixed evaluation set |
| `requirements.txt` | `cactus-compute`, pinned |
