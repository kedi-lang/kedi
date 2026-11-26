# Local vision

Read receipt images with models that run on your machine, then check them with ordinary
Kedi and Python logic. After a one-time download, nothing leaves the machine.

There are two programs:

- **`receipts.kedi`** reads the writing with Cactus Mirage, a 21 MB text reader, and
  parses it.
- **`receipt_record.kedi`** asks a vision-language model for a typed `Receipt`. The same
  program runs on Qwen3-VL-2B locally and on a cloud model; see
  [One program, local and cloud](#one-program-local-and-cloud).

## Reading text with Mirage

[`receipts.kedi`](receipts.kedi) passes the image as a reference,
`[writing] << Read the writing in <<photo>>.`, so the model gets the image itself. It then
parses the lines into items and a total, and checks that the items add up to that total.
The model only reads. Parsing and the check are deterministic, so a misread shows up as a
failed check, not as a plausible wrong number.

[`vision_model.py`](vision_model.py) wraps [Cactus
Mirage](https://huggingface.co/Cactus-Compute/mirage) as a LangChain chat model:

- **Images in.** Its profile declares `image_inputs`, so Kedi's LangChain adapter sends the
  image as an image part.
- **Text out.** Mirage returns each text region with a box, and the wrapper puts the regions
  in reading order, one line per row.
- **No tools or typed output.** The profile also declares `tool_calling: False`, so Kedi
  offers the reader no tools. A typed capture or a program tool fails with a clear error.

| Model | Package | License | Download |
|---|---|---|---|
| Cactus Mirage | `cactus-compute==4.2.7` | Apache 2.0 | 21 MB of weights, plus the Cactus engine |

The six receipts in [`receipts/`](receipts) are synthetic, and
[`make_receipts.py`](make_receipts.py) draws them with the font bundled with Pillow
(Aileron, SIL Open Font License). [`receipts/truth.json`](receipts/truth.json) holds the
expected text, items and total for each one. The bakery receipt's printed total is wrong
on purpose.

### Setup

```bash
uv venv .venv-local && source .venv-local/bin/activate
uv pip install -e ".[langchain]" -r examples/local_vision/requirements.txt
python -c "import os; os.environ['CACTUS_TELEMETRY']='0'; from cactus import Mirage; Mirage()"
```

The last line downloads the engine and the weights once, into `~/.cache/cactus`.

`vision_model.py` sets `CACTUS_TELEMETRY=0` and `DO_NOT_TRACK=1` before Cactus loads.

### Run

```bash
python examples/local_vision/score.py
```

For each receipt, `score.py` reports:

- the character error rate of what Mirage read, against the expected text;
- whether the parsed merchant, items and total match;
- whether the add-up check agrees with the truth.

It writes `examples/local_vision/out/report.json`.

### Results

Measured on 2026-11-26, on an Apple M1 with 8 GB of RAM and Python 3.12. Outbound
networking was denied with `sandbox-exec -p '(version 1)(allow default)(deny network*)'`.

| Measure | Result |
|---|---|
| Mean character error rate | 0.002 |
| Merchant | 6/6 |
| Items | 5/6 |
| Total | 6/6 |
| Add-up check agrees with the truth | 6/6, including the bakery's wrong total |
| Latency per image (median) | 0.15 s |
| Load from cache | 0.1 s; the first load, with its download, took 6.6 s |
| Peak process memory | 267 MB |

The one item error is "Vitamin C", read as "VitaminC": Mirage dropped the space. The sum
still matched, so the check passed.

### What Mirage is, and is not

Mirage is a vision model with several outputs: scene, objects, text, masks, relations, and
form fields (an enum choice, a bounded integer, a short string or a box). Three things we
found matter for this example:

- **Ask for the text section.** Without `sections=["text"]`, Mirage decides what to return
  for each image. On these receipts it sometimes returned only object boxes and no text.
- **Forms do not read receipts.** A form asking for the shop type (an enum), the item count
  (an integer) or the shop name (a string) came back empty or wrong on every receipt. Forms
  are for questions about a photo, such as which sign is shown.
- **Reading is the model's job here; deciding is not.** The 34 MB Needle 3 model from
  [Local extraction](../local_extraction/README.md) was tried on Mirage's text. It named the
  merchant correctly but chose the TOTAL line on only 1 of 4 receipts, so the program
  parses the lines itself.

Mirage reads Latin script only.

## One program, local and cloud

[`receipt_record.kedi`](receipt_record.kedi) has three steps:

1. **Read.** `[receipt: Receipt] << Read the receipt in <<photo>>.` asks for a typed record:
   the merchant, every item with its price, and the printed total.
2. **Check.** `adds_up` checks that the items add up to the total. A receipt that does not
   goes to a person, whether the model misread it or the shop misprinted it.
3. **No artifacts.** The program turns Kedi's artifact tools off with `> artifacts:` and
   `enabled: false`, because nothing in it is large. When the tools were offered,
   Qwen3-VL-2B called `read_artifact` until its retries ran out, on every receipt.

The local model is [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct-GGUF)
(Apache 2.0), served by llama.cpp's `llama-server`, an OpenAI-compatible server. Kedi reaches
it through its Pydantic adapter with Pydantic AI's OpenAI chat model, so the example adds no
model wrapper.

| Model | Runtime | License | Download |
|---|---|---|---|
| Qwen3-VL-2B-Instruct, Q4_K_M, pinned revision | llama.cpp `llama-server` (build 6970) | Apache 2.0 | 1,056 MB model + 424 MB image projector |

### Setup

```bash
uv venv .venv-local && source .venv-local/bin/activate
uv pip install -e .                    # compare.py needs only Kedi
brew install llama.cpp                 # or a release build from github.com/ggml-org/llama.cpp
uvx --from huggingface_hub hf download Qwen/Qwen3-VL-2B-Instruct-GGUF \
    Qwen3VL-2B-Instruct-Q4_K_M.gguf mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf \
    --revision 52d6c8ffea26cc873ac5ad116f8631268d7eb503
```

The files land in `~/.cache/huggingface/hub`. Homebrew installs the current llama.cpp, which
is newer than build 6970, the one measured here. For the cloud comparison, also run
`uv pip install "pydantic-ai-slim[google]"` and set `GOOGLE_API_KEY`.

### Run

```bash
M=~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct-GGUF/snapshots/52d6c8ffea26cc873ac5ad116f8631268d7eb503
llama-server -m $M/Qwen3VL-2B-Instruct-Q4_K_M.gguf --mmproj $M/mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf \
    --alias qwen3-vl-2b --host 127.0.0.1 --port 8092 -c 8192 --jinja
python examples/local_vision/compare.py --local http://127.0.0.1:8092/v1
python examples/local_vision/compare.py --cloud google:gemini-3.5-flash-lite --delay 5
```

`compare.py` reads every receipt 3 times and writes `out/compare_local.json` or
`out/compare_cloud.json`. It reports:

- runtime failures, apart from the model's answers;
- schema validity, on the first attempt and after any retry;
- per-field accuracy and task completion;
- whether the add-up check agrees with the truth;
- consistency across runs and latency.

The cloud run is an explicit comparison, never a fallback.

### Results

Measured on 2026-11-26, on an Apple M1 with 8 GB of RAM. For the local leg, `llama-server`
and `compare.py` both ran with outbound networking denied:

```
sandbox-exec -p '(version 1)(allow default)(deny network*)
                 (allow network* (local ip "localhost:*"))
                 (allow network* (remote ip "localhost:*"))'
```

| Measure | Qwen3-VL-2B, local | `gemini-3.5-flash-lite`, cloud |
|---|---|---|
| Runtime failures | 0/18 | 0/18 |
| Schema-valid answers | 18/18, all on the first attempt | 18/18, 17 on the first attempt |
| Merchant, items, total | 18/18 each | 18/18 each |
| Add-up check agrees with the truth | 18/18, including the bakery's wrong total | 18/18 |
| Identical answers across the 3 runs | 6/6 receipts | 6/6 receipts |
| Latency per receipt (median) | 3.8 s | 3.6 s |
| Model load from cache | 1.7 s | none |
| Server memory | 2.4 GiB maximum resident, 1.5 GiB peak footprint | none |

Qwen3-VL-2B read "Vitamin C" with its space, which Mirage dropped. Mirage is still about 25
times faster (0.15 s) and peaks at 267 MB, so it suits plain text reading where a 2B model
is too heavy.

These are six clean, synthetic receipts. They show that the program, the type and the check
carry over unchanged between backends. They do not show how either model handles photos of
real receipts.

### Two receipts: the same purchase?

[`same_purchase.kedi`](same_purchase.kedi) compares two receipt images:

- **The question.** `[match: Match] << Do the receipts <<first>> and <<second>> show the
  same shop and the same total?` asks for two booleans.
- **The decision.** The program calls a pair a duplicate only when both are true.
- **The same receipt twice.** Three of the six pairs in [`pairs.py`](pairs.py) give one
  receipt as both images. Kedi attaches the same content once, and the prompt names it
  `first` twice. The model must recognize one image under both names; a test checks
  exactly what reaches the model in both cases.

```bash
python examples/local_vision/pairs.py --local http://127.0.0.1:8092/v1
python examples/local_vision/pairs.py --cloud google:gemini-3.5-flash-lite --delay 5
```

| Measure, 3 runs of 6 pairs | Qwen3-VL-2B, local | `gemini-3.5-flash-lite`, cloud |
|---|---|---|
| Runtime failures | 0/18 | 0/18 |
| Schema-valid answers | 18/18 | 18/18 |
| Same shop, same total | 18/18 each | 18/18 each |
| Identical answers across the 3 runs | 6/6 pairs | 6/6 pairs |
| Latency per pair (median) | 2.3 s | 1.6 s |

The local leg ran under the same network-denied sandbox as above. As with single receipts,
these clean synthetic images show that the program carries over, not how the models handle
photographs.

### Cactus Engine

We tried Cactus Engine first, the Cactus runtime for vision-language models, and it did
not work here:

- **It cannot load the published models.** `cactus-engine` 2.2.2, the only release, rejects
  the vision-language bundles now published at huggingface.co/Cactus-Compute as "not a
  valid v2 Cactus bundle". That covers both the `-cq` archives and the older `int4`
  archives.
- **It sends requests to the cloud by default.** `cactus serve` hands low-confidence
  completions to Cactus's cloud unless it gets `--no-cloud-handoff`
  (`CACTUS_DISABLE_CLOUD_HANDOFF=1`). It also sends telemetry unless it gets
  `--no-cloud-tele` (`CACTUS_NO_CLOUD_TELE=1`). An offline setup with it needs both.
- **It conflicts with `cactus-compute`.** Both packages install the `cactus` module, so they
  need separate environments.

## Files

| File | Purpose |
|---|---|
| `receipts.kedi` | the Mirage program: read, parse, check |
| `vision_model.py` | Mirage as a LangChain chat model that reads text |
| `score.py` | score the Mirage program against the ground truth |
| `receipt_record.kedi` | the shared program: a typed `Receipt` from a vision-language model, then the check |
| `compare.py` | run the shared program locally or on a cloud model, and score it |
| `same_purchase.kedi` | compare two receipt images, then decide if they are a duplicate |
| `pairs.py` | run the comparison on six pairs, locally or on a cloud model |
| `make_receipts.py` | draw the receipts and the ground truth |
| `receipts/` | the images and `truth.json` |
| `requirements.txt` | `cactus-compute`, pinned, for Mirage |
