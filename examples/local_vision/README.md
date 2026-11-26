# Local vision

Read receipt images with a 21 MB vision model that runs on your machine, then check them
with ordinary Kedi and Python logic. After a one-time download, nothing leaves the machine.

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

## Setup

```bash
uv venv .venv-local && source .venv-local/bin/activate
uv pip install -e ".[langchain]" -r examples/local_vision/requirements.txt
python -c "import os; os.environ['CACTUS_TELEMETRY']='0'; from cactus import Mirage; Mirage()"
```

The last line downloads the engine and the weights once, into `~/.cache/cactus`.

`vision_model.py` sets `CACTUS_TELEMETRY=0` and `DO_NOT_TRACK=1` before Cactus loads.

## Run

```bash
python examples/local_vision/score.py
```

For each receipt, `score.py` reports:

- the character error rate of what Mirage read, against the expected text;
- whether the parsed merchant, items and total match;
- whether the add-up check agrees with the truth.

It writes `examples/local_vision/out/report.json`.

## Results

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

## What Mirage is, and is not

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

## Files

| File | Purpose |
|---|---|
| `receipts.kedi` | the program: read, parse, check |
| `vision_model.py` | Mirage as a LangChain chat model that reads text |
| `score.py` | score every receipt against the ground truth |
| `make_receipts.py` | draw the receipts and the ground truth |
| `receipts/` | the images and `truth.json` |
| `requirements.txt` | `cactus-compute`, pinned |
