# Kedi examples

Small, complete programs that solve a specific problem. Each directory contains
its own guide, inputs, and runnable entrypoint. The examples use synthetic data;
none sends a real customer message, changes an account, or executes a reviewed patch.

## Choose a workflow

| Example | What it does | Main concepts | Needs a model |
| --- | --- | --- | --- |
| [Meeting actions](meeting_actions/README.md) | Turn messy meeting notes into assigned actions | Templates, nested types, nullable fields, constraints | Yes |
| [Release notes](release_notes/README.md) | Filter internal changes and write user-facing notes | Model-evaluated conditions, sequential loops | Yes |
| [Support reply](support_reply/README.md) | Draft a policy-grounded delivery reply | Procedure tools, docstrings, lexical profiles | Yes |
| [Parallel review](parallel_review/README.md) | Review a patch from two independent perspectives | Subagent profiles, native task/await, typed results | Yes |
| [Artifact investigation](artifact_investigation/README.md) | Find the supported cause of a checkout incident | Bounded previews, lexical artifact retrieval | Yes |
| [Python API](python_api/README.md) | Extract an incident and draft a status notice | `@query`, `@bind`, `@type`, native Python values | Yes |
| [Jev routing](jev_routing/README.md) | Route a ticket using explicit decision criteria | Finite choices, thresholds, decision metadata | Jev |
| [Flappy Kedi](flappy_kedi/README.md) | Fly a pixel cat with typed model decisions, or play yourself | Kedi decision loop, choice criteria, Python binding, Pygame | Jev, local Laya, or manual |
| [Module imports](module_imports/README.md) | Compute an order quote in integer cents | Exports, imports, typed procedures, colocated tests | No |
| [Knapsack](knapsack/README.md) | Evaluate feasible, optimal item selections | Prompt spans, datasets, scoring, optimization | Yes |

Start with **Meeting actions** for the template model or **Module imports** for
a credential-free run. Parallel review and artifact investigation build on those
ideas; they are not prerequisites for writing a Kedi program.

## Setup

From the repository root:

```bash
uv sync --dev
source .venv/bin/activate
export KEDI_ADAPTER=pydantic
export KEDI_ADAPTER_MODEL=openai:gpt-5.6-luna
```

Set `OPENAI_API_KEY` in your environment or the checkout's `.env` for that provider.
Do not put credentials in example source files. Kedi loads `.env`; the Python API
entrypoint does so explicitly. A configured provider can be changed without editing
the examples. The Jev guide has its own optional-package and credential setup.

File-backed Kedi programs expose their absolute source path as `__file__`.
The examples bind `Path(__file__)` to a local variable in a Python block, then
use `path.with_name(...)` to read sibling inputs, so
`kedi examples/meeting_actions/main.kedi` works directly from the repository root.
Running `kedi main.kedi` inside the example directory also works. Kedi does not
change the process working directory; ordinary Python relative paths remain
CWD-relative. Kedi module imports resolve relative to their source file.
In PowerShell, activate
`.venv\Scripts\Activate.ps1` and use `$env:NAME = "value"` for environment variables.

To use a Codex subscription instead of a native OpenAI API key, on Python 3.11+:

```bash
uv sync --dev --extra codex-model
codex login
export KEDI_ADAPTER_MODEL=codex/gpt-5.6-luna
```

These are model calls, not canned output. Wording, correctness, request counts,
and latency can vary. A successful test with a stubbed provider is evidence of
the program's wiring, not a claim about model accuracy.

## Validation

From the repository root, run the offline regression suite:

```bash
python -m pytest tests/test_curated_examples.py tests/test_knapsack_example.py --record-mode=none
```

Tests parse the actual files, execute the programs with the real Kedi runtime
and deterministic model responses, exercise real local tools and artifact
retrieval, and check typed results. They never make paid model requests.
[Verification notes](VALIDATION.md) distinguish those checks from live runs.

The old loose smoke scripts, duplicate optimization demos, benchmark scratch
files, and unreferenced plots have been removed. Runtime edge-case tests belong
in `tests/`, not in the public example gallery.
