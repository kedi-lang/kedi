# Repository documentation

Start with the [public documentation](https://docs.kedi-lang.org/) for tutorials
and task-oriented guides. This directory contains the in-repository language
reference and development notes; it is not a second copy of the cookbook.

| Document | Purpose |
| --- | --- |
| [KEDI_DSL.md](KEDI_DSL.md) | Language syntax, semantics, runtime boundaries, and Python API |
| [GEPA_GUIDE.md](GEPA_GUIDE.md) | Optimizer integration and configuration |
| [RELEASE_VALIDATION.md](RELEASE_VALIDATION.md) | Distribution, platform, and release checks |
| [LOCAL_PACKAGE_REGISTRY.md](LOCAL_PACKAGE_REGISTRY.md) | Run and test the development registry |
| [EVENT_LOOPS.md](EVENT_LOOPS.md) | Design record: which event loop synchronous Kedi work runs on, and why |
| [Runnable examples](../examples/README.md) | Focused programs with input fixtures, guides, and tests |

For a complete optimization example, use [Knapsack](../examples/knapsack/README.md).
Its data, scoring, limitations, and verification results are kept beside the code.

## Background references

[gepa_aime.md](gepa_aime.md), [gepa_papillon.md](gepa_papillon.md), and
[gepa_structured.md](gepa_structured.md) are external DSPy/GEPA reference material.
They are not current Kedi examples, Kedi benchmark results, or instructions for
the current Kedi API. Prefer the maintained guide and executable example above.

## Updating examples

Keep a workflow in one directory under `examples/`, with its own README and any
small, non-sensitive fixtures. Explain the input, the result, how to run it, and
what the tests actually establish. Templates must form meaningful sentences when
their captures are filled and must start with `>>`.

Do not publish parser probes, private scratch programs, fabricated benchmark
speedups, or model self-reports as demonstrations of correctness. Keep Python
blocks direct; use helpers only when they express a reusable operation. Add
regressions against the actual example files, not a copied version of their code.

Changes to `KEDI_DSL.md` must also update the editor copies. `make check-docs`
verifies synchronization; the published site lives in the `kedi-docs` submodule.
