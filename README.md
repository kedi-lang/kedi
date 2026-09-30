# kedi

[![coverage](https://img.shields.io/badge/coverage-100%25-brightgreen)](https://github.com/kedi-lang/kedi/actions/workflows/tests.yml)

## Natural language, with the structure of a program

Kedi is a typed orchestration language for model-backed programs. A template
puts inputs and typed output captures in the sentence that gives them meaning.
Procedures, lexical agent profiles, control flow, and modules compose those
interactions. Python handles ordinary computation and interoperability.

```kedi
~Action(task: str, owner: str | None, deadline: str | None)

@extract_actions(notes: str) -> list[Action]:
  >> Extract agreed, unfinished work from <notes>.
  Use null for owners or deadlines that were not stated.
  The action items are [actions: list[Action]].
  = `actions`

= `extract_actions("Sam will add the regression test by Friday. The dashboard redesign is only a suggestion.")`
```

The return value is a list of typed `Action` objects. The capture is part of the
model-visible template, not only a field in a separately supplied schema.
Type validation checks structure; it does not prove that a model's answer is true.

[Documentation](https://docs.kedi-lang.org/) |
[Language reference](docs/KEDI_DSL.md) |
[Runnable examples](examples/README.md) |
[Discord](https://discord.gg/cuSbZd5he)

## Run From Source

The examples track this checkout, including its parser revision:

```bash
git clone --recurse-submodules https://github.com/kedi-lang/kedi.git
cd kedi
uv sync --dev
source .venv/bin/activate
export KEDI_ADAPTER=pydantic
export KEDI_ADAPTER_MODEL=openai:gpt-5.6-luna
```

Set the selected provider's credentials, for example `OPENAI_API_KEY`, in your
environment or a local `.env`. Never commit that file. Then:

```bash
cd examples/meeting_actions
kedi main.kedi
```

For an offline first run, use `examples/module_imports/main.kedi`; it performs
deterministic arithmetic without constructing a model client. See the
[example setup](examples/README.md#setup) for Codex subscription authentication,
and the [Jev guide](examples/jev_routing/README.md) for TypeSafe credentials.

## Explore the Language

| Workflow | What to look for |
| --- | --- |
| [Meeting actions](examples/meeting_actions/README.md) | Natural-language captures, nested result types, missing values |
| [Release notes](examples/release_notes/README.md) | Model-classified conditions and native loops |
| [Support reply](examples/support_reply/README.md) | Real procedure tools inside a scoped agent profile |
| [Parallel review](examples/parallel_review/README.md) | Two subagents, explicit tasks, typed results, no coordinator model call |
| [Artifact investigation](examples/artifact_investigation/README.md) | Large tool results, bounded reads, lexical retrieval |
| [Python API](examples/python_api/README.md) | Typed Python functions backed by Kedi docstrings or files |
| [Jev routing](examples/jev_routing/README.md) | Finite decisions, explicit criteria, thresholds, inspectable evidence |
| [Flappy Kedi](examples/flappy_kedi/README.md) | A Jev-controlled pixel game with a Kedi decision loop and Python physics |
| [Module imports](examples/module_imports/README.md) | Reusable typed code and colocated deterministic tests |
| [Knapsack](examples/knapsack/README.md) | Executable scoring and an optimizable prompt, without hiding incorrect answers |

Each example has its own source, guide, and regression coverage. Examples use
fictional inputs and never perform real business actions.
[Verification notes](examples/VALIDATION.md) distinguish deterministic tests
from live model observations.

## Python Interoperability

Keep the caller in Python when that suits your application:

```python
import kedi


@kedi.query
def summarize(update: str) -> str:
    """kedi
    >> Summarize the confirmed facts in <update> without inventing a recovery time.
    The customer-facing notice is [notice: str].
    = `notice`
    """
    ...


# Uses the configured adapter and model.
notice = summarize("Checkout is timing out; recovery has not been confirmed.")
```

`@kedi.bind` binds a separate `.kedi` file; `@kedi.type` exposes Python types.
`kedi.context(...)` provides scoped overrides. The
[complete Python example](examples/python_api/README.md) shows both styles and
passes a typed result from one call to the next.

## Models and Agents

Kedi's `AgentAdapter` is the runtime boundary. Pydantic AI and LangChain provide
framework-backed integrations; Codex, Claude, ACP, and A2A provide agent-harness
surfaces. They are integrations, not competing definitions of the language.

Use `> model:` and `> adapter:` for model/framework selection; `> agent:` selects
a harness. Profiles collect model settings, instructions, tools, and child
agents. Procedures capture their lexical configuration instead of inheriting
arbitrary overrides from their callers. See
[profiles and tools](https://docs.kedi-lang.org/agentic-engineering/).

Jev-backed programs use typed decision primitives rather than treating Jev as
a free-form text generator. A2A profiles can delegate to remote agents; a Kedi
program can also be served over A2A. Optional backends require their own extras
and credentials.

## Tools for Development

```bash
kedi main.kedi --parse        # Parse without executing
kedi main.kedi --test         # Run colocated deterministic tests
kedi main.kedi --eval         # Evaluate dataset-backed metrics
kedi --idle                  # Incremental terminal session
kedi notebook                # Local notebook (requires the notebook extra)
kedi-lsp                     # Language server for editors
```

Use `> show:` to display intermediate values. `=` returns from the current
procedure or regular program; it is not a substitute for printing multiple
values. Independent model work can overlap, while dependent captures wait for
the values they consume.

Editor sources live in [kedi-vscode](kedi-vscode/) and [kedi-zed](kedi-zed/);
the debugger lives in [debugger](debugger/). The notebook is a local-build
surface: install it with `uv sync --dev --extra notebook`.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup and quality gates,
[docs/README.md](docs/README.md) for the repository documentation map, and
[examples/README.md](examples/README.md) for example conventions.

```bash
make check
python -m pytest tests/test_curated_examples.py tests/test_knapsack_example.py --record-mode=none
make check-coverage
```
