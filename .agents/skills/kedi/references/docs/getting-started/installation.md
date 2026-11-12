# Installation

## Supported Python Versions

The core package requires Python 3.11 or newer and advertises Python 3.11
through 3.14. Optional integrations can require a newer interpreter; this is
not a promise that every extra runs on every core-supported Python version.

## Install with uv

For a new application project, create an environment first:

```bash
uv init my-kedi-project
cd my-kedi-project
uv add kedi
```

Run Kedi inside the project environment:

```bash
uv run kedi --help
uv run kedi -c '= ready'
```

For a temporary CLI invocation without adding a project dependency, use
`uvx kedi --help`. Use `uv run kedi` for the rest of this guide when using a
uv project, or activate `.venv` so the shorter `kedi` commands resolve correctly.
Commit `uv.lock` and use `uv sync --locked` in automation to preserve the tested
dependency resolution.

## Install with pip

Kedi can also be installed into an activated virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install kedi
```

Use `uv` for the repository's contributor workflow. The `pip` path is intended
for consumers whose environments are managed by another tool.

On Windows PowerShell, activate with `.\.venv\Scripts\Activate.ps1` instead.
The `source` command above is for POSIX shells.

## Configure Your First Model

No provider or API key is needed for parsing or deterministic Python expressions.
The model-backed learning examples explicitly select
`openai:gpt-6-luna` through the Pydantic adapter, and Kedi installs its OpenAI
provider support. Kedi 0.1.5 and earlier also need `pydantic-ai-slim[openai]`
in the same environment.

Set `OPENAI_API_KEY` in your shell or a local, untracked `.env` file:

```dotenv
OPENAI_API_KEY=your-key-here
```

The `kedi` command loads dotenv at startup. The Python API does not imply the
same CLI startup step: call `dotenv.load_dotenv()` yourself when embedding and
depending on a `.env` file. A key authenticates the provider; it does not select
a model. The source example supplies both `> adapter:` and `> model:`.

Do not commit `.env`, credentials, or captured secret-bearing output. A real
model call can incur provider charges. Start with the no-model verification below.

## Optional Backend Dependencies

The core distribution includes the Pydantic AI adapter and its OpenAI provider.
Other adapters, provider SDKs, and runtime surfaces are installed only when
selected. These are alternative additions for different applications, not one
installation sequence:

```bash
uv add "kedi[claude]"
uv add "kedi[codex-model]"
uv add "kedi[dspy]"
uv add "kedi[groq]"
uv add "kedi[langchain]"
uv add "kedi[langchain-aws]"
uv add "kedi[playground]"
uv add "kedi[notebook]"
uv add "kedi[typesafe]"
```

`dspy` installs DSPy for GEPA optimization, not a runtime adapter. `langchain`
installs the LangChain adapter's common OpenAI, OpenRouter, and MCP
integrations; add `langchain-aws` only for Bedrock. `groq` installs the provider
SDK used by Groq-backed Pydantic AI models. `claude` and `codex-model` install
their corresponding SDK bridges, while `playground` adds browser server
dependencies. Codex, Claude, and ACP harnesses may also require an executable,
authentication, or explicit command configuration.

Install only what the program uses. The Codex-model extra targets Python 3.11+
and Terminal-Bench tooling targets Python 3.12+. Notebook and Jev setup have
their own guides: [Notebook](../tooling/notebook.md) and
[Typesafe / Jev](../agent-adapters/typesafe.md). Installing an extra does not
authenticate a provider, start a harness, or change the model selection.

## Verify the CLI

```bash
kedi --help
kedi -p -c "= ready"
kedi -c "= ready"
```

The second command checks syntax; the third executes and prints `ready`.
Neither contacts a model. `kedi-lsp` is a stdio service launched by an editor,
not an interactive CLI verification command. Continue with
[Your First Program](first-program.md).

## Upgrade Kedi

With uv:

```bash
uv lock --upgrade-package kedi
uv sync
```

Review release notes before upgrading a production workflow. Kedi programs may
also depend on provider SDK behavior, agent harness versions, and installed
Kedi packages.
