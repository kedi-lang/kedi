# Environment Variables

## Runtime Selection

| Variable | Meaning |
| --- | --- |
| `KEDI_ADAPTER` | Default framework adapter for the Python API/CLI |
| `KEDI_AGENT` | Default harness for the Python API |
| `KEDI_ADAPTER_MODEL` | Default model for the selected backend |
| `MODEL_NAME` | CLI fallback model and codegen fallback |
| `KEDI_PARALLEL` | Override default concurrency or select sequential execution |

`KEDI_ADAPTER` and `KEDI_AGENT` are mutually exclusive. Python `configure()`,
decorator arguments, source directives, and CLI options can provide more local
selection. `KEDI_PARALLEL` accepts `0/1`, `true/false`, `on/off`, or a positive
worker count; invalid values fail rather than silently selecting a mode.
Unset or empty enables concurrency with eight workers. Use `KEDI_PARALLEL=0`
for sequential execution; no variable is needed to enable concurrency.

## Code Generation

| Variable | CLI equivalent |
| --- | --- |
| `KEDI_CODEGEN_AGENT` | `--codegen-agent` |
| `KEDI_CODEGEN_MODEL` | `--codegen-model` |
| `KEDI_CODEGEN_RETRIES` | `--codegen-retries` |

`MODEL_NAME` is also consulted by the Pydantic codegen service when no explicit
codegen model is supplied.

## Optimization

| Variable | CLI equivalent |
| --- | --- |
| `KEDI_OPTIMIZER` | `--optimizer` |
| `KEDI_OPTIMIZER_MODEL` | `--optimizer-model` |
| `KEDI_OPTIMIZER_REFLECTION_MODEL` | `--optimizer-reflection-model` |
| `KEDI_OPTIMIZER_MAX_METRIC_CALLS` | `--optimizer-max-metric-calls` |
| `KEDI_OPTIMIZER_MODEL_MAX_TOKENS` | `--optimizer-model-max-tokens` |
| `KEDI_OPTIMIZER_REFLECTION_MODEL_MAX_TOKENS` | `--optimizer-reflection-model-max-tokens` |
| `KEDI_OPTIMIZER_REFLECTION_MINIBATCH_SIZE` | `--optimizer-reflection-minibatch-size` |
| `KEDI_OPTIMIZER_MAX_VALIDATION_EXAMPLES` | `--optimizer-max-validation-examples` |
| `KEDI_DEBUG_OPTIMIZED=1` | Print optimized prompt-loading diagnostics |

The debug variable is for diagnosis and may reveal prompt content; do not leave
it enabled in normal production logs.

## Packages

| Variable | Contract |
| --- | --- |
| `KEDI_HOME` | Absolute Kedi state root; relative paths are rejected |
| `KEDI_REGISTRY_MOCK_ROOT` | Local mock registry root until the public registry exists |

Installed packages live beneath `$KEDI_HOME/registry` or the platform default
Kedi home. Git package operations set `GIT_TERMINAL_PROMPT=0` so unattended
installs fail instead of waiting for credentials.

## Filesystem and Interactive State

| Variable | Contract |
| --- | --- |
| `KEDI_WORKSPACE_POLICY` | Unset/empty or `strict` enforces workspace containment; `none` disables that containment, not all secret/deletion rules |
| `KEDI_HISTORY` | REPL input history file, default `~/.kedi_history` |
| `KEDI_CODEX_AUTH_FILE` | Explicit Codex Responses auth file; see [Codex models](../agent-adapters/codex-models.md) |

See [Filesystem](../modules-and-packaging/filesystem.md) before changing workspace
policy. It does not sandbox embedded Python or external agents.

## Notebook Server

| Variable | Contract |
| --- | --- |
| `HOST`, `PORT` | Notebook bind defaults, `127.0.0.1` and `8788` |
| `KEDI_NOTEBOOK_TOKEN` | Notebook API bearer token; required off loopback |
| `KEDI_NOTEBOOK_ENV_HOME` | Root for managed host virtual environments |
| `KEDI_NOTEBOOK_SECRETS_PATH` | Secret Manager JSON path; default `~/.kedi/notebook/secrets.json` |

Notebook reads `.env` from its working directory without overriding existing
process values, then applies its Secret Manager entries. Updating Secret Manager
resets active runtimes. These files and their contents must not be committed.

## A2A Server

| Variable | Contract |
| --- | --- |
| `KEDI_A2A_PUBLIC_URL` | Public discovery base URL used by `kedi a2a serve` |
| `KEDI_A2A_AUTH_SCHEME` | `basic`, `bearer`, or `api-key` |
| `KEDI_A2A_USERNAME` | Basic-auth username; default `kedi` |
| `KEDI_A2A_SECRET` | Direct server credential fallback |
| `KEDI_A2A_MODEL` | Model override for the exported server profile |

Prefer a named variable through `--secret-env` or a mounted file through
`--secret-file` over `KEDI_A2A_SECRET`. Client profiles reference their own
credential variable or file in the `> agent: a2a:` auth block.

## Benchmark Instrumentation

`KEDI_HARBOR_LOGFIRE=1` opts the Harbor integration into installing and
configuring Logfire and Kedi instrumentation. It is disabled when unset or
different from `1`. Exporting telemetry also depends on Logfire credentials;
this flag is not authentication. Leave it unset for an uninstrumented run.

## Provider Credentials

Provider SDKs read their own variables, such as `GROQ_API_KEY`,
`OPENAI_API_KEY`, `OPENROUTER_API_KEY`, or `ANTHROPIC_API_KEY`. Kedi forwards
selection and settings but does not rename provider credentials. The exact
required variable depends on the model string and adapter.

ACP, Claude, and Codex process settings can also supply an explicit `env`
mapping. It is merged with the current process environment where that adapter
documents process spawning.

## `.env` Loading and Security

The Python API's default configuration loads `.env` through its environment
setup. Explicit arguments and scoped configuration take precedence. Do not
depend on `.env` loading inside a separately spawned harness unless that process
also loads or receives the values.

Never put secrets in `.kedi` source, optimizer datasets, debug exports, or
committed environment files. `MarkdownDebugExporter` does not redact values.
Bundled filesystem tools treat `.env` names as sensitive, but arbitrary Python
code is not intercepted by the approval layer.
