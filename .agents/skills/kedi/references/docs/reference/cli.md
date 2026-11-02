# CLI Reference

## Invocation

```text
kedi SOURCE [OPTIONS] [PROGRAM_OPTIONS]
kedi -c SOURCE_TEXT [OPTIONS] [PROGRAM_OPTIONS]
kedi --idle [RUNTIME_BACKEND_OPTIONS] [--record] [--load SESSION_PATH] [--no-color]
kedi parse SOURCE
kedi SOURCE --parse
kedi install [PACKAGE.KEDI]
kedi add PACKAGE_NAME
kedi add git+https://github.com/OWNER/REPOSITORY.git
kedi skills add --path SKILL_DIRECTORY
kedi skills add --repo OWNER/REPOSITORY
kedi a2a serve SOURCE --entry PROFILE [OPTIONS]
kedi notebook [--host HOST] [--port PORT] [--python PATH] [--cwd PATH] [--token TOKEN] [--no-open]
```

Use `kedi --help`, `kedi install --help`, or `kedi add --help` for the relevant
usage surface.

`-c/--command` supports direct execution and `--parse` only. It cannot be
combined with tests, evals, or optimization because those modes require a
source-file identity and adjacent artifacts.

## Runtime Backend Options

| Option | Default | Meaning |
| --- | --- | --- |
| `--adapter NAME` | `pydantic` | `pydantic`, `dspy`, `langchain`, `claude`, `codex`, `acp`, or `a2a` |
| `--adapter-model MODEL` | `openai/gpt-6-luna` | Default model identifier; examples explicitly select `openai:gpt-6-luna` |

The historical option name is `--adapter`, but it accepts both framework
adapters and harnesses. In Kedi source and the Python API, `adapter` and `agent`
remain separate type-safe concepts.

## Interactive Mode

`kedi --idle` starts a process-local incremental session. It uses `+++` for a
new fragment and `...` for continuation input. Enter an empty continuation line
to execute a buffered multiline fragment. `:multiline` opens a one-shot editor
that submits on a complete double Enter; `Alt+Enter` forces submission.
`:show <expression>` inspects a value, `help` displays command help, and
`:exit`, `Ctrl+C`, or `Ctrl+D` closes the session.

`:dump` writes a strict snapshot and prints
`kedi --idle --load <session_path>`. `--record` performs that dump automatically
before the REPL exits. `--load` restores the supplied snapshot and keeps
recording subsequent changes to the same file.

`--snapshot-format portable|python` selects the format for `:dump`/recording.
New sessions default to the existing pickle-free `portable` format. Python mode
requires `kedi[snapshots]`; resuming it requires `--trust-snapshot`, acknowledging
arbitrary Python code execution and possible saved credentials. A loaded Python
session retains its format unless explicitly overridden. Format selection requires
`--idle`; trust acknowledgement requires `--idle --load`.

Live Kedi and embedded-Python syntax highlighting is enabled by default while
preserving the shared REPL history. `--no-color` disables input highlighting and
REPL error colors; a nonempty `NO_COLOR` environment variable does the same.
Redirected input or output uses plain line input. The existing `--highlight`
flag remains supported. Interactive mode accepts `--adapter` and
`--adapter-model`. It rejects a source
file, `-c/--command`, program arguments, `--parse`, `--test`, `--eval`, and
`--optimize`. `--record`, `--load`, `--highlight`, and `--no-color` require `--idle`. See
[Interactive Execution](../runtime/interactive-execution.md)
for state persistence, native results, imports, history, and complete terminal
behavior.

## Validation and Generation Options

| Option | Meaning |
| --- | --- |
| `--test` | Run all `@test:` cases and stop |
| `--eval` | Run all `@eval:` suites and stop |
| `--optimize` | Optimize every eligible `> optimize:` span before continuing |
| `--no-cache` | Disable/remove `> auto:` codegen cache for this command |
| `--quiet` | Suppress codegen and GEPA progress |
| `-p`, `--parse` | Parse source without compiling or executing |

If both `--test` and `--eval` are given, tests run and return first. Avoid the
ambiguous combination and invoke each validation mode separately.

## Code Generation Options

| Option | Default |
| --- | --- |
| `--codegen-agent` | `pydantic_ai` |
| `--codegen-model` | `openai/gpt-6-luna` |
| `--codegen-retries` | `5` |

The alternate codegen agent is `mock`, intended for deterministic tests.
Retries count generation stages. Generated implementations are cached beside
the source as `SOURCE.cache.kedi`.

## Optimizer Options

| Option | Default | Meaning |
| --- | --- | --- |
| `--optimizer` | `mock` | `mock` or `gepa` |
| `--optimizer-model` | `openai/gpt-6-luna` | GEPA main model |
| `--optimizer-reflection-model` | `openai/gpt-6.1-sol` | GEPA reflection model |
| `--optimizer-max-metric-calls` | `100` | Optimization evaluation budget |
| `--optimizer-model-max-tokens` | `8000` | Main-model output limit |
| `--optimizer-reflection-model-max-tokens` | `32000` | Reflection output limit |
| `--optimizer-reflection-minibatch-size` | `3` | Examples before reflection |
| `--optimizer-max-validation-examples` | all | Explicit validation-set cap |
| `--optimizer-fresh` | false | Delete prior optimized state/checkpoints |

`--optimize` does not imply GEPA; the CLI default optimizer is `mock`.
`--optimizer-fresh` removes `.optimized.json`, `.optimized_scores.json`, and
`.gepa/` artifacts. It does not control codegen cache.

## Program Arguments

Unrecognized options after the source are parsed into the reserved `args`
binding:

```bash
kedi report.kedi --incident-id 42 --dry-run
```

```kedi
[incident_id: int] = `int(args.incident_id)`
[dry_run: bool] = `args.dry_run is True`
```

Hyphens normalize to underscores. `--key value` stores a string; a flag without
a value stores `True`. The first duplicate option wins. Non-option extra tokens
are ignored. Missing attributes read as `None`.

## Package Commands

`kedi install` accepts zero or one manifest path and defaults to
`./package.kedi`. `kedi add` accepts exactly one registry name or credential-free
GitHub `git+https` URL. Package subcommands reject ordinary runtime options.

## A2A Server

`kedi a2a serve SOURCE --entry PROFILE` exposes one exported profile through
the official A2A protocol server. Install `kedi[a2a]` first.

| Option | Default / meaning |
| --- | --- |
| `--entry NAME` | Required exported profile |
| `--host`, `--port` | `127.0.0.1`, `8000` |
| `--public-url URL` | Required for a non-loopback bind |
| `--rpc-path PATH` | `/a2a` |
| `--auth` | `basic`, `bearer`, or `api-key`; default `basic` |
| `--username` | Basic username; default `kedi` |
| `--secret-env`, `--secret-file` | Secret reference; mutually exclusive |
| `--api-key-header` | API-key header; default `X-API-Key` |
| `--allow-unauthenticated` | Explicit loopback-only development mode |
| `--model` | Server entry model override |
| `--max-concurrency` | Concurrent profile executions; default `4` |
| `--max-pending-tasks` | Global and per-principal pending bound |
| `--max-sessions` | In-memory principal/context session bound |
| `--max-request-bytes` | HTTP request body bound |
| `--max-input-chars` | Task text bound |
| `--max-response-bytes` | Result bound |
| `--max-tasks-per-principal` | Retained task bound |

See [A2A Cloud Agents](../agent-adapters/a2a.md) for authentication, structured
results, lifecycle recovery, and deployment limits.

## Notebook Command { #notebook-command }

`kedi notebook` dispatches to the separately installed `kedi-notebook` package.
The source checkout supports `uv run --extra notebook kedi notebook` once its
notebook submodule is initialized. See [Local Notebook](../tooling/notebook.md).

| Option | Default / meaning |
| --- | --- |
| `--host` | `HOST` or `127.0.0.1` |
| `--port` | `PORT` or `8788` |
| `--python PATH` | Additional base interpreter; repeatable |
| `--cwd PATH` | Current directory; import and host command base |
| `--token TOKEN` | `KEDI_NOTEBOOK_TOKEN`; required for non-loopback serving |
| `--no-open` | Do not open the browser automatically |

Use `kedi-notebook --help` for the notebook server's full option list. These
are notebook options, not flags for ordinary `kedi program.kedi` execution.

## Skill Installation

`kedi skills add` requires exactly one of `--path` or `--repo`. The source must
provide a root `SKILL.md`. Installation copies that skill into
`$KEDI_HOME/registry/skills`; it does not activate it in a program. See
[Skills](../agentic-engineering/skills.md) for loading and trust boundaries.

## Exit and Output

Success exits `0`. Parse, compile, execution, validation, codegen, optimization,
package, and usage failures exit nonzero and render source-aware diagnostics
when available. Test mode exits `1` when any case fails; eval mode exits nonzero
for suite/configuration failures.

Ordinary top-level return values are written to stdout. Diagnostics and failed
operation details use stderr. `--quiet` does not hide final results or errors.
