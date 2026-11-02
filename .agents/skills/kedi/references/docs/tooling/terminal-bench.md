# Terminal-Bench 2.1

Kedi can run through Harbor's custom-agent interface for Terminal-Bench 2.1. The
integration owns the Kedi harness, terminal tools, non-interactive approval
policy, history, artifacts, and durable Kedi records. Harbor continues to own
the dataset, task containers, resource limits, timeouts, graders, task lock,
and job resume lifecycle.

This is the current integration guide. The [published engineering runs](terminal-bench-results.md)
used frozen source snapshots and a dedicated controller/transport setup; the
commands below are not a bit-for-bit reconstruction of those historical runs.

## Install

Harbor requires Python 3.12 or newer:

```bash
python3.12 -m pip install 'kedi[terminal-bench]'
```

`KediAgent` installs the Pydantic AI provider runtime inside each task container,
separate from Harbor's host environment. During local development, passing a
wheel built from the exact Kedi commit is the most reproducible path.

The host extra pins `codex-auth-helper==1.8.0` for credential management.
Codex task runtimes install `codex-auth-helper[websocket]==1.8.0` separately.
Do not combine `terminal-bench` with `codex-model` or Kedi's development group
in one environment: Harbor's LiteLLM dependency requires OpenAI `<3`, while
the WebSocket runtime requires OpenAI `>=3.8.0`.

## Freeze a Run

Build Kedi, then create the immutable manifest before observing benchmark
results:

Use a clean source checkout and its built wheel. Replace the Harbor revision
with the exact revision you install; the example SHA identifies a historical
revision, not a dynamically discovered version. `task-a` and `task-b` below are
placeholders: replace them with real task names from your pinned dataset before
running. A manifest can be constructed without proving those tasks exist.

```bash
uv build
kedi-terminal-bench manifest \
  --output runs/pilot.json \
  --harbor-revision 389bd4f8ce796ef4a97de4b62675021e262c8e76 \
  --model openrouter/openai/gpt-5.6-luna \
  --effort high \
  --kedi-wheel dist/kedi-0.4.0-py3-none-any.whl \
  --timeout-multiplier 1 \
  --agent-timeout-multiplier 1 \
  --verifier-timeout-multiplier 1 \
  --max-retries 0 \
  --task task-a \
  --task task-b
```

The manifest records:

- Terminal-Bench and Harbor versions;
- Harbor and Kedi revisions plus Kedi dirty state and the wheel SHA-256 or
  published package specification;
- adapter, model, effort, and non-secret model settings;
- exact task names, attempts, concurrency, environment, all Harbor timeout
  multipliers, and retry policy;
- history, provider prefix-cache placement, artifacts, compaction, and terminal
  limits.

Credential-like model setting keys and token values are rejected. Credentials
must be supplied through Harbor's provider environment.

For Daytona, configure the account and resource quota on the controller. For
local execution, select `--environment docker` and verify Docker is running.
Provider/model credentials and sandbox credentials are separate prerequisites.
Never place Codex auth JSON in a manifest, image, public bundle or task prompt.

Writing materially different settings to an existing manifest path is refused.
Its content digest excludes only the creation timestamp.

Agent setup and environment-build timeout multipliers are available alongside
the general, agent, and verifier controls. When retries are enabled, repeat
`--retry-include` and `--retry-exclude` to freeze the eligible Harbor exception
classes. `--retry-all-exceptions` explicitly clears Harbor's default exclusion
list.

## Run

Run the manifest with the recorded wheel:

```bash
kedi-terminal-bench run runs/pilot.json \
  --kedi-wheel dist/kedi-0.4.0-py3-none-any.whl \
  --jobs-dir runs/jobs \
  --job-name pilot-1
```

Use `--dry-run` to inspect the exact Harbor command without starting a job. If
the job directory and name are omitted, Kedi uses `./jobs` and a deterministic
name derived from the manifest digest.

The sample timeout multipliers are 1, not the six-hour agent override used in
the published runs. A multiplier scales a task's configured timeout; it does
not set the same absolute duration on every task. Record the final Harbor
configuration and inspect `--dry-run` before allocating resources.

Before a real job starts, Kedi checks that the selected `harbor` executable
reports the manifest's pinned Harbor version. A dry run only renders the
command and deliberately skips this executable check.

Kedi copies the manifest to `kedi-manifest.json` inside the Harbor job. Harbor's
generated `lock.json` records resolved task hashes, image digests, resources,
and grader inputs. Preserve both files with any reported result.

For a full-dataset run, freeze the task names explicitly and keep memory-heavy
tasks in a separate manifest when the sandbox account cannot run two of them
concurrently. The published 89-task engineering runs used 81 standard tasks at
concurrency 2 and 8 high-memory tasks at concurrency 1. They were aggregated
only after both manifests completed and each task appeared exactly once. See
[Terminal-Bench 2.1 Results](terminal-bench-results.md) for the exact
configuration, versions, metrics, and limitations.

To compare adapters, generate separate manifests and change only `--adapter`:

```bash
kedi-terminal-bench manifest \
  --output runs/pydantic.json \
  --harbor-revision "$HARBOR_REVISION" \
  --model codex/gpt-5.6-luna \
  --adapter pydantic \
  --effort high \
  --concurrency 2 \
  --attempts 1 \
  --max-retries 0 \
  --kedi-wheel dist/kedi-0.4.0-py3-none-any.whl \
  --task task-a \
  --task task-b

kedi-terminal-bench manifest \
  --output runs/langchain.json \
  --harbor-revision "$HARBOR_REVISION" \
  --model codex/gpt-5.6-luna \
  --adapter langchain \
  --effort high \
  --concurrency 2 \
  --attempts 1 \
  --max-retries 0 \
  --kedi-wheel dist/kedi-0.4.0-py3-none-any.whl \
  --task task-a \
  --task task-b
```

Keep model settings, task revisions, task order, prompt, limits, transport,
history, artifacts, and sandbox resources unchanged when the adapter is the
subject of the comparison. Credentials belong in the provider environment,
not in the manifest.

Installing the WebSocket helper extra or selecting `codex/...` does not itself
select WebSocket transport. The published run used an explicit WebSocket-first
factory/controller wrapper. For new custom runtimes, the public
[Codex model factory](../agent-adapters/codex-models.md) exposes
`connection="websocket", fallback="http"`; verify the actual transport trace
before claiming the historical transport configuration was reproduced.

Resume an interrupted job through Harbor's native resume path:

```bash
kedi-terminal-bench resume runs/jobs/pilot-1
```

The original `kedi-terminal-bench run` command is restart-safe when it is
repeated with the same manifest, jobs directory, and job name. Once Harbor's
`lock.json` or `config.json` exists, Kedi uses Harbor's native resume command
instead of starting a duplicate job. Repeating the command after all trials are
accounted for is a no-op. Materially different manifests remain rejected.

Keep the Harbor controller on a durable host. Daytona can still supply isolated
task environments, but the controller itself should not run inside an ephemeral
task sandbox: provider shutdown can otherwise interrupt the handoff from a
completed agent to its verifier. The persisted Kedi manifest and Harbor lock let
a durable controller continue after a process or host restart.

## Task Runtime

The benchmark profile is deliberately neutral to individual tasks. It tells the
agent to inspect before editing, ground effects in tool results, use exact argv
unless shell syntax is required, run focused verification, and stop when the
task is verified or no safe progress remains.

The task-container tool surface includes:

- sandbox-rooted filesystem reads, writes, directory creation, and structured
  patches;
- foreground argv and explicit Bash execution;
- background process start, bounded wait, status, bounded output reads, stdin,
  and stop;
- full capped process logs transported through Tool Artifacts;
- a verification state that becomes stale after later mutations.

Every process belongs to the trial session. A finite background job can be
waited on without terminating it when the wait expires; running wait results do
not repeat output previews. Waiting again on a completed verification process
does not revalidate a workspace changed since that verification. Timeout,
cancellation, output-limit termination,
and normal teardown terminate remaining process groups. Provider credentials
are stripped from terminal subprocess environments. Binary output is exposed
as base64 by bounded reads. A truncated text result identifies itself as a
head/tail excerpt and returns the exact `read_process_output` continuation for
the complete capped process log. The `output_continuations` field groups these
instructions by stream, before the potentially large stdout and stderr fields
in the artifact JSON representation. Commands start in the workspace but may use a
different path when the task explicitly names one inside its isolated
container.

Benchmark approval never opens an interactive prompt. Read-only operations and
declared mutations inside the isolated task container are allowed. Sensitive
requests and tools outside the explicit benchmark allowlist are denied.

### Execution Deadline

Single-step trials propagate Harbor's task timeout, override, cap, and multiplier
to the runner. An explicit `runner_timeout_seconds` can shorten this budget,
not extend it. The deadline starts at the agent's `run` entry and includes
instruction handoff and runner startup. The host and sandbox must have
synchronized clocks; Harbor still enforces its own outer timeout. Without
Harbor metadata, direct integration callers may supply an explicit runner
timeout. Multi-step phase budgets are not inferred automatically.

Commands cannot consume the finalization reserve. Near the deadline, one
terminal dictionary result adds `execution_budget`, reporting the remaining
seconds and reserve. The notice occurs in the last 20% of the runner's remaining
budget, capped at 120 seconds. It does not trigger an extra model call, replace
command output, repeat every turn, or change the stable prompt/history prefix.
The agent deadline does not shorten the bounded lifetime of services explicitly
retained for verification after successful completion. Cancellation, failure,
and sandbox teardown still terminate them.

## History and Artifacts

Stateful history and file-backed artifacts are enabled by default. Large tool
results remain available without placing their complete payload into every
model request. Stateful history also applies Kedi's provider-native prefix-cache
placement where the selected provider supports it.

Use `--no-history` or `--no-artifacts` when preparing controlled comparisons.
Disabling history also disables Kedi-managed prefix-cache placement. Native
compaction is opt-in:

```bash
kedi-terminal-bench manifest \
  --output runs/compacted.json \
  --harbor-revision 389bd4f8ce796ef4a97de4b62675021e262c8e76 \
  --model openrouter/openai/gpt-5.6-luna \
  --task task-a \
  --compaction-mode native \
  --compaction-threshold 100000
```

The fixed profile does not enable skill discovery, subagents, or dynamic
workflows. Those capabilities are not needed by Terminal-Bench tasks and
require separate experiments before they can become benchmark defaults.

## Evidence and Failures

Each trial preserves:

- `kedi-result.json` with state, phase, policy, verification, and usage;
- `runner-exit.json` with the runner process exit code and timestamp;
- `setup-runtime.log` with partial installation output and phase timestamps;
- `terminal-events.jsonl` with process lifecycle and verification changes;
- bounded command records and complete capped terminal stream files;
- file-backed artifact payloads;
- redacted error and cleanup diagnostics.

Terminal states distinguish completion, agent failure, integration failure,
timeout, and cancellation. The failure phase distinguishes setup, agent
execution, and teardown. Kedi usage and cache counters are projected into
Harbor's `AgentContext` after Harbor syncs the task-container logs to the host.
If timeout or cancellation interrupts aggregate usage reporting, completed
request and token counters are recovered from the append-only
`model-requests.jsonl` evidence. Provider cost is recovered only when every
completed request contains a measured cost; Kedi does not invent a partial cost.

Runtime installation output is saved while bootstrap, managed-Python creation,
and package installation are running, including when setup is interrupted.
The result timestamp, process exit timestamp, and Harbor completion timestamp
are separate: a sleeping controller or delayed remote-command polling must not
be mistaken for active model work. Keep the controller awake throughout a run,
including when tasks execute remotely. Missing exit evidence is not proof of a
clean shutdown.

After a tracked command exits, Kedi terminates descendants remaining in its
process group before draining output. To keep a service running, use the
background process tools and `retain_process`; the tracked main process must
remain alive. The command's exit code and captured output are preserved.

Remote runner cleanup on cancellation has a 25-second host-side timeout in
addition to its remote command timeout. A second cancellation also cancels the
cleanup operation. These bounds avoid an unbounded wait on cooperative network
clients; they cannot guarantee cleanup of an unreachable sandbox. Harbor still
owns environment teardown.

The integration does not include benchmark solutions or produce a score by
itself. Official graders remain the only source of task correctness.

## Record With Autobench

[Kedi Autobench](https://github.com/kedi-lang/kedi-autobench) imports completed
Harbor trials without replacing Harbor as the execution or grading authority.
Install it outside the task containers, then record the completed job:

```bash
python3.12 -m pip install kedi-autobench

kedi-autobench-terminal-bench record \
  --job-dir runs/jobs/pilot-1 \
  --record-dir runs/records/pilot-1

kedi-autobench-terminal-bench validate \
  --job-dir runs/jobs/pilot-1 \
  --record-dir runs/records/pilot-1
```

The record command maps every Harbor trial to one Autobench run, copies bounded
and redacted evidence, validates the live result, and replays the persisted
record before returning. A reward of zero is a valid benchmark outcome, not a
capture error. Replay inspects the frozen record without rerunning Harbor, Kedi,
or the model:

```bash
autobench replay runs/records/pilot-1
autobench report runs/records/pilot-1
```

To capture after a command in one operation, use the non-blocking wrapper. The
wrapped command's exit code remains authoritative even if post-run capture
fails:

```bash
kedi-autobench-terminal-bench run \
  --job-dir runs/jobs/pilot-1 \
  --record-dir runs/records/pilot-1 \
  -- kedi-terminal-bench run runs/pilot.json \
       --kedi-wheel dist/kedi-0.4.0-py3-none-any.whl \
       --jobs-dir runs/jobs \
       --job-name pilot-1
```

Here "non-blocking" means capture failure does not change the wrapped job's
exit status. It is not a detached scheduler: the wrapper waits for the command,
then records its output. Running it on a laptop does not make that laptop
independent of the run. Keep the controller process on the durable host.

Publish the immutable manifest, Harbor `lock.json`, sanitized Harbor evidence,
Autobench record, dependency versions, source and wheel hashes, aggregation
script, and checksums together. Never publish provider credentials, dotenv
files, authentication state, or unsanitized personal paths.
