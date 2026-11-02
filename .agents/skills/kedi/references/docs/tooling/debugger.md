# Debugger and Inspector

The optional `kedi-debugger` package serves both editor extensions through DAP.
It stops at Kedi statements and procedure boundaries while leaving the existing
runtime, adapters, approval policy, and concurrency in charge of execution.

!!! warning "Source-checkout feature"

    Install the matching local runtime, debugger, and editor extension. These
    changes are not an automatic Marketplace, Zed registry, or PyPI release.
    Older Kedi builds do not contain the required `kedi.debugging` hooks.

## One Interpreter

In managed mode, the updated extensions install the debugger automatically into
the shared `~/.kedi/editor-venv`, including upgrades of older managed environments.
The installer validates both the debugger package and the required runtime hooks.
This path requires matching published packages; the source-checkout warning above
still applies until release.

For a selected host interpreter or unpublished local development, install explicitly:

```sh
python -m pip install -e /path/to/kedi
python -m pip install --no-deps -e /path/to/kedi/debugger
python -c 'import kedi_debugger; from kedi.debugging import DebugEvent, observe_execution'
```

The shared managed environment is `~/.kedi/editor-venv`. Existing VS Code and Zed
host-Python overrides also govern debugging. Host interpreters are not modified
automatically. Project dependencies must be installed in that same environment.

## Start Without a Model

Save this as `debug.kedi`, put a breakpoint on the assignment inside `double`,
and launch it through the editor's debugger:

```kedi
@double(value: int) -> int:
  [result: int] = `value * 2`
  = `result`

[number: int] = `21`
[answer: int] = `double(number)`
> show: The computed answer is <answer>.
```

Inspect `value` and `result` in **Locals**, navigate the Kedi stack, step out,
and continue. The example makes no model calls. The debug console displays normal
program output; it is not a Python REPL.

## VS Code

Use **Run and Debug** with the Kedi launch configuration:

```json
{
  "version": "0.2.0",
  "configurations": [{
    "type": "kedi",
    "request": "launch",
    "name": "Debug Kedi",
    "program": "${file}",
    "cwd": "${workspaceFolder}",
    "stopOnEntry": true
  }]
}
```

The extension requires workspace trust. Select the interpreter using the existing
Kedi command or settings described in [VS Code](vscode.md).

## Zed

Use a Kedi debugger task with adapter `kedi` and the same launch fields.
The updated extension delegates to `python -m kedi_debugger --stdio` using the
same Python selection as its language server. See [Zed](zed.md) for host setup.
Zed reserves the task's `adapter` field for the debugger name, so use
`kediAdapter` for an optional model-framework override. For example, in
`.zed/debug.json`:

```json
[
  {
    "label": "Debug Kedi",
    "adapter": "kedi",
    "request": "launch",
    "program": "$ZED_FILE",
    "cwd": "$ZED_WORKTREE_ROOT",
    "stopOnEntry": true
  }
]
```

Save the file and trust the project before starting. The pinned Zed extension
API does not expose the VS Code-style trust or unsaved-buffer callbacks.

### Launch Environment

Both editors load the nearest `.env` from the launch `cwd` or its parents before
the worker imports Kedi or selects a model. Without `cwd`, discovery starts at
the program's directory. Only the nearest file is loaded. This includes API keys,
`KEDI_ADAPTER`, `KEDI_ADAPTER_MODEL`, and `MODEL_NAME`, even if the editor was
started outside a terminal.

Launch `env` entries override the inherited process environment, which overrides
`.env` values. `null` removes a variable from both sources; `""` keeps an empty
value. Dotenv quoting and `${NAME}` interpolation are supported. Explicit launch
`model` and adapter settings override environment defaults. These rules apply to
both VS Code and Zed. Keep credentials out of committed launch configurations.

To skip dotenv discovery, set `PYTHON_DOTENV_DISABLED=1` in launch `env` or the
inherited environment. The debugger sets this flag in the worker after loading
its environment so later automatic dotenv loads cannot restore removed keys or
load unrelated credentials. Restart the debug session after editing `.env`.
Known loaded credentials are masked in debug output and inspection. An invalid
or unreadable `.env` fails launch without exposing its contents; a missing file
is allowed.

## Model Boundaries

A model-backed program uses the usual Kedi directives:

```kedi
> model: openai:gpt-6-luna
[country] = Sweden
>> The capital of <country> is [capital].
> show: The capital is <capital>.
```

Enable the **Before model dispatch** and **After model result** event filters in
the debugger's breakpoint controls. Their IDs are `model_input` and `model_result`.
The first stop shows the prepared input; the second shows the result and available
request/usage observations. A request-transforming hook can change the assembled
request, so inspect `request` for the actual Kedi request rather than treating the
earlier input as a provider wire transcript.

Waiting on a capture does not stop its independent producer. Continue and step
commands resume paused peer lanes by default, so a capture can receive its result.
With explicit single-thread stepping, a paused producer needs its own continue
command. The debugger does not serialize parallel work or resend a request after
resume.

Request assembly and nested tool execution inside a model transaction are
observation-only: they cannot pause while holding that transaction's resources.
A propagated tool error can stop after those resources are released. Exceptions
from parallel model jobs also reach the exception filter.

## Inspection and Privacy

Source snapshots are frozen for the session. If a source changes before starting
or resuming, restart the debugger to run the edited program. Known environment
secrets are masked in source responses without shifting their line positions.

Snapshots are bounded and detached from live objects. Completed captures expose
their existing values and typed fields without resolving or calling the model
again. Pending promises are not forced; unknown Python objects remain opaque.
Large values do not consume the slots reserved for later sibling bindings.
If a scope exceeds its snapshot limits, a `<snapshot>` notice explains the partial
view. A resolved promise shows the memoized value actually read by the program,
even if the shared model-result mapping subsequently changes.
Names and captured paths such as
`answer`, `record.name`, or `items[0]` can be watched. Calls, assignments, operator
evaluation and dynamic property access are rejected. Resuming invalidates that
lane's inspection handles. Recent changes compare captured values only, not all
possible mutations of arbitrary Python objects.

Distinct dictionary keys such as `1` and `"1"` can share a displayed name. Watch
evaluation rejects those ambiguous paths instead of selecting an arbitrary value;
inspect their entries in the variables view.

Token usage counters, cache/reasoning token details and token limits remain
visible; the word `token` in a metric name does not make it a credential.
Access/refresh tokens, API keys and sensitive fields inside usage containers
remain redacted. Known environment secret values are masked even under otherwise
public field names.

Known environment secrets and sensitive fields are redacted, but unknown secrets
inside prose cannot be guaranteed absent. No persistent transcript or Logfire
exporter is configured. Treat screenshots and console output as potentially
sensitive. The older [executor debug exporter](../runtime/errors-and-debugging.md#executor-debug-events)
has a different privacy contract and is not this inspector.

## Limits and Lifecycle

Pause is cooperative. Already-running provider calls and their deadlines continue.
Embedded Python blocks are opaque; blocking Python may require **Terminate**.
The supervisor remains responsive and owns only its worker/process descendants.
Disconnect terminates the launched worker, not unrelated Python processes.

Use **Restart/Rerun** after saving source or `.env` changes. The editor ends the
old adapter session and launches a fresh one; the DAP restart shutdown flag is
accepted. This is a fresh execution, not hot reload. Source edits detected while
paused produce an explanatory console message instead of resuming stale code.

Thread names stay stable; DAP events carry running, stopped and exited states.
Capture waits use progress events when the editor supports them and end when
the wait or run finishes. Already-completed captures do not announce a wait.
Normal completion reports exit code `0`, ends all thread/progress states, and
then terminates the session. A closed session is not by itself a program failure.

Stepping stops before the next eligible Kedi statement, so assignments become
visible at the following stop. Internal model observations do not create extra
same-line steps unless their event filters are enabled. Editors may clear
Variables after completion; stop before the final return to inspect values.

Source edits require restart. There is no remote attach, notebook/REPL debugging,
reverse stepping, variable mutation, arbitrary Python evaluation, terminal input,
or async subagent-internal stepping. Tools still obey approval policy; DAP does
not provide an interactive terminal for approval prompts.
