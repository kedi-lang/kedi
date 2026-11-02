# Local Notebook

Kedi Notebook is a local browser interface over `InteractiveSession`. It keeps
definitions, values, imports, profiles, conversation state, and artifacts
available across cells without concatenating or replaying earlier source.

## Install and Start

From a Kedi source checkout, resolve the optional browser package and start the
notebook in one command:

```bash
uv run --extra notebook kedi notebook
```

The checkout must include the notebook submodule. If it is absent, initialize
it once with `git submodule update --init notebook` before the command above.
Initial dependency resolution may need network access.

The notebook is currently a local-build surface. This command uses the checked
out Kedi source and `notebook` submodule rather than a published distribution.

The server listens on `127.0.0.1:8788` and opens
`http://127.0.0.1:8788/notebook/`. It is a localhost development tool, not a
multi-user hosted service.

The standalone entry point has the same behavior:

```bash
kedi-notebook
```

Use `--host`, `--port`, `--cwd`, and `--no-open` to control serving. The working
directory is the base for relative Kedi imports and ordinary host filesystem
operations.

Loopback serving needs no credentials. Binding another interface requires an
explicit access token:

```bash
kedi notebook --host 0.0.0.0 --token "$KEDI_NOTEBOOK_TOKEN"
```

The token is required by every notebook API and browser bridge request. The
server also rejects cross-origin API requests. This protects the host Python
and shell execution surface; it does not turn the notebook into a multi-user
service.

The server loads `.env` from `--cwd` at startup without overriding existing
process values. Use **Secret Manager** in the top bar for model names,
credentials, and other environment values that should not enter notebook
source. Values may be entered individually or imported from an explicit
`.env` path; relative paths resolve from `--cwd`. They are stored in
`~/.kedi/notebook/secrets.json` with user-only permissions, and the browser
receives only configured variable names. Updating a value resets the active
runtime so later cells inherit the new environment.

`KEDI_NOTEBOOK_SECRETS_PATH` overrides that file location. This is local
plaintext storage protected by filesystem permissions, not an encrypted vault.
Do not print secrets into outputs or copy them into cell source.

## Browser and Host Python

See [Notebook Runtimes](notebook-runtimes.md) for the complete contract.

## Cell Semantics

Kedi cells execute when selected and run, against the current session state.
Earlier cells are not automatically replayed to satisfy dependencies. A cell
number represents the cell's current
position in the notebook, not its execution count. Rerunning a cell leaves that
number unchanged; adding, moving, or deleting cells recomputes affected
positions. A successful cell keeps its source editor and displays its output
directly below the source. It can be edited and run again; each rerun is a new
incremental execution against the current runtime state. The notebook does not
append an empty cell after execution. New cells are created only with an Add
cell control. Markdown cells render locally and never enter the Kedi runtime.

A cell whose first non-whitespace character is `!` is a terminal cell. You can
also create one explicitly from the cell type menu. Every non-empty line in a
multi-command terminal cell must begin with `!`.

In host mode, commands run in the notebook working directory. `!python` and
`!pip` always use the managed notebook environment. Other commands
run through the local shell, with the same permissions as the notebook server.
Standard output and standard error are streamed into the cell output while the
command is running. The terminal source remains editable and rerunnable after
the command finishes.

Browser mode has no operating-system shell. It supports:

- `!pip install <package>` and `!uv add <package>` through `micropip`;
- `!pip list`;
- `!echo <text>`;
- `!pwd`.

Browser `!uv add` is an installation convenience: it changes the live Pyodide
environment but does not edit `pyproject.toml`, lock files, or host project
state. Installed packages are immediately importable by later Kedi cells in
the same runtime session.

Notebook execution is intentionally non-transactional. State and external side
effects completed before an error remain visible after the cell fails. Running
any Kedi or terminal cell again creates a new execution attempt; Kedi does not
roll the session back or retry the source automatically.

Starting a new runtime clears live execution state and marks Kedi and terminal
source cells as unexecuted. There is no hidden replay and no automatic run-all
operation.

### Two Dependent Cells

Run this setup cell first:

```kedi
[values: list[int]] = `[2, 3, 5]`
```

Then run the result cell:

```kedi
> show: `sum(value * value for value in values)`
```

The result is `38`. After resetting the runtime, running only the second cell
fails because `values` does not exist. Rerun the setup cell, then the result
cell. Editing, hiding, moving or deleting a cell does not undo its earlier
effects on the live runtime.

## Notebook Files

See [Notebook Files and Recovery](notebook-files.md) for the complete contract.

## Editing

The active Kedi cell uses Monaco with Tree-sitter Kedi highlighting, embedded
Python highlighting, live Kedi and Pyright diagnostics, runtime diagnostics,
Kedi/Pyright completion, hover, references, rename, signature help, and
definition navigation. A rename that reaches into an earlier executed cell is
rejected instead of applying only the active-cell edits. If Monaco cannot load, the
notebook keeps editing available with a plain multiline fallback. Completed
cells remain editable and display output below their source.

Markdown cells safely render headings, paragraph groups, ordered and unordered
lists, fenced code, blockquotes, emphasis, links, and separators. Raw HTML is
not rendered.

Keyboard commands:

- `Shift+Enter`: run the active cell;
- `Ctrl/Cmd+S`: download the notebook;
- `Ctrl/Cmd+Enter`: insert after the active cell;
- `Ctrl/Cmd+Shift+Enter`: append a cell;
- `Ctrl/Cmd+Backspace`: delete the active cell;
- `Ctrl/Cmd+ArrowUp` or `Ctrl/Cmd+ArrowDown`: move the active cell.

All visible cells remain editable; changing focus does not collapse them or
replace highlighted source with plaintext. The eye action explicitly hides a
cell and persists that state in drafts and notebook files. The interface uses a
single dark notebook theme and keeps source/layout drafts in local browser
storage between reloads. Local draft recovery does not restore runtime state.

Monaco currently uses pinned jsDelivr assets. The default Pyodide runtime and
its startup packages are vendored with Kedi Notebook. Host execution remains
available through the fallback editor when Monaco cannot load.
