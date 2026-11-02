# Notebook Runtimes

## Execution Ownership

The Kedi compiler, interactive session and model adapters run in the local
server. The runtime menu selects where embedded Python executes, not a place
to move all Kedi execution or inference. Browser mode still needs that server.

## Browser Python

Browser execution through Pyodide 3.14 is selected by default. Embedded Python
runs in one persistent Web Worker, so Python values created by an earlier cell
remain available to later cells.

The worker starts loading when the notebook page opens. Runtime startup is
therefore paid during initial page load rather than deferred to the first cell.
The ready worker, imported modules, and browser-installed packages are reused
until the runtime is reset or the page is closed. Core Pyodide, Python standard
library, Micropip, and Pydantic files are shipped in the notebook wheel; this
startup does not require a Pyodide CDN. Packages installed later can still need
network access.

## Host Python

The server discovers compatible host Python installations. Select one as the
base for a managed environment; existing packages from the base environment
are not a promise of availability inside the new environment. Add or prioritize
an exact executable while starting the server:

```bash
kedi notebook --python /opt/homebrew/bin/python3.11
kedi notebook --python ~/.pyenv/versions/3.12.4/bin/python --port 8899
```

`--python` is repeatable. The selected executable is a base interpreter, not an
environment that Kedi modifies directly. On first use, the notebook creates a
project-specific virtual environment under `~/.kedi/notebook/venvs`, installs
the active Kedi checkout and its dependencies, and starts the persistent worker
from that environment. Its stable `kedi-notebook-py...` name is derived from the
base interpreter and working directory, so later server runs reuse it. Set
`KEDI_NOTEBOOK_ENV_HOME` to choose another environment root. Host execution is
not sandboxed and has the notebook server process's filesystem permissions.

After selecting a host runtime, use the package action beside the runtime menu
to inspect installed distributions or install newline-separated Python
requirements. Pip output streams in the dialog. These packages persist in the
managed environment and are available to both Kedi cells and host terminal
cells. The selected base interpreter remains unchanged.

The selector locks once a runtime session is created, including by a terminal
cell. Reset the runtime before choosing another interpreter.

## Interrupt and Recovery

Use the square interrupt action while a Kedi or terminal cell is running. The
active worker is terminated and replaced, so the cell remains editable and
rerunnable while its previous live state is discarded. The session's in-flight
Pydantic AI or LangChain model and async tool work receives cancellation too;
other sessions retain their own work. Completed effects are not rolled back,
and arbitrary synchronous tool code in a server thread cannot be forcibly
stopped by cancelling an async call.

Embedded-Python bridge
operations and terminal execution have 120-second timeouts; this is not a
universal 120-second deadline for a complete model-backed cell. Sessions with
no activity expire after 30 minutes.


See [Local Notebook](notebook.md) for installation and cell editing.
