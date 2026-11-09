# VS Code

## Install the Extension

Install **Kedi-Lang** (`dyigitpolat.kedi-vscode`) and the Microsoft Python
extension. VS Code 1.85 or newer is required.

## File Recognition

The extension registers `.kedi` as language ID `kedi` with aliases `Kedi` and
`kedi`.

## Syntax Highlighting

Semantic tokens from `kedi-lsp` highlight Kedi structure. Embedded fenced and
backtick Python regions are forwarded to Python tooling. Kedi query docstrings
inside `.py` files can also receive experimental semantic tokens.

## Language Server Setup

On first activation, the extension provisions Python 3.12 and installs Kedi
with its parser and debugger into `~/.kedi/editor-venv`. Zed shares this same environment.
An absolute `KEDI_HOME` environment variable relocates the managed directory.
Initial setup requires internet access; subsequent starts reuse the installed
environment. Concurrent editor starts serialize installation rather than
modifying the environment together.

Use **Kedi: Select Python Interpreter** to choose the managed environment,
follow the Microsoft Python extension's selected interpreter, or enter a host
Python executable. Host environments are never modified automatically and must
already have Kedi and its dependencies installed.

An explicit `kedi.lsp.pythonPath` takes precedence over Python-extension
selection. Without either override, the managed environment is used. An
explicit `kedi.lsp.serverCommand` can instead launch a custom server when no
host interpreter is configured. Workspace virtual environments are not selected
implicitly. Settings and selected-interpreter changes restart the server.

The managed installer pins `kedi==0.1.4`, `tree-sitter-kedi==0.4.6`, and
`kedi-debugger==0.1.0`. Existing managed environments gain the debugger on their
next setup check. Host environments still require explicit debugger installation.
Those releases must be available on the package index before this automatic
installation can be distributed, and the Kedi wheel must contain the required
debugger hooks; an unavailable or incompatible package produces an explicit
setup error rather than falling back to an older runtime.

See [Debugger and Inspector](debugger.md) for launch configuration and controls.

## Completion and Hover

Kedi completion, hover, definitions, references, rename, outline, signature
help, inlay hints, formatting, and diagnostics come from `kedi-lsp`. Embedded
Python hover/definition/references come from Pylance/Pyright forwarding.

## Run, Test and Debug

The editor title's run menu has **Run Kedi File** and **Debug Kedi File** for a
saved `.kedi` file. Each `@test:` and `@eval:` block shows a **Run tests** or
**Run evals** code lens. The Command Palette has the same commands:

| Command | Runs |
| --- | --- |
| **Kedi: Run Kedi File** | `kedi <file>` |
| **Kedi: Run Tests in File** | `kedi <file> --test` |
| **Kedi: Run Evals in File** | `kedi <file> --eval` |
| **Kedi: Debug Kedi File** | the [Kedi debugger](debugger.md), without stopping on entry |

`--test` and `--eval` run every block in the file. Runs use the language
server's Python, start in the workspace folder, save the file first and require
a trusted workspace. A new run replaces the previous run's terminal.

## Diagnostics

Open **Output → Kedi Language Server** for client/server failures. Set
`kedi.lsp.trace.server` to `messages` or `verbose` for protocol tracing.

## Extension Settings

```json
{
  "kedi.lsp.usePythonExtension": false,
  "kedi.embeddedPython.enable": true,
  "kedi.embeddedKediInPython.enable": true,
  "kedi.embeddedKediInPython.experimentalSemanticTokens": true
}
```

To use an existing host environment, set `kedi.lsp.pythonPath` to its Python
executable. Alternatively, clear that setting and enable
`kedi.lsp.usePythonExtension` to follow **Python: Select Interpreter**.
Check the Output log for the selected executable; the active terminal and
editor need not use the same environment.

## Troubleshooting

Confirm the selected interpreter runs `python -m kedi.lsp.server`. Reload the
window after installing Kedi into a new environment. Pylance shadow documents
are stored in extension storage, outside the workspace, and may be deleted;
the extension regenerates them.
