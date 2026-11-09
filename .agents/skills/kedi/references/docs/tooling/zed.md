# Zed

## Install the Extension

Open **Zed → Extensions**, search for **Kedi**, and install it. For local
development choose **Install Dev Extension** and select the `kedi-zed`
directory, not the grammar repository.

## File Recognition

The bundled language config registers `.kedi` as **Kedi**, with `#` line
comments, `###` block comments, bracket pairs, and Kedi indentation behavior.

## Tree-sitter Grammar

The extension pins `tree-sitter-kedi` to the revision in `extension.toml`.
Registry installs build/use the packaged extension. Local dev installs require
Rust via `rustup` and the declared `wasm32-wasip1` target.

## Syntax Highlighting

Tree-sitter queries provide highlights, outlines, brackets, indentation, and
Python injection. Recommended:

```json
{
  "languages": {
    "Kedi": {
      "formatter": "language_server",
      "format_on_save": "on",
      "semantic_tokens": "combined"
    }
  }
}
```

## Language Server Setup

On first language-server activation, the extension provisions Python 3.12 and
installs Kedi with its parser and debugger into `~/.kedi/editor-venv`, shared with VS Code.
An absolute `KEDI_HOME` environment variable relocates the managed directory.
The first setup requires internet access; subsequent starts reuse it. Parallel
editor starts share an installation lock.

To use a host Python instead, configure its executable explicitly:

```json
{
  "lsp": {
    "kedi-lsp": {
      "settings": {
        "python_path": "/path/to/python"
      }
    }
  }
}
```

This interpreter is used for Kedi language services and Python virtualizers.
It must already contain Kedi and its dependencies; the extension never installs
packages into a selected host environment. The extension API does not expose a
Python toolchain-selection callback, so Zed's selected project toolchain does
not implicitly override this setting. Restart language servers after changing it.
Advanced users can still configure `lsp.kedi-lsp.binary.path` and `arguments`
for a custom server; specify `settings.python_path` as well when that server
is a wrapper whose Python interpreter cannot be inferred.

The managed installer pins `kedi==0.1.4`, `tree-sitter-kedi==0.4.6`, and
`kedi-debugger==0.1.0`. Existing managed environments gain the debugger on their
next setup check. Host environments still require explicit debugger installation.
These releases must be published before distributing automatic installation,
and the Kedi wheel must contain the required debugger hooks. Missing or
incompatible packages produce an explicit setup error, not an older-runtime fallback.

See [Debugger and Inspector](debugger.md) for launch configuration and controls.

The extension also starts an embedded-Python proxy and auto-installs Pyright
through Zed's npm support. Python query docstrings use a separate virtualizer
server.

## Tasks and Commands

A run button appears beside the first item of a `.kedi` file and beside each
`@test:` and `@eval:` block. The same tasks are in **task: spawn**:

| Task | Command |
| --- | --- |
| `kedi: run <file>` | `kedi <file>` |
| `kedi: test <file>` | `kedi <file> --test` |
| `kedi: eval <file>` | `kedi <file> --eval` |

`--test` and `--eval` run every block in the file. The tasks run in Zed's
terminal from the worktree root. They use `kedi` from the terminal's `PATH`,
such as an activated project environment, and fall back to the shared managed
environment. To change a command, add a task with the same tag (`kedi-file`,
`kedi-test` or `kedi-eval`) to `.zed/tasks.json`.

## Extension Updates

Registry updates follow the published extension version. A dev extension uses
the selected local checkout and may leave ignored WASM/grammar build artifacts.

## Troubleshooting

If **Kedi** is absent from language selection, verify the extension installed
and `languages/kedi/config.toml` is present. If files are recognized but LSP
features fail, configure a Python environment that can import `kedi`.

The manifest requests broad `process:exec` because it may launch a configured
binary, managed-runtime installer, Python, Node, and Pyright proxy. Users
may narrow Zed's granted capability locally.
