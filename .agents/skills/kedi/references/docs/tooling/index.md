# Tools and Environments { #tooling }

Kedi ships one CLI, an LSP server, editor integrations, package commands, and
low-level browser/playground bridges.

## Command-Line Interface

```bash
kedi program.kedi
kedi -c '= hello'
kedi parse program.kedi
kedi program.kedi --test
kedi program.kedi --eval
```

The CLI loads `.env` and renders source-aware errors. Failed test cases exit
nonzero; a completed eval reports its score without enforcing a quality gate.

## Language Server

`kedi-lsp` provides diagnostics, semantic tokens, completion, hover,
definition/references, rename, formatting, symbols, signature help, and inlay
hints. It also understands Kedi query docstrings in Python.

## Editor Extensions

VS Code and Zed extensions recognize `.kedi`, package highlighting/snippets,
start `kedi-lsp`, and forward embedded Python regions to Python tooling.

## Browser Execution

`PlaygroundExecutor` moves embedded Python execution through a synchronous
worker bridge. `PyodideExecutor` is its browser-oriented alias/subclass.
`WebGPUAdapter` can route model requests to a browser-hosted model bridge.

## Package Tooling

`kedi install` installs a local `package.kedi`. `kedi add` supports explicit
credential-free GitHub sources. Named public registry installation is not yet
available; the local mock registry is for development only.

## Validation Workflows

Use parse-only checks first, deterministic tests second, evals for measured
quality, and optimization only after metrics are trustworthy.

## Terminal-Bench

The optional Harbor bridge runs Kedi as a custom agent against the official
Terminal-Bench 2.1 task containers and graders. It provides an immutable run
manifest, bounded terminal and process tools, non-interactive benchmark
approval, durable evidence, and Harbor-native resume. See
[Terminal-Bench 2.1](terminal-bench.md).

The frozen Pydantic AI and LangChain 89-task engineering runs, including cost,
latency, token, cache, request, and tool-call distributions, are reported in
[Terminal-Bench 2.1 Results](terminal-bench-results.md).

## In This Section

**Command Line**

- [Run and Parse](run-and-parse.md)
- [Test, Eval and Optimize](test-eval-and-optimize.md)
- [Package Commands](package-commands.md)
- [CLI Reference](../reference/cli.md)
- [Environment Variables](../reference/environment-variables.md)

**Editors**

- [Language Server](language-server.md)
- [VS Code](vscode.md)
- [Zed](zed.md)
- [Debugger and Inspector](debugger.md)

**Interactive Environments**

- [Terminal REPL](repl.md)
- [Local Notebook](notebook.md)
- [Notebook Runtimes](notebook-runtimes.md)
- [Notebook Files and Recovery](notebook-files.md)
- [Browser and Playground](browser-and-playground.md)

The REPL is a terminal session. Notebook is a separate local browser package
with cells and persistent executors. Playground is an integration/demo surface,
not a dependency of Notebook or a replacement for its file format.
