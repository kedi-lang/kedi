# Terminal REPL

The terminal submits complete fragments to an incremental runtime. For embedding, persistence restrictions, and failures, see [Incremental Execution](../runtime/interactive-execution.md).

## Terminal REPL

Start the terminal frontend without a source file:

```console
$ kedi --idle
 /\_/\
( o.o )
 > ^ <
Kedi 0.1.2 on darwin
Type "help" for interactive help, ":show" to inspect a value, ":multiline" for a multiline fragment, ":dump" to save, or ":exit" to leave.
+++ [base: int] = `40`
+++ @add_two() -> int:
...     = `base + 2`
...
+++ :show `add_two()`
42
+++
```

`+++` is the primary prompt. `...` indicates that the current fragment needs
more input. The REPL enters continuation mode for:

- a block header ending in `:`;
- an open parenthesis, bracket, or brace;
- an open inline-Python expression or Python fence;
- an explicit line continuation ending in `\`.

Press Tab to insert indentation at the continuation prompt. Submit an empty
continuation line to execute the buffered fragment exactly once. A complete
single-line fragment executes immediately.

### Explicit Multiline Input

Use `:multiline` when you want to compose a complete fragment before any part
of it executes:

```console
+++ :multiline
... [values: list[int]] = `[1, 2, 3]`
... > loop [value]: `values`:
...   `print(value)`
...
1
2
3
+++
```

Enter creates a new line. Press Enter again on the new empty line to submit the
fragment. Kedi keeps the editor open while the source has an open block,
delimiter, inline expression, or Python fence. `Alt+Enter` forces submission,
which is useful when you want the parser to diagnose incomplete source.

The multiline editor is one-shot: execution or an error returns to the normal
`+++` prompt. Terminal meta commands are recognized only at the primary prompt;
text such as `:exit` inside the editor is treated as Kedi source. The complete
fragment is executed once, so earlier lines cannot partially change session
state before submission. Terminal copy and paste preserves pasted newlines,
indentation, and blank lines without submitting the fragment. Press Enter twice
after pasting to execute it.

### Syntax Highlighting

Live Kedi and embedded-Python highlighting is enabled by default:

```bash
kedi --idle
```

Highlighting changes terminal presentation only. It does not start the Kedi
language server or add diagnostics, completion, or hover. Very large fragments
fall back to plain input to keep editing responsive. Highlighted input uses the
same `~/.kedi_history` file as ordinary and multiline input.

Use `kedi --idle --no-color` to disable highlighting and REPL error colors.
A nonempty `NO_COLOR` environment variable also disables them. Redirected
input or output uses plain line input rather than the interactive editor.
`--highlight` remains accepted for existing commands, but is no longer necessary.

### Inspecting Values

`:show <expression>` is an IDLE alias for native `> show:`. Both display once
and continue; neither declares output captures. Regular `.kedi` files use the
native form. For example:

```console
+++ :show <name>
rendered value
+++ :show `items[0]`
42
```

Both forms use print-like display. Interactive root returns are rejected before
execution; use show instead. Returns in procedures remain valid. Snapshot
fragment sources use the canonical `> show:` form.

### Commands, History, and Exit

The terminal understands these commands:

| Input | Behavior |
| --- | --- |
| `help` or `help()` | Show concise interactive help |
| `:multiline` | Compose and submit one complete multiline fragment |
| `:show <expression>` | Evaluate and print one value |
| `:dump` | Save the complete restorable session and print its resume command |
| `:exit` | Close the session |
| `Ctrl+C` | Exit silently, including during active execution |
| `Ctrl+D` | Close the session at the input prompt |

Python's `exit()` and `quit()` have no special terminal meaning. `:exit` is the
only textual exit command.

The first `:dump` writes an atomic snapshot under `~/.kedi/sessions`; later
dumps in the same REPL update the same file. Kedi prints the exact resume
command after every successful dump:

```console
To resume session, run -- kedi --idle --load <session_path>
```

Use `--record` to dump automatically before `:exit`, `Ctrl+C`, `Ctrl+D`, or a
`SystemExit` raised by inline Python. Recording happens before session resources
close and preserves the `SystemExit` status. `--load` restores a snapshot and
continues recording changes to that same path:

```bash
kedi --idle --record
kedi --idle --load ~/.kedi/sessions/idle-20260826T120000-ab12cd34.kedi-state
```

For supported local Python objects, install `kedi[snapshots]` and select
`--snapshot-format python`:

```bash
kedi --idle --snapshot-format python --record
kedi --idle --load PATH --trust-snapshot
```

Python snapshots can execute arbitrary code and include credentials. The printed
resume command includes the trust option and warning; never trust a file only
because it has a matching checksum. A loaded Python session retains Python mode
for recording. An explicit `--snapshot-format portable` requests conversion and
fails without replacing the existing file if the state is not portable.
Python versions and the versions of packages the saved graph depends on must match; see [Python object snapshots](../runtime/interactive-execution.md#python-object-snapshots)
for environment, executor, history/artifact and live-resource limits.

Readline history is stored in `~/.kedi_history`. Set `KEDI_HISTORY` to use a
different path:

```bash
KEDI_HISTORY="$HOME/.local/state/kedi/history" kedi --idle
```

Adapter selection remains available:

```bash
kedi --idle --adapter pydantic --adapter-model openai:gpt-6-luna
```

Interactive mode does not accept a source file, `-c/--command`, program
arguments, `--parse`, `--test`, `--eval`, or `--optimize`.
`--record`, `--load`, `--snapshot-format`, `--highlight`, and `--no-color` require
`--idle`. `--trust-snapshot` requires `--idle --load`.
