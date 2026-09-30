# Release notes from a change list

Keep internal maintenance out of customer-facing notes without maintaining a
keyword list. This example first classifies each change, then writes a note
only for a user-visible change.

## Run

Complete the [shared setup](../README.md#setup), then:

```bash
cd examples/release_notes
kedi main.kedi
```

Expect two notes: one about preserving names in CSV exports, and one about
resumable uploads. The second must retain **Linux only** and **100 MB**. The
internal fixture rename should not appear.

## Control flow

`> loop [change]:` visits changes sequentially. The `> if:` header is a natural
language claim, so the selected model evaluates it. A trailing-colon Python
condition such as `> if: \`enabled\`:` would instead be deterministic.
The procedure's system instruction scopes the judgment to the supplied text;
it is not asking the model to verify that a real deployment happened.

Only a true claim enters the `>>` block. `:=` updates the outer list with the
generated note, preserving input order. The final return waits for each required
value and returns the list.

For this input, the intended path has three classification calls and two
generation calls, before any provider/schema retries. This makes branching
visible, but it is not the cheapest way to summarize three lines: a single
template would be cheaper when per-item routing is unnecessary. Use this
pattern when selected branches have genuinely different work to do.
