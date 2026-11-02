# Notebook Files and Recovery

## Notebook Versus Progress

The Save command offers two `.kedinb` modes:

- **Just notebook** stores the title, cell layout, kinds, and source. Opening it
  restores unexecuted drafts.
- **Save progress** also stores retained outputs/results and a strict,
  pickle-free `InteractiveSession` snapshot containing the current KediEnv.
  Opening it restores that logical session when the next cell runs.

Neither mode includes Secret Manager or process environment values. A progress
save also does not serialize the Python worker or claim that external side
effects are reproducible. If live Kedi state cannot be represented without
changing its semantics, Save progress fails instead of writing a partial
snapshot.

This exclusion is not automatic redaction. A secret manually copied into cell
source, a Kedi value or printed output may be saved there. Inspect notebook
content before sharing it, even when Secret Manager itself is excluded.

## Limits and Recovery

Opening or creating a document asks before replacing unsaved work. Notebook
files are limited to 5 MB and 1,000 cells. Individual cell source is limited to
1 MB, and retained inline output is limited to 200,000 characters with an
explicit truncation marker.

Notebook uses the default portable `InteractiveSession.dump()`/`load()` boundary
and rejects state it cannot restore without replay. The separate, explicitly trusted
Python snapshot mode for local sessions does not enable worker serialization or
executable snapshot uploads in Notebook.

Browser draft recovery restores source/layout, not the live worker. A progress
snapshot restores supported logical state, not installed packages, open
connections, external processes or filesystem side effects. Recreate those
dependencies deliberately; loading a snapshot does not rerun setup cells.


See [Local Notebook](notebook.md) for installation and cell editing.
