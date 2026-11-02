# Filesystem Module

Import the tools you need or apply a profile. Workspace containment, secret-file opt-in, risk classification, and approval are separate checks; ordinary Python execution is not sandboxed by this module.

## `filesystem`

The `filesystem` module exports:

| Name | Behavior |
| --- | --- |
| `read_text_file` | Read one UTF-8 file |
| `write_text_file` | Replace/create one UTF-8 file |
| `apply_patch` | Add, append, or exact-once replace text |
| `path_exists` | Check file or directory existence |
| `list_directory` | Return sorted direct child names |
| `create_directory` | Create a directory tree |
| `remove_file` | Delete one file |
| `remove_directory` | Delete a directory tree |
| `get_file_info` | Return basic path metadata |
| `filesystem` | Read/write agent profile |
| `readonlyfs` | Read-only agent profile |

```kedi
> import: filesystem

@load_readme() -> str:
  = `read_text_file("README.md")`
```

## Workspace Policy

Paths resolve against the process working directory captured when the module
initializes, not a later working-directory change. `KEDI_WORKSPACE_POLICY`
selects containment at access time:

| Value | Behavior |
| --- | --- |
| Unset, empty, or `strict` | Resolved path must remain under the captured root, including after symlink resolution |
| `none` | Disable workspace containment; absolute and outside-root paths are permitted subject to other checks and OS permissions |
| Any other value | Configuration error rather than fallback |

`none` does not disable secret-file checks, delete protection, tool approval, or
read/list bounds. Avoid selecting it for untrusted agent workloads merely to
make a failing path work. Ordinary embedded Python can bypass these helpers;
host isolation is still necessary for untrusted programs.

## Bounded Reads

`read_text_file(file_path, secret_files=False, offset=0, max_chars=-1)` returns
UTF-8 text. `offset` is a nonnegative character offset, not a byte offset.
`max_chars=-1` permits an unbounded read only for files up to 1,000,000 bytes.
For larger files, pass a positive character limit. Zero and values below -1
are invalid. Direct calls return `str`; agent calls stream internally into
artifact admission without changing the declared return schema.

```kedi
> import: filesystem:
  read_text_file

# Requires an existing application.log in the workspace.
[head] = `read_text_file("application.log", max_chars=2000)`
[next_page] = `read_text_file("application.log", offset=2000, max_chars=2000)`
= <head>\n<next_page>
```

This is two bounded pages, not a claim that the file ends there. Directory
listing remains capped at 1,000 direct entries and returns alphabetically
sorted names; it has no pagination parameter.

## Secrets and Approval

Files named `.env` or beginning `.env.` are treated as secrets.
`read_text_file(path)` refuses them unless `secret_files=True`. In agent tool
use, that argument changes the call's risk classification so approval can be
required. Secret opt-in is not an authorization boundary by itself; the host
approval policy still decides.

## Exact Patches

`apply_patch("add", ...)` refuses an existing file. `update` requires a nonempty
`old_text` that occurs exactly once. `append` adds content. Prefer `apply_patch`
over whole-file writes for auditable agent edits.

When invoked as an agent tool, expected edit conflicts (an existing add target,
empty/missing/ambiguous `old_text`, or an invalid operation) return an explicit
`Patch rejected; no changes made` result. The agent can inspect the current file
and correct its edit without exhausting framework validation retries. Direct
programmatic calls still raise for these conflicts. Permission, path-boundary,
and unexpected I/O errors still propagate; replacement remains exact-once, never
fuzzy.

## Profiles and Destructive Operations

`readonlyfs` registers reads and metadata only. `filesystem` additionally
registers writes, patching, directory creation, and the destructive
`remove_file` and `remove_directory` tools. Removal cannot target the filesystem
captured workspace root, filesystem anchor, or user's home directory, including
under `KEDI_WORKSPACE_POLICY=none`. Agent approval still applies to destructive
calls. Direct Python/Kedi procedure calls retain path checks but are not an
agent tool approval loop.

## Callable Contracts

| Callable | Arguments and native result |
| --- | --- |
| `write_text_file` | `file_path, content` -> resolved path string; creates parents |
| `apply_patch` | `operation, file_path, content="", old_text="", new_text=""` -> edit summary; add/update/append only |
| `path_exists` | `path` -> bool |
| `list_directory` | `dir_path` -> sorted `list[str]` |
| `create_directory` | `dir_path` -> resolved path string; creates parents |
| `remove_file` | `file_path` -> deleted path string; missing target raises |
| `remove_directory` | `dir_path` -> deleted path string; recursively removes contents |
| `get_file_info` | `path` -> dict with `path`, `name`, `is_file`, `is_dir`, `size` (bytes), `modified_time` (timestamp) |
