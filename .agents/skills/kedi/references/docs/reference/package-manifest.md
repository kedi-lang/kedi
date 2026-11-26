# Package Manifest Reference

## Complete Shape

```kedi
> package: reports:
  author: Kedi Team
  contact: maintainers@example.com
  version: 1.2.0
  source: src/reports
  python: python@3.11-3.14
  license: MIT
  python_dependencies:
    pydantic>=2
    httpx[http2]>=0.27
```

The file must be named exactly `package.kedi`, contain exactly one package
directive, and contain no imports, exports, procedures, types, Python, values,
or executable statements. Comments are allowed.

## Fields

| Field | Required | Validation |
| --- | --- | --- |
| package name | yes | `[a-z][a-z0-9_]*` |
| `source` | installable package: yes | literal relative directory containing `main.kedi` |
| `version` | no | valid PEP 440 version |
| `python` | no | fixed version or inclusive closed range |
| `python_dependencies` | no | one list of valid PEP 508 requirements |
| `author` | no | metadata text |
| `contact` | no | metadata text |
| `license` | no | metadata text |
| `adapters` | no | one block of adapter entries (below) |

Unknown fields, duplicate scalar fields, a second dependency block, or a second
package directive fail validation.

## Adapters

```kedi
> package: pi:
  source: src/pi
  adapters:
    pi:
      kind: agent
      command: node bridge.mjs
    pi_local:
      kind: adapter
      class: kedi_pi:LocalAdapter
```

| Entry field | Required | Validation |
| --- | --- | --- |
| shortname | yes | `[a-z][a-z0-9_]*`, not a built-in adapter's (`pydantic`, `langchain`, `claude`, `codex`, `acp`, `a2a`, `rpc`, `dspy`) |
| `kind` | yes | `agent` (selected with `> agent:`, an agent harness) or `adapter` (selected with `> adapter:`, an agent framework) |
| `command` | one of `command` or `class` | a shell-style command line; arguments naming regular files of `source` become absolute paths |
| `class` | one of `command` or `class` | `module:Class`, a Python adapter class built with `Class(model=...)` |

Unknown entry fields, duplicate shortnames or fields, and a second adapters
block fail validation. Nothing is imported or launched when a package is
installed or imported; Kedi builds the adapter when a program selects it, and
refuses it if its kind differs from the declared one.

## Python Requirement

Accepted forms:

```text
python@3.11
python@3.11-3.14
```

The range includes both endpoints and its upper bound cannot precede its lower
bound. Installation compares it with the interpreter running `kedi install`.
This field does not create or select a virtual environment.

## Source Tree

For `source: src/reports`, installation requires:

```text
package.kedi
src/
  reports/
    main.kedi
```

The package root import loads `main.kedi`; nested imports load sibling `.kedi`
files. Absolute paths, `..`, symlink escapes, sparse-checkout patterns, special
files, missing roots, and oversized source trees are rejected.

## Dependencies

`kedi install` and `kedi add` install `python_dependencies` into `sys.executable`,
the interpreter running Kedi. Pip is used when available in that interpreter;
otherwise `uv pip --python` targets the same interpreter. Requirement extras,
version constraints and environment markers retain their PEP 508 meaning.
Reading a manifest or importing a package does not install Python dependencies.

## Installation

```bash
kedi install
kedi install path/to/package.kedi
kedi add package_name
kedi add git+https://github.com/owner/repository.git
```

Installed content and `.kedi-install.json` receipts live below the Kedi home
registry. Named registry installs use an exact verified source commit. Identity
and integrity are not safety. Imported packages may execute Python with
the host process's authority.
