# Package Manifests

`package.kedi` is a declarative package manifest written in Kedi directive
syntax. It is not an executable module.

## Complete Manifest

```kedi
> package: kedi_http:
  version: 1.2.0
  source: src/kedi_http
  python: python@3.11-3.14
  python_dependencies:
    httpx>=0.27
    pydantic>=2
```

The file must be named exactly `package.kedi` and contain exactly one package
directive. Comments are allowed; imports, exports, procedures, custom types,
Python blocks, variable initializations, assignments, and other executable
statements are rejected.

## Package Name

Package names use lowercase ASCII letters, digits, and underscores, start with a
letter, and match the installed registry directory:

```text
kedi_http    valid
http2        valid
KediHTTP     invalid
kedi-http    invalid
2http        invalid
```

The package name is also the root module name used by importers.

## Version

`version` is optional but, when present, must be a valid PEP 440 version:

```kedi
> package: reports:
  version: 2.0.0rc1
  source: src/reports
```

The manifest version is package metadata. It does not select a Python
environment or automatically enforce API compatibility between packages.

## Source Directory and `main.kedi`

`source` is required for installation. It is a literal relative directory
inside the package root and must contain `main.kedi`:

```text
package.kedi
src/
  kedi_http/
    main.kedi
    client.kedi
```

For `source: src/kedi_http`, `> import: kedi_http` loads `main.kedi` and
`> import: kedi_http/client` loads `client.kedi`.

Absolute paths, parent traversal, symlink escapes, sparse-checkout metacharacters
such as `*`, missing directories, and a source without `main.kedi` are rejected.

## Python Version Requirement

The optional `python` field accepts only:

```text
python@3.11
python@3.11-3.14
```

The range is closed and inclusive. The upper version cannot precede the lower
version. Installation checks the interpreter running the installer and refuses
an incompatible package.

This field expresses host compatibility, not a dependency to install.

## Python Dependencies

`python_dependencies` is an indented list of PEP 508 requirement strings:

```kedi
> package: reports:
  source: src/reports
  python_dependencies:
    pydantic>=2
    httpx[http2]>=0.27; python_version >= "3.11"
```

`kedi install` and `kedi add` validate these requirements and install them into
the Python interpreter running Kedi. A venv-local Kedi command installs into
that venv; a command running on another Python installs into that Python.
The installer uses `sys.executable`, not an unrelated `python` or `pip` on PATH.
It uses that interpreter's pip, falling back to `uv pip --python` if pip is absent.
At least one of pip or uv must be available.

PEP 508 extras, version constraints, and environment markers are passed to the
Python installer together, without forcing an upgrade of satisfied requirements.
Imports and manifest inspection never trigger installation. Use the bundled
`require` module for runtime checks of optional, undeclared features.

If dependency installation fails, the Kedi installation fails without replacing
the previous package. Kedi does not roll back changes the Python installer may
already have made to the environment. Externally managed system Python policies
are respected; use a writable virtual environment when system installation is
not permitted.

## Unknown and Duplicate Fields

Supported fields are `author`, `contact`, `license`, `version`, `source`,
`python`, and `python_dependencies`. See the [Manifest Reference](../reference/package-manifest.md)
for their exact validation rules. Unknown fields, duplicate scalar fields, multiple
dependency blocks, multiple package directives, and a missing package directive
are errors.

Manifest validation happens before installation and again when an installed
package is resolved. A modified or incomplete installed manifest is treated as
registry corruption rather than trusted package state.
