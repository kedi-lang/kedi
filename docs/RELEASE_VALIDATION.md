# Release Distribution Validation

## Clean Installation Gate

Run `make check-distribution` before publishing. Set `PYTHON=3.12` (or another
supported version) to select the interpreter. The check builds Kedi and its
parser from source distributions, installs their wheels in a temporary virtual
environment, and executes outside the checkout with Python isolated mode.
No editable installation, model credentials, or live model calls are needed.

The cross-platform equivalent, including Windows, is
`python scripts/check_distribution.py --python 3.10`. It does not require Bash
or Make. Wheel building requires a C compiler for the parser; ordinary users
installing published platform wheels do not need that compiler.

The smoke checks cover grammar AST nodes for display, budgets, native subagent
tasks, await, and task groups; CLI execution; bundled filesystem imports; and
optional-command help without installing the optional integrations. All three
installed console entrypoints are executed: `kedi`, `kedi-terminal-bench`, and
`kedi-lsp` (including an initialize/shutdown/exit protocol exchange).
`uv pip check` verifies the production dependency graph before any test-only
dependencies are installed. CI runs the same check on Python 3.10 through 3.14.
`make check-release` includes this gate.

## Linux and Windows CI

The `platform-python310` matrix expands to two separate jobs, on `ubuntu-latest`
and `windows-latest`, both using Python 3.10. Each job first builds and installs
candidate wheels in a clean environment and verifies entrypoints. It then
installs the locked development dependencies and runs the complete checkout
test suite with cassette recording disabled. These are separate environments:
the clean installation must not inherit optional SDKs from development.

Neither job uses `continue-on-error`. Harbor's sandbox process-group and shell
lifecycle tests are POSIX-only and are explicitly skipped on native Windows;
they still run on Linux. Portable Harbor metadata, routing, and model tests
remain enabled on Windows. This does not claim native Windows support for the
Linux Harbor sandbox runner.
JUnit results are uploaded for diagnosis. The existing Linux compatibility
matrix covers Python 3.11 through 3.14; Python 3.10 is covered by the new job.
Platform jobs use a pinned pytest-timeout plugin to limit individual tests to
150 seconds and the test session to 15 minutes. Timeout reports include thread
stack traces; large response fixtures use short parameter IDs to keep CI logs
and JUnit reports bounded.
Configuring a job is not proof it has passed: native Windows and Linux results
must be checked after the changes and submodule revisions are pushed.

The initial run `36058616061` passed the complete Linux Python 3.10 job. Windows
passed clean installation, dependency validation, all entrypoints, and the LSP
lifecycle, then stopped during test collection because a POSIX Harbor test
referenced `SIGKILL`. The Windows test suite must be rerun after the platform
guards are applied; the successful installation alone is not a suite pass.

## Parser Version Blocker

The published `tree-sitter-kedi==0.4.0` does not understand the current syntax.
An isolated installation of the Kedi 0.4.0 candidate reproduced parse failures
that the checkout's editable parser had hidden. The updated parser is version
0.4.1; Kedi's dependency pin and both editors' shared bootstrap now require it.

Passing the candidate-wheel gate does not mean the packages are available on
PyPI. Publish the parser first, then Kedi, and only then release editor packages
that automatically install those versions. Publish the notebook package after
its Kedi dependency is available. Verify a fresh index-only installation after
publication; do not replace an already published parser version with new code.

This gate is not a substitute for live model tests, notebook browser tests,
editor interaction tests, or platform-specific extension packaging checks.
Those remain separate release checks.

## Local Verification, 2026-09-24

- Clean candidate-wheel checks passed on Python 3.10, 3.11, 3.12, and 3.14
  (macOS ARM64). These are installation and behavioral smoke checks, not full
  regression-suite reruns for each interpreter.
- Shared editor bootstrap: 10 tests passed.
- The cross-platform driver and real console-entrypoint/LSP checks passed
  locally with Python 3.10. Three driver regression tests passed, including
  Windows path selection and failure propagation/cleanup. Native platform CI
  results are still pending; simulated path tests are not Windows execution.
- Task-group and task-group boundary regression suite: 46 tests passed.
- Dependency-pin validation, bootstrap artifact synchronization, and diff
  whitespace checks passed.
- `make check` passed. `ty` reported two unused-ignore warnings in existing
  source; basedpyright reported no errors or warnings.
- No packages were published. Index-only installation of the new parser pin
  remains blocked until parser 0.4.1 is published. Python 3.13 clean installation
  and live model/editor/notebook smoke checks have not been rerun in this pass.
