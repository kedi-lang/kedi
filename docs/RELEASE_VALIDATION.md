# Release Distribution Validation

## Clean Installation Gate

Run `make check-distribution` before publishing. Set `PYTHON=3.12` (or another
supported version) to select the interpreter. The check builds Kedi and its
parser from source distributions, installs their wheels in a temporary virtual
environment, and executes outside the checkout with Python isolated mode.
No editable installation, model credentials, or live model calls are needed.

The cross-platform equivalent, including Windows, is
`python scripts/check_distribution.py --python 3.11`. It does not require Bash
or Make. Wheel building requires a C compiler for the parser; ordinary users
installing published platform wheels do not need that compiler.

The smoke checks cover grammar AST nodes for display, budgets, native subagent
tasks, await, and task groups; CLI execution; bundled filesystem imports; and
optional-command help without installing the optional integrations. All three
installed console entrypoints are executed: `kedi`, `kedi-terminal-bench`, and
`kedi-lsp` (including an initialize/shutdown/exit protocol exchange).
`uv pip check` verifies the production dependency graph before any test-only
dependencies are installed. CI runs the same check on Python 3.11 through 3.14.
`make check-release` includes this gate.

## Supported Python CI

Kedi requires Python 3.11 or newer. The Linux compatibility matrix covers
Python 3.11 through 3.14, running the complete checkout test suite and clean
candidate-wheel installation checks. The quality-check matrix uses the same
versions. The former Python 3.10 job has been removed.

Clean installation and checkout tests use separate environments: the candidate
wheel must not inherit optional SDKs from development. Python 3.12 also runs
the full line/branch coverage gate. Notebook integration runs on Python 3.11;
the optional Harbor host integration requires Python 3.12.

Windows CI is currently disabled. The distribution driver still supports
Windows paths, but a passing Linux job does not establish native Windows
compatibility. Harbor's sandbox process-group and shell lifecycle tests remain
POSIX-only. Check CI results after pushing; workflow configuration alone is
not evidence of a passing run.

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

## Historical Local Verification, 2026-09-24

These results predate the Python 3.11 minimum and do not describe the current
support policy.

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
