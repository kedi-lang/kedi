from __future__ import annotations

import json
import re
import sys
import tomllib
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ACTION_REF = re.compile(r"^\s*uses:\s*([^\s#]+)@([^\s#]+)", re.MULTILINE)
EXACT_ACTION_REF = re.compile(r"[0-9a-f]{40}")
REQUIREMENT_OPERATOR = re.compile(r"(?:===|==|~=|!=|<=|>=|<|>)")


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def _requirement_has_bound(requirement: str) -> bool:
    requirement = requirement.split(";", 1)[0].strip()
    return bool(REQUIREMENT_OPERATOR.search(requirement))


def _requirement_is_exact(requirement: str) -> bool:
    requirement = requirement.split(";", 1)[0].strip()
    return "==" in requirement and "!=" not in requirement


def _check_python_metadata(errors: list[str]) -> None:
    project_paths = (
        Path("pyproject.toml"),
        Path("notebook/pyproject.toml"),
        Path("kedi-otel/pyproject.toml"),
        Path("playground/pyproject.toml"),
        Path("tree-sitter-kedi/pyproject.toml"),
    )
    exact_dev_paths = {
        Path("pyproject.toml"),
        Path("notebook/pyproject.toml"),
        Path("playground/pyproject.toml"),
    }

    for relative_path in project_paths:
        data = _load_toml(ROOT / relative_path)
        for requirement in data.get("build-system", {}).get("requires", ()):
            if not _requirement_is_exact(requirement):
                errors.append(f"{relative_path}: build requirement is not exact: {requirement}")

        project = data.get("project", {})
        published_groups = [project.get("dependencies", ())]
        published_groups.extend(project.get("optional-dependencies", {}).values())
        for requirements in published_groups:
            for requirement in requirements:
                if not _requirement_has_bound(requirement):
                    errors.append(
                        f"{relative_path}: published requirement is unbounded: {requirement}"
                    )

        if relative_path in exact_dev_paths:
            for requirement in data.get("dependency-groups", {}).get("dev", ()):
                if not _requirement_is_exact(requirement):
                    errors.append(f"{relative_path}: dev requirement is not exact: {requirement}")

    docs_requirements = ROOT / "kedi-docs/requirements-docs.txt"
    for line in docs_requirements.read_text(encoding="utf-8").splitlines():
        requirement = line.strip()
        if (
            requirement
            and not requirement.startswith("#")
            and not _requirement_is_exact(requirement)
        ):
            errors.append(f"kedi-docs: documentation requirement is not exact: {requirement}")


def _check_javascript_metadata(errors: list[str]) -> None:
    for relative_path in (
        Path("kedi-vscode/package.json"),
        Path("tree-sitter-kedi/package.json"),
    ):
        data = json.loads((ROOT / relative_path).read_text(encoding="utf-8"))
        for group in ("dependencies", "devDependencies"):
            for name, version in data.get(group, {}).items():
                if version.startswith(("^", "~", ">", "<", "*")) or version == "latest":
                    errors.append(
                        f"{relative_path}: {group} dependency {name} is not exact: {version}"
                    )


def _check_actions(errors: list[str]) -> None:
    workflow_roots = (
        ROOT / ".github/workflows",
        ROOT / "kedi-docs/.github/workflows",
        ROOT / "playground/.github/workflows",
        ROOT / "tree-sitter-kedi/.github/workflows",
    )
    for workflow_root in workflow_roots:
        for path in workflow_root.glob("*.y*ml"):
            text = path.read_text(encoding="utf-8")
            for match in ACTION_REF.finditer(text):
                action, ref = match.groups()
                if action.startswith("./") or action.startswith("docker://"):
                    continue
                if not EXACT_ACTION_REF.fullmatch(ref):
                    errors.append(f"{path.relative_to(ROOT)}: mutable action ref {action}@{ref}")


def _check_lockfiles(errors: list[str]) -> None:
    required = (
        Path("uv.lock"),
        Path("notebook/uv.lock"),
        Path("kedi-otel/uv.lock"),
        Path("playground/uv.lock"),
        Path("tree-sitter-kedi/uv.lock"),
        Path("kedi-vscode/package-lock.json"),
        Path("tree-sitter-kedi/package-lock.json"),
        Path("kedi-zed/Cargo.lock"),
    )
    for relative_path in required:
        if not (ROOT / relative_path).is_file():
            errors.append(f"missing lockfile: {relative_path}")

    if (ROOT / "requirements.txt").exists():
        errors.append("requirements.txt duplicates pyproject.toml and must not be restored")

    playground = _load_toml(ROOT / "playground/pyproject.toml")
    kedi_source = playground.get("tool", {}).get("uv", {}).get("sources", {}).get("kedi", {})
    revision = kedi_source.get("rev")
    if not isinstance(revision, str) or not EXACT_ACTION_REF.fullmatch(revision):
        errors.append("playground: Kedi source must use an exact 40-character git revision")
    if "branch" in kedi_source or "tag" in kedi_source:
        errors.append("playground: Kedi source cannot use a moving branch or tag")


def _check_pyodide_assets(errors: list[str]) -> None:
    vendor = ROOT / "notebook/src/kedi_notebook/static/vendor/pyodide"
    lock = json.loads((vendor / "pyodide-lock.json").read_text(encoding="utf-8"))
    packages = lock.get("packages", {})
    for package_name in (
        "annotated-types",
        "micropip",
        "pydantic",
        "pydantic-core",
        "typing-extensions",
        "typing-inspection",
    ):
        package = packages.get(package_name)
        if not isinstance(package, dict):
            errors.append(f"Pyodide lock is missing {package_name}")
            continue
        filename = package.get("file_name")
        if not isinstance(filename, str) or not (vendor / filename).is_file():
            errors.append(f"Pyodide asset is missing for {package_name}: {filename}")

    worker = (ROOT / "notebook/src/kedi_notebook/static/pyodide-worker.js").read_text(
        encoding="utf-8"
    )
    logfire_function = worker.split("async function installLogfire", 1)[-1].split(
        "globalThis.kediStreamOutput", 1
    )[0]
    install_match = re.search(r"micropip\.install\(\[(.*?)\]\)", logfire_function, re.DOTALL)
    if install_match is None:
        errors.append("Pyodide Logfire package list could not be found")
        return
    for requirement in re.findall(r'"([^"]+)"', install_match.group(1)):
        if not _requirement_is_exact(requirement):
            errors.append(f"Pyodide optional package is not exact: {requirement}")


def main() -> int:
    errors: list[str] = []
    _check_python_metadata(errors)
    _check_javascript_metadata(errors)
    _check_actions(errors)
    _check_lockfiles(errors)
    _check_pyodide_assets(errors)
    if errors:
        print("Dependency pin validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("Dependency pins and vendored runtime assets are consistent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
