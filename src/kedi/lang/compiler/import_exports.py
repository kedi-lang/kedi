"""Static export validation and publishing of imported bindings."""

from __future__ import annotations

import ast as py_ast
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from kedi.agent_profile import AgentProfile
from kedi.agent_profile.definitions import ProfileDefinition
from kedi.core import KediRuntime
from kedi.lang.ast import (
    ModuleImport,
    OutputSegment,
    Program,
    RawInvokeStmt,
    TemplateBlock,
    TemplateLine,
    VariableInitBlockStmt,
    VariableInitStmt,
)

if TYPE_CHECKING:
    from kedi.lang.compiler.program_compiler import ProgramCompiler


@dataclass(frozen=True)
class _StaticExportSurface:
    """Names provably exported without executing a module."""

    names: frozenset[str]
    complete: bool


def _validate_selected_import_before_execution(
    self: ProgramCompiler,
    module_import: ModuleImport,
    module_path: Path,
    imported_prog: Program,
) -> None:
    """Reject provably absent selective imports before module initialization."""
    if not module_import.names:
        return

    surface = self._static_export_surface(module_path, imported_prog, visiting=set())
    missing = [name for name in module_import.names if name not in surface.names]
    if missing and surface.complete:
        rendered = ", ".join(repr(name) for name in missing)
        raise ImportError(f"Kedi module {module_import.module!r} does not export {rendered}")


def _static_export_surface(
    self: ProgramCompiler,
    module_path: Path,
    prog: Program,
    *,
    visiting: set[Path],
) -> _StaticExportSurface:
    """Inventory an export surface without running embedded code.

    Explicit export directives are authoritative even when their value is
    created dynamically. ``export: *`` additionally receives statically
    knowable local and imported names. A genuinely dynamic prelude keeps
    the surface open, preserving the historical runtime fallback only for
    cases that cannot be proven before initialization.
    """
    path = module_path.resolve()
    if path in visiting:
        return _StaticExportSurface(frozenset(), complete=False)

    next_visiting = {*visiting, path}
    local_names, complete = self._static_local_runtime_names(prog)
    imported_names: set[str] = set()
    for nested_import in prog.imports:
        nested_path = self._resolve_import_path(prog, nested_import)
        nested_path, nested_prog = self.module_loader.program_for(
            nested_path,
            self._load_imported_program,
        )
        nested_surface = self._static_export_surface(
            nested_path,
            nested_prog,
            visiting=next_visiting,
        )
        complete = complete and nested_surface.complete
        if nested_import.names:
            imported_names.update(
                name for name in nested_import.names if name in nested_surface.names
            )
        else:
            imported_names.update(nested_surface.names)

    names: set[str] = set()
    for export in prog.exports:
        names.update(export.names)
        if export.export_all:
            names.update(
                name for name in (*local_names, *imported_names) if self._is_public_name(name)
            )

    # A module with only explicit exports is fully known even if its
    # prelude dynamically creates values: the explicit list is its public
    # contract. Dynamic names only affect an export-star surface.
    if not any(export.export_all for export in prog.exports):
        complete = True
    return _StaticExportSurface(frozenset(names), complete)


def _static_local_runtime_names(self: ProgramCompiler, prog: Program) -> tuple[set[str], bool]:
    names = {proc.name for proc in prog.procedures}
    names.update(typedef.name for typedef in prog.type_defs)
    names.update(prog.profile_defs)
    complete = True

    if prog.prelude is not None:
        prelude_names, prelude_complete = self._static_python_names(prog.prelude)
        names.update(prelude_names)
        complete = prelude_complete

    for stmt in prog.toplevel_body:
        if isinstance(stmt, (VariableInitStmt, VariableInitBlockStmt, RawInvokeStmt)):
            names.add(stmt.name)
        elif isinstance(stmt, TemplateBlock):
            names.update(self._template_output_names(stmt.lines))
        elif isinstance(stmt, TemplateLine):
            names.update(self._template_output_names([stmt]))
    return names, complete


def _static_python_names(self: ProgramCompiler, code: str) -> tuple[set[str], bool]:
    try:
        module = py_ast.parse(code)
    except SyntaxError:
        return set(), False

    names: set[str] = set()
    dynamic = False
    for stmt in module.body:
        if isinstance(stmt, (py_ast.FunctionDef, py_ast.AsyncFunctionDef, py_ast.ClassDef)):
            names.add(stmt.name)
            continue
        if isinstance(stmt, py_ast.Import):
            names.update(alias.asname or alias.name.partition(".")[0] for alias in stmt.names)
            continue
        if isinstance(stmt, py_ast.ImportFrom):
            if any(alias.name == "*" for alias in stmt.names):
                dynamic = True
            else:
                names.update(alias.asname or alias.name for alias in stmt.names)
            continue
        names.update(
            node.id
            for node in py_ast.walk(stmt)
            if isinstance(node, py_ast.Name) and isinstance(node.ctx, py_ast.Store)
        )
        dynamic = dynamic or any(
            isinstance(node, py_ast.Call)
            and isinstance(node.func, py_ast.Name)
            and node.func.id == "exec"
            for node in py_ast.walk(stmt)
        )
    return names, not dynamic


def _template_output_names(lines: list[TemplateLine]) -> set[str]:
    return {
        segment.name
        for line in lines
        for segment in line.segments
        if isinstance(segment, OutputSegment)
    }


def _merge_exports(
    self: ProgramCompiler,
    module_import: ModuleImport,
    module_path: Path,
    imported_prog: Program,
    imported_rt: KediRuntime,
    *,
    target_env: dict[str, Any] | None = None,
) -> None:
    from kedi.lang.compiler.environment import assert_not_cli_args_identifier

    export_names = self._export_names(imported_prog, imported_rt)
    names = self._selected_import_names(module_import, export_names)
    for name in names:
        assert_not_cli_args_identifier(name)
        kind, value = self._resolve_export(name, imported_rt, module_path)
        self._remove_imported_binding(name)
        if target_env is not None:
            target_env.pop(name, None)
        if kind == "procedure":
            self.rt._procedures[name] = self._wrap_imported_procedure(name, value, imported_rt)
        elif kind == "type":
            self.rt._type_env_stack[0][name] = value
        elif kind == "profile":
            self.rt._profile_registry[name] = self._bind_imported_profile(value, imported_rt)
        else:
            self.rt._import_env[name] = value
            if target_env is not None:
                target_env[name] = value


def _selected_import_names(
    self: ProgramCompiler,
    module_import: ModuleImport,
    export_names: list[str],
) -> list[str]:
    if not module_import.names:
        return export_names

    exported = set(export_names)
    missing = list(dict.fromkeys(name for name in module_import.names if name not in exported))
    if missing:
        rendered = ", ".join(repr(name) for name in missing)
        raise ImportError(f"Kedi module {module_import.module!r} does not export {rendered}")
    return list(dict.fromkeys(module_import.names))


def _remove_imported_binding(self: ProgramCompiler, name: str) -> None:
    """Make a later import authoritative even when the binding kind changes."""
    self.rt._procedures.pop(name, None)
    self.rt._type_env_stack[0].pop(name, None)
    self.rt._profile_registry.pop(name, None)
    self.rt._import_env.pop(name, None)


def _bind_imported_profile(
    self: ProgramCompiler,
    profile: AgentProfile | ProfileDefinition,
    imported_rt: KediRuntime,
    *,
    bound_profiles: dict[str, ProfileDefinition] | None = None,
) -> AgentProfile | ProfileDefinition:
    """Attach imported private dependencies to one profile without exporting them."""
    from kedi.agent_profile.use_resolution import python_capability_for_use

    if isinstance(profile, AgentProfile):
        return profile

    bound_profiles = {} if bound_profiles is None else bound_profiles
    existing = bound_profiles.get(profile.name)
    if existing is not None:
        return existing

    tool_bindings = dict(profile.tool_bindings)
    for tool_name in self._profile_tool_names(profile):
        spec = self._imported_tool_spec(tool_name, imported_rt)
        if spec is not None:
            tool_bindings[tool_name] = spec

    # Store the shell before recursively binding children so cycles do not
    # accidentally publish a caller-owned profile under the same name.
    bound = replace(
        profile,
        tool_bindings=tool_bindings,
        profile_bindings={},
        subagent_bindings={},
    )
    bound_profiles[profile.name] = bound
    profile_bindings: dict[str, ProfileDefinition] = {}
    capability_bindings = dict(profile.capability_bindings)
    for used_name in self._profile_use_names(profile):
        if used_name in tool_bindings:
            continue
        capability = capability_bindings.get(used_name) or python_capability_for_use(
            imported_rt, used_name, {}
        )
        if capability is not None:
            capability_bindings[used_name] = capability
            continue
        used_profile = imported_rt._profile_registry.get(used_name)
        if isinstance(used_profile, ProfileDefinition):
            bound_profile = self._bind_imported_profile(
                used_profile,
                imported_rt,
                bound_profiles=bound_profiles,
            )
            profile_bindings[used_name] = cast(ProfileDefinition, bound_profile)
    subagent_bindings: dict[str, ProfileDefinition] = {}
    for child_name in self._profile_subagent_names(profile):
        child = imported_rt._profile_registry.get(child_name)
        if isinstance(child, ProfileDefinition):
            bound_child = self._bind_imported_profile(
                child,
                imported_rt,
                bound_profiles=bound_profiles,
            )
            subagent_bindings[child_name] = cast(ProfileDefinition, bound_child)
    bound = replace(
        bound,
        profile_bindings=profile_bindings,
        capability_bindings=capability_bindings,
        subagent_bindings=subagent_bindings,
    )
    bound_profiles[profile.name] = bound
    return bound


def _imported_tool_spec(
    self: ProgramCompiler, tool_name: str, imported_rt: KediRuntime
) -> Any | None:
    from kedi.agent_adapter.tools.conversion import (
        kedi_procedure_tool_spec,
        python_callable_tool_spec,
    )
    from kedi.agent_profile.use_resolution import (
        procedure_def_for_tool,
        python_callable_for_tool,
        python_capability_for_use,
    )

    proc = procedure_def_for_tool(imported_rt, tool_name)
    if proc is not None:
        return kedi_procedure_tool_spec(imported_rt, proc)
    if python_capability_for_use(imported_rt, tool_name, {}) is not None:
        return None
    tool_fn = python_callable_for_tool(imported_rt, tool_name)
    return None if tool_fn is None else python_callable_tool_spec(tool_name, tool_fn)


def _profile_tool_names(
    self: ProgramCompiler, profile: AgentProfile | ProfileDefinition
) -> tuple[str, ...]:
    if isinstance(profile, AgentProfile):
        return profile.tools
    names: list[str] = []
    for member in profile.members:
        if member.kind == "tools":
            names.extend(member.tool_names)
        elif member.kind == "use":
            names.append(member.name)
    return tuple(names)


def _profile_use_names(self: ProgramCompiler, profile: ProfileDefinition) -> tuple[str, ...]:
    return tuple(
        name
        for member in profile.members
        for name in (
            (member.name,)
            if member.kind == "use"
            else member.tool_names
            if member.kind == "tools"
            else ()
        )
    )


def _profile_subagent_names(
    self: ProgramCompiler,
    profile: AgentProfile | ProfileDefinition,
) -> tuple[str, ...]:
    if isinstance(profile, AgentProfile):
        return profile.subagents
    return tuple(member.name for member in profile.members if member.kind == "subagents")


def _export_names(self: ProgramCompiler, prog: Program, imported_rt: KediRuntime) -> list[str]:
    names: list[str] = []
    for export in prog.exports:
        if export.export_all:
            names.extend(self._public_runtime_names(imported_rt))
        names.extend(export.names)
    return list(dict.fromkeys(names))


def _public_runtime_names(self: ProgramCompiler, rt: KediRuntime) -> list[str]:
    names: list[str] = []
    names.extend(name for name in rt._procedures if self._is_public_name(name))

    for layer in rt._type_env_stack or []:
        names.extend(name for name in layer if self._is_public_name(name))

    for env in (rt._globals_env, rt._prelude_env, rt._import_env):
        if env is not None:
            names.extend(name for name in env if self._is_public_name(name))

    names.extend(name for name in rt._profile_registry if self._is_public_name(name))

    return list(dict.fromkeys(names))


def _is_public_name(self: ProgramCompiler, name: str) -> bool:
    from kedi.lang.compiler.environment import cli_args_identifier

    return bool(name) and not name.startswith("_") and name != cli_args_identifier()


def _resolve_export(
    self: ProgramCompiler,
    name: str,
    imported_rt: KediRuntime,
    module_path: Path,
) -> tuple[str, Any]:
    if name in imported_rt._procedures:
        return "procedure", imported_rt._procedures[name]

    for layer in reversed(imported_rt._type_env_stack or []):
        if name in layer:
            return "type", layer[name]

    for env in (
        imported_rt._globals_env,
        imported_rt._prelude_env,
        imported_rt._import_env,
    ):
        if env is not None and name in env:
            return "value", env[name]

    if name in imported_rt._profile_registry:
        return "profile", imported_rt._profile_registry[name]

    raise RuntimeError(f"Kedi export {name!r} is not defined in {module_path}")


def _wrap_imported_procedure(
    self: ProgramCompiler,
    name: str,
    fn: Any,
    imported_rt: KediRuntime,
) -> Any:
    import inspect

    from kedi.lang.procedure_introspection import get_procedure_def, get_procedure_metadata

    signature = inspect.signature(fn)
    params = list(signature.parameters)
    defaults = {
        param_name: parameter.default
        for param_name, parameter in signature.parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }

    def _body(**kwargs: Any) -> Any:
        return imported_rt._invoke_procedure(fn, [kwargs[param] for param in params])

    wrapper = self.rt.executor.create_dynamic_function(name, params, _body, defaults)
    retry_policy = getattr(fn, "__kedi_tool_retry_policy__", None)
    if retry_policy is not None:
        # Preserve definition-site exception classes, including across re-exports.
        setattr(wrapper, "__kedi_tool_retry_policy__", retry_policy)
    metadata = get_procedure_metadata(fn)
    if metadata is not None:
        setattr(wrapper, "__signature__", metadata.introspection_signature())
        setattr(wrapper, "__kedi_procedure_metadata__", metadata)
        setattr(
            wrapper,
            "__kedi_procedure_def__",
            get_procedure_def(fn),
        )
        wrapper.__doc__ = metadata.docstring
    return wrapper
