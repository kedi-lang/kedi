"""Read-only inspection of model call sites in a Kedi program."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from types import FunctionType
from typing import Any, Literal, cast

from kedi.agent_adapter.meta import AdapterCapability, adapter_capability_names
from kedi.agent_adapter.registry import adapter_capabilities, known_adapter_shortnames
from kedi.agent_adapter.tools.retry import ToolRetryPolicy
from kedi.agent_profile.compile import profile_stmt_to_definition
from kedi.agent_profile.evaluation import apply_profile_members
from kedi.lang.ast import (
    AdapterDirectiveStmt,
    AgentDirectiveStmt,
    ArtifactsDirectiveStmt,
    BudgetDirectiveStmt,
    CodeModeDirectiveStmt,
    ConditionalLoopStmt,
    IfStmt,
    LoopStmt,
    McpDirectiveStmt,
    ModelDirectiveStmt,
    ModuleImport,
    ProcDefStmt,
    ProcedureDef,
    ProfileDefStmt,
    Program,
    RawInvokeStmt,
    RequiresDirectiveStmt,
    Span,
    TaskGroupStmt,
    TemplateBlock,
    TemplateLine,
    UseDirectiveStmt,
)
from kedi.lang.parser import parse_program


@dataclass(frozen=True)
class ExplainedValue:
    value: str | None
    source: str
    unresolved: str | None = None


@dataclass(frozen=True)
class ExplainedTool:
    name: str
    source: str
    risk: str | None = None
    retries: int | None = None
    effects: frozenset[str] | None = None


@dataclass(frozen=True)
class ExplainedRequirement:
    name: str
    status: Literal["satisfied", "missing", "unresolved"]
    source: str


@dataclass(frozen=True)
class SourceLocation:
    path: str | None
    line: int
    column: int


@dataclass(frozen=True)
class ExplainedCall:
    source_location: SourceLocation
    scope: str
    adapter: ExplainedValue
    model: ExplainedValue
    tools: tuple[ExplainedTool, ...]
    requirements: tuple[ExplainedRequirement, ...]
    unresolved: tuple[str, ...]
    configuration: tuple[tuple[str, ExplainedValue], ...] = ()


@dataclass(frozen=True)
class ExplainReport:
    calls: tuple[ExplainedCall, ...]
    diagnostics: tuple[str, ...]


@dataclass(frozen=True)
class _InspectionSource:
    program: Program
    kind: str
    reload_program: Any | None
    profile: Any
    adapter_shortname: str | None
    tools: dict[str, Any]


@dataclass(frozen=True)
class _State:
    adapter: ExplainedValue = ExplainedValue(None, "unspecified", "adapter is not selected")
    model: ExplainedValue = ExplainedValue(None, "unspecified", "model is not selected")
    tools: tuple[ExplainedTool, ...] = ()
    requirements: tuple[tuple[str, str], ...] = ()
    unresolved: tuple[str, ...] = ()
    configuration: tuple[tuple[str, ExplainedValue], ...] = ()


def _origin(prefix: str, loc: Span | None) -> str:
    return f"{prefix} at line {loc.start_line}" if loc is not None else prefix


def _selection(plain: str | None, code: str | None, *, origin: str, label: str) -> ExplainedValue:
    if code is not None:
        return ExplainedValue(None, origin, f"{label} uses a dynamic expression")
    if plain is None:
        return ExplainedValue(None, origin, f"{label} is not statically known")
    return ExplainedValue(plain, origin)


def _with_tool(state: _State, tool: ExplainedTool) -> _State:
    return replace(
        state, tools=tuple(item for item in state.tools if item.name != tool.name) + (tool,)
    )


def _with_requirement(state: _State, name: str, source: str) -> _State:
    if any(item[0] == name for item in state.requirements):
        return state
    return replace(state, requirements=(*state.requirements, (name, source)))


def _with_unknown(state: _State, message: str) -> _State:
    return replace(state, unresolved=(*state.unresolved, message))


class _Analyzer:
    def __init__(self, program: Program, initial: _State):
        self.program = program
        self.initial = initial
        self.calls: list[ExplainedCall] = []
        self.diagnostics: list[str] = []
        self.profiles: dict[str, ProfileDefStmt] = dict(program.profile_defs)
        self.procedures = {proc.name: proc for proc in program.procedures}
        self.available_procedures: set[str] = set()

    def analyze(self) -> ExplainReport:
        sequence: list[Any] = list(self.program.ordered_toplevel)
        if not sequence:
            sequence = [*self.program.toplevel_body, *self.program.procedures]
            sequence.sort(key=lambda item: item.loc.start_byte if item.loc else -1)
        self._walk(sequence, self.initial, "top-level")
        return ExplainReport(tuple(self.calls), tuple(self.diagnostics))

    def _profile(self, name: str, state: _State, seen: frozenset[str]) -> _State:
        if name in seen:
            return self._unknown_profile(state, f"cyclic profile reference: {name}")
        profile = self.profiles.get(name)
        if profile is None:
            return self._unknown_profile(state, f"profile {name} is not statically available")
        definition = profile_stmt_to_definition(profile)
        return apply_profile_members(
            definition.members,
            state,
            lambda current, member: self._directive(member, current, f"profile {name}"),
            lambda current, member: (
                self._tool_name(member.name, current, f"profile {name}")
                if member.name in self.available_procedures
                else self._profile(member.name, current, seen | {name})
            ),
        )

    def _unknown_profile(self, state: _State, reason: str) -> _State:
        unknown = ExplainedValue(None, "unresolved profile", reason)
        return replace(
            _with_unknown(state, reason),
            adapter=unknown,
            model=unknown,
            tools=tuple(
                replace(tool, risk=None, retries=None, effects=None) for tool in state.tools
            ),
            configuration=tuple((name, unknown) for name, _ in state.configuration),
        )

    def _tool_name(self, name: str, state: _State, origin: str) -> _State:
        proc = self.procedures.get(name) if name in self.available_procedures else None
        if proc is None:
            if any(tool.name == name for tool in state.tools):
                return state
            return _with_tool(state, ExplainedTool(name, f"{origin}: unresolved symbol {name}"))
        declaration = proc.tool
        return _with_tool(
            state,
            ExplainedTool(
                (declaration.name or name) if declaration is not None else name,
                f"{origin}: procedure {name}",
                declaration.risk if declaration is not None else "mutating",
                declaration.retries if declaration is not None else 0,
            ),
        )

    def _directive(self, item: Any, state: _State, prefix: str = "directive") -> _State:
        origin = _origin(prefix, getattr(item, "loc", None))
        kind = getattr(item, "kind", None)
        if kind == "conditional":
            return self._unknown_profile(state, f"{origin}: conditional profile members")
        if isinstance(item, (AgentDirectiveStmt, AdapterDirectiveStmt)) or kind in {
            "agent",
            "adapter",
        }:
            return replace(
                state,
                adapter=_selection(
                    item.plain_value, item.backtick_code, origin=origin, label="adapter"
                ),
            )
        if isinstance(item, ModelDirectiveStmt) or kind == "model":
            return replace(
                state,
                model=_selection(
                    item.plain_value, item.backtick_code, origin=origin, label="model"
                ),
            )
        if isinstance(item, RequiresDirectiveStmt) or kind == "requires":
            for name in item.requirements:
                state = _with_requirement(state, name, origin)
        elif isinstance(item, McpDirectiveStmt) or kind == "mcp":
            state = _with_requirement(state, "mcp", origin)
            state = _with_unknown(state, f"{origin}: MCP tools are discovered at runtime")
        elif isinstance(item, CodeModeDirectiveStmt) or kind == "codemode":
            state = self._configuration(state, "codemode", str(item.enabled), origin)
        elif isinstance(item, ArtifactsDirectiveStmt) or kind == "artifacts":
            enabled = next((field for field in item.fields if field.name == "enabled"), None)
            if enabled is not None:
                value = (
                    "unresolved"
                    if enabled.backtick_code is not None or enabled.plain_value is None
                    else str(enabled.plain_value.strip().lower() in {"true", "enabled"})
                )
                state = self._configuration(state, "artifacts", value, origin)
            elif "artifacts" not in dict(state.configuration):
                state = self._configuration(state, "artifacts", "True", origin)
        elif isinstance(item, BudgetDirectiveStmt) or kind == "budget":
            if item.request_limit is not None:
                state = _with_requirement(state, "request_budget", origin)
            if item.tool_attempt_limit is not None:
                state = _with_requirement(state, "tool_attempt_budget", origin)
            for name in ("request_limit", "tool_attempt_limit"):
                if getattr(item, name) is not None:
                    state = self._configuration(
                        state, f"budget.{name}", str(getattr(item, name)), origin
                    )
        elif kind == "tools":
            for name in item.tool_names:
                state = self._tool_name(name, state, origin)
        elif kind in {
            "system",
            "settings",
            "effort",
            "approval",
            "history",
            "hooks",
            "skills",
            "output",
            "subagent",
            "max_agents",
            "workflow",
        } or type(item).__name__ in {
            "SystemDirectiveStmt",
            "SettingsDirectiveStmt",
            "EffortDirectiveStmt",
            "ApprovalDirectiveStmt",
            "HistoryDirectiveStmt",
            "HooksDirectiveStmt",
            "SkillsDirectiveStmt",
        }:
            label = kind or type(item).__name__.removesuffix("DirectiveStmt").lower()
            # Configuration can contain credentials, input substitutions and executable
            # expressions. Inspection exposes its provenance, never evaluates it.
            state = self._configuration(state, label, "configured (value withheld)", origin)
        return state

    def _configuration(self, state: _State, name: str, value: str, origin: str) -> _State:
        config = dict(state.configuration)
        config[name] = ExplainedValue(value, origin)
        return replace(state, configuration=tuple(config.items()))

    def _walk(self, sequence: list[Any], state: _State, scope: str) -> _State:
        profiles, procedures, available = self.profiles, self.procedures, self.available_procedures
        self.profiles, self.procedures, self.available_procedures = (
            dict(profiles),
            dict(procedures),
            set(available),
        )
        try:
            return self._walk_scope(sequence, state, scope)
        finally:
            self.profiles, self.procedures, self.available_procedures = (
                profiles,
                procedures,
                available,
            )

    def _walk_scope(self, sequence: list[Any], state: _State, scope: str) -> _State:
        for item in sequence:
            if isinstance(item, ProfileDefStmt):
                self.profiles[item.name] = item
            elif isinstance(item, (ProcedureDef, ProcDefStmt)):
                proc = item if isinstance(item, ProcedureDef) else item.proc
                if proc is not None:
                    self.available_procedures.add(proc.name)
                    self.procedures[proc.name] = proc
                    self._walk(proc.body, state, f"procedure {proc.name}")
            elif isinstance(item, UseDirectiveStmt):
                if item.tool_names:
                    for name in item.tool_names:
                        state = self._tool_name(name, state, _origin("use", item.loc))
                elif item.target:
                    state = (
                        self._tool_name(item.target, state, _origin("use", item.loc))
                        if item.target in self.available_procedures
                        else self._profile(item.target, state, frozenset())
                    )
            elif isinstance(item, ModuleImport):
                state = _with_unknown(
                    state, "imported symbols and profiles require runtime resolution"
                )
            elif isinstance(item, (TemplateBlock, TemplateLine, RawInvokeStmt)):
                self._call(item, state, scope)
            elif isinstance(item, IfStmt):
                self._walk(item.body, state, f"{scope} (conditional)")
                if item.else_body:
                    self._walk(item.else_body, state, f"{scope} (alternative)")
            elif isinstance(item, (LoopStmt, ConditionalLoopStmt)):
                self._walk(item.body, state, f"{scope} (loop)")
                if isinstance(item, LoopStmt) and item.map_body:
                    self._walk(item.map_body, state, f"{scope} (map)")
            elif isinstance(item, TaskGroupStmt):
                for arm in item.arms:
                    self._walk(arm.body or [], state, f"{scope} (process {arm.wait.handle})")
            else:
                state = self._directive(item, state)
        return state

    def _call(self, item: Any, state: _State, scope: str) -> None:
        loc = item.loc
        location = (
            SourceLocation(loc.source_path, loc.start_line, loc.start_col)
            if loc
            else SourceLocation(None, 0, 0)
        )
        requirements = state.requirements
        for feature in ("codemode", "artifacts"):
            setting = dict(state.configuration).get(feature)
            if setting is not None and setting.value == "True":
                requirements = (*requirements, (feature, setting.source))
        if state.tools:
            requirements = (*requirements, ("tool_registration", "effective tools"))
        statuses: list[ExplainedRequirement] = []
        for name, source in dict(requirements).items():
            status: Literal["satisfied", "missing", "unresolved"] = "unresolved"
            if (
                state.adapter.value in known_adapter_shortnames()
                and name in adapter_capability_names()
            ):
                try:
                    supported = adapter_capabilities(state.adapter.value).capability_status(
                        cast(AdapterCapability, name)
                    )
                except Exception:
                    supported = None
                status = (
                    "unresolved" if supported is None else "satisfied" if supported else "missing"
                )
            if status == "missing":
                self.diagnostics.append(
                    f"{scope}:{location.line}: {name} is unsupported by {state.adapter.value}"
                )
            statuses.append(ExplainedRequirement(name, status, source))
        unresolved = list(state.unresolved)
        for value in (state.adapter, state.model):
            if value.unresolved:
                unresolved.append(value.unresolved)
        self.calls.append(
            ExplainedCall(
                location,
                scope,
                state.adapter,
                state.model,
                state.tools,
                tuple(statuses),
                tuple(dict.fromkeys(unresolved)),
                state.configuration,
            )
        )


def explain_program(program: Program, *, initial: _State | None = None) -> ExplainReport:
    """Inspect a parsed program without compiling or executing it."""
    return _Analyzer(program, initial or _State()).analyze()


def explain(target: Any) -> ExplainReport:
    """Inspect a ``@kedi.query`` or ``@kedi.bind`` callable without invoking it."""
    if not isinstance(target, FunctionType):
        raise TypeError("explain() expects a @kedi.query or @kedi.bind function")
    source = vars(target).get("__kedi_inspection_source__")
    if not isinstance(source, _InspectionSource):
        raise TypeError("explain() expects a @kedi.query or @kedi.bind function")
    from kedi.api.state import _config_var

    config = _config_var.get()
    profile = config.profile.merge(source.profile)
    adapter = profile.adapter_shortname or source.adapter_shortname
    if adapter is None and config.adapter is None:
        adapter = config.adapter_shortname
    model = profile.model or config.model
    initial = _State(
        adapter=ExplainedValue(
            adapter,
            "Python configuration",
            None if adapter else "custom Python adapter cannot be inspected without execution",
        ),
        model=ExplainedValue(
            model, "Python configuration", None if model else "model is not selected"
        ),
    )
    for name, fn in {**config.tools, **source.tools}.items():
        metadata = vars(fn) if isinstance(fn, FunctionType) else {}
        policy = metadata.get("__kedi_tool_retry_policy__")
        initial = _with_tool(
            initial,
            ExplainedTool(
                name,
                f"Python tool {name}",
                metadata.get("__kedi_tool_risk__"),
                policy.retries if isinstance(policy, ToolRetryPolicy) else None,
            ),
        )
    for name in profile.requirements:
        initial = _with_requirement(initial, name, "Python configuration")
    for name in profile.tools:
        if not any(tool.name == name for tool in initial.tools):
            initial = _with_tool(initial, ExplainedTool(name, f"Python profile: symbol {name}"))
    program = source.program if source.reload_program is None else source.reload_program()[0]
    return explain_program(program, initial=initial)


def explain_file(
    path: str | Path, *, adapter: str | None = None, model: str | None = None
) -> ExplainReport:
    """Parse a file directly; do not load its imports or run its prelude."""
    source_path = Path(path).resolve()
    program = parse_program(source_path.read_text(encoding="utf-8"), source_path=str(source_path))
    initial = _State(
        adapter=ExplainedValue(
            adapter,
            "CLI option" if adapter else "unspecified",
            None if adapter else "adapter is not selected",
        ),
        model=ExplainedValue(
            model,
            "CLI option" if model else "unspecified",
            None if model else "model is not selected",
        ),
    )
    return explain_program(program, initial=initial)


def render_explain(report: ExplainReport) -> str:
    """Render only configuration metadata, never prompt or tool payloads."""
    lines: list[str] = []
    for call in report.calls:
        loc = call.source_location
        lines.append(f"{call.scope} at {loc.path or '<memory>'}:{loc.line}:{loc.column}")
        for name, value in (("adapter", call.adapter), ("model", call.model)):
            lines.append(f"  {name}: {value.value or 'unresolved'} ({value.source})")
        for tool in call.tools:
            lines.append(
                f"  tool: {tool.name} ({tool.source}, risk={tool.risk or 'unknown'}, retries={tool.retries if tool.retries is not None else 'unknown'}, effects={'unknown' if tool.effects is None else ','.join(sorted(tool.effects))})"
            )
        for name, value in call.configuration:
            lines.append(f"  {name}: {value.value or 'unresolved'} ({value.source})")
        for requirement in call.requirements:
            lines.append(
                f"  requires: {requirement.name} {requirement.status} ({requirement.source})"
            )
        for message in call.unresolved:
            lines.append(f"  unresolved: {message}")
    for diagnostic in report.diagnostics:
        lines.append(f"diagnostic: {diagnostic}")
    return "\n".join(lines) if lines else "No model call sites found."
