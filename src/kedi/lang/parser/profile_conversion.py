from __future__ import annotations

from typing import TYPE_CHECKING, cast

from tree_sitter import Node

from kedi.agent_profile.capabilities import CAPABILITY_DIRECTIVES
from kedi.directive_values import (
    WORKFLOW_BACKGROUND_TOOLS,
    WORKFLOW_CONTROL_TOOLS,
    WORKFLOW_MODES,
)
from kedi.lang.ast import (
    ProfileAdapterEntry,
    ProfileAgentEntry,
    ProfileApprovalEntry,
    ProfileArtifactsEntry,
    ProfileBudgetEntry,
    ProfileCodeModeEntry,
    ProfileConditionalEntry,
    ProfileDeferEntry,
    ProfileDefStmt,
    ProfileEffortEntry,
    ProfileHistoryEntry,
    ProfileHooksEntry,
    ProfileMaxAgentsEntry,
    ProfileMcpEntry,
    ProfileModelEntry,
    ProfileOutputEntry,
    ProfileRequiresEntry,
    ProfileSettingsEntry,
    ProfileSkillsEntry,
    ProfileSubagentEntry,
    ProfileSystemEntry,
    ProfileWorkflowEntry,
    Span,
    UseDirectiveStmt,
    WorkflowMode,
)
from kedi.lang.parser.cst_helpers import (
    _decode_block_comment_body,
    _field,
    _identifier_name,
    _named_children,
    _src_text,
)
from kedi.lang.parser.diagnostics import KediDiagnostic

if TYPE_CHECKING:
    from kedi.lang.parser.cst_to_ast import _Walker


def _walk_profile_directive(self: _Walker, node: Node) -> ProfileDefStmt:
    name_node = _field(node, "name")
    assert name_node is not None
    name = _identifier_name(self.source, name_node)
    is_capability = node.type == "capability_directive"
    return _walk_profile_body(self, node, _field(node, "body"), name, is_capability)


WORKFLOW_TOOLS_MODE_ERROR = (
    "`tools` can only be used with `mode: delegate`. `mode: blocking` exposes only a "
    "synchronous `delegate_task` and `mode: dynamic` exposes only `run_workflow`, so there "
    "are no control tools to select. Remove `tools`, or use `mode: delegate` to choose "
    "which control tools the model sees."
)


def _walk_workflow_directive(self: _Walker, node: Node) -> ProfileWorkflowEntry | None:
    """``> workflow: <mode>`` or a block of ``> mode:`` and ``> tools:`` sub-directives."""
    mode_node = _field(node, "value")
    tools_node: Node | None = None
    body = _field(node, "body")
    if body is not None:
        for child in _named_children(body):
            if child.type == "workflow_mode_directive":
                if mode_node is not None:
                    self._error(child, "workflow defines `mode` twice", code="K-WORKFLOW-DUP")
                    continue
                mode_node = _field(child, "value")
            else:
                if tools_node is not None:
                    self._error(child, "workflow defines `tools` twice", code="K-WORKFLOW-DUP")
                    continue
                tools_node = child
    mode = "delegate" if mode_node is None else _identifier_name(self.source, mode_node)
    if mode not in WORKFLOW_MODES:
        assert mode_node is not None
        self._error(
            mode_node,
            "workflow mode must be 'delegate', 'dynamic' or 'blocking'",
            code="K-WORKFLOW-MODE",
        )
        return None
    tools = None
    if tools_node is not None:
        names = _workflow_tool_names(self, tools_node)
        problem = (
            WORKFLOW_TOOLS_MODE_ERROR if mode != "delegate" else _workflow_tools_problem(names)
        )
        if problem is not None:
            self._error(tools_node, problem, code="K-WORKFLOW-TOOLS")
            return None
        tools = None if names == ["all"] else tuple(names)
    return ProfileWorkflowEntry(mode=cast(WorkflowMode, mode), tools=tools, loc=self._span(node))


def _workflow_tool_names(self: _Walker, node: Node) -> list[str]:
    name_node = _field(node, "name")
    if name_node is not None:
        return [_identifier_name(self.source, name_node)]
    body = _field(node, "body")
    assert body is not None
    return [
        _identifier_name(self.source, identifier)
        for child in _named_children(body)
        if child.type == "workflow_tool_name" and (identifier := _field(child, "name"))
    ]


def _workflow_tools_problem(names: list[str]) -> str | None:
    if names == ["all"]:
        return None
    if "all" in names:
        return "`all` already selects every control tool; use `all` alone or list the tools"
    unknown = [name for name in names if name not in WORKFLOW_CONTROL_TOOLS]
    if unknown:
        return (
            f"unknown subagent control tool {', '.join(repr(name) for name in unknown)}; "
            f"choose from {', '.join(WORKFLOW_CONTROL_TOOLS)}, or use `all`"
        )
    if len(set(names)) != len(names):
        return "`> tools:` lists a control tool more than once"
    if "wait_subagent" not in names and set(names) & set(WORKFLOW_BACKGROUND_TOOLS):
        return (
            "`cancel_subagent` and `subagent_status` only apply to background runs, which "
            "are offered only when `wait_subagent` is listed; add `wait_subagent` or remove them"
        )
    return None


def _walk_profile_body(
    self: _Walker,
    node: Node,
    body_node: Node | None,
    name: str,
    is_capability: bool,
    *,
    branch: bool = False,
) -> ProfileDefStmt:
    conditional_members: list[ProfileConditionalEntry] = []
    defer_members: list[ProfileDeferEntry] = []
    members: list[ProfileModelEntry] = []
    agent_members: list[ProfileAgentEntry] = []
    adapter_members: list[ProfileAdapterEntry] = []
    effort_members: list[ProfileEffortEntry] = []
    approval_members: list[ProfileApprovalEntry] = []
    hooks_members: list[ProfileHooksEntry] = []
    history_members: list[ProfileHistoryEntry] = []
    skills_members: list[ProfileSkillsEntry] = []
    codemode_members: list[ProfileCodeModeEntry] = []
    requires_members: list[ProfileRequiresEntry] = []
    budget_members: list[ProfileBudgetEntry] = []
    system_members: list[ProfileSystemEntry] = []
    mcp_members: list[ProfileMcpEntry] = []
    settings_members: list[ProfileSettingsEntry] = []
    artifacts_members: list[ProfileArtifactsEntry] = []
    output_members: list[ProfileOutputEntry] = []
    subagent_members: list[ProfileSubagentEntry] = []
    max_agents_members: list[ProfileMaxAgentsEntry] = []
    workflow_members: list[ProfileWorkflowEntry] = []
    use_targets: list[str] = []
    tool_names: list[str] = []
    use_members: list[UseDirectiveStmt] = []
    docstring: str | None = None
    docstring_loc: Span | None = None
    if body_node is not None:
        body_children = _named_children(body_node)
        if not branch and body_children and body_children[0].type == "block_comment":
            comment = body_children[0]
            docstring = _decode_block_comment_body(_src_text(self.source, comment))
            docstring_loc = self._span(comment)
        for child in body_children:
            if child.type in {"profile_if_stmt", "capability_if_stmt"}:
                condition = _field(child, "condition")
                assert condition is not None
                code = _field(condition, "code")
                assert code is not None
                consequence = _field(child, "consequence")
                alternative = _field(child, "alternative")
                assert consequence is not None
                conditional_members.append(
                    ProfileConditionalEntry(
                        backtick_code=_src_text(self.source, code),
                        consequence=_walk_profile_body(
                            self, consequence, consequence, name, is_capability, branch=True
                        ),
                        alternative=(
                            _walk_profile_body(
                                self, alternative, alternative, name, is_capability, branch=True
                            )
                            if alternative is not None
                            else None
                        ),
                        loc=self._span(child),
                        condition_loc=self._span(condition),
                    )
                )
                continue
            member_name = child.type.removesuffix("_directive")
            if member_name == "system":
                member_name = "instructions"
            if (
                is_capability
                and child.type.endswith("_directive")
                and member_name not in CAPABILITY_DIRECTIVES
            ):
                self._error(
                    child,
                    f"'> {member_name}:' is not allowed in a capability; configure the agent "
                    "or a separate procedure/subagent profile instead",
                    code="K-CAPABILITY-MEMBER",
                )
                continue
            if child.type == "defer_directive":
                value = _field(child, "value")
                assert value is not None
                code = _field(value, "code") if value.type == "inline_python_expr" else None
                defer_members.append(
                    ProfileDeferEntry(
                        enabled=_src_text(self.source, value) == "true",
                        backtick_code=_src_text(self.source, code) if code is not None else None,
                        loc=self._span(child),
                    )
                )
                continue
            if child.type == "agent_directive":
                plain, code, command_plain, command_code, connection_fields = (
                    self._walk_agent_directive_parts(child)
                )
                agent_members.append(
                    ProfileAgentEntry(
                        plain_value=plain,
                        backtick_code=code,
                        command_plain_value=command_plain,
                        command_backtick_code=command_code,
                        connection_fields=connection_fields,
                        loc=self._span(child),
                    )
                )
            elif child.type == "adapter_directive":
                plain, code = self._walk_simple_directive_value(child)
                adapter_members.append(
                    ProfileAdapterEntry(
                        plain_value=plain,
                        backtick_code=code,
                        loc=self._span(child),
                    )
                )
            elif child.type == "model_directive":
                value_node = _field(child, "value")
                if value_node is None:
                    continue
                if value_node.type == "inline_python_expr":
                    code_node = _field(value_node, "code")
                    assert code_node is not None
                    members.append(
                        ProfileModelEntry(
                            backtick_code=_src_text(self.source, code_node),
                            loc=self._span(child),
                        )
                    )
                else:
                    members.append(
                        ProfileModelEntry(
                            plain_value=_src_text(self.source, value_node).strip(),
                            loc=self._span(child),
                        )
                    )
            elif child.type == "effort_directive":
                value_node = _field(child, "value")
                assert value_node is not None
                if value_node.type == "inline_python_expr":
                    code_node = _field(value_node, "code")
                    assert code_node is not None
                    effort_members.append(
                        ProfileEffortEntry(
                            backtick_code=_src_text(self.source, code_node),
                            loc=self._span(child),
                        )
                    )
                else:
                    plain = _src_text(self.source, value_node).strip()
                    self._validate_effort_plain_value(value_node, plain)
                    effort_members.append(
                        ProfileEffortEntry(
                            plain_value=plain,
                            loc=self._span(child),
                        )
                    )
            elif child.type == "approval_directive":
                approval_stmt = self._walk_approval_directive(child)
                approval_members.append(
                    ProfileApprovalEntry(
                        plain_value=approval_stmt.plain_value,
                        backtick_code=approval_stmt.backtick_code,
                        loc=self._span(child),
                    )
                )
            elif child.type == "hooks_directive":
                hooks_stmt = self._walk_hooks_directive(child)
                if is_capability and (
                    hooks_stmt.enabled is False
                    or any(field.name == "user_prompt_submit" for field in hooks_stmt.fields)
                ):
                    self._error(
                        child,
                        "Capabilities may add owned tool hooks, not disable hooks or set user_prompt_submit",
                        code="K-CAPABILITY-HOOK",
                    )
                    continue
                hooks_members.append(
                    ProfileHooksEntry(
                        enabled=hooks_stmt.enabled,
                        fields=hooks_stmt.fields,
                        loc=self._span(child),
                    )
                )
            elif child.type == "history_directive":
                history_stmt = self._walk_history_directive(child)
                history_members.append(
                    ProfileHistoryEntry(
                        enabled=history_stmt.enabled,
                        fields=history_stmt.fields,
                        loc=self._span(child),
                    )
                )
            elif child.type == "skills_directive":
                skills_stmt = self._walk_skills_directive(child)
                skills_members.append(
                    ProfileSkillsEntry(
                        enabled=skills_stmt.enabled,
                        fields=skills_stmt.fields,
                        loc=self._span(child),
                    )
                )
            elif child.type == "codemode_directive":
                codemode_stmt = self._walk_codemode_directive(child)
                codemode_members.append(
                    ProfileCodeModeEntry(
                        enabled=codemode_stmt.enabled,
                        fields=codemode_stmt.fields,
                        loc=self._span(child),
                    )
                )
            elif child.type == "requires_directive":
                requires_stmt = self._walk_requires_directive(child)
                requires_members.append(
                    ProfileRequiresEntry(
                        requirements=requires_stmt.requirements,
                        requirement_locs=requires_stmt.requirement_locs,
                        loc=self._span(child),
                    )
                )
            elif child.type == "budget_directive":
                budget_stmt = self._walk_budget_directive(child)
                budget_members.append(
                    ProfileBudgetEntry(
                        request_limit=budget_stmt.request_limit,
                        tool_attempt_limit=budget_stmt.tool_attempt_limit,
                        loc=self._span(child),
                    )
                )
            elif child.type == "system_directive":
                system_stmt = self._walk_system_directive(child)
                if system_stmt.template_lines:
                    system_members.append(
                        ProfileSystemEntry(
                            plain_value=system_stmt.plain_value,
                            template_lines=system_stmt.template_lines,
                            loc=self._span(child),
                        )
                    )
                    continue
                value_node = _field(child, "value")
                if value_node is None:
                    continue
                if value_node.type == "inline_python_expr":
                    code_node = _field(value_node, "code")
                    assert code_node is not None
                    system_members.append(
                        ProfileSystemEntry(
                            backtick_code=_src_text(self.source, code_node),
                            loc=self._span(child),
                        )
                    )
                else:
                    system_members.append(
                        ProfileSystemEntry(
                            plain_value=_src_text(self.source, value_node).strip(),
                            loc=self._span(child),
                        )
                    )
            elif child.type == "use_directive":
                use_stmt = self._walk_use_directive(child)
                use_members.append(use_stmt)
                if use_stmt.tool_names:
                    tool_names.extend(use_stmt.tool_names)
                elif use_stmt.name:
                    use_targets.append(use_stmt.name)
            elif child.type == "mcp_directive":
                mcp_stmt = self._walk_mcp_directive(child)
                mcp_members.append(ProfileMcpEntry(fields=mcp_stmt.fields, loc=self._span(child)))
            elif child.type == "settings_directive":
                settings_stmt = self._walk_settings_directive(child)
                settings_members.append(
                    ProfileSettingsEntry(
                        fields=settings_stmt.fields,
                        loc=self._span(child),
                    )
                )
            elif child.type == "artifacts_directive":
                artifacts_stmt = self._walk_artifacts_directive(child)
                artifacts_members.append(
                    ProfileArtifactsEntry(
                        fields=artifacts_stmt.fields,
                        loc=self._span(child),
                    )
                )
            elif child.type == "output_directive":
                type_node = _field(child, "type")
                assert type_node is not None
                type_expr = self._walk_type_expr(type_node)
                assert type_expr is not None
                output_members.append(
                    ProfileOutputEntry(
                        type_annotation=self._type_expr_to_str(type_node),
                        type_expr=type_expr,
                        loc=self._span(child),
                    )
                )
            elif child.type == "subagent_directive":
                subagent_node = _field(child, "name")
                assert subagent_node is not None
                subagent_members.append(
                    ProfileSubagentEntry(
                        name=_identifier_name(self.source, subagent_node),
                        loc=self._span(subagent_node),
                    )
                )
            elif child.type == "max_agents_directive":
                value_node = _field(child, "value")
                assert value_node is not None
                max_agents_members.append(
                    ProfileMaxAgentsEntry(
                        value=int(_src_text(self.source, value_node)),
                        loc=self._span(child),
                    )
                )
            elif child.type == "workflow_directive":
                entry = _walk_workflow_directive(self, child)
                if entry is not None:
                    workflow_members.append(entry)
    if len(workflow_members) > 1:
        for entry in workflow_members[1:]:
            assert entry.loc is not None
            self.diagnostics.append(
                KediDiagnostic(
                    severity="error",
                    message="profile can define only one workflow mode",
                    span=entry.loc,
                    code="K-WORKFLOW-DUP",
                )
            )
    if len(output_members) > 1:
        for entry in output_members[1:]:
            assert entry.loc is not None
            self.diagnostics.append(
                KediDiagnostic(
                    severity="error",
                    message="profile can define only one output type",
                    span=entry.loc,
                    code="K-PROFILE-OUTPUT-DUP",
                )
            )
    if (
        workflow_members
        and not branch
        and not conditional_members
        and workflow_members[-1].mode == "dynamic"
        and not subagent_members
        and not use_targets
    ):
        entry = workflow_members[-1]
        assert entry.loc is not None
        self.diagnostics.append(
            KediDiagnostic(
                severity="error",
                message="dynamic workflow requires at least one direct subagent",
                span=entry.loc,
                code="K-WORKFLOW-SUBAGENT",
            )
        )
    if (
        not members
        and not conditional_members
        and not defer_members
        and not agent_members
        and not adapter_members
        and not effort_members
        and not approval_members
        and not hooks_members
        and not history_members
        and not skills_members
        and not codemode_members
        and not requires_members
        and not budget_members
        and not system_members
        and not mcp_members
        and not settings_members
        and not artifacts_members
        and not output_members
        and not subagent_members
        and not max_agents_members
        and not workflow_members
        and not use_targets
        and not tool_names
    ):
        self._error(
            node,
            f"Profile {name!r} must contain at least one '> agent:', '> adapter:', '> model:', '> effort:', '> approval:', '> hooks:', '> history:', '> skills:', '> codemode:', '> requires:', '> instructions:', '> mcp:', '> settings:', '> artifacts:', '> output:', '> subagent:', '> max_agents:', '> workflow:', or '> use:' member",
            code="K-PROFILE-EMPTY",
        )
    return ProfileDefStmt(
        name=name,
        kind="capability" if is_capability else "profile",
        conditional_members=conditional_members,
        defer_members=defer_members,
        agent_members=agent_members,
        adapter_members=adapter_members,
        model_members=members,
        effort_members=effort_members,
        approval_members=approval_members,
        hooks_members=hooks_members,
        history_members=history_members,
        skills_members=skills_members,
        codemode_members=codemode_members,
        requires_members=requires_members,
        budget_members=budget_members,
        system_members=system_members,
        mcp_members=mcp_members,
        settings_members=settings_members,
        artifacts_members=artifacts_members,
        output_members=output_members,
        subagent_members=subagent_members,
        max_agents_members=max_agents_members,
        workflow_members=workflow_members,
        use_targets=use_targets,
        tool_names=tool_names,
        use_members=use_members,
        docstring=docstring,
        docstring_loc=docstring_loc,
        loc=self._span(node),
    )
