from __future__ import annotations

from typing import TYPE_CHECKING

from tree_sitter import Node

from kedi.lang.ast import (
    ProfileAdapterEntry,
    ProfileAgentEntry,
    ProfileApprovalEntry,
    ProfileArtifactsEntry,
    ProfileCodeModeEntry,
    ProfileDefStmt,
    ProfileEffortEntry,
    ProfileHistoryEntry,
    ProfileHooksEntry,
    ProfileMaxAgentsEntry,
    ProfileMcpEntry,
    ProfileModelEntry,
    ProfileOutputEntry,
    ProfileSettingsEntry,
    ProfileSkillsEntry,
    ProfileSubagentEntry,
    ProfileSystemEntry,
    ProfileWorkflowEntry,
    Span,
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
    members: list[ProfileModelEntry] = []
    agent_members: list[ProfileAgentEntry] = []
    adapter_members: list[ProfileAdapterEntry] = []
    effort_members: list[ProfileEffortEntry] = []
    approval_members: list[ProfileApprovalEntry] = []
    hooks_members: list[ProfileHooksEntry] = []
    history_members: list[ProfileHistoryEntry] = []
    skills_members: list[ProfileSkillsEntry] = []
    codemode_members: list[ProfileCodeModeEntry] = []
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
    docstring: str | None = None
    docstring_loc: Span | None = None
    body_node = _field(node, "body")
    if body_node is not None:
        body_children = _named_children(body_node)
        if body_children and body_children[0].type == "block_comment":
            comment = body_children[0]
            docstring = _decode_block_comment_body(_src_text(self.source, comment))
            docstring_loc = self._span(comment)
        for child in body_children:
            if child.type == "agent_directive":
                plain, code, command_plain, command_code = self._walk_agent_directive_parts(child)
                agent_members.append(
                    ProfileAgentEntry(
                        plain_value=plain,
                        backtick_code=code,
                        command_plain_value=command_plain,
                        command_backtick_code=command_code,
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
                value_node = _field(child, "value")
                assert value_node is not None
                mode = _identifier_name(self.source, value_node)
                workflow_mode: WorkflowMode
                if mode == "delegate":
                    workflow_mode = "delegate"
                elif mode == "dynamic":
                    workflow_mode = "dynamic"
                else:
                    self._error(
                        value_node,
                        "workflow mode must be 'delegate' or 'dynamic'",
                        code="K-WORKFLOW-MODE",
                    )
                    continue
                workflow_members.append(
                    ProfileWorkflowEntry(
                        mode=workflow_mode,
                        loc=self._span(child),
                    )
                )
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
    if workflow_members and workflow_members[-1].mode == "dynamic" and not subagent_members:
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
        and not agent_members
        and not adapter_members
        and not effort_members
        and not approval_members
        and not hooks_members
        and not history_members
        and not skills_members
        and not codemode_members
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
            f"Profile {name!r} must contain at least one '> agent:', '> adapter:', '> model:', '> effort:', '> approval:', '> hooks:', '> history:', '> skills:', '> codemode:', '> system:', '> mcp:', '> settings:', '> artifacts:', '> output:', '> subagent:', '> max_agents:', '> workflow:', or '> use:' member",
            code="K-PROFILE-EMPTY",
        )
    return ProfileDefStmt(
        name=name,
        agent_members=agent_members,
        adapter_members=adapter_members,
        model_members=members,
        effort_members=effort_members,
        approval_members=approval_members,
        hooks_members=hooks_members,
        history_members=history_members,
        skills_members=skills_members,
        codemode_members=codemode_members,
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
        docstring=docstring,
        docstring_loc=docstring_loc,
        loc=self._span(node),
    )
