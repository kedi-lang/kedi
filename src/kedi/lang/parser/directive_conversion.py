from __future__ import annotations

import re
from typing import TYPE_CHECKING, List

from tree_sitter import Node

from kedi.agent_adapter.hooks import HOOK_EVENT_NAMES
from kedi.agent_profile.codemode import CODEMODE_SETTING_FIELDS
from kedi.agent_profile.settings import (
    MODEL_SETTING_KEYS,
    is_valid_model_setting_key,
    parse_plain_setting_value,
)
from kedi.artifacts import ARTIFACT_POLICY_INPUT_FIELDS, ArtifactPolicy, ArtifactPolicyError
from kedi.effort import EFFORT_VALUES, is_valid_effort
from kedi.lang.ast import (
    AdapterDirectiveStmt,
    AgentDirectiveStmt,
    ApprovalDirectiveStmt,
    ArtifactsDirectiveStmt,
    ArtifactsField,
    AutoSpec,
    CodeModeDirectiveStmt,
    EffortDirectiveStmt,
    HistoryDirectiveStmt,
    HooksDirectiveStmt,
    McpDirectiveStmt,
    McpField,
    ModelDirectiveStmt,
    PackageMetadata,
    SettingsDirectiveStmt,
    SettingsField,
    SkillsDirectiveStmt,
    SystemDirectiveStmt,
    TemplateLine,
    TextSegment,
    UseDirectiveStmt,
)
from kedi.lang.parser.cst_helpers import _field, _identifier_name, _named_children, _src_text
from kedi.lang.parser.escapes import dedent_block
from kedi.mcp import MCP_TRANSPORT_VALUES, is_valid_mcp_transport

if TYPE_CHECKING:
    from kedi.lang.parser.cst_to_ast import _Walker


def _validate_effort_plain_value(self: _Walker, node: Node, value: str) -> None:
    if is_valid_effort(value):
        return
    expected = ", ".join(sorted(EFFORT_VALUES))
    self._error(
        node,
        f"effort directive must be one of: {expected}",
        code="K-EFFORT-VALUE",
    )


def _validate_approval_plain_value(self: _Walker, node: Node, value: str) -> None:
    if value in {"allow", "deny"}:
        return
    self._error(
        node,
        "approval directive must be 'allow', 'deny', or an inline Python handler; "
        "'edit' is only valid as ApprovalDecision.edit(...) from a handler",
        code="K-APPROVAL-VALUE",
    )


def _validate_mcp_transport_plain_value(self: _Walker, node: Node, value: str) -> None:
    if is_valid_mcp_transport(value):
        return
    expected = ", ".join(sorted(MCP_TRANSPORT_VALUES))
    self._error(
        node,
        f"MCP transport must be one of: {expected}",
        code="K-MCP-TRANSPORT",
    )


def _validate_setting_key(self: _Walker, node: Node, name: str) -> None:
    if is_valid_model_setting_key(name):
        return
    expected = ", ".join(sorted(MODEL_SETTING_KEYS))
    self._error(
        node,
        f"model setting must be one of: {expected}",
        code="K-SETTING-KEY",
    )


def _walk_package_directive(self: _Walker, node: Node) -> PackageMetadata:
    name_node = _field(node, "name")
    assert name_node is not None
    package = PackageMetadata(name=_identifier_name(self.source, name_node), loc=self._span(node))
    body = _field(node, "body")
    assert body is not None

    scalar_fields = {
        "author": "author",
        "contact": "contact",
        "version": "version",
        "source": "source",
        "python": "python_requirement",
        "license": "license",
    }
    for child in _named_children(body):
        if child.type == "package_python_dependencies":
            if package.python_dependencies:
                self._error(
                    child,
                    "package manifest can define only one python_dependencies block",
                    code="K-PACKAGE-DUP",
                )
                continue
            dependencies = _field(child, "body")
            assert dependencies is not None
            package.python_dependencies = [
                _src_text(self.source, value).strip()
                for dependency in _named_children(dependencies)
                if dependency.type == "package_dependency"
                for value in [_field(dependency, "value")]
                if value is not None and _src_text(self.source, value).strip()
            ]
            package.field_locs["python_dependencies"] = self._span(child)
            continue
        assert child.type == "package_field"
        field_node = _field(child, "name")
        value_node = _field(child, "value")
        assert field_node is not None and value_node is not None
        field_name = _identifier_name(self.source, field_node)
        attribute = scalar_fields.get(field_name)
        if attribute is None:
            self._error(
                field_node,
                f"unknown package metadata field {field_name!r}",
                code="K-PACKAGE-FIELD",
            )
            continue
        if field_name in package.field_locs:
            self._error(
                field_node,
                f"duplicate package metadata field {field_name!r}",
                code="K-PACKAGE-DUP",
            )
            continue
        value = _src_text(self.source, value_node).strip()
        setattr(package, attribute, value)
        package.field_locs[field_name] = self._span(field_node)

    if package.python_requirement is not None:
        match = re.fullmatch(r"python@(\d+)\.(\d+)(?:-(\d+)\.(\d+))?", package.python_requirement)
        if match is None:
            self._error(
                node,
                "python must be python@MAJOR.MINOR or python@MAJOR.MINOR-MAJOR.MINOR",
                code="K-PACKAGE-PYTHON",
            )
        elif match.group(3) is not None:
            lower = (int(match.group(1)), int(match.group(2)))
            upper = (int(match.group(3)), int(match.group(4)))
            if upper < lower:
                self._error(
                    node,
                    "python version range must end at or after its start",
                    code="K-PACKAGE-PYTHON",
                )
    return package


def _walk_auto_directive(self: _Walker, node: Node) -> AutoSpec:
    body_node = _field(node, "body")
    if body_node is None:
        return AutoSpec(prompt="", loc=self._span(node))
    prompt_lines: List[str] = []
    for tline in _named_children(body_node):
        if tline.type != "template_line":
            continue
        # Auto prompts are free text — render the raw source minus
        # trailing newline, then dedent to remove the body's leading
        # indentation.
        expr = next(
            (c for c in _named_children(tline) if c.type == "template_expr"),
            None,
        )
        if expr is None:
            prompt_lines.append("")
        else:
            prompt_lines.append(_src_text(self.source, expr).rstrip())
    # Strip common leading indent (the auto body is indented under
    # the directive header; the user's prompt content typically lives
    # at one consistent indent level).
    dedented = dedent_block(prompt_lines)
    prompt = "\n".join(dedented).strip("\n")
    return AutoSpec(prompt=prompt, loc=self._span(node))


def _walk_model_directive(self: _Walker, node: Node) -> ModelDirectiveStmt:
    plain, code = self._walk_simple_directive_value(node)
    return ModelDirectiveStmt(plain_value=plain, backtick_code=code, loc=self._span(node))


def _walk_agent_directive(self: _Walker, node: Node) -> AgentDirectiveStmt:
    plain, code, command_plain, command_code = self._walk_agent_directive_parts(node)
    return AgentDirectiveStmt(
        plain_value=plain,
        backtick_code=code,
        command_plain_value=command_plain,
        command_backtick_code=command_code,
        loc=self._span(node),
    )


def _walk_agent_directive_parts(
    self: _Walker,
    node: Node,
) -> tuple[str | None, str | None, str | None, str | None]:
    body_node = _field(node, "body")
    if body_node is None:
        plain, code = self._walk_simple_directive_value(node)
        return plain, code, None, None

    fields = [child for child in _named_children(body_node) if child.type == "agent_field"]
    if len(fields) != 1:
        self._error(
            body_node,
            "Multiline '> agent:' must contain exactly one agent entry",
            code="K-AGENT-BODY",
        )
        return None, None, None, None
    field = fields[0]
    name_node = _field(field, "name")
    value_node = _field(field, "value")
    assert name_node is not None
    assert value_node is not None
    plain = _identifier_name(self.source, name_node)
    if value_node.type == "inline_python_expr":
        code_node = _field(value_node, "code")
        assert code_node is not None
        return plain, None, None, _src_text(self.source, code_node)
    return plain, None, _src_text(self.source, value_node).strip(), None


def _walk_adapter_directive(self: _Walker, node: Node) -> AdapterDirectiveStmt:
    plain, code = self._walk_simple_directive_value(node)
    return AdapterDirectiveStmt(plain_value=plain, backtick_code=code, loc=self._span(node))


def _walk_simple_directive_value(self: _Walker, node: Node) -> tuple[str | None, str | None]:
    value_node = _field(node, "value")
    assert value_node is not None
    if value_node.type == "inline_python_expr":
        code_node = _field(value_node, "code")
        assert code_node is not None
        return None, _src_text(self.source, code_node)
    return _src_text(self.source, value_node).strip(), None


def _walk_effort_directive(self: _Walker, node: Node) -> EffortDirectiveStmt:
    value_node = _field(node, "value")
    assert value_node is not None
    if value_node.type == "inline_python_expr":
        code_node = _field(value_node, "code")
        assert code_node is not None
        return EffortDirectiveStmt(
            backtick_code=_src_text(self.source, code_node),
            loc=self._span(node),
        )
    plain = _src_text(self.source, value_node).strip()
    self._validate_effort_plain_value(value_node, plain)
    return EffortDirectiveStmt(plain_value=plain, loc=self._span(node))


def _walk_approval_directive(self: _Walker, node: Node) -> ApprovalDirectiveStmt:
    plain, code = self._walk_simple_directive_value(node)
    if plain is not None:
        value_node = _field(node, "value")
        assert value_node is not None
        self._validate_approval_plain_value(value_node, plain)
    return ApprovalDirectiveStmt(plain_value=plain, backtick_code=code, loc=self._span(node))


def _walk_hooks_directive(self: _Walker, node: Node) -> HooksDirectiveStmt:
    value_node = _field(node, "value")
    if value_node is not None:
        value = _src_text(self.source, value_node).strip()
        if value not in {"enabled", "disabled"}:
            self._error(
                value_node,
                "hooks directive must be 'enabled' or 'disabled'",
                code="K-HOOKS-VALUE",
            )
        return HooksDirectiveStmt(enabled=value == "enabled", loc=self._span(node))

    body_node = _field(node, "body")
    assert body_node is not None
    fields: list[SettingsField] = []
    for child in _named_children(body_node):
        if child.type in ("line_comment", "block_comment"):
            continue
        field = self._walk_settings_value_field(child)
        if field.name not in HOOK_EVENT_NAMES:
            expected = ", ".join(HOOK_EVENT_NAMES)
            self._error(
                child,
                f"Unknown hook event {field.name!r}; expected one of: {expected}",
                code="K-HOOKS-EVENT",
            )
            continue
        fields.append(field)
    return HooksDirectiveStmt(enabled=True, fields=fields, loc=self._span(node))


def _walk_history_directive(self: _Walker, node: Node) -> HistoryDirectiveStmt:
    value_node = _field(node, "value")
    if value_node is not None:
        value = _src_text(self.source, value_node).strip()
        if value not in {"enabled", "disabled"}:
            self._error(
                value_node,
                "history directive must be 'enabled' or 'disabled'",
                code="K-HISTORY-VALUE",
            )
        return HistoryDirectiveStmt(enabled=value == "enabled", loc=self._span(node))

    body_node = _field(node, "body")
    assert body_node is not None
    enabled: bool | None = None
    fields: list[SettingsField] = []
    seen: set[str] = set()
    for child in _named_children(body_node):
        if child.type in ("line_comment", "block_comment"):
            continue
        field = self._walk_settings_value_field(child)
        name = field.name
        if name not in {"enabled", "compaction_mode", "compaction_threshold"}:
            self._error(
                child,
                "history setting must be one of: enabled, compaction_mode, compaction_threshold",
                code="K-HISTORY-KEY",
            )
            continue
        if name in seen:
            self._error(
                child,
                f"Duplicate history setting: {name!r}",
                code="K-HISTORY-DUP",
            )
            continue
        seen.add(name)
        if name == "enabled":
            if field.backtick_code is not None:
                self._error(
                    child,
                    "history enabled setting must be the literal 'true' or 'false'",
                    code="K-HISTORY-ENABLED",
                )
                enabled = False
                continue
            value = (field.plain_value or "").strip()
            if value not in {"true", "false"}:
                self._error(
                    child,
                    "history enabled setting must be 'true' or 'false'",
                    code="K-HISTORY-ENABLED",
                )
            enabled = value == "true"
            continue
        if (
            name == "compaction_mode"
            and field.plain_value is not None
            and field.plain_value not in {"native", "disabled"}
        ):
            self._error(
                child,
                "history compaction_mode must be 'native' or 'disabled'",
                code="K-COMPACTION-MODE",
            )
        fields.append(field)
    if enabled is None:
        self._error(
            body_node,
            "history block requires an 'enabled: true' or 'enabled: false' setting",
            code="K-HISTORY-ENABLED-REQUIRED",
        )
    return HistoryDirectiveStmt(
        enabled=enabled is True,
        fields=fields,
        loc=self._span(node),
    )


def _walk_skills_directive(self: _Walker, node: Node) -> SkillsDirectiveStmt:
    value_node = _field(node, "value")
    if value_node is not None:
        value = _src_text(self.source, value_node).strip()
        if value not in {"enabled", "disabled"}:
            self._error(
                value_node,
                "skills directive must be 'enabled' or 'disabled'",
                code="K-SKILLS-VALUE",
            )
        return SkillsDirectiveStmt(enabled=value == "enabled", loc=self._span(node))

    body_node = _field(node, "body")
    assert body_node is not None
    enabled: bool | None = None
    fields: list[SettingsField] = []
    seen: set[str] = set()
    allowed = {
        "enabled",
        "cwd",
        "max_skills",
        "include_registry",
        "include_all",
        "exclude_paths",
    }
    boolean_fields = {"enabled", "include_registry", "include_all"}
    for child in _named_children(body_node):
        if child.type in ("line_comment", "block_comment"):
            continue
        field = self._walk_settings_value_field(child)
        name = field.name
        if name not in allowed:
            self._error(
                child,
                "skills setting must be one of: " + ", ".join(sorted(allowed)),
                code="K-SKILLS-KEY",
            )
            continue
        if name in seen:
            self._error(
                child,
                f"Duplicate skills setting: {name!r}",
                code="K-SKILLS-DUP",
            )
            continue
        seen.add(name)
        if name in boolean_fields and field.backtick_code is None:
            value = (field.plain_value or "").strip()
            if value not in {"true", "false"}:
                self._error(
                    child,
                    f"skills {name} setting must be 'true' or 'false'",
                    code="K-SKILLS-BOOL",
                )
        if name == "enabled":
            if field.backtick_code is not None:
                self._error(
                    child,
                    "skills enabled setting must be the literal 'true' or 'false'",
                    code="K-SKILLS-ENABLED",
                )
                enabled = False
            else:
                enabled = (field.plain_value or "").strip() == "true"
            continue
        fields.append(field)
    if enabled is None:
        self._error(
            body_node,
            "skills block requires an 'enabled: true' or 'enabled: false' setting",
            code="K-SKILLS-ENABLED-REQUIRED",
        )
    return SkillsDirectiveStmt(
        enabled=enabled is True,
        fields=fields,
        loc=self._span(node),
    )


def _walk_codemode_directive(self: _Walker, node: Node) -> CodeModeDirectiveStmt:
    value_node = _field(node, "value")
    if value_node is not None:
        value = _src_text(self.source, value_node).strip()
        if value not in {"enabled", "disabled"}:
            self._error(
                value_node,
                "codemode directive must be 'enabled' or 'disabled'",
                code="K-CODEMODE-VALUE",
            )
        return CodeModeDirectiveStmt(enabled=value == "enabled", loc=self._span(node))

    body_node = _field(node, "body")
    assert body_node is not None
    enabled = True
    fields: list[SettingsField] = []
    seen: set[str] = set()
    for child in _named_children(body_node):
        if child.type in ("line_comment", "block_comment"):
            continue
        field = self._walk_settings_value_field(child)
        name = field.name
        if name not in CODEMODE_SETTING_FIELDS:
            self._error(
                child,
                "codemode setting must be one of: " + ", ".join(sorted(CODEMODE_SETTING_FIELDS)),
                code="K-CODEMODE-KEY",
            )
            continue
        if name in seen:
            self._error(
                child,
                f"Duplicate codemode setting: {name!r}",
                code="K-CODEMODE-DUP",
            )
            continue
        seen.add(name)
        if name == "enabled":
            if field.backtick_code is not None:
                self._error(
                    child,
                    "codemode enabled setting must be the literal 'true' or 'false'",
                    code="K-CODEMODE-ENABLED",
                )
                enabled = False
            else:
                value = (field.plain_value or "").strip()
                if value not in {"true", "false"}:
                    self._error(
                        child,
                        "codemode enabled setting must be 'true' or 'false'",
                        code="K-CODEMODE-ENABLED",
                    )
                enabled = value == "true"
            continue
        fields.append(field)
    return CodeModeDirectiveStmt(enabled=enabled, fields=fields, loc=self._span(node))


def _walk_system_directive(self: _Walker, node: Node) -> SystemDirectiveStmt:
    head = _field(node, "head")
    body = _field(node, "body")
    lines: List[TemplateLine] = []
    if head is not None:
        lines.append(self._system_line_from_expr(head))
    if body is not None:
        for child in _named_children(body):
            if child.type != "system_line":
                continue
            expr_node = _field(child, "line")
            if expr_node is not None:
                lines.append(self._system_line_from_expr(expr_node))
    if lines:
        plain = "\n".join(
            "".join(seg.text for seg in line.segments if isinstance(seg, TextSegment)).strip()
            for line in lines
        ).strip()
        return SystemDirectiveStmt(
            plain_value=plain or None,
            template_lines=lines,
            loc=self._span(node),
        )

    value_node = _field(node, "value")
    if value_node is None:
        return SystemDirectiveStmt(loc=self._span(node))
    if value_node.type == "inline_python_expr":
        code_node = _field(value_node, "code")
        assert code_node is not None
        return SystemDirectiveStmt(
            backtick_code=_src_text(self.source, code_node),
            loc=self._span(node),
        )
    plain = _src_text(self.source, value_node).strip()
    return SystemDirectiveStmt(plain_value=plain, loc=self._span(node))


def _walk_mcp_directive(self: _Walker, node: Node) -> McpDirectiveStmt:
    body_node = _field(node, "body")
    assert body_node is not None
    fields: list[McpField] = []
    for child in _named_children(body_node):
        if child.type in ("line_comment", "block_comment"):
            continue
        fields.append(self._walk_mcp_field(child))
    return McpDirectiveStmt(fields=fields, loc=self._span(node))


def _walk_mcp_field(self: _Walker, node: Node) -> McpField:
    name_node = _field(node, "name")
    value_node = _field(node, "value")
    assert name_node is not None
    assert value_node is not None
    name = _identifier_name(self.source, name_node)
    if value_node.type == "inline_python_expr":
        code_node = _field(value_node, "code")
        assert code_node is not None
        return McpField(
            name=name,
            backtick_code=_src_text(self.source, code_node),
            loc=self._span(node),
        )
    plain = _src_text(self.source, value_node).strip()
    if name == "transport":
        self._validate_mcp_transport_plain_value(value_node, plain)
    return McpField(
        name=name,
        plain_value=plain,
        loc=self._span(node),
    )


def _walk_settings_directive(self: _Walker, node: Node) -> SettingsDirectiveStmt:
    body_node = _field(node, "body")
    assert body_node is not None
    fields: list[SettingsField] = []
    seen: set[str] = set()
    for child in _named_children(body_node):
        if child.type in ("line_comment", "block_comment"):
            continue
        setting = self._walk_settings_field(child)
        if setting.name in seen:
            self._error(
                child,
                f"Duplicate settings field: {setting.name!r}",
                code="K-SETTING-DUP",
            )
        seen.add(setting.name)
        fields.append(setting)
    return SettingsDirectiveStmt(fields=fields, loc=self._span(node))


def _walk_settings_field(self: _Walker, node: Node) -> SettingsField:
    field = self._walk_settings_value_field(node)
    name_node = _field(node, "name")
    assert name_node is not None
    self._validate_setting_key(name_node, field.name)
    if (
        field.name == "tool_reason"
        and field.plain_value is not None
        and field.plain_value not in {"enabled", "disabled"}
    ):
        self._error(
            node,
            "tool_reason must be enabled, disabled, or a Python boolean expression",
            code="K-SETTING-VALUE",
        )
    return field


def _walk_settings_value_field(self: _Walker, node: Node) -> SettingsField:
    name_node = _field(node, "name")
    value_node = _field(node, "value")
    assert name_node is not None
    assert value_node is not None
    name = _identifier_name(self.source, name_node)
    if value_node.type == "inline_python_expr":
        code_node = _field(value_node, "code")
        assert code_node is not None
        return SettingsField(
            name=name,
            backtick_code=_src_text(self.source, code_node),
            loc=self._span(node),
        )
    return SettingsField(
        name=name,
        plain_value=_src_text(self.source, value_node).strip(),
        loc=self._span(node),
    )


def _walk_artifacts_directive(self: _Walker, node: Node) -> ArtifactsDirectiveStmt:
    body_node = _field(node, "body")
    assert body_node is not None
    fields: list[ArtifactsField] = []
    seen: set[str] = set()
    for child in _named_children(body_node):
        if child.type in ("line_comment", "block_comment"):
            continue
        artifact_field = self._walk_artifacts_field(child)
        if artifact_field.name in seen:
            self._error(
                child,
                f"Duplicate artifacts field: {artifact_field.name!r}",
                code="K-ARTIFACTS-DUP",
            )
        seen.add(artifact_field.name)
        fields.append(artifact_field)
    return ArtifactsDirectiveStmt(fields=fields, loc=self._span(node))


def _walk_artifacts_field(self: _Walker, node: Node) -> ArtifactsField:
    name_node = _field(node, "name")
    value_node = _field(node, "value")
    assert name_node is not None
    assert value_node is not None
    name = _identifier_name(self.source, name_node)
    if name not in ARTIFACT_POLICY_INPUT_FIELDS:
        choices = ", ".join(sorted(ARTIFACT_POLICY_INPUT_FIELDS))
        self._error(
            name_node,
            f"Unknown artifacts field {name!r}; expected one of: {choices}",
            code="K-ARTIFACTS-FIELD",
        )
    if value_node.type == "inline_python_expr":
        code_node = _field(value_node, "code")
        assert code_node is not None
        return ArtifactsField(
            name=name,
            backtick_code=_src_text(self.source, code_node),
            loc=self._span(node),
        )
    plain_value = _src_text(self.source, value_node).strip()
    if name in ARTIFACT_POLICY_INPUT_FIELDS:
        try:
            ArtifactPolicy.patch({name: parse_plain_setting_value(plain_value)})
        except ArtifactPolicyError as exc:
            self._error(
                value_node,
                str(exc),
                code="K-ARTIFACTS-VALUE",
            )
    return ArtifactsField(
        name=name,
        plain_value=plain_value,
        loc=self._span(node),
    )


def _walk_use_directive(self: _Walker, node: Node) -> UseDirectiveStmt:
    body_node = _field(node, "body")
    if body_node is not None:
        tool_names = self._walk_use_tool_names(body_node)
        return UseDirectiveStmt(tool_names=tool_names, loc=self._span(node))
    name_node = _field(node, "name")
    assert name_node is not None
    return UseDirectiveStmt(
        target=_identifier_name(self.source, name_node),
        loc=self._span(node),
    )


def _walk_use_tool_names(self: _Walker, body_node: Node) -> list[str]:
    tool_names: list[str] = []
    for child in _named_children(body_node):
        if child.type == "use_tool_name":
            name_node = _field(child, "name")
            assert name_node is not None
            tool_names.append(_identifier_name(self.source, name_node))
        elif child.type == "use_tool_backtick":
            name_node = _field(child, "name")
            assert name_node is not None
            tool_names.append(_identifier_name(self.source, name_node))
    if not tool_names:
        self._error(
            body_node,
            "Multiline '> use:' must list at least one tool",
            code="K-USE-EMPTY",
        )
    return tool_names
