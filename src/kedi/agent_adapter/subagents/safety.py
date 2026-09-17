from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from ...agent_profile import AgentProfile
from ...errors import KediExecutionError
from ..approvals.policy import ApprovalPolicy

_SANDBOX_RANK = {
    "read-only": 0,
    "workspace-write": 1,
    "danger-full-access": 2,
}


@dataclass(frozen=True, slots=True)
class SubagentSafetyCeiling:
    """Security limits inherited by every descendant in one delegation tree."""

    approval_policies: tuple[ApprovalPolicy | None, ...]
    cwd: Path
    sandbox: str | None = None

    @classmethod
    def from_parent(
        cls,
        profile: AgentProfile,
        *,
        default_cwd: Path,
    ) -> SubagentSafetyCeiling:
        settings = dict(profile.settings)
        cwd = _resolved_cwd(settings.get("cwd"), default=default_cwd)
        sandbox = _sandbox_value(settings.get("sandbox"))
        return cls(
            approval_policies=(profile.approval,),
            cwd=cwd,
            sandbox=sandbox,
        )

    def constrain_child(
        self,
        profile: AgentProfile,
    ) -> tuple[AgentProfile, SubagentSafetyCeiling]:
        settings = dict(profile.settings)
        child_cwd = _resolved_cwd(settings.get("cwd"), default=self.cwd)
        if not child_cwd.is_relative_to(self.cwd):
            raise KediExecutionError(
                f"Subagent cwd {str(child_cwd)!r} escapes parent cwd boundary {str(self.cwd)!r}"
            )

        child_sandbox = _sandbox_value(settings.get("sandbox"))
        effective_sandbox = _narrower_sandbox(self.sandbox, child_sandbox)
        if settings.get("cwd") is not None:
            settings["cwd"] = str(child_cwd)
        if self.sandbox is not None and child_sandbox is None:
            settings["sandbox"] = self.sandbox

        effective_profile = (
            profile if settings == dict(profile.settings) else replace(profile, settings=settings)
        )
        return effective_profile, SubagentSafetyCeiling(
            approval_policies=(*self.approval_policies, profile.approval),
            cwd=child_cwd,
            sandbox=effective_sandbox,
        )


def _resolved_cwd(value: object, *, default: Path) -> Path:
    if value is None:
        return default.resolve()
    if not isinstance(value, (str, Path)):
        raise TypeError(f"Subagent cwd must be str or Path, got {type(value).__name__}")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = default / path
    return path.resolve()


def _sandbox_value(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or value not in _SANDBOX_RANK:
        raise KediExecutionError(
            "Subagent sandbox must be read-only, workspace-write, or danger-full-access"
        )
    return value


def _narrower_sandbox(parent: str | None, child: str | None) -> str | None:
    if parent is None:
        return child
    if child is None:
        return parent
    if _SANDBOX_RANK[child] > _SANDBOX_RANK[parent]:
        raise KediExecutionError(
            f"Subagent sandbox {child!r} is broader than parent sandbox {parent!r}"
        )
    return child
