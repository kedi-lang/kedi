"""Interactive approval handling for Kedi CLI runs."""

from __future__ import annotations

import inspect
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Literal

from .approval import ApprovalDecision, ApprovalRequest, ToolRisk

CliApprovalChoice = Literal["allow", "deny", "always"]
_PromptResult = CliApprovalChoice | None | Awaitable[CliApprovalChoice | None]
_Prompt = Callable[[ApprovalRequest], _PromptResult]


@dataclass
class CliApprovalHandler:
    """Ask once per risky tool call, with session-local allow-always choices."""

    prompt: _Prompt | None = None
    is_interactive: Callable[[], bool] = lambda: sys.stdin.isatty() and sys.stdout.isatty()
    _always_allowed: set[tuple[str, ToolRisk]] = field(default_factory=set, init=False)

    def __call__(self, request: ApprovalRequest) -> ApprovalDecision:
        """Resolve synchronous tool calls without turning sync tools into coroutines."""
        key = (request.tool_name, request.risk)
        if key in self._always_allowed:
            return ApprovalDecision.allow(reason="allowed for this CLI run")
        if not self.is_interactive():
            return ApprovalDecision.deny(reason="interactive CLI approval requires a TTY")

        choice = self._ask(request)
        return self._decision(request, choice)

    async def resolve_async(self, request: ApprovalRequest) -> ApprovalDecision:
        """Resolve native adapter approvals inside their active event loop."""
        key = (request.tool_name, request.risk)
        if key in self._always_allowed:
            return ApprovalDecision.allow(reason="allowed for this CLI run")
        if not self.is_interactive():
            return ApprovalDecision.deny(reason="interactive CLI approval requires a TTY")

        choice = await self._ask_async(request)
        return self._decision(request, choice)

    def _decision(
        self,
        request: ApprovalRequest,
        choice: CliApprovalChoice | None,
    ) -> ApprovalDecision:
        if choice == "allow":
            return ApprovalDecision.allow(reason="approved once in CLI")
        if choice == "always":
            self._always_allowed.add((request.tool_name, request.risk))
            return ApprovalDecision.allow(reason="allowed for this CLI run")
        return ApprovalDecision.deny(reason="denied in CLI")

    def _ask(self, request: ApprovalRequest) -> CliApprovalChoice | None:
        if self.prompt is not None:
            choice = self.prompt(request)
            if inspect.isawaitable(choice):
                if inspect.iscoroutine(choice):
                    choice.close()
                raise RuntimeError("async CLI approval prompts require an async adapter")
            return _as_choice(choice)

        import questionary

        message = f"Allow {request.risk.replace('_', ' ')} tool '{request.tool_name}'?"
        choice = questionary.select(
            message,
            choices=[
                questionary.Choice("Deny", value="deny"),
                questionary.Choice("Allow once", value="allow"),
                questionary.Choice("Allow always for this run", value="always"),
            ],
        ).ask()
        return _as_choice(choice)

    async def _ask_async(self, request: ApprovalRequest) -> CliApprovalChoice | None:
        if self.prompt is not None:
            choice = self.prompt(request)
            if inspect.isawaitable(choice):
                choice = await choice
            return _as_choice(choice)

        import questionary

        message = f"Allow {request.risk.replace('_', ' ')} tool '{request.tool_name}'?"
        choice = await questionary.select(
            message,
            choices=[
                questionary.Choice("Deny", value="deny"),
                questionary.Choice("Allow once", value="allow"),
                questionary.Choice("Allow always for this run", value="always"),
            ],
        ).ask_async()
        return _as_choice(choice)


def _as_choice(value: object) -> CliApprovalChoice | None:
    if value == "allow":
        return "allow"
    if value == "deny":
        return "deny"
    if value == "always":
        return "always"
    return None
