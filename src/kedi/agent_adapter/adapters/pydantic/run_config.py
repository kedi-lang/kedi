from __future__ import annotations

from collections.abc import Callable
from typing import (
    TYPE_CHECKING,
    Any,
    Mapping,
    Sequence,
    cast,
)

from pydantic_ai import (
    Agent,
)
from pydantic_ai.models import Model
from pydantic_ai.tools import (
    RunContext,
)

from kedi.agent_adapter.adapters.pydantic.capabilities import (
    _KediArtifactCapability,
    _KediRequiredToolCapability,
    _KediTelemetryCapability,
)
from kedi.agent_adapter.adapters.pydantic.compat import effective_model
from kedi.agent_adapter.adapters.pydantic.hooks import _KediHookCapability
from kedi.agent_adapter.adapters.pydantic.tools import _approval_required_toolset, _toolset_sequence
from kedi.agent_adapter.codemode import CodeModeLimits
from kedi.agent_adapter.codemode.pydantic import (
    PydanticCodeModeCapability,
    pydantic_code_mode_capability,
)
from kedi.agent_adapter.compaction.native_settings import with_pydantic_native_compaction
from kedi.agent_adapter.conversation import AdapterConversationContext
from kedi.agent_adapter.hooks import current_hook_dispatcher
from kedi.agent_adapter.prefix_cache import has_response_cache_settings, pydantic_cache_settings
from kedi.agent_adapter.request import current_agent_run_request
from kedi.agent_adapter.tools.scope import current_artifact_admission
from kedi.agent_adapter.tools.specs import ToolSpec
from kedi.agent_profile import normalize_codemode_settings
from kedi.agent_profile.settings import pydantic_model_settings
from kedi.effort import pydantic_effort_value
from kedi.run_budget import assert_run_budget_supported, current_run_budget

if TYPE_CHECKING:
    from kedi.agent_adapter.adapters.pydantic.adapter import PydanticAdapter


def _responses_session_model(
    agent: Agent[Any, Any],
    model: Model | str | None,
) -> Model | None:
    if not (
        hasattr(agent, "_model")
        and hasattr(agent, "_override_root_capability")
        and callable(getattr(agent, "_get_model_outside_run", None))
    ):
        return cast("Model", model) if callable(getattr(model, "responses_session", None)) else None
    selected_model = effective_model(agent, model)
    return selected_model if callable(getattr(selected_model, "responses_session", None)) else None


def _run_kwargs(
    self: PydanticAdapter[Any, Any],
    kwargs: Mapping[str, Any],
    *,
    tool_specs: Sequence[ToolSpec],
) -> dict[str, object]:
    run_kwargs: dict[str, object] = dict(kwargs)
    request = current_agent_run_request()
    if request is not None and request.prompt.kind == "claim":
        selected = effective_model(self, cast("Model | str | None", run_kwargs.get("model")))
        if getattr(selected, "system", None) == "typesafe" and not getattr(
            selected, "supports_decision_threshold", False
        ):
            raise ValueError(
                "Kedi Jev claims require kedi_typesafe.TypeSafeModel to enforce typesafe_threshold; "
                "the upstream TypeSafeModel uses a fixed decision policy"
            )
    self._apply_default_usage_limits(run_kwargs)
    if self.active_profile.system is not None and run_kwargs.get("instructions") is None:
        run_kwargs["instructions"] = self.active_profile.system
    model_settings = pydantic_model_settings(self.active_profile.settings)
    if self.active_profile.effort is not None:
        model_settings["thinking"] = pydantic_effort_value(self.active_profile.effort)
    if model_settings:
        existing_settings = run_kwargs.get("model_settings")
        if callable(existing_settings):
            run_kwargs["model_settings"] = _profile_model_settings(
                cast(
                    "Callable[[RunContext[Any]], Mapping[str, Any]]",
                    existing_settings,
                ),
                model_settings,
            )
        elif isinstance(existing_settings, Mapping):
            run_kwargs["model_settings"] = {**existing_settings, **model_settings}
        else:
            run_kwargs["model_settings"] = model_settings
    conversation = self._subagent_conversation_var.get()
    self._apply_conversation_history(run_kwargs, conversation)
    existing_settings = run_kwargs.get("model_settings")
    adapter_conversation = (
        conversation if isinstance(conversation, AdapterConversationContext) else None
    )
    needs_cache_settings = (
        adapter_conversation is not None
        or callable(existing_settings)
        or (
            isinstance(existing_settings, Mapping)
            and has_response_cache_settings(cast("Mapping[str, Any]", existing_settings))
        )
    )
    if not needs_cache_settings:
        selected_model = None
    else:
        selected_model = effective_model(
            self,
            cast("Model | str | None", run_kwargs.get("model")),
        )
    if callable(existing_settings) and selected_model is not None:
        run_kwargs["model_settings"] = _cached_model_settings(
            cast(
                "Callable[[RunContext[Any]], Mapping[str, Any]]",
                existing_settings,
            ),
            model=selected_model,
            conversation=adapter_conversation,
        )
    elif selected_model is not None:
        configured_settings = self.model_settings
        cache_settings = pydantic_cache_settings(
            {
                **(
                    cast("Mapping[str, Any]", configured_settings)
                    if isinstance(configured_settings, Mapping)
                    else {}
                ),
                **(
                    cast("Mapping[str, Any]", existing_settings)
                    if isinstance(existing_settings, Mapping)
                    else {}
                ),
            },
            model=selected_model,
            conversation=adapter_conversation,
        )
        if cache_settings:
            run_kwargs["model_settings"] = cache_settings
    from kedi.tool_context import inside_kedi_tool_call

    toolsets = self._toolsets_for_run(
        include_scoped_tools=(
            self._isolated_run_depth_for_current_adapter().get() > 0 or not inside_kedi_tool_call()
        )
    )
    caller_toolsets = _toolset_sequence(run_kwargs.get("toolsets"))
    budget = current_run_budget()
    assert_run_budget_supported(
        self,
        opaque_tools=bool(caller_toolsets)
        or bool(self.active_profile.mcp_servers)
        or any((tool.metadata or {}).get("source") == "external" for tool in tool_specs)
        or bool(
            budget is not None
            and budget.has_limit("tool_attempt")
            and self._has_native_approval_tools()
        ),
    )
    if caller_toolsets or toolsets:
        run_kwargs["toolsets"] = [
            *(
                _approval_required_toolset(
                    toolset,
                    adapter_shortname=self.shortname,
                )
                for toolset in caller_toolsets
            ),
            *(toolsets or ()),
        ]
    capabilities = run_kwargs.get("capabilities")
    if self.active_history_processor is not None:
        from .history_processing import reject_conflicting_history_capabilities

        reject_conflicting_history_capabilities(capabilities)
        reject_conflicting_history_capabilities(self._agent_init_kwargs.get("capabilities"))
    if self.active_tool_reason:
        from .reason import ToolReasonCapability

        capabilities = _with_capability(
            capabilities, ToolReasonCapability, lambda: ToolReasonCapability(self)
        )
    approval_resolution = self.active_approval_resolution
    dispatcher = current_hook_dispatcher()
    if any((tool.metadata or {}).get("source") != "external" for tool in tool_specs) or (
        dispatcher is not None
        and any(
            dispatcher.handlers_for(event)
            for event in (
                "pre_tool_use",
                "post_tool_use",
                "post_tool_use_failure",
                "post_tool_result_failure",
            )
        )
    ):
        capabilities = _with_capability(
            capabilities,
            _KediHookCapability,
            lambda: _KediHookCapability(self),
        )
    if current_artifact_admission() is not None:
        capabilities = _with_capability(
            capabilities,
            _KediArtifactCapability,
            lambda: _KediArtifactCapability(self),
        )
    codemode = normalize_codemode_settings(
        self.active_profile.codemode or getattr(self, "codemode", None)
    )
    if codemode is not None and codemode.enabled:
        if approval_resolution == "external":
            raise ValueError(
                "Pydantic CodeMode requires inline Kedi approval resolution; "
                "external deferred approval is not supported inside execute_code"
            )
        capabilities = _with_capability(
            capabilities,
            PydanticCodeModeCapability,
            lambda: pydantic_code_mode_capability(
                limits=CodeModeLimits(**codemode.limit_values()),
                preload_tools=codemode.preload_tools,
            ),
        )
    if approval_resolution == "kedi" and (
        caller_toolsets or toolsets is not None or self._has_native_approval_tools()
    ):
        capabilities = self._with_deferred_approval_capability(capabilities)
    elif approval_resolution == "external" and (
        caller_toolsets or toolsets is not None or self._has_native_approval_tools()
    ):
        capabilities = self._with_external_approval_capability(capabilities)
    if any((tool.metadata or {}).get("required_before_output") is True for tool in tool_specs):
        capabilities = _with_capability(
            capabilities,
            _KediRequiredToolCapability,
            _KediRequiredToolCapability,
        )
    compaction = self.active_profile.compaction
    if (
        self.active_profile.history is True
        and compaction is not None
        and compaction.mode == "native"
    ):
        selected_model = effective_model(
            self,
            cast("Model | str | None", run_kwargs.get("model")),
        )
        capabilities = with_pydantic_native_compaction(
            capabilities,
            model=selected_model,
            settings=compaction,
        )
    capabilities = self._with_telemetry_capability(capabilities)
    if capabilities:
        run_kwargs["capabilities"] = capabilities
    return run_kwargs


def _apply_default_usage_limits(
    self: PydanticAdapter[Any, Any], run_kwargs: dict[str, object]
) -> None:
    default_limits = getattr(self, "_default_usage_limits", None)
    if "usage_limits" not in run_kwargs and default_limits is not None:
        run_kwargs["usage_limits"] = default_limits


def _with_telemetry_capability(
    self: PydanticAdapter[Any, Any], existing: object | None
) -> list[Any]:
    from kedi.telemetry import owns_native_spans

    if existing is None:
        capabilities: list[Any] = []
    elif isinstance(existing, (list, tuple)):
        capabilities = list(existing)
    else:
        raise TypeError("Pydantic AI capabilities must be a list or tuple.")
    if (
        owns_native_spans("pydantic", "run_agent")
        and self.native_telemetry_enabled("run_agent")
        and not any(isinstance(capability, _KediTelemetryCapability) for capability in capabilities)
    ):
        capabilities.append(_KediTelemetryCapability())
    return capabilities


def _with_capability(
    existing: object | None,
    capability_type: type[Any],
    factory: Callable[[], Any],
) -> list[Any]:
    if existing is None:
        capabilities: list[Any] = []
    elif isinstance(existing, (list, tuple)):
        capabilities = list(existing)
    else:
        raise TypeError("Pydantic AI capabilities must be a list or tuple.")
    if not any(isinstance(capability, capability_type) for capability in capabilities):
        capabilities.append(factory())
    return capabilities


def _profile_model_settings(
    settings_factory: Callable[[RunContext[Any]], Mapping[str, Any]],
    profile_settings: Mapping[str, Any],
) -> Callable[[RunContext[Any]], dict[str, Any]]:
    def merged(ctx: RunContext[Any]) -> dict[str, Any]:
        return {**settings_factory(ctx), **profile_settings}

    return merged


def _cached_model_settings(
    settings_factory: Callable[[RunContext[Any]], Mapping[str, Any]],
    *,
    model: object,
    conversation: AdapterConversationContext | None,
) -> Callable[[RunContext[Any]], dict[str, Any]]:
    def merged(ctx: RunContext[Any]) -> dict[str, Any]:
        return pydantic_cache_settings(
            settings_factory(ctx),
            model=model,
            conversation=conversation,
        )

    return merged


def _agent_for_produce(self: PydanticAdapter[Any, Any]) -> Agent[Any, Any]:
    from kedi.tool_context import inside_kedi_tool_call

    if not inside_kedi_tool_call() and self._isolated_run_depth_for_current_adapter().get() == 0:
        return self
    # Keep nested Kedi tool calls isolated from the outer agent's scoped
    # Kedi tools. Native constructor and inherited toolsets are supplied
    # below through the normal per-run toolset surface with approval.
    init_kwargs = {
        name: value
        for name, value in self._agent_init_kwargs.items()
        if name not in {"tools", "toolsets"}
    }
    return Agent(self._agent_model_arg, **init_kwargs)
