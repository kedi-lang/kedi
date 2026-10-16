from __future__ import annotations

from concurrent.futures import Future
from contextlib import nullcontext
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Dict,
    List,
    Sequence,
    cast,
)

from pydantic import Field

from kedi.core.expressions import (
    CallExpr,
    ComputedExpr,
    Expression,
    InputExpr,
    OutputExpr,
    ValueEnv,
)
from kedi.decisions import (
    DecisionInput,
    DecisionOutput,
    DecisionRecord,
    DecisionReference,
    DecisionSource,
    collect_decision_evidence,
    fingerprint_text,
    fingerprint_value,
    new_call_id,
    publish_decision_record,
)
from kedi.errors import KediExecutionError
from kedi.promise import KediPromise

if TYPE_CHECKING:
    from kedi.agent_adapter.request import AgentRunRequest
    from kedi.core.runtime import KediRuntime


_CLAIM_OUTPUT_DESCRIPTION = (
    "Whether the claim is established. Treat current context and relevant tool results as "
    "authoritative, including when they conflict with general knowledge; otherwise use reliable "
    "existing knowledge. False if contradicted or unknown."
)


@dataclass(frozen=True, slots=True)
class _PromptInputSnapshot:
    text: str
    identity: DecisionInput


def _snapshot_prompt_input(name: str, value: Any) -> _PromptInputSnapshot:
    from kedi.lang.compiler.environment import model_value_and_text

    model_value, text = model_value_and_text(value)
    fingerprint = fingerprint_value(model_value)
    return _PromptInputSnapshot(
        text=text,
        identity=DecisionInput(
            name=name,
            fingerprint=fingerprint,
            supported=fingerprint is not None,
        ),
    )


def _invoke_text_parts(
    self: KediRuntime,
    parts: Sequence[Any],
    deps: Sequence[Any],
    env: ValueEnv,
    *,
    capture_key: str | None,
) -> Dict[str, Any]:
    if capture_key is not None:
        from kedi.lang.compiler.environment import assert_not_cli_args_identifier

        assert_not_cli_args_identifier(capture_key)
    parallel = bool(getattr(self._engine, "parallel", False))

    from kedi.agent_profile import current_block_agent_profile

    block_profile = current_block_agent_profile().frozen_copy()
    adapter = self._concrete_adapter(self._resolve_adapter(block_profile))
    captured_tool_specs = self._resolve_agent_tool_specs(block_profile, adapter)
    self._assert_profile_override_supported(block_profile, adapter)
    conversation_state, conversation_ticket, lane_identity = self._reserve_conversation_turn(
        block_profile, adapter, captured_tool_specs
    )

    def _call_invoke() -> str:
        resolved_parts = [p.resolve() if isinstance(p, KediPromise) else p for p in parts]
        prompt = "".join(resolved_parts)

        from kedi.agent_adapter.meta import SupportsProfileOverride
        from kedi.agent_adapter.request import RawPromptBuilder, TextOutput, lower_request_sync
        from kedi.agent_adapter.tools.specs import SupportsToolRegistration

        profile = block_profile
        override_ctx = nullcontext()
        if isinstance(adapter, SupportsProfileOverride):
            override_ctx = adapter.profile_override(profile)

        tool_adapter: Any = None
        tool_ctx: Any = nullcontext()
        if isinstance(adapter, SupportsToolRegistration):
            tool_adapter = adapter
            tool_ctx = tool_adapter.tool_scope()

        with tool_ctx:
            if tool_adapter is not None:
                for spec in captured_tool_specs:
                    tool_adapter.register_tool(spec)
            instruction_parts = self._agent_instruction_parts(profile, adapter)

            def invoke_request(request: AgentRunRequest) -> str:
                return str(lower_request_sync(adapter, request))

            with override_ctx:
                return self._run_conversation_call(
                    adapter,
                    RawPromptBuilder(content=prompt).build(),
                    invoke_request,
                    instructions=instruction_parts,
                    tools=captured_tool_specs,
                    profile=profile,
                    call_kind="invoke",
                    output=TextOutput(),
                    conversation_state=conversation_state,
                    conversation_ticket=conversation_ticket,
                    lane_identity=lane_identity,
                    retain_failed_turn=parallel,
                )

    if not parallel:
        try:
            response = _call_invoke()
        except BaseException:
            self._cancel_conversation_turn(conversation_state, conversation_ticket)
            raise
        if capture_key is None:
            return {}
        env[capture_key] = response
        return {capture_key: response}

    from kedi.lang.compiler.environment import snapshot_visible_values

    env_snapshot = snapshot_visible_values(env)

    def _job() -> Dict[str, Any]:
        token = self._current_env.set(env_snapshot)
        try:
            response = _call_invoke()
            return {capture_key: response} if capture_key is not None else {}
        except KediExecutionError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise self.build_execution_error(exc) from None
        finally:
            self._current_env.reset(token)

    try:
        future = self._engine.schedule(deps, _job)
    except BaseException:
        self._cancel_conversation_turn(conversation_state, conversation_ticket)
        raise
    future.add_done_callback(
        lambda completed: self._retire_conversation_turn_after_failed_future(
            conversation_state,
            conversation_ticket,
            completed,
        )
    )
    self._register_outstanding(future)
    if capture_key is None:
        return {}
    view = KediPromise(future, capture_key)
    env[capture_key] = view
    return {capture_key: view}


def _evaluate_claim(self: KediRuntime, claim: str) -> bool:
    """Classify a rendered control-flow claim through the active agent lane."""
    from kedi.agent_adapter.meta import SupportsProfileOverride
    from kedi.agent_adapter.tools.specs import SupportsToolRegistration
    from kedi.agent_profile import current_block_agent_profile

    profile = current_block_agent_profile().frozen_copy()
    adapter = self._concrete_adapter(self._resolve_adapter(profile))
    captured_tool_specs = self._resolve_agent_tool_specs(profile, adapter)
    self._assert_profile_override_supported(profile, adapter)
    from kedi.agent_adapter.request import (
        CLAIM_INSTRUCTION,
        ClaimPromptBuilder,
        InstructionPart,
        TypedOutput,
        lower_request_sync,
    )

    question = ClaimPromptBuilder(claim=claim).build()

    override_ctx = nullcontext()
    if isinstance(adapter, SupportsProfileOverride):
        override_ctx = adapter.profile_override(profile)

    tool_adapter: Any = None
    tool_ctx: Any = nullcontext()
    if isinstance(adapter, SupportsToolRegistration):
        tool_adapter = adapter
        tool_ctx = tool_adapter.tool_scope()

    with tool_ctx:
        if tool_adapter is not None:
            for spec in captured_tool_specs:
                tool_adapter.register_tool(spec)
        conversation_state, conversation_ticket, lane_identity = self._reserve_conversation_turn(
            profile, adapter, captured_tool_specs
        )
        try:
            instruction_parts = (
                *self._agent_instruction_parts(profile, adapter),
                InstructionPart(
                    identity="runtime.claim",
                    source="runtime",
                    content=CLAIM_INSTRUCTION,
                ),
            )
            claim_output_type = cast(
                type[Any],
                Annotated[bool, Field(description=_CLAIM_OUTPUT_DESCRIPTION)],
            )

            def produce_claim(request: AgentRunRequest) -> bool:
                result = lower_request_sync(adapter, request)
                if type(result) is not bool:
                    raise TypeError(
                        "Kedi template condition must return an exact bool; "
                        f"received {type(result).__name__}"
                    )
                return result

            with override_ctx:
                return self._run_conversation_call(
                    adapter,
                    question,
                    produce_claim,
                    instructions=instruction_parts,
                    tools=captured_tool_specs,
                    profile=profile,
                    call_kind="produce",
                    output=TypedOutput(output_type=claim_output_type),
                    conversation_state=conversation_state,
                    conversation_ticket=conversation_ticket,
                    lane_identity=lane_identity,
                )
        except BaseException:
            self._cancel_conversation_turn(conversation_state, conversation_ticket)
            raise


def _kedi_run(
    self: KediRuntime, expressions: Sequence[Expression], env: ValueEnv
) -> Dict[str, Any]:
    NotFound = object()
    # Prompt parts may include unresolved promises (dependency edges); collect their Futures
    # so the produce job is scheduled only after they resolve.
    parts: List[Any] = []
    outputs: List[OutputExpr] = []
    deps: List[Any] = []
    input_snapshots: list[_PromptInputSnapshot | KediPromise] = []
    inputs_complete = True
    for expr in expressions:
        if isinstance(expr, InputExpr):
            value = self._lookup_value(expr.key, env)
            if isinstance(value, KediPromise):
                captured = value.map(lambda item, key=expr.key: _snapshot_prompt_input(key, item))
                text = captured.map(lambda item: item.text)
                input_snapshots.append(captured)
            else:
                captured = _snapshot_prompt_input(expr.key, value)
                text = captured.text
                input_snapshots.append(captured)
            out = None
        else:
            text, out = self._evaluate_expression(expr, env)
            if isinstance(expr, (CallExpr, ComputedExpr)):
                inputs_complete = False
        if out is not None:
            outputs.append(out)
        if text is not None:
            parts.append(text)
            if isinstance(text, KediPromise):
                deps.append(text._future)

    required = [out.key for out in outputs]
    if required:
        from kedi.lang.compiler.environment import assert_not_cli_args_identifier

        for key in required:
            assert_not_cli_args_identifier(key)

    if not required:
        if not parts:
            return {}
        return self._invoke_text_parts(parts, deps, env, capture_key=None)

    # Build output schema mapping (default to str when unspecified)
    typed_fields: Dict[str, Any] = {
        out.key: (out.pytype if out.pytype is not None else str) for out in outputs
    }
    explicit_typed_fields = {out.key for out in outputs if out.pytype is not None}
    binding_contracts = {out.key: out.binding_contract for out in outputs}
    record_future: Future[DecisionRecord | None] = Future()
    call_id = new_call_id()
    frames = self.current_trace_frames()
    span = frames[-1].span if frames else None
    decision_source = DecisionSource(
        path=None if span is None else span.source_path,
        line=None if span is None else span.start_line,
        column=None if span is None else span.start_col,
    )

    def declare_output(key: str, value: Any) -> None:
        from kedi.lang.compiler.environment import BindingMetadata, ScopeFrame

        if isinstance(env, ScopeFrame):
            env.declare(
                key,
                value,
                BindingMetadata(
                    type_contract=binding_contracts[key],
                    decision=DecisionReference(record_future, (key,)),
                ),
            )
        else:
            env[key] = value

    parallel = bool(getattr(self._engine, "parallel", False))

    # Resolve tool specs NOW, on the main thread, while the lexical scope is correct. A
    # deferred job runs after the defining scope (e.g. a nested-procedure tool) has exited, so
    # re-resolving by name from rt._procedures then would bind the wrong procedure. Building
    # the spec here captures the compiled procedure fn for both registration AND invocation.
    from kedi.agent_profile import current_block_agent_profile

    block_profile = current_block_agent_profile().frozen_copy()
    adapter = self._concrete_adapter(self._resolve_adapter(block_profile))
    captured_tool_specs = self._resolve_agent_tool_specs(block_profile, adapter)
    # Validate the profile override on the main thread too, so it raises synchronously at the
    # call site in BOTH modes — symmetric with the tool guard above, instead of surfacing late
    # at drain (wrapped, differently typed) when _produce runs on a parallel worker.
    self._assert_profile_override_supported(block_profile, adapter)
    conversation_state, conversation_ticket, lane_identity = self._reserve_conversation_turn(
        block_profile, adapter, captured_tool_specs
    )

    def _produce() -> Dict[str, Any]:
        # Resolve any promise prompt parts (already complete by the time this job runs in the
        # parallel engine; concrete already in sequential), then build the prompt + call the
        # model + validate — all inside the captured context so profile/tools are correct.
        resolved_parts = [p.resolve() if isinstance(p, KediPromise) else p for p in parts]
        resolved_inputs = [
            item.resolve() if isinstance(item, KediPromise) else item for item in input_snapshots
        ]

        from kedi.agent_adapter.meta import SupportsProfileOverride, SupportsTemplatePromptMode
        from kedi.agent_adapter.request import (
            KediTemplatePromptBuilder,
            SchemaOutput,
            lower_request_sync,
        )
        from kedi.agent_adapter.tools.specs import SupportsToolRegistration

        # Support was already validated synchronously on the main thread
        # (_assert_profile_override_supported); here we only enter the override scope.
        profile = block_profile
        override_ctx = nullcontext()
        if isinstance(adapter, SupportsProfileOverride):
            override_ctx = adapter.profile_override(profile)

        # Every tool-capable model call gets its own scope, including calls with no Kedi
        # tools. This prevents a nested model call from inheriting its caller's registrations.
        tool_adapter: Any = None
        tool_ctx: Any = nullcontext()
        if isinstance(adapter, SupportsToolRegistration):
            tool_adapter = adapter
            tool_ctx = tool_adapter.tool_scope()

        captured_evidence: list[Any] = []

        def produce_and_validate(request: AgentRunRequest) -> Dict[str, Any]:
            with collect_decision_evidence() as evidence:
                result = lower_request_sync(adapter, request)
            captured_evidence.extend(evidence)
            out_map: Dict[str, Any] = {}
            artifact_policy = profile.artifacts
            for key in required:
                value = getattr(result, key, NotFound)
                if value is NotFound:
                    raise ValueError(f"LLM did not return required key: {key}")
                validated = self._validate_adapter_output(
                    value,
                    typed_fields[key],
                    explicit_type=key in explicit_typed_fields,
                )
                out_map[key] = (
                    self.artifacts.artifactize(
                        validated,
                        policy=artifact_policy,
                        source=f"output:{key}",
                    )
                    if artifact_policy is not None
                    else validated
                )
            return out_map

        with tool_ctx:
            if tool_adapter is not None:
                for spec in captured_tool_specs:
                    tool_adapter.register_tool(spec)
            prompt_builder = KediTemplatePromptBuilder(
                template="".join(resolved_parts),
                output_fields=tuple(required),
                mode=(
                    adapter.template_prompt_mode(profile)
                    if isinstance(adapter, SupportsTemplatePromptMode)
                    else "completion"
                ),
            )
            prompt = prompt_builder.build()
            instruction_parts = (
                *prompt_builder.instruction_parts(),
                *self._agent_instruction_parts(profile, adapter),
            )

            with override_ctx:
                result_map = self._run_conversation_call(
                    adapter,
                    prompt,
                    produce_and_validate,
                    instructions=instruction_parts,
                    tools=captured_tool_specs,
                    profile=profile,
                    call_kind="produce",
                    output=SchemaOutput(schema=typed_fields),
                    conversation_state=conversation_state,
                    conversation_ticket=conversation_ticket,
                    lane_identity=lane_identity,
                    retain_failed_turn=parallel,
                )
        if not captured_evidence:
            record_future.set_result(None)
            return result_map

        unique_inputs: dict[str, DecisionInput] = {}
        complete_input_set = inputs_complete
        for item in resolved_inputs:
            previous = unique_inputs.get(item.identity.name)
            if previous is not None and previous != item.identity:
                complete_input_set = False
            unique_inputs[item.identity.name] = item.identity
        output_identities: list[DecisionOutput] = []
        for key in required:
            output_fingerprint = fingerprint_value(result_map[key])
            output_identities.append(
                DecisionOutput(
                    field_path=(key,),
                    fingerprint=output_fingerprint,
                    supported=output_fingerprint is not None,
                )
            )
        record = DecisionRecord(
            call_id=call_id,
            source=decision_source,
            prompt_fingerprint=fingerprint_text(prompt.content),
            inputs=tuple(unique_inputs.values()),
            inputs_complete=complete_input_set,
            outputs=tuple(output_identities),
            evidence=captured_evidence[-1],
        )
        publish_decision_record(record)
        record_future.set_result(record)
        return result_map

    def finish_without_evidence() -> None:
        if not record_future.done():
            record_future.set_result(None)

    # Explicit sequential execution avoids a worker and returns concrete values.
    if not parallel:
        try:
            result_map = _produce()  # raises on the main thread, wrapped by the trace frame
        except BaseException:
            finish_without_evidence()
            self._cancel_conversation_turn(conversation_state, conversation_ticket)
            raise
        for key in required:
            declare_output(key, result_map[key])
        return {k: env[k] for k in required}

    # Parallel path. Snapshot the value-env by VALUE so the deferred worker sees the env as of
    # schedule time; copy_context() snapshots only the contextvar binding, leaving the dict
    # shared by reference, which would let later main-thread writes leak into a scheduled job.
    # Every per-call adapter scope (tool/profile ContextVars) is isolated per worker by the
    # copy_context() snapshot, so even tool-using templates pipeline safely.
    from kedi.lang.compiler.environment import snapshot_visible_values

    env_snapshot = snapshot_visible_values(env)

    def _job() -> Dict[str, Any]:
        # The job runs on a worker AFTER the originating template's trace frame has exited on the
        # main thread; attribute failures to the captured frames (carried via copy_context) so
        # errors still point at the template line.
        token = self._current_env.set(env_snapshot)
        try:
            return _produce()
        except KediExecutionError:
            raise
        except Exception as exc:  # noqa: BLE001
            # build_execution_error keeps the provider error on `.original`, so the job-manager
            # retry policy can still classify it (a 429/timeout stays retryable) even though the
            # job surfaces the wrapped KediExecutionError. `from None` avoids re-chaining it.
            raise self.build_execution_error(exc) from None
        finally:
            self._current_env.reset(token)

    try:
        future = self._engine.schedule(deps, _job)
    except BaseException:
        finish_without_evidence()
        self._cancel_conversation_turn(conversation_state, conversation_ticket)
        raise

    def _finish_parallel_produce(completed: Future[Any]) -> None:
        # The engine can cancel or fail the Future before `_job` starts. In that case none of
        # `_job`'s exception handlers run, so settle the evidence reference here as well.
        finish_without_evidence()
        self._retire_conversation_turn_after_failed_future(
            conversation_state,
            conversation_ticket,
            completed,
        )

    future.add_done_callback(_finish_parallel_produce)
    self._register_outstanding(future)
    views: Dict[str, Any] = {key: KediPromise(future, key) for key in required}
    for key in required:
        declare_output(key, views[key])
    return dict(views)
