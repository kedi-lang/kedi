from __future__ import annotations

import os
import subprocess
import sys

import pytest
from lsprotocol import types as lsp

import kedi.lsp.features.capabilities as lsp_capabilities_module
import kedi.lsp.features.capability_state as capability_state_module
from kedi.agent_adapter.meta import AdapterCapabilities
from kedi.agent_adapter.requirements import (
    MissingAdapterCapabilityError,
    check_adapter_requirements,
    normalize_adapter_requirements,
    validate_adapter_requirements,
)
from kedi.lang import compile_program, parse_program
from kedi.lang.ast import RequiresDirectiveStmt
from kedi.lang.parser.diagnostics import KediParseError
from kedi.lsp.features.diagnostics import compute_diagnostics
from tests.mock_adapter import MockAdapter


def _adapter(*, stream_events: bool) -> MockAdapter:
    adapter = MockAdapter()
    adapter.capabilities = AdapterCapabilities(
        supports_structured_output=True,
        supports_tool_registration=True,
        supports_profile_override=True,
        supports_stream_events=stream_events,
    )
    return adapter


def test_requires_parses_single_and_block_forms_with_stable_deduplication() -> None:
    program = parse_program(
        """
> requires: stream_events
> requires:
    tool_registration
    stream_events
"""
    )

    first, second = program.toplevel_body
    assert isinstance(first, RequiresDirectiveStmt)
    assert isinstance(second, RequiresDirectiveStmt)
    assert first.requirements == ["stream_events"]
    assert second.requirements == ["tool_registration", "stream_events"]
    assert set(second.requirement_locs) == {"tool_registration", "stream_events"}


def test_requires_rejects_unknown_capability_from_canonical_vocabulary() -> None:
    with pytest.raises(KediParseError) as exc_info:
        parse_program("> requires: imaginary_capability\n")

    diagnostic = exc_info.value.diagnostics[0]
    assert diagnostic.code == "K-REQUIRES-UNKNOWN"
    assert "stream_events" in diagnostic.message
    assert "imaginary_capability" in diagnostic.message


def test_profile_requirements_are_additive_and_preserve_declaration_origins() -> None:
    runtime = compile_program(
        parse_program(
            """
> requires: tool_registration
> profile: streaming:
    > requires:
        stream_events
        tool_registration
> use: streaming

@inspect():
    >> Inspection result is [answer].
"""
        ),
        adapter=_adapter(stream_events=True),
    )

    procedure = runtime._procedures["inspect"].__kedi_procedure_def__
    profile = procedure.captured_agent_profile
    assert profile.requirements == ("tool_registration", "stream_events")
    assert len(profile.requirement_origins["tool_registration"]) == 2
    assert len(profile.requirement_origins["stream_events"]) == 1


def test_unsupported_requirement_fails_before_model_io() -> None:
    adapter = _adapter(stream_events=False)
    runtime = compile_program(
        parse_program("> requires: stream_events\n>> Result is [answer].\n"),
        adapter=adapter,
    )

    with pytest.raises(MissingAdapterCapabilityError) as exc_info:
        runtime.run_main()

    assert exc_info.value.adapter_name == "mock"
    assert exc_info.value.requirement == "stream_events"
    assert exc_info.value.source_location is not None
    assert adapter.request_calls == []


def test_supported_requirement_allows_model_io() -> None:
    adapter = _adapter(stream_events=True)
    runtime = compile_program(
        parse_program("> requires: stream_events\n>> Result is [answer].\n"),
        adapter=adapter,
    )

    runtime.run_main()

    assert len(adapter.request_calls) == 1
    assert adapter.request_calls[0].endswith("Result is [answer].")


def test_procedure_requirement_is_checked_at_its_model_boundary() -> None:
    adapter = _adapter(stream_events=False)
    runtime = compile_program(
        parse_program(
            """
@inspect():
    > requires: stream_events
    >> Inspection result is [answer].

= `inspect()`
"""
        ),
        adapter=adapter,
    )

    with pytest.raises(MissingAdapterCapabilityError):
        runtime.run_main()

    assert adapter.request_calls == []


def test_requirement_validation_remains_active_under_python_optimized_mode() -> None:
    script = """
from kedi.agent_adapter.meta import AdapterCapabilities
from kedi.agent_adapter.requirements import MissingAdapterCapabilityError, validate_adapter_requirements

class Adapter:
    shortname = "optimized"
    capabilities = AdapterCapabilities(supports_stream_events=False)

try:
    validate_adapter_requirements(adapter=Adapter(), requirements=("stream_events",))
except MissingAdapterCapabilityError:
    raise SystemExit(0)
raise SystemExit(1)
"""
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}

    completed = subprocess.run(
        [sys.executable, "-O", "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode == 0, completed.stderr


def test_normalize_requirements_rejects_invalid_python_inputs() -> None:
    assert normalize_adapter_requirements(None) == ()
    assert normalize_adapter_requirements(["stream_events", "stream_events"]) == ("stream_events",)
    with pytest.raises(TypeError, match="sequence"):
        normalize_adapter_requirements("stream_events")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="only strings"):
        normalize_adapter_requirements(["stream_events", 1])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="unknown adapter requirement"):
        normalize_adapter_requirements(["imaginary"])


def test_requirement_checks_preserve_unresolved_state_and_last_origin() -> None:
    first_origin = object()
    last_origin = object()

    checks = check_adapter_requirements(
        capabilities=None,
        requirements=("stream_events",),
        origins={"stream_events": (first_origin, last_origin)},
    )

    assert len(checks) == 1
    assert checks[0].status == "unresolved"
    assert checks[0].source_location is last_origin


def test_requirement_checks_keep_unknown_capability_state_unresolved() -> None:
    checks = check_adapter_requirements(
        capabilities=AdapterCapabilities(supports_stream_events=None),
        requirements=("stream_events",),
    )

    assert checks[0].status == "unresolved"


def test_only_explicitly_negotiated_requirements_may_defer_runtime_validation() -> None:
    adapter = MockAdapter()
    adapter.capabilities = AdapterCapabilities(supports_structured_output=None)
    with pytest.raises(MissingAdapterCapabilityError):
        validate_adapter_requirements(adapter=adapter, requirements=("structured_output",))

    adapter.capabilities = AdapterCapabilities(
        supports_structured_output=None,
        negotiated_capabilities=frozenset({"structured_output"}),
    )
    checks = validate_adapter_requirements(adapter=adapter, requirements=("structured_output",))
    assert checks[0].status == "unresolved"

    adapter.capabilities = AdapterCapabilities(
        supports_structured_output=False,
        negotiated_capabilities=frozenset({"structured_output"}),
    )
    with pytest.raises(MissingAdapterCapabilityError):
        validate_adapter_requirements(adapter=adapter, requirements=("structured_output",))


@pytest.mark.parametrize(
    "source",
    [
        "> adapter: pydantic\n> requires: stream_events\n",
        "> requires: stream_events\n> adapter: pydantic\n",
        ("> profile: streaming:\n    > adapter: pydantic\n    > requires: stream_events\n"),
    ],
)
def test_lsp_reports_explicit_unsupported_requirements_as_errors(
    source: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unsupported = AdapterCapabilities(supports_stream_events=False)
    monkeypatch.setattr(
        lsp_capabilities_module,
        "adapter_capabilities",
        lambda _name: unsupported,
    )
    monkeypatch.setattr(
        capability_state_module,
        "adapter_capabilities",
        lambda _name: unsupported,
    )

    diagnostics = [
        diagnostic
        for diagnostic in compute_diagnostics(source, "main.kedi")
        if diagnostic.code == "K-ADAPTER-REQUIREMENT"
    ]

    assert len(diagnostics) == 1
    assert diagnostics[0].severity == lsp.DiagnosticSeverity.Error
    assert diagnostics[0].range.start.line == source.splitlines().index(
        next(line for line in source.splitlines() if "requires" in line)
    )
    assert "stream_events" in diagnostics[0].message


def test_lsp_defers_unknown_static_capability_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        capability_state_module,
        "adapter_capabilities",
        lambda _name: AdapterCapabilities(),
    )
    diagnostics = compute_diagnostics(
        "> adapter: pydantic\n> requires: stream_events\n",
        "main.kedi",
    )

    assert all(item.code != "K-ADAPTER-REQUIREMENT" for item in diagnostics)


def test_lsp_defers_dynamic_adapter_requirement_to_runtime() -> None:
    diagnostics = compute_diagnostics(
        "> adapter: `adapter_name`\n> requires: stream_events\n",
        "main.kedi",
    )

    assert all(item.code != "K-ADAPTER-REQUIREMENT" for item in diagnostics)


def test_lsp_requirement_helpers_skip_supported_and_unlocated_requirements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = capability_state_module._AdapterDiagnosticState(shortname="test")
    span = parse_program("> requires: stream_events\n").toplevel_body[0].loc
    assert span is not None

    monkeypatch.setattr(
        lsp_capabilities_module,
        "_adapter_capability_status",
        lambda *_args: True,
    )
    assert (
        lsp_capabilities_module._explicit_requirement_diagnostics(
            state,
            entries=(("stream_events", span),),
        )
        == []
    )

    monkeypatch.setattr(
        lsp_capabilities_module,
        "_adapter_capability_status",
        lambda *_args: False,
    )
    assert (
        lsp_capabilities_module._explicit_requirement_diagnostics(
            state,
            entries=(("stream_events", None),),
        )
        == []
    )

    entry = ("stream_events", span)
    assert capability_state_module._merge_requirement_entries((entry,), (entry,)) == (entry,)
