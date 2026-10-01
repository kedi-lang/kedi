import builtins

import pytest

import kedi
from kedi.agent_adapter import LangChainAdapter, PydanticAdapter
from kedi.agent_adapter.models import is_decision_model_id, laya_model
from kedi.agent_profile import AgentProfile

pytest.importorskip("kedi_laya")


@pytest.mark.parametrize("adapter", [PydanticAdapter, LangChainAdapter])
@pytest.mark.parametrize("model_id", ["laya-mlx/example/model", "laya/example/model"])
def test_laya_model_id_caches_model_without_loading_weights(adapter, model_id):
    pytest.importorskip("kedi_laya")
    instance = adapter(model_id)
    assert is_decision_model_id(model_id)
    assert instance._resolve_model(model_id) is instance._resolve_model(model_id)
    assert not instance.supports_artifact_contract(AgentProfile(model=model_id))


@pytest.mark.parametrize("model_id", ["laya/", "laya-mlx/ ", "other/model"])
def test_laya_model_id_requires_checkpoint(model_id):
    pytest.importorskip("kedi_laya")
    with pytest.raises(ValueError):
        laya_model(model_id, "pydantic")


def test_laya_rejects_unknown_adapter():
    with pytest.raises(ValueError, match="Pydantic or LangChain"):
        laya_model("laya/test", "other")


@pytest.mark.parametrize("missing", ["kedi_laya", "numpy"])
def test_laya_optional_dependency_errors_preserve_transitive_failures(monkeypatch, missing):
    original = builtins.__import__

    def imported(name, *args, **kwargs):
        if name == "kedi_laya":
            raise ModuleNotFoundError("missing dependency", name=missing)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", imported)
    with pytest.raises(
        ModuleNotFoundError, match="missing dependency" if missing == "numpy" else "kedi-laya"
    ):
        laya_model("laya/test", "pydantic")


@pytest.mark.parametrize("adapter", ["pydantic", "langchain"])
def test_kedi_capture_keeps_local_provider_confidence_and_typed_output(adapter, tmp_path):
    from kedi_laya import LayaClient

    class Predictor:
        def predict(self, state, questions):
            assert "duplicate charge" in str(state)
            return {
                "answers": {
                    key: {
                        "type": "choice",
                        "choice": "billing",
                        "confidence": 0.8,
                        "probabilities": {"billing": 0.9, "technical": 0.1},
                    }
                    for key in questions
                },
                "usage": {"input_tokens": 90, "output_tokens": 0},
            }

    source = tmp_path / "route.kedi"
    source.write_text(
        '>> The department for a duplicate charge is [department: Literal["billing", "technical"]].\n'
        "= ```\n"
        "from kedi import decision_info\n"
        "resolved = department\n"
        'info = decision_info("department")\n'
        'return {"department": resolved, "provider": info.provider, "confidence": info.confidence}\n'
        "```\n"
    )

    @kedi.bind(file=source)
    def route() -> dict: ...

    with LayaClient("test/checkpoint", predictor=Predictor()) as client:
        engine = (
            PydanticAdapter(client.as_pydantic_model())
            if adapter == "pydantic"
            else LangChainAdapter(client.as_langchain_model())
        )
        with kedi.context(adapter=engine, model=None):
            assert route() == {"department": "billing", "provider": "laya", "confidence": 0.8}


@pytest.mark.parametrize("selected", ["", ":\n  Probability\n  Rubric"])
def test_decisions_module_exports_shared_primitives_without_selecting_model(selected):
    from kedi_decisions import Probability, Rubric

    from kedi.lang import compile_program, parse_program
    from kedi.decisions import Probability as ExportedProbability
    from kedi.lsp.module_surface import load_imported_surface
    from kedi.module_registry import resolve_module_path

    assert ExportedProbability is Probability
    profile = AgentProfile(model="openai:unchanged")
    runtime = compile_program(
        parse_program(f"> import: decisions{selected}\n= imported\n"),
        default_agent_profile=profile,
    )
    assert runtime.run_main() == "imported"
    assert runtime.default_agent_profile.model == profile.model
    assert runtime._import_env["Probability"] is Probability
    assert runtime._import_env["Rubric"] is Rubric
    surface = load_imported_surface(
        resolve_module_path("decisions", source_path=None), set(), "decisions"
    )
    assert surface.exported_names() == {
        "BooleanCriteria",
        "ChoiceCriteria",
        "Probability",
        "Rubric",
    }


@pytest.mark.parametrize("adapter", [PydanticAdapter, LangChainAdapter])
def test_laya_object_binding_disables_artifact_tools(adapter):
    from kedi_laya import LayaClient

    with LayaClient("unloaded") as client:
        model = (
            client.as_pydantic_model()
            if adapter is PydanticAdapter
            else client.as_langchain_model()
        )
        assert not adapter(model).supports_artifact_contract(AgentProfile())


@pytest.mark.parametrize("adapter_type", [PydanticAdapter, LangChainAdapter])
@pytest.mark.parametrize("threshold,expected", [(0.85, True), (0.95, False)])
@pytest.mark.parametrize("claim", [False, True])
def test_laya_kedi_settings_control_claims_and_captures(adapter_type, threshold, expected, claim):
    from kedi_laya import LayaClient

    from kedi.lang import compile_program, parse_program

    class Predictor:
        def predict(self, state, questions):
            return {"answers": {key: {"type": "noul", "noul": 0.9} for key in questions}}

    source = f"> settings:\n  decision_threshold: {threshold}\n"
    if claim:
        source += "> if: This request is approved\n  = `True`\n> else:\n  = `False`\n"
    else:
        source += ">> Approval of this request is [approved: bool].\n= `bool(approved)`\n"
    with LayaClient("test", predictor=Predictor()) as client:
        model = (
            client.as_pydantic_model()
            if adapter_type is PydanticAdapter
            else client.as_langchain_model()
        )
        runtime = compile_program(parse_program(source), adapter=adapter_type(model))
        assert runtime.run_main() is expected
