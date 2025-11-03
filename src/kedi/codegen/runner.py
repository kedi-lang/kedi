from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from ..lang import parse_program
from ..lang.ast import ProcedureDef
from .interfaces import CodegenConfig
from .manager import CodegenManager
from .mock_agents import MockPassingAgent
from .pydantic_ai_agent import PydanticAIAgent


def _reconstruct_header(proc: ProcedureDef) -> str:
    params_parts: List[str] = []
    for p in proc.params:
        t = None
        if proc.param_type_annotations is not None:
            t = proc.param_type_annotations.get(p)
        params_parts.append(f"{p}: {t}" if t else p)
    header = f"@{proc.name}(" + ", ".join(params_parts) + ")"
    if proc.return_type_annotation:
        header += f" -> {proc.return_type_annotation}"
    header += ":"
    return header


def _select_agent(kind: str, model: str | None):
    k = (kind or "pydantic_ai").lower()
    if k == "mock":
        return MockPassingAgent()
    return PydanticAIAgent(model=model)


def run_codegen(
    source_path: Path,
    *,
    agent_kind: str = "pydantic_ai",
    model: str | None = None,
    max_retries: int = 5,
    use_cache: bool = True,
) -> bool:
    text = Path(source_path).read_text(encoding="utf-8")
    prog = parse_program(text)
    procs: List[ProcedureDef] = prog.procedures
    mgr = CodegenManager(
        _select_agent(agent_kind, model),
        CodegenConfig(max_retries=max_retries),
        use_cache=use_cache,
    )
    success_all = True
    for proc in procs:
        # Check for new auto_spec first, fall back to legacy ai_prompt
        auto_spec = getattr(proc, "auto_spec", None)
        if auto_spec:
            prompt = auto_spec.prompt
        else:
            prompt = getattr(proc, "ai_prompt", None)
        
        if not prompt:
            continue
        
        # Always process AI-prompted procedures; cache may already exist and be skipped by manager
        header = _reconstruct_header(proc)
        ok = mgr.orchestrate_generation(Path(source_path), header, prompt, spec="")
        success_all = success_all and ok
    return success_all
