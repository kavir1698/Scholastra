"""Document-contract node: generate the whole-document coherence contract.

Runs once after outline approval and before research fan-out, using the full
outline, scope, audience and section purposes. A failure degrades to a contract
carrying the configured source policy (so downstream prompts still get policy
guidance) and is recorded as a blocking quality warning rather than silently
proceeding.
"""

from __future__ import annotations

from typing import Any

from theo_explore.agents.prompting import render, system_preamble
from theo_explore.config import load_source_policy
from theo_explore.graph.deps import Deps
from theo_explore.models.schemas import DocumentContract, NodeUsage, Outline, ScopeProposal
from theo_explore.sources.policy import policy_lines


def make_contract_node(deps: Deps):
    async def contract(state: dict[str, Any]) -> dict[str, Any]:
        outline = Outline.model_validate(state.get("outline") or {"title": "", "sections": []})
        scope = ScopeProposal.model_validate(state.get("scope") or {})
        policy = load_source_policy(deps.settings)
        lines = policy_lines(policy)
        prompt = render(
            "contract", scope=scope, outline=outline, source_policy_lines=lines
        )
        failure: str | None = None
        try:
            result, usage = await deps.llm.structured(
                "contract",
                DocumentContract,
                [{"role": "user", "content": prompt}],
                extra_system=system_preamble("Editorial Architect"),
                max_repairs=1,
            )
        except Exception as exc:  # noqa: BLE001 - degrade to a policy-only contract
            deps.log("contract_failed", error=str(exc))
            result, usage = DocumentContract(), NodeUsage()
            failure = f"contract generation failed: {exc}"

        result = result.model_copy(update={"source_policy": result.source_policy or lines})
        deps.workspace(state["project_id"]).save_contract(result)
        deps.log(
            "contract",
            glossary=len(result.glossary),
            arc=len(result.argument_arc),
            failed=bool(failure),
        )
        update: dict[str, Any] = {
            "contract": result.model_dump(),
            "status": "plan_research",
            **deps.budget_delta("contract", usage),
        }
        if failure:
            update["errors"] = [failure]
        return update

    return contract
