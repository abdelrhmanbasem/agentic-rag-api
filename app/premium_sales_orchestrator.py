# app/premium_sales_orchestrator.py
# Adaptive premium sales reasoning orchestrator.
#
# This is the "breathtaking when needed" layer.
# It does:
# - mode-aware broad retrieval
# - evidence judgment
# - premium answer generation
# - critic pass
# - safe revision if needed

from typing import Dict, Any, List, Tuple

from app.llm import chat_text, chat_json, model_for_tier
from app.premium_retrieval import retrieve_premium_evidence
from app.premium_prompts import (
    build_language_rule,
    build_premium_sales_system_prompt,
    build_premium_context,
    build_evidence_judge_prompt,
    build_answer_critic_prompt,
)
from app.intelligence_modes import choose_intelligence_mode
from app.variables import extract_variables, apply_variable_patch


def safe_json_result(value: Any, fallback: Dict[str, Any]) -> Dict[str, Any]:
    return value if isinstance(value, dict) else fallback


def judge_evidence(
    *,
    model: str,
    user_message: str,
    variables: Dict[str, Any],
    knowledge: List[Dict[str, Any]],
    memories: List[Dict[str, Any]],
) -> Dict[str, Any]:
    prompt = build_evidence_judge_prompt(
        user_message=user_message,
        variables=variables,
        knowledge=knowledge,
        memories=memories,
    )

    result = chat_json(
        model,
        [{"role": "user", "content": prompt}],
        max_tokens=500,
    )

    return safe_json_result(
        result,
        {
            "enough_to_answer": bool(knowledge or variables),
            "confidence": 0.6 if knowledge or variables else 0.3,
            "missing_facts": [],
            "conflicting_facts": [],
            "best_evidence_titles": [],
            "answer_risk": "medium",
            "should_ask_followup": not bool(knowledge or variables),
            "notes": "Fallback evidence judgment.",
        },
    )


def generate_premium_answer(
    *,
    model: str,
    assistant: Dict[str, Any],
    user_message: str,
    summary: str,
    variables: Dict[str, Any],
    recent_messages: List[Dict[str, Any]],
    knowledge: List[Dict[str, Any]],
    memories: List[Dict[str, Any]],
    mode_decision: Dict[str, Any],
    evidence_judgment: Dict[str, Any],
) -> str:
    mode = mode_decision.get("mode", "premium_sales")

    system_prompt = (
        build_premium_sales_system_prompt(assistant=assistant, mode=mode)
        + "\n"
        + build_language_rule(user_message)
    )

    context = build_premium_context(
        user_message=user_message,
        summary=summary,
        variables=variables,
        recent_messages=recent_messages,
        knowledge=knowledge,
        memories=memories,
        mode_decision=mode_decision,
        evidence_judgment=evidence_judgment,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": context},
        {"role": "user", "content": user_message},
    ]

    return chat_text(model, messages, max_tokens=650)


def critique_answer(
    *,
    model: str,
    user_message: str,
    answer: str,
    variables: Dict[str, Any],
    knowledge: List[Dict[str, Any]],
) -> Dict[str, Any]:
    prompt = build_answer_critic_prompt(
        user_message=user_message,
        answer=answer,
        variables=variables,
        knowledge=knowledge,
    )

    result = chat_json(
        model,
        [{"role": "user", "content": prompt}],
        max_tokens=450,
    )

    return safe_json_result(
        result,
        {
            "passes": True,
            "unsupported_claims": [],
            "overpromises": [],
            "wrong_language": False,
            "did_not_answer_user": False,
            "too_pushy": False,
            "too_long": False,
            "revision_instruction": "",
        },
    )


def revise_answer(
    *,
    model: str,
    user_message: str,
    original_answer: str,
    critique: Dict[str, Any],
) -> str:
    revision_instruction = critique.get("revision_instruction") or "Fix the answer safely."

    messages = [
        {
            "role": "system",
            "content": (
                "Revise the answer according to the critic. "
                "Do not add unsupported facts. Keep the same language as the user. "
                "Return only the revised final answer."
            ),
        },
        {
            "role": "user",
            "content": f"""
Latest user message:
{user_message}

Original answer:
{original_answer}

Critique:
{critique}

Revision instruction:
{revision_instruction}
""",
        },
    ]

    return chat_text(model, messages, max_tokens=500)


def should_revise(critique: Dict[str, Any]) -> bool:
    if not critique:
        return False

    if critique.get("passes") is False:
        return True

    risky_keys = [
        "unsupported_claims",
        "overpromises",
    ]

    if any(critique.get(key) for key in risky_keys):
        return True

    if critique.get("wrong_language"):
        return True

    if critique.get("did_not_answer_user"):
        return True

    if critique.get("too_pushy"):
        return True

    return False


def run_adaptive_premium_turn(
    *,
    assistant: Dict[str, Any],
    assistant_id: str,
    user_id: str,
    schema: Dict[str, Any],
    variables: Dict[str, Any],
    recent_messages: List[Dict[str, Any]],
    summary: str,
    user_message: str,
    route: Dict[str, Any] | None = None,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Returns:
    (handled, result)

    handled=False means caller should continue existing normal pipeline.
    """

    route = route or {}
    variables = dict(variables or {})

    mode_decision = choose_intelligence_mode(
        message=user_message,
        variables=variables,
        route=route,
        schema=schema,
        assistant_id=assistant_id,
    )

    if not mode_decision.get("should_use_premium"):
        return False, {
            "mode_decision": mode_decision,
        }

    selected_tier = mode_decision.get("selected_model_tier", "normal")
    model = model_for_tier(selected_tier)

    extraction = extract_variables(
        schema=schema,
        existing_variables=variables,
        recent_messages=recent_messages,
        user_message=user_message,
    )

    updated_variables = apply_variable_patch(
        variables,
        extraction.get("updates", {}),
        extraction.get("deletions", []),
    )

    if extraction.get("intent"):
        updated_variables["intent"] = extraction.get("intent")

    retrieval = retrieve_premium_evidence(
        assistant_id=assistant_id,
        user_id=user_id,
        user_message=user_message,
        variables=updated_variables,
        mode=mode_decision.get("mode", "premium_sales"),
    )

    knowledge = retrieval.get("knowledge", [])
    memories = retrieval.get("memories", [])

    evidence_judgment = judge_evidence(
        model=model,
        user_message=user_message,
        variables=updated_variables,
        knowledge=knowledge,
        memories=memories,
    )

    answer = generate_premium_answer(
        model=model,
        assistant=assistant,
        user_message=user_message,
        summary=summary,
        variables=updated_variables,
        recent_messages=recent_messages,
        knowledge=knowledge,
        memories=memories,
        mode_decision=mode_decision,
        evidence_judgment=evidence_judgment,
    )

    critique = critique_answer(
        model=model,
        user_message=user_message,
        answer=answer,
        variables=updated_variables,
        knowledge=knowledge,
    )

    if should_revise(critique):
        answer = revise_answer(
            model=model,
            user_message=user_message,
            original_answer=answer,
            critique=critique,
        )

    result = {
        "answer": answer,
        "variables": updated_variables,
        "variable_updates": extraction.get("updates", {}),
        "variable_deletions": extraction.get("deletions", []),
        "missing_variables": extraction.get("missing_variables", []),
        "intent": updated_variables.get("intent", extraction.get("intent", "general_question")),
        "mode_decision": mode_decision,
        "route": {
            **route,
            "answer_mode": "adaptive_premium",
            "premium_mode": mode_decision.get("mode"),
            "needs_rag": True,
            "needs_memory": True,
            "rag_cache_hit": False,
            "reason": mode_decision.get("reason"),
        },
        "knowledge_used": knowledge,
        "knowledge_source": retrieval.get("knowledge_source", "none"),
        "memories_used": memories,
        "retrieval_queries": retrieval.get("queries", []),
        "evidence_judgment": evidence_judgment,
        "critique": critique,
        "model_used": model,
        "model_tier": selected_tier,
        "recommended_next_action": mode_decision.get("mode", "adaptive_premium"),
    }

    return True, result
