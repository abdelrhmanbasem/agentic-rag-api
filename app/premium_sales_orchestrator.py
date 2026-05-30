# app/premium_sales_orchestrator.py
# Adaptive premium sales reasoning orchestrator.
#
# This is the "breathtaking when needed" layer.
#
# It does:
# - mode-aware broad retrieval
# - evidence judgment
# - premium memory decision
# - premium answer generation
# - critic pass
# - safe revision if needed
#
# Important safety rule:
# The premium LLM is allowed to extract user-provided facts,
# but it is NOT allowed to directly write system-computed fields
# like lead_score, lead_temperature, workflow_stage, workflow, etc.

from typing import Dict, Any, List, Tuple

from app.llm import chat_text, chat_json, model_for_tier
from app.premium_retrieval import retrieve_premium_evidence
from app.premium_memory import build_premium_memory_decision
from app.premium_prompts import (
    build_language_rule,
    build_premium_sales_system_prompt,
    build_premium_context,
    build_evidence_judge_prompt,
    build_answer_critic_prompt,
)
from app.intelligence_modes import choose_intelligence_mode
from app.variables import extract_variables, apply_variable_patch


SYSTEM_COMPUTED_VARIABLES = {
    "lead_score",
    "lead_temperature",
    "lead_score_reasons",
    "workflow",
    "workflow_stage",
    "recommended_next_action",
    "next_best_action",
    "knowledge_source",
    "model_used",
    "model_tier",
    "token_usage",
    "memory_saved",
    "long_term_memories_written",
}


SERVICE_DECISION_VARIABLES = {
    "recommended_section",
    "service_needed",
    "customer_facing_section",
    "diagnostic_stage",
    "next_service_action",
}


def safe_json_result(value: Any, fallback: Dict[str, Any]) -> Dict[str, Any]:
    return value if isinstance(value, dict) else fallback


def clean_premium_extraction_updates(updates: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = dict(updates or {})

    for key in SYSTEM_COMPUTED_VARIABLES:
        cleaned.pop(key, None)

    for key in list(cleaned.keys()):
        if str(key).startswith("_"):
            cleaned.pop(key, None)

    return cleaned


def clean_premium_extraction_deletions(deletions: List[str]) -> List[str]:
    safe_deletions = []

    for key in deletions or []:
        if key in SYSTEM_COMPUTED_VARIABLES:
            continue
        if str(key).startswith("_"):
            continue
        safe_deletions.append(key)

    return safe_deletions


def ensure_service_decision_variables(
    variables: Dict[str, Any],
    answer: str,
    assistant_id: str = "",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Ensures the service assistant stores structured service decision variables.

    Why this exists:
    - The main brain may correctly say "قسم كشف الموتور" in the answer.
    - The booking sub-agent needs the internal variable "recommended_section".
    - This repair layer makes the main brain's visible decision explicit.

    Scope:
    - Only applies to service_center_agentic_rag.
    - Does not affect future assistants.
    - Does not make the booking sub-agent diagnose.
    """

    variables = dict(variables or {})
    assistant_id = str(assistant_id or "")

    if assistant_id != "service_center_agentic_rag":
        return variables, {}

    existing_recommended = variables.get("recommended_section")
    existing_customer = variables.get("customer_facing_section")

    if existing_recommended and existing_customer:
        return variables, {}

    symptoms = variables.get("symptoms", [])
    if isinstance(symptoms, list):
        symptoms_text = " ".join(str(item) for item in symptoms)
    else:
        symptoms_text = str(symptoms or "")

    text = (
        str(answer or "")
        + " "
        + str(variables.get("issue_description", ""))
        + " "
        + symptoms_text
    ).lower()

    section_map = [
        {
            "internal": "Engine Diagnostics",
            "customer": "قسم كشف الموتور",
            "keys": [
                "قسم كشف الموتور",
                "engine diagnostics",
                "overheating",
                "heat smell",
                "smell of overheating",
                "gauge rising",
                "سخونة",
                "بتسخن",
                "ريحة سخونة",
                "مؤشر الحرارة",
                "المؤشر بيعلى",
                "ريداتير",
                "ردياتير",
                "دورة التبريد",
                "مروحة الردياتير",
                "مروحة الريداتير",
            ],
        },
        {
            "internal": "AC Cooling",
            "customer": "قسم التكييف",
            "keys": [
                "قسم التكييف",
                "ac cooling",
                "weak ac",
                "تكييف",
                "مش بيبرد",
                "فريون",
                "كمبروسر",
                "مروحة كوندنسر",
            ],
        },
        {
            "internal": "Brakes & Safety",
            "customer": "قسم الفرامل",
            "keys": [
                "قسم الفرامل",
                "brakes",
                "brake",
                "فرامل",
                "تيل",
                "طنابير",
                "بتصفر",
                "صوت صفارة",
            ],
        },
        {
            "internal": "Electrical & Battery",
            "customer": "قسم الكهرباء والبطارية",
            "keys": [
                "قسم الكهرباء",
                "قسم الكهرباء والبطارية",
                "battery",
                "electrical",
                "بطارية",
                "دينامو",
                "مارش",
                "مش بتدور",
                "تك تك",
                "كهرباء",
            ],
        },
        {
            "internal": "Suspension & Steering",
            "customer": "قسم العفشة والدركسيون",
            "keys": [
                "قسم العفشة",
                "قسم العفشة والدركسيون",
                "suspension",
                "steering",
                "عفشة",
                "دركسيون",
                "رعشة",
                "اهتزاز",
                "تخبيط",
            ],
        },
        {
            "internal": "Tires & Alignment",
            "customer": "قسم الزوايا والكاوتش",
            "keys": [
                "قسم الزوايا",
                "قسم الزوايا والكاوتش",
                "tires",
                "alignment",
                "زوايا",
                "كاوتش",
                "ترصيص",
                "بتحدف",
            ],
        },
        {
            "internal": "Transmission",
            "customer": "قسم الفتيس",
            "keys": [
                "قسم الفتيس",
                "transmission",
                "gearbox",
                "فتيس",
                "نتشة",
                "نقلات",
                "غيار",
            ],
        },
        {
            "internal": "Quick Service",
            "customer": "قسم الصيانة السريعة",
            "keys": [
                "قسم الصيانة السريعة",
                "quick service",
                "صيانة سريعة",
                "تغيير زيت",
                "زيت",
                "فلتر",
            ],
        },
    ]

    updates: Dict[str, Any] = {}

    for item in section_map:
        if any(str(key).lower() in text for key in item["keys"]):
            if not variables.get("recommended_section"):
                variables["recommended_section"] = item["internal"]
                updates["recommended_section"] = item["internal"]

            if not variables.get("service_needed"):
                variables["service_needed"] = item["internal"]
                updates["service_needed"] = item["internal"]

            if not variables.get("customer_facing_section"):
                variables["customer_facing_section"] = item["customer"]
                updates["customer_facing_section"] = item["customer"]

            if not variables.get("diagnostic_stage"):
                variables["diagnostic_stage"] = "qualified"
                updates["diagnostic_stage"] = "qualified"

            if not variables.get("next_service_action"):
                variables["next_service_action"] = "offer_booking"
                updates["next_service_action"] = "offer_booking"

            break

    return variables, updates


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
            "confidence": 0.6 if knowledge or variables else 0.25,
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
    premium_memory_decision: Dict[str, Any],
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
        premium_memory_decision=premium_memory_decision,
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
            "too_generic": False,
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
                "Do not add unsupported facts. "
                "Keep the same language as the user. "
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

    if critique.get("too_generic"):
        return True

    return False


def build_premium_debug(
    *,
    mode_decision: Dict[str, Any],
    retrieval: Dict[str, Any],
    evidence_judgment: Dict[str, Any],
    critique: Dict[str, Any],
    premium_memory_decision: Dict[str, Any],
    service_decision_updates: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    knowledge = retrieval.get("knowledge", []) or []
    memories = retrieval.get("memories", []) or []

    return {
        "mode": mode_decision.get("mode"),
        "reason": mode_decision.get("reason"),
        "model_tier": mode_decision.get("selected_model_tier"),
        "retrieval_queries": retrieval.get("queries", []),
        "evidence_count": len(knowledge),
        "memory_count": len(memories),
        "evidence_confidence": evidence_judgment.get("confidence"),
        "evidence_enough": evidence_judgment.get("enough_to_answer"),
        "answer_risk": evidence_judgment.get("answer_risk"),
        "critic_passed": critique.get("passes"),
        "critic_too_generic": critique.get("too_generic", False),
        "memory_allow_long_term": premium_memory_decision.get("allow_long_term_memory"),
        "memory_suppress_long_term": premium_memory_decision.get("suppress_long_term_memory"),
        "memory_session_signals": premium_memory_decision.get("session_signals", []),
        "service_decision_updates": service_decision_updates or {},
    }


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

    raw_updates = extraction.get("updates", {}) or {}
    raw_deletions = extraction.get("deletions", []) or []

    extraction_updates = clean_premium_extraction_updates(raw_updates)
    extraction_deletions = clean_premium_extraction_deletions(raw_deletions)

    updated_variables = apply_variable_patch(
        variables,
        extraction_updates,
        extraction_deletions,
    )

    if extraction.get("intent"):
        updated_variables["intent"] = extraction.get("intent")

    premium_memory_decision = build_premium_memory_decision(
        user_message=user_message,
        variables=updated_variables,
        recent_messages=recent_messages,
        mode_decision=mode_decision,
    )

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
        premium_memory_decision=premium_memory_decision,
    )

    service_decision_updates: Dict[str, Any] = {}

    updated_variables, service_decision_updates = ensure_service_decision_variables(
        variables=updated_variables,
        answer=answer,
        assistant_id=assistant_id,
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

        updated_variables, revised_service_decision_updates = ensure_service_decision_variables(
            variables=updated_variables,
            answer=answer,
            assistant_id=assistant_id,
        )

        service_decision_updates.update(revised_service_decision_updates)

    combined_variable_updates = dict(extraction_updates or {})
    combined_variable_updates.update(service_decision_updates)

    premium_debug = build_premium_debug(
        mode_decision=mode_decision,
        retrieval=retrieval,
        evidence_judgment=evidence_judgment,
        critique=critique,
        premium_memory_decision=premium_memory_decision,
        service_decision_updates=service_decision_updates,
    )

    result = {
        "answer": answer,
        "variables": updated_variables,
        "variable_updates": combined_variable_updates,
        "variable_deletions": extraction_deletions,
        "raw_variable_updates": raw_updates,
        "raw_variable_deletions": raw_deletions,
        "service_decision_updates": service_decision_updates,
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
        "premium_memory_decision": premium_memory_decision,
        "premium_debug": premium_debug,
        "model_used": model,
        "model_tier": selected_tier,
        "recommended_next_action": (
            updated_variables.get("next_service_action")
            or mode_decision.get("mode", "adaptive_premium")
        ),
    }

    return True, result
