# app/assistant_brain.py
# Universal assistant brain.
#
# Purpose:
# - Diagnose user state.
# - Choose next-best move.
# - Decide whether GPT is needed.
# - Produce deterministic premium answers when possible.
# - Work for this assistant and future assistants.
#
# Important:
# - This brain should make the assistant human-quality, not deceptive.
# - If asked whether it is AI/human, it must be honest through the normal system path.

import re
from typing import Dict, Any, List, Optional

from app.brain_types import BrainDecision, ComposeResult
from app.domain_playbooks import get_domain_playbook, infer_style_for_workflow
from app.objection_playbooks import build_objection_reply, detect_objection_type
from app.response_composer import (
    compose_premium_entry_reply,
    compose_fact_reply,
    compose_cta,
    get_selected_item,
)
from app.self_check import self_check_answer


def normalize_arabic(text: str) -> str:
    text = text or ""
    replacements = {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ى": "ي",
        "ة": "ه",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def normalize_text(text: str) -> str:
    return normalize_arabic((text or "").lower().strip())


def is_arabic_text(text: str) -> bool:
    return bool(re.search(r"[\u0600-\u06FF]", text or ""))


def has_any(text: str, markers: List[str]) -> bool:
    return any(marker in text for marker in markers)


def infer_workflow_from_state(schema: Dict[str, Any], assistant_id: str, variables: Dict[str, Any]) -> str:
    schema = schema or {}
    assistant_id = (assistant_id or "").lower()
    variables = variables or {}

    car_keys = [
        "car_brand",
        "car_condition",
        "transmission",
        "budget_max",
        "matched_car_model",
        "matched_car_year",
        "matched_car_km",
        "matched_car_price",
        "preferred_viewing_date",
        "preferred_viewing_time",
    ]

    service_keys = [
        "service_needed",
        "appointment_date",
        "appointment_time",
        "doctor_preference",
        "patient_name",
    ]

    if any(k in schema for k in car_keys) or any(k in variables for k in car_keys) or any(
        x in assistant_id for x in ["car", "cars", "auto", "vehicle", "dealer"]
    ):
        return "car_sales"

    if any(k in schema for k in service_keys) or any(k in variables for k in service_keys) or any(
        x in assistant_id for x in ["clinic", "doctor", "medical", "dental", "dentist", "health"]
    ):
        return "service_booking"

    if any(x in assistant_id for x in ["real_estate", "property", "rent", "broker"]):
        return "real_estate"

    if any(x in assistant_id for x in ["shop", "store", "ecommerce", "order"]):
        return "ecommerce"

    return "general"


def detect_fact_type(message: str) -> Optional[str]:
    text = normalize_text(message)

    if has_any(text, ["اوتوماتيك", "اتوماتيك", "أوتوماتيك", "مانيوال", "automatic", "manual", "transmission"]):
        return "transmission"

    if has_any(text, ["كام كيلو", "عاملة كام", "عامله كام", "ماشية كام", "ماشيه كام", "km", "mileage"]):
        return "km"

    if has_any(text, ["بكام", "سعر", "سعرها", "سعره", "price", "cost", "how much"]):
        return "price"

    return None


def detect_user_state(message: str, variables: Dict[str, Any]) -> Dict[str, Any]:
    text = normalize_text(message)
    variables = variables or {}
    signals = []

    objection = detect_objection_type(message)
    if objection:
        signals.append(f"objection:{objection}")
        if objection == "price":
            return {"state": "price_sensitive", "signals": signals}
        if objection == "comparison":
            return {"state": "comparing", "signals": signals}
        if objection == "trust":
            return {"state": "skeptical", "signals": signals}
        if objection == "hesitation":
            return {"state": "hesitant", "signals": signals}

    if has_any(text, ["احجز", "اشوفها", "أشوفها", "معاينة", "معاينه", "بكرة", "الساعة", "ميعاد", "موعد", "book", "schedule", "viewing"]):
        signals.append("ready_or_booking")
        return {"state": "ready", "signals": signals}

    if has_any(text, ["تنصحني", "رأيك", "رايك", "اخدها", "استنى", "استني", "تستاهل", "recommend", "should i", "worth"]):
        signals.append("advice_requested")
        return {"state": "advisor_needed", "signals": signals}

    if has_any(text, ["مش فاهم", "مش فاهمني", "مش قصدي", "wrong", "misunderstood"]):
        signals.append("repair_needed")
        return {"state": "confused", "signals": signals}

    if variables.get("budget_max") and (variables.get("selected_item") or variables.get("matched_car_model")):
        signals.append("qualified_with_item")
        return {"state": "qualified", "signals": signals}

    if variables.get("selected_item") or variables.get("matched_car_model"):
        signals.append("has_selected_item")
        return {"state": "interested", "signals": signals}

    return {"state": "curious", "signals": signals}


def should_use_gpt_for_message(message: str, variables: Dict[str, Any], workflow: str) -> bool:
    text = normalize_text(message)

    # Simple factual state questions should not use GPT.
    if detect_fact_type(message):
        return False

    advisor_markers = [
        "تنصحني",
        "رأيك",
        "رايك",
        "اخدها",
        "استنى",
        "استني",
        "تستاهل",
        "صفقة",
        "صفقه",
        "احسن",
        "افضل",
        "قارن",
        "مقارنة",
        "مقارنه",
        "عيوب",
        "مميزات",
        "صيانة",
        "صيانه",
        "recommend",
        "should i",
        "worth",
        "compare",
        "better",
        "pros",
        "cons",
        "maintenance",
        "resale",
    ]

    if has_any(text, advisor_markers):
        return True

    if text.startswith("ليه") or text.startswith("why"):
        return True

    return False


def diagnose_brain(
    *,
    message: str,
    schema: Dict[str, Any],
    assistant_id: str,
    variables: Dict[str, Any],
    recent_messages: Optional[List[Dict[str, Any]]] = None,
) -> BrainDecision:
    variables = variables or {}
    workflow = infer_workflow_from_state(schema, assistant_id, variables)
    playbook = get_domain_playbook(workflow)
    style_profile = infer_style_for_workflow(workflow)

    state_result = detect_user_state(message, variables)
    user_state = state_result["state"]
    signals = state_result["signals"]

    stage = variables.get("workflow_stage") or "general"
    gpt_needed = should_use_gpt_for_message(message, variables, workflow)

    sales_move = "answer_directly"
    cta = "continue"
    recommended_action = "continue_conversation"
    gpt_policy = "avoid"

    if gpt_needed:
        sales_move = "advise"
        cta = "advisor_response"
        recommended_action = "advisor_response"
        gpt_policy = "recommended"

    elif user_state == "price_sensitive":
        sales_move = "reassure_then_soft_close"
        cta = "offer_human_followup"
        recommended_action = "handle_objection"

    elif user_state == "comparing":
        sales_move = "compare_options"
        cta = "advisor_response"
        recommended_action = "advisor_response"
        gpt_needed = True
        gpt_policy = "recommended"

    elif user_state == "ready":
        sales_move = "book_or_confirm"
        cta = playbook.get("primary_cta", "continue")
        recommended_action = "collect_details"

    elif user_state == "hesitant":
        sales_move = "reassure_then_soft_close"
        cta = "advisor_or_viewing"
        recommended_action = "reassure"

    elif user_state == "confused":
        sales_move = "repair"
        cta = "clarify"
        recommended_action = "repair_conversation"

    elif variables.get("selected_item") or variables.get("matched_car_model"):
        sales_move = "answer_then_soft_close"
        cta = playbook.get("primary_cta", "continue")
        recommended_action = "continue_conversation"

    return BrainDecision(
        user_state=user_state,
        workflow=workflow,
        conversation_stage=stage,
        sales_move=sales_move,
        answer_style=playbook.get("preferred_style", "short"),
        should_use_gpt=gpt_needed,
        gpt_policy=gpt_policy,
        reason="Brain diagnosed message and selected next best move.",
        confidence=0.9 if signals else 0.75,
        cta=cta,
        detected_signals=signals,
        recommended_action=recommended_action,
        metadata={
            "playbook_goal": playbook.get("goal"),
            "style_profile": style_profile,
        },
    )


def build_brain_deterministic_response(
    *,
    message: str,
    schema: Dict[str, Any],
    assistant_id: str,
    variables: Dict[str, Any],
    recent_messages: Optional[List[Dict[str, Any]]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Try to answer through the assistant brain without GPT.
    Return None when GPT/advisor or existing paths should handle it.
    """
    decision = diagnose_brain(
        message=message,
        schema=schema,
        assistant_id=assistant_id,
        variables=variables,
        recent_messages=recent_messages,
    )

    if decision.should_use_gpt:
        return {
            "should_use_gpt": True,
            "decision": decision.to_dict(),
        }

    objection = build_objection_reply(message, variables)
    if objection and not objection.get("should_use_gpt"):
        answer = self_check_answer(
            objection["answer"],
            user_message=message,
            variables=variables,
            recent_messages=recent_messages,
        )

        return ComposeResult(
            answer=answer,
            action=objection.get("action", "handle_objection"),
            updates=objection.get("updates", {}),
            skip_summary=False if objection.get("updates") else True,
            skip_memory=False if objection.get("updates") else True,
            model_tier="assistant_brain",
            answer_mode="assistant_brain_objection",
        ).to_dict() | {"brain_decision": decision.to_dict(), "should_use_gpt": False}

    fact_type = detect_fact_type(message)
    if fact_type and (variables.get("selected_item") or variables.get("matched_car_model")):
        answer = compose_fact_reply(
            message=message,
            variables=variables,
            fact_type=fact_type,
            recent_messages=recent_messages,
        )

        if answer:
            answer = self_check_answer(
                answer,
                user_message=message,
                variables=variables,
                recent_messages=recent_messages,
            )

            return ComposeResult(
                answer=answer,
                action="answer_from_brain_state",
                updates={},
                skip_summary=True,
                skip_memory=True,
                model_tier="assistant_brain",
                answer_mode="assistant_brain_fact",
            ).to_dict() | {"brain_decision": decision.to_dict(), "should_use_gpt": False}

    # Entry-like premium reply if there is a selected item but no simple fact.
    if decision.sales_move == "answer_then_soft_close" and (variables.get("selected_item") or variables.get("matched_car_model")):
        answer = compose_premium_entry_reply(
            message=message,
            variables=variables,
            recent_messages=recent_messages,
        )

        answer = self_check_answer(
            answer,
            user_message=message,
            variables=variables,
            recent_messages=recent_messages,
        )

        return ComposeResult(
            answer=answer,
            action="brain_soft_close",
            updates={},
            skip_summary=True,
            skip_memory=True,
            model_tier="assistant_brain",
            answer_mode="assistant_brain",
        ).to_dict() | {"brain_decision": decision.to_dict(), "should_use_gpt": False}

    return {
        "should_use_gpt": False,
        "decision": decision.to_dict(),
        "no_direct_answer": True,
    }


def build_brain_advisor_hint(
    *,
    message: str,
    schema: Dict[str, Any],
    assistant_id: str,
    variables: Dict[str, Any],
    recent_messages: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    decision = diagnose_brain(
        message=message,
        schema=schema,
        assistant_id=assistant_id,
        variables=variables,
        recent_messages=recent_messages,
    )

    return {
        "brain_decision": decision.to_dict(),
        "advisor_instruction": (
            "Use this brain decision as strategy. Answer naturally, briefly, and practically. "
            "Do not reveal internal strategy. Do not pretend to be human. "
            "Use one clear next step."
        ),
    }
