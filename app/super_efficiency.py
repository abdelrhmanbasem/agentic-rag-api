# app/super_efficiency.py
# Extra token/cost guards:
# - deterministic mini-router
# - summary cooldown
# - memory-write cooldown
# - production/debug response mode

import os
from typing import Dict, Any, Optional, List


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


def deterministic_route_guess(
    *,
    message: str,
    schema: Dict[str, Any],
    variables: Dict[str, Any],
    assistant_id: str = "",
) -> Optional[Dict[str, Any]]:
    """
    Cheap route before GPT router.
    Return None when unsure, so GPT router can still handle complex cases.
    """
    text = normalize_text(message)
    schema = schema or {}
    variables = variables or {}
    assistant_id = (assistant_id or "").lower()

    if not text:
        return None

    def has_any(words: List[str]) -> bool:
        return any(word in text for word in words)

    if has_any(["شكرا", "شكرًا", "thanks", "thank you", "تسلم"]):
        return {
            "intent_hint": "general_question",
            "answer_mode": "no_llm",
            "needs_rag": False,
            "needs_memory": False,
            "needs_variable_extraction": False,
            "selected_model_tier": "cheap",
            "reason": "Deterministic route: closing/thanks.",
            "risk_score": 0.0,
            "complexity_score": 0.0,
        }

    if has_any(["كلموني", "حد يكلمني", "human", "agent", "call me", "مندوب", "موظف"]):
        return {
            "intent_hint": "human_handoff",
            "answer_mode": "no_llm",
            "needs_rag": False,
            "needs_memory": True,
            "needs_variable_extraction": True,
            "selected_model_tier": "cheap",
            "reason": "Deterministic route: human handoff.",
            "risk_score": 0.1,
            "complexity_score": 0.1,
        }

    car_markers = [
        "bmw",
        "mercedes",
        "hyundai",
        "toyota",
        "kia",
        "nissan",
        "audi",
        "عربية",
        "عربيه",
        "اشتري",
        "اشترى",
        "مستعملة",
        "مستعمله",
        "زيرو",
        "بي ام",
        "مرسيدس",
        "هيونداي",
        "تويوتا",
    ]

    if has_any(car_markers) or any(k in schema for k in ["car_brand", "budget_max", "matched_car_model"]):
        if has_any(["احجز", "معاينة", "معاينه", "اشوفها", "أشوفها", "viewing", "test drive"]):
            return {
                "intent_hint": "viewing_request",
                "answer_mode": "generate",
                "needs_rag": False,
                "needs_memory": True,
                "needs_variable_extraction": True,
                "selected_model_tier": "cheap",
                "reason": "Deterministic route: car viewing request.",
                "risk_score": 0.1,
                "complexity_score": 0.2,
            }

        if has_any(car_markers):
            return {
                "intent_hint": "car_search",
                "answer_mode": "generate",
                "needs_rag": True,
                "needs_memory": False,
                "needs_variable_extraction": True,
                "selected_model_tier": "cheap",
                "reason": "Deterministic route: car search.",
                "risk_score": 0.1,
                "complexity_score": 0.2,
            }

    service_markers = [
        "احجز",
        "حجز",
        "ميعاد",
        "موعد",
        "كشف",
        "دكتور",
        "عيادة",
        "عياده",
        "appointment",
        "book",
        "doctor",
        "clinic",
    ]

    if has_any(service_markers) or any(k in schema for k in ["service_needed", "appointment_date"]):
        return {
            "intent_hint": "booking_request",
            "answer_mode": "generate",
            "needs_rag": True,
            "needs_memory": True,
            "needs_variable_extraction": True,
            "selected_model_tier": "cheap",
            "reason": "Deterministic route: service booking.",
            "risk_score": 0.1,
            "complexity_score": 0.25,
        }

    return None


def should_update_summary(
    *,
    message: str,
    answer: str,
    variables: Dict[str, Any],
    model_tier: str,
    route: Dict[str, Any],
    recent_messages: List[Dict[str, Any]],
) -> bool:
    """
    Summary updates are useful but expensive.
    Skip them for cheap/state turns unless there is important progress.
    """
    variables = variables or {}
    route = route or {}
    text = normalize_text(message)

    if model_tier in ["fast_path", "state", "no_llm"]:
        important = any(
            variables.get(k)
            for k in [
                "preferred_viewing_date",
                "preferred_viewing_time",
                "appointment_date",
                "appointment_time",
                "location",
                "phone_number",
                "needs_human",
            ]
        )

        if not important:
            return False

    if model_tier == "entry_path":
        return True

    if variables.get("workflow_stage") in ["confirmed", "viewing_details_collected", "booking_details_collected"]:
        return True

    if variables.get("needs_human"):
        return True

    if any(x in text for x in ["احجز", "ميعاد", "موعد", "كلموني", "شكوى", "complaint", "refund"]):
        return True

    user_turn_count = len([m for m in recent_messages or [] if m.get("role") == "user"])

    return user_turn_count > 0 and user_turn_count % 8 == 0


def should_write_memory(
    *,
    message: str,
    variables: Dict[str, Any],
    model_tier: str,
    route: Dict[str, Any],
    recent_messages: List[Dict[str, Any]],
) -> bool:
    """
    Long-term memory should only run when stable preference changed.
    """
    variables = variables or {}
    text = normalize_text(message)

    if model_tier in ["fast_path", "state", "no_llm"]:
        return False

    stable_preference_keys = [
        "car_brand",
        "car_condition",
        "budget_max",
        "preferred_contact_method",
        "location",
        "service_needed",
        "doctor_preference",
        "insurance_provider",
    ]

    if any(variables.get(k) for k in stable_preference_keys):
        if any(
            marker in text
            for marker in [
                "افضل",
                "بحب",
                "عايز",
                "عاوز",
                "حابب",
                "ميزانيتي",
                "whatsapp",
                "واتساب",
                "ماتتصلش",
                "prefer",
                "budget",
                "looking for",
            ]
        ):
            return True

    if variables.get("workflow_stage") in ["confirmed", "booking_details_collected", "viewing_details_collected"]:
        return True

    return False


def debug_response_enabled() -> bool:
    return os.getenv("DEBUG_RESPONSE", "true").lower() in ["1", "true", "yes", "on"]


def compact_chat_response(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    In production, avoid huge API responses to n8n/WhatsApp.
    Full debug remains available when DEBUG_RESPONSE=true.
    """
    if debug_response_enabled():
        return payload

    return {
        "answer": payload.get("answer"),
        "assistant_id": payload.get("assistant_id"),
        "conversation_id": payload.get("conversation_id"),
        "intent": payload.get("intent"),
        "model_used": payload.get("model_used"),
        "model_tier": payload.get("model_tier"),
        "token_usage": payload.get("token_usage"),
        "recommended_next_action": payload.get("recommended_next_action"),
        "memory_saved": payload.get("memory_saved"),
    }
