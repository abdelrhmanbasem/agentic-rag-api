# app/premium_memory.py
# Premium memory filter for adaptive premium mode.
#
# Goal:
# - Prevent one-off objections from becoming permanent memory.
# - Keep useful session signals for the current turn.
# - Only allow stable, explicit preferences into long-term memory.
#
# This file does NOT write memory directly.
# It decides what should be allowed/suppressed.

import re
from typing import Dict, Any, List


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


def has_any(text: str, markers: List[str]) -> bool:
    return any(marker in text for marker in markers)


def detect_price_sensitivity(message: str) -> bool:
    text = normalize_text(message)
    return has_any(
        text,
        [
            "expensive",
            "too expensive",
            "discount",
            "installment",
            "installments",
            "lower price",
            "final price",
            "غالي",
            "غالية",
            "غاليه",
            "كتير",
            "السعر عالي",
            "خصم",
            "تقسيط",
            "قسط",
            "نهائي",
            "اخره",
            "آخره",
            "مش مناسب",
        ],
    )


def detect_stable_budget_preference(message: str, variables: Dict[str, Any]) -> bool:
    text = normalize_text(message)
    variables = variables or {}

    if not variables.get("budget_max"):
        return False

    return has_any(
        text,
        [
            "budget",
            "my budget",
            "up to",
            "under",
            "ميزانيتي",
            "ميزانية",
            "لحد",
            "حدي",
            "مليون",
            "الف",
            "ألف",
        ],
    )


def detect_contact_preference(message: str, variables: Dict[str, Any]) -> bool:
    text = normalize_text(message)
    variables = variables or {}

    if variables.get("preferred_contact_method"):
        return True

    return has_any(
        text,
        [
            "whatsapp",
            "whats app",
            "no calls",
            "don't call",
            "dont call",
            "واتساب",
            "واتس",
            "ماتتصلش",
            "ما تتصلش",
            "متتصلش",
            "بلاش مكالمات",
            "كلمني واتساب",
        ],
    )


def detect_repeated_signal(
    *,
    message: str,
    recent_messages: List[Dict[str, Any]],
    signal_type: str,
) -> bool:
    text = normalize_text(message)

    if signal_type == "price_sensitive":
        current = detect_price_sensitivity(text)
        if not current:
            return False

        count = 0
        for msg in recent_messages or []:
            if msg.get("role") != "user":
                continue
            if detect_price_sensitivity(msg.get("content", "")):
                count += 1

        return count >= 2

    return False


def build_premium_memory_decision(
    *,
    user_message: str,
    variables: Dict[str, Any],
    recent_messages: List[Dict[str, Any]],
    mode_decision: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Returns a memory policy object for this turn.

    Important:
    - session_signals can be used in debug/reasoning.
    - suppress_long_term_memory means main.py should avoid writing long-term memory for this premium turn.
    - allow_long_term_memory means main.py may write memory using its existing memory function.
    """

    variables = variables or {}
    recent_messages = recent_messages or {}
    mode_decision = mode_decision or {}

    session_signals: List[str] = []
    long_term_candidates: List[Dict[str, Any]] = []
    suppress_reasons: List[str] = []

    price_sensitive_now = detect_price_sensitivity(user_message)
    repeated_price_sensitive = detect_repeated_signal(
        message=user_message,
        recent_messages=recent_messages,
        signal_type="price_sensitive",
    )

    if price_sensitive_now:
        session_signals.append("price_sensitive_now")

        if repeated_price_sensitive:
            long_term_candidates.append(
                {
                    "type": "constraint",
                    "text": "User is consistently price-sensitive when evaluating purchases.",
                    "confidence": 0.75,
                    "importance": 0.7,
                }
            )
        else:
            suppress_reasons.append("one_off_price_objection")

    if detect_stable_budget_preference(user_message, variables):
        long_term_candidates.append(
            {
                "type": "constraint",
                "text": f"User budget is around {variables.get('budget_max')} {variables.get('currency', '')}".strip(),
                "confidence": 0.85,
                "importance": 0.8,
            }
        )

    if detect_contact_preference(user_message, variables):
        long_term_candidates.append(
            {
                "type": "preference",
                "text": f"User prefers {variables.get('preferred_contact_method', 'WhatsApp')} for contact.",
                "confidence": 0.9,
                "importance": 0.85,
            }
        )

    has_confirmed_stage = variables.get("workflow_stage") in [
        "confirmed",
        "booking_details_collected",
        "viewing_details_collected",
    ]

    needs_human = bool(variables.get("needs_human") and variables.get("handoff_reason"))

    allow_long_term_memory = bool(
        long_term_candidates
        or has_confirmed_stage
        or needs_human
    )

    suppress_long_term_memory = False

    if suppress_reasons and not allow_long_term_memory:
        suppress_long_term_memory = True

    return {
        "session_signals": session_signals,
        "long_term_candidates": long_term_candidates,
        "allow_long_term_memory": allow_long_term_memory,
        "suppress_long_term_memory": suppress_long_term_memory,
        "suppress_reasons": suppress_reasons,
        "mode": mode_decision.get("mode"),
        "reason": (
            "Allowing long-term memory for stable/confirmed signal."
            if allow_long_term_memory
            else "Suppressing long-term memory for temporary/one-off signal."
            if suppress_long_term_memory
            else "No strong memory signal."
        ),
    }
