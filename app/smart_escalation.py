# app/smart_escalation.py
# Smart escalation gate.
#
# Purpose:
# - Use GPT only when judgment, comparison, reasoning, or nuance is genuinely useful.
# - Keep factual/stateful replies deterministic.
# - Keep advisor prompts very compact to reduce paid input tokens.

import re
from typing import Dict, Any


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


def should_escalate_to_advisor(
    *,
    message: str,
    variables: Dict[str, Any],
    schema: Dict[str, Any],
    assistant_id: str,
) -> bool:
    text = normalize_text(message)
    variables = variables or {}

    advisor_markers = [
        "تنصحني",
        "رايك",
        "رأيك",
        "اخدها",
        "استنى",
        "استني",
        "تستاهل",
        "صفقة",
        "صفقه",
        "كويسة",
        "كويسه",
        "قرار شراء",
        "قرار الشراء",
        "ذكي",
        "حلل",
        "تحليل",
        "احسن",
        "افضل",
        "قارن",
        "مقارنة",
        "مقارنه",
        "عيوب",
        "مميزات",
        "صيانة",
        "صيانه",
        "اعادة بيع",
        "إعادة بيع",
        "مخاطر",
        "recommend",
        "should i",
        "worth it",
        "worth",
        "compare",
        "better",
        "pros",
        "cons",
        "maintenance",
        "resale",
        "risk",
        "risks",
        "analyze",
        "analysis",
    ]

    if any(marker in text for marker in advisor_markers):
        return True

    if text.startswith("ليه") or text.startswith("why"):
        return True

    if variables.get("selected_item") and any(x in text for x in ["ولا", "or", "بديل", "alternative"]):
        return True

    return False


def build_advisor_route(reason: str = "User needs advice, comparison, or judgment.") -> Dict[str, Any]:
    return {
        "intent_hint": "advisory_question",
        "answer_mode": "advisor",
        "needs_rag": True,
        "needs_memory": False,
        "needs_variable_extraction": False,
        "selected_model_tier": "normal",
        "reason": reason,
        "risk_score": 0.25,
        "complexity_score": 0.8,
    }


def build_advisor_system_prompt(
    *,
    assistant: Dict[str, Any],
    message: str,
    variables: Dict[str, Any],
) -> str:
    arabic = is_arabic_text(message)

    language_rule = (
        "Reply in natural Egyptian Arabic. Keep car names/models in English."
        if arabic
        else "Reply in the same language as the user."
    )

    return f"""
You are a practical advisor.

{language_rule}

Rules:
- Be balanced, not pushy.
- Use only known facts.
- Do not invent condition, warranty, discounts, inspection, or service history.
- For cars, reason from model, year, mileage, price, budget, transmission, and user concern.
- Prefer "worth viewing/checking" over "buy it now".
- Keep it 2-3 short sentences.
- Give one next step.
- Do not reveal internal routing or variables.
"""


def build_advisor_context(
    *,
    message: str,
    summary: str,
    variables: Dict[str, Any],
    knowledge,
    memories,
) -> str:
    return f"""
User:
{message}

Known facts:
{variables}

Relevant item:
{knowledge}
"""
