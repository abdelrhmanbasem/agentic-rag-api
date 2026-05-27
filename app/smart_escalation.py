# app/smart_escalation.py
# Smart escalation gate.
#
# Purpose:
# - Keep simple turns zero-token.
# - Use GPT only when the user needs real reasoning, advice, comparison, persuasion,
#   uncertainty handling, or complex judgment.
#
# This is what makes the agent feel much smarter without using GPT for every turn.

import re
from typing import Dict, Any, List, Optional


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


def should_escalate_to_advisor(
    *,
    message: str,
    variables: Dict[str, Any],
    schema: Dict[str, Any],
    assistant_id: str = "",
) -> bool:
    """
    Escalate only when the user needs judgment/advice/reasoning.
    Do NOT escalate for simple factual/state questions.
    """
    text = normalize_text(message)
    variables = variables or {}

    if not text:
        return False

    # Simple factual questions should stay cheap.
    simple_fact_markers = [
        "اوتوماتيك",
        "اتوماتيك",
        "مانيوال",
        "كام كيلو",
        "عامله كام",
        "عاملة كام",
        "بكام",
        "سعرها كام",
        "سعره كام",
        "متاح",
        "available",
        "automatic",
        "manual",
        "km",
        "mileage",
        "price",
    ]

    if has_any(text, simple_fact_markers):
        # Exception: "is the price good?" is advice, not just price.
        advice_price_markers = ["سعر كويس", "السعر كويس", "worth", "good deal", "تستاهل"]
        if not has_any(text, advice_price_markers):
            return False

    advisor_markers = [
        # Arabic advice
        "تنصحني",
        "ترشحلي",
        "رأيك",
        "رايك",
        "ايه الاحسن",
        "ايه افضل",
        "أنهي احسن",
        "انهي احسن",
        "اختار ايه",
        "اخدها",
        "استنى",
        "استني",
        "تستاهل",
        "صفقه كويسه",
        "صفقة كويسة",
        "سعر كويس",
        "مناسبه ليا",
        "مناسبة ليا",
        "هل دي كويسه",
        "هل دي كويسة",

        # Arabic comparison
        "قارن",
        "مقارنه",
        "مقارنة",
        "احسن من",
        "افضل من",
        "ولا مرسيدس",
        "ولا bmw",
        "بدل",
        "بديل",

        # Arabic fear / doubt
        "خايف",
        "قلقان",
        "مش متاكد",
        "مش متأكد",
        "مش عارف",
        "هل العداد كتير",
        "العداد كتير",
        "عيوب",
        "مميزات",
        "اعطال",
        "صيانة",
        "صيانه",
        "إعادة بيع",
        "اعادة بيع",

        # English advice/comparison
        "recommend",
        "should i",
        "what do you think",
        "is it worth",
        "worth it",
        "good deal",
        "compare",
        "better than",
        "pros and cons",
        "concerned",
        "worried",
        "maintenance",
        "resale",
    ]

    if has_any(text, advisor_markers):
        return True

    # If the user asks a broad "why" about a product/service, escalate.
    if text.startswith("ليه") or text.startswith("why"):
        return True

    return False


def build_advisor_route(reason: str = "User needs advice or reasoning.") -> Dict[str, Any]:
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
        "rag_cache_hit": False,
    }


def build_advisor_system_prompt(
    *,
    assistant: Dict[str, Any],
    message: str,
    variables: Dict[str, Any],
) -> str:
    arabic = is_arabic_text(message)

    language_rule = (
        "Reply in natural Egyptian Arabic. Be sharp, practical, and concise."
        if arabic
        else "Reply in the same language as the user. Be sharp, practical, and concise."
    )

    return f"""
You are an elite business conversation advisor inside an agentic RAG assistant.

Assistant base behavior:
{assistant.get("system_prompt", "")}

Language:
{language_rule}

Style:
- Sound like a very sharp human operator, not a generic chatbot.
- Be practical, honest, and persuasive.
- Do not overpromise.
- Use known state and knowledge.
- If the user asks for advice, give a clear recommendation with a reason.
- If comparing options, explain the tradeoff simply.
- If there is uncertainty, say what should be checked next.
- End with one useful next step, not multiple questions.
- Keep the answer short enough for WhatsApp.

Important:
- Do not invent inventory, prices, kilometers, years, policies, appointment slots, or guarantees.
- Do not reveal internal routing, variables, RAG, or memory.
"""


def build_advisor_context(
    *,
    message: str,
    summary: str,
    variables: Dict[str, Any],
    knowledge: List[Dict[str, Any]],
    memories: List[Dict[str, Any]],
) -> str:
    return f"""
Latest user message:
{message}

Conversation summary:
{summary}

Current known state:
{variables}

Relevant knowledge:
{knowledge}

Relevant memories:
{memories}
"""
