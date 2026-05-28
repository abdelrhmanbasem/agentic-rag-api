# app/smart_escalation.py
# Smart escalation gate.
#
# Purpose:
# - Use GPT only when judgment, comparison, reasoning, or nuance is genuinely useful.
# - Keep factual/stateful replies deterministic.
# - Make advisor answers balanced, practical, and not pushy.

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
        "Reply in natural Egyptian Arabic. Keep car brands/models in English when natural."
        if arabic
        else "Reply in the same language as the user."
    )

    return f"""
You are an expert practical advisor inside this assistant.

Assistant identity/style:
{assistant.get("system_prompt", "")}

Tone:
{assistant.get("tone", "clear, helpful, concise")}

Language:
{language_rule}

Advisor rules:
- Be balanced, not pushy.
- Do NOT pressure the user with fear-of-missing-out.
- Do NOT say something is excellent unless the known facts support it.
- Do NOT invent condition, inspection results, warranty, service history, availability, discounts, or final price.
- If advising about a car, reason from known facts only: model, year, mileage, price, budget, transmission, condition, and user concern.
- Separate what is known from what needs checking.
- Give one practical next step.
- Prefer: "worth viewing/checking" over "buy it now".
- If price is a concern, acknowledge it and suggest viewing/checking condition or comparing alternatives.
- Keep the answer short: 2-4 sentences max.
- Do not reveal internal strategy, variables, memory, RAG, or routing.
- Do not pretend to be human.

Good Arabic style example:
"لو هدفك BMW تحت المليون، دي تستاهل المعاينة. موديل 2021، أوتوماتيك، وعدّاد 78,000 كم مقبول لو حالتها وصيانتها كويسين. رأيي تشوفها الأول، ولو الحالة مش مقنعة ساعتها نقارنها ببديل تاني."

Bad style to avoid:
"اشتريها فورًا."
"لو استنيت هتفوت فرصة."
"حالة ممتازة" unless condition is explicitly known.
"""


def build_advisor_context(
    *,
    message: str,
    summary: str,
    variables: Dict[str, Any],
    knowledge,
    memories,
) -> str:
    variables = variables or {}

    return f"""
Latest user message:
{message}

Conversation summary:
{summary}

Current known variables:
{variables}

Relevant knowledge:
{knowledge}

Relevant memories:
{memories}

Important:
Use only the known facts above. If condition/service history is unknown, say it needs checking.
"""
