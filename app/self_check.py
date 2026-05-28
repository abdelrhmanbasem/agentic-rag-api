# app/self_check.py
# Lightweight final answer self-check.
#
# Purpose:
# - Keep deterministic replies natural.
# - Reduce repetition, weird punctuation, overlong replies, and robotic CTAs.
# - Does not call GPT.

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


def cleanup_spacing(answer: str) -> str:
    answer = answer or ""
    answer = re.sub(r"\s+", " ", answer).strip()
    answer = answer.replace("..", ".")
    answer = answer.replace("،،", "،")
    answer = answer.replace(" .", ".")
    answer = answer.replace(" ،", "،")
    answer = answer.replace(" ?", "?")
    answer = answer.replace(" ؟", "؟")
    return answer.strip()


def trim_overlong(answer: str, max_sentences: int = 3) -> str:
    answer = cleanup_spacing(answer)
    if not answer:
        return answer

    # Split on Arabic/English sentence punctuation while keeping it readable.
    parts = re.split(r"(?<=[.!؟])\s+", answer)

    if len(parts) <= max_sentences:
        return answer

    return " ".join(parts[:max_sentences]).strip()


def remove_duplicate_sentences(answer: str) -> str:
    answer = cleanup_spacing(answer)
    if not answer:
        return answer

    parts = re.split(r"(?<=[.!؟])\s+", answer)
    seen = set()
    clean = []

    for part in parts:
        key = normalize_text(part)
        if key and key not in seen:
            clean.append(part)
            seen.add(key)

    return " ".join(clean).strip()


def avoid_repeated_cta(
    answer: str,
    recent_messages: Optional[List[Dict[str, Any]]],
) -> str:
    if not answer:
        return answer

    recent_text = " ".join(
        msg.get("content", "")
        for msg in (recent_messages or [])[-5:]
        if msg.get("role") == "assistant"
    )
    recent_norm = normalize_text(recent_text)
    answer_norm = normalize_text(answer)

    repeated_viewing = any(x in recent_norm for x in ["تحب تشوفها", "معاينة", "معاينه", "viewing"]) and any(
        x in answer_norm for x in ["تحب تشوفها", "تحب اظبطلك معاينة", "تحب أظبطلك معاينة", "arrange a viewing"]
    )

    if repeated_viewing:
        answer = re.sub(r"تحب أظبطلك معاينة[؟?]?", "أقدر أكمّلك تفاصيلها أو نثبت المعاد لما يناسبك.", answer)
        answer = re.sub(r"تحب اظبطلك معاينة[؟?]?", "أقدر أكمّلك تفاصيلها أو نثبت المعاد لما يناسبك.", answer)
        answer = re.sub(r"Would you like me to arrange a viewing\??", "I can share more details or set it up when you’re ready.", answer)

    return cleanup_spacing(answer)


def remove_robotic_phrases(answer: str) -> str:
    replacements = {
        "بناءً على المعلومات المتاحة": "حسب التفاصيل المتاحة",
        "هل ترغب": "تحب",
        "يرجى": "ممكن",
        "فضلاً": "لو سمحت",
        "I would be happy to": "I can",
        "Based on the available information": "From the available details",
    }

    for old, new in replacements.items():
        answer = answer.replace(old, new)

    return answer


def ensure_not_deceptive(answer: str) -> str:
    # Do not let deterministic templates claim human identity.
    blocked_phrases = [
        "أنا إنسان",
        "انا انسان",
        "I am human",
        "I'm human",
    ]

    for phrase in blocked_phrases:
        answer = answer.replace(phrase, "")

    return cleanup_spacing(answer)


def self_check_answer(
    answer: str,
    *,
    user_message: str = "",
    variables: Optional[Dict[str, Any]] = None,
    recent_messages: Optional[List[Dict[str, Any]]] = None,
    max_sentences: int = 3,
) -> str:
    answer = answer or ""

    answer = cleanup_spacing(answer)
    answer = remove_robotic_phrases(answer)
    answer = remove_duplicate_sentences(answer)
    answer = avoid_repeated_cta(answer, recent_messages)
    answer = ensure_not_deceptive(answer)
    answer = trim_overlong(answer, max_sentences=max_sentences)
    answer = cleanup_spacing(answer)

    return answer
