# app/intelligence_modes.py
# Adaptive intelligence mode selector.
#
# Goal:
# - Keep cheap deterministic paths for trivial turns.
# - Trigger premium/deep reasoning only when the sales moment deserves it.
#
# Modes:
# - deterministic: no GPT needed
# - light_rag: simple fact / availability / price
# - balanced: normal generated answer
# - premium_sales: objections, warm leads, booking hesitation
# - deep_advisor: comparisons, "should I?", worth-it, alternatives
# - careful_strong: complaints, anger, urgent/sensitive cases

import os
import re
from typing import Dict, Any, List


AGENT_MODE = os.getenv("AGENT_MODE", "adaptive_premium").lower()

MODE_DETERMINISTIC = "deterministic"
MODE_LIGHT_RAG = "light_rag"
MODE_BALANCED = "balanced"
MODE_PREMIUM_SALES = "premium_sales"
MODE_DEEP_ADVISOR = "deep_advisor"
MODE_CAREFUL_STRONG = "careful_strong"


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


def is_arabic_text(text: str) -> bool:
    return bool(re.search(r"[\u0600-\u06FF]", text or ""))


def is_trivial_turn(message: str) -> bool:
    text = normalize_text(message)

    exact = {
        "ok",
        "okay",
        "yes",
        "no",
        "thanks",
        "thank you",
        "تمام",
        "اوكي",
        "أوكي",
        "ماشي",
        "حاضر",
        "شكرا",
        "شكرًا",
        "تسلم",
        "اه",
        "ايوه",
        "لا",
    }

    if text in exact:
        return True

    cleaned_phone = re.sub(r"[\s\-\+\(\)]", "", message or "")
    if cleaned_phone.isdigit() and 8 <= len(cleaned_phone) <= 15:
        return True

    return len(text) <= 2


def is_simple_fact_question(message: str, variables: Dict[str, Any]) -> bool:
    text = normalize_text(message)
    variables = variables or {}

    has_item = bool(
        variables.get("selected_item")
        or variables.get("matched_car_model")
        or variables.get("matched_car_price")
    )

    fact_markers = [
        "price",
        "cost",
        "how much",
        "km",
        "mileage",
        "automatic",
        "manual",
        "transmission",
        "بكام",
        "سعر",
        "سعرها",
        "كام كيلو",
        "عاملة كام",
        "عامله كام",
        "اوتوماتيك",
        "أوتوماتيك",
        "مانيوال",
    ]

    return has_item and has_any(text, fact_markers)


def is_booking_collection(message: str) -> bool:
    text = normalize_text(message)

    booking_markers = [
        "tomorrow",
        "today",
        "saturday",
        "sunday",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "3 pm",
        "4 pm",
        "بكرة",
        "بكره",
        "النهارده",
        "السبت",
        "الاحد",
        "الأحد",
        "الاتنين",
        "الثلاثاء",
        "الاربع",
        "الخميس",
        "الجمعة",
        "العصر",
        "الصبح",
        "المغرب",
        "بالليل",
        "التجمع",
        "مدينة نصر",
        "المعادي",
        "زايد",
    ]

    return has_any(text, booking_markers)


def is_objection(message: str) -> bool:
    text = normalize_text(message)

    markers = [
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
    ]

    return has_any(text, markers)


def is_advice_or_comparison(message: str) -> bool:
    text = normalize_text(message)

    markers = [
        "recommend",
        "should i",
        "worth",
        "worth it",
        "compare",
        "better",
        "pros",
        "cons",
        "alternative",
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
        "بديل",
    ]

    if has_any(text, markers):
        return True

    if " ولا " in text:
        return True

    return False


def is_complaint_or_risk(message: str) -> bool:
    text = normalize_text(message)

    markers = [
        "angry",
        "complaint",
        "refund",
        "unacceptable",
        "lawsuit",
        "report you",
        "cancel",
        "emergency",
        "urgent",
        "شكوى",
        "غاضب",
        "زعلان",
        "غير مقبول",
        "عايز فلوسي",
        "استرداد",
        "طوارئ",
        "مستعجل",
        "ضروري",
    ]

    return has_any(text, markers)


def lead_score_from_variables(variables: Dict[str, Any]) -> int:
    variables = variables or {}

    try:
        return int(variables.get("lead_score") or 0)
    except Exception:
        return 0


def has_high_value_sales_context(variables: Dict[str, Any]) -> bool:
    variables = variables or {}

    return bool(
        variables.get("selected_item")
        or variables.get("matched_car_model")
        or variables.get("budget_max")
        or variables.get("phone_number")
        or variables.get("preferred_viewing_date")
        or variables.get("needs_human")
        or lead_score_from_variables(variables) >= 60
    )


def choose_intelligence_mode(
    *,
    message: str,
    variables: Dict[str, Any],
    route: Dict[str, Any] | None = None,
    schema: Dict[str, Any] | None = None,
    assistant_id: str = "",
) -> Dict[str, Any]:
    """
    Returns:
    {
      "mode": "...",
      "should_use_premium": bool,
      "selected_model_tier": "cheap|normal|strong",
      "reason": "..."
    }
    """

    route = route or {}
    variables = variables or {}

    if AGENT_MODE in ["off", "efficiency"]:
        return {
            "mode": MODE_DETERMINISTIC,
            "should_use_premium": False,
            "selected_model_tier": "cheap",
            "reason": "Premium mode disabled by AGENT_MODE.",
        }

    if is_trivial_turn(message):
        return {
            "mode": MODE_DETERMINISTIC,
            "should_use_premium": False,
            "selected_model_tier": "cheap",
            "reason": "Trivial turn.",
        }

    if is_complaint_or_risk(message):
        return {
            "mode": MODE_CAREFUL_STRONG,
            "should_use_premium": True,
            "selected_model_tier": "strong",
            "reason": "Complaint, urgency, anger, or risk needs careful handling.",
        }

    if is_advice_or_comparison(message):
        return {
            "mode": MODE_DEEP_ADVISOR,
            "should_use_premium": True,
            "selected_model_tier": "strong",
            "reason": "User asked for advice, comparison, judgment, or worth-it reasoning.",
        }

    if is_objection(message):
        return {
            "mode": MODE_PREMIUM_SALES,
            "should_use_premium": True,
            "selected_model_tier": "normal",
            "reason": "Sales objection detected.",
        }

    if lead_score_from_variables(variables) >= 70:
        return {
            "mode": MODE_PREMIUM_SALES,
            "should_use_premium": True,
            "selected_model_tier": "normal",
            "reason": "High-value warm/hot lead.",
        }

    if is_simple_fact_question(message, variables):
        return {
            "mode": MODE_LIGHT_RAG,
            "should_use_premium": False,
            "selected_model_tier": "cheap",
            "reason": "Simple factual follow-up can stay light.",
        }

    if is_booking_collection(message):
        return {
            "mode": MODE_DETERMINISTIC,
            "should_use_premium": False,
            "selected_model_tier": "cheap",
            "reason": "Likely date/time/location collection.",
        }

    if has_high_value_sales_context(variables):
        return {
            "mode": MODE_BALANCED,
            "should_use_premium": True,
            "selected_model_tier": "normal",
            "reason": "Meaningful sales context; use premium reasoning lightly.",
        }

    return {
        "mode": MODE_BALANCED,
        "should_use_premium": False,
        "selected_model_tier": route.get("selected_model_tier", "normal"),
        "reason": "Default balanced mode.",
    }
