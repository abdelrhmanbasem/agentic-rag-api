# app/objection_playbooks.py
# Deterministic objection and hesitation handling.
#
# Purpose:
# - Handle common objections without GPT when possible.
# - Keep replies human, short, persuasive, and safe.

import re
from typing import Dict, Any, Optional


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


def format_money(amount: Any, currency: str = "EGP", arabic: bool = True) -> str:
    try:
        formatted = f"{int(amount):,}"
    except Exception:
        formatted = str(amount)

    if arabic and currency == "EGP":
        return f"{formatted} جنيه"

    return f"{formatted} {currency}"


def get_selected_item(variables: Dict[str, Any]) -> Dict[str, Any]:
    variables = variables or {}

    selected = variables.get("selected_item")
    if isinstance(selected, dict):
        return selected

    if variables.get("matched_car_model"):
        return {
            "type": "car",
            "brand": variables.get("car_brand"),
            "model": variables.get("matched_car_model"),
            "year": variables.get("matched_car_year"),
            "km": variables.get("matched_car_km"),
            "price": variables.get("matched_car_price"),
            "currency": variables.get("currency") or "EGP",
            "transmission": variables.get("transmission"),
            "condition": variables.get("car_condition"),
        }

    return {}


def detect_objection_type(message: str) -> Optional[str]:
    text = normalize_text(message)

    price_markers = [
        "غالي",
        "غاليه",
        "غالية",
        "كتير",
        "السعر عالي",
        "خصم",
        "تقسيط",
        "قسط",
        "نهائي",
        "اخره",
        "آخره",
        "اقل",
        "أقل",
        "expensive",
        "too expensive",
        "discount",
        "installment",
        "installments",
        "final price",
        "lower price",
    ]

    mileage_markers = [
        "عداد",
        "كيلو كتير",
        "عاملة كتير",
        "عامله كتير",
        "mileage",
        "too many km",
        "high mileage",
    ]

    trust_markers = [
        "مضمون",
        "اضمن",
        "ثقة",
        "ثقه",
        "خايف",
        "قلقان",
        "نصب",
        "trust",
        "guarantee",
        "worried",
        "concerned",
    ]

    comparison_markers = [
        "احسن من",
        "افضل من",
        "ولا",
        "قارن",
        "مقارنة",
        "مقارنه",
        "compare",
        "better than",
    ]

    hesitation_markers = [
        "هفكر",
        "لسه",
        "مش متاكد",
        "مش متأكد",
        "مش عارف",
        "maybe",
        "not sure",
    ]

    if any(x in text for x in price_markers):
        return "price"

    if any(x in text for x in mileage_markers):
        return "mileage"

    if any(x in text for x in trust_markers):
        return "trust"

    if any(x in text for x in comparison_markers):
        return "comparison"

    if any(x in text for x in hesitation_markers):
        return "hesitation"

    return None


def build_price_objection_reply(message: str, variables: Dict[str, Any]) -> str:
    arabic = is_arabic_text(message)
    item = get_selected_item(variables)

    model = item.get("model") or variables.get("matched_car_model") or variables.get("car_brand") or ("العربية" if arabic else "the car")
    price = item.get("price") or variables.get("matched_car_price")
    currency = item.get("currency") or variables.get("currency") or "EGP"
    budget = variables.get("budget_max")

    if arabic:
        if price and budget:
            try:
                diff = int(budget) - int(price)
            except Exception:
                diff = None

            if diff is not None and diff >= 0:
                return (
                    f"فاهمك، السعر مهم. {model} سعرها {format_money(price, currency, True)}، "
                    f"وده داخل ميزانيتك ولسه سايب حوالي {format_money(diff, currency, True)}. "
                    f"الأذكى تشوف حالتها الأول، وبعد المعاينة نعرف مساحة التفاوض."
                )

        if price:
            return (
                f"فاهمك، السعر الحالي لـ {model} هو {format_money(price, currency, True)}. "
                f"لو شايفه عالي، نقدر نشوف بديل أقرب لميزانيتك أو نخلي الفريق يتابع معاك بخصوص التفاوض."
            )

        return "فاهمك، السعر مهم طبعًا. أقدر أشوفلك بديل أقرب لميزانيتك أو أخلي الفريق يتابع معاك."

    if price:
        return (
            f"I understand. {model} is currently {format_money(price, currency, False)}. "
            f"If that feels high, we can compare alternatives or have the team follow up about negotiation."
        )

    return "I understand. We can compare alternatives or have the team follow up about pricing."


def build_mileage_objection_reply(message: str, variables: Dict[str, Any]) -> str:
    arabic = is_arabic_text(message)
    item = get_selected_item(variables)

    model = item.get("model") or variables.get("matched_car_model") or variables.get("car_brand") or ("العربية" if arabic else "the car")
    km = item.get("km") or variables.get("matched_car_km")

    try:
        km_text = f"{int(km):,}"
    except Exception:
        km_text = str(km or "")

    if arabic:
        if km:
            return (
                f"سؤالك في محله. {model} عاملة {km_text} كيلو، وده مش مقلق لو الصيانة منتظمة والحالة كويسة. "
                f"الأهم في المعاينة نتأكد من الصيانة، الموتور، الفتيس، والعفشة."
            )
        return "سؤالك في محله. العداد مهم، بس الأهم الحالة والصيانة. نقدر نأكد التفاصيل في المعاينة."

    if km:
        return (
            f"Good question. {model} has {km_text} km. That is not automatically a problem if maintenance and condition are good. "
            f"The key is checking service history, engine, transmission, and suspension."
        )

    return "Good question. Mileage matters, but condition and service history matter more."


def build_trust_objection_reply(message: str, variables: Dict[str, Any]) -> str:
    arabic = is_arabic_text(message)

    if arabic:
        return (
            "قلقك طبيعي. عشان كده الأفضل تشوفها على الطبيعة وتراجع الحالة والتفاصيل قبل أي قرار. "
            "أقدر أظبطلك معاينة وتاخد قرارك براحتك."
        )

    return (
        "That concern makes sense. The best next step is to view it, check the condition and details, then decide. "
        "I can help arrange a viewing."
    )


def build_hesitation_reply(message: str, variables: Dict[str, Any]) -> str:
    arabic = is_arabic_text(message)
    item = get_selected_item(variables)

    model = item.get("model") or variables.get("matched_car_model") or variables.get("car_brand") or ("الاختيار ده" if arabic else "this option")

    if arabic:
        return (
            f"ولا يهمك. لو لسه بتفكر، أقدر ألخصلك {model} في نقطتين: هل مناسب لميزانيتك، وهل حالته تستاهل المعاينة. "
            "تحب أقولك رأيي بصراحة؟"
        )

    return (
        f"No problem. If you are still thinking, I can summarize {model} in two points: budget fit and whether it is worth viewing. "
        "Would you like my honest take?"
    )


def build_comparison_reply(message: str, variables: Dict[str, Any]) -> Optional[str]:
    # Comparison usually benefits from GPT advisor if multiple options/tradeoffs are involved.
    return None


def build_objection_reply(message: str, variables: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    objection_type = detect_objection_type(message)

    if not objection_type:
        return None

    if objection_type == "price":
        answer = build_price_objection_reply(message, variables)
        return {
            "answer": answer,
            "objection_type": "price",
            "action": "handle_price_objection",
            "should_use_gpt": False,
            "updates": {
                "needs_human": True,
                "handoff_reason": "price_or_negotiation",
            },
        }

    if objection_type == "mileage":
        answer = build_mileage_objection_reply(message, variables)
        return {
            "answer": answer,
            "objection_type": "mileage",
            "action": "handle_mileage_objection",
            "should_use_gpt": False,
            "updates": {},
        }

    if objection_type == "trust":
        answer = build_trust_objection_reply(message, variables)
        return {
            "answer": answer,
            "objection_type": "trust",
            "action": "handle_trust_objection",
            "should_use_gpt": False,
            "updates": {},
        }

    if objection_type == "hesitation":
        answer = build_hesitation_reply(message, variables)
        return {
            "answer": answer,
            "objection_type": "hesitation",
            "action": "handle_hesitation",
            "should_use_gpt": False,
            "updates": {},
        }

    if objection_type == "comparison":
        return {
            "answer": "",
            "objection_type": "comparison",
            "action": "advisor_comparison",
            "should_use_gpt": True,
            "updates": {},
        }

    return None
