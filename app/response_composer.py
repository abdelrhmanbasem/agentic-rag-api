# app/response_composer.py
# Premium deterministic response composer.
#
# Purpose:
# - Make zero-token answers sound like a sharp human operator.
# - Keep replies short, natural, and useful.
# - Avoid robotic phrasing and repeated CTAs.

import re
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


def format_km(km: Any, arabic: bool = True) -> str:
    try:
        formatted = f"{int(km):,}"
    except Exception:
        formatted = str(km)

    return f"{formatted} كيلو" if arabic else f"{formatted} km"


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


def recent_assistant_text(recent_messages: Optional[List[Dict[str, Any]]]) -> str:
    texts = []
    for msg in recent_messages or []:
        if msg.get("role") == "assistant":
            texts.append(msg.get("content", ""))
    return " ".join(texts[-3:])


def already_asked_viewing(recent_messages: Optional[List[Dict[str, Any]]], variables: Dict[str, Any]) -> bool:
    asked = variables.get("asked_questions")
    if isinstance(asked, list) and "viewing_interest" in asked:
        return True

    text = normalize_text(recent_assistant_text(recent_messages))
    return any(x in text for x in ["تحب تشوف", "معاينة", "معاينه", "viewing"])


def compose_item_summary(message: str, variables: Dict[str, Any]) -> str:
    arabic = is_arabic_text(message)
    item = get_selected_item(variables)

    model = item.get("model") or variables.get("matched_car_model") or variables.get("car_brand") or ("العربية" if arabic else "the car")
    year = item.get("year") or variables.get("matched_car_year")
    km = item.get("km") or variables.get("matched_car_km")
    price = item.get("price") or variables.get("matched_car_price")
    currency = item.get("currency") or variables.get("currency") or "EGP"
    transmission = item.get("transmission") or variables.get("transmission")

    if arabic:
        parts = [f"عندنا {model}"]
        if year:
            parts.append(f"موديل {year}")
        if transmission == "automatic":
            parts.append("أوتوماتيك")
        elif transmission == "manual":
            parts.append("مانيوال")
        if km:
            parts.append(f"عاملة {format_km(km, True)}")
        if price:
            parts.append(f"وسعرها {format_money(price, currency, True)}")
        return "، ".join(parts) + "."

    parts = [f"We have {model}"]
    if year:
        parts.append(f"from {year}")
    if transmission:
        parts.append(f"with {transmission} transmission")
    if km:
        parts.append(f"and {format_km(km, False)}")
    if price:
        parts.append(f"priced at {format_money(price, currency, False)}")
    return " ".join(parts) + "."


def compose_value_frame(message: str, variables: Dict[str, Any]) -> str:
    arabic = is_arabic_text(message)
    item = get_selected_item(variables)

    price = item.get("price") or variables.get("matched_car_price")
    budget = variables.get("budget_max")
    condition = item.get("condition") or variables.get("car_condition")
    transmission = item.get("transmission") or variables.get("transmission")
    brand = item.get("brand") or variables.get("car_brand") or "BMW"

    if arabic:
        if price and budget:
            try:
                diff = int(budget) - int(price)
            except Exception:
                diff = None

            if diff is not None and diff >= 0:
                return f"اختيار قوي لو عايز {brand} مستعملة وتحت ميزانيتك بحوالي {format_money(diff, arabic=True)}."
            if diff is not None:
                return f"اختيار قوي، بس أعلى من ميزانيتك بحوالي {format_money(abs(diff), arabic=True)}."

        if condition == "used" and transmission == "automatic":
            return f"اختيار مناسب لو عايز {brand} مستعملة ومريحة في الاستخدام اليومي."

        if transmission == "automatic":
            return "مناسبة جدًا للاستخدام اليومي والزحمة."

        return "اختيار مناسب لو المواصفات دي قريبة من اللي بتدور عليه."

    if price and budget:
        try:
            diff = int(budget) - int(price)
        except Exception:
            diff = None

        if diff is not None and diff >= 0:
            return f"It is a strong fit and about {format_money(diff, arabic=False)} under your budget."
        if diff is not None:
            return f"It is a strong option, but about {format_money(abs(diff), arabic=False)} above your budget."

    if condition == "used" and transmission == "automatic":
        return "It is a practical used option for daily driving."

    return "It is a relevant option for what you are looking for."


def compose_cta(
    message: str,
    variables: Dict[str, Any],
    recent_messages: Optional[List[Dict[str, Any]]] = None,
    strategy: str = "soft_close",
) -> str:
    arabic = is_arabic_text(message)

    has_item = bool(variables.get("selected_item") or variables.get("matched_car_model"))
    has_budget = bool(variables.get("budget_max"))
    asked_viewing = already_asked_viewing(recent_messages, variables)

    if arabic:
        if strategy == "ask_phone":
            return "محتاج رقم للتواصل عشان نأكد المعاد."

        if strategy == "book":
            return "تحب أظبطلك معاينة؟"

        if strategy == "compare":
            return "تحب أقارنها لك ببديل تاني؟"

        if strategy == "advisor":
            return "تحب أقولك رأيي بصراحة؟"

        if has_item and has_budget and not asked_viewing:
            return "تحب أظبطلك معاينة؟"

        if has_item and not asked_viewing:
            return "تحب أقولك تفاصيلها أكتر ولا أظبطلك معاينة؟"

        return "تحب نكمل على الاختيار ده؟"

    if strategy == "ask_phone":
        return "I just need a contact number to confirm it."

    if strategy == "book":
        return "Would you like me to arrange a viewing?"

    if strategy == "compare":
        return "Would you like me to compare it with another option?"

    if has_item and not asked_viewing:
        return "Would you like more details or should I arrange a viewing?"

    return "Would you like to continue with this option?"


def compose_premium_entry_reply(
    message: str,
    variables: Dict[str, Any],
    recent_messages: Optional[List[Dict[str, Any]]] = None,
) -> str:
    summary = compose_item_summary(message, variables)
    frame = compose_value_frame(message, variables)
    cta = compose_cta(message, variables, recent_messages, strategy="soft_close")

    return f"{summary} {frame} {cta}".strip()


def compose_fact_reply(
    message: str,
    variables: Dict[str, Any],
    fact_type: str,
    recent_messages: Optional[List[Dict[str, Any]]] = None,
) -> Optional[str]:
    arabic = is_arabic_text(message)
    item = get_selected_item(variables)

    model = item.get("model") or variables.get("matched_car_model") or variables.get("car_brand") or ("العربية" if arabic else "the car")
    transmission = item.get("transmission") or variables.get("transmission")
    km = item.get("km") or variables.get("matched_car_km")
    price = item.get("price") or variables.get("matched_car_price")
    currency = item.get("currency") or variables.get("currency") or "EGP"

    cta = compose_cta(message, variables, recent_messages)

    if fact_type == "transmission" and transmission:
        if arabic:
            if transmission == "automatic":
                return f"أيوه، {model} أوتوماتيك ومريحة جدًا للاستخدام اليومي. {cta}"
            return f"{model} فتيسها {transmission}. {cta}"
        return f"Yes, {model} is {transmission}. {cta}"

    if fact_type == "km" and km:
        if arabic:
            return f"{model} عاملة {format_km(km, True)}. عداد مناسب لو حالتها وصيانتها كويسين. {cta}"
        return f"{model} has {format_km(km, False)}. {cta}"

    if fact_type == "price" and price:
        if arabic:
            return f"سعر {model} هو {format_money(price, currency, True)}. {cta}"
        return f"{model} is {format_money(price, currency, False)}. {cta}"

    return None


def compose_confirmation_reply(message: str, variables: Dict[str, Any]) -> str:
    arabic = is_arabic_text(message)
    item = get_selected_item(variables)

    model = item.get("model") or variables.get("matched_car_model") or variables.get("car_brand") or ("العربية" if arabic else "the car")
    date = variables.get("preferred_viewing_date")
    time_value = variables.get("preferred_viewing_time")
    location = variables.get("location")

    if arabic:
        parts = []
        if date:
            parts.append(str(date))
        if time_value:
            parts.append(f"الساعة {time_value}")
        if location:
            parts.append(f"في {location}")

        joined = " ".join(parts)
        return f"تمام، هظبطلك معاينة {model} {joined}. هنتواصل معاك لتأكيد التفاصيل."

    parts = []
    if date:
        parts.append(str(date))
    if time_value:
        parts.append(f"at {time_value}")
    if location:
        parts.append(f"in {location}")

    joined = " ".join(parts)
    return f"Great, I’ll set up the viewing for {model} {joined}. We’ll contact you to confirm the details."
