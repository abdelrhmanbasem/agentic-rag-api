# app/conversation_brain.py
# Universal deterministic conversation intelligence layer.
#
# Goal:
# - Make cheap paths sound like a sharp human operator, not a robotic FAQ bot.
# - Keep token cost near-zero by using state + templates, not GPT.
# - Work across future assistants through workflow-aware response composition.
#
# Design principle:
# Clear > clever.
# Human > over-composed.
# Short but intelligent.
# Answer first, then guide the next step.

import re
import hashlib
from typing import Dict, Any, List, Optional


def is_arabic_text(text: str) -> bool:
    return bool(re.search(r"[\u0600-\u06FF]", text or ""))


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


def stable_pick(options: List[str], seed: str = "") -> str:
    if not options:
        return ""

    seed = seed or "default"
    digest = hashlib.md5(seed.encode("utf-8")).hexdigest()
    index = int(digest[:8], 16) % len(options)

    return options[index]


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


def car_model_label(variables: Dict[str, Any], arabic: bool = True) -> str:
    item = get_selected_item(variables)

    return (
        item.get("model")
        or variables.get("matched_car_model")
        or variables.get("car_brand")
        or ("العربية" if arabic else "the car")
    )


def get_recent_assistant_texts(recent_messages: Optional[List[Dict[str, Any]]]) -> List[str]:
    texts = []

    for msg in recent_messages or []:
        if msg.get("role") == "assistant":
            content = msg.get("content") or ""
            if content:
                texts.append(content)

    return texts[-5:]


def recently_said(recent_messages: Optional[List[Dict[str, Any]]], markers: List[str]) -> bool:
    recent_text = " ".join(get_recent_assistant_texts(recent_messages)).lower()
    recent_text = normalize_text(recent_text)

    return any(marker in recent_text for marker in markers)


def infer_buyer_state(message: str, variables: Dict[str, Any]) -> str:
    text = normalize_text(message)
    variables = variables or {}

    if any(x in text for x in ["غالي", "غاليه", "غالية", "كتير", "نهائي", "اخره", "آخره", "خصم", "تقسيط", "discount", "installment", "final price"]):
        return "price_sensitive"

    if any(x in text for x in ["احجز", "اشوفها", "أشوفها", "معاينة", "معاينه", "بكرة", "النهارده", "الساعة", "ميعاد", "موعد", "book", "schedule", "viewing"]):
        return "ready"

    if any(x in text for x in ["مش عارف", "اقارن", "مقارنة", "احسن", "افضل", "بديل", "alternative", "compare"]):
        return "comparing"

    if any(x in text for x in ["مش متاكد", "مش متأكد", "هفكر", "لسه", "maybe", "not sure"]):
        return "hesitant"

    if variables.get("budget_max"):
        return "qualified"

    if variables.get("matched_car_model") or variables.get("selected_item"):
        return "interested"

    return "curious"


def budget_fit_phrase(variables: Dict[str, Any], arabic: bool = True) -> str:
    item = get_selected_item(variables)

    price = item.get("price") or variables.get("matched_car_price")
    budget = variables.get("budget_max")
    currency = item.get("currency") or variables.get("currency") or "EGP"

    if not price or not budget:
        return ""

    try:
        price_int = int(price)
        budget_int = int(budget)
    except Exception:
        return ""

    diff = budget_int - price_int

    if arabic:
        if diff > 0:
            return f"وده تحت ميزانيتك بحوالي {format_money(diff, currency, arabic=True)}"
        if diff == 0:
            return "وده على حدود ميزانيتك بالظبط"
        return f"بس أعلى من ميزانيتك بحوالي {format_money(abs(diff), currency, arabic=True)}"

    if diff > 0:
        return f"it is about {format_money(diff, currency, arabic=False)} under your budget"
    if diff == 0:
        return "it is exactly at your budget"
    return f"it is about {format_money(abs(diff), currency, arabic=False)} above your budget"


def item_summary_sentence(variables: Dict[str, Any], message: str) -> str:
    arabic = is_arabic_text(message)
    item = get_selected_item(variables)

    model = car_model_label(variables, arabic)
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
            parts.append(f"عاملة {format_km(km, arabic=True)}")

        if price:
            parts.append(f"وسعرها {format_money(price, currency, arabic=True)}")

        return "، ".join(parts) + "."

    parts = [f"We have {model}"]

    if year:
        parts.append(f"from {year}")

    if transmission:
        parts.append(f"with {transmission} transmission")

    if km:
        parts.append(f"and {format_km(km, arabic=False)}")

    if price:
        parts.append(f"priced at {format_money(price, currency, arabic=False)}")

    return " ".join(parts) + "."


def simple_value_frame(variables: Dict[str, Any], message: str) -> str:
    arabic = is_arabic_text(message)
    item = get_selected_item(variables)

    model = car_model_label(variables, arabic)
    price = item.get("price") or variables.get("matched_car_price")
    budget = variables.get("budget_max")
    transmission = item.get("transmission") or variables.get("transmission")
    condition = item.get("condition") or variables.get("car_condition")

    if arabic:
        if price and budget:
            fit = budget_fit_phrase(variables, arabic=True)
            return f"اختيار قوي لو عايز BMW مستعملة وفي نفس الوقت {fit}."

        if transmission == "automatic" and condition == "used":
            return "اختيار مناسب لو عايز عربية مستعملة شيك ومريحة في الاستخدام اليومي."

        if transmission == "automatic":
            return "اختيار مناسب لو عايز عربية مريحة في الاستخدام اليومي والزحمة."

        return f"اختيار مناسب لو بتدور على {model} بالمواصفات دي."

    if price and budget:
        fit = budget_fit_phrase(variables, arabic=False)
        return f"It is a strong option if you want something in this range, and {fit}."

    if transmission == "automatic" and condition == "used":
        return "It is a strong option if you want a used car that is still comfortable for daily driving."

    if transmission == "automatic":
        return "It is a practical option for daily driving."

    return f"It is a relevant option if you are looking for {model}."


def choose_clean_cta(
    *,
    variables: Dict[str, Any],
    message: str,
    recent_messages: Optional[List[Dict[str, Any]]] = None,
    intent: str = "",
) -> str:
    arabic = is_arabic_text(message)
    buyer_state = infer_buyer_state(message, variables)
    asked = variables.get("asked_questions")
    if not isinstance(asked, list):
        asked = []

    already_asked_viewing = "viewing_interest" in asked or recently_said(
        recent_messages,
        ["تحب تشوفها", "معاينة", "معاينه", "viewing"],
    )

    already_asked_budget = "budget" in asked or recently_said(
        recent_messages,
        ["ميزانيتك", "budget"],
    )

    has_item = bool(variables.get("selected_item") or variables.get("matched_car_model"))
    has_budget = bool(variables.get("budget_max"))

    seed = message + str(variables.get("conversation_id", "")) + str(variables.get("workflow_stage", ""))

    if arabic:
        if buyer_state == "price_sensitive":
            return stable_pick(
                [
                    "تحب أخلي حد من الفريق يتابع معاك بخصوص التفاوض أو التقسيط؟",
                    "تحب أطلبلك متابعة من الفريق بخصوص السعر؟",
                    "لو حابب، نكمل على المعاينة ونشوف إمكانية التفاوض بعدها.",
                ],
                seed,
            )

        if buyer_state == "ready":
            return stable_pick(
                [
                    "تحب أظبطلك المعاينة على أي يوم؟",
                    "تمام، أنسب معاد ليك إمتى؟",
                    "حلو، نثبت المعاينة على إمتى؟",
                ],
                seed,
            )

        if buyer_state == "comparing":
            return stable_pick(
                [
                    "تحب أقارنلك بينها وبين اختيار تاني؟",
                    "تحب أشوفلك بديل قريب منها في نفس الرينج؟",
                    "أقدر أقولك هي مميزة في إيه مقارنة ببديل تاني.",
                ],
                seed,
            )

        if buyer_state == "hesitant":
            return stable_pick(
                [
                    "تحب أقولك أهم نقطتين فيها يساعدوك تقرر؟",
                    "تحب تعرف عنها حاجة معينة قبل ما تقرر؟",
                    "ولا يهمك، نقدر نمشيها واحدة واحدة.",
                ],
                seed,
            )

        if has_item and has_budget and not already_asked_viewing:
            return stable_pick(
                [
                    "تحب أظبطلك معاد تشوفها؟",
                    "لو مناسبة ليك، نقدر نرتبلك معاينة.",
                    "تحب تشوفها على الطبيعة؟",
                ],
                seed,
            )

        if has_item and not already_asked_viewing:
            return stable_pick(
                [
                    "تحب أقولك تفاصيلها أكتر ولا نرتبلك معاينة؟",
                    "تحب تعرف حالتها أكتر ولا تشوفها على الطبيعة؟",
                    "تحب أكملك تفاصيلها ولا أظبطلك معاد معاينة؟",
                ],
                seed,
            )

        if not has_budget and not already_asked_budget:
            return stable_pick(
                [
                    "ميزانيتك في حدود كام؟",
                    "تحب أدورلك في رينج كام؟",
                    "إيه حدود الميزانية اللي حابب تمشي فيها؟",
                ],
                seed,
            )

        return stable_pick(
            [
                "تحب نكمل على الاختيار ده؟",
                "تحب أقولك تفاصيل أكتر؟",
                "تحب أشوفلك بديل كمان؟",
            ],
            seed,
        )

    if buyer_state == "price_sensitive":
        return "Would you like someone from the team to follow up about negotiation or installments?"

    if buyer_state == "ready":
        return "Great, what day works best for the viewing?"

    if has_item and not already_asked_viewing:
        return "Would you like the details or should we arrange a viewing?"

    if not has_budget and not already_asked_budget:
        return "What budget range would you like to stay within?"

    return "Would you like to continue with this option?"


def compose_car_entry_answer(
    *,
    variables: Dict[str, Any],
    message: str,
    recent_messages: Optional[List[Dict[str, Any]]] = None,
) -> str:
    arabic = is_arabic_text(message)

    summary = item_summary_sentence(variables, message)
    frame = simple_value_frame(variables, message)
    cta = choose_clean_cta(
        variables=variables,
        message=message,
        recent_messages=recent_messages,
        intent="entry",
    )

    if arabic:
        return f"{summary} {frame} {cta}"

    return f"{summary} {frame} {cta}"


def compose_car_fact_answer(
    *,
    message: str,
    variables: Dict[str, Any],
    fact_type: str,
    recent_messages: Optional[List[Dict[str, Any]]] = None,
) -> Optional[str]:
    arabic = is_arabic_text(message)
    item = get_selected_item(variables)

    model = car_model_label(variables, arabic)
    transmission = item.get("transmission") or variables.get("transmission")
    km = item.get("km") or variables.get("matched_car_km")
    price = item.get("price") or variables.get("matched_car_price")
    currency = item.get("currency") or variables.get("currency") or "EGP"

    cta = choose_clean_cta(
        variables=variables,
        message=message,
        recent_messages=recent_messages,
        intent=fact_type,
    )

    if fact_type == "transmission" and transmission:
        if arabic:
            if transmission == "automatic":
                return f"أيوه، {model} أوتوماتيك. مريحة جدًا للاستخدام اليومي والزحمة. {cta}"
            if transmission == "manual":
                return f"{model} مانيوال. لو ده مناسب لاستخدامك، أقدر أكملك باقي التفاصيل. {cta}"
            return f"{model} فتيسها {transmission}. {cta}"

        if transmission == "automatic":
            return f"Yes, {model} is automatic, which is convenient for daily driving. {cta}"
        if transmission == "manual":
            return f"{model} is manual. {cta}"
        return f"{model} transmission is {transmission}. {cta}"

    if fact_type == "km" and km:
        if arabic:
            return f"{model} عاملة {format_km(km, arabic=True)}. عداد مناسب على موديلها وسعرها. {cta}"
        return f"{model} has {format_km(km, arabic=False)}. {cta}"

    if fact_type == "price" and price:
        fit = budget_fit_phrase(variables, arabic=arabic)

        if arabic:
            if fit:
                return f"سعر {model} هو {format_money(price, currency, arabic=True)}، {fit}. {cta}"
            return f"سعر {model} هو {format_money(price, currency, arabic=True)}. {cta}"

        if fit:
            return f"{model} price is {format_money(price, currency, arabic=False)}, and {fit}. {cta}"
        return f"{model} price is {format_money(price, currency, arabic=False)}. {cta}"

    if fact_type == "budget" and price:
        fit = budget_fit_phrase(variables, arabic=arabic)

        if arabic:
            if fit:
                return f"تمام، كده أنت في الرينج الصح. {model} سعرها {format_money(price, currency, arabic=True)}، {fit}. {cta}"
            return f"{model} سعرها {format_money(price, currency, arabic=True)}. {cta}"

        if fit:
            return f"Good, {model} is {format_money(price, currency, arabic=False)}, and {fit}. {cta}"
        return f"{model} is {format_money(price, currency, arabic=False)}. {cta}"

    return None


def compose_price_objection_answer(
    *,
    message: str,
    variables: Dict[str, Any],
    recent_messages: Optional[List[Dict[str, Any]]] = None,
) -> str:
    arabic = is_arabic_text(message)
    item = get_selected_item(variables)

    model = car_model_label(variables, arabic)
    price = item.get("price") or variables.get("matched_car_price")
    currency = item.get("currency") or variables.get("currency") or "EGP"
    fit = budget_fit_phrase(variables, arabic=arabic)

    cta = choose_clean_cta(
        variables=variables,
        message=message,
        recent_messages=recent_messages,
        intent="price_objection",
    )

    if arabic:
        if price:
            fit_part = f"، {fit}" if fit else ""
            return (
                f"فاهمك، طبيعي تسأل على السعر. السعر الحالي لـ {model} هو "
                f"{format_money(price, currency, arabic=True)}{fit_part}. "
                f"التفاوض أو التقسيط بيتحدد حسب الجدية والمعاينة. {cta}"
            )

        return f"فاهمك، السعر مهم طبعًا. أقدر أشوفلك بديل أقرب لميزانيتك أو أخلي حد من الفريق يتابع معاك. {cta}"

    if price:
        return (
            f"I understand. {model} is currently {format_money(price, currency, arabic=False)}. "
            f"Negotiation or installments depend on the viewing and seriousness. {cta}"
        )

    return f"I understand. I can suggest a closer option or have the team follow up about pricing. {cta}"


def compose_soft_close(message: str, variables: Dict[str, Any]) -> str:
    arabic = is_arabic_text(message)

    if arabic:
        return "تمام، ولا يهمك. لو حبيت تشوف اختيار تاني أو ترجع للعربية دي، أنا معاك."

    return "No problem. If you want another option or want to come back to this one, I’m here."


def compose_handoff_ack(message: str, variables: Dict[str, Any]) -> str:
    arabic = is_arabic_text(message)

    if arabic:
        return "تمام، هخلي حد من الفريق يتابع معاك. ولو تحب تضيف أي تفاصيل قبل ما يكلموك ابعتها هنا."

    return "Sure, I’ll have someone from the team follow up with you. You can send any extra details here before they contact you."


def compose_service_entry_answer(
    *,
    variables: Dict[str, Any],
    message: str,
    recent_messages: Optional[List[Dict[str, Any]]] = None,
) -> str:
    arabic = is_arabic_text(message)
    service = variables.get("service_needed")

    if arabic:
        if service:
            return f"تمام، أقدر أساعدك تحجز {service}. عشان أظبطهولك صح، تحب المعاد يكون إمتى؟"
        return "تمام، أقدر أساعدك في الحجز. تحب تحجز لأي خدمة بالظبط؟"

    if service:
        return f"Sure, I can help you book {service}. What day works best for you?"

    return "Sure, I can help with booking. What service would you like to book?"
