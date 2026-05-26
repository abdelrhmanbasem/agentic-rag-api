# app/fast_path.py
# Universal pre-router fast path for all assistants.
# Goal: answer obvious state/workflow messages before router/extraction/RAG/GPT.
# This saves tokens while keeping conversations fluid.

import re
from typing import Dict, Any, Optional, List


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


def schema_has_any(schema: Dict[str, Any], keys: List[str]) -> bool:
    schema = schema or {}
    return any(key in schema for key in keys)


def infer_workflow_type(schema: Dict[str, Any], assistant_id: str = "") -> str:
    schema = schema or {}
    assistant_id = (assistant_id or "").lower()

    car_keys = [
        "car_brand",
        "car_condition",
        "transmission",
        "budget_max",
        "matched_car_model",
        "matched_car_year",
        "matched_car_km",
        "matched_car_price",
        "preferred_viewing_date",
        "preferred_viewing_time",
    ]

    service_keys = [
        "service_needed",
        "appointment_date",
        "appointment_time",
        "doctor_preference",
        "patient_name",
        "insurance_provider",
    ]

    real_estate_keys = [
        "property_type",
        "property_location",
        "bedrooms",
        "bathrooms",
        "rent_budget",
        "purchase_budget",
        "viewing_date",
        "viewing_time",
    ]

    ecommerce_keys = [
        "product_name",
        "product_category",
        "order_id",
        "shipping_address",
        "delivery_date",
    ]

    if schema_has_any(schema, car_keys) or any(
        marker in assistant_id
        for marker in ["car", "cars", "auto", "vehicle", "dealer"]
    ):
        return "car_sales"

    if schema_has_any(schema, service_keys) or any(
        marker in assistant_id
        for marker in ["clinic", "doctor", "medical", "dental", "dentist", "health", "salon", "spa"]
    ):
        return "service_booking"

    if schema_has_any(schema, real_estate_keys) or any(
        marker in assistant_id
        for marker in ["real_estate", "property", "rent", "broker"]
    ):
        return "real_estate"

    if schema_has_any(schema, ecommerce_keys) or any(
        marker in assistant_id
        for marker in ["shop", "store", "ecommerce", "order"]
    ):
        return "ecommerce"

    return "general"


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


def format_money_ar(amount, currency="EGP") -> str:
    try:
        formatted = f"{int(amount):,}"
    except Exception:
        formatted = str(amount)

    if currency == "EGP":
        return f"{formatted} جنيه"

    return f"{formatted} {currency}"


def format_money_en(amount, currency="EGP") -> str:
    try:
        formatted = f"{int(amount):,}"
    except Exception:
        formatted = str(amount)

    return f"{formatted} {currency}"


def get_model_name(variables: Dict[str, Any], arabic: bool = False) -> str:
    item = get_selected_item(variables)
    return (
        item.get("model")
        or variables.get("matched_car_model")
        or variables.get("car_brand")
        or ("العربية" if arabic else "the item")
    )


def user_is_thanks(message: str) -> bool:
    text = normalize_text(message)

    markers = [
        "شكرا",
        "شكرًا",
        "تسلم",
        "ميرسي",
        "تمام شكرا",
        "تمام شكرًا",
        "thanks",
        "thank you",
        "thx",
        "ok thanks",
    ]

    return any(marker in text for marker in markers)


def user_is_short_ack(message: str) -> bool:
    text = normalize_text(message)

    exact = {
        "تمام",
        "ماشي",
        "حاضر",
        "اشطا",
        "اوك",
        "اوكي",
        "ok",
        "okay",
        "yes",
        "yep",
        "اه",
        "اها",
        "ايوه",
        "ايوة",
        "تمام ماشي",
    }

    return text in exact


def user_is_no(message: str) -> bool:
    text = normalize_text(message)

    exact = {
        "لا",
        "لأ",
        "لاء",
        "no",
        "nope",
        "مش عايز",
        "مش عاوز",
        "لا شكرا",
        "لا شكرًا",
    }

    return text in exact


def user_wants_human(message: str) -> bool:
    text = normalize_text(message)

    markers = [
        "كلموني",
        "حد يكلمني",
        "عايز اكلم حد",
        "عايز حد",
        "اكلم انسان",
        "موظف",
        "مندوب",
        "call me",
        "human",
        "agent",
        "representative",
        "talk to someone",
    ]

    return any(marker in text for marker in markers)


def user_is_repairing(message: str) -> bool:
    text = normalize_text(message)

    markers = [
        "مش قصدي",
        "مش ده قصدي",
        "مش فاهمني",
        "انت مش فاهم",
        "لا مش كده",
        "wrong",
        "not what i mean",
        "you misunderstood",
    ]

    return any(marker in text for marker in markers)


def user_has_objection(message: str) -> bool:
    text = normalize_text(message)

    markers = [
        "غالي",
        "غاليه",
        "غالية",
        "السعر عالي",
        "كتير",
        "خصم",
        "تقسيط",
        "قسط",
        "مش مناسب",
        "اقل",
        "أقل",
        "expensive",
        "too expensive",
        "discount",
        "installment",
        "installments",
        "lower price",
    ]

    return any(marker in text for marker in markers)


def asks_transmission(message: str) -> bool:
    text = normalize_text(message)

    markers = [
        "اوتوماتيك",
        "اتوماتيك",
        "أوتوماتيك",
        "مانيوال",
        "automatic",
        "manual",
        "transmission",
    ]

    return any(marker in text for marker in markers)


def asks_km_or_mileage(message: str) -> bool:
    text = normalize_text(message)

    markers = [
        "كام كيلو",
        "عامله كام",
        "عاملة كام",
        "ماشية كام",
        "ماشيه كام",
        "كيلو",
        "km",
        "kilometers",
        "mileage",
    ]

    return any(marker in text for marker in markers)


def asks_price(message: str) -> bool:
    text = normalize_text(message)

    markers = [
        "بكام",
        "سعر",
        "سعرها",
        "سعره",
        "price",
        "cost",
        "how much",
    ]

    return any(marker in text for marker in markers)


def asks_budget_fit(message: str) -> bool:
    text = normalize_text(message)

    markers = [
        "ميزانيتي",
        "ميزانية",
        "الميزانية",
        "لحد",
        "مليون",
        "ينفع",
        "مناسب",
        "budget",
        "under",
        "up to",
        "fit",
    ]

    return any(marker in text for marker in markers)


def wants_action_or_viewing(message: str) -> bool:
    text = normalize_text(message)

    markers = [
        "احجز",
        "احجزها",
        "عايز اشوف",
        "عايز اشوفها",
        "اشوفها",
        "أشوفها",
        "معاينة",
        "معاينه",
        "اتفرج",
        "اروح",
        "book",
        "schedule",
        "view",
        "visit",
        "see it",
    ]

    return any(marker in text for marker in markers)


def has_date(variables: Dict[str, Any], workflow: str) -> bool:
    if workflow == "car_sales":
        return bool(variables.get("preferred_viewing_date"))

    if workflow == "service_booking":
        return bool(variables.get("appointment_date"))

    return bool(
        variables.get("preferred_date")
        or variables.get("appointment_date")
        or variables.get("viewing_date")
    )


def has_time(variables: Dict[str, Any], workflow: str) -> bool:
    if workflow == "car_sales":
        return bool(variables.get("preferred_viewing_time") or variables.get("appointment_time"))

    if workflow == "service_booking":
        return bool(variables.get("appointment_time"))

    return bool(
        variables.get("preferred_time")
        or variables.get("appointment_time")
        or variables.get("viewing_time")
    )


def has_location(variables: Dict[str, Any]) -> bool:
    return bool(
        variables.get("location")
        or variables.get("location_branch")
        or variables.get("property_location")
    )


def has_phone(variables: Dict[str, Any]) -> bool:
    return bool(variables.get("phone_number"))


def get_last_assistant_text(recent_messages: List[Dict[str, Any]]) -> str:
    for msg in reversed(recent_messages or []):
        if msg.get("role") == "assistant":
            return msg.get("content", "") or ""

    return ""


def infer_yes_no_context(last_assistant_message: str) -> str:
    text = normalize_text(last_assistant_message)

    if any(marker in text for marker in ["تحب تشوف", "معاينة", "viewing", "schedule a viewing"]):
        return "viewing_interest"

    if any(marker in text for marker in ["بديل", "alternative", "اختيار تاني"]):
        return "alternative_interest"

    if any(marker in text for marker in ["تفاوض", "تقسيط", "discount", "installment"]):
        return "human_followup_interest"

    if any(marker in text for marker in ["المعاد", "ميعاد", "امتى", "when", "what day"]):
        return "date_time_collection"

    return "general_ack"


def build_state_fact_answer(message: str, variables: Dict[str, Any]) -> Optional[str]:
    arabic = is_arabic_text(message)
    item = get_selected_item(variables)

    if not item and not variables.get("matched_car_model"):
        return None

    model = get_model_name(variables, arabic)
    transmission = item.get("transmission") or variables.get("transmission")
    km = item.get("km") or variables.get("matched_car_km")
    price = item.get("price") or variables.get("matched_car_price")
    currency = item.get("currency") or variables.get("currency") or "EGP"
    budget = variables.get("budget_max")

    if asks_transmission(message) and transmission:
        if arabic:
            if transmission == "automatic":
                return f"أيوه، {model} أوتوماتيك."
            if transmission == "manual":
                return f"{model} مانيوال."
            return f"{model} فتيسها {transmission}."

        if transmission == "automatic":
            return f"Yes, {model} is automatic."
        if transmission == "manual":
            return f"{model} is manual."
        return f"{model} transmission is {transmission}."

    if asks_km_or_mileage(message) and km:
        try:
            km_text = f"{int(km):,}"
        except Exception:
            km_text = str(km)

        if arabic:
            return f"{model} عاملة {km_text} كيلو."

        return f"{model} has {km_text} km."

    if asks_price(message) and price:
        if arabic:
            return f"سعر {model} هو {format_money_ar(price, currency)}."

        return f"{model} price is {format_money_en(price, currency)}."

    if asks_budget_fit(message) and budget and price:
        try:
            fits = int(price) <= int(budget)
        except Exception:
            fits = None

        if fits is True:
            if arabic:
                return f"أيوه، {model} مناسبة لميزانيتك. سعرها {format_money_ar(price, currency)}."

            return f"Yes, {model} fits your budget. Its price is {format_money_en(price, currency)}."

        if fits is False:
            if arabic:
                return f"{model} أعلى من ميزانيتك شوية. سعرها {format_money_ar(price, currency)}."

            return f"{model} is slightly above your budget. Its price is {format_money_en(price, currency)}."

    return None


def build_workflow_fast_answer(
    message: str,
    schema: Dict[str, Any],
    assistant_id: str,
    variables: Dict[str, Any],
    recent_messages: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Returns:
    {
        "answer": "...",
        "model_tier": "fast_path",
        "action": "...",
        "updates": {...},
        "skip_summary": True,
        "skip_memory": True
    }

    Or None if the normal pipeline should continue.
    """
    variables = variables or {}
    workflow = infer_workflow_type(schema, assistant_id)
    arabic = is_arabic_text(message)
    stage = variables.get("workflow_stage", "")
    item = get_selected_item(variables)
    last_assistant = get_last_assistant_text(recent_messages)
    yes_no_context = infer_yes_no_context(last_assistant)

    if user_is_thanks(message):
        return {
            "answer": "العفو، تحت أمرك في أي وقت." if arabic else "You’re welcome. I’m here if you need anything else.",
            "model_tier": "fast_path",
            "action": "close_conversation",
            "skip_summary": True,
            "skip_memory": True,
        }

    if user_wants_human(message):
        return {
            "answer": "تمام، هخلي حد من الفريق يتابع معاك." if arabic else "Sure, I’ll have someone from the team follow up with you.",
            "model_tier": "fast_path",
            "action": "handoff_to_human",
            "updates": {
                "needs_human": True,
                "handoff_reason": "user_requested_human",
            },
            "skip_summary": False,
            "skip_memory": False,
        }

    if user_is_repairing(message):
        return {
            "answer": (
                "تمام، حقك عليا. تقصد تعدل النوع أو الميزانية، ولا تدور على اختيار مختلف تمامًا؟"
                if arabic
                else "Got it, sorry about that. Do you want to change the brand or budget, or look for something different?"
            ),
            "model_tier": "fast_path",
            "action": "repair_conversation",
            "skip_summary": True,
            "skip_memory": True,
        }

    fact_answer = build_state_fact_answer(message, variables)
    if fact_answer:
        return {
            "answer": fact_answer,
            "model_tier": "fast_path",
            "action": "answer_from_state",
            "skip_summary": True,
            "skip_memory": True,
        }

    if user_has_objection(message):
        model = get_model_name(variables, arabic)
        price = item.get("price") or variables.get("matched_car_price")
        currency = item.get("currency") or variables.get("currency") or "EGP"
        budget = variables.get("budget_max")

        if arabic:
            if price and budget:
                try:
                    if int(price) <= int(budget):
                        answer = (
                            f"فاهمك. سعر {model} هو {format_money_ar(price, currency)}، "
                            f"وده داخل ميزانيتك اللي قلتها. لو حابب، أقدر أخلي حد من الفريق يتابع معاك بخصوص التفاوض أو التقسيط."
                        )
                    else:
                        answer = (
                            f"فاهمك. سعر {model} الحالي {format_money_ar(price, currency)}. "
                            f"أقدر أقولك على بديل أقرب لميزانيتك، أو أخلي حد من الفريق يتابع معاك بخصوص التفاوض أو التقسيط."
                        )
                except Exception:
                    answer = "فاهمك. أقدر أقولك على بديل أقرب لميزانيتك، أو أخلي حد من الفريق يتابع معاك بخصوص السعر."
            elif price:
                answer = (
                    f"فاهمك. سعر {model} الحالي {format_money_ar(price, currency)}. "
                    f"أقدر أقولك على بديل أقرب لميزانيتك، أو أخلي حد من الفريق يتابع معاك بخصوص التفاوض أو التقسيط."
                )
            else:
                answer = "فاهمك. أقدر أقولك على بديل أقرب لميزانيتك، أو أخلي حد من الفريق يتابع معاك بخصوص السعر."
        else:
            if price:
                answer = (
                    f"I understand. {model} is currently {format_money_en(price, currency)}. "
                    f"I can suggest a closer alternative or have someone follow up about negotiation or installments."
                )
            else:
                answer = "I understand. I can suggest a closer alternative or have someone follow up about the price."

        return {
            "answer": answer,
            "model_tier": "fast_path",
            "action": "handle_objection",
            "updates": {
                "needs_human": True,
                "handoff_reason": "price_or_objection",
            },
            "skip_summary": False,
            "skip_memory": False,
        }

    if workflow == "car_sales":
        model = get_model_name(variables, arabic)

        if user_is_short_ack(message):
            if yes_no_context == "viewing_interest":
                return {
                    "answer": "تمام، تحب المعاد يكون إمتى؟" if arabic else "Great, what day would you prefer for the viewing?",
                    "model_tier": "fast_path",
                    "action": "ask_viewing_date",
                    "updates": {
                        "intent": "viewing_request",
                        "workflow_stage": "viewing_requested",
                    },
                    "skip_summary": True,
                    "skip_memory": True,
                }

            if yes_no_context == "alternative_interest":
                return None

            if yes_no_context == "human_followup_interest":
                return {
                    "answer": "تمام، هخلي حد من الفريق يتابع معاك." if arabic else "Sure, I’ll have someone from the team follow up with you.",
                    "model_tier": "fast_path",
                    "action": "handoff_to_human",
                    "updates": {
                        "needs_human": True,
                        "handoff_reason": "accepted_human_followup",
                    },
                    "skip_summary": False,
                    "skip_memory": False,
                }

            if stage == "viewing_requested":
                if not has_date(variables, workflow):
                    return {
                        "answer": "تمام، تحب المعاد يكون إمتى؟" if arabic else "Sure, what day would you prefer?",
                        "model_tier": "fast_path",
                        "action": "ask_viewing_date",
                        "skip_summary": True,
                        "skip_memory": True,
                    }

                if not has_time(variables, workflow):
                    return {
                        "answer": "تمام، والساعة كام يناسبك؟" if arabic else "Great, what time works best for you?",
                        "model_tier": "fast_path",
                        "action": "ask_viewing_time",
                        "skip_summary": True,
                        "skip_memory": True,
                    }

                if not has_location(variables):
                    return {
                        "answer": "تمام، والمكان فين؟" if arabic else "Great, what location works for you?",
                        "model_tier": "fast_path",
                        "action": "ask_location",
                        "skip_summary": True,
                        "skip_memory": True,
                    }

            if item or variables.get("matched_car_model"):
                return {
                    "answer": "تحب أحددلك معاد للمعاينة؟" if arabic else "Would you like me to schedule a viewing?",
                    "model_tier": "fast_path",
                    "action": "ask_viewing_interest",
                    "skip_summary": True,
                    "skip_memory": True,
                }

        if user_is_no(message):
            return {
                "answer": "تمام، ولا يهمك. لو حبيت تشوف اختيار تاني أنا معاك." if arabic else "No problem. If you want to see another option, I’m here.",
                "model_tier": "fast_path",
                "action": "soft_close",
                "skip_summary": True,
                "skip_memory": True,
            }

        if wants_action_or_viewing(message):
            if has_date(variables, workflow) and has_time(variables, workflow) and has_location(variables) and has_phone(variables):
                date = variables.get("preferred_viewing_date")
                time = variables.get("preferred_viewing_time") or variables.get("appointment_time")
                location = variables.get("location")

                answer = (
                    f"تمام، كده طلب المعاينة لـ {model} يوم {date} الساعة {time} في {location}. هنتواصل معاك على نفس الرقم لتأكيد التفاصيل."
                    if arabic
                    else f"Great, the viewing request for {model} is set for {date} at {time} in {location}. We’ll contact you on the same number to confirm the details."
                )

                return {
                    "answer": answer,
                    "model_tier": "fast_path",
                    "action": "confirm_viewing",
                    "updates": {
                        "lead_stage": "confirmed",
                        "workflow_stage": "confirmed",
                    },
                    "skip_summary": False,
                    "skip_memory": False,
                }

            if not has_date(variables, workflow):
                return {
                    "answer": "تمام، تحب المعاد يكون إمتى؟" if arabic else "Sure, what day would you prefer for the viewing?",
                    "model_tier": "fast_path",
                    "action": "ask_viewing_date",
                    "updates": {
                        "intent": "viewing_request",
                        "workflow_stage": "viewing_requested",
                    },
                    "skip_summary": True,
                    "skip_memory": True,
                }

            if not has_time(variables, workflow):
                return {
                    "answer": "تمام، والساعة كام يناسبك؟" if arabic else "Great, what time works best for you?",
                    "model_tier": "fast_path",
                    "action": "ask_viewing_time",
                    "updates": {
                        "intent": "viewing_request",
                        "workflow_stage": "viewing_requested",
                    },
                    "skip_summary": True,
                    "skip_memory": True,
                }

            if not has_location(variables):
                return {
                    "answer": "تمام، والمكان فين؟" if arabic else "Great, what location works for you?",
                    "model_tier": "fast_path",
                    "action": "ask_location",
                    "updates": {
                        "intent": "viewing_request",
                        "workflow_stage": "viewing_requested",
                    },
                    "skip_summary": True,
                    "skip_memory": True,
                }

            if not has_phone(variables):
                return {
                    "answer": "تمام، ابعتلي رقم للتواصل عشان نأكد التفاصيل." if arabic else "Great, please send a contact number so we can confirm the details.",
                    "model_tier": "fast_path",
                    "action": "ask_phone",
                    "updates": {
                        "intent": "viewing_request",
                        "workflow_stage": "viewing_requested",
                    },
                    "skip_summary": True,
                    "skip_memory": True,
                }

    if workflow == "service_booking":
        if user_is_short_ack(message):
            if not variables.get("service_needed"):
                return {
                    "answer": "تمام، تحب تحجز لأي خدمة؟" if arabic else "Sure, what service would you like to book?",
                    "model_tier": "fast_path",
                    "action": "ask_service",
                    "skip_summary": True,
                    "skip_memory": True,
                }

            if not has_date(variables, workflow):
                return {
                    "answer": "تمام، تحب المعاد يكون إمتى؟" if arabic else "Sure, what day would you prefer?",
                    "model_tier": "fast_path",
                    "action": "ask_booking_date",
                    "skip_summary": True,
                    "skip_memory": True,
                }

            if not has_time(variables, workflow):
                return {
                    "answer": "تمام، والساعة كام يناسبك؟" if arabic else "Great, what time works best for you?",
                    "model_tier": "fast_path",
                    "action": "ask_booking_time",
                    "skip_summary": True,
                    "skip_memory": True,
                }

    return None


def should_try_fast_path(message: str, variables: Dict[str, Any], schema: Dict[str, Any]) -> bool:
    """
    Conservative gate. Only try fast path when we have state or the message is obviously simple.
    This avoids blocking rich first-turn RAG/intelligence.
    """
    text = normalize_text(message)
    variables = variables or {}

    if not text:
        return False

    obvious_simple = (
        user_is_thanks(message)
        or user_is_short_ack(message)
        or user_is_no(message)
        or user_wants_human(message)
        or user_is_repairing(message)
        or user_has_objection(message)
        or asks_transmission(message)
        or asks_km_or_mileage(message)
        or asks_price(message)
        or asks_budget_fit(message)
        or wants_action_or_viewing(message)
    )

    has_state = bool(
        variables.get("selected_item")
        or variables.get("matched_car_model")
        or variables.get("workflow_stage")
        or variables.get("intent") in ["viewing_request", "booking_request"]
    )

    return bool(obvious_simple and has_state)
