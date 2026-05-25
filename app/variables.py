import re
from app.config import MOCK_MODE
from app.llm import chat_json, extraction_model


CANONICAL_INTENTS = {
    "general_question",
    "car_search",
    "viewing_request",
    "booking_request",
    "service_question",
    "insurance_question",
    "complaint",
    "human_handoff",
    "urgent_medical_issue",
    "variable_update",
    "provide_contact",
}


INTENT_ALIASES = {
    "car_inquiry": "car_search",
    "car_enquiry": "car_search",
    "car_purchase": "car_search",
    "car_buying": "car_search",
    "buy_car": "car_search",
    "vehicle_search": "car_search",
    "vehicle_inquiry": "car_search",
    "vehicle_enquiry": "car_search",
    "vehicle_purchase": "car_search",
    "inventory_question": "car_search",
    "product_search": "car_search",
    "product_inquiry": "car_search",

    "schedule_viewing": "viewing_request",
    "book_viewing": "viewing_request",
    "viewing": "viewing_request",
    "viewing_inquiry": "viewing_request",
    "test_drive": "viewing_request",
    "schedule_test_drive": "viewing_request",
    "visit_request": "viewing_request",

    "appointment": "booking_request",
    "appointment_request": "booking_request",
    "schedule_appointment": "booking_request",
    "book_appointment": "booking_request",
    "clinic_booking": "booking_request",
    "reservation": "booking_request",

    "handoff": "human_handoff",
    "human": "human_handoff",
    "agent_request": "human_handoff",
    "talk_to_human": "human_handoff",

    "complain": "complaint",
    "customer_complaint": "complaint",

    "emergency": "urgent_medical_issue",
    "urgent": "urgent_medical_issue",
    "urgent_case": "urgent_medical_issue",
    "medical_emergency": "urgent_medical_issue",
}


def normalize_intent(intent: str) -> str:
    intent = (intent or "general_question").strip().lower()

    if intent in CANONICAL_INTENTS:
        return intent

    return INTENT_ALIASES.get(intent, intent)


def apply_variable_patch(existing, updates, deletions):
    result = dict(existing or {})

    for key in deletions or []:
        if key in result:
            del result[key]

    for key, value in (updates or {}).items():
        if value is not None:
            result[key] = value

    return result


def schema_has(schema, key):
    return key in (schema or {})


def has_arabic(text):
    return bool(re.search(r"[\u0600-\u06FF]", text or ""))


def normalize_arabic(text):
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


def extract_budget(text, existing_variables):
    normalized = normalize_arabic(text.lower())

    budget_match = re.search(
        r"(\d+(?:\.\d+)?)\s*(million|m|k|thousand|مليون|الف|ألف)?",
        normalized,
    )
    if not budget_match:
        return None

    budget_words = [
        "budget",
        "under",
        "up to",
        "million",
        "egp",
        "price",
        "cost",
        "ميزانيه",
        "ميزانية",
        "لحد",
        "حدي",
        "حد اقصي",
        "حد أقصي",
        "مليون",
        "جنيه",
        "سعر",
        "بكام",
        "تكلفه",
        "تكلفة",
    ]

    if not any(word in normalized for word in budget_words):
        return None

    number = float(budget_match.group(1))
    unit = budget_match.group(2)

    if unit in ["million", "m", "مليون"]:
        number = number * 1000000
    elif unit in ["k", "thousand", "الف", "ألف"]:
        number = number * 1000

    return int(number)


def extract_phone(user_message):
    phone_match = re.search(r"(\+?\d[\d\s\-]{7,}\d)", user_message)
    if not phone_match:
        return None

    return phone_match.group(1).strip()


def extract_name(user_message):
    patterns = [
        r"my name is\s+([a-zA-Z\u0600-\u06FF ]+)",
        r"اسمي\s+([a-zA-Z\u0600-\u06FF ]+)",
        r"انا اسمي\s+([a-zA-Z\u0600-\u06FF ]+)",
        r"أنا اسمي\s+([a-zA-Z\u0600-\u06FF ]+)",
    ]

    for pattern in patterns:
        match = re.search(user_message=user_message, pattern=pattern, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()

    return None
