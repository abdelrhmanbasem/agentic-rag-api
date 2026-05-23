import re
from app.config import MOCK_MODE
from app.llm import chat_json, extraction_model


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


def extract_budget(text, existing_variables):
    budget_match = re.search(r"(\d+(?:\.\d+)?)\s*(million|m|k|thousand)?", text)
    if not budget_match:
        return None

    if not any(word in text for word in ["budget", "under", "up to", "million", "egp", "price", "cost"]):
        return None

    number = float(budget_match.group(1))
    unit = budget_match.group(2)

    if unit in ["million", "m"]:
        number = number * 1000000
    elif unit in ["k", "thousand"]:
        number = number * 1000

    return int(number)


def mock_extract_variables(schema, existing_variables, recent_messages, user_message):
    text = user_message.lower()
    updates = {}
    deletions = []
    intent = existing_variables.get("intent", "general_question") if existing_variables else "general_question"

    if "whatsapp" in text or "whats app" in text:
        if schema_has(schema, "preferred_contact_method"):
            updates["preferred_contact_method"] = "WhatsApp"

    if "call" in text and ("no call" in text or "no calls" in text or "don't call" in text or "dont call" in text):
        if schema_has(schema, "preferred_contact_method"):
            updates["preferred_contact_method"] = "WhatsApp"

    known_brands = ["bmw", "mercedes", "audi", "toyota", "hyundai", "kia", "nissan", "ford", "tesla"]
    for brand in known_brands:
        if brand in text and schema_has(schema, "car_brand"):
            updates["car_brand"] = brand.upper() if brand == "bmw" else brand.title()
            intent = "car_search"

    if "used" in text and schema_has(schema, "car_condition"):
        updates["car_condition"] = "used"
        intent = "car_search"

    if ("brand new" in text or "new car" in text or "new one" in text) and schema_has(schema, "car_condition"):
        updates["car_condition"] = "new"
        intent = "car_search"

    if "automatic" in text and schema_has(schema, "transmission"):
        updates["transmission"] = "automatic"

    if "manual" in text and schema_has(schema, "transmission"):
        updates["transmission"] = "manual"

    budget = extract_budget(text, existing_variables or {})
    if budget and schema_has(schema, "budget_max"):
        updates["budget_max"] = budget
        if schema_has(schema, "currency"):
            updates["currency"] = "EGP"
        intent = "car_search"

    if "forget" in text and "budget" in text:
        deletions.append("budget_max")

    clinic_services = [
        "cleaning",
        "teeth cleaning",
        "dental",
        "dentist",
        "consultation",
        "checkup",
        "check-up",
        "orthodontics",
        "xray",
        "x-ray",
        "blood test",
        "dermatology",
        "cardiology",
    ]

    for service in clinic_services:
        if service in text and schema_has(schema, "service_needed"):
            updates["service_needed"] = service
            intent = "service_question"

    if "doctor" in text and schema_has(schema, "doctor_preference"):
        doctor_match = re.search(r"doctor\s+([a-zA-Z]+)", user_message)
        if doctor_match:
            updates["doctor_preference"] = doctor_match.group(1)

    if "insurance" in text and schema_has(schema, "insurance_provider"):
        updates["insurance_provider"] = "mentioned"

    if "tomorrow" in text and schema_has(schema, "appointment_date"):
        updates["appointment_date"] = "tomorrow"

    if "today" in text and schema_has(schema, "appointment_date"):
        updates["appointment_date"] = "today"

    if "saturday" in text and schema_has(schema, "appointment_date"):
        updates["appointment_date"] = "Saturday"

    if "morning" in text and schema_has(schema, "appointment_time"):
        updates["appointment_time"] = "morning"

    if "afternoon" in text and schema_has(schema, "appointment_time"):
        updates["appointment_time"] = "afternoon"

    if "evening" in text and schema_has(schema, "appointment_time"):
        updates["appointment_time"] = "evening"

    if ("book" in text or "appointment" in text or "visit" in text) and (
        schema_has(schema, "appointment_date") or schema_has(schema, "service_needed")
    ):
        intent = "booking_request"

    phone_match = re.search(r"(\+?\d[\d\s\-]{7,}\d)", user_message)
    if phone_match and schema_has(schema, "phone_number"):
        updates["phone_number"] = phone_match.group(1).strip()

    name_match = re.search(r"my name is\s+([a-zA-Z\u0600-\u06FF ]+)", user_message, re.IGNORECASE)
    if name_match and schema_has(schema, "patient_name"):
        updates["patient_name"] = name_match.group(1).strip()

    if any(word in text for word in ["urgent", "emergency", "severe pain", "bleeding", "chest pain"]):
        if schema_has(schema, "urgency"):
            updates["urgency"] = "emergency"
        if schema_has(schema, "needs_human"):
            updates["needs_human"] = True
        intent = "urgent_medical_issue"

    if "refund" in text or "complaint" in text or "angry" in text or "unacceptable" in text:
        intent = "complaint"

    if updates and schema_has(schema, "lead_stage"):
        if intent == "booking_request":
            updates["lead_stage"] = "collecting_details"
        elif intent == "urgent_medical_issue":
            updates["lead_stage"] = "needs_human"
        else:
            updates["lead_stage"] = "qualified"

    missing = []
    for key, config in (schema or {}).items():
        if key == "intent":
            continue
        required = config.get("required", False) if isinstance(config, dict) else False
        if required and key not in updates and key not in (existing_variables or {}):
            missing.append(key)

    return {
        "intent": intent,
        "updates": updates,
        "deletions": deletions,
        "missing_variables": list(dict.fromkeys(missing)),
        "confidence": 0.75,
        "notes": "Mock extraction used because MOCK_MODE is enabled.",
    }


def gpt_extract_variables(schema, existing_variables, recent_messages, user_message):
    prompt = f"""
You are a variable extraction engine.

Update the conversation variables based on the latest user message.

Variable schema:
{schema}

Existing variables:
{existing_variables}

Recent messages:
{recent_messages}

Latest user message:
{user_message}

Rules:
- Extract only variables that exist in the schema.
- If the user changes their mind, update the old value.
- If the user contradicts previous info, prefer the latest explicit statement.
- If the user says to forget/remove something, add that key to deletions.
- Do not guess values.
- Return JSON only.

Return:
{{
  "intent": "string",
  "updates": {{}},
  "deletions": [],
  "missing_variables": [],
  "confidence": 0.0,
  "notes": "short internal summary"
}}
"""

    result = chat_json(
        extraction_model(),
        [{"role": "user", "content": prompt}],
        max_tokens=600,
    )

    return {
        "intent": result.get("intent", "general_question"),
        "updates": result.get("updates", {}),
        "deletions": result.get("deletions", []),
        "missing_variables": result.get("missing_variables", []),
        "confidence": result.get("confidence", 0.5),
        "notes": result.get("notes", ""),
    }


def extract_variables(schema, existing_variables, recent_messages, user_message):
    if MOCK_MODE:
        return mock_extract_variables(schema, existing_variables, recent_messages, user_message)

    return gpt_extract_variables(schema, existing_variables, recent_messages, user_message)
