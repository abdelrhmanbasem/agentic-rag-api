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


def mock_extract_variables(schema, existing_variables, recent_messages, user_message):
    text = user_message.lower()
    updates = {}
    deletions = []
    intent = existing_variables.get("intent", "general_question") if existing_variables else "general_question"

    known_brands = ["bmw", "mercedes", "audi", "toyota", "hyundai", "kia", "nissan", "ford", "tesla"]
    for brand in known_brands:
        if brand in text:
            updates["car_brand"] = brand.upper() if brand in ["bmw"] else brand.title()
            intent = "car_search"

    if "used" in text:
        updates["car_condition"] = "used"
        intent = "car_search"

    if "new" in text and "new one" in text or "brand new" in text:
        updates["car_condition"] = "new"
        intent = "car_search"

    if "automatic" in text:
        updates["transmission"] = "automatic"

    if "manual" in text:
        updates["transmission"] = "manual"

    if "egp" in text or "egyptian" in text:
        updates["currency"] = "EGP"

    budget_match = re.search(r"(\d+(?:\.\d+)?)\s*(million|m|k|thousand)?", text)
    if budget_match and any(word in text for word in ["budget", "under", "up to", "million", "egp", "price"]):
        number = float(budget_match.group(1))
        unit = budget_match.group(2)

        if unit in ["million", "m"]:
            number = number * 1000000
        elif unit in ["k", "thousand"]:
            number = number * 1000

        updates["budget_max"] = int(number)
        if "currency" not in updates:
            updates["currency"] = existing_variables.get("currency", "EGP") if existing_variables else "EGP"
        intent = "car_search"

    if "actually" in text or "instead" in text or "change" in text or "make it" in text:
        intent = existing_variables.get("intent", intent) if existing_variables else intent

    if "forget" in text and "budget" in text:
        deletions.append("budget_max")

    if "book" in text or "appointment" in text or "visit" in text:
        intent = "booking_request"

    if "refund" in text or "complaint" in text or "angry" in text or "unacceptable" in text:
        intent = "complaint"

    if updates and "lead_stage" in schema:
        updates["lead_stage"] = "qualified"

    missing = []
    for key, config in (schema or {}).items():
        if key == "intent":
            continue
        required = config.get("required", False) if isinstance(config, dict) else False
        if required and key not in updates and key not in (existing_variables or {}):
            missing.append(key)

    if "phone_number" in schema and "phone_number" not in updates and "phone_number" not in (existing_variables or {}):
        missing.append("phone_number")

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
