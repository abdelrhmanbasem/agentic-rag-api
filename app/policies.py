import re


SHORT_ACKS = {
    "hi",
    "hello",
    "hey",
    "thanks",
    "thank you",
    "ok",
    "okay",
    "yes",
    "no",
    "تمام",
    "شكرا",
}


def is_short_ack(message: str) -> bool:
    text = message.lower().strip()
    return text in SHORT_ACKS or len(text) <= 3


def is_phone_only(message: str) -> bool:
    text = message.strip()
    cleaned = re.sub(r"[\s\-\+\(\)]", "", text)
    return cleaned.isdigit() and 8 <= len(cleaned) <= 15


def is_simple_variable_update(message: str) -> bool:
    text = message.lower()

    update_markers = [
        "actually",
        "change",
        "make it",
        "instead",
        "my phone",
        "my number",
        "whatsapp",
        "no calls",
        "don't call",
        "dont call",
        "call me",
        "tomorrow",
        "morning",
        "afternoon",
        "evening",
    ]

    return any(marker in text for marker in update_markers)


def should_skip_generation(message: str, variable_updates: dict, intent: str) -> bool:
    if is_short_ack(message):
        return True

    if is_phone_only(message):
        return True

    if variable_updates and is_simple_variable_update(message):
        return True

    if intent in ["provide_contact", "variable_update"]:
        return True

    return False


def build_no_llm_answer(message: str, variables: dict, variable_updates: dict, missing_variables: list) -> str:
    text = message.lower().strip()

    if is_short_ack(message):
        return "Got it."

    if is_phone_only(message) or "phone_number" in variable_updates:
        return "Got it — I saved your phone number."

    if "preferred_contact_method" in variable_updates:
        method = variable_updates.get("preferred_contact_method")
        return f"Got it — I’ll use {method} as your preferred contact method."

    if "budget_max" in variable_updates:
        budget = variable_updates.get("budget_max")
        currency = variables.get("currency", "")
        return f"Got it — I updated your budget to {budget} {currency}."

    if "appointment_date" in variable_updates or "appointment_time" in variable_updates:
        date = variables.get("appointment_date", "")
        time = variables.get("appointment_time", "")
        return f"Got it — I updated your appointment preference to {date} {time}."

    if variable_updates:
        return "Got it — I updated your details."

    if missing_variables:
        return "Got it. I still need a few details to continue."

    return "Got it."
