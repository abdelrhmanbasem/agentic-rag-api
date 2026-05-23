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

    budget_match = re.search(r"(\d+(?:\.\d+)?)\s*(million|m|k|thousand|مليون|الف|ألف)?", normalized)
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
        match = re.search(pattern, user_message, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    return None


def mock_extract_variables(schema, existing_variables, recent_messages, user_message):
    text = user_message.lower()
    ar_text = normalize_arabic(text)
    updates = {}
    deletions = []
    intent = existing_variables.get("intent", "general_question") if existing_variables else "general_question"

    # Contact preference
    whatsapp_words = [
        "whatsapp",
        "whats app",
        "واتساب",
        "واتس",
        "الواتساب",
        "كلمني واتساب",
        "ابعتلي واتساب",
    ]

    no_call_words = [
        "no calls",
        "don't call",
        "dont call",
        "ماتتصلش",
        "ما تتصلش",
        "متتصلش",
        "بلاش مكالمات",
        "مش عايز مكالمات",
        "مش عايزه مكالمات",
        "ما تكلمنيش مكالمات",
        "ماتكلمنيش مكالمات",
    ]

    if any(word in text for word in whatsapp_words) or any(word in ar_text for word in whatsapp_words):
        if schema_has(schema, "preferred_contact_method"):
            updates["preferred_contact_method"] = "WhatsApp"

    if any(word in text for word in no_call_words) or any(word in ar_text for word in no_call_words):
        if schema_has(schema, "preferred_contact_method"):
            updates["preferred_contact_method"] = "WhatsApp"

    # Cars
    brand_map = {
        "bmw": "BMW",
        "بي ام": "BMW",
        "بي ام دبليو": "BMW",
        "مرسيدس": "Mercedes",
        "mercedes": "Mercedes",
        "audi": "Audi",
        "اودي": "Audi",
        "toyota": "Toyota",
        "تويوتا": "Toyota",
        "hyundai": "Hyundai",
        "هيونداي": "Hyundai",
        "kia": "Kia",
        "كيا": "Kia",
        "nissan": "Nissan",
        "نيسان": "Nissan",
        "ford": "Ford",
        "فورد": "Ford",
        "tesla": "Tesla",
        "تسلا": "Tesla",
    }

    for brand_word, brand_value in brand_map.items():
        if brand_word in text or brand_word in ar_text:
            if schema_has(schema, "car_brand"):
                updates["car_brand"] = brand_value
                intent = "car_search"

    used_words = ["used", "مستعمل", "مستعمله", "استعمال", "كسر زيرو"]
    new_words = ["brand new", "new car", "new one", "زيرو", "جديد", "جديده"]

    if any(word in text for word in used_words) or any(word in ar_text for word in used_words):
        if schema_has(schema, "car_condition"):
            updates["car_condition"] = "used"
            intent = "car_search"

    if any(word in text for word in new_words) or any(word in ar_text for word in new_words):
        if schema_has(schema, "car_condition"):
            updates["car_condition"] = "new"
            intent = "car_search"

    automatic_words = ["automatic", "اوتوماتيك", "اوتو", "اتوماتيك"]
    manual_words = ["manual", "مانيوال", "عادي"]

    if any(word in text for word in automatic_words) or any(word in ar_text for word in automatic_words):
        if schema_has(schema, "transmission"):
            updates["transmission"] = "automatic"

    if any(word in text for word in manual_words) or any(word in ar_text for word in manual_words):
        if schema_has(schema, "transmission"):
            updates["transmission"] = "manual"

    budget = extract_budget(text, existing_variables or {})
    if budget and schema_has(schema, "budget_max"):
        updates["budget_max"] = budget
        if schema_has(schema, "currency"):
            updates["currency"] = "EGP"
        intent = "car_search"

    forget_budget_words = [
        "forget budget",
        "forget the budget",
        "ignore budget",
        "شيل الميزانيه",
        "شيل الميزانية",
        "انس الميزانيه",
        "انس الميزانية",
        "مش مهم الميزانيه",
        "مش مهم الميزانية",
    ]

    if any(word in text for word in forget_budget_words) or any(word in ar_text for word in forget_budget_words):
        deletions.append("budget_max")

    # Clinic/services
    service_map = {
        "cleaning": "cleaning",
        "teeth cleaning": "teeth cleaning",
        "dental": "dental",
        "dentist": "dentist",
        "consultation": "consultation",
        "checkup": "checkup",
        "check-up": "checkup",
        "orthodontics": "orthodontics",
        "xray": "x-ray",
        "x-ray": "x-ray",
        "blood test": "blood test",
        "dermatology": "dermatology",
        "cardiology": "cardiology",
        "تنظيف": "teeth cleaning",
        "تنضيف": "teeth cleaning",
        "اسنان": "dental",
        "سنان": "dental",
        "كشف": "consultation",
        "استشاره": "consultation",
        "استشارة": "consultation",
        "اشعه": "x-ray",
        "أشعه": "x-ray",
        "اشعة": "x-ray",
        "تحليل": "blood test",
        "تحاليل": "blood test",
        "جلديه": "dermatology",
        "جلدية": "dermatology",
        "قلب": "cardiology",
    }

    for service_word, service_value in service_map.items():
        if service_word in text or service_word in ar_text:
            if schema_has(schema, "service_needed"):
                updates["service_needed"] = service_value
                intent = "service_question"

    # Doctor preference
    if schema_has(schema, "doctor_preference"):
        doctor_patterns = [
            r"doctor\s+([a-zA-Z]+)",
            r"dr\.?\s+([a-zA-Z]+)",
            r"دكتور\s+([a-zA-Z\u0600-\u06FF]+)",
            r"دكتوره\s+([a-zA-Z\u0600-\u06FF]+)",
            r"الدكتور\s+([a-zA-Z\u0600-\u06FF]+)",
            r"الدكتوره\s+([a-zA-Z\u0600-\u06FF]+)",
        ]

        for pattern in doctor_patterns:
            doctor_match = re.search(pattern, user_message, re.IGNORECASE)
            if doctor_match:
                updates["doctor_preference"] = doctor_match.group(1).strip()
                break

    # Insurance
    insurance_words = ["insurance", "تأمين", "تامين", "التأمين", "التامين"]
    if any(word in text for word in insurance_words) or any(word in ar_text for word in insurance_words):
        if schema_has(schema, "insurance_provider"):
            updates["insurance_provider"] = "mentioned"
        intent = "insurance_question"

    # Dates
    if schema_has(schema, "appointment_date"):
        if "tomorrow" in text or "بكره" in ar_text or "بكرة" in text:
            updates["appointment_date"] = "tomorrow"
        elif "today" in text or "النهارده" in ar_text or "انهارده" in ar_text:
            updates["appointment_date"] = "today"
        elif "saturday" in text or "السبت" in ar_text:
            updates["appointment_date"] = "Saturday"
        elif "sunday" in text or "الاحد" in ar_text:
            updates["appointment_date"] = "Sunday"
        elif "monday" in text or "الاتنين" in ar_text:
            updates["appointment_date"] = "Monday"
        elif "tuesday" in text or "التلات" in ar_text or "الثلاث" in ar_text:
            updates["appointment_date"] = "Tuesday"
        elif "wednesday" in text or "الاربع" in ar_text:
            updates["appointment_date"] = "Wednesday"
        elif "thursday" in text or "الخميس" in ar_text:
            updates["appointment_date"] = "Thursday"
        elif "friday" in text or "الجمعه" in ar_text or "الجمعة" in text:
            updates["appointment_date"] = "Friday"

    # Times
    if schema_has(schema, "appointment_time"):
        if "morning" in text or "الصبح" in ar_text or "صباح" in ar_text:
            updates["appointment_time"] = "morning"
        elif "afternoon" in text or "بعد الضهر" in ar_text or "بعد الظهر" in text:
            updates["appointment_time"] = "afternoon"
        elif "evening" in text or "بليل" in ar_text or "بالليل" in ar_text or "المسا" in ar_text:
            updates["appointment_time"] = "evening"

    # Branch/location preference
    if schema_has(schema, "location_branch"):
        branch_patterns = [
            r"branch\s+([a-zA-Z\u0600-\u06FF ]+)",
            r"فرع\s+([a-zA-Z\u0600-\u06FF ]+)",
            r"في فرع\s+([a-zA-Z\u0600-\u06FF ]+)",
        ]

        for pattern in branch_patterns:
            branch_match = re.search(pattern, user_message, re.IGNORECASE)
            if branch_match:
                updates["location_branch"] = branch_match.group(1).strip()
                break

    # Booking intent
    booking_words = [
        "book",
        "appointment",
        "visit",
        "احجز",
        "حجز",
        "ميعاد",
        "موعد",
        "اقابل",
        "اروح",
        "زيارة",
        "زياره",
    ]

    if any(word in text for word in booking_words) or any(word in ar_text for word in booking_words):
        if schema_has(schema, "appointment_date") or schema_has(schema, "service_needed"):
            intent = "booking_request"

    # Phone/name
    phone = extract_phone(user_message)
    if phone and schema_has(schema, "phone_number"):
        updates["phone_number"] = phone

    name = extract_name(user_message)
    if name and schema_has(schema, "patient_name"):
        updates["patient_name"] = name

    # Urgency / human handoff
    urgent_words = [
        "urgent",
        "emergency",
        "severe pain",
        "bleeding",
        "chest pain",
        "طوارئ",
        "مستعجل",
        "ضروري",
        "الم شديد",
        "وجع شديد",
        "نزيف",
        "الم في الصدر",
        "مش قادر استحمل",
    ]

    if any(word in text for word in urgent_words) or any(word in ar_text for word in urgent_words):
        if schema_has(schema, "urgency"):
            updates["urgency"] = "emergency"
        if schema_has(schema, "needs_human"):
            updates["needs_human"] = True
        intent = "urgent_medical_issue"

    complaint_words = [
        "refund",
        "complaint",
        "angry",
        "unacceptable",
        "شكوى",
        "زعلان",
        "غاضب",
        "مش مقبول",
        "غير مقبول",
        "عايز فلوسي",
        "استرداد",
    ]

    if any(word in text for word in complaint_words) or any(word in ar_text for word in complaint_words):
        intent = "complaint"

    # Lead stage
    if updates and schema_has(schema, "lead_stage"):
        if intent == "booking_request":
            updates["lead_stage"] = "collecting_details"
        elif intent == "urgent_medical_issue":
            updates["lead_stage"] = "needs_human"
        elif intent == "complaint":
            updates["lead_stage"] = "needs_human"
        else:
            updates["lead_stage"] = "qualified"

    # Missing required variables
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
- Support English, Arabic, and Egyptian Arabic.
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
