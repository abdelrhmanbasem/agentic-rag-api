# app/variables.py
# Combined Level 1 + "breath-taking smart" upgrade.
# Replace your existing app/variables.py with this file.

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
    "objection",
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

    "price_objection": "objection",
    "budget_objection": "objection",
    "too_expensive": "objection",
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


def extract_budget(text, existing_variables=None):
    normalized = normalize_arabic((text or "").lower())

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
        number *= 1000000
    elif unit in ["k", "thousand", "الف", "ألف"]:
        number *= 1000

    return int(number)


def extract_phone(user_message):
    phone_match = re.search(r"(\+?\d[\d\s\-]{7,}\d)", user_message or "")
    return phone_match.group(1).strip() if phone_match else None


def extract_name(user_message):
    patterns = [
        r"my name is\s+([a-zA-Z\u0600-\u06FF ]+)",
        r"اسمي\s+([a-zA-Z\u0600-\u06FF ]+)",
        r"انا اسمي\s+([a-zA-Z\u0600-\u06FF ]+)",
        r"أنا اسمي\s+([a-zA-Z\u0600-\u06FF ]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, user_message or "", re.IGNORECASE)
        if match:
            return match.group(1).strip()

    return None


def extract_location(user_message):
    text = user_message or ""
    ar_text = normalize_arabic(text.lower())

    patterns = [
        r"انا في\s+([a-zA-Z\u0600-\u06FF0-9 ]+)",
        r"أنا في\s+([a-zA-Z\u0600-\u06FF0-9 ]+)",
        r"في\s+([a-zA-Z\u0600-\u06FF0-9 ]+)",
        r"location is\s+([a-zA-Z\u0600-\u06FF0-9 ]+)",
        r"i am in\s+([a-zA-Z\u0600-\u06FF0-9 ]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            value = re.sub(
                r"(بكره|بكرة|tomorrow|today|الساعة|الساعه|at).*$",
                "",
                value,
                flags=re.IGNORECASE,
            ).strip()
            if value:
                return value

    known_locations = [
        "التجمع",
        "القاهرة الجديدة",
        "مدينتي",
        "الرحاب",
        "مدينة نصر",
        "مصر الجديدة",
        "المعادي",
        "6 اكتوبر",
        "اكتوبر",
        "زايد",
        "الشيخ زايد",
        "الدقي",
        "المهندسين",
        "الزمالك",
        "alexandria",
        "cairo",
        "new cairo",
        "maadi",
        "zayed",
    ]

    for loc in known_locations:
        if loc in ar_text or loc in text.lower():
            return loc

    return None


def extract_viewing_time(user_message):
    text = user_message or ""
    lower = normalize_arabic(text.lower())

    hour_match = re.search(
        r"(?:الساعة|الساعه|at)?\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm|الصبح|صباح|العصر|الضهر|الظهر|المغرب|بالليل|بليل|المسا|المساء)?",
        lower,
    )
    if hour_match:
        hour = hour_match.group(1)
        minute = hour_match.group(2)
        period = hour_match.group(3)

        value = f"{hour}:{minute}" if minute else hour

        if period:
            value = f"{value} {period}"

        return value.strip()

    if "الصبح" in lower or "صباح" in lower:
        return "morning"

    if "بعد الضهر" in lower or "بعد الظهر" in lower or "العصر" in lower:
        return "afternoon"

    if "بليل" in lower or "بالليل" in lower or "المسا" in lower or "المساء" in lower:
        return "evening"

    return None


def mock_extract_variables(schema, existing_variables, recent_messages, user_message):
    text = (user_message or "").lower()
    ar_text = normalize_arabic(text)
    updates = {}
    deletions = []

    existing_variables = existing_variables or {}
    intent = normalize_intent(existing_variables.get("intent", "general_question"))

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
            intent = "variable_update"

    if any(word in text for word in no_call_words) or any(word in ar_text for word in no_call_words):
        if schema_has(schema, "preferred_contact_method"):
            updates["preferred_contact_method"] = "WhatsApp"
            intent = "variable_update"

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

    used_words = [
        "used",
        "مستعمل",
        "مستعمله",
        "مستعملة",
        "استعمال",
        "كسر زيرو",
    ]

    new_words = [
        "brand new",
        "new car",
        "new one",
        "زيرو",
        "جديد",
        "جديده",
        "جديدة",
    ]

    if any(word in text for word in used_words) or any(word in ar_text for word in used_words):
        if schema_has(schema, "car_condition"):
            updates["car_condition"] = "used"
            intent = "car_search"

    if any(word in text for word in new_words) or any(word in ar_text for word in new_words):
        if schema_has(schema, "car_condition"):
            updates["car_condition"] = "new"
            intent = "car_search"

    automatic_words = [
        "automatic",
        "اوتوماتيك",
        "أوتوماتيك",
        "اوتو",
        "اتوماتيك",
    ]

    manual_words = [
        "manual",
        "مانيوال",
        "عادي",
    ]

    if any(word in text for word in automatic_words) or any(word in ar_text for word in automatic_words):
        if schema_has(schema, "transmission"):
            updates["transmission"] = "automatic"
            intent = "car_search"

    if any(word in text for word in manual_words) or any(word in ar_text for word in manual_words):
        if schema_has(schema, "transmission"):
            updates["transmission"] = "manual"
            intent = "car_search"

    budget = extract_budget(text, existing_variables)
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
        intent = "variable_update"

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
            doctor_match = re.search(pattern, user_message or "", re.IGNORECASE)
            if doctor_match:
                updates["doctor_preference"] = doctor_match.group(1).strip()
                break

    insurance_words = [
        "insurance",
        "تأمين",
        "تامين",
        "التأمين",
        "التامين",
    ]

    if any(word in text for word in insurance_words) or any(word in ar_text for word in insurance_words):
        if schema_has(schema, "insurance_provider"):
            updates["insurance_provider"] = "mentioned"
        intent = "insurance_question"

    date_value = None

    if "tomorrow" in text or "بكره" in ar_text or "بكرة" in text:
        date_value = "tomorrow"
    elif "today" in text or "النهارده" in ar_text or "انهارده" in ar_text:
        date_value = "today"
    elif "saturday" in text or "السبت" in ar_text:
        date_value = "Saturday"
    elif "sunday" in text or "الاحد" in ar_text:
        date_value = "Sunday"
    elif "monday" in text or "الاتنين" in ar_text:
        date_value = "Monday"
    elif "tuesday" in text or "التلات" in ar_text or "الثلاث" in ar_text:
        date_value = "Tuesday"
    elif "wednesday" in text or "الاربع" in ar_text:
        date_value = "Wednesday"
    elif "thursday" in text or "الخميس" in ar_text:
        date_value = "Thursday"
    elif "friday" in text or "الجمعه" in ar_text or "الجمعة" in text:
        date_value = "Friday"

    if date_value and schema_has(schema, "appointment_date"):
        updates["appointment_date"] = date_value

    if date_value and schema_has(schema, "preferred_viewing_date"):
        updates["preferred_viewing_date"] = date_value

    time_value = extract_viewing_time(user_message)

    if time_value and schema_has(schema, "appointment_time"):
        updates["appointment_time"] = time_value

    if time_value and schema_has(schema, "preferred_viewing_time"):
        updates["preferred_viewing_time"] = time_value

    location = extract_location(user_message)

    if location and schema_has(schema, "location"):
        updates["location"] = location

    if schema_has(schema, "location_branch"):
        branch_patterns = [
            r"branch\s+([a-zA-Z\u0600-\u06FF ]+)",
            r"فرع\s+([a-zA-Z\u0600-\u06FF ]+)",
            r"في فرع\s+([a-zA-Z\u0600-\u06FF ]+)",
        ]

        for pattern in branch_patterns:
            branch_match = re.search(pattern, user_message or "", re.IGNORECASE)
            if branch_match:
                updates["location_branch"] = branch_match.group(1).strip()
                break

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
        if schema_has(schema, "preferred_viewing_date") or schema_has(schema, "matched_car_model") or schema_has(schema, "car_brand"):
            intent = "viewing_request"
        elif schema_has(schema, "appointment_date") or schema_has(schema, "service_needed"):
            intent = "booking_request"

    viewing_words = [
        "see it",
        "view it",
        "book viewing",
        "book a viewing",
        "schedule viewing",
        "test drive",
        "visit to see",
        "viewing",
        "عايز اشوفها",
        "عايز أشوفها",
        "عايز اشوفه",
        "اشوفها",
        "أشوفها",
        "اشوفه",
        "احجز معاينة",
        "معاينة",
        "معاينه",
        "اتفرج عليها",
        "اجربها",
        "تجربة قيادة",
        "تجربه قياده",
        "احجزها",
    ]

    if any(word in text for word in viewing_words) or any(word in ar_text for word in viewing_words):
        intent = "viewing_request"

        if schema_has(schema, "lead_stage"):
            updates["lead_stage"] = "viewing_requested"

    phone = extract_phone(user_message)

    if phone and schema_has(schema, "phone_number"):
        updates["phone_number"] = phone

        if intent == "general_question":
            intent = "provide_contact"

    name = extract_name(user_message)

    if name and schema_has(schema, "patient_name"):
        updates["patient_name"] = name

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

    objection_words = [
        "expensive",
        "too expensive",
        "discount",
        "installment",
        "installments",
        "غالي",
        "غالية",
        "السعر عالي",
        "كتير",
        "خصم",
        "تقسيط",
        "مش مناسب",
    ]

    if any(word in text for word in objection_words) or any(word in ar_text for word in objection_words):
        intent = "objection"

    intent = normalize_intent(intent)

    if updates and schema_has(schema, "lead_stage"):
        if intent == "booking_request":
            updates["lead_stage"] = "collecting_details"
        elif intent == "viewing_request":
            updates["lead_stage"] = "viewing_requested"
        elif intent == "urgent_medical_issue":
            updates["lead_stage"] = "needs_human"
        elif intent in ["complaint", "objection"]:
            updates["lead_stage"] = "needs_attention"
        else:
            updates["lead_stage"] = "qualified"

    missing = []

    for key, config in (schema or {}).items():
        if key == "intent":
            continue

        required = config.get("required", False) if isinstance(config, dict) else False

        if required and key not in updates and key not in existing_variables:
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

Canonical intent names:
- general_question
- car_search
- viewing_request
- booking_request
- service_question
- insurance_question
- complaint
- human_handoff
- urgent_medical_issue
- variable_update
- provide_contact
- objection

Intent rules:
- If the user wants to buy, search for, ask about, or inquire about a car/vehicle/product availability, use intent = car_search.
- Do NOT use car_inquiry, car_purchase, vehicle_search, or product_inquiry. Map them to car_search.
- If the user wants to see a car/product, schedule a viewing, test drive, or book a viewing, use intent = viewing_request.
- If the user wants to book a clinic/service appointment/reservation, use intent = booking_request.
- If the user gives only a phone/contact detail, use intent = provide_contact.
- If the user changes/corrects previous info, use intent = variable_update unless another stronger intent applies.
- If urgent medical/emergency language appears, use intent = urgent_medical_issue.
- If complaint/refund/angry escalation language appears, use intent = complaint.
- If the user says price is high, asks for discount, installments, or says it is expensive, use intent = objection.

Extraction rules:
- Extract only variables that exist in the schema.
- If the user changes their mind, update the old value.
- If the user contradicts previous info, prefer the latest explicit statement.
- If the user says to forget/remove something, add that key to deletions.
- Do not guess values.
- Support English, Arabic, Egyptian Arabic, and common Franco Arabic.
- Extract location from phrases like "أنا في التجمع", "في المعادي", "I am in New Cairo".
- Extract preferred_viewing_time from phrases like "الساعة 3 العصر", "3 pm", "بكرة ٣".
- Respect intent-specific required variables if present, but do not invent missing values.
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
        max_tokens=700,
    )

    intent = normalize_intent(result.get("intent", "general_question"))

    return {
        "intent": intent,
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
