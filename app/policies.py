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
    "تماما",
    "تم",
    "اوكي",
    "أوكي",
    "حاضر",
    "ماشي",
    "شكرا",
    "شكرًا",
    "تسلم",
    "اه",
    "ايوه",
    "طيب",
    "اشطا",
}


ARABIC_ACKS = {
    "تمام",
    "تماما",
    "تم",
    "اوكي",
    "أوكي",
    "حاضر",
    "ماشي",
    "شكرا",
    "شكرًا",
    "تسلم",
    "اه",
    "ايوه",
    "طيب",
    "اشطا",
}


def is_arabic_message(message: str) -> bool:
    return bool(re.search(r"[\u0600-\u06FF]", message or ""))


def is_short_ack(message: str) -> bool:
    text = (message or "").lower().strip()
    return text in SHORT_ACKS or len(text) <= 3


def is_phone_only(message: str) -> bool:
    text = (message or "").strip()
    cleaned = re.sub(r"[\s\-\+\(\)]", "", text)
    return cleaned.isdigit() and 8 <= len(cleaned) <= 15


def is_simple_variable_update(message: str) -> bool:
    text = (message or "").lower()

    update_markers = [
        "actually",
        "change",
        "make it",
        "instead",
        "my phone",
        "my number",
        "whatsapp",
        "whats app",
        "no calls",
        "don't call",
        "dont call",
        "call me",
        "tomorrow",
        "morning",
        "afternoon",
        "evening",
        "في الواتساب",
        "واتساب",
        "واتس",
        "ماتتصلش",
        "ما تتصلش",
        "متتصلش",
        "كلمني واتساب",
        "ابعتلي واتساب",
        "رقمي",
        "نمرة",
        "نمرتي",
        "بكرة",
        "بكره",
        "النهارده",
        "انهارده",
        "الصبح",
        "بعد الضهر",
        "بعد الظهر",
        "بليل",
        "بالليل",
        "المسا",
        "المساء",
        "لحد",
        "ميزانيتي",
        "ميزانية",
        "الميزانية",
        "مليون",
        "مستعمل",
        "مستعمله",
        "زيرو",
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


def _clean_join(parts: list[str]) -> str:
    return "، ".join([part.strip() for part in parts if part and part.strip()])


def build_no_llm_answer(
    message: str,
    variables: dict,
    variable_updates: dict,
    missing_variables: list,
) -> str:
    arabic = is_arabic_message(message)
    variables = variables or {}
    variable_updates = variable_updates or {}
    missing_variables = missing_variables or []

    if is_short_ack(message):
        if arabic:
            return "تمام."
        return "Got it."

    if is_phone_only(message) or "phone_number" in variable_updates:
        if arabic:
            return "تمام، سجلت رقمك."
        return "Got it — I saved your phone number."

    if variable_updates:
        if arabic:
            parts = []

            if "car_brand" in variable_updates:
                parts.append(f"نوع العربية {variable_updates.get('car_brand')}")

            if "car_condition" in variable_updates:
                condition = variable_updates.get("car_condition")
                if condition == "used":
                    parts.append("مستعملة")
                elif condition == "new":
                    parts.append("زيرو")
                else:
                    parts.append(str(condition))

            if "budget_max" in variable_updates:
                budget = variable_updates.get("budget_max")
                currency = variables.get("currency", "")
                parts.append(f"الميزانية لحد {budget} {currency}".strip())

            if "transmission" in variable_updates:
                transmission = variable_updates.get("transmission")
                if transmission == "automatic":
                    parts.append("الفتيس أوتوماتيك")
                elif transmission == "manual":
                    parts.append("الفتيس مانيوال")
                else:
                    parts.append(f"الفتيس {transmission}")

            if "preferred_contact_method" in variable_updates:
                method = variable_updates.get("preferred_contact_method")
                parts.append(f"التواصل على {method}")

            if "appointment_date" in variable_updates or "appointment_time" in variable_updates:
                date = variables.get("appointment_date", "")
                time = variables.get("appointment_time", "")
                appointment_text = f"الميعاد {date} {time}".strip()
                parts.append(appointment_text)

            if "service_needed" in variable_updates:
                parts.append(f"الخدمة المطلوبة: {variable_updates.get('service_needed')}")

            if "doctor_preference" in variable_updates:
                parts.append(f"الدكتور المفضل: {variable_updates.get('doctor_preference')}")

            if "location_branch" in variable_updates:
                parts.append(f"الفرع: {variable_updates.get('location_branch')}")

            if "insurance_provider" in variable_updates:
                parts.append("التأمين تم تسجيله")

            if "patient_name" in variable_updates:
                parts.append(f"الاسم {variable_updates.get('patient_name')}")

            if "urgency" in variable_updates:
                urgency = variable_updates.get("urgency")
                if urgency == "emergency":
                    parts.append("الحالة طارئة")
                else:
                    parts.append(f"درجة الاستعجال {urgency}")

            if "needs_human" in variable_updates and variable_updates.get("needs_human"):
                parts.append("محتاج متابعة من شخص من الفريق")

            if "lead_stage" in variable_updates:
                pass

            if parts:
                return "تمام، سجلت: " + _clean_join(parts) + "."

            return "تمام، حدثت البيانات."

        parts = []

        if "car_brand" in variable_updates:
            parts.append(f"brand: {variable_updates.get('car_brand')}")

        if "car_condition" in variable_updates:
            parts.append(f"condition: {variable_updates.get('car_condition')}")

        if "budget_max" in variable_updates:
            budget = variable_updates.get("budget_max")
            currency = variables.get("currency", "")
            parts.append(f"budget up to {budget} {currency}".strip())

        if "transmission" in variable_updates:
            parts.append(f"transmission: {variable_updates.get('transmission')}")

        if "preferred_contact_method" in variable_updates:
            parts.append(f"contact by {variable_updates.get('preferred_contact_method')}")

        if "appointment_date" in variable_updates or "appointment_time" in variable_updates:
            date = variables.get("appointment_date", "")
            time = variables.get("appointment_time", "")
            parts.append(f"appointment preference: {date} {time}".strip())

        if "service_needed" in variable_updates:
            parts.append(f"service: {variable_updates.get('service_needed')}")

        if "doctor_preference" in variable_updates:
            parts.append(f"doctor preference: {variable_updates.get('doctor_preference')}")

        if "location_branch" in variable_updates:
            parts.append(f"branch/location: {variable_updates.get('location_branch')}")

        if "insurance_provider" in variable_updates:
            parts.append("insurance mentioned")

        if "patient_name" in variable_updates:
            parts.append(f"name: {variable_updates.get('patient_name')}")

        if "urgency" in variable_updates:
            parts.append(f"urgency: {variable_updates.get('urgency')}")

        if "needs_human" in variable_updates and variable_updates.get("needs_human"):
            parts.append("human follow-up needed")

        if "lead_stage" in variable_updates:
            pass

        if parts:
            return "Got it — I updated " + ", ".join(parts) + "."

        return "Got it — I updated your details."

    if missing_variables:
        if arabic:
            return "تمام. لسه محتاج شوية تفاصيل عشان أكمل."
        return "Got it. I still need a few details to continue."

    if arabic:
        return "تمام."

    return "Got it."
