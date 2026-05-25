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


CAR_SEARCH_INTENTS = {
    "car_search",
    "car_inquiry",
    "car_enquiry",
    "car_purchase",
    "car_buying",
    "buy_car",
    "vehicle_search",
    "vehicle_inquiry",
    "vehicle_enquiry",
    "vehicle_purchase",
    "inventory_question",
    "product_search",
    "product_inquiry",
}


VARIABLE_LABELS_EN = {
    "location": "your location",
    "phone_number": "your phone number",
    "patient_name": "your name",
    "service_needed": "the service you need",
    "appointment_date": "the appointment date",
    "appointment_time": "the appointment time",
    "preferred_contact_method": "your preferred contact method",
    "car_brand": "the car brand",
    "car_condition": "the car condition",
    "budget_max": "your budget",
    "currency": "the currency",
    "transmission": "the transmission",
    "doctor_preference": "your preferred doctor",
    "location_branch": "the preferred branch",
    "insurance_provider": "your insurance provider",
    "preferred_viewing_date": "your preferred viewing date",
}


VARIABLE_LABELS_AR = {
    "location": "المكان",
    "phone_number": "رقم الموبايل",
    "patient_name": "الاسم",
    "service_needed": "الخدمة المطلوبة",
    "appointment_date": "تاريخ الميعاد",
    "appointment_time": "وقت الميعاد",
    "preferred_contact_method": "طريقة التواصل المفضلة",
    "car_brand": "نوع العربية",
    "car_condition": "حالة العربية",
    "budget_max": "الميزانية",
    "currency": "العملة",
    "transmission": "نوع الفتيس",
    "doctor_preference": "الدكتور المفضل",
    "location_branch": "الفرع المفضل",
    "insurance_provider": "التأمين",
    "preferred_viewing_date": "ميعاد المعاينة المفضل",
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
        "معاينة",
        "معاينه",
        "اشوفها",
        "أشوفها",
        "عايز اشوفها",
        "عايز أشوفها",
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


def _english_join(items: list[str]) -> str:
    cleaned = [item for item in items if item]

    if not cleaned:
        return ""

    if len(cleaned) == 1:
        return cleaned[0]

    if len(cleaned) == 2:
        return f"{cleaned[0]} and {cleaned[1]}"

    return ", ".join(cleaned[:-1]) + f", and {cleaned[-1]}"


def _labels_for_missing(missing_variables: list[str], arabic: bool) -> list[str]:
    labels = VARIABLE_LABELS_AR if arabic else VARIABLE_LABELS_EN

    return [
        labels.get(key, key.replace("_", " "))
        for key in missing_variables or []
    ]


def _previous_assistant_question(recent_messages: list[dict]) -> str:
    if not recent_messages:
        return ""

    for message in reversed(recent_messages):
        if message.get("role") != "assistant":
            continue

        content = (message.get("content") or "").strip()

        if not content:
            continue

        if "?" in content or "؟" in content:
            return content

    return ""


def _next_relevant_question(variables: dict, missing_variables: list[str], arabic: bool) -> str:
    variables = variables or {}
    intent = (variables.get("intent") or "").lower()

    if missing_variables:
        labels = _labels_for_missing(missing_variables, arabic)

        if arabic:
            return "لسه محتاج " + _clean_join(labels) + " عشان أكمل."

        return "I still need " + _english_join(labels) + " to continue."

    if intent in CAR_SEARCH_INTENTS:
        if not variables.get("budget_max"):
            if arabic:
                return "ميزانيتك في حدود كام؟"
            return "What budget range should I look within?"

        if not variables.get("transmission"):
            if arabic:
                return "تحبها أوتوماتيك ولا مانيوال؟"
            return "Do you prefer automatic or manual?"

        if not variables.get("car_brand"):
            if arabic:
                return "تحب نوع عربية معين؟"
            return "Do you have a preferred car brand?"

        if arabic:
            return "تحب موديل معين أو سنة معينة؟"

        return "Do you prefer a specific model or year?"

    if intent == "viewing_request":
        if arabic:
            return "تمام، نقدر نكمل طلب المعاينة."
        return "Great, we can continue arranging the viewing."

    if intent == "booking_request":
        needed = []

        if not variables.get("service_needed"):
            needed.append("service_needed")

        if not variables.get("appointment_date"):
            needed.append("appointment_date")

        if not variables.get("appointment_time"):
            needed.append("appointment_time")

        if needed:
            labels = _labels_for_missing(needed, arabic)

            if arabic:
                return "لسه محتاج " + _clean_join(labels) + " عشان أكمل الحجز."

            return "I still need " + _english_join(labels) + " to continue the booking."

        if arabic:
            return "تحب نكمل بيانات الحجز؟"

        return "Would you like to continue with the booking details?"

    return ""


def _contextual_ack_answer(
    message: str,
    variables: dict,
    missing_variables: list,
    recent_messages: list[dict],
    recommended_next_action: str,
) -> str:
    arabic = is_arabic_message(message)
    variables = variables or {}
    missing_variables = missing_variables or []

    next_question = _next_relevant_question(variables, missing_variables, arabic)

    if next_question:
        if arabic:
            return "تمام، " + next_question
        return "Got it — " + next_question

    previous_question = _previous_assistant_question(recent_messages)

    if previous_question:
        if arabic:
            return "تمام، خلينا نكمل على نفس النقطة: " + previous_question

        return "Got it — let’s continue from there: " + previous_question

    if recommended_next_action == "collect_booking_details":
        if arabic:
            return "تمام، خلينا نكمل بيانات الحجز."
        return "Got it — let’s continue with the booking details."

    if recommended_next_action == "collect_viewing_details":
        if arabic:
            return "تمام، خلينا نكمل بيانات المعاينة."
        return "Got it — let’s continue with the viewing details."

    if recommended_next_action == "urgent_human_handoff":
        if arabic:
            return "تمام، الحالة محتاجة متابعة فورية من شخص من الفريق."
        return "Got it — this needs urgent human follow-up."

    if recommended_next_action == "consider_human_handoff":
        if arabic:
            return "تمام، ممكن نحول الموضوع لشخص من الفريق يساعدك."
        return "Got it — we may need a human team member to help with this."

    if variables:
        if arabic:
            return "تمام، مكمل معاك."
        return "Got it — I’ll continue with that."

    if arabic:
        return "تمام."

    return "Got it."


def build_no_llm_answer(
    message: str,
    variables: dict,
    variable_updates: dict,
    missing_variables: list,
    recent_messages: list[dict] | None = None,
    recommended_next_action: str = "continue_conversation",
) -> str:
    arabic = is_arabic_message(message)
    variables = variables or {}
    variable_updates = variable_updates or {}
    missing_variables = missing_variables or []
    recent_messages = recent_messages or []

    if is_short_ack(message):
        return _contextual_ack_answer(
            message=message,
            variables=variables,
            missing_variables=missing_variables,
            recent_messages=recent_messages,
            recommended_next_action=recommended_next_action,
        )

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

            if "preferred_viewing_date" in variable_updates:
                parts.append(f"ميعاد المعاينة {variable_updates.get('preferred_viewing_date')}")

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

            response = "تمام، سجلت: " + _clean_join(parts) + "." if parts else "تمام، حدثت البيانات."

            next_question = _next_relevant_question(variables, missing_variables, arabic=True)

            if next_question:
                response += " " + next_question

            return response

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

        if "preferred_viewing_date" in variable_updates:
            parts.append(f"viewing date: {variable_updates.get('preferred_viewing_date')}")

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

        response = "Got it — I updated " + ", ".join(parts) + "." if parts else "Got it — I updated your details."

        next_question = _next_relevant_question(variables, missing_variables, arabic=False)

        if next_question:
            response += " " + next_question

        return response

    if missing_variables:
        if arabic:
            labels = _labels_for_missing(missing_variables, arabic=True)
            return "تمام، لسه محتاج " + _clean_join(labels) + " عشان أكمل."

        labels = _labels_for_missing(missing_variables, arabic=False)
        return "Got it — I still need " + _english_join(labels) + " to continue."

    next_question = _next_relevant_question(variables, missing_variables, arabic)

    if next_question:
        if arabic:
            return "تمام، " + next_question
        return "Got it — " + next_question

    if arabic:
        return "تمام."

    return "Got it."
