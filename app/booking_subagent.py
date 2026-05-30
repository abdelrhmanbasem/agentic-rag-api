# app/booking_subagent.py

import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo


BOOKING_AGENT_NAME = "booking_agent"
DEFAULT_TZ = "Africa/Cairo"


def _norm(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _lower(value: Any) -> str:
    return _norm(value).lower()


def _has_any(text: str, words: List[str]) -> bool:
    text = _lower(text)
    return any(w.lower() in text for w in words)


def _clean_phone_from_user_id(user_id: str) -> str:
    user_id = _norm(user_id).replace("=", "")
    return user_id.replace("@s.whatsapp.net", "").strip()


def _today_egypt() -> datetime:
    return datetime.now(ZoneInfo(DEFAULT_TZ))


def _next_weekday(target_weekday: int, include_today: bool = False) -> str:
    """
    target_weekday: Monday=0, Sunday=6
    """
    today = _today_egypt().date()
    days_ahead = target_weekday - today.weekday()

    if days_ahead < 0 or (days_ahead == 0 and not include_today):
        days_ahead += 7

    return (today + timedelta(days=days_ahead)).isoformat()


def _parse_day_month(day: int, month: int) -> str:
    """
    Converts day/month without year into the next valid upcoming date.
    """
    today = _today_egypt().date()
    year = today.year

    try:
        candidate = datetime(year, month, day, tzinfo=ZoneInfo(DEFAULT_TZ)).date()
    except ValueError:
        return ""

    if candidate < today:
        candidate = datetime(year + 1, month, day, tzinfo=ZoneInfo(DEFAULT_TZ)).date()

    return candidate.isoformat()


def _arabic_digits_to_latin(text: str) -> str:
    arabic_digits = "٠١٢٣٤٥٦٧٨٩"
    eastern_digits = "۰۱۲۳۴۵۶۷۸۹"
    result = str(text)

    for i, digit in enumerate(arabic_digits):
        result = result.replace(digit, str(i))

    for i, digit in enumerate(eastern_digits):
        result = result.replace(digit, str(i))

    return result


def detect_booking_intent(message: str, variables: Dict[str, Any]) -> bool:
    """
    Generic-ish booking intent detector.
    Works for Arabic/English appointment/booking language.
    """
    text = _lower(message)
    variables = variables or {}

    if variables.get("active_subagent") == BOOKING_AGENT_NAME:
        return True

    if variables.get("booking_stage"):
        return True

    booking_words = [
        "احجز",
        "احجزلي",
        "حجز",
        "أحجز",
        "عايز احجز",
        "عايز أروح",
        "عايز اروح",
        "عايز اكشف",
        "عايز أكشف",
        "اظبط",
        "أظبط",
        "ميعاد",
        "معاد",
        "موعد",
        "appointment",
        "book",
        "booking",
        "reserve",
        "schedule",
        "visit",
    ]

    return _has_any(text, booking_words)


def detect_confirmation(message: str) -> bool:
    text = _lower(message)
    confirmation_words = [
        "اه",
        "أه",
        "ايوه",
        "أيوه",
        "ايوا",
        "أيوة",
        "نعم",
        "تمام",
        "ماشي",
        "اوكي",
        "أوكي",
        "ok",
        "okay",
        "yes",
        "confirm",
        "confirmed",
        "ثبت",
        "ثبته",
        "اثبت",
        "أثبت",
        "احجز",
        "احجزه",
        "احجزلي",
    ]
    return _has_any(text, confirmation_words)


def extract_branch(message: str, variables: Dict[str, Any]) -> Optional[str]:
    """
    First version: branch aliases for the service assistant.
    Later this can load branches from assistant playbook/schema.
    """
    existing = variables.get("location_branch") or variables.get("branch")
    if existing:
        return existing

    text = _lower(message)

    branch_map = {
        "new cairo": "New Cairo",
        "القاهرة الجديدة": "New Cairo",
        "القاهره الجديده": "New Cairo",
        "التجمع": "New Cairo",
        "nasr city": "Nasr City",
        "مدينة نصر": "Nasr City",
        "مدينه نصر": "Nasr City",
        "sheikh zayed": "Sheikh Zayed",
        "zayed": "Sheikh Zayed",
        "زايد": "Sheikh Zayed",
        "الشيخ زايد": "Sheikh Zayed",
        "maadi": "Maadi",
        "المعادي": "Maadi",
        "alexandria": "Alexandria",
        "alex": "Alexandria",
        "اسكندرية": "Alexandria",
        "إسكندرية": "Alexandria",
        "الإسكندرية": "Alexandria",
        "الاسكندرية": "Alexandria",
    }

    for key, branch in branch_map.items():
        if key in text:
            return branch

    return None


def _normalize_time(hour: int, minute: int = 0, pm_hint: bool = False, am_hint: bool = False) -> str:
    if pm_hint and hour < 12:
        hour += 12

    if am_hint and hour == 12:
        hour = 0

    return f"{hour:02d}:{minute:02d}"


def extract_time(message: str, variables: Dict[str, Any]) -> Optional[str]:
    existing = variables.get("appointment_time") or variables.get("time")
    if existing:
        return existing

    text = _lower(_arabic_digits_to_latin(message))

    phrase_map = {
        "2 الظهر": "14:00",
        "2 ظهر": "14:00",
        "2 العصر": "14:00",
        "10 الصبح": "10:00",
        "10 صباح": "10:00",
        "12 الظهر": "12:00",
        "12 ظهر": "12:00",
        "4 العصر": "16:00",
        "4 مساء": "16:00",
        "6 المغرب": "18:00",
        "6 مساء": "18:00",
    }

    for phrase, value in phrase_map.items():
        if phrase in text:
            return value

    # Match 14:00, 2:30, etc.
    match = re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", text)
    if match:
        return _normalize_time(int(match.group(1)), int(match.group(2)))

    # Match "2 pm", "10 am".
    match = re.search(r"\b(\d{1,2})\s*(am|pm)\b", text)
    if match:
        hour = int(match.group(1))
        suffix = match.group(2)
        return _normalize_time(hour, 0, pm_hint=suffix == "pm", am_hint=suffix == "am")

    # Match "الساعة 2", "at 2".
    match = re.search(r"(?:الساعة|الساعه|at)\s*(\d{1,2})", text)
    if match:
        hour = int(match.group(1))
        pm_hint = any(x in text for x in ["مساء", "بالليل", "الظهر", "العصر", "المغرب", "pm"])
        am_hint = any(x in text for x in ["صباح", "الصبح", "am"])
        return _normalize_time(hour, 0, pm_hint=pm_hint, am_hint=am_hint)

    # If user says only "10 الصبح" without "الساعة".
    match = re.search(r"\b(\d{1,2})\s*(الصبح|صباح|الظهر|ظهر|العصر|مساء|المغرب|بالليل)\b", text)
    if match:
        hour = int(match.group(1))
        hint = match.group(2)
        pm_hint = hint in ["الظهر", "ظهر", "العصر", "مساء", "المغرب", "بالليل"]
        am_hint = hint in ["الصبح", "صباح"]
        return _normalize_time(hour, 0, pm_hint=pm_hint, am_hint=am_hint)

    return None


def extract_date(message: str, variables: Dict[str, Any]) -> Optional[str]:
    """
    Handles:
    - ISO dates: 2026-06-01
    - relative Arabic: النهارده، بكرة، بعد بكرة
    - weekdays Arabic/English: الأحد، Monday, next Tuesday
    - simple June examples: 1 يونيو, June 1
    """
    existing = variables.get("appointment_date") or variables.get("date")
    if existing:
        return existing

    text = _lower(_arabic_digits_to_latin(message))
    today = _today_egypt().date()

    # ISO date.
    match = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", text)
    if match:
        return match.group(1)

    # Relative dates.
    # Important: check "day after tomorrow" before "tomorrow".
    if any(x in text for x in ["بعد بكرة", "بعد بكره", "day after tomorrow"]):
        return (today + timedelta(days=2)).isoformat()

    if any(x in text for x in ["النهارده", "النهاردة", "اليوم", "today"]):
        return today.isoformat()

    if any(x in text for x in ["بكرة", "بكره", "tomorrow"]):
        return (today + timedelta(days=1)).isoformat()

    # Arabic weekdays. Python weekday: Monday=0, Sunday=6.
    weekday_map = {
        "الاثنين": 0,
        "الإثنين": 0,
        "الاتنين": 0,
        "اتنين": 0,
        "التلات": 1,
        "الثلاثاء": 1,
        "الثلاثا": 1,
        "تلات": 1,
        "الأربعاء": 2,
        "الاربعاء": 2,
        "اربع": 2,
        "الخميس": 3,
        "خميس": 3,
        "الجمعة": 4,
        "الجمعه": 4,
        "جمعه": 4,
        "جمعة": 4,
        "السبت": 5,
        "سبت": 5,
        "الأحد": 6,
        "الاحد": 6,
        "حد": 6,
    }

    for word, weekday in weekday_map.items():
        if word in text:
            include_today = any(x in text for x in ["النهارده", "النهاردة", "اليوم", "today"])
            return _next_weekday(weekday, include_today=include_today)

    english_weekday_map = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6,
    }

    for word, weekday in english_weekday_map.items():
        if word in text:
            include_today = "today" in text
            return _next_weekday(weekday, include_today=include_today)

    june_map = {
        "1 يونيو": 1,
        "واحد يونيو": 1,
        "اول يونيو": 1,
        "أول يونيو": 1,
        "2 يونيو": 2,
        "اتنين يونيو": 2,
        "تاني يونيو": 2,
        "3 يونيو": 3,
        "تلاتة يونيو": 3,
    }

    for phrase, day in june_map.items():
        if phrase in text:
            return _parse_day_month(day, 6)

    match = re.search(r"\b(?:june)\s+(\d{1,2})\b", text)
    if match:
        return _parse_day_month(int(match.group(1)), 6)

    match = re.search(r"\b(\d{1,2})\s+(?:june)\b", text)
    if match:
        return _parse_day_month(int(match.group(1)), 6)

    return None


def infer_section_fallback(message: str, variables: Dict[str, Any]) -> Optional[str]:
    """
    Fallback only.

    Preferred behavior:
    - Main brain / diagnostic advisor sets recommended_section.
    - Booking sub-agent uses that section and does not diagnose.

    This fallback is intentionally not called automatically.
    """
    existing = (
        variables.get("recommended_section")
        or variables.get("service_needed")
        or variables.get("section")
    )
    if existing:
        return existing

    text = _lower(message)
    symptoms = variables.get("symptoms") or []

    if isinstance(symptoms, str):
        symptoms = [symptoms]

    if "overheating" in symptoms or _has_any(text, ["بتسخن", "سخونة", "حرارة", "المؤشر", "ريداتير", "ردياتير"]):
        return "Engine Diagnostics"

    if _has_any(text, ["تكييف", "مش بيبرد", "فريون", "ac"]):
        return "AC Cooling"

    if _has_any(text, ["فرامل", "تيل", "طنابير", "بتصفر"]):
        return "Brakes & Safety"

    if _has_any(text, ["بطارية", "دينامو", "مارش", "مش بتدور", "تك تك"]):
        return "Electrical & Battery"

    if _has_any(text, ["عفشة", "دركسيون", "رعشة", "اهتزاز"]):
        return "Suspension & Steering"

    if _has_any(text, ["زوايا", "كاوتش", "بتحدف", "ترصيص"]):
        return "Tires & Alignment"

    if _has_any(text, ["فتيس", "نتشة", "نقلات", "غيار"]):
        return "Transmission"

    return None


def collect_slot_variables(
    message: str,
    variables: Dict[str, Any],
) -> Dict[str, Any]:
    variables = dict(variables or {})

    branch = extract_branch(message, variables)
    date = extract_date(message, variables)
    time = extract_time(message, variables)

    # IMPORTANT:
    # Booking sub-agent should not diagnose or choose the service section.
    # The main brain / diagnostic advisor should set recommended_section first.
    section = (
        variables.get("recommended_section")
        or variables.get("service_needed")
        or variables.get("section")
    )

    if branch:
        variables["location_branch"] = branch
    if date:
        variables["appointment_date"] = date
    if time:
        variables["appointment_time"] = time
    if section:
        variables["recommended_section"] = section
        variables["service_needed"] = section

    return variables


def missing_slot_fields(variables: Dict[str, Any]) -> List[str]:
    missing = []

    if not variables.get("location_branch"):
        missing.append("location_branch")

    if not variables.get("appointment_date"):
        missing.append("appointment_date")

    if not variables.get("appointment_time"):
        missing.append("appointment_time")

    if not variables.get("recommended_section") and not variables.get("service_needed"):
        missing.append("recommended_section")

    return missing


def ask_for_missing_slot_fields(missing: List[str]) -> str:
    if not missing:
        return ""

    missing_set = set(missing)

    if missing_set >= {"location_branch", "appointment_date", "appointment_time", "recommended_section"}:
        return (
            "تمام، أظبطهولك. بس محتاج أفهم المشكلة باختصار عشان أحددلك القسم الصح، "
            "وكمان تحب أنهي فرع ويوم ووقت مناسبين ليك؟"
        )

    if missing_set >= {"location_branch", "appointment_date", "appointment_time"}:
        return "تمام، أظبطهولك. تحب أنهي فرع، ويوم ووقت مناسبين ليك؟"

    if "recommended_section" in missing_set and len(missing_set) == 1:
        return "تمام، أظبطهولك. بس محتاج أعرف المشكلة أو نوع الكشف المطلوب عشان أحدد القسم الصح."

    parts = []

    if "location_branch" in missing_set:
        parts.append("الفرع")

    if "appointment_date" in missing_set:
        parts.append("اليوم")

    if "appointment_time" in missing_set:
        parts.append("الوقت")

    if parts:
        return "تمام، محتاج بس أعرف " + " و".join(parts) + " المناسبين ليك."

    return "تمام، محتاج كام تفصيلة بسيطة عشان أظبطهولك."


def action_check_availability(variables: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": "check_availability",
        "payload": {
            "branch": variables.get("location_branch"),
            "date": variables.get("appointment_date"),
            "time": variables.get("appointment_time"),
            "section": variables.get("recommended_section") or variables.get("service_needed"),
        },
    }


def normalize_unavailable_reason(reason: str) -> str:
    reason_l = _lower(reason)

    if not reason_l:
        return "المعاد ده مش متاح حاليًا"

    if "exact slot not listed" in reason_l:
        return "الوقت ده مش مفتوح في جدول المواعيد"

    if "slot is full" in reason_l or "full" in reason_l:
        return "المعاد ده اتحجز بالكامل"

    if "equipment" in reason_l:
        return "القسم مش متاح في الوقت ده بسبب صيانة أو تجهيزات"

    if "manager blocked" in reason_l or "blocked" in reason_l:
        return "المعاد ده مقفول من إدارة الفرع حاليًا"

    return reason


def handle_availability_result(
    tool_result: Dict[str, Any],
    variables: Dict[str, Any],
) -> Dict[str, Any]:
    variables = dict(variables or {})
    available = bool(tool_result.get("available"))

    requested = tool_result.get("requested") or {}

    if requested:
        variables["location_branch"] = requested.get("branch") or variables.get("location_branch")
        variables["appointment_date"] = requested.get("date") or variables.get("appointment_date")
        variables["appointment_time"] = requested.get("time") or variables.get("appointment_time")
        variables["recommended_section"] = requested.get("section") or variables.get("recommended_section")
        variables["service_needed"] = variables.get("recommended_section") or variables.get("service_needed")

    variables["active_subagent"] = BOOKING_AGENT_NAME

    if available:
        variables["slot_status"] = "available"
        variables["booking_status"] = "slot_available"
        variables["booking_stage"] = "availability_available_waiting_confirmation"
        variables["booking_confirmation_requested"] = True

        branch = variables.get("location_branch", "الفرع")
        date = variables.get("appointment_date", "اليوم")
        time = variables.get("appointment_time", "الوقت")

        answer = (
            f"تمام، الميعاد متاح في {branch} يوم {date} الساعة {time}. "
            "تحب أثبتهولك؟"
        )

        return {
            "handled": True,
            "answer": answer,
            "variables": variables,
            "active_subagent": BOOKING_AGENT_NAME,
            "booking_stage": variables["booking_stage"],
            "action_required": None,
            "missing_variables": [],
            "recommended_next_action": "await_slot_confirmation",
            "reason": "Handled availability result: available.",
        }

    variables["slot_status"] = "unavailable"
    variables["booking_status"] = "slot_unavailable"
    variables["booking_stage"] = "availability_unavailable_waiting_new_slot"
    variables["unavailable_reason"] = tool_result.get("reason") or tool_result.get("unavailable_reason")
    variables["nearest_slots"] = tool_result.get("nearest_slots") or []
    variables["nearest_slots_text"] = tool_result.get("nearest_slots_text") or ""

    reason_text = normalize_unavailable_reason(variables.get("unavailable_reason", ""))
    nearest_text = _norm(variables.get("nearest_slots_text"))

    if nearest_text:
        answer = (
            f"تمام، الميعاد ده مش متاح للأسف لأن {reason_text}. "
            "بس لقيتلك أقرب اختيارات متاحة:\n\n"
            f"{nearest_text}\n\n"
            "تحب أثبتلك واحد منهم؟"
        )
    else:
        answer = (
            f"تمام، الميعاد ده مش متاح للأسف لأن {reason_text}. "
            "تحب نجرب وقت تاني في نفس الفرع، ولا أدوّرلك في فرع قريب؟"
        )

    return {
        "handled": True,
        "answer": answer,
        "variables": variables,
        "active_subagent": BOOKING_AGENT_NAME,
        "booking_stage": variables["booking_stage"],
        "action_required": None,
        "missing_variables": [],
        "recommended_next_action": "choose_new_slot",
        "reason": "Handled availability result: unavailable.",
    }


def _extract_plate_digits(message: str) -> Optional[str]:
    text = _arabic_digits_to_latin(message)
    matches = re.findall(r"\b\d{3,6}\b", text)

    if matches:
        return matches[-1]

    return None


def _extract_customer_name(message: str) -> Optional[str]:
    """
    Simple first version.
    Later this should become LLM extraction for smoother handling.
    """
    text = _norm(message)

    patterns = [
        r"(?:اسمي|الاسم|انا اسمي|أنا اسمي)\s+([^،,\n]+)",
        r"(?:name is|my name is)\s+([^،,\n]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            name = match.group(1).strip()
            if 2 <= len(name) <= 60:
                return name

    return None


def _detect_phone_confirmed(message: str) -> bool:
    text = _lower(message)

    return _has_any(
        text,
        [
            "نفس الرقم",
            "اه نفس الرقم",
            "أه نفس الرقم",
            "ايوه نفس الرقم",
            "أيوه نفس الرقم",
            "ايوا نفس الرقم",
            "نعم نفس الرقم",
            "same number",
            "yes same",
            "use this number",
            "this number",
        ],
    )


def collect_booking_confirmation_details(
    message: str,
    variables: Dict[str, Any],
) -> Dict[str, Any]:
    variables = dict(variables or {})

    name = _extract_customer_name(message)
    if name and not variables.get("customer_full_name"):
        variables["customer_full_name"] = name

    plate = _extract_plate_digits(message)
    if plate and not variables.get("plate_digits"):
        variables["plate_digits"] = plate

    if _detect_phone_confirmed(message):
        variables["phone_confirmed"] = True

    return variables


def missing_confirmation_fields(variables: Dict[str, Any]) -> List[str]:
    missing = []

    if not variables.get("customer_full_name"):
        missing.append("customer_full_name")

    if not variables.get("plate_digits"):
        missing.append("plate_digits")

    if not variables.get("phone_confirmed"):
        missing.append("phone_confirmed")

    return missing


def ask_for_missing_confirmation_fields(missing: List[str]) -> str:
    missing_set = set(missing)

    if missing_set == {"customer_full_name", "plate_digits", "phone_confirmed"}:
        return (
            "تمام، عشان أثبت الحجز محتاج اسم حضرتك بالكامل، "
            "ونمر العربية، وتأكيد إننا نستخدم نفس رقم الواتساب للتواصل."
        )

    parts = []

    if "customer_full_name" in missing_set:
        parts.append("اسم حضرتك بالكامل")

    if "plate_digits" in missing_set:
        parts.append("نمر العربية")

    if "phone_confirmed" in missing_set:
        parts.append("تأكيد نستخدم نفس رقم الواتساب للتواصل")

    return "تمام، باقي بس " + " و".join(parts) + "."


def action_create_booking(variables: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": "create_booking",
        "payload": {
            "branch": variables.get("location_branch"),
            "date": variables.get("appointment_date"),
            "time": variables.get("appointment_time"),
            "section": variables.get("recommended_section") or variables.get("service_needed"),
            "customer_full_name": variables.get("customer_full_name"),
            "plate_digits": variables.get("plate_digits"),
            "phone_number": variables.get("phone_number"),
        },
    }


def handle_booking_result(
    tool_result: Dict[str, Any],
    variables: Dict[str, Any],
) -> Dict[str, Any]:
    variables = dict(variables or {})
    variables["active_subagent"] = BOOKING_AGENT_NAME

    success = bool(tool_result.get("success"))

    if success:
        visit_id = tool_result.get("visit_id") or tool_result.get("booking_id") or ""

        variables["booking_status"] = "confirmed"
        variables["booking_stage"] = "booking_confirmed"
        variables["visit_id"] = visit_id

        branch = variables.get("location_branch", "الفرع")
        date = variables.get("appointment_date", "اليوم")
        time = variables.get("appointment_time", "الوقت")
        section = (
            variables.get("customer_facing_section")
            or variables.get("recommended_section")
            or "القسم"
        )

        if visit_id:
            answer = (
                f"تمام، كده الحجز اتأكد. رقم الزيارة {visit_id}. "
                f"ميعادك في {branch} يوم {date} الساعة {time} في {section}. "
                "خلي رقم الزيارة معاك وتوريه للمهندس أول ما توصل."
            )
        else:
            answer = (
                f"تمام، كده الحجز اتأكد. "
                f"ميعادك في {branch} يوم {date} الساعة {time} في {section}. "
                "هنأكد التفاصيل معاك على نفس الرقم."
            )

        return {
            "handled": True,
            "answer": answer,
            "variables": variables,
            "active_subagent": BOOKING_AGENT_NAME,
            "booking_stage": variables["booking_stage"],
            "action_required": None,
            "missing_variables": [],
            "recommended_next_action": "booking_confirmed",
            "reason": "Handled booking result: success.",
        }

    variables["booking_status"] = "failed"
    variables["booking_stage"] = "booking_failed"

    reason = tool_result.get("reason") or tool_result.get("error") or "حصلت مشكلة أثناء تثبيت الحجز"

    answer = (
        f"معلش، مقدرتش أثبت الحجز دلوقتي بسبب: {reason}. "
        "تحب أجربلك نفس الميعاد تاني، ولا نختار ميعاد مختلف؟"
    )

    return {
        "handled": True,
        "answer": answer,
        "variables": variables,
        "active_subagent": BOOKING_AGENT_NAME,
        "booking_stage": variables["booking_stage"],
        "action_required": None,
        "missing_variables": [],
        "recommended_next_action": "booking_retry_or_new_slot",
        "reason": "Handled booking result: failed.",
    }


def run_booking_subagent(
    assistant: Dict[str, Any],
    assistant_id: str,
    user_id: str,
    conversation_id: str,
    message: str,
    variables: Dict[str, Any],
    recent_messages: List[Dict[str, Any]],
    summary: str,
    schema: Dict[str, Any],
    tool_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Internal booking sub-agent.

    Handles:
    - booking intent
    - branch/date/time collection
    - check_availability action
    - availability_result tool output
    - available-slot confirmation
    - customer name / plate / phone confirmation
    - create_booking action
    - booking_result tool output
    """

    variables = dict(variables or {})

    if tool_result and tool_result.get("type") == "availability_result":
        return handle_availability_result(tool_result, variables)

    if tool_result and tool_result.get("type") == "booking_result":
        return handle_booking_result(tool_result, variables)

    is_booking = detect_booking_intent(message, variables)

    if not is_booking:
        return {"handled": False}

    variables["active_subagent"] = BOOKING_AGENT_NAME
    variables["intent"] = "booking_request"
    variables["workflow"] = variables.get("workflow") or "service_booking"

    if user_id and not variables.get("phone_number"):
        variables["phone_number"] = _clean_phone_from_user_id(user_id)
        variables["phone_source"] = "whatsapp_user_id"

    booking_stage = variables.get("booking_stage")

    if booking_stage == "availability_available_waiting_confirmation":
        if detect_confirmation(message):
            variables["booking_stage"] = "booking_collecting_customer_details"
            variables = collect_booking_confirmation_details(message, variables)

            missing_details = missing_confirmation_fields(variables)

            if missing_details:
                return {
                    "handled": True,
                    "answer": ask_for_missing_confirmation_fields(missing_details),
                    "variables": variables,
                    "active_subagent": BOOKING_AGENT_NAME,
                    "booking_stage": variables["booking_stage"],
                    "action_required": None,
                    "missing_variables": missing_details,
                    "recommended_next_action": "collect_booking_confirmation_details",
                    "reason": "User confirmed available slot; collecting booking details.",
                }

            variables["booking_stage"] = "booking_create_ready"

            return {
                "handled": True,
                "answer": "تمام، هثبتلك الحجز دلوقتي.",
                "variables": variables,
                "active_subagent": BOOKING_AGENT_NAME,
                "booking_stage": variables["booking_stage"],
                "action_required": action_create_booking(variables),
                "missing_variables": [],
                "recommended_next_action": "create_booking",
                "reason": "Booking confirmation details complete; create booking required.",
            }

        return {
            "handled": True,
            "answer": "تمام، تحب أثبتلك الميعاد ده ولا تختار وقت تاني؟",
            "variables": variables,
            "active_subagent": BOOKING_AGENT_NAME,
            "booking_stage": booking_stage,
            "action_required": None,
            "missing_variables": [],
            "recommended_next_action": "await_slot_confirmation",
            "reason": "Waiting for user to confirm available slot.",
        }

    if booking_stage == "booking_collecting_customer_details":
        variables = collect_booking_confirmation_details(message, variables)
        missing_details = missing_confirmation_fields(variables)

        if missing_details:
            return {
                "handled": True,
                "answer": ask_for_missing_confirmation_fields(missing_details),
                "variables": variables,
                "active_subagent": BOOKING_AGENT_NAME,
                "booking_stage": variables["booking_stage"],
                "action_required": None,
                "missing_variables": missing_details,
                "recommended_next_action": "collect_booking_confirmation_details",
                "reason": "Still missing booking confirmation details.",
            }

        variables["booking_stage"] = "booking_create_ready"

        return {
            "handled": True,
            "answer": "تمام، هثبتلك الحجز دلوقتي.",
            "variables": variables,
            "active_subagent": BOOKING_AGENT_NAME,
            "booking_stage": variables["booking_stage"],
            "action_required": action_create_booking(variables),
            "missing_variables": [],
            "recommended_next_action": "create_booking",
            "reason": "Booking confirmation details complete; create booking required.",
        }

    variables = collect_slot_variables(message, variables)
    missing = missing_slot_fields(variables)

    if missing:
        variables["booking_stage"] = "booking_collecting_slot"
        answer = ask_for_missing_slot_fields(missing)

        return {
            "handled": True,
            "answer": answer,
            "variables": variables,
            "active_subagent": BOOKING_AGENT_NAME,
            "booking_stage": variables["booking_stage"],
            "action_required": None,
            "missing_variables": missing,
            "recommended_next_action": "collect_booking_slot_fields",
            "reason": "Booking intent detected; missing slot fields.",
        }

    variables["booking_stage"] = "availability_check_ready"
    variables["slot_status"] = "pending_check"

    return {
        "handled": True,
        "answer": "تمام، هراجعلك الميعاد ده لحظة.",
        "variables": variables,
        "active_subagent": BOOKING_AGENT_NAME,
        "booking_stage": variables["booking_stage"],
        "action_required": action_check_availability(variables),
        "missing_variables": [],
        "recommended_next_action": "check_availability",
        "reason": "Booking slot fields complete; availability check required.",
    }
