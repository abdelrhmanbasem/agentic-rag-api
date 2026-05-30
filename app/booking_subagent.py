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

    for i, d in enumerate(arabic_digits):
        result = result.replace(d, str(i))

    for i, d in enumerate(eastern_digits):
        result = result.replace(d, str(i))

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

    # Arabic common phrases.
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
    m = re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", text)
    if m:
        return _normalize_time(int(m.group(1)), int(m.group(2)))

    # Match "2 pm", "10 am".
    m = re.search(r"\b(\d{1,2})\s*(am|pm)\b", text)
    if m:
        hour = int(m.group(1))
        suffix = m.group(2)
        return _normalize_time(hour, 0, pm_hint=suffix == "pm", am_hint=suffix == "am")

    # Match "الساعة 2", "at 2".
    m = re.search(r"(?:الساعة|الساعه|at)\s*(\d{1,2})", text)
    if m:
        hour = int(m.group(1))
        pm_hint = any(x in text for x in ["مساء", "بالليل", "الظهر", "العصر", "المغرب", "pm"])
        am_hint = any(x in text for x in ["صباح", "الصبح", "am"])
        return _normalize_time(hour, 0, pm_hint=pm_hint, am_hint=am_hint)

    # If user says only "10 الصبح" without "الساعة".
    m = re.search(r"\b(\d{1,2})\s*(الصبح|صباح|الظهر|ظهر|العصر|مساء|المغرب|بالليل)\b", text)
    if m:
        hour = int(m.group(1))
        hint = m.group(2)
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
    m = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", text)
    if m:
        return m.group(1)

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

    # English weekdays.
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

    # Arabic June examples.
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

    # English June.
    m = re.search(r"\b(?:june)\s+(\d{1,2})\b", text)
    if m:
        return _parse_day_month(int(m.group(1)), 6)

    m = re.search(r"\b(\d{1,2})\s+(?:june)\b", text)
    if m:
        return _parse_day_month(int(m.group(1)), 6)

    return None


def infer_section_fallback(message: str, variables: Dict[str, Any]) -> Optional[str]:
    """
    Fallback only.

    Preferred behavior:
    - Main brain / diagnostic advisor sets recommended_section.
    - Booking sub-agent uses that section and does not diagnose.

    This fallback is intentionally not called automatically unless we explicitly decide
    to use it later.
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
        return "تمام، أظبطهولك. بس محتاج أفهم المشكلة باختصار عشان أحددلك القسم الصح للكشف."

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
        variables["booking_stage"] = "availability_available_waiting_confirmation"

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
            "reason": "Handled availability result: available.",
        }

    variables["slot_status"] = "unavailable"
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
        "reason": "Handled availability result: unavailable.",
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

    First version:
    - handles booking intent
    - collects branch/date/time
    - requires main brain to provide recommended_section/service_needed
    - requests availability check
    - handles availability_result tool output
    """

    variables = dict(variables or {})

    if tool_result and tool_result.get("type") == "availability_result":
        return handle_availability_result(tool_result, variables)

    is_booking = detect_booking_intent(message, variables)

    if not is_booking:
        return {"handled": False}

    variables["active_subagent"] = BOOKING_AGENT_NAME
    variables["intent"] = "booking_request"
    variables["workflow"] = variables.get("workflow") or "service_booking"

    # Capture phone from WhatsApp user_id if available.
    if user_id and not variables.get("phone_number"):
        variables["phone_number"] = _clean_phone_from_user_id(user_id)
        variables["phone_source"] = "whatsapp_user_id"

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
