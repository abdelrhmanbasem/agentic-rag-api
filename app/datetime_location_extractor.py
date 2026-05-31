# app/datetime_location_extractor.py
# Deterministic Arabic/English date, time, and location extraction.
#
# Goal: 
# - Silently extract common WhatsApp dates/times ("بكرة 3 في التجمع") into variables.
# - Saves tokens and prevents GPT hallucination on relative Arabic dates.
# - Leaves the actual talking to the LangGraph generator node.

import re
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

def normalize_arabic(text: str) -> str:
    text = text or ""
    replacements = {"أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي", "ة": "ه"}
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text

def normalize_text(text: str) -> str:
    return normalize_arabic((text or "").lower().strip())

def is_iso_date(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        datetime.strptime(value, "%Y-%m-%d")
        return True
    except Exception:
        return False

def extract_time(message: str) -> Optional[str]:
    text = normalize_text(message)

    if "العصر" in text: return "16:00"
    if "بعد المغرب" in text or "المغرب" in text: return "19:00"
    if "بالليل" in text or "ليل" in text: return "20:00"
    if "الصبح" in text or "صباح" in text: return "10:00"

    match = re.search(r"(?:الساعه|الساعة|على|علي)?\s*\b(\d{1,2})(?::(\d{2}))?\s*(م|مساء|pm|ص|صباح|am)?\b", text)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2) or 0)
        suffix = match.group(3)

        if hour > 23: return None

        if suffix in ["م", "مساء", "pm"] and hour < 12: hour += 12
        elif suffix in ["ص", "صباح", "am"]: pass
        elif 1 <= hour <= 7: hour += 12

        if 0 <= hour <= 23 and 0 <= minute <= 59:
            return f"{hour:02d}:{minute:02d}"

    return None

def extract_date(message: str, now: Optional[datetime] = None) -> Optional[str]:
    text = normalize_text(message)
    now = now or datetime.utcnow()

    if "بعد بكره" in text or "بعد بكرة" in message:
        return (now + timedelta(days=2)).date().isoformat()
    if "بكره" in text or "بكرة" in message or "tomorrow" in text:
        return (now + timedelta(days=1)).date().isoformat()
    if "النهارده" in text or "اليوم" in text or "today" in text:
        return now.date().isoformat()

    weekdays = {
        "السبت": 5, "الاحد": 6, "الأحد": 6, "الاتنين": 0, "الاثنين": 0,
        "التلات": 1, "الثلاث": 1, "الثلاثاء": 1, "الاربع": 2, "الأربع": 2,
        "الاربعاء": 2, "الخميس": 3, "الجمعه": 4, "الجمعة": 4,
    }

    today_weekday = now.weekday()
    for word, target_weekday in weekdays.items():
        if normalize_text(word) in text:
            days_ahead = (target_weekday - today_weekday) % 7
            if days_ahead == 0: days_ahead = 7
            return (now + timedelta(days=days_ahead)).date().isoformat()

    match = re.search(r"\b(20\d{2})-(\d{1,2})-(\d{1,2})\b", text)
    if match:
        y, m, d = match.groups()
        try:
            return datetime(int(y), int(m), int(d)).date().isoformat()
        except Exception:
            return None
    return None

def extract_location(message: str) -> Optional[str]:
    original = message or ""
    text = normalize_text(original)

    known_locations = [
        "التجمع الخامس", "التجمع", "مدينة نصر", "مدينه نصر", "مصر الجديدة",
        "مصر الجديده", "المعادي", "الزمالك", "الشيخ زايد", "زايد", "اكتوبر",
        "6 اكتوبر", "الدقي", "المهندسين", "الرحاب", "مدينتي", "الشروق",
        "العبور", "الجيزة", "جيزة", "القاهرة", "القاهره", "اسكندرية", "اسكندريه",
    ]

    for loc in known_locations:
        if normalize_text(loc) in text:
            return loc

    parts = re.split(r"\b(?:في|فى)\b", original)
    if len(parts) >= 2:
        candidate = parts[-1].strip()
        candidate = re.split(r"[،,.!?؟\n]", candidate)[0].strip()
        stop_words = ["بكرة", "بكره", "الساعة", "الساعه", "تمام", "ممكن", "عايز", "عاوز", "معاينة", "معاينه", "احجز", "اشوف"]
        for stop in stop_words:
            candidate = candidate.replace(stop, "").strip()

        blocked = {"معاينة", "معاينه", "المعاينة", "المعاينه", "الحجز", "المعاد", "ميعاد", "موعد", "الميعاد", "الموعد", "العربية", "العربيه", "السيارة"}
        normalized_candidate = normalize_text(candidate)

        if candidate and normalized_candidate not in blocked and 3 <= len(candidate) <= 30:
            return candidate
    return None

def should_override_date(existing_value: Any, new_value: Optional[str]) -> bool:
    if not new_value: return False
    if not existing_value: return True
    if not is_iso_date(existing_value): return True
    return False

def extract_datetime_location_patch(message: str, variables: Dict[str, Any], workflow: str) -> Dict[str, Any]:
    """
    Silently extracts date/time/location into variables. 
    LangGraph will use these updated variables to craft a human-like response.
    """
    variables = variables or {}
    updates: Dict[str, Any] = {}

    date_value = extract_date(message)
    time_value = extract_time(message)
    location_value = extract_location(message)

    if workflow == "car_sales":
        if should_override_date(variables.get("preferred_viewing_date"), date_value):
            updates["preferred_viewing_date"] = date_value
        if time_value and not variables.get("preferred_viewing_time"):
            updates["preferred_viewing_time"] = time_value
        if location_value and not variables.get("location"):
            updates["location"] = location_value
        if updates:
            updates["intent"] = "viewing_request"
            updates["workflow_stage"] = "viewing_requested"

    elif workflow == "service_booking":
        if should_override_date(variables.get("appointment_date"), date_value):
            updates["appointment_date"] = date_value
        if time_value and not variables.get("appointment_time"):
            updates["appointment_time"] = time_value
        if location_value and not variables.get("location"):
            updates["location"] = location_value
        if updates:
            updates["intent"] = "booking_request"
            updates["workflow_stage"] = "booking_requested"

    else:
        if should_override_date(variables.get("preferred_date"), date_value):
            updates["preferred_date"] = date_value
        if time_value and not variables.get("preferred_time"):
            updates["preferred_time"] = time_value
        if location_value and not variables.get("location"):
            updates["location"] = location_value

    return updates
