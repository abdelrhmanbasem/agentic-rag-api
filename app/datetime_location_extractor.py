# app/datetime_location_extractor.py
# Deterministic Arabic/English date, time, and location extraction.
#
# Goal:
# - Make booking/viewing flows smoother.
# - Avoid GPT extraction for common WhatsApp messages:
#   "بكرة 3 في التجمع", "السبت العصر", "مدينة نصر", "بعد المغرب"

import re
from datetime import datetime, timedelta
from typing import Dict, Any, Optional


def normalize_arabic(text: str) -> str:
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


def normalize_text(text: str) -> str:
    return normalize_arabic((text or "").lower().strip())


def extract_time(message: str) -> Optional[str]:
    text = normalize_text(message)

    # 3, الساعة 3, بكرة 3
    match = re.search(r"(?:الساعه|الساعة|على|علي)?\s*(\d{1,2})(?::(\d{2}))?\s*(م|مساء|pm|ص|صباح|am)?", text)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2) or 0)
        suffix = match.group(3)

        # Heuristic for Arabic sales chats: bare 3 usually means afternoon.
        if suffix in ["م", "مساء", "pm"] and hour < 12:
            hour += 12
        elif suffix in ["ص", "صباح", "am"]:
            pass
        elif 1 <= hour <= 7:
            hour += 12

        if 0 <= hour <= 23 and 0 <= minute <= 59:
            return f"{hour:02d}:{minute:02d}"

    if "العصر" in text:
        return "16:00"

    if "بعد المغرب" in text or "المغرب" in text:
        return "19:00"

    if "بالليل" in text or "ليل" in text:
        return "20:00"

    if "الصبح" in text or "صباح" in text:
        return "10:00"

    return None


def extract_date(message: str, now: Optional[datetime] = None) -> Optional[str]:
    text = normalize_text(message)
    now = now or datetime.utcnow()

    if "النهارده" in text or "اليوم" in text or "today" in text:
        return now.date().isoformat()

    if "بكره" in text or "بكرة" in message or "tomorrow" in text:
        return (now + timedelta(days=1)).date().isoformat()

    if "بعد بكره" in text or "بعد بكرة" in message:
        return (now + timedelta(days=2)).date().isoformat()

    weekdays = {
        "السبت": 5,
        "الاحد": 6,
        "الأحد": 6,
        "الاتنين": 0,
        "الاثنين": 0,
        "التلات": 1,
        "الثلاث": 1,
        "الثلاثاء": 1,
        "الاربع": 2,
        "الأربع": 2,
        "الاربعاء": 2,
        "الخميس": 3,
        "الجمعه": 4,
        "الجمعة": 4,
    }

    today_weekday = now.weekday()

    for word, target_weekday in weekdays.items():
        if normalize_text(word) in text:
            days_ahead = (target_weekday - today_weekday) % 7
            if days_ahead == 0:
                days_ahead = 7
            return (now + timedelta(days=days_ahead)).date().isoformat()

    # ISO-ish date, e.g. 2026-05-30
    match = re.search(r"\b(20\d{2})-(\d{1,2})-(\d{1,2})\b", text)
    if match:
        y, m, d = match.groups()
        try:
            return datetime(int(y), int(m), int(d)).date().isoformat()
        except Exception:
            return None

    return None


def extract_location(message: str) -> Optional[str]:
    text = normalize_text(message)

    known_locations = [
        "التجمع",
        "التجمع الخامس",
        "مدينة نصر",
        "مدينه نصر",
        "مصر الجديدة",
        "مصر الجديده",
        "المعادي",
        "الزمالك",
        "الشيخ زايد",
        "زايد",
        "اكتوبر",
        "اكتوبر",
        "6 اكتوبر",
        "الدقي",
        "المهندسين",
        "الرحاب",
        "مدينتي",
        "الشروق",
        "العبور",
        "الجيزة",
        "جيزة",
        "القاهرة",
        "القاهره",
        "اسكندرية",
        "اسكندريه",
    ]

    for loc in known_locations:
        if normalize_text(loc) in text:
            return loc

    # Pattern: "في التجمع", "في مدينة نصر"
    match = re.search(r"(?:في|فى)\s+([\u0600-\u06FF\s]{3,30})", message or "")
    if match:
        candidate = match.group(1).strip()
        candidate = re.split(r"[،,.!?؟]", candidate)[0].strip()
        if len(candidate) >= 3:
            return candidate

    return None


def extract_datetime_location_patch(
    *,
    message: str,
    variables: Dict[str, Any],
    workflow: str,
) -> Dict[str, Any]:
    variables = variables or {}
    updates: Dict[str, Any] = {}

    date_value = extract_date(message)
    time_value = extract_time(message)
    location_value = extract_location(message)

    if workflow == "car_sales":
        if date_value and not variables.get("preferred_viewing_date"):
            updates["preferred_viewing_date"] = date_value

        if time_value and not variables.get("preferred_viewing_time"):
            updates["preferred_viewing_time"] = time_value

        if location_value and not variables.get("location"):
            updates["location"] = location_value

        if updates:
            updates["intent"] = "viewing_request"
            updates["workflow_stage"] = "viewing_requested"

    elif workflow == "service_booking":
        if date_value and not variables.get("appointment_date"):
            updates["appointment_date"] = date_value

        if time_value and not variables.get("appointment_time"):
            updates["appointment_time"] = time_value

        if location_value and not variables.get("location"):
            updates["location"] = location_value

        if updates:
            updates["intent"] = "booking_request"
            updates["workflow_stage"] = "booking_requested"

    else:
        if date_value:
            updates["preferred_date"] = date_value
        if time_value:
            updates["preferred_time"] = time_value
        if location_value:
            updates["location"] = location_value

    return updates
