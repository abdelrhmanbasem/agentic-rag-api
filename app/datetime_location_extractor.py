# app/datetime_location_extractor.py
# Deterministic Arabic/English date, time, and location extraction.
#
# Goal:
# - Make booking/viewing flows smoother.
# - Avoid GPT extraction for common WhatsApp messages:
#   "بكرة 3 في التجمع", "السبت العصر", "مدينة نصر", "بعد المغرب"
#
# This version:
# - Overrides bad non-ISO dates from GPT extraction.
# - Exposes a fast response builder for zero-token scheduling turns.

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


def is_arabic_text(text: str) -> bool:
    return bool(re.search(r"[\u0600-\u06FF]", text or ""))


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

    if "العصر" in text:
        return "16:00"

    if "بعد المغرب" in text or "المغرب" in text:
        return "19:00"

    if "بالليل" in text or "ليل" in text:
        return "20:00"

    if "الصبح" in text or "صباح" in text:
        return "10:00"

    # Examples:
    # الساعة 3
    # بكرة 3
    # 3:30
    # 3 مساء
    match = re.search(
        r"(?:الساعه|الساعة|على|علي)?\s*\b(\d{1,2})(?::(\d{2}))?\s*(م|مساء|pm|ص|صباح|am)?\b",
        text,
    )

    if match:
        hour = int(match.group(1))
        minute = int(match.group(2) or 0)
        suffix = match.group(3)

        if hour > 23:
            return None

        # Arabic sales chats usually use bare 3/4/5 as afternoon.
        if suffix in ["م", "مساء", "pm"] and hour < 12:
            hour += 12
        elif suffix in ["ص", "صباح", "am"]:
            pass
        elif 1 <= hour <= 7:
            hour += 12

        if 0 <= hour <= 23 and 0 <= minute <= 59:
            return f"{hour:02d}:{minute:02d}"

    return None


def extract_date(message: str, now: Optional[datetime] = None) -> Optional[str]:
    text = normalize_text(message)
    now = now or datetime.utcnow()

    # Important: check "after tomorrow" before "tomorrow".
    if "بعد بكره" in text or "بعد بكرة" in message:
        return (now + timedelta(days=2)).date().isoformat()

    if "بكره" in text or "بكرة" in message or "tomorrow" in text:
        return (now + timedelta(days=1)).date().isoformat()

    if "النهارده" in text or "اليوم" in text or "today" in text:
        return now.date().isoformat()

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
        "التجمع الخامس",
        "التجمع",
        "مدينة نصر",
        "مدينه نصر",
        "مصر الجديدة",
        "مصر الجديده",
        "المعادي",
        "الزمالك",
        "الشيخ زايد",
        "زايد",
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

    match = re.search(r"(?:في|فى)\s+([\u0600-\u06FF\s]{3,30})", message or "")
    if match:
        candidate = match.group(1).strip()
        candidate = re.split(r"[،,.!?؟]", candidate)[0].strip()

        # Avoid taking a whole sentence as location.
        stop_words = ["بكرة", "بكره", "الساعة", "الساعه", "تمام", "ممكن", "عايز"]
        for stop in stop_words:
            candidate = candidate.replace(stop, "").strip()

        if 3 <= len(candidate) <= 30:
            return candidate

    return None


def looks_like_datetime_location_message(message: str) -> bool:
    text = normalize_text(message)

    date_markers = [
        "بكره",
        "بكرة",
        "بعد بكره",
        "بعد بكرة",
        "النهارده",
        "اليوم",
        "السبت",
        "الاحد",
        "الأحد",
        "الاتنين",
        "الاثنين",
        "التلات",
        "الثلاث",
        "الثلاثاء",
        "الاربع",
        "الاربعاء",
        "الخميس",
        "الجمعه",
        "الجمعة",
        "today",
        "tomorrow",
    ]

    time_markers = [
        "الساعة",
        "الساعه",
        "العصر",
        "المغرب",
        "بالليل",
        "الصبح",
        "pm",
        "am",
    ]

    location = extract_location(message)
    date = extract_date(message)
    time_value = extract_time(message)

    if date and (time_value or location):
        return True

    if time_value and location:
        return True

    if any(marker in text for marker in date_markers) and any(marker in text for marker in time_markers):
        return True

    # "بكرة 3" without location still matters in an active viewing flow.
    if date and time_value:
        return True

    return False


def should_override_date(existing_value: Any, new_value: Optional[str]) -> bool:
    if not new_value:
        return False

    if not existing_value:
        return True

    # Override GPT mistakes like "بكرة 3 في التجمع".
    if not is_iso_date(existing_value):
        return True

    return False


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


def format_user_friendly_date(message: str, iso_date: Optional[str]) -> str:
    text = normalize_text(message)

    if "بكره" in text or "بكرة" in message:
        return "بكرة"

    if "بعد بكره" in text or "بعد بكرة" in message:
        return "بعد بكرة"

    if "النهارده" in text or "اليوم" in text:
        return "النهارده"

    return iso_date or "المعاد"


def format_user_friendly_time(time_value: Optional[str]) -> str:
    if not time_value:
        return ""

    try:
        hour, minute = time_value.split(":")
        hour_i = int(hour)
        minute_i = int(minute)

        display_hour = hour_i
        suffix = ""

        if hour_i == 0:
            display_hour = 12
            suffix = " صباحًا"
        elif 1 <= hour_i < 12:
            display_hour = hour_i
            suffix = " صباحًا"
        elif hour_i == 12:
            display_hour = 12
            suffix = " ظهرًا"
        else:
            display_hour = hour_i - 12
            suffix = " مساءً"

        if minute_i:
            return f"{display_hour}:{minute_i:02d}{suffix}"

        return f"{display_hour}{suffix}"
    except Exception:
        return time_value


def build_datetime_location_fast_response(
    *,
    message: str,
    variables: Dict[str, Any],
    workflow: str,
) -> Optional[Dict[str, Any]]:
    """
    Returns a zero-token scheduling response if the message clearly provides date/time/location.
    Caller should merge updates, save variables, save assistant message, and return.
    """
    variables = variables or {}

    if not looks_like_datetime_location_message(message):
        return None

    updates = extract_datetime_location_patch(
        message=message,
        variables=variables,
        workflow=workflow,
    )

    if not updates:
        return None

    merged = dict(variables)
    merged.update(updates)

    arabic = is_arabic_text(message)

    if workflow == "car_sales":
        model = (
            (merged.get("selected_item") or {}).get("model")
            if isinstance(merged.get("selected_item"), dict)
            else None
        ) or merged.get("matched_car_model") or merged.get("car_brand") or "العربية"

        date_value = merged.get("preferred_viewing_date")
        time_value = merged.get("preferred_viewing_time") or merged.get("appointment_time")
        location = merged.get("location")
        phone = merged.get("phone_number")

        friendly_date = format_user_friendly_date(message, date_value)
        friendly_time = format_user_friendly_time(time_value)

        if arabic:
            details = []
            if friendly_date:
                details.append(friendly_date)
            if friendly_time:
                details.append(f"الساعة {friendly_time}")
            if location:
                details.append(f"في {location}")

            joined = " ".join(details).strip()

            if phone:
                answer = f"تمام، هظبطلك معاينة {model} {joined}. هنتواصل معاك على نفس الرقم لتأكيد التفاصيل."
                updates["lead_stage"] = "confirmed"
                updates["workflow_stage"] = "confirmed"
                action = "confirm_viewing"
                skip_summary = False
                skip_memory = False
            else:
                answer = f"تمام، هظبطلك معاينة {model} {joined}. محتاج رقم للتواصل عشان نأكد المعاد."
                action = "ask_phone"
                skip_summary = True
                skip_memory = True

        else:
            joined_parts = []
            if friendly_date:
                joined_parts.append(friendly_date)
            if friendly_time:
                joined_parts.append(f"at {friendly_time}")
            if location:
                joined_parts.append(f"in {location}")
            joined = " ".join(joined_parts).strip()

            if phone:
                answer = f"Great, I’ll set up the viewing for {model} {joined}. We’ll contact you on the same number to confirm the details."
                updates["lead_stage"] = "confirmed"
                updates["workflow_stage"] = "confirmed"
                action = "confirm_viewing"
                skip_summary = False
                skip_memory = False
            else:
                answer = f"Great, I’ll set up the viewing for {model} {joined}. I just need a contact number to confirm it."
                action = "ask_phone"
                skip_summary = True
                skip_memory = True

        return {
            "answer": answer,
            "updates": updates,
            "model_tier": "fast_path",
            "action": action,
            "skip_summary": skip_summary,
            "skip_memory": skip_memory,
        }

    if workflow == "service_booking":
        service = merged.get("service_needed") or ("الخدمة" if arabic else "the service")
        date_value = merged.get("appointment_date")
        time_value = merged.get("appointment_time")
        phone = merged.get("phone_number")

        friendly_date = format_user_friendly_date(message, date_value)
        friendly_time = format_user_friendly_time(time_value)

        if arabic:
            if phone:
                answer = f"تمام، هظبطلك حجز {service} {friendly_date} الساعة {friendly_time}. هنتواصل معاك على نفس الرقم لتأكيد التفاصيل."
                updates["workflow_stage"] = "confirmed"
                action = "confirm_booking"
                skip_summary = False
                skip_memory = False
            else:
                answer = f"تمام، هظبطلك حجز {service} {friendly_date} الساعة {friendly_time}. محتاج رقم للتواصل عشان نأكد الحجز."
                action = "ask_phone"
                skip_summary = True
                skip_memory = True
        else:
            if phone:
                answer = f"Great, I’ll set up your {service} booking for {friendly_date} at {friendly_time}. We’ll contact you on the same number to confirm."
                updates["workflow_stage"] = "confirmed"
                action = "confirm_booking"
                skip_summary = False
                skip_memory = False
            else:
                answer = f"Great, I’ll set up your {service} booking for {friendly_date} at {friendly_time}. I just need a contact number to confirm it."
                action = "ask_phone"
                skip_summary = True
                skip_memory = True

        return {
            "answer": answer,
            "updates": updates,
            "model_tier": "fast_path",
            "action": action,
            "skip_summary": skip_summary,
            "skip_memory": skip_memory,
        }

    return None
