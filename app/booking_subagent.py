# app/booking_subagent.py

import json
import math
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from app.llm import chat_text, model_for_tier


BOOKING_AGENT_NAME = "booking_agent"
DEFAULT_TZ = "Africa/Cairo"


BRANCH_COORDINATES = {
    "New Cairo": (30.0074, 31.4913),
    "Nasr City": (30.0561, 31.3300),
    "Sheikh Zayed": (30.0131, 30.9769),
    "Maadi": (29.9602, 31.2569),
    "Alexandria": (31.2001, 29.9187),
}


BRANCH_DISPLAY_NAMES = {
    "New Cairo": "التجمع",
    "Nasr City": "مدينة نصر",
    "Sheikh Zayed": "الشيخ زايد",
    "Maadi": "المعادي",
    "Alexandria": "إسكندرية",
}


AREA_COORDINATES = {
    # East / New Cairo side
    "Obour": (30.2285, 31.4799),
    "Shorouk": (30.1417, 31.6167),
    "Madinaty": (30.0934, 31.6385),
    "Rehab": (30.0637, 31.4880),
    "Badr City": (30.1360, 31.7060),
    "Mostakbal City": (30.0606, 31.5862),
    "10th of Ramadan": (30.3065, 31.7415),
    "New Cairo Area": (30.0074, 31.4913),
    "Tagamoa": (30.0074, 31.4913),
    "Fifth Settlement": (30.0085, 31.4815),
    "First Settlement": (30.0610, 31.4430),

    # Cairo center / east
    "Nasr City Area": (30.0561, 31.3300),
    "Heliopolis": (30.0912, 31.3220),
    "Nozha": (30.1114, 31.3490),
    "Sheraton": (30.1069, 31.3767),
    "Gesr Suez": (30.1242, 31.3319),
    "Ain Shams": (30.1290, 31.3310),
    "Matariya": (30.1237, 31.3136),
    "Salam City": (30.1706, 31.4215),
    "Marg": (30.1521, 31.3359),
    "Shubra": (30.1056, 31.2459),
    "Mokattam": (30.0217, 31.3034),
    "Downtown Cairo": (30.0444, 31.2357),
    "Zamalek": (30.0617, 31.2195),
    "Garden City": (30.0375, 31.2313),
    "Abbasiya": (30.0728, 31.2806),
    "Ramses": (30.0626, 31.2477),

    # South Cairo
    "Maadi Area": (29.9602, 31.2569),
    "Zahraa Maadi": (29.9689, 31.3147),
    "Katameya": (29.9953, 31.4314),
    "Helwan": (29.8414, 31.3008),
    "Tora": (29.9368, 31.2704),

    # Giza / west
    "Giza": (30.0131, 31.2089),
    "Dokki": (30.0384, 31.2122),
    "Mohandessin": (30.0566, 31.2006),
    "Agouza": (30.0522, 31.2108),
    "Haram": (29.9911, 31.1607),
    "Faisal": (30.0020, 31.1693),
    "Moneeb": (29.9816, 31.2129),
    "October": (29.9285, 30.9188),
    "6 October": (29.9285, 30.9188),
    "Sheikh Zayed Area": (30.0131, 30.9769),
    "Hadayek October": (29.9140, 30.8655),
    "Hadayek Ahram": (29.9544, 31.0965),

    # Alexandria
    "Alexandria Area": (31.2001, 29.9187),
    "Smouha": (31.2089, 29.9447),
    "Sidi Gaber": (31.2183, 29.9422),
    "Gleem": (31.2361, 29.9604),
    "Miami Alexandria": (31.2647, 29.9990),
    "Stanley": (31.2372, 29.9567),
    "Mandara": (31.2809, 30.0144),
    "Agami": (31.0957, 29.7604),
}


AREA_ALIASES = {
    # Obour
    "el obour": "Obour",
    "el-obour": "Obour",
    "al obour": "Obour",
    "al-obour": "Obour",
    "el 3obour": "Obour",
    "3obour": "Obour",
    "obour": "Obour",
    "العبور": "Obour",
    "عبور": "Obour",

    # East / New Cairo side
    "الشروق": "Shorouk",
    "شروق": "Shorouk",
    "shorouk": "Shorouk",
    "el shorouk": "Shorouk",
    "مدينتي": "Madinaty",
    "madinaty": "Madinaty",
    "الرحاب": "Rehab",
    "رحاب": "Rehab",
    "rehab": "Rehab",
    "بدر": "Badr City",
    "badr": "Badr City",
    "مدينة بدر": "Badr City",
    "مدينه بدر": "Badr City",
    "المستقبل": "Mostakbal City",
    "mostakbal": "Mostakbal City",
    "مستقبل سيتي": "Mostakbal City",
    "العاشر": "10th of Ramadan",
    "العاشر من رمضان": "10th of Ramadan",
    "10th ramadan": "10th of Ramadan",
    "10th of ramadan": "10th of Ramadan",
    "التجمع": "New Cairo Area",
    "tagamoa": "Tagamoa",
    "tagamo3": "Tagamoa",
    "new cairo": "New Cairo Area",
    "القاهرة الجديدة": "New Cairo Area",
    "القاهره الجديده": "New Cairo Area",
    "التجمع الخامس": "Fifth Settlement",
    "fifth settlement": "Fifth Settlement",
    "5th settlement": "Fifth Settlement",
    "التجمع الاول": "First Settlement",
    "التجمع الأول": "First Settlement",
    "first settlement": "First Settlement",

    # Cairo center / east
    "مدينة نصر": "Nasr City Area",
    "مدينه نصر": "Nasr City Area",
    "nasr city": "Nasr City Area",
    "مصر الجديدة": "Heliopolis",
    "مصر الجديده": "Heliopolis",
    "heliopolis": "Heliopolis",
    "النزهة": "Nozha",
    "النزهه": "Nozha",
    "nozha": "Nozha",
    "sheraton": "Sheraton",
    "شيراتون": "Sheraton",
    "جسر السويس": "Gesr Suez",
    "gesr suez": "Gesr Suez",
    "عين شمس": "Ain Shams",
    "ain shams": "Ain Shams",
    "المطرية": "Matariya",
    "المطريه": "Matariya",
    "matariya": "Matariya",
    "السلام": "Salam City",
    "مدينة السلام": "Salam City",
    "مدينه السلام": "Salam City",
    "el salam": "Salam City",
    "المرج": "Marg",
    "marg": "Marg",
    "شبرا": "Shubra",
    "شبرا مصر": "Shubra",
    "shubra": "Shubra",
    "shobra": "Shubra",
    "shubra masr": "Shubra",
    "روض الفرج": "Shubra",
    "rod el farag": "Shubra",
    "الساحل": "Shubra",
    "el sahel": "Shubra",
    "المقطم": "Mokattam",
    "mokattam": "Mokattam",
    "وسط البلد": "Downtown Cairo",
    "downtown": "Downtown Cairo",
    "زمالك": "Zamalek",
    "الزمالك": "Zamalek",
    "zamalek": "Zamalek",
    "جاردن سيتي": "Garden City",
    "garden city": "Garden City",
    "العباسية": "Abbasiya",
    "العباسيه": "Abbasiya",
    "abbasiya": "Abbasiya",
    "رمسيس": "Ramses",
    "ramses": "Ramses",

    # South Cairo
    "المعادي": "Maadi Area",
    "معادي": "Maadi Area",
    "maadi": "Maadi Area",
    "زهراء المعادي": "Zahraa Maadi",
    "zahraa maadi": "Zahraa Maadi",
    "القطامية": "Katameya",
    "القطاميه": "Katameya",
    "katameya": "Katameya",
    "helwan": "Helwan",
    "حلوان": "Helwan",
    "طرة": "Tora",
    "طره": "Tora",
    "tora": "Tora",

    # Giza / west
    "giza": "Giza",
    "الجيزة": "Giza",
    "الجيزه": "Giza",
    "جيزة": "Giza",
    "جيزه": "Giza",
    "الدقي": "Dokki",
    "الدقى": "Dokki",
    "dokki": "Dokki",
    "المهندسين": "Mohandessin",
    "mohandessin": "Mohandessin",
    "العجوزة": "Agouza",
    "العجوزه": "Agouza",
    "agouza": "Agouza",
    "الهرم": "Haram",
    "هرم": "Haram",
    "haram": "Haram",
    "فيصل": "Faisal",
    "faisal": "Faisal",
    "المنيب": "Moneeb",
    "moneeb": "Moneeb",
    "اكتوبر": "6 October",
    "أكتوبر": "6 October",
    "6 اكتوبر": "6 October",
    "٦ اكتوبر": "6 October",
    "six october": "6 October",
    "6 october": "6 October",
    "october": "October",
    "الشيخ زايد": "Sheikh Zayed Area",
    "زايد": "Sheikh Zayed Area",
    "zayed": "Sheikh Zayed Area",
    "sheikh zayed": "Sheikh Zayed Area",
    "حدائق اكتوبر": "Hadayek October",
    "حدائق أكتوبر": "Hadayek October",
    "hadayek october": "Hadayek October",
    "حدائق الاهرام": "Hadayek Ahram",
    "حدائق الأهرام": "Hadayek Ahram",
    "hadayek ahram": "Hadayek Ahram",

    # Alexandria
    "اسكندرية": "Alexandria Area",
    "إسكندرية": "Alexandria Area",
    "الاسكندرية": "Alexandria Area",
    "الإسكندرية": "Alexandria Area",
    "alex": "Alexandria Area",
    "alexandria": "Alexandria Area",
    "سموحة": "Smouha",
    "سموحه": "Smouha",
    "smouha": "Smouha",
    "سيدي جابر": "Sidi Gaber",
    "سيدى جابر": "Sidi Gaber",
    "sidi gaber": "Sidi Gaber",
    "جليم": "Gleem",
    "gleem": "Gleem",
    "ميامي": "Miami Alexandria",
    "miami": "Miami Alexandria",
    "ستانلي": "Stanley",
    "stanley": "Stanley",
    "المندرة": "Mandara",
    "المندره": "Mandara",
    "mandara": "Mandara",
    "العجمي": "Agami",
    "العجمى": "Agami",
    "agami": "Agami",
}


def _norm(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _lower(value: Any) -> str:
    return _norm(value).lower()


def _has_any(text: str, words: List[str]) -> bool:
    text = _lower(text)
    return any(word.lower() in text for word in words)


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value

    if value is None:
        return False

    if isinstance(value, (int, float)):
        return value == 1

    text = str(value).strip().lower()

    if text in {"true", "1", "yes", "y", "available", "متاح"}:
        return True

    if text in {"false", "0", "no", "n", "unavailable", "not_available", "not available", "غير متاح"}:
        return False

    return False


def _clean_phone_from_user_id(user_id: str) -> str:
    user_id = _norm(user_id).replace("=", "")
    return user_id.replace("@s.whatsapp.net", "").strip()


def _today_egypt() -> datetime:
    return datetime.now(ZoneInfo(DEFAULT_TZ))


def _next_weekday(target_weekday: int, include_today: bool = False) -> str:
    today = _today_egypt().date()
    days_ahead = target_weekday - today.weekday()

    if days_ahead < 0 or (days_ahead == 0 and not include_today):
        days_ahead += 7

    return (today + timedelta(days=days_ahead)).isoformat()


def _parse_day_month(day: int, month: int) -> str:
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
    result = str(text)

    for i, digit in enumerate("٠١٢٣٤٥٦٧٨٩"):
        result = result.replace(digit, str(i))

    for i, digit in enumerate("۰۱۲۳۴۵۶۷۸۹"):
        result = result.replace(digit, str(i))

    return result


def _clean_location_text(text: str) -> str:
    text = _lower(_arabic_digits_to_latin(text))
    text = text.replace("-", " ")
    text = text.replace("_", " ")
    text = re.sub(r"[،,.;:()\[\]{}!?؟]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _distance_km(coord_a: Tuple[float, float], coord_b: Tuple[float, float]) -> float:
    lat1, lon1 = coord_a
    lat2, lon2 = coord_b

    radius = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    )

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return radius * c


def branch_display_name(branch: str) -> str:
    branch = _norm(branch)
    return BRANCH_DISPLAY_NAMES.get(branch, branch or "الفرع")


def normalize_reason_for_customer(reason: str) -> str:
    reason_l = _lower(reason)

    if not reason_l:
        return "المعاد ده مش متاح حاليًا"

    if "exact slot not listed" in reason_l:
        return "الوقت ده مش مفتوح في جدول المواعيد"

    if "slot is full" in reason_l or "fully booked" in reason_l or "full" in reason_l:
        return "المعاد ده اتحجز بالكامل"

    if (
        "equipment" in reason_l
        or "maintenance" in reason_l
        or "manager blocked" in reason_l
        or "blocked" in reason_l
    ):
        return "القسم مقفول في الوقت ده بسبب صيانة/تجهيزات"

    return reason


def parse_availability_tool_result(tool_result: Dict[str, Any]) -> Dict[str, Any]:
    read = tool_result.get("read_availability") or tool_result or {}
    requested = tool_result.get("requested") or read.get("requested") or {}

    slot_status = str(read.get("slot_status") or "").strip().lower()

    if "exact_slot_available" in read:
        exact_available = parse_bool(read.get("exact_slot_available"))
    elif "available" in read:
        exact_available = parse_bool(read.get("available"))
    else:
        exact_available = False

    exact_slot_found = parse_bool(read.get("exact_slot_found"))

    available = exact_available is True or slot_status == "available"

    if slot_status in {"unavailable", "not_available", "not available"}:
        available = False

    raw_reason = (
        read.get("unavailable_reason")
        or read.get("reason")
        or read.get("unavailableReason")
        or ""
    )

    nearest_slots = read.get("nearest_slots") or []
    nearest_slots_text = read.get("nearest_slots_text") or ""

    return {
        "available": available,
        "slot_status": slot_status,
        "exact_slot_found": exact_slot_found,
        "exact_slot_available": exact_available,
        "unavailable_reason": normalize_reason_for_customer(raw_reason),
        "raw_unavailable_reason": raw_reason,
        "nearest_slots": nearest_slots,
        "nearest_slots_text": nearest_slots_text,
        "requested": requested,
        "raw_read_availability": read,
    }


def extract_user_area(message: str) -> Optional[str]:
    text = _clean_location_text(message)
    sorted_aliases = sorted(AREA_ALIASES.items(), key=lambda item: len(item[0]), reverse=True)

    for alias, area in sorted_aliases:
        alias_clean = _clean_location_text(alias)

        if not alias_clean:
            continue

        pattern = r"(?<!\w)" + re.escape(alias_clean) + r"(?!\w)"

        if re.search(pattern, text):
            return area

    return None


def recommend_nearest_branches_for_area(area: str, limit: int = 2) -> List[Dict[str, Any]]:
    area_coord = AREA_COORDINATES.get(_norm(area))

    if not area_coord:
        return []

    ranked = []

    for branch, branch_coord in BRANCH_COORDINATES.items():
        distance = _distance_km(area_coord, branch_coord)
        ranked.append(
            {
                "branch": branch,
                "branch_ar": branch_display_name(branch),
                "distance_km_estimate": round(distance, 1),
            }
        )

    ranked.sort(key=lambda item: item["distance_km_estimate"])

    return ranked[:limit]


def recommend_branch_for_area(area: str) -> Optional[str]:
    ranked = recommend_nearest_branches_for_area(area, limit=1)

    if not ranked:
        return None

    return ranked[0]["branch"]


def detect_nearest_branch_question(message: str) -> bool:
    text = _clean_location_text(message)

    question_markers = [
        "اقرب فرع",
        "أقرب فرع",
        "ايه اقرب فرع",
        "إيه أقرب فرع",
        "اقرب فرع ليا",
        "أقرب فرع ليا",
        "اقرب مركز",
        "أقرب مركز",
        "في فرع هنا",
        "لو في فرع هنا",
        "في مركز هنا",
        "فرع هنا",
        "مركز هنا",
        "قريب مني",
        "قريب ليا",
        "قريبة مني",
        "قريبه مني",
        "nearest branch",
        "closest branch",
        "near branch",
        "nearest center",
        "closest center",
    ]

    if any(marker.lower() in text for marker in question_markers):
        return True

    location_intro_markers = [
        "انا في",
        "أنا في",
        "انا ساكن في",
        "انا قريب من",
        "أنا قريب من",
        "i am in",
        "i'm in",
        "im in",
        "near",
        "close to",
    ]

    if any(marker.lower() in text for marker in location_intro_markers):
        return bool(extract_user_area(message))

    return False


def extract_branch(message: str, variables: Dict[str, Any]) -> Optional[str]:
    existing = variables.get("location_branch") or variables.get("branch")

    if existing:
        return existing

    text = _clean_location_text(message)

    branch_map = {
        "new cairo": "New Cairo",
        "القاهرة الجديدة": "New Cairo",
        "القاهره الجديده": "New Cairo",
        "التجمع": "New Cairo",
        "tagamoa": "New Cairo",
        "tagamo3": "New Cairo",
        "nasr city": "Nasr City",
        "مدينة نصر": "Nasr City",
        "مدينه نصر": "Nasr City",
        "sheikh zayed": "Sheikh Zayed",
        "zayed": "Sheikh Zayed",
        "زايد": "Sheikh Zayed",
        "الشيخ زايد": "Sheikh Zayed",
        "maadi": "Maadi",
        "المعادي": "Maadi",
        "معادي": "Maadi",
        "alexandria": "Alexandria",
        "alex": "Alexandria",
        "اسكندرية": "Alexandria",
        "إسكندرية": "Alexandria",
        "الإسكندرية": "Alexandria",
        "الاسكندرية": "Alexandria",
    }

    sorted_branches = sorted(branch_map.items(), key=lambda item: len(item[0]), reverse=True)

    for key, branch in sorted_branches:
        key_clean = _clean_location_text(key)

        if key_clean in text:
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

    match = re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", text)

    if match:
        return _normalize_time(int(match.group(1)), int(match.group(2)))

    match = re.search(r"\b(\d{1,2})\s*(am|pm)\b", text)

    if match:
        hour = int(match.group(1))
        suffix = match.group(2)
        return _normalize_time(hour, 0, pm_hint=suffix == "pm", am_hint=suffix == "am")

    match = re.search(r"(?:الساعة|الساعه|at)\s*(\d{1,2})", text)

    if match:
        hour = int(match.group(1))
        pm_hint = any(x in text for x in ["مساء", "بالليل", "الظهر", "العصر", "المغرب", "pm"])
        am_hint = any(x in text for x in ["صباح", "الصبح", "am"])
        return _normalize_time(hour, 0, pm_hint=pm_hint, am_hint=am_hint)

    match = re.search(r"\b(\d{1,2})\s*(الصبح|صباح|الظهر|ظهر|العصر|مساء|المغرب|بالليل)\b", text)

    if match:
        hour = int(match.group(1))
        hint = match.group(2)
        pm_hint = hint in ["الظهر", "ظهر", "العصر", "مساء", "المغرب", "بالليل"]
        am_hint = hint in ["الصبح", "صباح"]
        return _normalize_time(hour, 0, pm_hint=pm_hint, am_hint=am_hint)

    return None


def extract_date(message: str, variables: Dict[str, Any]) -> Optional[str]:
    existing = variables.get("appointment_date") or variables.get("date")

    if existing:
        return existing

    text = _lower(_arabic_digits_to_latin(message))
    today = _today_egypt().date()

    match = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", text)

    if match:
        return match.group(1)

    if any(x in text for x in ["بعد بكرة", "بعد بكره", "day after tomorrow"]):
        return (today + timedelta(days=2)).isoformat()

    if any(x in text for x in ["النهارده", "النهاردة", "اليوم", "today"]):
        return today.isoformat()

    if any(x in text for x in ["بكرة", "بكره", "tomorrow"]):
        return (today + timedelta(days=1)).isoformat()

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


def collect_slot_variables(message: str, variables: Dict[str, Any]) -> Dict[str, Any]:
    variables = dict(variables or {})

    branch = extract_branch(message, variables)
    date = extract_date(message, variables)
    time = extract_time(message, variables)
    user_area = extract_user_area(message)

    if user_area:
        variables["user_area"] = user_area

    if not branch and user_area:
        recommended_branch = recommend_branch_for_area(user_area)
        nearest_branches = recommend_nearest_branches_for_area(user_area, limit=2)

        if recommended_branch:
            branch = recommended_branch
            variables["location_branch"] = recommended_branch
            variables["branch_recommendation_reason"] = f"nearest_known_branch_for_{user_area}"
            variables["nearest_branch_options"] = nearest_branches

    if detect_nearest_branch_question(message):
        variables["nearest_branch_question"] = True

    section = (
        variables.get("recommended_section")
        or variables.get("service_needed")
        or variables.get("section")
    )

    symptoms = variables.get("symptoms") or []
    issue_description = _lower(variables.get("issue_description", ""))

    if not section:
        if (
            (isinstance(symptoms, list) and any("overheating" in str(s).lower() for s in symptoms))
            or "سخونة" in issue_description
            or "بتسخن" in issue_description
            or "المؤشر بيعلى" in issue_description
        ):
            section = "Engine Diagnostics"
            variables["recommended_section"] = section
            variables["service_needed"] = section
            variables.setdefault("customer_facing_section", "قسم كشف الموتور")

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


def missing_confirmation_fields(variables: Dict[str, Any]) -> List[str]:
    missing = []

    if not variables.get("customer_full_name"):
        missing.append("customer_full_name")

    if not variables.get("plate_digits"):
        missing.append("plate_digits")

    if not variables.get("phone_confirmed"):
        missing.append("phone_confirmed")

    return missing


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


def detect_booking_intent(message: str, variables: Dict[str, Any]) -> bool:
    text = _lower(message)
    variables = variables or {}

    if variables.get("active_subagent") == BOOKING_AGENT_NAME:
        return True

    if variables.get("booking_stage"):
        return True

    if variables.get("customer_agreed_to_visit") is True:
        return True

    if variables.get("intent") == "booking_request":
        return True

    if variables.get("workflow_stage") == "booking_requested":
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
        "ظبطلي",
        "أظبطلي",
        "حددلي",
        "ميعاد",
        "معاد",
        "موعد",
        "ياريت",
        "اه ياريت",
        "أه ياريت",
        "لو سمحت",
        "appointment",
        "book",
        "booking",
        "reserve",
        "schedule",
        "visit",
    ]

    if _has_any(text, booking_words):
        return True

    if detect_nearest_branch_question(message):
        return True

    agreement_words = [
        "اه",
        "أه",
        "ايوه",
        "أيوه",
        "ايوا",
        "تمام",
        "ماشي",
        "ok",
        "okay",
        "yes",
    ]

    if (
        variables.get("next_service_action") == "offer_booking"
        or variables.get("recommended_next_action") == "offer_booking"
    ) and _has_any(text, agreement_words):
        return True

    return False


def detect_confirmation(message: str) -> bool:
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

    return _has_any(message, confirmation_words)


def _extract_plate_digits(message: str) -> Optional[str]:
    text = _arabic_digits_to_latin(message)
    matches = re.findall(r"\b\d{3,6}\b", text)

    if matches:
        return matches[-1]

    return None


def _extract_customer_name(message: str) -> Optional[str]:
    raw_text = _norm(message)
    text = _arabic_digits_to_latin(raw_text)

    patterns = [
        r"(?:اسمي|الاسم|انا اسمي|أنا اسمي|انا|أنا)\s+([^،,\n\d]+)",
        r"(?:name is|my name is|i am|i'm)\s+([^،,\n\d]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)

        if match:
            name = match.group(1).strip()
            name = re.sub(
                r"\b(?:ونفس الرقم|نفس الرقم|same number|yes same|this number)\b",
                "",
                name,
                flags=re.IGNORECASE,
            ).strip()

            if 2 <= len(name) <= 60:
                return name

    cleaned = re.sub(r"\b\d{3,6}\b", " ", text)
    cleaned = re.sub(
        r"(ونفس الرقم|نفس الرقم|اه نفس الرقم|أه نفس الرقم|ايوه نفس الرقم|أيوه نفس الرقم|same number|yes same|this number|use this number)",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"[،,.;:()\[\]{}]", " ", cleaned)
    cleaned = " ".join(cleaned.split()).strip()

    bad_phrases = [
        "تمام",
        "ماشي",
        "اوكي",
        "أوكي",
        "احجز",
        "احجزلي",
        "ثبته",
        "ثبت",
        "نفس الرقم",
    ]

    if any(cleaned.lower() == bad.lower() for bad in bad_phrases):
        return None

    words = cleaned.split()

    if 2 <= len(words) <= 4 and 2 <= len(cleaned) <= 60:
        return cleaned

    return None


def _detect_phone_confirmed(message: str) -> bool:
    return _has_any(
        message,
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


def collect_booking_confirmation_details(message: str, variables: Dict[str, Any]) -> Dict[str, Any]:
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


def build_booking_facts(
    *,
    message: str,
    variables: Dict[str, Any],
    stage: str,
    missing_variables: Optional[List[str]] = None,
    action_required: Optional[Dict[str, Any]] = None,
    availability: Optional[Dict[str, Any]] = None,
    booking_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    nearest_options = variables.get("nearest_branch_options") or []

    facts = {
        "latest_customer_message": message,
        "stage": stage,
        "missing_variables": missing_variables or [],
        "action_required": action_required,
        "availability": availability,
        "booking_result": booking_result,
        "known": {
            "intent": variables.get("intent"),
            "workflow": variables.get("workflow"),
            "booking_stage": variables.get("booking_stage"),
            "location_branch": variables.get("location_branch"),
            "location_branch_ar": branch_display_name(variables.get("location_branch")),
            "user_area": variables.get("user_area"),
            "appointment_date": variables.get("appointment_date"),
            "appointment_time": variables.get("appointment_time"),
            "recommended_section": variables.get("recommended_section") or variables.get("service_needed"),
            "customer_facing_section": variables.get("customer_facing_section"),
            "customer_full_name": variables.get("customer_full_name"),
            "plate_digits": variables.get("plate_digits"),
            "phone_confirmed": variables.get("phone_confirmed"),
            "phone_number": variables.get("phone_number"),
            "nearest_branch_question": bool(variables.get("nearest_branch_question")),
            "nearest_branch_options": nearest_options,
        },
    }

    return facts


def fallback_booking_reply(facts: Dict[str, Any]) -> str:
    stage = facts.get("stage", "")
    known = facts.get("known") or {}
    availability = facts.get("availability") or {}
    missing = set(facts.get("missing_variables") or [])

    if availability:
        if availability.get("available") is True:
            branch = known.get("location_branch_ar") or "الفرع"
            date = known.get("appointment_date") or ""
            time = known.get("appointment_time") or ""
            return f"تمام، الميعاد متاح في فرع {branch} يوم {date} الساعة {time}. تحب أثبتهولك؟"

        reason = availability.get("unavailable_reason") or "المعاد ده مش متاح حاليًا"
        nearest = availability.get("nearest_slots_text") or ""

        if nearest:
            return (
                f"تمام، الميعاد ده مش متاح للأسف عشان {reason}. "
                f"بس لقيتلك أقرب مواعيد متاحة:\n\n{nearest}\n\n"
                "تحب أثبتلك واحد منهم؟"
            )

        return (
            f"تمام، الميعاد ده مش متاح للأسف عشان {reason}. "
            "تحب نجرب وقت تاني في نفس الفرع، ولا أدوّرلك في فرع قريب؟"
        )

    if known.get("nearest_branch_question") and not known.get("user_area") and not known.get("location_branch"):
        return "أكيد، قولّي إنت فين أو أقرب منطقة ليك، وأنا أقولك أنسب فرع. وبعدها نحدد اليوم والوقت المناسبين."

    if known.get("location_branch") and {"appointment_date", "appointment_time"} & missing:
        branch = known.get("location_branch_ar") or "الفرع"
        return f"تمام، نبدأ بفرع {branch}. قولّي اليوم والوقت اللي يناسبك."

    if known.get("nearest_branch_options") and {"appointment_date", "appointment_time"} & missing:
        options = known["nearest_branch_options"]
        first = options[0]
        second = options[1] if len(options) > 1 else None

        if second:
            return (
                f"تمام، أقرب فرع مناسب ليك غالبًا فرع {first['branch_ar']} "
                f"وبعده فرع {second['branch_ar']}. "
                f"نبدأ بفرع {first['branch_ar']}؟ قولّي اليوم والوقت اللي يناسبك."
            )

        return f"تمام، أقرب فرع مناسب ليك غالبًا فرع {first['branch_ar']}. قولّي اليوم والوقت اللي يناسبك."

    if stage == "booking_collecting_customer_details":
        parts = []

        if "customer_full_name" in missing:
            parts.append("اسم حضرتك بالكامل")

        if "plate_digits" in missing:
            parts.append("نمر العربية")

        if "phone_confirmed" in missing:
            parts.append("تأكيد نستخدم نفس رقم الواتساب للتواصل")

        if parts:
            return "تمام، عشان أثبت الحجز محتاج " + " و".join(parts) + "."

    if missing:
        if {"location_branch", "appointment_date", "appointment_time"} <= missing:
            return "تمام، تحب أنهي فرع، ويوم ووقت مناسبين ليك؟"

        if "appointment_date" in missing and "appointment_time" in missing:
            return "تمام، قولّي اليوم والوقت اللي يناسبك."

        if "appointment_date" in missing:
            return "تمام، تحب أنهي يوم؟"

        if "appointment_time" in missing:
            return "تمام، تحب الساعة كام؟"

    return "تمام، معاك."


def validate_booking_llm_answer(answer: str, facts: Dict[str, Any]) -> bool:
    if not answer or not answer.strip():
        return False

    answer_l = answer.lower()
    availability = facts.get("availability") or {}
    available = availability.get("available")

    if available is False:
        says_unavailable = (
            "مش متاح" in answer
            or "غير متاح" in answer
            or "للأسف" in answer
            or "unavailable" in answer_l
            or "not available" in answer_l
        )

        bad_available_phrases = [
            "المعاد متاح",
            "الموعد متاح",
            "ميعاد متاح",
            "slot is available",
            "appointment slot is available",
            "available",
            "تحب أثبته",
            "تأكيد الحجز",
            "confirm this booking",
        ]

        if any(phrase in answer_l for phrase in bad_available_phrases) and not says_unavailable:
            return False

    if available is True:
        bad_unavailable_phrases = [
            "مش متاح",
            "غير متاح",
            "unavailable",
            "not available",
            "للأسف الميعاد",
        ]

        if any(phrase in answer_l for phrase in bad_unavailable_phrases):
            return False

    known = facts.get("known") or {}

    if known.get("nearest_branch_question") and not known.get("user_area") and not known.get("location_branch"):
        asks_location = (
            "فين" in answer
            or "منطقة" in answer
            or "منطقتك" in answer
            or "اقرب مكان" in answer
            or "أقرب مكان" in answer
            or "where" in answer_l
            or "area" in answer_l
        )

        asks_time_first = (
            ("اليوم" in answer or "يوم" in answer)
            and ("الوقت" in answer or "الساعة" in answer)
            and not asks_location
        )

        if asks_time_first:
            return False

    forbidden = [
        "json",
        "tool",
        "backend",
        "variable",
        "workflow",
        "facts",
        "availability api",
    ]

    if any(word in answer_l for word in forbidden):
        return False

    return True


def llm_compose_booking_reply(
    *,
    assistant: Dict[str, Any],
    message: str,
    facts: Dict[str, Any],
) -> str:
    model = model_for_tier("normal")

    assistant_name = assistant.get("name") or assistant.get("id") or "the assistant"
    tone = assistant.get("tone") or "natural, helpful, concise"

    system_prompt = f"""
You are the booking sub-agent for {assistant_name}.

Your job:
Write the next customer-facing WhatsApp reply.

You must use the FACTS as the source of truth.
Do not invent availability, booking status, prices, branch names, dates, times, or visit IDs.

Critical rules:
- If facts.availability.available is false, never say the slot is available.
- If facts.availability.available is true, never say the slot is unavailable.
- If facts.availability.unavailable_reason exists, explain it naturally.
- If facts.availability.nearest_slots_text exists, offer those slots.
- If the slot is unavailable and no nearest slots are provided, ask whether to try another time in the same branch or another nearby branch.
- If customer asks for nearest branch and user_area is unknown, ask for their area/location first.
- If recommended nearest branch exists, recommend it naturally.
- If branch is clear but date/time are missing, ask for date/time.
- If slot is available and confirmation details are missing, ask only for missing details.
- If action_required exists, do not send filler like "I will check"; either return an empty reply or a very short transition only if explicitly needed.
- Use natural Egyptian Arabic unless the customer wrote in English.
- Keep it short: 1 to 3 sentences.
- Ask at most one grouped question.
- Do not mention JSON, variables, tools, backend, workflow, or facts.
- Tone: {tone}
"""

    user_prompt = f"""
Latest customer message:
{message}

FACTS:
{json.dumps(facts, ensure_ascii=False, indent=2)}

Write the next WhatsApp reply only.
"""

    try:
        answer = chat_text(
            model,
            [
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": user_prompt.strip()},
            ],
            max_tokens=260,
        ).strip()
    except Exception:
        answer = ""

    if validate_booking_llm_answer(answer, facts):
        return answer

    retry_prompt = f"""
Your previous answer violated the facts or routing rules.

Rewrite correctly.

Important:
- If availability.available is false, clearly say the requested slot is not available and why.
- If nearest branch was asked but user area is unknown, ask where the customer is first.
- Do not mention internal systems.

FACTS:
{json.dumps(facts, ensure_ascii=False, indent=2)}

Correct final WhatsApp reply only.
"""

    try:
        retry_answer = chat_text(
            model,
            [
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": retry_prompt.strip()},
            ],
            max_tokens=220,
        ).strip()
    except Exception:
        retry_answer = ""

    if validate_booking_llm_answer(retry_answer, facts):
        return retry_answer

    return fallback_booking_reply(facts)


def handle_availability_result(
    tool_result: Dict[str, Any],
    variables: Dict[str, Any],
    assistant: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    variables = dict(variables or {})
    assistant = assistant or {}

    availability = parse_availability_tool_result(tool_result)
    requested = availability.get("requested") or {}

    if requested:
        variables["location_branch"] = requested.get("branch") or variables.get("location_branch")
        variables["appointment_date"] = requested.get("date") or variables.get("appointment_date")
        variables["appointment_time"] = requested.get("time") or variables.get("appointment_time")
        variables["recommended_section"] = requested.get("section") or variables.get("recommended_section")
        variables["service_needed"] = variables.get("recommended_section") or variables.get("service_needed")

    variables["active_subagent"] = BOOKING_AGENT_NAME
    variables["raw_read_availability"] = availability.get("raw_read_availability") or {}

    if availability["available"]:
        variables["slot_status"] = "available"
        variables["booking_status"] = "slot_available"
        variables["booking_stage"] = "availability_available_waiting_confirmation"
        variables["booking_confirmation_requested"] = True

        facts = build_booking_facts(
            message="__availability_result__",
            variables=variables,
            stage=variables["booking_stage"],
            missing_variables=[],
            availability=availability,
        )

        answer = llm_compose_booking_reply(
            assistant=assistant,
            message="__availability_result__",
            facts=facts,
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
    variables["unavailable_reason"] = availability.get("unavailable_reason") or ""
    variables["nearest_slots"] = availability.get("nearest_slots") or []
    variables["nearest_slots_text"] = availability.get("nearest_slots_text") or ""

    facts = build_booking_facts(
        message="__availability_result__",
        variables=variables,
        stage=variables["booking_stage"],
        missing_variables=[],
        availability=availability,
    )

    answer = llm_compose_booking_reply(
        assistant=assistant,
        message="__availability_result__",
        facts=facts,
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


def handle_booking_result(
    tool_result: Dict[str, Any],
    variables: Dict[str, Any],
    assistant: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    variables = dict(variables or {})
    assistant = assistant or {}
    variables["active_subagent"] = BOOKING_AGENT_NAME

    success = parse_bool(tool_result.get("success"))

    if success:
        variables["booking_status"] = "confirmed"
        variables["booking_stage"] = "booking_confirmed"
        variables["visit_id"] = tool_result.get("visit_id") or tool_result.get("booking_id") or ""

        facts = build_booking_facts(
            message="__booking_result__",
            variables=variables,
            stage=variables["booking_stage"],
            booking_result=tool_result,
        )

        answer = llm_compose_booking_reply(
            assistant=assistant,
            message="__booking_result__",
            facts=facts,
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

    facts = build_booking_facts(
        message="__booking_result__",
        variables=variables,
        stage=variables["booking_stage"],
        booking_result=tool_result,
    )

    answer = llm_compose_booking_reply(
        assistant=assistant,
        message="__booking_result__",
        facts=facts,
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
    variables = dict(variables or {})

    if tool_result and tool_result.get("type") == "availability_result":
        return handle_availability_result(tool_result, variables, assistant=assistant)

    if tool_result and tool_result.get("type") == "booking_result":
        return handle_booking_result(tool_result, variables, assistant=assistant)

    if not detect_booking_intent(message, variables):
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
                facts = build_booking_facts(
                    message=message,
                    variables=variables,
                    stage=variables["booking_stage"],
                    missing_variables=missing_details,
                )

                answer = llm_compose_booking_reply(
                    assistant=assistant,
                    message=message,
                    facts=facts,
                )

                return {
                    "handled": True,
                    "answer": answer,
                    "variables": variables,
                    "active_subagent": BOOKING_AGENT_NAME,
                    "booking_stage": variables["booking_stage"],
                    "action_required": None,
                    "missing_variables": missing_details,
                    "recommended_next_action": "collect_booking_confirmation_details",
                    "reason": "User confirmed available slot; collecting booking details.",
                }

            variables["booking_stage"] = "booking_create_ready"
            action_required = action_create_booking(variables)

            facts = build_booking_facts(
                message=message,
                variables=variables,
                stage=variables["booking_stage"],
                action_required=action_required,
            )

            answer = llm_compose_booking_reply(
                assistant=assistant,
                message=message,
                facts=facts,
            )

            return {
                "handled": True,
                "answer": answer,
                "variables": variables,
                "active_subagent": BOOKING_AGENT_NAME,
                "booking_stage": variables["booking_stage"],
                "action_required": action_required,
                "missing_variables": [],
                "recommended_next_action": "create_booking",
                "reason": "Booking confirmation details complete; create booking required.",
            }

        facts = build_booking_facts(
            message=message,
            variables=variables,
            stage=booking_stage,
            missing_variables=[],
        )

        answer = llm_compose_booking_reply(
            assistant=assistant,
            message=message,
            facts=facts,
        )

        return {
            "handled": True,
            "answer": answer,
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
            facts = build_booking_facts(
                message=message,
                variables=variables,
                stage=variables["booking_stage"],
                missing_variables=missing_details,
            )

            answer = llm_compose_booking_reply(
                assistant=assistant,
                message=message,
                facts=facts,
            )

            return {
                "handled": True,
                "answer": answer,
                "variables": variables,
                "active_subagent": BOOKING_AGENT_NAME,
                "booking_stage": variables["booking_stage"],
                "action_required": None,
                "missing_variables": missing_details,
                "recommended_next_action": "collect_booking_confirmation_details",
                "reason": "Still missing booking confirmation details.",
            }

        variables["booking_stage"] = "booking_create_ready"
        action_required = action_create_booking(variables)

        facts = build_booking_facts(
            message=message,
            variables=variables,
            stage=variables["booking_stage"],
            action_required=action_required,
        )

        answer = llm_compose_booking_reply(
            assistant=assistant,
            message=message,
            facts=facts,
        )

        return {
            "handled": True,
            "answer": answer,
            "variables": variables,
            "active_subagent": BOOKING_AGENT_NAME,
            "booking_stage": variables["booking_stage"],
            "action_required": action_required,
            "missing_variables": [],
            "recommended_next_action": "create_booking",
            "reason": "Booking confirmation details complete; create booking required.",
        }

    variables = collect_slot_variables(message, variables)
    missing = missing_slot_fields(variables)

    if missing:
        variables["booking_stage"] = "booking_collecting_slot"

        facts = build_booking_facts(
            message=message,
            variables=variables,
            stage=variables["booking_stage"],
            missing_variables=missing,
        )

        answer = llm_compose_booking_reply(
            assistant=assistant,
            message=message,
            facts=facts,
        )

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
    action_required = action_check_availability(variables)

    return {
        "handled": True,
        "answer": "",
        "variables": variables,
        "active_subagent": BOOKING_AGENT_NAME,
        "booking_stage": variables["booking_stage"],
        "action_required": action_required,
        "missing_variables": [],
        "recommended_next_action": "check_availability",
        "reason": "Booking slot fields complete; availability check required.",
    }
