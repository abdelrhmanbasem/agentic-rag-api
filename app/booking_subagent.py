# app/booking_subagent.py
# Pure business logic for resolving Egyptian locations and calculating nearest branches.
# Stripped of all conversational/LLM logic (handled by LangGraph).

import math
import re
from typing import Any, Dict, List, Optional, Tuple

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

# (Keep all your excellent AREA_COORDINATES and AREA_ALIASES dictionaries here exactly as you wrote them)
# ... [Paste your existing AREA_COORDINATES and AREA_ALIASES dicts here] ...

def _norm(value: Any) -> str:
    return str(value).strip() if value is not None else ""

def _lower(value: Any) -> str:
    return _norm(value).lower()

def _arabic_digits_to_latin(text: str) -> str:
    result = str(text)
    for i, digit in enumerate("٠١٢٣٤٥٦٧٨٩"): result = result.replace(digit, str(i))
    for i, digit in enumerate("۰۱۲۳۴۵۶۷۸۹"): result = result.replace(digit, str(i))
    return result

def _clean_location_text(text: str) -> str:
    text = _lower(_arabic_digits_to_latin(text))
    text = text.replace("-", " ").replace("_", " ")
    text = re.sub(r"[،,.;:()\[\]{}!?؟]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def _distance_km(coord_a: Tuple[float, float], coord_b: Tuple[float, float]) -> float:
    lat1, lon1 = coord_a
    lat2, lon2 = coord_b
    radius = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    return radius * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))

def branch_display_name(branch: str) -> str:
    return BRANCH_DISPLAY_NAMES.get(_norm(branch), branch or "الفرع")

def extract_user_area(message: str) -> Optional[str]:
    # Placeholder: Assuming AREA_ALIASES is pasted above
    text = _clean_location_text(message)
    sorted_aliases = sorted(AREA_ALIASES.items(), key=lambda item: len(item[0]), reverse=True)
    for alias, area in sorted_aliases:
        alias_clean = _clean_location_text(alias)
        if alias_clean and re.search(r"(?<!\w)" + re.escape(alias_clean) + r"(?!\w)", text):
            return area
    return None

def recommend_nearest_branches_for_area(area: str, limit: int = 2) -> List[Dict[str, Any]]:
    # Placeholder: Assuming AREA_COORDINATES is pasted above
    area_coord = AREA_COORDINATES.get(_norm(area))
    if not area_coord: return []

    ranked = [
        {
            "branch": branch,
            "branch_ar": branch_display_name(branch),
            "distance_km_estimate": round(_distance_km(area_coord, branch_coord), 1),
        }
        for branch, branch_coord in BRANCH_COORDINATES.items()
    ]
    ranked.sort(key=lambda item: item["distance_km_estimate"])
    return ranked[:limit]
