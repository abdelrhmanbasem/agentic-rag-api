# app/structured_inventory.py
# Structured inventory index for ultra-low-token / low-embedding lookups.
#
# Fixes:
# - Auto-creates /app/data.
# - Can rebuild structured inventory from Qdrant/RAG knowledge results when local file is missing.
# - Makes first-turn inventory lookup recover after restart/redeploy.
# - Searches by query variables and raw Arabic/English query.

import json
import os
import re
from typing import Dict, Any, List, Optional


DATA_DIR = os.getenv("STRUCTURED_DATA_DIR", "/app/data")
INVENTORY_PATH = os.path.join(DATA_DIR, "structured_inventory.json")


def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def load_inventory() -> List[Dict[str, Any]]:
    ensure_data_dir()

    if not os.path.exists(INVENTORY_PATH):
        return []

    try:
        with open(INVENTORY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        return []


def save_inventory(items: List[Dict[str, Any]]):
    ensure_data_dir()

    with open(INVENTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


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


def detect_brand(text: str) -> Optional[str]:
    normalized = normalize_text(text)

    brand_map = {
        "bmw": "BMW",
        "بي ام": "BMW",
        "بي ام دبليو": "BMW",
        "mercedes": "Mercedes",
        "مرسيدس": "Mercedes",
        "hyundai": "Hyundai",
        "هيونداي": "Hyundai",
        "toyota": "Toyota",
        "تويوتا": "Toyota",
        "kia": "Kia",
        "كيا": "Kia",
        "nissan": "Nissan",
        "نيسان": "Nissan",
        "audi": "Audi",
        "اودي": "Audi",
        "ford": "Ford",
        "فورد": "Ford",
        "tesla": "Tesla",
        "تسلا": "Tesla",
    }

    for raw, brand in brand_map.items():
        if raw in normalized:
            return brand

    return None


def detect_condition(text: str) -> Optional[str]:
    normalized = normalize_text(text)

    if any(x in normalized for x in ["used", "preowned", "pre-owned", "مستعمل", "مستعمله", "مستعملة", "استعمال"]):
        return "used"

    if any(x in normalized for x in ["new", "brand new", "زيرو", "جديد", "جديده", "جديدة"]):
        return "new"

    return None


def detect_transmission(text: str) -> Optional[str]:
    normalized = normalize_text(text)

    if any(x in normalized for x in ["automatic", "اوتوماتيك", "أوتوماتيك", "اتوماتيك", "اوتو"]):
        return "automatic"

    if any(x in normalized for x in ["manual", "مانيوال", "عادي"]):
        return "manual"

    return None


def detect_brand_from_model(model: str) -> Optional[str]:
    model = model or ""
    first = model.split()[0] if model.split() else ""
    return detect_brand(first) or first or None


def parse_inventory_items(
    assistant_id: str,
    document_id: str,
    title: str,
    text: str,
    metadata: Dict[str, Any],
) -> List[Dict[str, Any]]:
    text = text or ""
    metadata = metadata or {}

    chunks = re.split(r"[\n\.]+", text)
    items: List[Dict[str, Any]] = []

    model_pattern = re.compile(
        r"\b(BMW\s+\w+|Mercedes\s+\w+|Hyundai\s+\w+|Toyota\s+\w+|Kia\s+\w+|Nissan\s+\w+|Audi\s+\w+|Ford\s+\w+|Tesla\s+\w+)\b",
        re.IGNORECASE,
    )

    for chunk in chunks:
        line = chunk.strip()
        if not line:
            continue

        lower = line.lower()

        inventory_signal = any(
            marker in lower
            for marker in [
                "available",
                "egp",
                "price",
                "km",
                "متاح",
                "متاحة",
                "سعر",
                "جنيه",
                "كيلو",
            ]
        )

        if not inventory_signal:
            continue

        model_match = model_pattern.search(line)
        if not model_match:
            continue

        model = model_match.group(1).strip()
        brand = detect_brand(line) or detect_brand_from_model(model)

        year = None
        year_match = re.search(r"\b(20\d{2})\b", line)
        if year_match:
            year = year_match.group(1)

        km = None
        km_match = re.search(r"(\d{2,6})\s*(?:km|كيلو)", line, re.IGNORECASE)
        if km_match:
            try:
                km = int(km_match.group(1))
            except Exception:
                km = None

        price = None
        price_match = re.search(r"(\d{5,})\s*EGP", line, re.IGNORECASE)
        if price_match:
            try:
                price = int(price_match.group(1))
            except Exception:
                price = None

        if price is None:
            price_match_ar = re.search(r"(\d{5,})\s*جنيه", line, re.IGNORECASE)
            if price_match_ar:
                try:
                    price = int(price_match_ar.group(1))
                except Exception:
                    price = None

        item = {
            "assistant_id": assistant_id,
            "document_id": document_id,
            "title": title,
            "text": line,
            "raw_text": line,
            "search_text": normalize_text(line),
            "metadata": metadata,
            "type": "car",
            "brand": brand,
            "model": model,
            "year": year,
            "km": km,
            "price": price,
            "currency": "EGP" if price else metadata.get("currency", "EGP"),
            "transmission": detect_transmission(line),
            "condition": detect_condition(line) or metadata.get("condition"),
        }

        items.append(item)

    return items


def upsert_structured_inventory_from_text(
    assistant_id: str,
    document_id: str,
    title: str,
    text: str,
    metadata: Dict[str, Any],
) -> int:
    metadata = metadata or {}
    title_l = (title or "").lower()
    doc_l = (document_id or "").lower()
    meta_type = str(metadata.get("type", "")).lower()
    text_l = (text or "").lower()

    should_index = (
        meta_type == "inventory"
        or "inventory" in title_l
        or "inventory" in doc_l
        or "available for" in text_l
        or "egp" in text_l
    )

    if not should_index:
        return 0

    existing = load_inventory()

    existing = [
        item
        for item in existing
        if not (
            item.get("assistant_id") == assistant_id
            and item.get("document_id") == document_id
        )
    ]

    parsed = parse_inventory_items(
        assistant_id=assistant_id,
        document_id=document_id,
        title=title,
        text=text,
        metadata=metadata,
    )

    save_inventory(existing + parsed)

    return len(parsed)


def rebuild_structured_inventory_from_knowledge(
    assistant_id: str,
    knowledge: List[Dict[str, Any]],
) -> int:
    """
    Rebuild local structured_inventory.json from retrieved RAG/Qdrant results.

    This is a recovery path for restarts/redeploys where /app/data is not persisted.
    It lets the system fall back to Qdrant once, rebuild the local structured file,
    then continue using structured_inventory afterward.
    """
    if not knowledge:
        return 0

    existing = load_inventory()
    parsed_all: List[Dict[str, Any]] = []

    for idx, item in enumerate(knowledge):
        text = item.get("text") or ""
        metadata = item.get("metadata") or {}

        document_id = item.get("document_id") or metadata.get("document_id") or "inventory_rebuilt"
        title = item.get("title") or metadata.get("title") or "Rebuilt Inventory"

        parsed = parse_inventory_items(
            assistant_id=assistant_id,
            document_id=document_id,
            title=title,
            text=text,
            metadata={
                **metadata,
                "source": metadata.get("source") or "rebuilt_from_rag",
                "rebuilt": True,
            },
        )

        parsed_all.extend(parsed)

    if not parsed_all:
        return 0

    # Remove duplicates for this assistant/model/year/price.
    combined = [
        item
        for item in existing
        if item.get("assistant_id") != assistant_id
    ]

    seen = set()

    for item in parsed_all:
        key = (
            item.get("assistant_id"),
            item.get("model"),
            item.get("year"),
            item.get("price"),
        )

        if key in seen:
            continue

        seen.add(key)
        combined.append(item)

    save_inventory(combined)

    return len(parsed_all)


def extract_query_hints(raw_query: str) -> Dict[str, Any]:
    raw_query = raw_query or ""
    hints: Dict[str, Any] = {}

    brand = detect_brand(raw_query)
    condition = detect_condition(raw_query)
    transmission = detect_transmission(raw_query)

    if brand:
        hints["car_brand"] = brand

    if condition:
        hints["car_condition"] = condition

    if transmission:
        hints["transmission"] = transmission

    text = normalize_text(raw_query)

    if "مليون" in text:
        match = re.search(r"(\d+(?:\.\d+)?)?\s*مليون", text)
        if match:
            value = float(match.group(1) or 1)
            hints["budget_max"] = int(value * 1000000)

    elif any(x in text for x in ["budget", "ميزاني", "لحد", "حد اقصي", "حد اقصى", "الف", "ألف"]):
        match = re.search(r"(\d+(?:\.\d+)?)\s*(k|thousand|الف|ألف)?", text)
        if match:
            value = float(match.group(1))
            unit = match.group(2)
            if unit in ["k", "thousand", "الف", "ألف"]:
                value *= 1000
            hints["budget_max"] = int(value)

    return hints


def score_inventory_item(item: Dict[str, Any], query_variables: Dict[str, Any], raw_query: str = "") -> float:
    score = 0.0
    query_variables = query_variables or {}
    raw_query_norm = normalize_text(raw_query)

    merged_query = dict(extract_query_hints(raw_query))
    merged_query.update({k: v for k, v in query_variables.items() if v not in [None, ""]})

    q_brand = str(merged_query.get("car_brand") or "").lower()
    q_condition = str(merged_query.get("car_condition") or "").lower()
    q_transmission = str(merged_query.get("transmission") or "").lower()
    q_budget = merged_query.get("budget_max")

    item_brand = str(item.get("brand") or "").lower()
    item_condition = str(item.get("condition") or "").lower()
    item_transmission = str(item.get("transmission") or "").lower()
    item_price = item.get("price")
    item_model = str(item.get("model") or "").lower()
    item_text = normalize_text(item.get("text") or "")

    if q_brand:
        if q_brand == item_brand:
            score += 120
        elif q_brand in item_model or q_brand in item_text:
            score += 80
        else:
            score -= 80

    if raw_query_norm:
        for token in raw_query_norm.split():
            if len(token) >= 3 and token in item_text:
                score += 5

    if q_condition and item_condition:
        score += 25 if q_condition == item_condition else -10

    if q_transmission:
        score += 35 if q_transmission == item_transmission else -15

    if q_budget and item_price:
        try:
            score += 80 if int(item_price) <= int(q_budget) else -80
        except Exception:
            pass

    if item_price:
        score += 8

    if item.get("km"):
        score += 5

    if item.get("year"):
        score += 4

    if item.get("model"):
        score += 4

    return score


def search_structured_inventory(
    assistant_id: str,
    query_variables: Dict[str, Any],
    limit: int = 4,
    raw_query: str = "",
) -> List[Dict[str, Any]]:
    items = [
        item
        for item in load_inventory()
        if item.get("assistant_id") == assistant_id
    ]

    if not items:
        return []

    scored = []

    for item in items:
        score = score_inventory_item(item, query_variables or {}, raw_query=raw_query)
        enriched = dict(item)
        enriched["score"] = score

        if score > 0:
            scored.append(enriched)

    scored.sort(key=lambda x: x.get("score", 0), reverse=True)

    return scored[:limit]


def inventory_items_to_knowledge(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    knowledge = []

    for idx, item in enumerate(items):
        knowledge.append(
            {
                "assistant_id": item.get("assistant_id"),
                "document_id": item.get("document_id"),
                "title": item.get("title") or "Structured Inventory",
                "chunk_index": idx,
                "text": item.get("text") or "",
                "metadata": {
                    **(item.get("metadata") or {}),
                    "source": "structured_inventory",
                    "type": item.get("type"),
                    "brand": item.get("brand"),
                    "model": item.get("model"),
                    "year": item.get("year"),
                    "km": item.get("km"),
                    "price": item.get("price"),
                    "currency": item.get("currency"),
                    "transmission": item.get("transmission"),
                    "condition": item.get("condition"),
                },
                "score": item.get("score", 0),
            }
        )

    return knowledge
