# app/entry_path.py
# Universal first-turn / early-turn entry path.
# Goal:
# - Handle obvious business-intent messages before GPT router/extraction/generation.
# - Keep the agent smart and fluid while minimizing tokens.
# - Works for this assistant and future assistants through schema/workflow detection.
# - Uses conversation_brain.py to make deterministic responses feel intelligent and human-like.
# - IMPORTANT: entry_path should NOT write long-term memory by default.

import re
from typing import Dict, Any, Optional, List

from app.conversation_brain import (
    compose_car_entry_answer,
    compose_service_entry_answer,
)


def is_arabic_text(text: str) -> bool:
    return bool(re.search(r"[\u0600-\u06FF]", text or ""))


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


def schema_has_any(schema: Dict[str, Any], keys: List[str]) -> bool:
    schema = schema or {}
    return any(key in schema for key in keys)


def infer_workflow_type(schema: Dict[str, Any], assistant_id: str = "") -> str:
    schema = schema or {}
    assistant_id = (assistant_id or "").lower()

    car_keys = [
        "car_brand",
        "car_condition",
        "transmission",
        "budget_max",
        "matched_car_model",
        "matched_car_year",
        "matched_car_km",
        "matched_car_price",
        "preferred_viewing_date",
        "preferred_viewing_time",
    ]

    service_keys = [
        "service_needed",
        "appointment_date",
        "appointment_time",
        "doctor_preference",
        "patient_name",
        "insurance_provider",
    ]

    real_estate_keys = [
        "property_type",
        "property_location",
        "bedrooms",
        "bathrooms",
        "rent_budget",
        "purchase_budget",
        "viewing_date",
        "viewing_time",
    ]

    ecommerce_keys = [
        "product_name",
        "product_category",
        "order_id",
        "shipping_address",
        "delivery_date",
    ]

    if schema_has_any(schema, car_keys) or any(
        marker in assistant_id
        for marker in ["car", "cars", "auto", "vehicle", "dealer"]
    ):
        return "car_sales"

    if schema_has_any(schema, service_keys) or any(
        marker in assistant_id
        for marker in ["clinic", "doctor", "medical", "dental", "dentist", "health", "salon", "spa"]
    ):
        return "service_booking"

    if schema_has_any(schema, real_estate_keys) or any(
        marker in assistant_id
        for marker in ["real_estate", "property", "rent", "broker"]
    ):
        return "real_estate"

    if schema_has_any(schema, ecommerce_keys) or any(
        marker in assistant_id
        for marker in ["shop", "store", "ecommerce", "order"]
    ):
        return "ecommerce"

    return "general"


def schema_has(schema: Dict[str, Any], key: str) -> bool:
    return key in (schema or {})


def extract_budget(text: str) -> Optional[int]:
    normalized = normalize_text(text)

    budget_words = [
        "budget",
        "under",
        "up to",
        "price",
        "cost",
        "million",
        "egp",
        "ميزانيه",
        "ميزانية",
        "لحد",
        "حدي",
        "حد اقصي",
        "حد اقصى",
        "مليون",
        "الف",
        "ألف",
        "جنيه",
        "سعر",
        "بكام",
    ]

    if not any(word in normalized for word in budget_words):
        return None

    match = re.search(
        r"(\d+(?:\.\d+)?)\s*(million|m|k|thousand|مليون|الف|ألف)?",
        normalized,
    )

    if not match:
        return None

    value = float(match.group(1))
    unit = match.group(2)

    if unit in ["million", "m", "مليون"]:
        value *= 1000000
    elif unit in ["k", "thousand", "الف", "ألف"]:
        value *= 1000

    return int(value)


def detect_car_brand(text: str) -> Optional[str]:
    normalized = normalize_text(text)

    brand_map = {
        "bmw": "BMW",
        "بي ام": "BMW",
        "بي ام دبليو": "BMW",
        "مرسيدس": "Mercedes",
        "mercedes": "Mercedes",
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

    for raw, canonical in brand_map.items():
        if raw in normalized:
            return canonical

    return None


def detect_car_condition(text: str) -> Optional[str]:
    normalized = normalize_text(text)

    used_words = [
        "used",
        "preowned",
        "pre-owned",
        "مستعمل",
        "مستعمله",
        "مستعملة",
        "استعمال",
        "كسر زيرو",
    ]

    new_words = [
        "new",
        "brand new",
        "زيرو",
        "جديد",
        "جديده",
        "جديدة",
    ]

    if any(word in normalized for word in used_words):
        return "used"

    if any(word in normalized for word in new_words):
        return "new"

    return None


def detect_transmission(text: str) -> Optional[str]:
    normalized = normalize_text(text)

    if any(word in normalized for word in ["automatic", "اوتوماتيك", "أوتوماتيك", "اتوماتيك", "اوتو"]):
        return "automatic"

    if any(word in normalized for word in ["manual", "مانيوال", "عادي"]):
        return "manual"

    return None


def detect_car_intent(text: str) -> bool:
    normalized = normalize_text(text)

    intent_words = [
        "car",
        "vehicle",
        "buy",
        "want",
        "looking for",
        "searching for",
        "available",
        "inventory",
        "عربيه",
        "عربية",
        "اشتري",
        "اشترى",
        "عايز",
        "عاوز",
        "حابب",
        "بدور",
        "موجود",
        "متاح",
        "فيه",
    ]

    brand = detect_car_brand(text)

    return bool(brand or any(word in normalized for word in intent_words))


def extract_car_variables(schema: Dict[str, Any], message: str) -> Dict[str, Any]:
    updates: Dict[str, Any] = {}

    brand = detect_car_brand(message)
    condition = detect_car_condition(message)
    transmission = detect_transmission(message)
    budget = extract_budget(message)

    if schema_has(schema, "intent"):
        updates["intent"] = "car_search"

    if brand and schema_has(schema, "car_brand"):
        updates["car_brand"] = brand

    if condition and schema_has(schema, "car_condition"):
        updates["car_condition"] = condition

    if transmission and schema_has(schema, "transmission"):
        updates["transmission"] = transmission

    if budget and schema_has(schema, "budget_max"):
        updates["budget_max"] = budget

    if budget and schema_has(schema, "currency"):
        updates["currency"] = "EGP"

    if schema_has(schema, "lead_stage"):
        updates["lead_stage"] = "qualified"

    return updates


def extract_service_needed(message: str) -> Optional[str]:
    normalized = normalize_text(message)

    service_map = {
        "cleaning": "cleaning",
        "teeth cleaning": "teeth cleaning",
        "dental": "dental",
        "dentist": "dentist",
        "consultation": "consultation",
        "checkup": "checkup",
        "check-up": "checkup",
        "orthodontics": "orthodontics",
        "xray": "x-ray",
        "x-ray": "x-ray",
        "blood test": "blood test",
        "dermatology": "dermatology",
        "cardiology": "cardiology",
        "تنظيف": "teeth cleaning",
        "تنضيف": "teeth cleaning",
        "اسنان": "dental",
        "سنان": "dental",
        "كشف": "consultation",
        "استشاره": "consultation",
        "استشارة": "consultation",
        "اشعه": "x-ray",
        "اشعة": "x-ray",
        "تحليل": "blood test",
        "تحاليل": "blood test",
        "جلديه": "dermatology",
        "جلدية": "dermatology",
        "قلب": "cardiology",
    }

    for raw, canonical in service_map.items():
        if raw in normalized:
            return canonical

    return None


def detect_service_intent(text: str) -> bool:
    normalized = normalize_text(text)

    intent_words = [
        "book",
        "appointment",
        "reservation",
        "schedule",
        "clinic",
        "doctor",
        "احجز",
        "حجز",
        "ميعاد",
        "موعد",
        "دكتور",
        "دكتوره",
        "عيادة",
        "عياده",
        "كشف",
    ]

    return bool(extract_service_needed(text) or any(word in normalized for word in intent_words))


def extract_service_variables(schema: Dict[str, Any], message: str) -> Dict[str, Any]:
    updates: Dict[str, Any] = {}

    service = extract_service_needed(message)

    if schema_has(schema, "intent"):
        updates["intent"] = "booking_request" if detect_service_intent(message) else "service_question"

    if service and schema_has(schema, "service_needed"):
        updates["service_needed"] = service

    if schema_has(schema, "lead_stage"):
        updates["lead_stage"] = "qualified"

    return updates


def extract_prices_from_text(text: str) -> List[int]:
    text = text or ""
    prices: List[int] = []

    for match in re.findall(r"(\d{5,})\s*EGP", text, re.IGNORECASE):
        try:
            prices.append(int(match))
        except Exception:
            pass

    for match in re.findall(r"(\d{5,})\s*جنيه", text, re.IGNORECASE):
        try:
            prices.append(int(match))
        except Exception:
            pass

    return prices


def choose_best_knowledge_for_variables(knowledge: List[Dict[str, Any]], variables: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not knowledge:
        return None

    variables = variables or {}
    brand = (variables.get("car_brand") or "").lower()
    budget = variables.get("budget_max")
    transmission = (variables.get("transmission") or "").lower()
    condition = (variables.get("car_condition") or "").lower()

    brand_aliases = {
        "bmw": ["bmw", "بي ام", "بي ام دبليو"],
        "mercedes": ["mercedes", "مرسيدس"],
        "hyundai": ["hyundai", "هيونداي"],
        "toyota": ["toyota", "تويوتا"],
        "kia": ["kia", "كيا"],
        "nissan": ["nissan", "نيسان"],
        "audi": ["audi", "اودي"],
    }

    def score_item(item: Dict[str, Any]) -> float:
        text = (item.get("text") or "").lower()
        score = 0.0

        if brand:
            aliases = brand_aliases.get(brand, [brand])
            score += 120 if any(alias in text for alias in aliases) else -70

        if budget:
            prices = extract_prices_from_text(text)
            if prices:
                best_price = min(prices)
                score += 90 if best_price <= int(budget) else -90

        if transmission:
            score += 30 if transmission in text else 0

        if condition:
            condition_words = {
                "used": ["used", "مستعملة", "مستعمله", "مستعمل"],
                "new": ["new", "brand new", "زيرو", "جديدة", "جديده"],
            }.get(condition, [condition])

            score += 20 if any(word in text for word in condition_words) else 0

        score += float(item.get("score", 0.0) or 0.0) * 5

        return score

    return sorted(knowledge, key=score_item, reverse=True)[0]


def extract_knowledge_facts(item: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not item:
        return {}

    text = item.get("text", "") or ""
    lower = text.lower()
    facts: Dict[str, Any] = {}

    model_match = re.search(
        r"\b(BMW\s+320i|Mercedes\s+C180|Hyundai\s+Tucson|Toyota\s+\w+|Kia\s+\w+|Nissan\s+\w+|Audi\s+\w+)\b",
        text,
        re.IGNORECASE,
    )
    if model_match:
        facts["matched_car_model"] = model_match.group(1)

    year_match = re.search(r"\b(20\d{2})\b", text)
    if year_match:
        facts["matched_car_year"] = year_match.group(1)

    km_match = re.search(r"(\d{2,6})\s*km", text, re.IGNORECASE)
    if km_match:
        try:
            facts["matched_car_km"] = int(km_match.group(1))
        except Exception:
            pass

    price_match = re.search(r"(\d{5,})\s*EGP", text, re.IGNORECASE)
    if price_match:
        try:
            facts["matched_car_price"] = int(price_match.group(1))
            facts["currency"] = "EGP"
        except Exception:
            pass

    if "automatic" in lower or "اوتوماتيك" in lower or "أوتوماتيك" in lower:
        facts["transmission"] = "automatic"

    if "manual" in lower or "مانيوال" in lower:
        facts["transmission"] = "manual"

    if "bmw" in lower or "بي ام" in lower:
        facts["car_brand"] = "BMW"
    elif "mercedes" in lower or "مرسيدس" in lower:
        facts["car_brand"] = "Mercedes"
    elif "hyundai" in lower or "هيونداي" in lower:
        facts["car_brand"] = "Hyundai"
    elif "toyota" in lower or "تويوتا" in lower:
        facts["car_brand"] = "Toyota"
    elif "kia" in lower or "كيا" in lower:
        facts["car_brand"] = "Kia"
    elif "nissan" in lower or "نيسان" in lower:
        facts["car_brand"] = "Nissan"
    elif "audi" in lower or "اودي" in lower:
        facts["car_brand"] = "Audi"

    if any(word in lower for word in ["used", "مستعملة", "مستعمله", "مستعمل"]):
        facts["car_condition"] = "used"

    if any(word in lower for word in ["brand new", "new", "زيرو", "جديدة", "جديده"]):
        facts["car_condition"] = "new"

    return facts


def build_selected_item(variables: Dict[str, Any]) -> Dict[str, Any]:
    if not variables.get("matched_car_model"):
        return {}

    return {
        "type": "car",
        "brand": variables.get("car_brand"),
        "model": variables.get("matched_car_model"),
        "year": variables.get("matched_car_year"),
        "km": variables.get("matched_car_km"),
        "price": variables.get("matched_car_price"),
        "currency": variables.get("currency") or "EGP",
        "transmission": variables.get("transmission"),
        "condition": variables.get("car_condition"),
    }


def build_car_entry_answer(variables: Dict[str, Any], message: str) -> str:
    return compose_car_entry_answer(
        variables=variables,
        message=message,
        recent_messages=None,
    )


def build_service_entry_answer(variables: Dict[str, Any], message: str) -> str:
    return compose_service_entry_answer(
        variables=variables,
        message=message,
        recent_messages=None,
    )


def should_try_entry_path(message: str, schema: Dict[str, Any], variables: Dict[str, Any], assistant_id: str = "") -> bool:
    variables = variables or {}
    workflow = infer_workflow_type(schema, assistant_id)

    has_selected_state = bool(
        variables.get("selected_item")
        or variables.get("matched_car_model")
        or variables.get("workflow_stage") in ["matched_inventory", "viewing_requested", "confirmed"]
    )

    if has_selected_state:
        return False

    if workflow == "car_sales":
        return detect_car_intent(message)

    if workflow == "service_booking":
        return detect_service_intent(message)

    return False


def build_entry_path_response(
    *,
    message: str,
    schema: Dict[str, Any],
    assistant_id: str,
    variables: Dict[str, Any],
    knowledge: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Builds deterministic first-turn answer after caller provides knowledge.

    IMPORTANT:
    Entry path returns skip_summary=True and skip_memory=True by default.
    This keeps first-turn obvious business intents extremely cheap for all assistants.
    """
    variables = dict(variables or {})
    workflow = infer_workflow_type(schema, assistant_id)

    if workflow == "car_sales":
        if not detect_car_intent(message):
            return None

        updates = extract_car_variables(schema, message)
        variables.update(updates)

        best_item = choose_best_knowledge_for_variables(knowledge, variables)
        facts = extract_knowledge_facts(best_item)
        variables.update(facts)

        selected_item = build_selected_item(variables)
        if selected_item:
            variables["selected_item"] = selected_item

        variables["intent"] = "car_search"
        variables["workflow"] = "car_sales"

        if variables.get("matched_car_model"):
            variables["workflow_stage"] = "matched_inventory"
        else:
            variables["workflow_stage"] = "qualified_interest"

        if schema_has(schema, "lead_stage"):
            variables["lead_stage"] = "qualified"

        answer = build_car_entry_answer(variables, message)

        return {
            "answer": answer,
            "updates": variables,
            "model_tier": "entry_path",
            "action": "entry_car_search",
            "knowledge_used": knowledge,
            "knowledge_source": "qdrant" if knowledge else "none",
            "skip_summary": True,
            "skip_memory": True,
        }

    if workflow == "service_booking":
        if not detect_service_intent(message):
            return None

        updates = extract_service_variables(schema, message)
        variables.update(updates)

        variables["workflow"] = "service_booking"
        variables["workflow_stage"] = "service_identified" if variables.get("service_needed") else "new_lead"

        answer = build_service_entry_answer(variables, message)

        return {
            "answer": answer,
            "updates": variables,
            "model_tier": "entry_path",
            "action": "entry_service_booking",
            "knowledge_used": knowledge,
            "knowledge_source": "qdrant" if knowledge else "none",
            "skip_summary": True,
            "skip_memory": True,
        }

    return None
