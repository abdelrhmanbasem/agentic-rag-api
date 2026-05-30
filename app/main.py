# app/main.py
# Smart stateful Agentic RAG API
# Includes:
# - selected_item state
# - deterministic state answers
# - Arabic/Egyptian Arabic language guard
# - WhatsApp phone auto-fill
# - workflow stages
# - lead scoring
# - objection/repair/final confirmation handling
# - token_usage reporting
# - universal entry_path for obvious first-turn efficiency
# - structured inventory before vector RAG
# - structured inventory auto-rebuild from RAG if /app/data disappears
# - deterministic date/time/location extraction
# - early zero-token datetime scheduling path
# - universal assistant brain before fast_path
# - universal pre-router fast_path for simple follow-ups
# - smart escalation gate
# - advisory GPT fallback only when useful
# - workflow playbooks + assistant profiles

import re
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from app.config import (
    APP_SECRET,
    MOCK_MODE,
    RECENT_MESSAGES_LIMIT,
    KNOWLEDGE_TOP_K,
    MEMORY_TOP_K,
    RAG_CACHE_ENABLED,
    RAG_CACHE_MAX_AGE_MINUTES,
)
from app.db import (
    init_db,
    upsert_assistant,
    get_or_create_assistant,
    ensure_conversation,
    save_message,
    get_recent_messages,
    get_summary,
    save_schema,
    get_schema,
    get_variables,
    save_variables,
    upsert_knowledge_document,
    list_knowledge_documents,
    list_long_term_memories,
    save_rag_cache,
    get_rag_cache,
    log_estimated_usage,
)
from app.llm import chat_text, route_message, model_for_tier
from app.rag import (
    ensure_qdrant,
    ingest_document,
    search_knowledge,
    search_memories,
    compress_knowledge,
)
from app.variables import extract_variables, apply_variable_patch
from app.memory import update_conversation_summary, decide_and_write_long_term_memories
from app.policies import should_skip_generation, build_no_llm_answer
from app.token_usage import build_token_usage_report
from app.fast_path import should_try_fast_path, build_workflow_fast_answer
from app.entry_path import should_try_entry_path, build_entry_path_response, extract_car_variables
from app.structured_inventory import (
    upsert_structured_inventory_from_text,
    search_structured_inventory,
    inventory_items_to_knowledge,
    rebuild_structured_inventory_from_knowledge,
)
from app.super_efficiency import (
    deterministic_route_guess,
    should_update_summary,
    should_write_memory,
    compact_chat_response,
)
from app.smart_escalation import (
    should_escalate_to_advisor,
    build_advisor_route,
    build_advisor_system_prompt,
    build_advisor_context,
)
from app.datetime_location_extractor import (
    extract_datetime_location_patch,
    build_datetime_location_fast_response,
)
from app.playbooks import get_playbook, get_assistant_profile
from app.assistant_brain import (
    build_brain_deterministic_response,
    build_brain_advisor_hint,
)

from app.premium_sales_orchestrator import run_adaptive_premium_turn
from app.booking_subagent import run_booking_subagent


app = FastAPI(title="Agentic RAG API")


class AssistantRequest(BaseModel):
    assistant_id: str
    name: str
    system_prompt: str
    tone: str = "clear, helpful, concise"
    memory_policy: str = "Remember stable preferences, goals, constraints, and important facts."


class SchemaRequest(BaseModel):
    assistant_id: str
    schema: Dict[str, Any]


class IngestRequest(BaseModel):
    assistant_id: str
    document_id: str
    title: str
    text: str
    metadata: Dict[str, Any] = {}


class KnowledgeSearchRequest(BaseModel):
    assistant_id: str
    query: str
    limit: int = 4


class ChatRequest(BaseModel):
    assistant_id: str
    user_id: str
    conversation_id: str
    message: str
    channel: str = "n8n"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    tool_result: Optional[Dict[str, Any]] = None

class PatchVariablesRequest(BaseModel):
    assistant_id: str
    user_id: str
    conversation_id: str
    updates: Dict[str, Any] = {}
    deletions: List[str] = []


def check_auth(x_api_key: str):
    if x_api_key != APP_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")


def normalize_intent(intent: str) -> str:
    intent = (intent or "general_question").strip().lower()

    aliases = {
        "car_inquiry": "car_search",
        "car_enquiry": "car_search",
        "car_purchase": "car_search",
        "car_buying": "car_search",
        "buy_car": "car_search",
        "vehicle_search": "car_search",
        "vehicle_inquiry": "car_search",
        "vehicle_enquiry": "car_search",
        "vehicle_purchase": "car_search",
        "inventory_question": "car_search",
        "product_search": "car_search",
        "product_inquiry": "car_search",
        "schedule_viewing": "viewing_request",
        "book_viewing": "viewing_request",
        "viewing": "viewing_request",
        "viewing_inquiry": "viewing_request",
        "test_drive": "viewing_request",
        "schedule_test_drive": "viewing_request",
        "visit_request": "viewing_request",
        "appointment": "booking_request",
        "appointment_request": "booking_request",
        "schedule_appointment": "booking_request",
        "book_appointment": "booking_request",
        "clinic_booking": "booking_request",
        "reservation": "booking_request",
        "handoff": "human_handoff",
        "human": "human_handoff",
        "agent_request": "human_handoff",
        "talk_to_human": "human_handoff",
        "complain": "complaint",
        "customer_complaint": "complaint",
        "emergency": "urgent_medical_issue",
        "urgent": "urgent_medical_issue",
        "urgent_case": "urgent_medical_issue",
        "medical_emergency": "urgent_medical_issue",
        "price_objection": "objection",
        "budget_objection": "objection",
        "too_expensive": "objection",
    }

    return aliases.get(intent, intent)


def _normalize_text_for_stage(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        try:
            return str(value)
        except Exception:
            return ""
    return str(value).strip().lower()


def _append_unique_list(values: Any, item: str) -> list:
    result = values if isinstance(values, list) else []
    if item and item not in result:
        result.append(item)
    return result


def _extract_last_assistant_question(recent_messages: list) -> str:
    """
    Returns the most recent assistant question if available.
    Works generically for any assistant.
    """
    for msg in reversed(recent_messages or []):
        role = msg.get("role") if isinstance(msg, dict) else None
        content = msg.get("content") if isinstance(msg, dict) else None

        if role == "assistant" and content:
            text = str(content).strip()
            if "?" in text or "؟" in text:
                return text

    return ""


def _looks_like_user_answered_question(message: str) -> bool:
    """
    Generic signal that the user is answering a previous question.
    Works for Arabic/English without domain-specific logic.
    """
    text = _normalize_text_for_stage(message)

    answer_markers = [
        "yes",
        "no",
        "yeah",
        "yep",
        "nope",
        "اه",
        "أه",
        "ايوه",
        "أيوه",
        "لا",
        "تمام",
        "معظم",
        "غالبا",
        "غالبًا",
        "في ",
        "مفيش",
        "مافيش",
        "مش",
        "بيحصل",
        "بتحصل",
        "بيظهر",
        "بتظهر",
        "بيطلع",
        "بيعلى",
        "عايز",
        "محتاج",
        "اختار",
        "احجز",
    ]

    if any(marker in text for marker in answer_markers):
        return True

    # Short user replies after an assistant question are often answers.
    if 1 <= len(text.split()) <= 12:
        return True

    return False


def apply_conversation_stage_governor(
    message: str,
    variables: Dict[str, Any],
    recent_messages: list,
    assistant_id: str,
) -> Dict[str, Any]:
    """
    Universal state governor for all assistants.

    It does NOT contain domain-specific service/car/medical logic.
    It prevents repeated questions and helps the LLM know where the conversation is.

    Generic stages:
    - discovery: assistant is still understanding the user's need
    - awaiting_answer: assistant asked a question and is waiting for user reply
    - qualified: user answered enough to move forward
    - action_ready: enough info exists for workflow/action
    """

    variables = dict(variables or {})
    text = _normalize_text_for_stage(message)

    known_facts = variables.get("known_facts")
    if not isinstance(known_facts, list):
        known_facts = []

    answered_questions = variables.get("answered_questions")
    if not isinstance(answered_questions, list):
        answered_questions = []

    do_not_repeat = variables.get("do_not_repeat")
    if not isinstance(do_not_repeat, list):
        do_not_repeat = []

    last_question = variables.get("last_assistant_question") or _extract_last_assistant_question(recent_messages)

    user_answered = bool(last_question and _looks_like_user_answered_question(message))

    if last_question:
        variables["last_assistant_question"] = last_question

    if user_answered:
        variables["conversation_stage"] = "qualified"
        variables["last_user_answer"] = message
        variables["answered_questions"] = _append_unique_list(answered_questions, last_question)

        # Store a generic fact from the user answer.
        known_facts = _append_unique_list(known_facts, message)
        variables["known_facts"] = known_facts

        # Tell GPT not to ask the same question again.
        do_not_repeat = _append_unique_list(do_not_repeat, last_question)
        variables["do_not_repeat"] = do_not_repeat

        variables["next_conversation_action"] = "consult_playbook_and_move_forward"
        variables["_stage_instruction"] = (
            "The user answered the previous assistant question. "
            "Treat known_facts and last_user_answer as authoritative context. "
            "Do not ask the same question again. "
            "Do not continue discovery unless the user introduces a new issue or the playbook says more information is required. "
            "Use the assistant playbook to decide the next business step. "
            "Summarize what is known briefly, then move forward."
        )

    else:
        # If no previous question was answered, keep discovery/action flow.
        variables.setdefault("conversation_stage", "discovery")
        variables.setdefault("next_conversation_action", "ask_or_answer")

        variables["_stage_instruction"] = (
            "Use known_facts, recent conversation, and the assistant playbook before asking anything. "
            "Do not repeat questions listed in do_not_repeat or answered_questions. "
            "Ask at most one useful question only if the playbook says discovery is still needed. "
            "If enough context exists, move to the next playbook step."
        )

    # Generic anti-loop instruction, useful for every future assistant.
    variables["_do_not_repeat_instruction"] = (
        "Never ask again about information already present in known_facts, "
        "last_user_answer, issue_description, summary, answered_questions, or do_not_repeat. "
        "If enough context exists, move forward instead of continuing discovery."
    )

    return variables


def infer_workflow_type(schema: Dict[str, Any], assistant_id: str = "") -> str:
    assistant_id = (assistant_id or "").lower()

    # Hard override:
    # This assistant is for car service diagnostics + visit booking.
    # Never classify it as car_sales.
    if assistant_id == "service_center_agentic_rag":
        return "service_booking"

    schema = schema or {}
    keys = set(schema.keys())

    car_sales_keys = {
        "car_brand",
        "car_model",
        "budget_max",
        "matched_car_model",
        "matched_car_price",
        "matched_car_year",
        "matched_car_km",
        "selected_item",
        "transmission",
        "car_condition",
    }

    service_booking_keys = {
        "service_needed",
        "issue_description",
        "symptoms",
        "recommended_section",
        "appointment_date",
        "appointment_time",
        "customer_full_name",
        "plate_digits",
        "visit_id",
        "slot_status",
        "booking_status",
        "location_branch",
    }

    medical_keys = {
        "doctor_name",
        "specialty",
        "insurance_provider",
        "appointment_type",
    }

    if assistant_id in {"car_sales_demo", "car_sales", "cars", "auto_sales"}:
        return "car_sales"

    if "service" in assistant_id or "maintenance" in assistant_id or "repair" in assistant_id:
        return "service_booking"

    if "medical" in assistant_id or "clinic" in assistant_id or "doctor" in assistant_id:
        return "medical_booking"

    if keys & service_booking_keys:
        return "service_booking"

    if keys & car_sales_keys:
        return "car_sales"

    if keys & medical_keys:
        return "medical_booking"

    return "generic"


def normalize_intent_for_schema(intent: str, schema: Dict[str, Any], assistant_id: str = "") -> str:
    intent = normalize_intent(intent)
    workflow = infer_workflow_type(schema, assistant_id)

    if workflow == "car_sales" and intent == "booking_request":
        return "viewing_request"

    return intent


def calculate_missing_required_variables(
    schema: Dict[str, Any],
    variables: Dict[str, Any],
    intent: str = "general_question",
    assistant_id: str = "",
) -> List[str]:
    missing = []
    variables = variables or {}
    schema = schema or {}

    intent = normalize_intent_for_schema(
        intent,
        schema=schema,
        assistant_id=assistant_id,
    )

    for key, config in schema.items():
        if key == "intent" or not isinstance(config, dict):
            continue

        value = variables.get(key)
        if value is not None and value != "":
            continue

        globally_required = config.get("required", False)
        required_for_intents = config.get("required_for_intents", []) or []
        not_required_for_intents = config.get("not_required_for_intents", []) or []

        required_for_intents = [
            normalize_intent_for_schema(i, schema=schema, assistant_id=assistant_id)
            for i in required_for_intents
        ]
        not_required_for_intents = [
            normalize_intent_for_schema(i, schema=schema, assistant_id=assistant_id)
            for i in not_required_for_intents
        ]

        if intent in not_required_for_intents:
            continue

        if globally_required or intent in required_for_intents:
            missing.append(key)

    return list(dict.fromkeys(missing))


def is_arabic_text(text: str) -> bool:
    return bool(re.search(r"[\u0600-\u06FF]", text or ""))


def looks_mostly_english(text: str) -> bool:
    text = text or ""

    arabic_chars = len(re.findall(r"[\u0600-\u06FF]", text))
    latin_chars = len(re.findall(r"[A-Za-z]", text))

    return latin_chars > arabic_chars * 2


def is_user_question(message: str) -> bool:
    text = (message or "").lower().strip()

    if not text:
        return False

    question_markers = [
        "?",
        "؟",
        "is ",
        "are ",
        "do ",
        "does ",
        "can ",
        "could ",
        "would ",
        "how ",
        "what ",
        "when ",
        "where ",
        "why ",
        "which ",
        "هي ",
        "هو ",
        "هل ",
        "ده ",
        "دي ",
        "دا ",
        "دة ",
        "فيه ",
        "في ",
        "ممكن ",
        "ينفع ",
        "كام",
        "قد ايه",
        "قد إيه",
        "فين",
        "امتى",
        "إمتى",
        "ايه",
        "إيه",
        "ليه",
        "ازاي",
        "إزاي",
        "بكام",
        "متاح",
        "متاحة",
        "موجود",
        "موجودة",
        "لسه موجود",
        "لسه موجودة",
        "عاملة كام",
        "عامله كام",
        "سعرها",
        "سعره",
        "لونها",
        "اوتوماتيك",
        "أوتوماتيك",
        "مانيوال",
    ]

    return any(marker in text for marker in question_markers)


def detect_reply_language_instruction(user_message: str) -> str:
    if is_arabic_text(user_message):
        return """
LANGUAGE RULE:
- The latest user message is Arabic or Egyptian Arabic.
- You MUST reply in natural Egyptian Arabic.
- Do NOT reply in English.
- Keep brand names, model names, doctor names, service names, and technical terms in English only when natural.
- Keep car brands and models like BMW, Mercedes, Hyundai, 320i, C180 in English if needed.
- Use Egyptian Arabic phrasing like: عربية، مستعملة، أوتوماتيك، سعرها، عاملة كام كيلو، تحب تشوفها؟
- Keep the answer friendly, clear, and short.
"""

    return """
LANGUAGE RULE:
- Reply in the same language as the latest user message.
- If the latest user message is English, reply in English.
- Keep the answer friendly, clear, and short.
"""


def enforce_reply_language(user_message: str, answer: str, model: str) -> str:
    if not is_arabic_text(user_message):
        return answer

    if not looks_mostly_english(answer):
        return answer

    rewrite_messages = [
        {
            "role": "system",
            "content": (
                "Rewrite the assistant answer into natural Egyptian Arabic. "
                "Do not add new facts. Do not remove important facts. "
                "Keep brand/model names like BMW, 320i, Mercedes, C180 in English. "
                "Keep prices, years, phone numbers, dates, and kilometers exactly the same. "
                "Use natural Egyptian Arabic phrasing. Return only the rewritten answer."
            ),
        },
        {"role": "user", "content": answer},
    ]

    return chat_text(model, rewrite_messages, max_tokens=300)


def extract_phone_from_whatsapp_user_id(user_id: str) -> Optional[str]:
    match = re.search(r"(\d{8,15})", user_id or "")
    return match.group(1) if match else None


def autofill_channel_context(variables: Dict[str, Any], req: ChatRequest) -> Dict[str, Any]:
    variables = dict(variables or {})
    channel = (req.channel or "").lower()

    if channel == "whatsapp" and not variables.get("phone_number"):
        phone = extract_phone_from_whatsapp_user_id(req.user_id)
        if phone:
            variables["phone_number"] = phone
            variables["phone_source"] = "whatsapp_user_id"

    if channel and not variables.get("channel"):
        variables["channel"] = channel

    return variables


def extract_prices_from_text(text: str) -> List[int]:
    text = (text or "").lower()
    prices = []

    for match in re.findall(r"(\d{5,})\s*egp", text):
        try:
            prices.append(int(match))
        except Exception:
            pass

    for match in re.findall(r"(\d{5,})\s*جنيه", text):
        try:
            prices.append(int(match))
        except Exception:
            pass

    return prices


def choose_best_knowledge_for_variables(knowledge: List[Dict[str, Any]], variables: Dict[str, Any]):
    if not knowledge:
        return None

    variables = variables or {}
    brand = (variables.get("car_brand") or "").lower()
    budget = variables.get("budget_max")

    brand_aliases = {
        "bmw": ["bmw", "بي ام", "بي ام دبليو"],
        "mercedes": ["mercedes", "مرسيدس"],
        "hyundai": ["hyundai", "هيونداي"],
        "toyota": ["toyota", "تويوتا"],
        "kia": ["kia", "كيا"],
        "nissan": ["nissan", "نيسان"],
        "audi": ["audi", "اودي"],
    }

    def score_item(item):
        text = (item.get("text") or "").lower()
        score = 0

        if brand:
            aliases = brand_aliases.get(brand, [brand])
            score += 120 if any(alias in text for alias in aliases) else -60

        if budget:
            prices = extract_prices_from_text(text)
            if prices:
                best_price = min(prices)
                score += 90 if best_price <= budget else -90

        if variables.get("transmission"):
            transmission = str(variables.get("transmission")).lower()
            if transmission in text:
                score += 30

        score += float(item.get("score", 0.0) or 0.0) * 5

        return score

    return sorted(knowledge, key=score_item, reverse=True)[0]


def extract_knowledge_facts(item):
    if not item:
        return {}

    text = item.get("text", "") or ""
    lower = text.lower()
    facts = {}

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


def enrich_variables_from_best_knowledge(variables, knowledge):
    variables = dict(variables or {})
    best_item = choose_best_knowledge_for_variables(knowledge, variables)

    if not best_item:
        return variables

    facts = extract_knowledge_facts(best_item)

    for key, value in facts.items():
        if value is not None and value != "":
            variables[key] = value

    return variables


def build_selected_item_from_variables(variables: Dict[str, Any]) -> Dict[str, Any]:
    variables = variables or {}
    item = dict(variables.get("selected_item") or {}) if isinstance(variables.get("selected_item"), dict) else {}

    if variables.get("matched_car_model"):
        item["type"] = "car"
        item["brand"] = variables.get("car_brand")
        item["model"] = variables.get("matched_car_model")
        item["year"] = variables.get("matched_car_year")
        item["km"] = variables.get("matched_car_km")
        item["price"] = variables.get("matched_car_price")
        item["currency"] = variables.get("currency") or "EGP"
        item["transmission"] = variables.get("transmission")
        item["condition"] = variables.get("car_condition")

    return item


def sync_selected_item_state(variables: Dict[str, Any]) -> Dict[str, Any]:
    variables = dict(variables or {})
    item = build_selected_item_from_variables(variables)

    if item:
        variables["selected_item"] = item

    return variables


def set_workflow_stage(schema: Dict[str, Any], assistant_id: str, variables: Dict[str, Any], intent: str) -> Dict[str, Any]:
    variables = dict(variables or {})
    workflow = infer_workflow_type(schema, assistant_id)
    intent = normalize_intent_for_schema(intent, schema, assistant_id)

    variables["workflow"] = workflow

    if workflow == "car_sales":
        if variables.get("lead_stage") == "confirmed":
            variables["workflow_stage"] = "confirmed"
        elif intent == "viewing_request" and variables.get("preferred_viewing_date") and variables.get("location") and variables.get("phone_number"):
            variables["workflow_stage"] = "viewing_details_collected"
        elif intent == "viewing_request":
            variables["workflow_stage"] = "viewing_requested"
        elif variables.get("matched_car_model") and variables.get("budget_max"):
            variables["workflow_stage"] = "budget_confirmed"
        elif variables.get("matched_car_model"):
            variables["workflow_stage"] = "matched_inventory"
        elif variables.get("car_brand") or variables.get("budget_max") or variables.get("car_condition"):
            variables["workflow_stage"] = "qualified_interest"
        else:
            variables["workflow_stage"] = "new_lead"

    elif workflow == "service_booking":
        if intent == "booking_request" and variables.get("appointment_date") and variables.get("appointment_time") and variables.get("phone_number"):
            variables["workflow_stage"] = "booking_details_collected"
        elif intent == "booking_request":
            variables["workflow_stage"] = "booking_requested"
        elif variables.get("service_needed"):
            variables["workflow_stage"] = "service_identified"
        else:
            variables["workflow_stage"] = "new_lead"

    else:
        variables["workflow_stage"] = intent or "general"

    return variables


def add_asked_question(variables: Dict[str, Any], question_key: str) -> Dict[str, Any]:
    variables = dict(variables or {})
    asked = variables.get("asked_questions")

    if not isinstance(asked, list):
        asked = []

    if question_key not in asked:
        asked.append(question_key)

    variables["asked_questions"] = asked[-20:]

    return variables


def compute_lead_score(variables: Dict[str, Any], intent: str) -> Dict[str, Any]:
    variables = dict(variables or {})
    intent = normalize_intent(intent)

    score = 0
    reasons = []

    workflow = variables.get("workflow") or "general"

    def add(points: int, reason: str):
        nonlocal score
        score += points
        reasons.append(reason)

    has_interest = bool(
        variables.get("car_brand")
        or variables.get("service_needed")
        or variables.get("selected_item")
        or variables.get("matched_car_model")
    )

    has_matched_item = bool(
        variables.get("selected_item")
        or variables.get("matched_car_model")
    )

    has_budget = bool(variables.get("budget_max"))
    has_date = bool(variables.get("preferred_viewing_date") or variables.get("appointment_date"))
    has_time = bool(variables.get("preferred_viewing_time") or variables.get("appointment_time"))
    has_location = bool(variables.get("location"))
    has_phone = bool(variables.get("phone_number"))

    is_action_intent = intent in ["viewing_request", "booking_request"]

    is_price_sensitive = bool(
        variables.get("price_sensitive")
        or variables.get("last_objection_type") == "price"
        or variables.get("handoff_reason") == "price_or_negotiation"
    )

    needs_human = bool(variables.get("needs_human"))

    if has_interest:
        add(20, "clear interest")

    if has_matched_item:
        add(25, "matched item/service")

    if has_budget:
        add(15, "budget provided")

    if has_date:
        add(15, "date provided")

    if has_time:
        add(10, "time provided")

    if has_location:
        add(10, "location provided")

    if has_phone:
        add(15, "contact available")

    if is_action_intent:
        add(20, "action requested")

    if is_price_sensitive:
        add(15, "price-sensitive but engaged")

    if needs_human:
        add(10, "human follow-up needed")

    # Smart floor rules:
    # A user who selected an item and objected to price is not cold.
    # They are commercially engaged and should be treated as warm.
    if workflow == "car_sales" and has_matched_item and is_price_sensitive:
        score = max(score, 60)
        if "price-sensitive qualified lead" not in reasons:
            reasons.append("price-sensitive qualified lead")

    if workflow == "car_sales" and has_matched_item and has_phone:
        score = max(score, 70)
        if "matched item with contact" not in reasons:
            reasons.append("matched item with contact")

    if workflow == "car_sales" and has_date and has_time and has_location:
        score = max(score, 80)
        if "viewing details collected" not in reasons:
            reasons.append("viewing details collected")

    if variables.get("lead_stage") in ["confirmed", "negotiation"]:
        score = max(score, 65)
        stage_reason = f"lead stage: {variables.get('lead_stage')}"
        if stage_reason not in reasons:
            reasons.append(stage_reason)

    score = min(score, 100)

    if score >= 80:
        temperature = "hot"
    elif score >= 50:
        temperature = "warm"
    else:
        temperature = "cold"

    variables["lead_score"] = score
    variables["lead_temperature"] = temperature
    variables["lead_score_reasons"] = list(dict.fromkeys(reasons))

    return variables


def should_force_rag_for_actionable_intent(intent: str, variables: Dict[str, Any], message: str) -> bool:
    intent = normalize_intent(intent)
    variables = variables or {}
    text = (message or "").lower()

    if intent == "car_search":
        return bool(
            variables.get("car_brand")
            or variables.get("budget_max")
            or variables.get("car_condition")
            or variables.get("transmission")
            or any(
                marker in text
                for marker in [
                    "car",
                    "vehicle",
                    "buy",
                    "bmw",
                    "mercedes",
                    "hyundai",
                    "toyota",
                    "kia",
                    "nissan",
                    "audi",
                    "عربية",
                    "عربيه",
                    "اشتري",
                    "اشترى",
                    "مستعملة",
                    "مستعمله",
                    "زيرو",
                    "بي ام",
                    "مرسيدس",
                    "هيونداي",
                    "تويوتا",
                    "كيا",
                    "نيسان",
                ]
            )
        )

    if intent == "service_question":
        return bool(
            variables.get("service_needed")
            or variables.get("doctor_preference")
            or variables.get("insurance_provider")
        )

    if intent == "booking_request":
        return bool(
            variables.get("service_needed")
            or variables.get("appointment_date")
            or variables.get("appointment_time")
            or variables.get("doctor_preference")
        )

    if intent == "insurance_question":
        return bool(variables.get("insurance_provider") or variables.get("service_needed"))

    if intent == "viewing_request":
        return True

    return False


def format_money_ar(amount, currency="EGP") -> str:
    try:
        formatted = f"{int(amount):,}"
    except Exception:
        formatted = str(amount)

    return f"{formatted} جنيه" if currency == "EGP" else f"{formatted} {currency}"


def format_money_en(amount, currency="EGP") -> str:
    try:
        formatted = f"{int(amount):,}"
    except Exception:
        formatted = str(amount)

    return f"{formatted} {currency}"


def deterministic_answer_from_state(
    user_message: str,
    variables: Dict[str, Any],
    recommended_next_action: str = "continue_conversation",
) -> Optional[str]:
    variables = variables or {}
    text = (user_message or "").lower().strip()
    arabic = is_arabic_text(user_message)

    selected_item = variables.get("selected_item") if isinstance(variables.get("selected_item"), dict) else {}

    transmission = selected_item.get("transmission") or variables.get("transmission")
    km = selected_item.get("km") or variables.get("matched_car_km")
    price = selected_item.get("price") or variables.get("matched_car_price")
    currency = selected_item.get("currency") or variables.get("currency") or "EGP"
    model = selected_item.get("model") or variables.get("matched_car_model") or variables.get("car_brand") or "العربية"
    budget = variables.get("budget_max")

    if any(marker in text for marker in ["thanks", "thank you", "شكرا", "شكرًا", "تسلم", "تمام شكرا", "تمام شكرًا"]):
        return "العفو، تحت أمرك في أي وقت." if arabic else "You’re welcome. I’m here if you need anything else."

    if any(marker in text for marker in ["كلموني", "عايز اكلم حد", "عايز حد يكلمني", "call me", "talk to human", "agent"]):
        return "تمام، هخلي حد من الفريق يتابع معاك." if arabic else "Sure, I’ll have someone from the team follow up with you."

    if any(marker in text for marker in ["automatic", "manual", "اوتوماتيك", "أوتوماتيك", "اتوماتيك", "مانيوال"]) and transmission:
        if arabic:
            if transmission == "automatic":
                return f"أيوه، {model} أوتوماتيك."
            if transmission == "manual":
                return f"{model} مانيوال."
            return f"{model} فتيسها {transmission}."

        if transmission == "automatic":
            return f"Yes, the {model} is automatic."
        if transmission == "manual":
            return f"The {model} is manual."
        return f"The {model} transmission is {transmission}."

    if any(marker in text for marker in ["km", "kilometers", "mileage", "كام كيلو", "عاملة كام", "عامله كام", "ماشية كام"]) and km:
        try:
            km_text = f"{int(km):,}"
        except Exception:
            km_text = str(km)

        if arabic:
            return f"{model} عاملة {km_text} كيلو."
        return f"The {model} has {km_text} km."

    if any(marker in text for marker in ["price", "cost", "how much", "بكام", "سعر", "سعرها", "سعره"]) and price:
        if arabic:
            return f"سعر {model} هو {format_money_ar(price, currency)}."
        return f"The {model} price is {format_money_en(price, currency)}."

    if any(marker in text for marker in ["budget", "under", "up to", "million", "ميزانيتي", "ميزانية", "الميزانية", "لحد", "مليون", "مناسب"]) and budget and price:
        try:
            fits_budget = int(price) <= int(budget)
        except Exception:
            fits_budget = None

        if fits_budget is True:
            if arabic:
                return f"أيوه، {model} مناسبة لميزانيتك. سعرها {format_money_ar(price, currency)}. تحب نرتب معاد تشوفها؟"
            return f"Yes, the {model} fits your budget. Its price is {format_money_en(price, currency)}. Would you like to schedule a viewing?"

        if fits_budget is False:
            if arabic:
                return f"{model} أعلى من ميزانيتك شوية. سعرها {format_money_ar(price, currency)}. تحب أقولك على بديل أقرب لميزانيتك؟"
            return f"The {model} is slightly above your budget. Its price is {format_money_en(price, currency)}. Would you like an alternative closer to your budget?"

    return None


def build_objection_answer(user_message: str, variables: Dict[str, Any]) -> Optional[str]:
    text = (user_message or "").lower()
    arabic = is_arabic_text(user_message)
    selected_item = variables.get("selected_item") if isinstance(variables.get("selected_item"), dict) else {}

    model = selected_item.get("model") or variables.get("matched_car_model") or variables.get("car_brand") or "العربية"
    price = selected_item.get("price") or variables.get("matched_car_price")
    currency = selected_item.get("currency") or variables.get("currency") or "EGP"

    if not any(
        marker in text
        for marker in [
            "غالية",
            "غالي",
            "كتير",
            "السعر عالي",
            "اقل",
            "أقل",
            "خصم",
            "تقسيط",
            "مش مناسب",
            "too expensive",
            "expensive",
            "discount",
            "installment",
            "installments",
        ]
    ):
        return None

    if arabic:
        if price:
            return f"فاهمك. سعر {model} الحالي {format_money_ar(price, currency)}. أقدر أقولك على بديل أقرب لميزانيتك، أو أخلي حد من الفريق يتابع معاك بخصوص التفاوض أو التقسيط."
        return "فاهمك. أقدر أقولك على بديل أقرب لميزانيتك، أو أخلي حد من الفريق يتابع معاك بخصوص السعر."

    if price:
        return f"I understand. The current price for {model} is {format_money_en(price, currency)}. I can suggest a closer alternative or have someone from the team follow up about negotiation or installments."

    return "I understand. I can suggest a closer alternative or have someone from the team follow up about the price."


def build_conversation_repair_answer(user_message: str, variables: Dict[str, Any]) -> Optional[str]:
    text = (user_message or "").lower().strip()
    arabic = is_arabic_text(user_message)

    if not any(
        marker in text
        for marker in ["مش قصدي", "مش ده قصدي", "انت مش فاهم", "مش فاهمني", "wrong", "not what i mean", "you misunderstood"]
    ):
        return None

    return (
        "تمام، حقك عليا. تقصد تعدل النوع/الميزانية، ولا تقصد عربية مختلفة تمامًا؟"
        if arabic
        else "Got it, sorry about that. Do you want to change the brand/budget, or are you looking for something completely different?"
    )


def build_final_confirmation(
    schema: Dict[str, Any],
    assistant_id: str,
    variables: Dict[str, Any],
    user_message: str,
) -> Optional[str]:
    workflow = infer_workflow_type(schema, assistant_id)
    variables = variables or {}
    arabic = is_arabic_text(user_message)

    if workflow == "car_sales":
        intent = normalize_intent_for_schema(variables.get("intent"), schema, assistant_id)

        if intent != "viewing_request":
            return None

        selected_item = variables.get("selected_item") if isinstance(variables.get("selected_item"), dict) else {}

        model = selected_item.get("model") or variables.get("matched_car_model") or variables.get("car_brand") or "العربية"
        date = variables.get("preferred_viewing_date")
        time = variables.get("preferred_viewing_time") or variables.get("appointment_time")
        location = variables.get("location")
        phone = variables.get("phone_number")

        if not date or not location or not phone:
            return None

        if arabic:
            if time:
                return f"تمام، كده طلب المعاينة لـ {model} يوم {date} الساعة {time} في {location}. هنتواصل معاك على نفس الرقم لتأكيد التفاصيل."
            return f"تمام، كده طلب المعاينة لـ {model} يوم {date} في {location}. هنتواصل معاك على نفس الرقم لتأكيد التفاصيل."

        if time:
            return f"Great, the viewing request for {model} is set for {date} at {time} in {location}. We’ll contact you on the same number to confirm the details."

        return f"Great, the viewing request for {model} is set for {date} in {location}. We’ll contact you on the same number to confirm the details."

    if workflow == "service_booking":
        intent = normalize_intent_for_schema(variables.get("intent"), schema, assistant_id)

        if intent != "booking_request":
            return None

        service = variables.get("service_needed") or "الخدمة"
        date = variables.get("appointment_date")
        time = variables.get("appointment_time")
        phone = variables.get("phone_number")
        patient = variables.get("patient_name")

        if not service or not date or not time or not phone:
            return None

        if arabic:
            name_part = f" باسم {patient}" if patient else ""
            return f"تمام، كده طلب الحجز{name_part} لـ {service} يوم {date} الساعة {time}. هنتواصل معاك على نفس الرقم لتأكيد التفاصيل."

        name_part = f" for {patient}" if patient else ""
        return f"Great, the booking request{name_part} for {service} is set for {date} at {time}. We’ll contact you on the same number to confirm the details."

    return None


def update_asked_questions_from_answer(variables: Dict[str, Any], answer: str) -> Dict[str, Any]:
    variables = dict(variables or {})
    answer_text = (answer or "").lower()

    if any(x in answer_text for x in ["ميزانيتك", "budget"]):
        variables = add_asked_question(variables, "budget")

    if any(x in answer_text for x in ["تحب تشوف", "معاينة", "viewing"]):
        variables = add_asked_question(variables, "viewing_interest")

    if any(x in answer_text for x in ["المكان", "location", "فين"]):
        variables = add_asked_question(variables, "location")

    if any(x in answer_text for x in ["رقم", "phone", "number"]):
        variables = add_asked_question(variables, "phone")

    if any(x in answer_text for x in ["معاد", "ميعاد", "date", "time"]):
        variables = add_asked_question(variables, "date_time")

    return variables


def is_context_sensitive_variable_update(message, updates, knowledge):
    text = (message or "").lower()
    updates = updates or {}

    if not knowledge:
        return False

    sensitive_update_keys = {
        "budget_max",
        "currency",
        "car_brand",
        "car_condition",
        "transmission",
        "preferred_viewing_date",
        "preferred_viewing_time",
    }

    context_markers = [
        "budget",
        "ميزانيتي",
        "ميزانية",
        "الميزانية",
        "لحد",
        "مليون",
        "اشوفها",
        "أشوفها",
        "معاينة",
        "معاينه",
        "احجزها",
        "نفسها",
        "العربية",
        "العربيه",
        "دي",
        "ده",
        "دا",
    ]

    return any(key in updates for key in sensitive_update_keys) and any(marker in text for marker in context_markers)


def build_mock_answer(intent, variables, missing_variables, knowledge):
    intent = normalize_intent(intent)

    if intent == "car_search":
        brand = variables.get("car_brand", "a suitable car")
        budget = variables.get("budget_max")
        condition = variables.get("car_condition", "")
        best_match = choose_best_knowledge_for_variables(knowledge, variables)

        if best_match:
            title = best_match.get("title", "the knowledge base")
            text = best_match.get("text", "")

            if budget:
                return f"Got it — you're looking for a {condition} {brand} up to {budget}. I found a relevant match in {title}: {text[:220]}"

            return f"Got it — you're interested in {brand}. I found relevant information in {title}: {text[:220]}"

        if budget:
            return f"Got it — you're looking for a {condition} {brand} up to {budget}. I can help narrow that down. What model or body type do you prefer?"

        return f"Got it — you're interested in {brand}. What budget range should I look within?"

    if intent == "viewing_request":
        return "Sure — I can help arrange a viewing. I just need the remaining viewing details."

    if intent == "booking_request":
        return "Sure — I can help with booking. What day and time works best for you?"

    if intent == "complaint":
        return "I’m sorry about that. I’ll help you resolve it. Can you share a few more details so I can escalate it properly?"

    if intent == "urgent_medical_issue":
        return "This may need urgent attention. Please contact emergency services or the business directly right away."

    return "Got it. I updated the details I understood. How can I help next?"


def generate_answer(
    assistant,
    recent_messages,
    summary,
    variables,
    knowledge,
    memories,
    user_message,
    intent,
    missing_variables,
    selected_model_tier,
):
    intent = normalize_intent(intent)
    model = model_for_tier(selected_model_tier)

    if MOCK_MODE:
        return build_mock_answer(intent, variables, missing_variables, knowledge), model, selected_model_tier

    language_instruction = detect_reply_language_instruction(user_message)

    system_prompt = f"""
{assistant["system_prompt"]}

Tone:
{assistant["tone"]}

Memory policy:
{assistant["memory_policy"]}

{language_instruction}

Rules:
- Use current variables when relevant.
- Use selected_item when relevant.
- Use retrieved knowledge when relevant.
- Use long-term memories only when relevant.
- Ask for missing important variables naturally.
- Always follow the LANGUAGE RULE above.
- Continue the conversation naturally after acknowledging what was captured.
- If the user asks a follow-up question about a selected item, answer it directly before asking another question.
- If the user provides a budget and the selected item has a known price, tell whether it fits the budget before asking the next useful question.
- Avoid repeating the same CTA/question if it was already asked.
- If the user wants action, move the workflow forward.
- If the user objects to price, offer alternatives or human follow-up.
- If the user is angry, urgent, or asks for a human, suggest human follow-up.
- If knowledge contains prices, kilometers, model years, services, policies, or availability, use them accurately.
- Do not invent cars, prices, appointment slots, doctors, services, or policies.
- Be concise and useful.
- Do not reveal internal routing, memory, variables, or RAG.
"""

    context = f"""
Latest user message:
{user_message}

Conversation summary:
{summary}

Current variables:
{variables}

Relevant long-term memories:
{memories}

Relevant knowledge:
{knowledge}

Intent:
{intent}

Missing variables:
{missing_variables}
"""

    messages = [{"role": "system", "content": system_prompt}, {"role": "system", "content": context}]
    messages.extend(recent_messages)

    answer = chat_text(model, messages, max_tokens=700)
    answer = enforce_reply_language(user_message, answer, model)

    return answer, model, selected_model_tier



def build_compact_advisor_variables(variables: Dict[str, Any]) -> Dict[str, Any]:
    variables = variables or {}

    selected = variables.get("selected_item") if isinstance(variables.get("selected_item"), dict) else {}

    compact = {
        "workflow": variables.get("workflow"),
        "intent": variables.get("intent"),
        "car_brand": variables.get("car_brand") or selected.get("brand"),
        "car_condition": variables.get("car_condition") or selected.get("condition"),
        "budget_max": variables.get("budget_max"),
        "matched_car_model": variables.get("matched_car_model") or selected.get("model"),
        "matched_car_year": variables.get("matched_car_year") or selected.get("year"),
        "matched_car_km": variables.get("matched_car_km") or selected.get("km"),
        "matched_car_price": variables.get("matched_car_price") or selected.get("price"),
        "currency": variables.get("currency") or selected.get("currency"),
        "transmission": variables.get("transmission") or selected.get("transmission"),
        "last_objection_type": variables.get("last_objection_type"),
        "price_sensitive": variables.get("price_sensitive"),
    }

    return {k: v for k, v in compact.items() if v not in [None, "", [], {}]}


def build_compact_advisor_knowledge(knowledge: List[Dict[str, Any]], limit: int = 1) -> List[Dict[str, Any]]:
    compact = []

    for item in (knowledge or [])[:limit]:
        metadata = item.get("metadata") or {}

        clean = {
            "title": item.get("title"),
            "text": item.get("text"),
            "model": metadata.get("model"),
            "year": metadata.get("year"),
            "km": metadata.get("km"),
            "price": metadata.get("price"),
            "currency": metadata.get("currency"),
            "transmission": metadata.get("transmission"),
            "condition": metadata.get("condition"),
        }

        compact.append({k: v for k, v in clean.items() if v not in [None, "", [], {}]})

    return compact


def generate_advisor_answer(
    assistant,
    summary,
    variables,
    knowledge,
    memories,
    user_message,
    selected_model_tier="normal",
):
    model = model_for_tier(selected_model_tier)

    if MOCK_MODE:
        mock_answer = (
            "فاهمك. بناءً على التفاصيل المتاحة، الاختيار مناسب مبدئيًا، "
            "بس الأفضل نأكد الحالة والمعاينة قبل القرار النهائي."
        )
        return mock_answer, model, selected_model_tier

    system_prompt = build_advisor_system_prompt(
        assistant=assistant,
        message=user_message,
        variables=variables,
    )

    context = build_advisor_context(
        message=user_message,
        summary=summary,
        variables=variables,
        knowledge=knowledge,
        memories=memories,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": context},
        {"role": "user", "content": user_message},
    ]

    answer = chat_text(model, messages, max_tokens=350)
    answer = enforce_reply_language(user_message, answer, model)

    return answer, model, selected_model_tier


def should_use_cached_rag(message, route):
    if not RAG_CACHE_ENABLED:
        return False

    text = (message or "").lower().strip()

    followup_markers = [
        "it",
        "that",
        "this",
        "its",
        "is it",
        "does it",
        "what about",
        "how many",
        "automatic",
        "manual",
        "km",
        "color",
        "available",
        "book it",
        "see it",
        "price",
        "budget",
        "under",
        "up to",
        "million",
        "same one",
        "the car",
        "the bmw",
        "ده",
        "دي",
        "دا",
        "العربية",
        "العربيه",
        "متاح",
        "اوتوماتيك",
        "أوتوماتيك",
        "مانيوال",
        "كام كيلو",
        "عاملة كام",
        "عامله كام",
        "بكام",
        "سعرها",
        "ميزانيتي",
        "ميزانية",
        "الميزانية",
        "لحد",
        "مليون",
        "نفسها",
        "احجزها",
        "اشوفها",
        "أشوفها",
        "اشوفه",
        "معاينة",
        "معاينه",
        "غالية",
        "تقسيط",
        "خصم",
    ]

    return any(marker in text for marker in followup_markers)


@app.on_event("startup")
def startup():
    init_db()
    ensure_qdrant()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "mock_mode": MOCK_MODE,
        "rag_cache_enabled": RAG_CACHE_ENABLED,
    }


@app.post("/assistants")
def create_assistant(req: AssistantRequest, x_api_key: str = Header(default="")):
    check_auth(x_api_key)
    upsert_assistant(req.assistant_id, req.name, req.system_prompt, req.tone, req.memory_policy)

    return {
        "status": "saved",
        "assistant_id": req.assistant_id,
    }


@app.post("/schemas")
def create_schema(req: SchemaRequest, x_api_key: str = Header(default="")):
    check_auth(x_api_key)
    save_schema(req.assistant_id, req.schema)

    return {
        "status": "saved",
        "assistant_id": req.assistant_id,
        "schema_keys": list(req.schema.keys()),
    }


@app.get("/schemas/{assistant_id}")
def read_schema(assistant_id: str, x_api_key: str = Header(default="")):
    check_auth(x_api_key)

    return {
        "assistant_id": assistant_id,
        "schema": get_schema(assistant_id),
    }


@app.post("/ingest")
def ingest(req: IngestRequest, x_api_key: str = Header(default="")):
    check_auth(x_api_key)

    chunks = ingest_document(
        assistant_id=req.assistant_id,
        document_id=req.document_id,
        title=req.title,
        text=req.text,
        metadata=req.metadata,
    )

    upsert_knowledge_document(
        assistant_id=req.assistant_id,
        document_id=req.document_id,
        title=req.title,
        metadata=req.metadata,
        chunk_count=chunks,
    )

    structured_items = upsert_structured_inventory_from_text(
        assistant_id=req.assistant_id,
        document_id=req.document_id,
        title=req.title,
        text=req.text,
        metadata=req.metadata,
    )

    return {
        "status": "ingested",
        "assistant_id": req.assistant_id,
        "document_id": req.document_id,
        "chunks": chunks,
        "structured_items": structured_items,
    }


@app.get("/knowledge/{assistant_id}")
def read_knowledge_documents(assistant_id: str, x_api_key: str = Header(default="")):
    check_auth(x_api_key)

    return {
        "assistant_id": assistant_id,
        "documents": list_knowledge_documents(assistant_id),
    }


@app.post("/knowledge/search")
def knowledge_search(req: KnowledgeSearchRequest, x_api_key: str = Header(default="")):
    check_auth(x_api_key)

    results = search_knowledge(
        assistant_id=req.assistant_id,
        query=req.query,
        limit=req.limit,
    )

    compressed = compress_knowledge(results, req.query)

    return {
        "assistant_id": req.assistant_id,
        "query": req.query,
        "count": len(compressed),
        "results": compressed,
    }


@app.get("/variables/{conversation_id}")
def read_variables(conversation_id: str, x_api_key: str = Header(default="")):
    check_auth(x_api_key)

    return {
        "conversation_id": conversation_id,
        "variables": get_variables(conversation_id),
    }


@app.patch("/variables")
def patch_variables(req: PatchVariablesRequest, x_api_key: str = Header(default="")):
    check_auth(x_api_key)

    existing = get_variables(req.conversation_id)
    updated = apply_variable_patch(existing, req.updates, req.deletions)

    save_variables(
        req.conversation_id,
        req.assistant_id,
        req.user_id,
        updated,
    )

    return {
        "status": "updated",
        "conversation_id": req.conversation_id,
        "variables": updated,
    }


@app.get("/memories/{assistant_id}/{user_id}")
def read_user_memories(assistant_id: str, user_id: str, x_api_key: str = Header(default="")):
    check_auth(x_api_key)

    return {
        "assistant_id": assistant_id,
        "user_id": user_id,
        "memories": list_long_term_memories(assistant_id, user_id),
    }


@app.post("/chat")
def chat(req: ChatRequest, x_api_key: str = Header(default="")):
    check_auth(x_api_key)

    assistant = get_or_create_assistant(req.assistant_id)

    ensure_conversation(
        req.conversation_id,
        req.assistant_id,
        req.user_id,
        req.channel,
    )

    save_message(
        req.conversation_id,
        req.assistant_id,
        req.user_id,
        "user",
        req.message,
    )

    recent_messages = get_recent_messages(req.conversation_id, limit=RECENT_MESSAGES_LIMIT)
    summary = get_summary(req.conversation_id)
    schema = get_schema(req.assistant_id)
    existing_variables = get_variables(req.conversation_id)

    existing_variables = apply_conversation_stage_governor(
        message=req.message,
        variables=existing_variables,
        recent_messages=recent_messages,
        assistant_id=req.assistant_id,
    )

    workflow_type = infer_workflow_type(schema, req.assistant_id)
    playbook = get_playbook(workflow_type)
    assistant_profile = get_assistant_profile(req.assistant_id)

    # ------------------------------------------------------------
    # Universal entry path
    # ------------------------------------------------------------
    entry_path_result = None

    if req.assistant_id != "service_center_agentic_rag" and should_try_entry_path(req.message, schema, existing_variables, req.assistant_id):
        entry_knowledge = []
        entry_knowledge_source = "none"
        entry_query_variables = {}

        if workflow_type == "car_sales" and req.assistant_id != "service_center_agentic_rag":
            entry_query_variables = extract_car_variables(schema, req.message)

            structured_items = search_structured_inventory(
                assistant_id=req.assistant_id,
                query_variables=entry_query_variables,
                limit=KNOWLEDGE_TOP_K,
                raw_query=req.message,
            )

            if structured_items:
                entry_knowledge = inventory_items_to_knowledge(structured_items)
                entry_knowledge_source = "structured_inventory"

        if not entry_knowledge:
            raw_entry_knowledge = search_knowledge(
                req.assistant_id,
                req.message,
                limit=KNOWLEDGE_TOP_K,
            )
            compressed_entry_knowledge = compress_knowledge(raw_entry_knowledge, req.message)

            rebuilt_count = 0

            if (
                workflow_type == "car_sales"
                and req.assistant_id != "service_center_agentic_rag"
                and compressed_entry_knowledge
            ):
                rebuilt_count = rebuild_structured_inventory_from_knowledge(
                    assistant_id=req.assistant_id,
                    knowledge=compressed_entry_knowledge,
                )

                if rebuilt_count:
                    structured_items = search_structured_inventory(
                        assistant_id=req.assistant_id,
                        query_variables=entry_query_variables,
                        limit=KNOWLEDGE_TOP_K,
                        raw_query=req.message,
                    )

                    if structured_items:
                        entry_knowledge = inventory_items_to_knowledge(structured_items)
                        entry_knowledge_source = "structured_inventory_rebuilt"

            if not entry_knowledge:
                entry_knowledge = compressed_entry_knowledge
                entry_knowledge_source = "qdrant" if entry_knowledge else "none"

        entry_path_result = build_entry_path_response(
            message=req.message,
            schema=schema,
            assistant_id=req.assistant_id,
            variables=existing_variables,
            knowledge=entry_knowledge,
        )

        if entry_path_result:
            entry_path_result["knowledge_source"] = entry_knowledge_source

    if entry_path_result:
        updated_variables = entry_path_result.get("updates") or dict(existing_variables or {})
        answer = entry_path_result["answer"]

        save_variables(
            req.conversation_id,
            req.assistant_id,
            req.user_id,
            updated_variables,
        )

        save_message(
            req.conversation_id,
            req.assistant_id,
            req.user_id,
            "assistant",
            answer,
        )

        updated_summary = summary
        long_term_memories_written = []

        if not entry_path_result.get("skip_summary", True):
            updated_summary = update_conversation_summary(
                conversation_id=req.conversation_id,
                assistant_id=req.assistant_id,
                user_id=req.user_id,
                variables=updated_variables,
            )

        if not entry_path_result.get("skip_memory", True):
            long_term_memories_written = decide_and_write_long_term_memories(
                assistant_id=req.assistant_id,
                user_id=req.user_id,
                conversation_id=req.conversation_id,
                summary=updated_summary,
                recent_messages=get_recent_messages(req.conversation_id, limit=8),
                variables=updated_variables,
            )

        entry_usage_input_obj = {
            "message": req.message,
            "variables": updated_variables,
            "entry_path": entry_path_result,
            "playbook": playbook,
            "assistant_profile": assistant_profile,
        }

        token_usage = build_token_usage_report(
            model_used="none",
            model_tier=entry_path_result.get("model_tier", "entry_path"),
            answer_mode="entry_path",
            input_obj=entry_usage_input_obj,
            output_text=answer,
            knowledge_source=entry_path_result.get("knowledge_source", "none"),
            rag_cache_hit=False,
        )

        log_estimated_usage(
            assistant_id=req.assistant_id,
            conversation_id=req.conversation_id,
            user_id=req.user_id,
            model="none",
            purpose="chat_entry_path",
            input_obj=entry_usage_input_obj,
            output_text=answer,
            metadata={
                "model_tier": entry_path_result.get("model_tier", "entry_path"),
                "answer_mode": "entry_path",
                "needs_rag": True,
                "needs_memory": not entry_path_result.get("skip_memory", True),
                "rag_cache_hit": False,
                "token_usage": token_usage,
            },
        )

        response_payload = {
            "answer": answer,
            "assistant_id": req.assistant_id,
            "conversation_id": req.conversation_id,
            "intent": updated_variables.get("intent", "general_question"),
            "variables": updated_variables,
            "variable_updates": updated_variables,
            "variable_deletions": [],
            "missing_variables": [],
            "recommended_next_action": entry_path_result.get("action", "entry_path"),
            "next_best_action": {
                "action": entry_path_result.get("action", "entry_path"),
                "reason": "Handled by universal entry path before GPT routing.",
                "confidence": 0.9,
            },
            "route": {
                "answer_mode": "entry_path",
                "needs_rag": True,
                "needs_memory": not entry_path_result.get("skip_memory", True),
                "rag_cache_hit": False,
            },
            "knowledge_used": entry_path_result.get("knowledge_used", []),
            "knowledge_source": entry_path_result.get("knowledge_source", "none"),
            "memories_used": [],
            "summary": updated_summary,
            "long_term_memories_written": long_term_memories_written,
            "model_used": "none",
            "model_tier": entry_path_result.get("model_tier", "entry_path"),
            "token_usage": token_usage,
            "mock_mode": MOCK_MODE,
            "memory_saved": bool(long_term_memories_written),
        }

        return compact_chat_response(response_payload)

    # ------------------------------------------------------------
    # Early deterministic date/time/location scheduling path
    # ------------------------------------------------------------
    datetime_fast_result = build_datetime_location_fast_response(
        message=req.message,
        variables=existing_variables,
        workflow=workflow_type,
    )

    if datetime_fast_result:
        datetime_updates = datetime_fast_result.get("updates") or {}
        updated_variables = dict(existing_variables or {})
        updated_variables.update(datetime_updates)
        updated_variables = autofill_channel_context(updated_variables, req)
        updated_variables = sync_selected_item_state(updated_variables)

        updated_variables = set_workflow_stage(
            schema,
            req.assistant_id,
            updated_variables,
            updated_variables.get("intent", "viewing_request"),
        )

        updated_variables = compute_lead_score(
            updated_variables,
            updated_variables.get("intent", "viewing_request"),
        )

        answer = datetime_fast_result["answer"]

        save_variables(
            req.conversation_id,
            req.assistant_id,
            req.user_id,
            updated_variables,
        )

        save_message(
            req.conversation_id,
            req.assistant_id,
            req.user_id,
            "assistant",
            answer,
        )

        updated_summary = summary
        long_term_memories_written = []

        if not datetime_fast_result.get("skip_summary", True):
            updated_summary = update_conversation_summary(
                conversation_id=req.conversation_id,
                assistant_id=req.assistant_id,
                user_id=req.user_id,
                variables=updated_variables,
            )

        if not datetime_fast_result.get("skip_memory", True):
            long_term_memories_written = decide_and_write_long_term_memories(
                assistant_id=req.assistant_id,
                user_id=req.user_id,
                conversation_id=req.conversation_id,
                summary=updated_summary,
                recent_messages=get_recent_messages(req.conversation_id, limit=8),
                variables=updated_variables,
            )

        usage_input_obj = {
            "message": req.message,
            "variables": updated_variables,
            "datetime_fast_path": datetime_fast_result,
            "playbook": playbook,
            "assistant_profile": assistant_profile,
        }

        token_usage = build_token_usage_report(
            model_used="none",
            model_tier=datetime_fast_result.get("model_tier", "fast_path"),
            answer_mode="datetime_fast_path",
            input_obj=usage_input_obj,
            output_text=answer,
            knowledge_source="none",
            rag_cache_hit=False,
        )

        log_estimated_usage(
            assistant_id=req.assistant_id,
            conversation_id=req.conversation_id,
            user_id=req.user_id,
            model="none",
            purpose="chat_datetime_fast_path",
            input_obj=usage_input_obj,
            output_text=answer,
            metadata={
                "model_tier": datetime_fast_result.get("model_tier", "fast_path"),
                "answer_mode": "datetime_fast_path",
                "needs_rag": False,
                "needs_memory": not datetime_fast_result.get("skip_memory", True),
                "rag_cache_hit": False,
                "token_usage": token_usage,
            },
        )

        response_payload = {
            "answer": answer,
            "assistant_id": req.assistant_id,
            "conversation_id": req.conversation_id,
            "intent": updated_variables.get("intent", "viewing_request"),
            "variables": updated_variables,
            "variable_updates": datetime_updates,
            "variable_deletions": [],
            "missing_variables": calculate_missing_required_variables(
                schema=schema,
                variables=updated_variables,
                intent=updated_variables.get("intent", "viewing_request"),
                assistant_id=req.assistant_id,
            ),
            "recommended_next_action": datetime_fast_result.get("action", "datetime_fast_path"),
            "next_best_action": {
                "action": datetime_fast_result.get("action", "datetime_fast_path"),
                "reason": "Handled by deterministic date/time/location fast path.",
                "confidence": 0.95,
            },
            "route": {
                "answer_mode": "datetime_fast_path",
                "needs_rag": False,
                "needs_memory": not datetime_fast_result.get("skip_memory", True),
                "rag_cache_hit": False,
            },
            "knowledge_used": [],
            "knowledge_source": "none",
            "memories_used": [],
            "summary": updated_summary,
            "long_term_memories_written": long_term_memories_written,
            "model_used": "none",
            "model_tier": datetime_fast_result.get("model_tier", "fast_path"),
            "token_usage": token_usage,
            "mock_mode": MOCK_MODE,
            "memory_saved": bool(long_term_memories_written),
        }

        return compact_chat_response(response_payload)

        # ------------------------------------------------------------
    # Universal assistant brain before fast_path
    # ------------------------------------------------------------
    # Runs before fast_path so objections/advice/commercial psychology
    # are handled by the real brain first.
    brain_wants_gpt = False

    brain_result = build_brain_deterministic_response(
        message=req.message,
        schema=schema,
        variables=existing_variables,
        assistant_id=req.assistant_id,
        recent_messages=recent_messages,
    )

    if req.assistant_id == "service_center_agentic_rag":
        brain_result = None
        brain_wants_gpt = True

    if brain_result and brain_result.get("should_use_gpt"):
        brain_wants_gpt = True

        brain_hint = build_brain_advisor_hint(
            message=req.message,
            schema=schema,
            variables=existing_variables,
            assistant_id=req.assistant_id,
        )

        existing_variables = dict(existing_variables or {})
        existing_variables["_brain_hint"] = brain_hint
   
    # ------------------------------------------------------------
    # Internal booking sub-agent
    # ------------------------------------------------------------

    booking_result = run_booking_subagent(
        assistant=assistant,
        assistant_id=req.assistant_id,
        user_id=req.user_id,
        conversation_id=req.conversation_id,
        message=req.message,
        variables=existing_variables,
        recent_messages=recent_messages,
        summary=summary,
        schema=schema,
        tool_result=req.tool_result,
    )

    if booking_result.get("handled"):
        updated_variables = dict(existing_variables or {})
        updated_variables.update(booking_result.get("variables", {}))

        save_variables(req.conversation_id, updated_variables)

        answer = booking_result.get("answer", "")

        save_message(req.conversation_id, "user", req.message)

        if answer:
            save_message(req.conversation_id, "assistant", answer)

        response_payload = {
            "answer": answer,
            "assistant_id": req.assistant_id,
            "conversation_id": req.conversation_id,
            "intent": updated_variables.get("intent", "booking_request"),
            "variables": updated_variables,
            "variable_updates": booking_result.get("variables", {}),
            "variable_deletions": [],
            "missing_variables": booking_result.get("missing_variables", []),
            "active_subagent": booking_result.get("active_subagent"),
            "booking_stage": booking_result.get("booking_stage"),
            "action_required": booking_result.get("action_required"),
            "recommended_next_action": booking_result.get("recommended_next_action", "booking_flow"),
            "next_best_action": {
                "action": booking_result.get("recommended_next_action", "booking_flow"),
                "reason": booking_result.get("reason", "Handled by internal booking sub-agent."),
                "confidence": 0.9,
            },
            "route": {
                "answer_mode": "booking_subagent",
                "active_subagent": booking_result.get("active_subagent"),
                "booking_stage": booking_result.get("booking_stage"),
                "needs_rag": False,
                "needs_memory": False,
                "rag_cache_hit": False,
                "reason": booking_result.get("reason", "Handled by internal booking sub-agent."),
            },
            "knowledge_used": [],
            "knowledge_source": "booking_subagent",
            "memories_used": [],
            "summary": summary,
            "long_term_memories_written": [],
            "model_used": booking_result.get("model_used", "none"),
            "model_tier": booking_result.get("model_tier", "booking_subagent"),
            "token_usage": {
                "model_used": booking_result.get("model_used", "none"),
                "model_tier": booking_result.get("model_tier", "booking_subagent"),
                "answer_mode": "booking_subagent",
                "input_tokens_estimate": 0,
                "output_tokens_estimate": 0,
                "total_tokens_estimate": 0,
                "estimated_cost_usd": 0.0,
                "is_estimate": True,
                "notes": "Handled by internal booking sub-agent before premium/RAG generation.",
            },
            "mock_mode": MOCK_MODE,
            "memory_saved": False,
        }

        return compact_chat_response(response_payload)
    
    # ------------------------------------------------------------
    # Adaptive premium intelligence layer
    # ------------------------------------------------------------

    if existing_variables.get("_stage_instruction"):
        existing_variables["_conversation_governor_instruction"] = (
            "HIGH PRIORITY CONVERSATION STATE INSTRUCTION:\n"
            + existing_variables.get("_stage_instruction", "")
            + "\n\nANTI-REPEAT RULE:\n"
            + existing_variables.get("_do_not_repeat_instruction", "")
            + "\n\nPLAYBOOK RULE:\n"
            "Use the assistant's prompt, knowledge base, and conversation playbook to decide the next best step. "
            "Do not behave like this is the first message if known_facts, last_user_answer, summary, or issue_description already contain context."
        )

    premium_handled, premium_result = run_adaptive_premium_turn(
        assistant=assistant,
        assistant_id=req.assistant_id,
        user_id=req.user_id,
        schema=schema,
        variables=existing_variables,
        recent_messages=recent_messages,
        summary=summary,
        user_message=req.message,
        route={
            "brain_wants_gpt": brain_wants_gpt,
            "brain_hint": existing_variables.get("_brain_hint") if isinstance(existing_variables, dict) else None,
        },
    )

    if premium_handled:
        updated_variables = premium_result.get("variables") or dict(existing_variables or {})
        answer = premium_result["answer"]

        updated_variables = autofill_channel_context(updated_variables, req)
        updated_variables = sync_selected_item_state(updated_variables)

        current_intent = (
            premium_result.get("intent")
            or updated_variables.get("intent")
            or "general_question"
        )

        updated_variables = set_workflow_stage(
            schema,
            req.assistant_id,
            updated_variables,
            current_intent,
        )

        updated_variables = compute_lead_score(
            updated_variables,
            current_intent,
        )

        save_variables(
            req.conversation_id,
            req.assistant_id,
            req.user_id,
            updated_variables,
        )

        save_message(
            req.conversation_id,
            req.assistant_id,
            req.user_id,
            "assistant",
            answer,
        )

        updated_summary = update_conversation_summary(
            conversation_id=req.conversation_id,
            assistant_id=req.assistant_id,
            user_id=req.user_id,
            variables=updated_variables,
        )

        premium_memory_decision = premium_result.get("premium_memory_decision", {}) or {}

        if premium_memory_decision.get("suppress_long_term_memory"):
            long_term_memories_written = []
        else:
            long_term_memories_written = decide_and_write_long_term_memories(
                assistant_id=req.assistant_id,
                user_id=req.user_id,
                conversation_id=req.conversation_id,
                summary=updated_summary,
                recent_messages=get_recent_messages(req.conversation_id, limit=8),
                variables=updated_variables,
            )

        premium_usage_input_obj = {
            "message": req.message,
            "summary": summary,
            "variables": updated_variables,
            "mode_decision": premium_result.get("mode_decision"),
            "retrieval_queries": premium_result.get("retrieval_queries"),
            "knowledge": premium_result.get("knowledge_used"),
            "memories": premium_result.get("memories_used"),
            "evidence_judgment": premium_result.get("evidence_judgment"),
            "critique": premium_result.get("critique"),
            "premium_memory_decision": premium_result.get("premium_memory_decision"),
            "premium_debug": premium_result.get("premium_debug"),
            "playbook": playbook,
            "assistant_profile": assistant_profile,
        }

        token_usage = build_token_usage_report(
            model_used=premium_result.get("model_used"),
            model_tier=premium_result.get("model_tier"),
            answer_mode="adaptive_premium",
            input_obj=premium_usage_input_obj,
            output_text=answer,
            knowledge_source=premium_result.get("knowledge_source", "none"),
            rag_cache_hit=False,
            notes="Adaptive premium estimate includes retrieval, evidence judgment, answer generation, critique, and revision payloads.",
        )

        log_estimated_usage(
            assistant_id=req.assistant_id,
            conversation_id=req.conversation_id,
            user_id=req.user_id,
            model=premium_result.get("model_used"),
            purpose="chat_adaptive_premium",
            input_obj=premium_usage_input_obj,
            output_text=answer,
            metadata={
                "model_tier": premium_result.get("model_tier"),
                "answer_mode": "adaptive_premium",
                "premium_mode": premium_result.get("mode_decision", {}).get("mode"),
                "knowledge_source": premium_result.get("knowledge_source", "none"),
                "rag_cache_hit": False,
                "token_usage": token_usage,
            },
        )

        response_payload = {
            "answer": answer,
            "assistant_id": req.assistant_id,
            "conversation_id": req.conversation_id,
            "intent": current_intent,
            "variables": updated_variables,
            "variable_updates": premium_result.get("variable_updates", {}),
            "variable_deletions": premium_result.get("variable_deletions", []),
            "missing_variables": calculate_missing_required_variables(
                schema=schema,
                variables=updated_variables,
                intent=current_intent,
                assistant_id=req.assistant_id,
            ),
            "recommended_next_action": premium_result.get("recommended_next_action", "adaptive_premium"),
            "next_best_action": {
                "action": premium_result.get("recommended_next_action", "adaptive_premium"),
                "reason": premium_result.get("mode_decision", {}).get("reason", "Handled by adaptive premium layer."),
                "confidence": premium_result.get("evidence_judgment", {}).get("confidence", 0.75),
            },
            "route": premium_result.get("route", {}),
            "premium_debug": premium_result.get("premium_debug", {}),
            "premium_memory_decision": premium_result.get("premium_memory_decision", {}),
            "knowledge_used": premium_result.get("knowledge_used", []),
            "knowledge_source": premium_result.get("knowledge_source", "none"),
            "memories_used": premium_result.get("memories_used", []),
            "summary": updated_summary,
            "long_term_memories_written": long_term_memories_written,
            "model_used": premium_result.get("model_used"),
            "model_tier": premium_result.get("model_tier"),
            "token_usage": token_usage,
            "mock_mode": MOCK_MODE,
            "memory_saved": bool(long_term_memories_written),
        }

        return compact_chat_response(response_payload)

    elif brain_result and not brain_result.get("no_direct_answer"):
        # existing old brain block continues here
        brain_updates = brain_result.get("updates") or {}
        updated_variables = dict(existing_variables or {})
        updated_variables.update(brain_updates)

        updated_variables = autofill_channel_context(updated_variables, req)
        updated_variables = sync_selected_item_state(updated_variables)

        current_intent = updated_variables.get("intent") or existing_variables.get("intent") or "general_question"

        updated_variables = set_workflow_stage(
            schema,
            req.assistant_id,
            updated_variables,
            current_intent,
        )

        updated_variables = compute_lead_score(
            updated_variables,
            current_intent,
        )

        answer = brain_result["answer"]

        save_variables(
            req.conversation_id,
            req.assistant_id,
            req.user_id,
            updated_variables,
        )

        save_message(
            req.conversation_id,
            req.assistant_id,
            req.user_id,
            "assistant",
            answer,
        )

        updated_summary = summary
        long_term_memories_written = []

        if not brain_result.get("skip_summary", True):
            updated_summary = update_conversation_summary(
                conversation_id=req.conversation_id,
                assistant_id=req.assistant_id,
                user_id=req.user_id,
                variables=updated_variables,
            )

        if not brain_result.get("skip_memory", True):
            long_term_memories_written = decide_and_write_long_term_memories(
                assistant_id=req.assistant_id,
                user_id=req.user_id,
                conversation_id=req.conversation_id,
                summary=updated_summary,
                recent_messages=get_recent_messages(req.conversation_id, limit=8),
                variables=updated_variables,
            )

        usage_input_obj = {
            "message": req.message,
            "variables": updated_variables,
            "brain_result": brain_result,
            "playbook": playbook,
            "assistant_profile": assistant_profile,
        }

        token_usage = build_token_usage_report(
            model_used="none",
            model_tier=brain_result.get("model_tier", "assistant_brain"),
            answer_mode=brain_result.get("answer_mode", "assistant_brain"),
            input_obj=usage_input_obj,
            output_text=answer,
            knowledge_source="none",
            rag_cache_hit=False,
        )

        log_estimated_usage(
            assistant_id=req.assistant_id,
            conversation_id=req.conversation_id,
            user_id=req.user_id,
            model="none",
            purpose="chat_assistant_brain",
            input_obj=usage_input_obj,
            output_text=answer,
            metadata={
                "model_tier": brain_result.get("model_tier", "assistant_brain"),
                "answer_mode": brain_result.get("answer_mode", "assistant_brain"),
                "needs_rag": False,
                "needs_memory": not brain_result.get("skip_memory", True),
                "rag_cache_hit": False,
                "token_usage": token_usage,
                "brain_decision": brain_result.get("brain_decision"),
            },
        )

        response_payload = {
            "answer": answer,
            "assistant_id": req.assistant_id,
            "conversation_id": req.conversation_id,
            "intent": updated_variables.get("intent", current_intent),
            "variables": updated_variables,
            "variable_updates": brain_updates,
            "variable_deletions": [],
            "missing_variables": calculate_missing_required_variables(
                schema=schema,
                variables=updated_variables,
                intent=updated_variables.get("intent", current_intent),
                assistant_id=req.assistant_id,
            ),
            "recommended_next_action": brain_result.get("action", "assistant_brain"),
            "next_best_action": {
                "action": brain_result.get("action", "assistant_brain"),
                "reason": "Handled by universal assistant brain before fast path and GPT routing.",
                "confidence": 0.9,
            },
            "route": {
                "answer_mode": brain_result.get("answer_mode", "assistant_brain"),
                "needs_rag": False,
                "needs_memory": not brain_result.get("skip_memory", True),
                "rag_cache_hit": False,
                "brain_decision": brain_result.get("brain_decision"),
            },
            "knowledge_used": [],
            "knowledge_source": "none",
            "memories_used": [],
            "summary": updated_summary,
            "long_term_memories_written": long_term_memories_written,
            "model_used": "none",
            "model_tier": brain_result.get("model_tier", "assistant_brain"),
            "token_usage": token_usage,
            "mock_mode": MOCK_MODE,
            "memory_saved": bool(long_term_memories_written),
        }

        return compact_chat_response(response_payload)

    # ------------------------------------------------------------
    # Universal pre-router fast path
    # ------------------------------------------------------------
    # Safety default:
    # If the assistant brain block did not run or did not define this flag,
    # keep fast_path allowed by default.
    if "brain_wants_gpt" not in locals():
        brain_wants_gpt = False

    fast_path_result = None

    # Hard guard:
    # If assistant_brain requested GPT/advisor, fast_path must NOT steal the message.
    brain_has_advisor_hint = isinstance(existing_variables, dict) and bool(existing_variables.get("_brain_hint"))

    if (
        not brain_wants_gpt
        and not brain_has_advisor_hint
        and should_try_fast_path(req.message, existing_variables, schema)
    ):
        fast_path_result = build_workflow_fast_answer(
            message=req.message,
            schema=schema,
            assistant_id=req.assistant_id,
            variables=existing_variables,
            recent_messages=recent_messages,
        )

    if fast_path_result:
        fast_updates = fast_path_result.get("updates") or {}
        updated_variables = dict(existing_variables or {})
        updated_variables.update(fast_updates)

        save_variables(
            req.conversation_id,
            req.assistant_id,
            req.user_id,
            updated_variables,
        )

        answer = fast_path_result["answer"]

        save_message(
            req.conversation_id,
            req.assistant_id,
            req.user_id,
            "assistant",
            answer,
        )

        updated_summary = summary
        long_term_memories_written = []

        if not fast_path_result.get("skip_summary", True):
            updated_summary = update_conversation_summary(
                conversation_id=req.conversation_id,
                assistant_id=req.assistant_id,
                user_id=req.user_id,
                variables=updated_variables,
            )

        if not fast_path_result.get("skip_memory", True):
            long_term_memories_written = decide_and_write_long_term_memories(
                assistant_id=req.assistant_id,
                user_id=req.user_id,
                conversation_id=req.conversation_id,
                summary=updated_summary,
                recent_messages=get_recent_messages(req.conversation_id, limit=8),
                variables=updated_variables,
            )

        fast_usage_input_obj = {
            "message": req.message,
            "variables": updated_variables,
            "fast_path": fast_path_result,
            "playbook": playbook,
            "assistant_profile": assistant_profile,
        }

        token_usage = build_token_usage_report(
            model_used="none",
            model_tier=fast_path_result.get("model_tier", "fast_path"),
            answer_mode="fast_path",
            input_obj=fast_usage_input_obj,
            output_text=answer,
            knowledge_source="none",
            rag_cache_hit=False,
        )

        log_estimated_usage(
            assistant_id=req.assistant_id,
            conversation_id=req.conversation_id,
            user_id=req.user_id,
            model="none",
            purpose="chat_fast_path",
            input_obj=fast_usage_input_obj,
            output_text=answer,
            metadata={
                "model_tier": fast_path_result.get("model_tier", "fast_path"),
                "answer_mode": "fast_path",
                "needs_rag": False,
                "needs_memory": not fast_path_result.get("skip_memory", True),
                "rag_cache_hit": False,
                "token_usage": token_usage,
            },
        )

        response_payload = {
            "answer": answer,
            "assistant_id": req.assistant_id,
            "conversation_id": req.conversation_id,
            "intent": updated_variables.get("intent", existing_variables.get("intent", "general_question")),
            "variables": updated_variables,
            "variable_updates": fast_updates,
            "variable_deletions": [],
            "missing_variables": [],
            "recommended_next_action": fast_path_result.get("action", "fast_path"),
            "next_best_action": {
                "action": fast_path_result.get("action", "fast_path"),
                "reason": "Handled by universal pre-router fast path.",
                "confidence": 0.95,
            },
            "route": {
                "answer_mode": "fast_path",
                "needs_rag": False,
                "needs_memory": not fast_path_result.get("skip_memory", True),
                "rag_cache_hit": False,
            },
            "knowledge_used": [],
            "knowledge_source": "none",
            "memories_used": [],
            "summary": updated_summary,
            "long_term_memories_written": long_term_memories_written,
            "model_used": "none",
            "model_tier": fast_path_result.get("model_tier", "fast_path"),
            "token_usage": token_usage,
            "mock_mode": MOCK_MODE,
            "memory_saved": bool(long_term_memories_written),
        }

        return compact_chat_response(response_payload)

    # ------------------------------------------------------------
    # Smart advisor escalation
    # ------------------------------------------------------------
    if should_escalate_to_advisor(
        message=req.message,
        variables=existing_variables,
        schema=schema,
        assistant_id=req.assistant_id,
    ):
        route = build_advisor_route("User needs advice, comparison, or judgment.")

        knowledge = []
        knowledge_source = "none"

        cached = get_rag_cache(req.conversation_id, RAG_CACHE_MAX_AGE_MINUTES) if RAG_CACHE_ENABLED else None

        if cached:
            knowledge = cached.get("compressed_payload") or cached.get("knowledge_payload") or []
            knowledge_source = "cache"
            route["rag_cache_hit"] = True
        else:
            raw_knowledge = search_knowledge(
                req.assistant_id,
                req.message,
                limit=KNOWLEDGE_TOP_K,
            )
            knowledge = compress_knowledge(raw_knowledge, req.message)
            knowledge_source = "qdrant" if knowledge else "none"
            route["rag_cache_hit"] = False

            if raw_knowledge:
                save_rag_cache(
                    conversation_id=req.conversation_id,
                    assistant_id=req.assistant_id,
                    user_id=req.user_id,
                    query=req.message,
                    knowledge_payload=raw_knowledge,
                    compressed_payload=knowledge,
                )

        memories = []

        advisor_variables_raw = dict(existing_variables or {})
        brain_hint = advisor_variables_raw.pop("_brain_hint", None)

        advisor_variables = build_compact_advisor_variables(advisor_variables_raw)

        if brain_hint:
            decision = brain_hint.get("brain_decision", {}) if isinstance(brain_hint, dict) else {}
            advisor_variables["brain_strategy"] = {
                "user_state": decision.get("user_state"),
                "sales_move": decision.get("sales_move"),
                "recommended_action": decision.get("recommended_action"),
            }

        compact_knowledge = build_compact_advisor_knowledge(knowledge, limit=1)

        answer, model, tier = generate_advisor_answer(
            assistant=assistant,
            summary="",
            variables=advisor_variables,
            knowledge=compact_knowledge,
            memories=[],
            user_message=req.message,
            selected_model_tier=route.get("selected_model_tier", "normal"),
        )

        save_message(
            req.conversation_id,
            req.assistant_id,
            req.user_id,
            "assistant",
            answer,
        )

        token_usage = build_token_usage_report(
            model_used=model,
            model_tier=tier,
            answer_mode="advisor",
            input_obj={
                "message": req.message,
                "summary": "",
                "variables": advisor_variables,
                "knowledge": compact_knowledge,
                "route": route,
                "playbook": playbook,
                "assistant_profile": assistant_profile,
            },
            output_text=answer,
            knowledge_source=knowledge_source,
            rag_cache_hit=route.get("rag_cache_hit", False),
        )

        log_estimated_usage(
            assistant_id=req.assistant_id,
            conversation_id=req.conversation_id,
            user_id=req.user_id,
            model=model,
            purpose="chat_advisor",
            input_obj={
                "message": req.message,
                "summary": "",
                "variables": advisor_variables,
                "knowledge": compact_knowledge,
                "route": route,
                "playbook": playbook,
                "assistant_profile": assistant_profile,
            },
            output_text=answer,
            metadata={
                "model_tier": tier,
                "answer_mode": "advisor",
                "knowledge_source": knowledge_source,
                "rag_cache_hit": route.get("rag_cache_hit", False),
                "token_usage": token_usage,
            },
        )

        response_payload = {
            "answer": answer,
            "assistant_id": req.assistant_id,
            "conversation_id": req.conversation_id,
            "intent": "advisory_question",
            "variables": existing_variables,
            "variable_updates": {},
            "variable_deletions": [],
            "missing_variables": [],
            "recommended_next_action": "advisor_response",
            "route": route,
            "knowledge_used": compact_knowledge,
            "knowledge_source": knowledge_source,
            "memories_used": memories,
            "summary": summary,
            "long_term_memories_written": [],
            "model_used": model,
            "model_tier": tier,
            "token_usage": token_usage,
            "mock_mode": MOCK_MODE,
            "memory_saved": False,
        }

        return compact_chat_response(response_payload)

    route = deterministic_route_guess(
        message=req.message,
        schema=schema,
        variables=existing_variables,
        assistant_id=req.assistant_id,
    )

    if route is None:
        route = route_message(
            assistant=assistant,
            summary=summary,
            variables=existing_variables,
            recent_messages=recent_messages,
            user_message=req.message,
        )

    route["intent_hint"] = normalize_intent_for_schema(
        route.get("intent_hint", "general_question"),
        schema=schema,
        assistant_id=req.assistant_id,
    )

    if route.get("needs_variable_extraction", True):
        extraction = extract_variables(
            schema=schema,
            existing_variables=existing_variables,
            recent_messages=recent_messages,
            user_message=req.message,
        )
    else:
        extraction = {
            "intent": existing_variables.get("intent", route.get("intent_hint", "general_question")),
            "updates": {},
            "deletions": [],
            "missing_variables": [],
            "confidence": 1.0,
            "notes": "Variable extraction skipped by router.",
        }

    extraction["intent"] = normalize_intent_for_schema(
        extraction.get("intent", "general_question"),
        schema=schema,
        assistant_id=req.assistant_id,
    )

    updated_variables = apply_variable_patch(
        existing_variables,
        extraction.get("updates", {}),
        extraction.get("deletions", []),
    )

    updated_variables = autofill_channel_context(updated_variables, req)

    deterministic_time_place_updates = extract_datetime_location_patch(
        message=req.message,
        variables=updated_variables,
        workflow=workflow_type,
    )

    if deterministic_time_place_updates:
        updated_variables = apply_variable_patch(
            updated_variables,
            deterministic_time_place_updates,
            [],
        )

    current_intent = normalize_intent_for_schema(
        extraction.get("intent") or updated_variables.get("intent") or "general_question",
        schema=schema,
        assistant_id=req.assistant_id,
    )

    updated_variables["intent"] = current_intent

    missing_required_variables = calculate_missing_required_variables(
        schema=schema,
        variables=updated_variables,
        intent=current_intent,
        assistant_id=req.assistant_id,
    )

    if not missing_required_variables:
        missing_required_variables = extraction.get("missing_variables", [])

    recommended_next_action = "continue_conversation"

    if missing_required_variables:
        recommended_next_action = "ask_clarifying_question"

    if current_intent == "booking_request":
        recommended_next_action = "collect_booking_details"

    if current_intent == "viewing_request":
        recommended_next_action = "collect_viewing_details"

    if current_intent == "human_handoff":
        recommended_next_action = "human_handoff"

    if current_intent == "complaint":
        recommended_next_action = "consider_human_handoff"

    if current_intent == "urgent_medical_issue":
        recommended_next_action = "urgent_human_handoff"

    force_rag_for_actionable_intent = should_force_rag_for_actionable_intent(
        current_intent,
        updated_variables,
        req.message,
    )

    if force_rag_for_actionable_intent:
        route["answer_mode"] = "generate"
        route["needs_rag"] = True
        route["needs_memory"] = True

    skip_generation = should_skip_generation(
        message=req.message,
        variable_updates=extraction.get("updates", {}),
        intent=current_intent,
    )

    if is_user_question(req.message) or force_rag_for_actionable_intent:
        skip_generation = False
        route["answer_mode"] = "generate"
        route["needs_rag"] = True
        route["needs_memory"] = True

    if skip_generation:
        route["answer_mode"] = "no_llm"
        route["needs_rag"] = False
        route["needs_memory"] = False
        route["selected_model_tier"] = "cheap"

    knowledge = []
    knowledge_source = "none"

    cached = get_rag_cache(req.conversation_id, RAG_CACHE_MAX_AGE_MINUTES) if RAG_CACHE_ENABLED else None
    use_cached_followup = bool(cached) and should_use_cached_rag(req.message, {**route, "needs_rag": True})

    if use_cached_followup:
        knowledge = cached.get("compressed_payload") or cached.get("knowledge_payload") or []
        knowledge_source = "cache"
        route["needs_rag"] = True
        route["rag_cache_hit"] = True

    elif route.get("needs_rag", True):
        raw_knowledge = search_knowledge(
            req.assistant_id,
            req.message,
            limit=KNOWLEDGE_TOP_K,
        )

        knowledge = compress_knowledge(raw_knowledge, req.message)
        knowledge_source = "qdrant"
        route["rag_cache_hit"] = False

        if raw_knowledge:
            save_rag_cache(
                conversation_id=req.conversation_id,
                assistant_id=req.assistant_id,
                user_id=req.user_id,
                query=req.message,
                knowledge_payload=raw_knowledge,
                compressed_payload=knowledge,
            )

    if knowledge:
        updated_variables = enrich_variables_from_best_knowledge(updated_variables, knowledge)

    updated_variables = sync_selected_item_state(updated_variables)
    updated_variables = set_workflow_stage(schema, req.assistant_id, updated_variables, current_intent)
    updated_variables = compute_lead_score(updated_variables, current_intent)

    missing_required_variables = calculate_missing_required_variables(
        schema=schema,
        variables=updated_variables,
        intent=current_intent,
        assistant_id=req.assistant_id,
    )

    save_variables(
        req.conversation_id,
        req.assistant_id,
        req.user_id,
        updated_variables,
    )

    if (
        route.get("answer_mode") == "no_llm"
        and is_context_sensitive_variable_update(req.message, extraction.get("updates", {}), knowledge)
    ):
        route["answer_mode"] = "generate"
        route["needs_rag"] = True
        route["needs_memory"] = True

    memories = []

    if route.get("needs_memory", True):
        memories = search_memories(
            req.assistant_id,
            req.user_id,
            req.message,
            limit=MEMORY_TOP_K,
        )

    final_confirmation = build_final_confirmation(schema, req.assistant_id, updated_variables, req.message)
    repair_answer = build_conversation_repair_answer(req.message, updated_variables)
    objection_answer = build_objection_answer(req.message, updated_variables)
    state_answer = deterministic_answer_from_state(req.message, updated_variables, recommended_next_action)

    if final_confirmation:
        answer = final_confirmation
        model = "none"
        tier = "state"
        updated_variables["lead_stage"] = "confirmed"
        updated_variables["workflow_stage"] = "confirmed"

    elif repair_answer:
        answer = repair_answer
        model = "none"
        tier = "state"

    elif objection_answer:
        answer = objection_answer
        model = "none"
        tier = "state"
        updated_variables["needs_human"] = True
        updated_variables["handoff_reason"] = "price_or_objection"

    elif state_answer:
        answer = state_answer
        model = "none"
        tier = "state"

    elif route.get("answer_mode") == "no_llm":
        answer = build_no_llm_answer(
            message=req.message,
            variables=updated_variables,
            variable_updates=extraction.get("updates", {}),
            missing_variables=missing_required_variables,
            recent_messages=recent_messages,
            recommended_next_action=recommended_next_action,
        )
        model = "none"
        tier = "no_llm"

    else:
        answer, model, tier = generate_answer(
            assistant=assistant,
            recent_messages=recent_messages,
            summary=summary,
            variables=updated_variables,
            knowledge=knowledge,
            memories=memories,
            user_message=req.message,
            intent=current_intent,
            missing_variables=missing_required_variables,
            selected_model_tier=route.get("selected_model_tier", "normal"),
        )

    updated_variables = update_asked_questions_from_answer(updated_variables, answer)

    save_variables(
        req.conversation_id,
        req.assistant_id,
        req.user_id,
        updated_variables,
    )

    save_message(
        req.conversation_id,
        req.assistant_id,
        req.user_id,
        "assistant",
        answer,
    )

    if should_update_summary(
        message=req.message,
        answer=answer,
        variables=updated_variables,
        model_tier=tier,
        route=route,
        recent_messages=recent_messages,
    ):
        updated_summary = update_conversation_summary(
            conversation_id=req.conversation_id,
            assistant_id=req.assistant_id,
            user_id=req.user_id,
            variables=updated_variables,
        )
    else:
        updated_summary = summary

    if should_write_memory(
        message=req.message,
        variables=updated_variables,
        model_tier=tier,
        route=route,
        recent_messages=recent_messages,
    ):
        long_term_memories_written = decide_and_write_long_term_memories(
            assistant_id=req.assistant_id,
            user_id=req.user_id,
            conversation_id=req.conversation_id,
            summary=updated_summary,
            recent_messages=get_recent_messages(req.conversation_id, limit=8),
            variables=updated_variables,
        )
    else:
        long_term_memories_written = []

    usage_input_obj = {
        "message": req.message,
        "summary": summary,
        "variables": updated_variables,
        "knowledge": knowledge,
        "memories": memories,
        "route": route,
        "playbook": playbook,
        "assistant_profile": assistant_profile,
    }

    token_usage = build_token_usage_report(
        model_used=model,
        model_tier=tier,
        answer_mode=route.get("answer_mode", "generate"),
        input_obj=usage_input_obj,
        output_text=answer,
        knowledge_source=knowledge_source,
        rag_cache_hit=route.get("rag_cache_hit", False),
    )

    log_estimated_usage(
        assistant_id=req.assistant_id,
        conversation_id=req.conversation_id,
        user_id=req.user_id,
        model=model,
        purpose="chat",
        input_obj=usage_input_obj,
        output_text=answer,
        metadata={
            "mock_mode": MOCK_MODE,
            "knowledge_source": knowledge_source,
            "model_tier": tier,
            "answer_mode": route.get("answer_mode"),
            "needs_rag": route.get("needs_rag"),
            "needs_memory": route.get("needs_memory"),
            "rag_cache_hit": route.get("rag_cache_hit", False),
            "token_usage": token_usage,
        },
    )

    response_payload = {
        "answer": answer,
        "assistant_id": req.assistant_id,
        "conversation_id": req.conversation_id,
        "intent": current_intent,
        "variables": updated_variables,
        "variable_updates": extraction.get("updates", {}),
        "variable_deletions": extraction.get("deletions", []),
        "missing_variables": missing_required_variables,
        "recommended_next_action": recommended_next_action,
        "route": route,
        "knowledge_used": knowledge,
        "knowledge_source": knowledge_source,
        "memories_used": memories,
        "summary": updated_summary,
        "long_term_memories_written": long_term_memories_written,
        "model_used": model,
        "model_tier": tier,
        "token_usage": token_usage,
        "mock_mode": MOCK_MODE,
        "memory_saved": bool(long_term_memories_written),
    }

    return compact_chat_response(response_payload)
