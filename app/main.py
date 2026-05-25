import re
from typing import Dict, Any

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

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
from app.memory import (
    update_conversation_summary,
    decide_and_write_long_term_memories,
)
from app.policies import should_skip_generation, build_no_llm_answer

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
    metadata: Dict[str, Any] = {}


class PatchVariablesRequest(BaseModel):
    assistant_id: str
    user_id: str
    conversation_id: str
    updates: Dict[str, Any] = {}
    deletions: list[str] = []


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
    }

    return aliases.get(intent, intent)


def calculate_missing_required_variables(
    schema: Dict[str, Any],
    variables: Dict[str, Any],
    intent: str = "general_question",
) -> list[str]:
    missing = []
    variables = variables or {}
    schema = schema or {}
    intent = normalize_intent(intent)

    for key, config in schema.items():
        if key == "intent":
            continue

        if not isinstance(config, dict):
            continue

        value = variables.get(key)
        is_empty = value is None or value == ""

        if not is_empty:
            continue

        globally_required = config.get("required", False)
        required_for_intents = config.get("required_for_intents", []) or []
        not_required_for_intents = config.get("not_required_for_intents", []) or []

        required_for_intents = [normalize_intent(i) for i in required_for_intents]
        not_required_for_intents = [normalize_intent(i) for i in not_required_for_intents]

        if intent in not_required_for_intents:
            continue

        if globally_required:
            missing.append(key)
            continue

        if intent in required_for_intents:
            missing.append(key)
            continue

    return missing


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


def should_force_rag_for_actionable_intent(intent: str, variables: Dict[str, Any], message: str) -> bool:
    """
    Production rule:
    If the user has given enough searchable/actionable info, do not stop at no-LLM acknowledgement.
    Use RAG/knowledge immediately.

    This is generic enough for future assistants:
    - car_search + brand/budget/condition/transmission -> search inventory
    - service_question + service_needed -> search service/pricing docs
    - booking_request + service/date/time -> search relevant booking/policy docs if available
    - insurance_question + insurance_provider/service -> search insurance/policy docs
    """
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
        return bool(
            variables.get("insurance_provider")
            or variables.get("service_needed")
        )

    if intent == "viewing_request":
        return True

    return False


def detect_reply_language_instruction(user_message: str) -> str:
    message = user_message or ""

    if is_arabic_text(message):
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
                "Use natural Egyptian Arabic phrasing. "
                "Return only the rewritten answer."
            ),
        },
        {
            "role": "user",
            "content": answer,
        },
    ]

    return chat_text(model, rewrite_messages, max_tokens=300)


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

    upsert_assistant(
        req.assistant_id,
        req.name,
        req.system_prompt,
        req.tone,
        req.memory_policy,
    )

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

    return {
        "status": "ingested",
        "assistant_id": req.assistant_id,
        "document_id": req.document_id,
        "chunks": chunks,
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


def extract_prices_from_text(text):
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


def choose_best_knowledge_for_variables(knowledge, variables):
    if not knowledge:
        return None

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
            if any(alias in text for alias in aliases):
                score += 120
            else:
                score -= 60

        if budget:
            prices = extract_prices_from_text(text)

            if prices:
                best_price = min(prices)

                if best_price <= budget:
                    score += 90
                else:
                    score -= 90

        if variables.get("transmission"):
            transmission = str(variables.get("transmission")).lower()
            if transmission in text:
                score += 30

        score += float(item.get("score", 0.0) or 0.0) * 5

        return score

    ranked = sorted(knowledge, key=score_item, reverse=True)
    return ranked[0]


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

    return facts


def enrich_variables_from_best_knowledge(variables, knowledge):
    variables = dict(variables or {})

    if not knowledge:
        return variables

    best_item = choose_best_knowledge_for_variables(knowledge, variables)
    if not best_item:
        return variables

    facts = extract_knowledge_facts(best_item)

    for key, value in facts.items():
        if value is not None and value != "":
            variables[key] = value

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
    }

    if not any(key in updates for key in sensitive_update_keys):
        return False

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

    return any(marker in text for marker in context_markers)


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
                return (
                    f"Got it — you're looking for a {condition} {brand} up to {budget}. "
                    f"I found a relevant match in {title}: {text[:220]}"
                )

            return (
                f"Got it — you're interested in {brand}. "
                f"I found relevant information in {title}: {text[:220]}"
            )

        if budget:
            return (
                f"Got it — you're looking for a {condition} {brand} up to {budget}. "
                f"I can help narrow that down. What model or body type do you prefer?"
            )

        return f"Got it — you're interested in {brand}. What budget range should I look within?"

    if intent == "viewing_request":
        return "Sure — I can help arrange a viewing. I just need the remaining viewing details."

    if intent == "booking_request":
        return "Sure — I can help with booking. What day and time works best for you?"

    if intent == "complaint":
        return (
            "I’m sorry about that. I’ll help you resolve it. "
            "Can you share a few more details so I can escalate it properly?"
        )

    if intent == "urgent_medical_issue":
        return (
            "This may need urgent medical attention. Please contact emergency services "
            "or the clinic directly right away."
        )

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
- Use retrieved knowledge when relevant.
- Use long-term memories only when relevant.
- Ask for missing important variables naturally.
- Always follow the LANGUAGE RULE above.
- Continue the conversation naturally after acknowledging what was captured.
- If the user asks about buying, searching, booking, viewing, pricing, availability, or services, do not stop at acknowledgement; ask the next useful question.
- If the user asks a follow-up question about a retrieved item, answer it directly from the current knowledge/cache before asking another question.
- If the user provides a budget and the current matched item has a known price, tell whether it fits the budget before asking the next useful question.
- If knowledge contains prices, kilometers, model years, or availability, use them accurately.
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

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": context},
    ]
    messages.extend(recent_messages)

    answer = chat_text(model, messages, max_tokens=700)
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
    ]

    return any(marker in text for marker in followup_markers)


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

    route = route_message(
        assistant=assistant,
        summary=summary,
        variables=existing_variables,
        recent_messages=recent_messages,
        user_message=req.message,
    )

    route["intent_hint"] = normalize_intent(route.get("intent_hint", "general_question"))

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

    extraction["intent"] = normalize_intent(extraction.get("intent", "general_question"))

    updated_variables = apply_variable_patch(
        existing_variables,
        extraction.get("updates", {}),
        extraction.get("deletions", []),
    )

    current_intent = normalize_intent(
        extraction.get("intent") or updated_variables.get("intent") or "general_question"
    )

    updated_variables["intent"] = current_intent

    missing_required_variables = calculate_missing_required_variables(
        schema=schema,
        variables=updated_variables,
        intent=current_intent,
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

    if is_user_question(req.message):
        skip_generation = False
        route["answer_mode"] = "generate"
        route["needs_rag"] = True
        route["needs_memory"] = True

    if force_rag_for_actionable_intent:
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

    cached = None
    if RAG_CACHE_ENABLED:
        cached = get_rag_cache(req.conversation_id, RAG_CACHE_MAX_AGE_MINUTES)

    use_cached_followup = bool(cached) and should_use_cached_rag(
        req.message,
        {
            **route,
            "needs_rag": True,
        },
    )

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

        missing_required_variables = calculate_missing_required_variables(
            schema=schema,
            variables=updated_variables,
            intent=current_intent,
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

    if route.get("answer_mode") == "no_llm":
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

    long_term_memories_written = decide_and_write_long_term_memories(
        assistant_id=req.assistant_id,
        user_id=req.user_id,
        conversation_id=req.conversation_id,
        summary=updated_summary,
        recent_messages=get_recent_messages(req.conversation_id, limit=8),
        variables=updated_variables,
    )

    log_estimated_usage(
        assistant_id=req.assistant_id,
        conversation_id=req.conversation_id,
        user_id=req.user_id,
        model=model,
        purpose="chat",
        input_obj={
            "message": req.message,
            "summary": summary,
            "variables": updated_variables,
            "knowledge": knowledge,
            "memories": memories,
            "route": route,
        },
        output_text=answer,
        metadata={
            "mock_mode": MOCK_MODE,
            "knowledge_source": knowledge_source,
            "model_tier": tier,
            "answer_mode": route.get("answer_mode"),
            "needs_rag": route.get("needs_rag"),
            "needs_memory": route.get("needs_memory"),
            "rag_cache_hit": route.get("rag_cache_hit", False),
        },
    )

    return {
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
        "mock_mode": MOCK_MODE,
        "memory_saved": bool(long_term_memories_written),
    }
