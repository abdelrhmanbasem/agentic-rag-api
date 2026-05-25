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


def calculate_missing_required_variables(
    schema: Dict[str, Any],
    variables: Dict[str, Any],
    intent: str = "general_question",
) -> list[str]:
    missing = []
    variables = variables or {}
    schema = schema or {}
    intent = intent or "general_question"

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


def detect_reply_language_instruction(user_message: str) -> str:
    message = user_message or ""

    if is_arabic_text(message):
        return """
LANGUAGE RULE:
- The latest user message is Arabic or Egyptian Arabic.
- You MUST reply in natural Egyptian Arabic.
- Do NOT reply in English.
- Keep car brands and model names like BMW, Mercedes, Hyundai, 320i, C180 in English if needed.
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
    """
    Safety layer:
    If the user wrote Arabic/Egyptian Arabic but the model answered mostly in English,
    rewrite the answer into natural Egyptian Arabic before returning it.
    """
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
                "Keep car brands/models like BMW, 320i, Mercedes, C180 in English. "
                "Keep prices, years, and kilometers exactly the same. "
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


def build_mock_answer(intent, variables, missing_variables, knowledge):
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
        "مانيوال",
        "كام كيلو",
        "بكام",
        "سعرها",
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

    updated_variables = apply_variable_patch(
        existing_variables,
        extraction.get("updates", {}),
        extraction.get("deletions", []),
    )

    if extraction.get("intent"):
        updated_variables["intent"] = extraction["intent"]

    current_intent = extraction.get("intent") or updated_variables.get("intent") or "general_question"

    missing_required_variables = calculate_missing_required_variables(
        schema=schema,
        variables=updated_variables,
        intent=current_intent,
    )

    if not missing_required_variables:
        missing_required_variables = extraction.get("missing_variables", [])

    save_variables(
        req.conversation_id,
        req.assistant_id,
        req.user_id,
        updated_variables,
    )

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

    skip_generation = should_skip_generation(
        message=req.message,
        variable_updates=extraction.get("updates", {}),
        intent=current_intent,
    )

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
