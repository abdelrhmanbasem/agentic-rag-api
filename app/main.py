from typing import Dict, Any

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from app.config import (
    APP_SECRET,
    MOCK_MODE,
    RECENT_MESSAGES_LIMIT,
    KNOWLEDGE_TOP_K,
    MEMORY_TOP_K,
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
)
from app.llm import chat_text, route_message, model_for_tier
from app.rag import (
    ensure_qdrant,
    ingest_document,
    search_knowledge,
    search_memories,
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


@app.on_event("startup")
def startup():
    init_db()
    ensure_qdrant()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "mock_mode": MOCK_MODE,
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

    return {
        "assistant_id": req.assistant_id,
        "query": req.query,
        "count": len(results),
        "results": results,
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


def build_mock_answer(intent, variables, missing_variables, knowledge):
    if intent == "car_search":
        brand = variables.get("car_brand", "a suitable car")
        budget = variables.get("budget_max")
        condition = variables.get("car_condition", "")

        if knowledge:
            first_match = knowledge[0]
            title = first_match.get("title", "the knowledge base")
            text = first_match.get("text", "")

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

    system_prompt = f"""
{assistant["system_prompt"]}

Tone:
{assistant["tone"]}

Memory policy:
{assistant["memory_policy"]}

Rules:
- Use current variables when relevant.
- Use retrieved knowledge when relevant.
- Use long-term memories only when relevant.
- Ask for missing important variables naturally.
- Be concise and useful.
- Do not reveal internal routing, memory, or RAG.
"""

    context = f"""
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
    return answer, model, selected_model_tier


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

    save_variables(
        req.conversation_id,
        req.assistant_id,
        req.user_id,
        updated_variables,
    )

    skip_generation = should_skip_generation(
        message=req.message,
        variable_updates=extraction.get("updates", {}),
        intent=extraction.get("intent", "general_question"),
    )

    if skip_generation:
        route["answer_mode"] = "no_llm"
        route["needs_rag"] = False
        route["needs_memory"] = False
        route["selected_model_tier"] = "cheap"

    knowledge = []
    if route.get("needs_rag", True):
        knowledge = search_knowledge(req.assistant_id, req.message, limit=KNOWLEDGE_TOP_K)

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
            missing_variables=extraction.get("missing_variables", []),
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
            intent=extraction.get("intent", "general_question"),
            missing_variables=extraction.get("missing_variables", []),
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

    recommended_next_action = "continue_conversation"

    if extraction.get("missing_variables"):
        recommended_next_action = "ask_clarifying_question"

    if extraction.get("intent") == "booking_request":
        recommended_next_action = "collect_booking_details"

    if extraction.get("intent") == "complaint":
        recommended_next_action = "consider_human_handoff"

    if extraction.get("intent") == "urgent_medical_issue":
        recommended_next_action = "urgent_human_handoff"

    return {
        "answer": answer,
        "assistant_id": req.assistant_id,
        "conversation_id": req.conversation_id,
        "intent": extraction.get("intent", "general_question"),
        "variables": updated_variables,
        "variable_updates": extraction.get("updates", {}),
        "variable_deletions": extraction.get("deletions", []),
        "missing_variables": extraction.get("missing_variables", []),
        "recommended_next_action": recommended_next_action,
        "route": route,
        "knowledge_used": knowledge,
        "memories_used": memories,
        "summary": updated_summary,
        "long_term_memories_written": long_term_memories_written,
        "model_used": model,
        "model_tier": tier,
        "mock_mode": MOCK_MODE,
        "memory_saved": bool(long_term_memories_written),
    }
