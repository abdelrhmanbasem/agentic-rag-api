# app/main.py
# Smart stateful Agentic RAG API - LangGraph Upgraded
import re
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

# --- New LangGraph Imports ---
from app.graph import app_graph 
from langchain_core.messages import HumanMessage, AIMessage

# --- Existing Imports ---
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
from app.rag import (
    ensure_qdrant,
    ingest_document,
    search_knowledge,
    search_memories,
    compress_knowledge,
)
from app.variables import apply_variable_patch
from app.memory import update_conversation_summary, decide_and_write_long_term_memories
from app.super_efficiency import compact_chat_response
from app.structured_inventory import upsert_structured_inventory_from_text

app = FastAPI(title="Agentic RAG API - LangGraph Edition")


# --- Pydantic Models ---
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


# --- Helpers ---
def check_auth(x_api_key: str):
    if x_api_key != APP_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")

def is_arabic_text(text: str) -> bool:
    return bool(re.search(r"[\u0600-\u06FF]", text or ""))

def detect_reply_language_instruction(user_message: str) -> str:
    if is_arabic_text(user_message):
        return """
LANGUAGE RULE:
- The latest user message is Arabic or Egyptian Arabic.
- You MUST reply in natural Egyptian Arabic.
- Do NOT reply in English.
- Keep brand names, model names, doctor names, service names, and technical terms in English only when natural.
- Use natural Egyptian Arabic phrasing.
- Keep the answer friendly, clear, and short.
"""
    return """
LANGUAGE RULE:
- Reply in the same language as the latest user message.
- If the latest user message is English, reply in English.
- Keep the answer friendly, clear, and short.
"""


# --- API Endpoints ---
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
        "architecture": "LangGraph"
    }

@app.post("/assistants")
def create_assistant(req: AssistantRequest, x_api_key: str = Header(default="")):
    check_auth(x_api_key)
    upsert_assistant(req.assistant_id, req.name, req.system_prompt, req.tone, req.memory_policy)
    return {"status": "saved", "assistant_id": req.assistant_id}

@app.post("/schemas")
def create_schema(req: SchemaRequest, x_api_key: str = Header(default="")):
    check_auth(x_api_key)
    save_schema(req.assistant_id, req.schema)
    return {"status": "saved", "assistant_id": req.assistant_id, "schema_keys": list(req.schema.keys())}

@app.get("/schemas/{assistant_id}")
def read_schema(assistant_id: str, x_api_key: str = Header(default="")):
    check_auth(x_api_key)
    return {"assistant_id": assistant_id, "schema": get_schema(assistant_id)}

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
    return {"assistant_id": assistant_id, "documents": list_knowledge_documents(assistant_id)}

@app.post("/knowledge/search")
def knowledge_search(req: KnowledgeSearchRequest, x_api_key: str = Header(default="")):
    check_auth(x_api_key)
    results = search_knowledge(assistant_id=req.assistant_id, query=req.query, limit=req.limit)
    compressed = compress_knowledge(results, req.query)
    return {"assistant_id": req.assistant_id, "query": req.query, "count": len(compressed), "results": compressed}

@app.get("/variables/{conversation_id}")
def read_variables(conversation_id: str, x_api_key: str = Header(default="")):
    check_auth(x_api_key)
    return {"conversation_id": conversation_id, "variables": get_variables(conversation_id)}

@app.patch("/variables")
def patch_variables(req: PatchVariablesRequest, x_api_key: str = Header(default="")):
    check_auth(x_api_key)
    existing = get_variables(req.conversation_id)
    updated = apply_variable_patch(existing, req.updates, req.deletions)
    save_variables(req.conversation_id, req.assistant_id, req.user_id, updated)
    return {"status": "updated", "conversation_id": req.conversation_id, "variables": updated}

@app.get("/memories/{assistant_id}/{user_id}")
def read_user_memories(assistant_id: str, user_id: str, x_api_key: str = Header(default="")):
    check_auth(x_api_key)
    return {"assistant_id": assistant_id, "user_id": user_id, "memories": list_long_term_memories(assistant_id, user_id)}


# ------------------------------------------------------------
# The New LangGraph Chat Endpoint
# ------------------------------------------------------------
@app.post("/chat")
def chat(req: ChatRequest, x_api_key: str = Header(default="")):
    check_auth(x_api_key)

    # 1. Initialize dependencies
    assistant = get_or_create_assistant(req.assistant_id)
    ensure_conversation(req.conversation_id, req.assistant_id, req.user_id, req.channel)

    # Save the incoming user message immediately
    save_message(req.conversation_id, req.assistant_id, req.user_id, "user", req.message)

    # 2. Fetch context from DB
    recent_messages = get_recent_messages(req.conversation_id, limit=RECENT_MESSAGES_LIMIT)
    summary = get_summary(req.conversation_id)
    existing_variables = get_variables(req.conversation_id)

    # 3. Format history for LangGraph
    formatted_history = []
    for msg in recent_messages:
        role = msg.get("role")
        content = msg.get("content")
        if role == "user":
            formatted_history.append(HumanMessage(content=content))
        elif role == "assistant":
            formatted_history.append(AIMessage(content=content))

    # 4. Initialize the Graph State
    initial_state = {
        "messages": formatted_history,
        "assistant_id": req.assistant_id,
        "user_id": req.user_id,
        "conversation_id": req.conversation_id,
        "variables": existing_variables,
        "summary": summary,
        "system_prompt": assistant["system_prompt"],
        "tone": assistant.get("tone", "clear, helpful, concise"),
        "language_instruction": detect_reply_language_instruction(req.message)
    }

    # 5. Run the LangGraph orchestration (The Brain takes over here)
    final_state = app_graph.invoke(initial_state)

    # 6. Extract the generated human-like answer and any updated variables
    answer = final_state["messages"][-1].content
    updated_variables = final_state.get("variables", existing_variables)

    # 7. Save Assistant Answer
    save_message(req.conversation_id, req.assistant_id, req.user_id, "assistant", answer)
    save_variables(req.conversation_id, req.assistant_id, req.user_id, updated_variables)

    # 8. Memory & Summary Management (Your existing smart memory logic!)
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

    # 9. Return the structured payload expected by your frontend/n8n
    response_payload = {
        "answer": answer,
        "assistant_id": req.assistant_id,
        "conversation_id": req.conversation_id,
        "variables": updated_variables,
        "missing_variables": [], # Handled naturally in chat now
        "route": {"answer_mode": "langgraph_agentic"},
        "summary": updated_summary,
        "long_term_memories_written": long_term_memories_written,
        "mock_mode": MOCK_MODE,
        "memory_saved": bool(long_term_memories_written),
    }

    return compact_chat_response(response_payload)
