import re
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, Header, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.callbacks import get_openai_callback

from app.config import APP_SECRET, MOCK_MODE, RECENT_MESSAGES_LIMIT, validate_runtime_config
from app.graph import app_graph
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
    log_model_usage,
)
from app.rag import ensure_qdrant, ingest_document, search_knowledge, compress_knowledge
from app.variables import apply_variable_patch
from app.memory import update_conversation_summary, decide_and_write_long_term_memories


app = FastAPI(title="No-Hardcoding Agentic RAG API - LangGraph")


class AssistantRequest(BaseModel):
    assistant_id: str
    name: str
    system_prompt: str
    agent_config: Dict[str, Any] = Field(default_factory=dict)


class SchemaRequest(BaseModel):
    assistant_id: str
    schema: Dict[str, Any]


class IngestRequest(BaseModel):
    assistant_id: str
    document_id: str
    title: str
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class KnowledgeSearchRequest(BaseModel):
    assistant_id: str
    query: str
    limit: int = 6


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
    updates: Dict[str, Any] = Field(default_factory=dict)
    deletions: List[str] = Field(default_factory=list)


def check_auth(x_api_key: str):
    if not APP_SECRET:
        return
    if x_api_key != APP_SECRET:
        raise HTTPException(status_code=401, detail="Unauthorized")


def is_arabic_text(text: str) -> bool:
    return bool(re.search(r"[\u0600-\u06FF]", text or ""))


def detect_reply_language_instruction(user_message: str) -> str:
    if is_arabic_text(user_message):
        return """
Language instruction:
- The latest user message is Arabic or Egyptian Arabic.
- Reply in natural Egyptian Arabic unless the user explicitly requests another language.
- Keep names, brands, and technical terms in English only when natural.
"""
    return """
Language instruction:
- Reply in the same language as the latest user message.
"""


def post_response_memory_tasks(assistant_id: str, user_id: str, conversation_id: str, variables: Dict[str, Any], agent_config: Dict[str, Any]):
    updated_summary = update_conversation_summary(conversation_id, assistant_id, user_id, variables)
    decide_and_write_long_term_memories(
        assistant_id=assistant_id,
        user_id=user_id,
        conversation_id=conversation_id,
        summary=updated_summary,
        recent_messages=get_recent_messages(conversation_id, limit=8),
        variables=variables,
        agent_config=agent_config,
    )


@app.on_event("startup")
def startup():
    validate_runtime_config()
    init_db()
    ensure_qdrant()


@app.get("/health")
def health():
    return {"status": "ok", "mock_mode": MOCK_MODE, "architecture": "No-Hardcoding LangGraph Agentic RAG"}


@app.post("/assistants")
def create_assistant(req: AssistantRequest, x_api_key: str = Header(default="")):
    check_auth(x_api_key)
    upsert_assistant(req.assistant_id, req.name, req.system_prompt, req.agent_config)
    return {"status": "saved", "assistant_id": req.assistant_id}


@app.get("/assistants/{assistant_id}")
def read_assistant(assistant_id: str, x_api_key: str = Header(default="")):
    check_auth(x_api_key)
    return get_or_create_assistant(assistant_id)


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
    chunks = ingest_document(req.assistant_id, req.document_id, req.title, req.text, req.metadata)
    upsert_knowledge_document(req.assistant_id, req.document_id, req.title, req.metadata, chunks)
    return {"status": "ingested", "assistant_id": req.assistant_id, "document_id": req.document_id, "chunks": chunks}


@app.get("/knowledge/{assistant_id}")
def read_knowledge_documents(assistant_id: str, x_api_key: str = Header(default="")):
    check_auth(x_api_key)
    return {"assistant_id": assistant_id, "documents": list_knowledge_documents(assistant_id)}


@app.post("/knowledge/search")
def knowledge_search(req: KnowledgeSearchRequest, x_api_key: str = Header(default="")):
    check_auth(x_api_key)
    results = search_knowledge(req.assistant_id, req.query, req.limit)
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


@app.post("/chat")
def chat(req: ChatRequest, background_tasks: BackgroundTasks, x_api_key: str = Header(default="")):
    check_auth(x_api_key)

    assistant = get_or_create_assistant(req.assistant_id)
    agent_config = assistant["agent_config"]

    ensure_conversation(req.conversation_id, req.assistant_id, req.user_id, req.channel)
    save_message(req.conversation_id, req.assistant_id, req.user_id, "user", req.message)

    recent_messages = get_recent_messages(req.conversation_id, limit=RECENT_MESSAGES_LIMIT)
    summary = get_summary(req.conversation_id)
    variables = get_variables(req.conversation_id)
    schema = get_schema(req.assistant_id)

    formatted_history = []
    for msg in recent_messages:
        if msg.get("role") == "user":
            formatted_history.append(HumanMessage(content=msg.get("content", "")))
        elif msg.get("role") == "assistant":
            formatted_history.append(AIMessage(content=msg.get("content", "")))

    initial_state = {
        "messages": formatted_history,
        "assistant_id": req.assistant_id,
        "user_id": req.user_id,
        "conversation_id": req.conversation_id,
        "variables": variables,
        "summary": summary,
        "system_prompt": assistant["system_prompt"],
        "agent_config": agent_config,
        "language_instruction": detect_reply_language_instruction(req.message),
        "schema": schema,
        "tool_result": req.tool_result or {},
    }

    with get_openai_callback() as cb:
        final_state = app_graph.invoke(initial_state)
        exact_input_tokens = cb.prompt_tokens
        exact_output_tokens = cb.completion_tokens
        exact_total_cost = cb.total_cost

    answer = final_state.get("final_answer") or final_state["messages"][-1].content
    updated_variables = final_state.get("variables", variables)
    planner = final_state.get("planner", {})
    selected_subagent = final_state.get("selected_subagent", {})
    subagent_analysis = final_state.get("subagent_analysis", {})
    quality = final_state.get("quality", {})

    save_message(req.conversation_id, req.assistant_id, req.user_id, "assistant", answer)
    save_variables(req.conversation_id, req.assistant_id, req.user_id, updated_variables)

    log_model_usage(
        assistant_id=req.assistant_id,
        conversation_id=req.conversation_id,
        user_id=req.user_id,
        model="langgraph-no-hardcoding-agentic-rag",
        purpose="chat_turn",
        input_tokens=exact_input_tokens,
        output_tokens=exact_output_tokens,
        estimated_cost_usd=exact_total_cost,
        metadata={
            "planner": planner,
            "selected_subagent": selected_subagent,
            "subagent_analysis": subagent_analysis,
            "quality": quality,
        },
    )

    background_tasks.add_task(
        post_response_memory_tasks,
        req.assistant_id,
        req.user_id,
        req.conversation_id,
        updated_variables,
        agent_config,
    )

    return {
        "answer": answer,
        "assistant_id": req.assistant_id,
        "conversation_id": req.conversation_id,
        "variables": updated_variables,
        "route": {
            "user_intent": planner.get("user_intent"),
            "selected_subagent_id": planner.get("selected_subagent_id"),
            "needs_knowledge": planner.get("needs_knowledge"),
            "needs_memory": planner.get("needs_memory"),
            "needs_tool": planner.get("needs_tool"),
            "risk_level": planner.get("risk_level"),
            "confidence": planner.get("confidence"),
        },
        "quality": quality,
        "token_usage": {
            "input_tokens": exact_input_tokens,
            "output_tokens": exact_output_tokens,
            "cost_usd": exact_total_cost,
        },
        "mock_mode": MOCK_MODE,
    }
