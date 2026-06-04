import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, List

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from app.graph import app_graph
from app.config_loader import load_assistant_and_schema, get_config_source
from app.db import (
    init_db,
    ensure_conversation,
    save_message,
    get_recent_messages,
    load_conversation_state,
    save_conversation_state,
    clear_conversation_data,
)


APP_SECRET = os.getenv("APP_SECRET", os.getenv("API_KEY", ""))
DATA_DIR = Path(os.getenv("DATA_DIR", "/app/data"))

ASSISTANTS_DIR = DATA_DIR / "assistants"
SCHEMAS_DIR = DATA_DIR / "schemas"
CONVERSATIONS_DIR = DATA_DIR / "conversations"

ASSISTANTS_DIR.mkdir(parents=True, exist_ok=True)
SCHEMAS_DIR.mkdir(parents=True, exist_ok=True)
CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Modular Agentic LangGraph API")


@app.on_event("startup")
def startup_event():
    init_db()


class ChatRequest(BaseModel):
    assistant_id: str
    user_id: str
    conversation_id: str
    message: str
    channel: str = "api"
    variables: Dict[str, Any] = Field(default_factory=dict)
    debug: bool = False


def require_api_key(x_api_key: Optional[str]) -> None:
    if APP_SECRET and x_api_key != APP_SECRET:
        raise HTTPException(status_code=401, detail="Invalid API key")


def safe_json_load(path: Path, default: Any) -> Any:
    if not path.exists():
        return default

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def safe_json_write(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def assistant_path(assistant_id: str) -> Path:
    return ASSISTANTS_DIR / f"{assistant_id}.json"


def schema_path(assistant_id: str) -> Path:
    return SCHEMAS_DIR / f"{assistant_id}.json"


def safe_conversation_id(conversation_id: str) -> str:
    return conversation_id.replace("/", "_")


def conversation_path(assistant_id: str, conversation_id: str) -> Path:
    return CONVERSATIONS_DIR / assistant_id / f"{safe_conversation_id(conversation_id)}.json"


def load_assistant_legacy(assistant_id: str) -> Dict[str, Any]:
    data = safe_json_load(assistant_path(assistant_id), {})

    if not data:
        raise HTTPException(status_code=404, detail=f"Assistant not found: {assistant_id}")

    return data


def load_schema_legacy(assistant_id: str) -> Dict[str, Any]:
    return safe_json_load(schema_path(assistant_id), {})


def load_conversation_legacy(assistant_id: str, conversation_id: str) -> Dict[str, Any]:
    return safe_json_load(
        conversation_path(assistant_id, conversation_id),
        {
            "variables": {},
            "messages": [],
            "traces": [],
            "summary": ""
        }
    )


def save_conversation_legacy(assistant_id: str, conversation_id: str, data: Dict[str, Any]) -> None:
    safe_json_write(
        conversation_path(assistant_id, conversation_id),
        data
    )


def normalize_pg_conversation_state(
    assistant_id: str,
    conversation_id: str,
    user_id: str,
    channel: str
) -> Dict[str, Any]:
    state_row = load_conversation_state(conversation_id)

    if state_row:
        state = state_row.get("state", {})

        if not isinstance(state, dict):
            state = {}

        messages = state.get("messages", [])
        traces = state.get("traces", [])

        if not isinstance(messages, list):
            messages = []

        if not isinstance(traces, list):
            traces = []

        return {
            "variables": state_row.get("variables", {}) if isinstance(state_row.get("variables", {}), dict) else {},
            "messages": messages,
            "traces": traces,
            "summary": state_row.get("summary", "") or "",
            "message_count": state_row.get("message_count", 0) or 0,
            "channel": state_row.get("channel", channel) or channel,
            "source": "postgres"
        }

    legacy = load_conversation_legacy(assistant_id, conversation_id)

    if not isinstance(legacy, dict):
        legacy = {}

    return {
        "variables": legacy.get("variables", {}) if isinstance(legacy.get("variables", {}), dict) else {},
        "messages": legacy.get("messages", []) if isinstance(legacy.get("messages", []), list) else [],
        "traces": legacy.get("traces", []) if isinstance(legacy.get("traces", []), list) else [],
        "summary": legacy.get("summary", "") or "",
        "message_count": len(legacy.get("messages", [])) if isinstance(legacy.get("messages", []), list) else 0,
        "channel": channel,
        "source": "legacy_json"
    }


def save_conversation_pg(
    request: ChatRequest,
    conversation: Dict[str, Any],
    variables: Dict[str, Any]
) -> None:
    state = {
        "messages": conversation.get("messages", [])[-60:],
        "traces": conversation.get("traces", [])[-40:]
    }

    save_conversation_state(
        conversation_id=request.conversation_id,
        assistant_id=request.assistant_id,
        user_id=request.user_id,
        channel=request.channel,
        state=state,
        variables=variables,
        summary=conversation.get("summary", "") or "",
        message_count=len(state.get("messages", []))
    )


def append_messages(
    conversation: Dict[str, Any],
    user_message: str,
    assistant_answer: str
) -> Dict[str, Any]:
    messages = conversation.get("messages", [])

    if not isinstance(messages, list):
        messages = []

    messages.append({
        "role": "user",
        "content": user_message
    })

    messages.append({
        "role": "assistant",
        "content": assistant_answer
    })

    conversation["messages"] = messages[-60:]
    return conversation


def append_trace(conversation: Dict[str, Any], trace: Dict[str, Any]) -> Dict[str, Any]:
    traces = conversation.get("traces", [])

    if not isinstance(traces, list):
        traces = []

    traces.append(trace)
    conversation["traces"] = traces[-40:]
    return conversation


def build_langchain_history(
    conversation: Dict[str, Any],
    latest_user_message: str
) -> List[BaseMessage]:
    output: List[BaseMessage] = []
    messages = conversation.get("messages", [])

    if not isinstance(messages, list):
        messages = []

    for item in messages[-24:]:
        if not isinstance(item, dict):
            continue

        role = item.get("role")
        content = item.get("content")

        if not content:
            continue

        if role == "user":
            output.append(HumanMessage(content=str(content)))
        elif role == "assistant":
            output.append(AIMessage(content=str(content)))

    output.append(HumanMessage(content=latest_user_message))
    return output


def merge_variables(existing: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(existing or {})

    if not isinstance(incoming, dict):
        return merged

    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_variables(merged[key], value)
        else:
            merged[key] = value

    return merged


def attach_request_metadata(
    variables: Dict[str, Any],
    request: ChatRequest
) -> Dict[str, Any]:
    updated = dict(variables or {})
    updated["conversation_id"] = request.conversation_id
    updated["user_id"] = request.user_id
    updated["channel"] = request.channel
    return updated


def build_graph_input(
    request: ChatRequest,
    assistant_config: Dict[str, Any],
    schema: Dict[str, Any],
    conversation: Dict[str, Any],
    variables: Dict[str, Any]
) -> Dict[str, Any]:
    return {
        "messages": build_langchain_history(conversation, request.message),
        "assistant_id": request.assistant_id,
        "user_id": request.user_id,
        "conversation_id": request.conversation_id,
        "variables": variables,
        "summary": conversation.get("summary", ""),
        "system_prompt": assistant_config.get("system_prompt", ""),
        "agent_config": assistant_config,
        "schema": schema,
        "tool_result": {},
        "language_instruction": assistant_config.get("language_policy", "")
    }


def build_debug_trace(
    request: ChatRequest,
    result: Dict[str, Any],
    variables: Dict[str, Any],
    answer: str
) -> Dict[str, Any]:
    manifest = result.get("manifest", {}) or {}
    tool_result = result.get("tool_result", {}) or {}
    quality = result.get("quality", {}) or {}

    return {
        "message": request.message,
        "selected_subagent": manifest.get("selected_subagent_id", ""),
        "chained_subagent": manifest.get("chained_subagent_id", ""),
        "detected_intents": manifest.get("detected_intents", []),
        "manifest": manifest,
        "tool_result": tool_result,
        "subagent_analysis": result.get("subagent_analysis", {}) or {},
        "memory_writer": result.get("memory_writer", {}) or {},
        "quality": quality,
        "state_after": variables,
        "final_answer": answer
    }


@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "modular-agentic-langgraph-api"
    }


@app.post("/assistants")
def save_assistant_endpoint(
    payload: Dict[str, Any],
    x_api_key: Optional[str] = Header(default=None)
):
    require_api_key(x_api_key)

    assistant_id = payload.get("assistant_id")

    if not assistant_id:
        raise HTTPException(status_code=400, detail="assistant_id is required")

    safe_json_write(assistant_path(assistant_id), payload)

    return {
        "status": "saved",
        "assistant_id": assistant_id
    }


@app.get("/assistants/{assistant_id}")
def get_assistant_endpoint(
    assistant_id: str,
    x_api_key: Optional[str] = Header(default=None)
):
    require_api_key(x_api_key)

    assistant_config, _schema = load_assistant_and_schema(assistant_id)

    if assistant_config:
        return assistant_config

    return load_assistant_legacy(assistant_id)


@app.post("/schemas")
def save_schema_endpoint(
    payload: Dict[str, Any],
    x_api_key: Optional[str] = Header(default=None)
):
    require_api_key(x_api_key)

    assistant_id = payload.get("assistant_id") or payload.get("id") or payload.get("name")

    if not assistant_id:
        raise HTTPException(status_code=400, detail="assistant_id is required")

    safe_json_write(schema_path(assistant_id), payload)

    return {
        "status": "saved",
        "assistant_id": assistant_id
    }


@app.get("/schemas/{assistant_id}")
def get_schema_endpoint(
    assistant_id: str,
    x_api_key: Optional[str] = Header(default=None)
):
    require_api_key(x_api_key)

    _assistant_config, schema = load_assistant_and_schema(assistant_id)

    if schema:
        return schema

    return load_schema_legacy(assistant_id)


@app.get("/config-source/{assistant_id}")
def get_config_source_endpoint(
    assistant_id: str,
    x_api_key: Optional[str] = Header(default=None)
):
    require_api_key(x_api_key)
    return get_config_source(assistant_id)


@app.post("/chat")
def chat(
    request: ChatRequest,
    x_api_key: Optional[str] = Header(default=None)
):
    require_api_key(x_api_key)

    assistant_config, schema = load_assistant_and_schema(request.assistant_id)

    if not assistant_config:
        raise HTTPException(
            status_code=404,
            detail=f"Assistant not found: {request.assistant_id}"
        )

    ensure_conversation(
        conversation_id=request.conversation_id,
        assistant_id=request.assistant_id,
        user_id=request.user_id,
        channel=request.channel
    )

    conversation = normalize_pg_conversation_state(
        assistant_id=request.assistant_id,
        conversation_id=request.conversation_id,
        user_id=request.user_id,
        channel=request.channel
    )

    existing_variables = conversation.get("variables", {})

    if not isinstance(existing_variables, dict):
        existing_variables = {}

    variables = merge_variables(existing_variables, request.variables)
    variables = attach_request_metadata(variables, request)

    graph_input = build_graph_input(
        request=request,
        assistant_config=assistant_config,
        schema=schema,
        conversation=conversation,
        variables=variables
    )

    result = app_graph.invoke(graph_input)

    answer = str(result.get("final_answer", "") or "").strip()
    variables = result.get("variables", variables)

    if not isinstance(variables, dict):
        variables = {}

    trace = build_debug_trace(
        request=request,
        result=result,
        variables=variables,
        answer=answer
    )

    append_messages(conversation, request.message, answer)
    append_trace(conversation, trace)

    save_message(
        conversation_id=request.conversation_id,
        assistant_id=request.assistant_id,
        user_id=request.user_id,
        role="user",
        content=request.message
    )

    save_message(
        conversation_id=request.conversation_id,
        assistant_id=request.assistant_id,
        user_id=request.user_id,
        role="assistant",
        content=answer
    )

    conversation["variables"] = variables

    save_conversation_pg(
        request=request,
        conversation=conversation,
        variables=variables
    )

    tool_result = result.get("tool_result", {}) or {}
    manifest = result.get("manifest", {}) or {}

    response = {
        "answer": answer,
        "assistant_id": request.assistant_id,
        "conversation_id": request.conversation_id,
        "variables": variables,
        "selected_subagent": manifest.get("selected_subagent_id", ""),
        "chained_subagent": manifest.get("chained_subagent_id", ""),
        "detected_intents": manifest.get("detected_intents", []),
        "action": tool_result.get("action", "reply") if isinstance(tool_result, dict) else "reply",
        "tool_calls_used": tool_result.get("tool_calls_used", 0) if isinstance(tool_result, dict) else 0
    }

    if request.debug:
        response["debug"] = trace
        response["config_source"] = get_config_source(request.assistant_id)
        response["state_source"] = conversation.get("source", "postgres")

    return response


@app.post("/conversations/{assistant_id}/{conversation_id}/clear")
def clear_conversation_endpoint(
    assistant_id: str,
    conversation_id: str,
    x_api_key: Optional[str] = Header(default=None)
):
    require_api_key(x_api_key)

    clear_conversation_data(conversation_id)

    path = conversation_path(assistant_id, conversation_id)

    if path.exists():
        path.unlink()

    return {
        "ok": True,
        "cleared": True,
        "assistant_id": assistant_id,
        "conversation_id": conversation_id
    }


@app.get("/conversations/{assistant_id}/{conversation_id}")
def get_conversation_endpoint(
    assistant_id: str,
    conversation_id: str,
    x_api_key: Optional[str] = Header(default=None)
):
    require_api_key(x_api_key)

    state_row = load_conversation_state(conversation_id)

    if state_row:
        return {
            "assistant_id": assistant_id,
            "conversation_id": conversation_id,
            "source": "postgres",
            "variables": state_row.get("variables", {}),
            "summary": state_row.get("summary", ""),
            "state": state_row.get("state", {}),
            "message_count": state_row.get("message_count", 0),
            "updated_at": state_row.get("updated_at")
        }

    legacy = load_conversation_legacy(assistant_id, conversation_id)
    legacy["source"] = "legacy_json"
    return legacy
