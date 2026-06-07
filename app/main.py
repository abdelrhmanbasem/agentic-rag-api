# Architecture batch: 6.33-config-driven-main-error-handling-no-hardcoding
import copy
import hashlib
import json
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

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
from app.subagents.base import apply_variable_patch, deep_get


APP_SECRET = os.getenv("APP_SECRET", os.getenv("API_KEY", ""))
DATA_DIR = Path(os.getenv("DATA_DIR", "/app/data"))

ASSISTANTS_DIR = DATA_DIR / "assistants"
SCHEMAS_DIR = DATA_DIR / "schemas"
CONVERSATIONS_DIR = DATA_DIR / "conversations"

ASSISTANTS_DIR.mkdir(parents=True, exist_ok=True)
SCHEMAS_DIR.mkdir(parents=True, exist_ok=True)
CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)

API_TITLE = os.getenv("API_TITLE", "Modular Agentic LangGraph API")
API_SERVICE_NAME = os.getenv("API_SERVICE_NAME", "modular-agentic-langgraph-api")

app = FastAPI(title=API_TITLE)

_CONVERSATION_LOCKS: Dict[str, threading.RLock] = {}
_CONVERSATION_LOCKS_LAST_USED: Dict[str, float] = {}
_CONVERSATION_LOCKS_GUARD = threading.RLock()


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

    # Optional generic idempotency / observability fields.
    # They are not required, preserving backward compatibility.
    request_id: Optional[str] = None
    message_id: Optional[str] = None
    idempotency_key: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


def require_api_key(x_api_key: Optional[str]) -> None:
    if APP_SECRET and x_api_key != APP_SECRET:
        raise HTTPException(status_code=401, detail="Invalid API key")


def now_ms() -> int:
    return int(time.time() * 1000)


def safe_json_load(path: Path, default: Any) -> Any:
    if not path.exists():
        return copy.deepcopy(default)

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return copy.deepcopy(default)


def safe_json_write(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, indent=2)
    tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}.{uuid.uuid4().hex}")

    tmp_path.write_text(payload, encoding="utf-8")
    tmp_path.replace(path)


def safe_storage_id(value: str, fallback: str = "default") -> str:
    text = str(value or "").strip()

    if not text:
        return fallback

    # Filesystem-safe only; does not change the actual conversation_id used in DB/state.
    safe = re.sub(r"[^A-Za-z0-9_.@+-]+", "_", text)
    safe = safe.strip("._")

    return safe or fallback


def assistant_path(assistant_id: str) -> Path:
    return ASSISTANTS_DIR / f"{safe_storage_id(assistant_id)}.json"


def schema_path(assistant_id: str) -> Path:
    return SCHEMAS_DIR / f"{safe_storage_id(assistant_id)}.json"


def safe_conversation_id(conversation_id: str) -> str:
    return safe_storage_id(conversation_id)


def conversation_path(assistant_id: str, conversation_id: str) -> Path:
    return CONVERSATIONS_DIR / safe_storage_id(assistant_id) / f"{safe_conversation_id(conversation_id)}.json"


def get_config_path(config: Dict[str, Any], path: str, default: Any = None) -> Any:
    current: Any = config

    for part in str(path or "").split("."):
        if not part:
            continue

        if not isinstance(current, dict):
            return default

        if part not in current:
            return default

        current = current.get(part)

    return current if current is not None else default


def get_config_int(config: Dict[str, Any], path: str, default: int) -> int:
    value = get_config_path(config, path, default)

    try:
        return int(value)
    except Exception:
        return default


def get_config_bool(config: Dict[str, Any], path: str, default: bool) -> bool:
    value = get_config_path(config, path, default)

    if isinstance(value, bool):
        return value

    if value is None:
        return default

    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def compact_list(value: Any, limit: int) -> List[Any]:
    if not isinstance(value, list):
        return []

    return value[-max(limit, 0):]


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
            "summary": "",
            "processed_requests": []
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
        row_assistant_id = state_row.get("assistant_id")

        # If the DB stores assistant_id and it conflicts, avoid leaking state
        # across assistants that accidentally reuse a conversation_id.
        if row_assistant_id and str(row_assistant_id) != str(assistant_id):
            legacy = load_conversation_legacy(assistant_id, conversation_id)
        else:
            state = state_row.get("state", {})

            if not isinstance(state, dict):
                state = {}

            messages = state.get("messages", [])
            traces = state.get("traces", [])
            processed_requests = state.get("processed_requests", [])

            if not isinstance(messages, list):
                messages = []

            if not isinstance(traces, list):
                traces = []

            if not isinstance(processed_requests, list):
                processed_requests = []

            return {
                "variables": state_row.get("variables", {}) if isinstance(state_row.get("variables", {}), dict) else {},
                "messages": messages,
                "traces": traces,
                "processed_requests": processed_requests,
                "summary": state_row.get("summary", "") or "",
                "message_count": state_row.get("message_count", 0) or len(messages),
                "channel": state_row.get("channel", channel) or channel,
                "source": "postgres"
            }
    else:
        legacy = load_conversation_legacy(assistant_id, conversation_id)

    if not isinstance(legacy, dict):
        legacy = {}

    messages = legacy.get("messages", []) if isinstance(legacy.get("messages", []), list) else []

    return {
        "variables": legacy.get("variables", {}) if isinstance(legacy.get("variables", {}), dict) else {},
        "messages": messages,
        "traces": legacy.get("traces", []) if isinstance(legacy.get("traces", []), list) else [],
        "processed_requests": legacy.get("processed_requests", []) if isinstance(legacy.get("processed_requests", []), list) else [],
        "summary": legacy.get("summary", "") or "",
        "message_count": legacy.get("message_count", len(messages)) if isinstance(legacy.get("message_count", len(messages)), int) else len(messages),
        "channel": channel,
        "source": "legacy_json"
    }


def get_state_limits(assistant_config: Dict[str, Any]) -> Dict[str, int]:
    return {
        "messages": get_config_int(assistant_config, "request_handling.max_stored_messages", 80),
        "traces": get_config_int(assistant_config, "request_handling.max_stored_traces", 50),
        "history": get_config_int(assistant_config, "request_handling.max_history_messages", 24),
        "processed_requests": get_config_int(assistant_config, "request_handling.max_processed_requests", 120),
    }


def save_conversation_pg(
    request: ChatRequest,
    assistant_config: Dict[str, Any],
    conversation: Dict[str, Any],
    variables: Dict[str, Any]
) -> None:
    limits = get_state_limits(assistant_config)

    state = {
        "messages": compact_list(conversation.get("messages", []), limits["messages"]),
        "traces": compact_list(conversation.get("traces", []), limits["traces"]),
        "processed_requests": compact_list(conversation.get("processed_requests", []), limits["processed_requests"])
    }

    message_count = conversation.get("message_count")
    if not isinstance(message_count, int):
        message_count = len(state.get("messages", []))

    save_conversation_state(
        conversation_id=request.conversation_id,
        assistant_id=request.assistant_id,
        user_id=request.user_id,
        channel=request.channel,
        state=state,
        variables=variables,
        summary=conversation.get("summary", "") or "",
        message_count=message_count
    )


def append_messages(
    conversation: Dict[str, Any],
    user_message: str,
    assistant_answer: str,
    request_key: str = "",
    trace_id: str = "",
    assistant_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    assistant_config = assistant_config or {}
    limits = get_state_limits(assistant_config)
    messages = conversation.get("messages", [])

    if not isinstance(messages, list):
        messages = []

    timestamp_ms = now_ms()

    messages.append({
        "role": "user",
        "content": user_message,
        "request_key": request_key,
        "trace_id": trace_id,
        "created_at_ms": timestamp_ms
    })

    messages.append({
        "role": "assistant",
        "content": assistant_answer,
        "request_key": request_key,
        "trace_id": trace_id,
        "created_at_ms": timestamp_ms
    })

    conversation["messages"] = compact_list(messages, limits["messages"])
    conversation["message_count"] = int(conversation.get("message_count", 0) or 0) + 2
    return conversation


def append_trace(
    conversation: Dict[str, Any],
    trace: Dict[str, Any],
    assistant_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    assistant_config = assistant_config or {}
    limits = get_state_limits(assistant_config)
    traces = conversation.get("traces", [])

    if not isinstance(traces, list):
        traces = []

    traces.append(trace)
    conversation["traces"] = compact_list(traces, limits["traces"])
    return conversation


def append_processed_request(
    conversation: Dict[str, Any],
    request_key: str,
    trace: Dict[str, Any],
    assistant_config: Dict[str, Any]
) -> Dict[str, Any]:
    if not request_key:
        return conversation

    limits = get_state_limits(assistant_config)
    processed = conversation.get("processed_requests", [])

    if not isinstance(processed, list):
        processed = []

    processed.append({
        "request_key": request_key,
        "trace_id": trace.get("trace_id", ""),
        "final_answer": trace.get("final_answer", ""),
        "created_at_ms": trace.get("created_at_ms", now_ms()),
        "selected_subagent": trace.get("selected_subagent", ""),
        "chained_subagent": trace.get("chained_subagent", ""),
        "detected_intents": trace.get("detected_intents", []),
        "action": trace.get("action", "reply"),
        "tool_calls_used": trace.get("tool_calls_used", 0),
    })

    # Keep newest per key.
    deduped: Dict[str, Dict[str, Any]] = {}
    for item in processed:
        if not isinstance(item, dict):
            continue
        key = str(item.get("request_key") or "").strip()
        if key:
            deduped[key] = item

    conversation["processed_requests"] = compact_list(list(deduped.values()), limits["processed_requests"])
    return conversation


def find_processed_request(conversation: Dict[str, Any], request_key: str) -> Optional[Dict[str, Any]]:
    if not request_key:
        return None

    processed = conversation.get("processed_requests", [])

    if not isinstance(processed, list):
        return None

    for item in reversed(processed):
        if not isinstance(item, dict):
            continue

        if str(item.get("request_key") or "") == request_key:
            return item

    return None


def build_langchain_history(
    conversation: Dict[str, Any],
    latest_user_message: str,
    assistant_config: Optional[Dict[str, Any]] = None
) -> List[BaseMessage]:
    assistant_config = assistant_config or {}
    limits = get_state_limits(assistant_config)
    output: List[BaseMessage] = []
    messages = conversation.get("messages", [])

    if not isinstance(messages, list):
        messages = []

    for item in messages[-limits["history"]:]:
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


def merge_variables(
    existing: Dict[str, Any],
    incoming: Dict[str, Any],
    assistant_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    assistant_config = assistant_config or {}

    if not isinstance(existing, dict):
        existing = {}

    if not isinstance(incoming, dict):
        return dict(existing)

    allow_source_of_truth = get_config_bool(
        assistant_config,
        "request_variable_policy.allow_source_of_truth",
        False
    )

    allow_empty_updates = get_config_bool(
        assistant_config,
        "request_variable_policy.allow_empty_updates",
        False
    )

    return apply_variable_patch(
        existing,
        incoming,
        [],
        allow_source_of_truth=allow_source_of_truth,
        assistant_config=assistant_config,
        allow_empty_updates=allow_empty_updates
    )


def attach_request_metadata(
    variables: Dict[str, Any],
    request: ChatRequest,
    request_key: str = "",
    trace_id: str = ""
) -> Dict[str, Any]:
    updated = dict(variables or {})
    updated["conversation_id"] = request.conversation_id
    updated["user_id"] = request.user_id
    updated["channel"] = request.channel

    if request_key:
        updated["request_key"] = request_key

    if trace_id:
        updated["trace_id"] = trace_id

    if isinstance(request.metadata, dict) and request.metadata:
        updated["request_metadata"] = request.metadata

    if request.message_id:
        updated["message_id"] = request.message_id

    if request.request_id:
        updated["request_id"] = request.request_id

    return updated


def build_graph_input(
    request: ChatRequest,
    assistant_config: Dict[str, Any],
    schema: Dict[str, Any],
    conversation: Dict[str, Any],
    variables: Dict[str, Any]
) -> Dict[str, Any]:
    return {
        "messages": build_langchain_history(conversation, request.message, assistant_config),
        "assistant_id": request.assistant_id,
        "user_id": request.user_id,
        "conversation_id": request.conversation_id,
        "variables": variables,
        "summary": conversation.get("summary", ""),
        "system_prompt": assistant_config.get("system_prompt", ""),
        "agent_config": assistant_config,
        "schema": schema,
        "tool_result": {},
        "multi_tool_results": [],
        "multi_intents": [],
        "parallel_tool_requests": [],
        "knowledge_queries": [],
        "multi_knowledge": [],
        "response_synthesis": {},
        "language_instruction": assistant_config.get("language_policy", "")
    }


def get_client_request_key(request: ChatRequest, assistant_config: Dict[str, Any]) -> str:
    direct = (
        request.idempotency_key
        or request.message_id
        or request.request_id
        or ""
    )

    if direct:
        return str(direct).strip()

    metadata = request.metadata if isinstance(request.metadata, dict) else {}

    metadata_keys = get_config_path(
        assistant_config,
        "request_handling.idempotency_metadata_keys",
        ["idempotency_key", "message_id", "request_id"]
    )

    if not isinstance(metadata_keys, list):
        metadata_keys = ["idempotency_key", "message_id", "request_id"]

    for key in metadata_keys:
        value = metadata.get(str(key))
        if value:
            return str(value).strip()

    return ""


def get_trace_id(request: ChatRequest, request_key: str = "") -> str:
    seed = request_key or f"{request.assistant_id}|{request.conversation_id}|{time.time_ns()}|{uuid.uuid4().hex}"
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]
    return f"trace_{digest}"


def build_debug_trace(
    request: ChatRequest,
    result: Dict[str, Any],
    variables: Dict[str, Any],
    answer: str,
    request_key: str = "",
    trace_id: str = "",
    duration_ms: int = 0,
    state_source: str = "",
    persistence_errors: Optional[List[str]] = None,
    graph_error: Optional[str] = None
) -> Dict[str, Any]:
    manifest = result.get("manifest", {}) or {}
    tool_result = result.get("tool_result", {}) or {}
    quality = result.get("quality", {}) or {}

    return {
        "trace_id": trace_id,
        "request_key": request_key,
        "created_at_ms": now_ms(),
        "duration_ms": duration_ms,
        "assistant_id": request.assistant_id,
        "conversation_id": request.conversation_id,
        "user_id": request.user_id,
        "channel": request.channel,
        "message": request.message,
        "message_id": request.message_id or "",
        "selected_subagent": manifest.get("selected_subagent_id", ""),
        "chained_subagent": manifest.get("chained_subagent_id", ""),
        "detected_intents": manifest.get("detected_intents", []),
        "multi_intents": manifest.get("multi_intents", []),
        "manifest": manifest,
        "tool_result": tool_result,
        "multi_tool_results": result.get("multi_tool_results", []) or tool_result.get("multi_tool_results", []) if isinstance(tool_result, dict) else [],
        "knowledge_queries": result.get("knowledge_queries", []),
        "multi_knowledge": result.get("multi_knowledge", []),
        "response_synthesis": result.get("response_synthesis", {}) or manifest.get("response_synthesis", {}),
        "subagent_analysis": result.get("subagent_analysis", {}) or {},
        "memory_writer": result.get("memory_writer", {}) or {},
        "quality": quality,
        "state_after": variables,
        "final_answer": answer,
        "state_source": state_source,
        "persistence_errors": persistence_errors or [],
        "graph_error": graph_error or "",
        "action": tool_result.get("action", "reply") if isinstance(tool_result, dict) else "reply",
        "tool_calls_used": tool_result.get("tool_calls_used", 0) if isinstance(tool_result, dict) else 0
    }


def make_response_payload(
    request: ChatRequest,
    answer: str,
    variables: Dict[str, Any],
    result: Dict[str, Any],
    trace: Dict[str, Any],
    duplicate: bool = False
) -> Dict[str, Any]:
    manifest = result.get("manifest", {}) or {}
    tool_result = result.get("tool_result", {}) or {}

    response = {
        "answer": answer,
        "assistant_id": request.assistant_id,
        "conversation_id": request.conversation_id,
        "request_key": trace.get("request_key", ""),
        "trace_id": trace.get("trace_id", ""),
        "duplicate": duplicate,
        "variables": variables,
        "selected_subagent": manifest.get("selected_subagent_id", trace.get("selected_subagent", "")),
        "chained_subagent": manifest.get("chained_subagent_id", trace.get("chained_subagent", "")),
        "detected_intents": manifest.get("detected_intents", trace.get("detected_intents", [])),
        "action": tool_result.get("action", trace.get("action", "reply")) if isinstance(tool_result, dict) else trace.get("action", "reply"),
        "tool_calls_used": tool_result.get("tool_calls_used", trace.get("tool_calls_used", 0)) if isinstance(tool_result, dict) else trace.get("tool_calls_used", 0)
    }

    if request.debug:
        response["debug"] = trace
        response["config_source"] = get_config_source(request.assistant_id)
        response["state_source"] = trace.get("state_source", "")

    return response


def save_messages_safely(
    request: ChatRequest,
    answer: str,
    persistence_errors: List[str]
) -> None:
    for role, content in [
        ("user", request.message),
        ("assistant", answer)
    ]:
        try:
            save_message(
                conversation_id=request.conversation_id,
                assistant_id=request.assistant_id,
                user_id=request.user_id,
                role=role,
                content=content
            )
        except Exception as exc:
            persistence_errors.append(f"save_message:{role}:{type(exc).__name__}: {exc}")


def get_lock_key(assistant_id: str, conversation_id: str) -> str:
    return f"{assistant_id}::{conversation_id}"


def cleanup_conversation_locks(max_locks: int = 2000) -> None:
    if len(_CONVERSATION_LOCKS) <= max_locks:
        return

    ordered = sorted(
        _CONVERSATION_LOCKS_LAST_USED.items(),
        key=lambda item: item[1]
    )

    remove_count = max(len(_CONVERSATION_LOCKS) - max_locks, 0)

    for key, _last_used in ordered[:remove_count]:
        _CONVERSATION_LOCKS.pop(key, None)
        _CONVERSATION_LOCKS_LAST_USED.pop(key, None)


def get_conversation_lock(assistant_id: str, conversation_id: str) -> threading.RLock:
    key = get_lock_key(assistant_id, conversation_id)

    with _CONVERSATION_LOCKS_GUARD:
        if key not in _CONVERSATION_LOCKS:
            _CONVERSATION_LOCKS[key] = threading.RLock()
        _CONVERSATION_LOCKS_LAST_USED[key] = time.time()
        cleanup_conversation_locks()

        return _CONVERSATION_LOCKS[key]



def clean_config_text(value: Any) -> str:
    if value in [None, "", [], {}]:
        return ""

    return str(value).strip()


def get_config_list(config: Dict[str, Any], path: str, default: Optional[List[Any]] = None) -> List[Any]:
    value = get_config_path(config, path, default if default is not None else [])

    if isinstance(value, list):
        return value

    return default if default is not None else []


def get_fallback_messages(assistant_config: Dict[str, Any]) -> Dict[str, str]:
    fallback = assistant_config.get("fallback_messages", {})

    if not isinstance(fallback, dict):
        return {}

    output: Dict[str, str] = {}

    for key, value in fallback.items():
        text = clean_config_text(value)
        if text:
            output[str(key)] = text

    return output


def get_configured_message(
    assistant_config: Dict[str, Any],
    path: str,
    default: str = "",
    **format_values: Any
) -> str:
    value = clean_config_text(get_config_path(assistant_config, path, ""))

    if not value:
        value = clean_config_text(default)

    if not value:
        return ""

    if format_values:
        try:
            return value.format(**format_values)
        except Exception:
            return value

    return value


def get_api_error_message(
    assistant_config: Dict[str, Any],
    key: str,
    default: str,
    **format_values: Any
) -> str:
    # API error messages are developer/client-facing, not the assistant's final
    # answer. Still, assistants can override them in config when desired.
    return get_configured_message(
        assistant_config=assistant_config,
        path=f"api_error_messages.{key}",
        default=default,
        **format_values
    )


def get_error_answer(assistant_config: Dict[str, Any], error_type: str = "graph_error") -> str:
    """
    Config-driven final-answer fallback.

    User-facing error/empty-answer wording must come from domain_bundle.json
    under fallback_messages. Python does not infer language-specific wording.
    The last English fallback is an emergency platform default only.
    """
    fallback = get_fallback_messages(assistant_config)

    configured_order = get_config_list(
        assistant_config,
        "request_handling.error_fallback_order",
        []
    )

    order: List[str] = []

    if error_type:
        order.append(str(error_type))

    for key in configured_order:
        key_text = str(key or "").strip()
        if key_text and key_text not in order:
            order.append(key_text)

    for key in ["default_final", "empty_answer", "graph_error", "emergency_final"]:
        if key not in order:
            order.append(key)

    for key in order:
        answer = clean_config_text(fallback.get(key))
        if answer:
            return answer

    return get_configured_message(
        assistant_config=assistant_config,
        path="request_handling.emergency_error_answer",
        default="Something went wrong. Please try again."
    )


@app.get("/health")
def health():
    return {
        "ok": True,
        "service": API_SERVICE_NAME
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
    x_api_key: Optional[str] = Header(default=None),
    x_request_id: Optional[str] = Header(default=None)
):
    require_api_key(x_api_key)

    if x_request_id and not request.request_id:
        request = request.copy(update={"request_id": x_request_id})

    assistant_config, schema = load_assistant_and_schema(request.assistant_id)

    if not assistant_config:
        raise HTTPException(
            status_code=404,
            detail=f"Assistant not found: {request.assistant_id}"
        )

    if not str(request.message or "").strip():
        raise HTTPException(
            status_code=400,
            detail=get_api_error_message(assistant_config, "message_required", "message is required")
        )

    max_message_chars = get_config_int(
        assistant_config,
        "request_handling.max_message_chars",
        12000
    )

    if len(str(request.message)) > max_message_chars:
        raise HTTPException(
            status_code=400,
            detail=get_api_error_message(assistant_config, "message_too_long", "message is too long")
        )

    lock_timeout = get_config_int(
        assistant_config,
        "request_handling.conversation_lock_timeout_seconds",
        30
    )

    lock = get_conversation_lock(request.assistant_id, request.conversation_id)

    acquired = lock.acquire(timeout=max(lock_timeout, 1))

    if not acquired:
        raise HTTPException(
            status_code=409,
            detail=get_api_error_message(
                assistant_config,
                "conversation_busy",
                "Conversation is busy; retry shortly"
            )
        )

    try:
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

        request_key = get_client_request_key(request, assistant_config)
        trace_id = get_trace_id(request, request_key)

        duplicate_record = find_processed_request(conversation, request_key)

        if duplicate_record:
            variables = conversation.get("variables", {})
            if not isinstance(variables, dict):
                variables = {}

            trace = {
                "trace_id": duplicate_record.get("trace_id", trace_id),
                "request_key": request_key,
                "created_at_ms": now_ms(),
                "assistant_id": request.assistant_id,
                "conversation_id": request.conversation_id,
                "user_id": request.user_id,
                "channel": request.channel,
                "message": request.message,
                "final_answer": duplicate_record.get("final_answer", ""),
                "selected_subagent": duplicate_record.get("selected_subagent", ""),
                "chained_subagent": duplicate_record.get("chained_subagent", ""),
                "detected_intents": duplicate_record.get("detected_intents", []),
                "action": duplicate_record.get("action", "reply"),
                "tool_calls_used": duplicate_record.get("tool_calls_used", 0),
                "state_after": variables,
                "state_source": conversation.get("source", ""),
                "duplicate": True,
            }

            return make_response_payload(
                request=request,
                answer=duplicate_record.get("final_answer", ""),
                variables=variables,
                result={},
                trace=trace,
                duplicate=True
            )

        existing_variables = conversation.get("variables", {})

        if not isinstance(existing_variables, dict):
            existing_variables = {}

        variables = merge_variables(existing_variables, request.variables, assistant_config)
        variables = attach_request_metadata(
            variables=variables,
            request=request,
            request_key=request_key,
            trace_id=trace_id
        )

        graph_input = build_graph_input(
            request=request,
            assistant_config=assistant_config,
            schema=schema,
            conversation=conversation,
            variables=variables
        )

        started_at = time.time()
        graph_error = ""
        result: Dict[str, Any] = {}

        try:
            result = app_graph.invoke(graph_input)
            answer = str(result.get("final_answer", "") or "").strip()
            variables = result.get("variables", variables)

            if not isinstance(variables, dict):
                variables = {}

            if not answer:
                answer = get_error_answer(assistant_config, "empty_answer")

        except Exception as exc:
            graph_error = f"{type(exc).__name__}: {exc}"
            return_error_answer = get_config_bool(
                assistant_config,
                "request_handling.return_error_answer",
                True
            )

            if not return_error_answer:
                raise HTTPException(
                    status_code=500,
                    detail=get_api_error_message(
                        assistant_config,
                        "graph_execution_failed",
                        "Graph execution failed"
                    )
                )

            answer = get_error_answer(assistant_config, "graph_error")
            result = {
                "manifest": {},
                "tool_result": {
                    "ok": False,
                    "error_type": "graph_execution_error",
                    "error": graph_error,
                    "action": "reply",
                    "tool_calls_used": 0
                },
                "quality": {}
            }

        duration_ms = int((time.time() - started_at) * 1000)
        persistence_errors: List[str] = []

        trace = build_debug_trace(
            request=request,
            result=result,
            variables=variables,
            answer=answer,
            request_key=request_key,
            trace_id=trace_id,
            duration_ms=duration_ms,
            state_source=conversation.get("source", ""),
            persistence_errors=persistence_errors,
            graph_error=graph_error
        )

        append_messages(
            conversation=conversation,
            user_message=request.message,
            assistant_answer=answer,
            request_key=request_key,
            trace_id=trace_id,
            assistant_config=assistant_config
        )
        append_trace(conversation, trace, assistant_config)
        append_processed_request(conversation, request_key, trace, assistant_config)

        save_messages_safely(request, answer, persistence_errors)

        conversation["variables"] = variables

        try:
            save_conversation_pg(
                request=request,
                assistant_config=assistant_config,
                conversation=conversation,
                variables=variables
            )
        except Exception as exc:
            persistence_errors.append(f"save_conversation_state:{type(exc).__name__}: {exc}")
            save_conversation_legacy(request.assistant_id, request.conversation_id, conversation)

        if persistence_errors:
            trace["persistence_errors"] = persistence_errors
            # Best-effort second save so the trace includes persistence warnings.
            try:
                conversation["traces"][-1] = trace
                save_conversation_pg(
                    request=request,
                    assistant_config=assistant_config,
                    conversation=conversation,
                    variables=variables
                )
            except Exception:
                pass

        return make_response_payload(
            request=request,
            answer=answer,
            variables=variables,
            result=result,
            trace=trace,
            duplicate=False
        )

    finally:
        lock.release()


@app.post("/conversations/{assistant_id}/{conversation_id}/clear")
def clear_conversation_endpoint(
    assistant_id: str,
    conversation_id: str,
    x_api_key: Optional[str] = Header(default=None)
):
    require_api_key(x_api_key)

    lock = get_conversation_lock(assistant_id, conversation_id)

    acquired = lock.acquire(timeout=30)

    if not acquired:
        raise HTTPException(status_code=409, detail="Conversation is busy; retry shortly")

    try:
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

    finally:
        lock.release()


@app.get("/conversations/{assistant_id}/{conversation_id}")
def get_conversation_endpoint(
    assistant_id: str,
    conversation_id: str,
    x_api_key: Optional[str] = Header(default=None)
):
    require_api_key(x_api_key)

    state_row = load_conversation_state(conversation_id)

    if state_row:
        row_assistant_id = state_row.get("assistant_id")
        if row_assistant_id and str(row_assistant_id) != str(assistant_id):
            raise HTTPException(status_code=404, detail="Conversation not found for assistant")

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
