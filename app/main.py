import os
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

try:
    from langchain_community.callbacks.manager import get_openai_callback
except Exception:
    try:
        from langchain.callbacks import get_openai_callback
    except Exception:
        get_openai_callback = None

from app.graph import app_graph


APP_SECRET = os.getenv("APP_SECRET", os.getenv("API_KEY", ""))
DATA_DIR = Path(os.getenv("DATA_DIR", "/app/data"))

ASSISTANTS_DIR = DATA_DIR / "assistants"
SCHEMAS_DIR = DATA_DIR / "schemas"
CONVERSATIONS_DIR = DATA_DIR / "conversations"

ASSISTANTS_DIR.mkdir(parents=True, exist_ok=True)
SCHEMAS_DIR.mkdir(parents=True, exist_ok=True)
CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Agentic RAG API")


class ChatRequest(BaseModel):
    assistant_id: str
    user_id: str
    conversation_id: str
    message: str
    channel: str = "api"

    variables: Dict[str, Any] = Field(default_factory=dict)
    tool_result: Dict[str, Any] = Field(default_factory=dict)
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
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def assistant_path(assistant_id: str) -> Path:
    return ASSISTANTS_DIR / f"{assistant_id}.json"


def schema_path(assistant_id: str) -> Path:
    return SCHEMAS_DIR / f"{assistant_id}.json"


def conversation_path(assistant_id: str, conversation_id: str) -> Path:
    return CONVERSATIONS_DIR / assistant_id / f"{conversation_id}.json"


def load_assistant(assistant_id: str) -> Dict[str, Any]:
    data = safe_json_load(assistant_path(assistant_id), {})

    if not data:
        raise HTTPException(status_code=404, detail=f"Assistant not found: {assistant_id}")

    return data


def load_schema(assistant_id: str) -> Dict[str, Any]:
    return safe_json_load(schema_path(assistant_id), {})


def load_conversation(assistant_id: str, conversation_id: str) -> Dict[str, Any]:
    return safe_json_load(
        conversation_path(assistant_id, conversation_id),
        {
            "variables": {},
            "summary": "",
            "messages": [],
        },
    )


def save_conversation(assistant_id: str, conversation_id: str, data: Dict[str, Any]) -> None:
    safe_json_write(conversation_path(assistant_id, conversation_id), data)


def message_to_dict(message: BaseMessage) -> Dict[str, str]:
    if isinstance(message, HumanMessage):
        role = "user"
    elif isinstance(message, AIMessage):
        role = "assistant"
    else:
        role = "system"

    return {
        "role": role,
        "content": str(message.content),
    }


def dict_to_message(item: Dict[str, Any]) -> BaseMessage:
    role = item.get("role")
    content = item.get("content", "")

    if role == "assistant":
        return AIMessage(content=content)

    return HumanMessage(content=content)


def recent_messages_from_conversation(conversation: Dict[str, Any], limit: int = 12) -> List[BaseMessage]:
    raw_messages = conversation.get("messages", [])

    if not isinstance(raw_messages, list):
        raw_messages = []

    return [dict_to_message(item) for item in raw_messages[-limit:] if isinstance(item, dict)]


def append_conversation_messages(
    conversation: Dict[str, Any],
    user_message: str,
    assistant_answer: str,
    is_tool_result_turn: bool = False,
) -> Dict[str, Any]:
    messages = conversation.get("messages", [])

    if not isinstance(messages, list):
        messages = []

    if not is_tool_result_turn and user_message:
        messages.append({
            "role": "user",
            "content": user_message,
        })

    if assistant_answer:
        messages.append({
            "role": "assistant",
            "content": assistant_answer,
        })

    # Keep recent history compact.
    conversation["messages"] = messages[-30:]

    return conversation


def build_agent_config(assistant_doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Supports two assistant.json shapes:

    1. {
         "assistant_id": "...",
         "system_prompt": "...",
         "agent_config": {...}
       }

    2. {
         "assistant_id": "...",
         "assistant_goal": "...",
         "subagents": [...]
       }

    The graph expects agent_config.
    """
    agent_config = assistant_doc.get("agent_config")

    if isinstance(agent_config, dict):
        return agent_config

    fallback = dict(assistant_doc)
    fallback.pop("system_prompt", None)
    return fallback


def build_language_instruction(assistant_doc: Dict[str, Any], channel: str) -> str:
    agent_config = build_agent_config(assistant_doc)

    language_policy = agent_config.get("language_policy", "")
    conversation_style = agent_config.get("conversation_style", "")

    return (
        f"Channel: {channel}\n"
        f"Language policy: {language_policy}\n"
        f"Conversation style: {conversation_style}\n"
        "Use the customer's language naturally unless the assistant config says otherwise."
    )


def build_route_contract(final_state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Public route contract for n8n.

    This exposes the tool decision clearly without hardcoding any domain logic.
    Tool names and payloads come from the graph planner/manifest.
    """
    planner = final_state.get("planner", {}) or {}
    manifest = final_state.get("manifest", {}) or {}

    return {
        "user_intent": planner.get("user_intent") or manifest.get("user_intent", ""),
        "selected_subagent_id": planner.get("selected_subagent_id") or manifest.get("selected_subagent_id", ""),

        "needs_knowledge": bool(planner.get("needs_knowledge", manifest.get("needs_knowledge", False))),
        "needs_memory": bool(planner.get("needs_memory", manifest.get("needs_memory", False))),
        "needs_tool": bool(planner.get("needs_tool", manifest.get("needs_tool", False))),

        "requested_tool_name": (
            planner.get("requested_tool_name")
            or manifest.get("requested_tool_name")
            or ""
        ),

        "missing_tool_inputs": (
            planner.get("missing_tool_inputs")
            or manifest.get("missing_tool_inputs")
            or []
        ),

        "tool_request_payload": (
            planner.get("tool_request_payload")
            or manifest.get("tool_request_payload")
            or {}
        ),

        "risk_level": planner.get("risk_level") or manifest.get("risk_level", "low"),
        "confidence": planner.get("confidence", manifest.get("confidence", 0.0)),

        "conversation_stage": planner.get("conversation_stage") or manifest.get("conversation_stage", ""),
        "workflow_stage": planner.get("workflow_stage") or manifest.get("workflow_stage", ""),

        "simple_response_mode": bool(
            planner.get("simple_response_mode", manifest.get("simple_response_mode", False))
        ),

        "manifest_profile_used": (
            planner.get("manifest_profile_used")
            or manifest.get("manifest_profile_used")
            or ""
        ),
    }


def build_action_required_contract(final_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Backward-compatible action contract for n8n.

    This replaces old action_required logic with a generic contract.
    No domain-specific fields are hardcoded here.
    The payload comes from planner.tool_request_payload / manifest.tool_request_payload.
    """
    route = build_route_contract(final_state)

    if not route.get("needs_tool"):
        return None

    tool_name = route.get("requested_tool_name") or ""

    if not tool_name:
        return None

    return {
        "type": tool_name,
        "payload": route.get("tool_request_payload") or {},
        "missing_inputs": route.get("missing_tool_inputs") or [],
    }


def should_hide_answer_for_tool_request(route: Dict[str, Any], request: ChatRequest) -> bool:
    """
    If a tool is ready to run, n8n should execute it first.
    So the API response should not send a normal customer-facing answer.

    If inputs are missing, keep the generated answer because it should ask the customer
    for missing information.
    """
    if request.tool_result:
        return False

    if not route.get("needs_tool"):
        return False

    if not route.get("requested_tool_name"):
        return False

    missing_inputs = route.get("missing_tool_inputs") or []

    if missing_inputs:
        return False

    return True


def get_token_usage_from_callback(cb: Any) -> Dict[str, Any]:
    if cb is None:
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
        }

    return {
        "input_tokens": int(getattr(cb, "prompt_tokens", 0) or 0),
        "output_tokens": int(getattr(cb, "completion_tokens", 0) or 0),
        "cost_usd": float(getattr(cb, "total_cost", 0.0) or 0.0),
    }


@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "agentic-rag-api",
    }


@app.post("/assistants")
def save_assistant_endpoint(payload: Dict[str, Any], x_api_key: Optional[str] = Header(default=None)):
    require_api_key(x_api_key)

    assistant_id = payload.get("assistant_id")

    if not assistant_id:
        raise HTTPException(status_code=400, detail="assistant_id is required")

    safe_json_write(assistant_path(assistant_id), payload)

    return {
        "status": "saved",
        "assistant_id": assistant_id,
    }


@app.get("/assistants/{assistant_id}")
def get_assistant_endpoint(assistant_id: str, x_api_key: Optional[str] = Header(default=None)):
    require_api_key(x_api_key)

    return load_assistant(assistant_id)


@app.post("/schemas")
def save_schema_endpoint(payload: Dict[str, Any], x_api_key: Optional[str] = Header(default=None)):
    require_api_key(x_api_key)

    assistant_id = (
        payload.get("assistant_id")
        or payload.get("id")
        or payload.get("name")
    )

    if not assistant_id:
        # Keep compatibility with the assistant you are currently testing.
        assistant_id = "service_center_agentic_rag"

    safe_json_write(schema_path(assistant_id), payload)

    return {
        "status": "saved",
        "assistant_id": assistant_id,
    }


@app.get("/schemas/{assistant_id}")
def get_schema_endpoint(assistant_id: str, x_api_key: Optional[str] = Header(default=None)):
    require_api_key(x_api_key)

    return load_schema(assistant_id)


@app.post("/chat")
def chat(request: ChatRequest, x_api_key: Optional[str] = Header(default=None)):
    require_api_key(x_api_key)

    assistant_doc = load_assistant(request.assistant_id)
    schema = load_schema(request.assistant_id)
    conversation = load_conversation(request.assistant_id, request.conversation_id)

    stored_variables = conversation.get("variables", {})
    if not isinstance(stored_variables, dict):
        stored_variables = {}

    incoming_variables = request.variables or {}
    if not isinstance(incoming_variables, dict):
        incoming_variables = {}

    variables = {
        **stored_variables,
        **incoming_variables,
    }

    messages = recent_messages_from_conversation(conversation, limit=12)
    messages.append(HumanMessage(content=request.message))

    system_prompt = assistant_doc.get("system_prompt", "")
    agent_config = build_agent_config(assistant_doc)
    language_instruction = build_language_instruction(assistant_doc, request.channel)

    initial_state = {
        "messages": messages,
        "assistant_id": request.assistant_id,
        "user_id": request.user_id,
        "conversation_id": request.conversation_id,

        "variables": variables,
        "summary": conversation.get("summary", ""),
        "system_prompt": system_prompt,
        "agent_config": agent_config,
        "language_instruction": language_instruction,
        "schema": schema,
        "tool_result": request.tool_result or {},
    }

    cb = None

    if get_openai_callback is not None:
        with get_openai_callback() as callback:
            final_state = app_graph.invoke(initial_state)
            cb = callback
    else:
        final_state = app_graph.invoke(initial_state)

    route = build_route_contract(final_state)
    action_required = build_action_required_contract(final_state)

    answer = final_state.get("final_answer", "") or ""

    if should_hide_answer_for_tool_request(route, request):
        answer = ""

    final_variables = final_state.get("variables", {}) or {}

    conversation["variables"] = final_variables
    conversation["summary"] = final_state.get("summary", conversation.get("summary", ""))

    is_tool_result_turn = bool(request.tool_result) or request.message == "__tool_result__"
    append_conversation_messages(
        conversation,
        user_message=request.message,
        assistant_answer=answer,
        is_tool_result_turn=is_tool_result_turn,
    )

    save_conversation(request.assistant_id, request.conversation_id, conversation)

    response_payload = {
        "answer": answer,
        "assistant_id": request.assistant_id,
        "conversation_id": request.conversation_id,
        "variables": final_variables,

        # n8n-friendly tool/routing contract
        "route": route,

        # backward-compatible n8n contract
        "action_required": action_required,

        "quality": final_state.get("quality", {}) or {},
        "token_usage": get_token_usage_from_callback(cb),
        "mock_mode": False,
    }

    if request.debug:
        response_payload["debug"] = {
            "manifest": final_state.get("manifest", {}) or {},
            "planner": final_state.get("planner", {}) or {},
            "selected_subagent": final_state.get("selected_subagent", {}) or {},
            "knowledge_items": final_state.get("knowledge_items", []) or [],
            "subagent_analysis": final_state.get("subagent_analysis", {}) or {},
        }

    return response_payload
