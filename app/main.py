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

    # Supported modes:
    # - normal: original behavior
    # - planner: return clean planner/data_request output for n8n/Activepieces
    # - final_with_external_context: use external_context.result and return final answer only
    mode: str = "normal"

    variables: Dict[str, Any] = Field(default_factory=dict)
    tool_result: Dict[str, Any] = Field(default_factory=dict)

    # Used by n8n/Activepieces-style fetch-then-answer flow.
    planner_result: Dict[str, Any] = Field(default_factory=dict)
    conversation_state: Dict[str, Any] = Field(default_factory=dict)
    external_context: Dict[str, Any] = Field(default_factory=dict)

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


def merge_dicts(*items: Any) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}

    for item in items:
        if isinstance(item, dict):
            merged.update(item)

    return merged


def compact_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keeps false/0 values but removes None and empty strings/lists/dicts.
    """
    out: Dict[str, Any] = {}

    for key, value in data.items():
        if value is None:
            continue
        if value == "":
            continue
        if value == []:
            continue
        if value == {}:
            continue
        out[key] = value

    return out


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

    if request.mode == "final_with_external_context":
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


def extract_data_request_from_response(
    route: Dict[str, Any],
    action_required: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Generic planner output for n8n/Activepieces.
    Does not hardcode any operation names.
    """
    if action_required and isinstance(action_required, dict):
        payload = action_required.get("payload") or {}
        if isinstance(payload, dict):
            return payload

    payload = route.get("tool_request_payload") or {}
    if isinstance(payload, dict):
        return payload

    return {}


def build_planner_response(
    request: ChatRequest,
    response_payload: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Converts the old graph route/action format into a cleaner n8n planner format.

    This is generic:
    - It does not know Apps Script.
    - It does not know service center sheet structure.
    - It only exposes route/action/data_request from the assistant planner.
    """
    route = response_payload.get("route") or {}
    action_required = response_payload.get("action_required")

    missing_inputs = []
    if isinstance(action_required, dict):
        missing_inputs = action_required.get("missing_inputs") or []
    if not missing_inputs:
        missing_inputs = route.get("missing_tool_inputs") or []

    requested_tool_name = ""
    if isinstance(action_required, dict):
        requested_tool_name = action_required.get("type") or ""
    if not requested_tool_name:
        requested_tool_name = route.get("requested_tool_name") or ""

    data_request = extract_data_request_from_response(route, action_required)

    needs_external_data = bool(
        requested_tool_name
        and data_request
        and not missing_inputs
    )

    answer = response_payload.get("answer") or ""
    quality = response_payload.get("quality") or {}

    # If planner needs external data, do not send customer-facing waiting text.
    if needs_external_data:
        answer = ""

    return {
        "answer": answer,
        "assistant_id": response_payload.get("assistant_id"),
        "conversation_id": response_payload.get("conversation_id"),
        "variables": response_payload.get("variables") or {},

        "mode": "needs_external_data" if needs_external_data else "answer_directly",
        "needs_external_data": needs_external_data,
        "data_source": requested_tool_name,
        "data_request": data_request if needs_external_data else {},

        "route": route,
        "action_required": action_required if not needs_external_data else None,

        "missing_inputs": missing_inputs,
        "quality": quality,
        "token_usage": response_payload.get("token_usage") or {},
        "mock_mode": response_payload.get("mock_mode", False),
    }


def get_external_result(external_context: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(external_context, dict):
        return {}

    result = external_context.get("result")

    if isinstance(result, dict):
        return result

    return {}


def get_external_operation(external_context: Dict[str, Any], planner_result: Dict[str, Any]) -> str:
    if isinstance(external_context, dict):
        op = external_context.get("operation")
        if op:
            return str(op)

        result = external_context.get("result")
        if isinstance(result, dict) and result.get("operation"):
            return str(result.get("operation"))

    if isinstance(planner_result, dict):
        data_request = planner_result.get("data_request")
        if isinstance(data_request, dict) and data_request.get("operation"):
            return str(data_request.get("operation"))

        route = planner_result.get("route")
        if isinstance(route, dict):
            payload = route.get("tool_request_payload")
            if isinstance(payload, dict) and payload.get("operation"):
                return str(payload.get("operation"))

    return ""


def build_tool_result_from_external_context(request: ChatRequest) -> Dict[str, Any]:
    """
    Converts n8n/Activepieces fetched data into the existing graph's tool_result format.

    This is generic:
    - Does not hardcode sheet names.
    - Does not hardcode URLs.
    - Does not call external APIs.
    - Uses whatever n8n sends in external_context.result.
    """
    external_context = request.external_context or {}
    planner_result = request.planner_result or {}
    result = get_external_result(external_context)

    if not result:
        return {}

    operation = get_external_operation(external_context, planner_result)
    data_source = ""

    if isinstance(planner_result, dict):
        data_source = (
            planner_result.get("data_source")
            or planner_result.get("tool_name")
            or ""
        )

    if not data_source and isinstance(external_context, dict):
        data_source = (
            external_context.get("source")
            or external_context.get("tool_name")
            or ""
        )

    tool_result = dict(result)

    tool_result.setdefault("operation", operation)
    tool_result.setdefault("tool_name", data_source)
    tool_result.setdefault("status", "success" if result.get("ok") is True else "failed")
    tool_result.setdefault("raw", result)

    return tool_result


def infer_state_updates_from_external_context(request: ChatRequest) -> Dict[str, Any]:
    """
    State extraction from external_context.result using common field names if present.

    This is still generic:
    - It does not call a fixed tool.
    - It does not know any URL.
    - It only preserves fields if the external result already contains them.
    """
    result = get_external_result(request.external_context or {})
    operation = get_external_operation(request.external_context or {}, request.planner_result or {})

    if not result:
        return {}

    branch = (
        result.get("branch")
        or result.get("nearest_branch")
        or result.get("location_branch")
        or ""
    )

    section = (
        result.get("section")
        or result.get("service_needed")
        or result.get("recommended_section")
        or ""
    )

    date = (
        result.get("date")
        or result.get("appointment_date")
        or ""
    )

    time = (
        result.get("time")
        or result.get("appointment_time")
        or ""
    )

    updates = {
        "last_external_operation": operation,
        "last_external_result": result,
    }

    if branch:
        updates.update({
            "location_branch": branch,
            "nearest_branch": branch,
            "selected_branch": branch,
        })

    if section:
        updates.update({
            "service_needed": section,
            "recommended_section": section,
        })

    if date:
        updates.update({
            "appointment_date": date,
            "requested_date": date,
        })

    if time:
        updates.update({
            "appointment_time": time,
            "requested_time": time,
        })

    if result.get("available_slots") is not None:
        updates["available_slots"] = result.get("available_slots") or []
        updates["available_slots_text"] = result.get("available_slots_text") or ""
        updates["slots_found"] = bool(result.get("slots_found"))

    if result.get("exact_slot") is not None:
        updates["exact_slot"] = result.get("exact_slot") or {}

    if result.get("requested_slot") is not None:
        updates["requested_slot"] = result.get("requested_slot") or {}

    if result.get("nearest_slots") is not None:
        updates["nearest_slots"] = result.get("nearest_slots") or []
        updates["nearest_slots_text"] = result.get("nearest_slots_text") or ""

    if result.get("slot_status") is not None:
        updates["slot_status"] = result.get("slot_status") or ""

    reason = result.get("unavailable_reason") or result.get("reason") or ""
    if reason:
        updates["unavailable_reason"] = reason
        updates["reason"] = reason

    if result.get("booking") is not None:
        updates["booking"] = result.get("booking") or {}

    if result.get("booking_status") is not None:
        updates["booking_status"] = result.get("booking_status") or ""

    if result.get("visit_id") is not None:
        updates["visit_id"] = result.get("visit_id") or ""

    if operation:
        updates["active_goal"] = operation

    return compact_dict(updates)


def build_final_mode_instruction(request: ChatRequest) -> str:
    external_context = request.external_context or {}
    planner_result = request.planner_result or {}
    conversation_state = request.conversation_state or {}

    return (
        "MODE: final_with_external_context\n"
        "You are now writing the final customer-facing reply.\n"
        "You already received external data in external_context.result.\n"
        "Do not request any tool. Do not return action_required. Do not say you will check, search, fetch, or look it up.\n"
        "Use external_context.result as the source of truth for external facts.\n"
        "external_context.result is verified data already fetched by the workflow before this call.\n"
        "Treat facts from external_context.result as confirmed external/tool data.\n"
        "The quality check must not claim external_context.result facts are unverified.\n"
        "If external_context.result contains the requested answer, answer directly and naturally.\n"
        "If the external result says something is missing or not found, ask only for the missing customer detail.\n"
        "Do not expose internal fields, JSON, tools, operations, routes, payloads, or system logic.\n\n"
        f"Planner result:\n{json.dumps(planner_result, ensure_ascii=False)}\n\n"
        f"Conversation state:\n{json.dumps(conversation_state, ensure_ascii=False)}\n\n"
        f"External context:\n{json.dumps(external_context, ensure_ascii=False)}\n"
    )


def run_graph_once(
    request: ChatRequest,
    assistant_doc: Dict[str, Any],
    schema: Dict[str, Any],
    conversation: Dict[str, Any],
    variables: Dict[str, Any],
    override_tool_result: Optional[Dict[str, Any]] = None,
    extra_system_instruction: str = "",
) -> Dict[str, Any]:
    messages = recent_messages_from_conversation(conversation, limit=12)
    messages.append(HumanMessage(content=request.message))

    system_prompt = assistant_doc.get("system_prompt", "")
    if extra_system_instruction:
        system_prompt = f"{system_prompt}\n\n{extra_system_instruction}"

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
        "tool_result": override_tool_result if override_tool_result is not None else (request.tool_result or {}),
    }

    cb = None

    if get_openai_callback is not None:
        with get_openai_callback() as callback:
            final_state = app_graph.invoke(initial_state)
            cb = callback
    else:
        final_state = app_graph.invoke(initial_state)

    return {
        "final_state": final_state,
        "token_usage": get_token_usage_from_callback(cb),
    }


def build_response_payload(
    request: ChatRequest,
    final_state: Dict[str, Any],
    token_usage: Dict[str, Any],
    force_no_action_required: bool = False,
) -> Dict[str, Any]:
    route = build_route_contract(final_state)

    if force_no_action_required:
        action_required = None
    else:
        action_required = build_action_required_contract(final_state)

    answer = final_state.get("final_answer", "") or ""

    if not force_no_action_required and should_hide_answer_for_tool_request(route, request):
        answer = ""

    response_payload = {
        "answer": answer,
        "assistant_id": request.assistant_id,
        "conversation_id": request.conversation_id,
        "variables": final_state.get("variables", {}) or {},

        "route": route,
        "action_required": action_required,

        "quality": final_state.get("quality", {}) or {},
        "token_usage": token_usage,
        "mock_mode": False,
    }

    if force_no_action_required:
        response_payload["route"]["needs_tool"] = False
        response_payload["route"]["requested_tool_name"] = ""
        response_payload["route"]["tool_request_payload"] = {}
        response_payload["route"]["missing_tool_inputs"] = []
        response_payload["action_required"] = None

        if not response_payload["answer"]:
            revised = response_payload.get("quality", {}).get("revised_answer", "")
            if revised:
                response_payload["answer"] = revised

    if request.debug:
        response_payload["debug"] = {
            "manifest": final_state.get("manifest", {}) or {},
            "planner": final_state.get("planner", {}) or {},
            "selected_subagent": final_state.get("selected_subagent", {}) or {},
            "knowledge_items": final_state.get("knowledge_items", []) or [],
            "subagent_analysis": final_state.get("subagent_analysis", {}) or {},
        }

    return response_payload


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

    conversation_state = request.conversation_state or {}
    if not isinstance(conversation_state, dict):
        conversation_state = {}

    external_state_updates: Dict[str, Any] = {}
    if request.mode == "final_with_external_context":
        external_state_updates = infer_state_updates_from_external_context(request)

    variables = merge_dicts(
        stored_variables,
        conversation_state,
        incoming_variables,
        external_state_updates,
    )

    if request.mode == "final_with_external_context":
        external_tool_result = build_tool_result_from_external_context(request)

        graph_output = run_graph_once(
            request=request,
            assistant_doc=assistant_doc,
            schema=schema,
            conversation=conversation,
            variables=variables,
            override_tool_result=external_tool_result,
            extra_system_instruction=build_final_mode_instruction(request),
        )

        final_state = graph_output["final_state"]
        token_usage = graph_output["token_usage"]

        final_variables = merge_dicts(
            variables,
            final_state.get("variables", {}) or {},
            external_state_updates,
        )
        final_state["variables"] = final_variables

        response_payload = build_response_payload(
            request=request,
            final_state=final_state,
            token_usage=token_usage,
            force_no_action_required=True,
        )

        conversation["variables"] = final_variables
        conversation["summary"] = final_state.get("summary", conversation.get("summary", ""))

        # This final answer responds to the original user message.
        # Avoid duplicating the same user message in history because planner mode already stored it.
        append_conversation_messages(
            conversation,
            user_message=request.message,
            assistant_answer=response_payload.get("answer", ""),
            is_tool_result_turn=True,
        )

        save_conversation(request.assistant_id, request.conversation_id, conversation)

        return response_payload

    graph_output = run_graph_once(
        request=request,
        assistant_doc=assistant_doc,
        schema=schema,
        conversation=conversation,
        variables=variables,
    )

    final_state = graph_output["final_state"]
    token_usage = graph_output["token_usage"]

    response_payload = build_response_payload(
        request=request,
        final_state=final_state,
        token_usage=token_usage,
        force_no_action_required=False,
    )

    final_variables = response_payload.get("variables", {}) or {}

    conversation["variables"] = final_variables
    conversation["summary"] = final_state.get("summary", conversation.get("summary", ""))

    is_tool_result_turn = bool(request.tool_result) or request.message == "__tool_result__"

    append_conversation_messages(
        conversation,
        user_message=request.message,
        assistant_answer=response_payload.get("answer", ""),
        is_tool_result_turn=is_tool_result_turn,
    )

    save_conversation(request.assistant_id, request.conversation_id, conversation)

    if request.mode == "planner":
        return build_planner_response(request, response_payload)

    return response_payload
