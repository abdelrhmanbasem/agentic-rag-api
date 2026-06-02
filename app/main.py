import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from app.agentic_brain import AgenticBrain


APP_SECRET = os.getenv("APP_SECRET", os.getenv("API_KEY", ""))
DATA_DIR = Path(os.getenv("DATA_DIR", "/app/data"))

ASSISTANTS_DIR = DATA_DIR / "assistants"
SCHEMAS_DIR = DATA_DIR / "schemas"
CONVERSATIONS_DIR = DATA_DIR / "conversations"

ASSISTANTS_DIR.mkdir(parents=True, exist_ok=True)
SCHEMAS_DIR.mkdir(parents=True, exist_ok=True)
CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Modular Agentic Brain API")
brain = AgenticBrain()


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
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def assistant_path(assistant_id: str) -> Path:
    return ASSISTANTS_DIR / f"{assistant_id}.json"


def schema_path(assistant_id: str) -> Path:
    return SCHEMAS_DIR / f"{assistant_id}.json"


def safe_conversation_id(conversation_id: str) -> str:
    return conversation_id.replace("/", "_")


def conversation_path(assistant_id: str, conversation_id: str) -> Path:
    return CONVERSATIONS_DIR / assistant_id / f"{safe_conversation_id(conversation_id)}.json"


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
            "messages": [],
            "traces": []
        }
    )


def save_conversation(assistant_id: str, conversation_id: str, data: Dict[str, Any]) -> None:
    save_path = conversation_path(assistant_id, conversation_id)
    safe_json_write(save_path, data)


def append_messages(conversation: Dict[str, Any], user_message: str, assistant_answer: str) -> Dict[str, Any]:
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

    conversation["messages"] = messages[-40:]
    return conversation


def append_trace(conversation: Dict[str, Any], trace: Dict[str, Any]) -> Dict[str, Any]:
    traces = conversation.get("traces", [])

    if not isinstance(traces, list):
        traces = []

    traces.append(trace)
    conversation["traces"] = traces[-30:]
    return conversation


@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "modular-agentic-brain-api"
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
        "assistant_id": assistant_id
    }


@app.get("/assistants/{assistant_id}")
def get_assistant_endpoint(assistant_id: str, x_api_key: Optional[str] = Header(default=None)):
    require_api_key(x_api_key)
    return load_assistant(assistant_id)


@app.post("/schemas")
def save_schema_endpoint(payload: Dict[str, Any], x_api_key: Optional[str] = Header(default=None)):
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
def get_schema_endpoint(assistant_id: str, x_api_key: Optional[str] = Header(default=None)):
    require_api_key(x_api_key)
    return load_schema(assistant_id)


@app.post("/chat")
def chat(request: ChatRequest, x_api_key: Optional[str] = Header(default=None)):
    require_api_key(x_api_key)

    assistant_config = load_assistant(request.assistant_id)
    schema = load_schema(request.assistant_id)
    conversation = load_conversation(request.assistant_id, request.conversation_id)

    result = brain.run(
        assistant_config=assistant_config,
        schema=schema,
        conversation=conversation,
        user_message=request.message,
        incoming_variables=request.variables,
        max_tool_calls=int(assistant_config.get("max_tool_calls", 4))
    )

    answer = result.get("answer", "")
    variables = result.get("variables", {})
    trace = result.get("trace", {})

    conversation["variables"] = variables
    append_messages(conversation, request.message, answer)
    append_trace(conversation, trace)

    save_conversation(request.assistant_id, request.conversation_id, conversation)

    response = {
        "answer": answer,
        "assistant_id": request.assistant_id,
        "conversation_id": request.conversation_id,
        "variables": variables,
        "selected_subagent": trace.get("selected_subagent"),
        "action": result.get("action", "reply"),
        "tool_calls_used": result.get("tool_calls_used", 0)
    }

    if request.debug:
        response["debug"] = trace

    return response


@app.post("/conversations/{assistant_id}/{conversation_id}/clear")
def clear_conversation_endpoint(
    assistant_id: str,
    conversation_id: str,
    x_api_key: Optional[str] = Header(default=None)
):
    require_api_key(x_api_key)

    path = conversation_path(assistant_id, conversation_id)

    if path.exists():
        path.unlink()

    return {
        "ok": True,
        "cleared": True,
        "assistant_id": assistant_id,
        "conversation_id": conversation_id
    }
