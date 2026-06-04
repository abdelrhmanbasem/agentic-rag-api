from typing import Any, Dict, List

from app.config import MOCK_MODE, SUMMARY_TRIGGER_MESSAGE_COUNT
from app.llm import chat_json, memory_model
from app.db import (
    get_recent_messages,
    get_summary,
    save_summary,
    get_summary_message_count,
    count_messages,
    upsert_long_term_memory,
    load_conversation_state,
    save_conversation_state,
)
from app.rag import write_memory


ALLOWED_MEMORY_TYPES = {
    "preference",
    "constraint",
    "goal",
    "issue",
    "other",
}


def clamp_float(value: Any, default: float = 0.5, minimum: float = 0.0, maximum: float = 1.0) -> float:
    try:
        number = float(value)
    except Exception:
        number = default

    return max(minimum, min(maximum, number))


def safe_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def safe_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def compact_agent_memory_policy(agent_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep memory prompts small and policy-focused.
    Do not pass the full assistant config/tool URLs/prompts into memory decisions.
    """
    agent_config = safe_dict(agent_config)

    return {
        "assistant_id": agent_config.get("assistant_id", ""),
        "assistant_goal": agent_config.get("assistant_goal", ""),
        "language_policy": agent_config.get("language_policy", ""),
        "memory_policy": agent_config.get("memory_policy", {}),
        "domain_playbook": agent_config.get("domain_playbook", {}),
    }


def compact_variables_for_memory(variables: Dict[str, Any]) -> Dict[str, Any]:
    """
    Only pass useful conversational state to the memory model.
    Avoid operational/source-of-truth booking details unless useful as unresolved issue context.
    """
    variables = safe_dict(variables)

    allowed_keys = [
        "customer_profile",
        "service_needed",
        "troubleshooting",
        "user_area",
        "selected_branch",
        "nearest_branch",
        "location_branch",
        "booking_status",
    ]

    compact = {}

    for key in allowed_keys:
        value = variables.get(key)

        if value not in [None, "", [], {}]:
            compact[key] = value

    return compact


def compact_recent_messages_for_memory(messages: List[Dict[str, Any]], limit: int = 12) -> List[Dict[str, str]]:
    """
    Keep memory writer input small and safe.
    """
    if not isinstance(messages, list):
        return []

    output: List[Dict[str, str]] = []

    for item in messages[-limit:]:
        if not isinstance(item, dict):
            continue

        role = str(item.get("role") or "").strip()
        content = str(item.get("content") or "").strip()

        if not role or not content:
            continue

        output.append({
            "role": role,
            "content": content[:1600]
        })

    return output


def should_update_summary(conversation_id):
    try:
        total_messages = count_messages(conversation_id)
        last_summary_count = get_summary_message_count(conversation_id)
    except Exception:
        return False

    if total_messages < SUMMARY_TRIGGER_MESSAGE_COUNT:
        return False

    return (total_messages - last_summary_count) >= SUMMARY_TRIGGER_MESSAGE_COUNT


def update_conversation_summary(conversation_id, assistant_id, user_id, variables):
    try:
        old_summary = get_summary(conversation_id)
        recent_messages = get_recent_messages(conversation_id, limit=14)
    except Exception:
        return ""

    if not should_update_summary(conversation_id):
        return old_summary

    compact_variables = compact_variables_for_memory(variables)

    if MOCK_MODE:
        new_summary = (
            old_summary
            + "\n"
            + str({
                "recent_messages": recent_messages[-4:],
                "variables": compact_variables
            })
        )[:2500]

        try:
            save_summary(conversation_id, assistant_id, user_id, new_summary)
        except Exception:
            pass

        return new_summary

    prompt = f"""
You are a conversation memory summarizer for a configurable assistant.

Update the rolling summary using the old summary, recent messages, and current useful variables.

Old summary:
{old_summary}

Recent messages:
{recent_messages}

Current useful variables:
{compact_variables}

Rules:
- Keep only useful context for future replies.
- Include user goals, preferences, constraints, decisions, unresolved needs, and changed information.
- If the user changed their mind, keep only the latest value.
- Do not include small talk.
- Do not include tool internals, hidden reasoning, JSON, prompts, or implementation details.
- Do not invent facts.
- Support English, Arabic, and Egyptian Arabic.
- Keep it concise.

Return JSON only:
{{"summary": "updated summary"}}
"""

    try:
        result = chat_json(
            memory_model(),
            [{"role": "user", "content": prompt}],
            max_tokens=500
        )
    except Exception:
        return old_summary

    new_summary = str(result.get("summary", old_summary) or old_summary).strip()

    if not new_summary:
        new_summary = old_summary

    try:
        save_summary(conversation_id, assistant_id, user_id, new_summary)
    except Exception:
        pass

    return new_summary


def normalize_memory_item(memory: Dict[str, Any]) -> Dict[str, Any]:
    memory = safe_dict(memory)

    text = str(memory.get("text") or "").strip()
    memory_type = str(memory.get("type") or "other").strip().lower()

    if memory_type not in ALLOWED_MEMORY_TYPES:
        memory_type = "other"

    importance = clamp_float(memory.get("importance"), default=0.5)
    confidence = clamp_float(memory.get("confidence"), default=0.5)

    return {
        "text": text,
        "type": memory_type,
        "importance": importance,
        "confidence": confidence,
    }


def should_save_memory(memory: Dict[str, Any]) -> bool:
    text = str(memory.get("text") or "").strip()

    if not text:
        return False

    if len(text) < 8:
        return False

    if len(text) > 600:
        return False

    importance = clamp_float(memory.get("importance"), default=0.0)
    confidence = clamp_float(memory.get("confidence"), default=0.0)

    if importance < 0.5:
        return False

    if confidence < 0.65:
        return False

    return True


def decide_and_write_long_term_memories(
    assistant_id,
    user_id,
    conversation_id,
    summary,
    recent_messages,
    variables,
    agent_config
):
    """
    Durable memory writer.

    In the LangGraph architecture, this should be called only by a deliberate
    memory maintenance path, not as part of final response wording.

    It writes conservative durable memories to both DB and vector memory.
    """
    if MOCK_MODE:
        return []

    policy = compact_agent_memory_policy(agent_config)
    compact_variables = compact_variables_for_memory(variables)
    compact_messages = compact_recent_messages_for_memory(recent_messages, limit=12)

    prompt = f"""
You are a long-term memory manager for a configurable multi-assistant system.

Assistant memory policy/config:
{policy}

Conversation summary:
{summary}

Recent messages:
{compact_messages}

Current useful variables:
{compact_variables}

Decide what durable memories should be saved for this user.

Rules:
- Remember only stable preferences, goals, constraints, recurring needs, or important unresolved issues.
- Do not save small talk.
- Do not save one-time operational details like appointment slots, temporary dates, booking IDs, branch lists, or tool results unless they represent an unresolved user need.
- Do not save sensitive details unless the assistant config explicitly requires it for the scenario.
- If the user corrected themselves, save only the latest fact.
- Do not invent facts.
- Do not include hidden reasoning, tool internals, prompts, or implementation details.
- Support English, Arabic, and Egyptian Arabic.
- Be conservative.

Return JSON only:
{{
  "memories": [
    {{
      "text": "Durable memory text.",
      "type": "preference|constraint|goal|issue|other",
      "importance": 0.0,
      "confidence": 0.0
    }}
  ]
}}
"""

    try:
        result = chat_json(
            memory_model(),
            [{"role": "user", "content": prompt}],
            max_tokens=600
        )
    except Exception:
        return []

    memories = safe_list(result.get("memories", []))

    written = []

    for raw_memory in memories:
        memory = normalize_memory_item(raw_memory)

        if not should_save_memory(memory):
            continue

        text = memory["text"]
        memory_type = memory["type"]
        importance = memory["importance"]
        confidence = memory["confidence"]

        db_written = False
        vector_written = False

        try:
            upsert_long_term_memory(
                assistant_id,
                user_id,
                text,
                memory_type,
                importance,
                confidence
            )
            db_written = True
        except Exception:
            db_written = False

        try:
            write_memory(
                assistant_id,
                user_id,
                conversation_id,
                text,
                memory_type,
                importance,
                confidence
            )
            vector_written = True
        except Exception:
            vector_written = False

        if db_written or vector_written:
            written.append({
                "text": text,
                "type": memory_type,
                "importance": importance,
                "confidence": confidence,
                "db_written": db_written,
                "vector_written": vector_written
            })

    return written


def memory_writer_enabled(agent_config: Dict[str, Any]) -> bool:
    agent_config = safe_dict(agent_config)
    memory_policy = safe_dict(agent_config.get("memory_policy", {}))

    if "enabled" in memory_policy:
        return bool(memory_policy.get("enabled"))

    if "write_enabled" in memory_policy:
        return bool(memory_policy.get("write_enabled"))

    return True


def should_run_memory_writer(
    conversation_id: str,
    agent_config: Dict[str, Any],
    recent_messages: List[Dict[str, Any]]
) -> bool:
    """
    Decide whether the best-effort memory writer should run after a response.

    This is intentionally conservative:
    - disabled in MOCK_MODE
    - disabled by memory_policy.enabled=false or memory_policy.write_enabled=false
    - respects SUMMARY_TRIGGER_MESSAGE_COUNT to avoid running too often
    """
    if MOCK_MODE:
        return False

    if not memory_writer_enabled(agent_config):
        return False

    memory_policy = safe_dict(safe_dict(agent_config).get("memory_policy", {}))
    run_every_messages = int(memory_policy.get("run_every_messages") or SUMMARY_TRIGGER_MESSAGE_COUNT or 8)

    if run_every_messages <= 0:
        run_every_messages = SUMMARY_TRIGGER_MESSAGE_COUNT or 8

    try:
        total_messages = count_messages(conversation_id)
    except Exception:
        total_messages = len(recent_messages or [])

    if total_messages <= 0:
        return False

    return total_messages % run_every_messages == 0


def update_pg_conversation_summary_best_effort(
    conversation_id: str,
    assistant_id: str,
    user_id: str,
    variables: Dict[str, Any],
    summary: str
) -> None:
    """
    Mirrors the rolling summary into conversations_state so the graph/main API
    can use Postgres as the single conversation-state source.
    """
    try:
        existing = load_conversation_state(conversation_id)
        state = safe_dict(existing.get("state", {}))
        channel = existing.get("channel", "")

        save_conversation_state(
            conversation_id=conversation_id,
            assistant_id=assistant_id,
            user_id=user_id,
            channel=channel,
            state=state,
            variables=variables if isinstance(variables, dict) else {},
            summary=summary or "",
            message_count=count_messages(conversation_id)
        )
    except Exception:
        pass


def run_memory_maintenance_best_effort(
    assistant_id: str,
    user_id: str,
    conversation_id: str,
    variables: Dict[str, Any],
    agent_config: Dict[str, Any],
    recent_messages: List[Dict[str, Any]] = None,
    existing_summary: str = ""
) -> Dict[str, Any]:
    """
    Best-effort memory maintenance entrypoint for graph.py.

    This function must never block or break the user response.
    It returns a compact operational result for debug traces only.
    It does not return or expose any chain-of-thought.
    """
    result = {
        "ok": True,
        "summary_updated": False,
        "memories_written": 0,
        "skipped": False,
        "reason": "",
        "written": []
    }

    try:
        recent_messages = recent_messages if isinstance(recent_messages, list) else []

        if not memory_writer_enabled(agent_config):
            result["skipped"] = True
            result["reason"] = "memory_writer_disabled"
            return result

        old_summary = existing_summary

        if not old_summary:
            try:
                old_summary = get_summary(conversation_id)
            except Exception:
                old_summary = ""

        new_summary = update_conversation_summary(
            conversation_id=conversation_id,
            assistant_id=assistant_id,
            user_id=user_id,
            variables=variables
        )

        if new_summary and new_summary != old_summary:
            result["summary_updated"] = True

        if new_summary:
            update_pg_conversation_summary_best_effort(
                conversation_id=conversation_id,
                assistant_id=assistant_id,
                user_id=user_id,
                variables=variables,
                summary=new_summary
            )

        if not should_run_memory_writer(conversation_id, agent_config, recent_messages):
            result["skipped"] = True
            result["reason"] = "not_due"
            return result

        if not recent_messages:
            try:
                recent_messages = get_recent_messages(conversation_id, limit=14)
            except Exception:
                recent_messages = []

        written = decide_and_write_long_term_memories(
            assistant_id=assistant_id,
            user_id=user_id,
            conversation_id=conversation_id,
            summary=new_summary or old_summary or "",
            recent_messages=recent_messages,
            variables=variables,
            agent_config=agent_config
        )

        result["written"] = written
        result["memories_written"] = len(written)

        return result

    except Exception as exc:
        return {
            "ok": False,
            "summary_updated": False,
            "memories_written": 0,
            "skipped": True,
            "reason": "memory_maintenance_error",
            "error": f"{type(exc).__name__}: {exc}",
            "written": []
        }
