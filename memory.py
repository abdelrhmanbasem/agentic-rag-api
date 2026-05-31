from app.config import MOCK_MODE, SUMMARY_TRIGGER_MESSAGE_COUNT
from app.llm import chat_json, memory_model
from app.db import (
    get_recent_messages,
    get_summary,
    save_summary,
    get_summary_message_count,
    count_messages,
    upsert_long_term_memory,
)
from app.rag import write_memory


def should_update_summary(conversation_id):
    total_messages = count_messages(conversation_id)
    last_summary_count = get_summary_message_count(conversation_id)
    if total_messages < SUMMARY_TRIGGER_MESSAGE_COUNT:
        return False
    return (total_messages - last_summary_count) >= SUMMARY_TRIGGER_MESSAGE_COUNT


def update_conversation_summary(conversation_id, assistant_id, user_id, variables):
    old_summary = get_summary(conversation_id)
    recent_messages = get_recent_messages(conversation_id, limit=14)

    if not should_update_summary(conversation_id):
        return old_summary

    if MOCK_MODE:
        new_summary = (old_summary + "\n" + str({"recent_messages": recent_messages[-4:], "variables": variables}))[:2500]
        save_summary(conversation_id, assistant_id, user_id, new_summary)
        return new_summary

    prompt = f"""
You are a conversation memory summarizer.

Update the rolling summary using the old summary, recent messages, and current variables.

Old summary:
{old_summary}

Recent messages:
{recent_messages}

Current variables:
{variables}

Rules:
- Keep only useful context for future replies.
- Include user goals, preferences, constraints, decisions, unresolved needs, and changed information.
- If the user changed their mind, keep only the latest value.
- Do not include small talk.
- Support English, Arabic, and Egyptian Arabic.
- Keep it concise.

Return JSON only:
{{"summary": "updated summary"}}
"""
    result = chat_json(memory_model(), [{"role": "user", "content": prompt}], max_tokens=500)
    new_summary = result.get("summary", old_summary)
    save_summary(conversation_id, assistant_id, user_id, new_summary)
    return new_summary


def decide_and_write_long_term_memories(assistant_id, user_id, conversation_id, summary, recent_messages, variables, agent_config):
    if MOCK_MODE:
        return []

    prompt = f"""
You are a long-term memory manager for a configurable multi-assistant system.

Assistant memory policy/config:
{agent_config}

Conversation summary:
{summary}

Recent messages:
{recent_messages}

Current variables:
{variables}

Decide what durable memories should be saved for this user.

Rules:
- Remember only stable preferences, goals, constraints, recurring needs, or important unresolved issues.
- Do not save small talk.
- Do not save sensitive details unless the assistant config explicitly requires it for the scenario.
- If the user corrected themselves, save only the latest fact.
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
    result = chat_json(memory_model(), [{"role": "user", "content": prompt}], max_tokens=600)
    memories = result.get("memories", [])

    written = []
    for memory in memories:
        text = (memory.get("text") or "").strip()
        memory_type = memory.get("type", "other")
        importance = float(memory.get("importance", 0.5))
        confidence = float(memory.get("confidence", 0.5))

        if not text or importance < 0.5 or confidence < 0.65:
            continue

        upsert_long_term_memory(assistant_id, user_id, text, memory_type, importance, confidence)
        write_memory(assistant_id, user_id, conversation_id, text, memory_type, importance, confidence)
        written.append(memory)

    return written
