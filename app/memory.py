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


def mock_update_summary(old_summary, recent_messages, variables):
    parts = []

    if old_summary:
        parts.append(old_summary)

    if variables:
        parts.append(f"Current known variables: {variables}")

    latest_user_messages = [
        m["content"] for m in recent_messages
        if m.get("role") == "user"
    ][-3:]

    if latest_user_messages:
        parts.append("Recent user messages: " + " | ".join(latest_user_messages))

    summary = "\n".join(parts)
    return summary[:2500]


def update_conversation_summary(conversation_id, assistant_id, user_id, variables):
    old_summary = get_summary(conversation_id)
    recent_messages = get_recent_messages(conversation_id, limit=12)

    if not should_update_summary(conversation_id):
        return old_summary

    if MOCK_MODE:
        new_summary = mock_update_summary(old_summary, recent_messages, variables)
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
- If the user changed their mind, keep the latest value.
- Do not include small talk.
- Support English, Arabic, and Egyptian Arabic.
- Keep it concise.

Return JSON only:
{{
  "summary": "updated summary"
}}
"""

    result = chat_json(
        memory_model(),
        [{"role": "user", "content": prompt}],
        max_tokens=500,
    )

    new_summary = result.get("summary", old_summary)
    save_summary(conversation_id, assistant_id, user_id, new_summary)
    return new_summary


def mock_memory_decision(variables):
    memories = []

    preferred_contact = variables.get("preferred_contact_method") or variables.get("preferred_contact")
    if preferred_contact:
        memories.append({
            "text": f"User prefers {preferred_contact} as contact method.",
            "type": "preference",
            "importance": 0.8,
            "confidence": 0.8,
        })

    car_brand = variables.get("car_brand")
    if car_brand:
        memories.append({
            "text": f"User is interested in {car_brand} cars.",
            "type": "preference",
            "importance": 0.7,
            "confidence": 0.75,
        })

    car_condition = variables.get("car_condition")
    if car_condition:
        memories.append({
            "text": f"User is interested in {car_condition} cars.",
            "type": "preference",
            "importance": 0.6,
            "confidence": 0.7,
        })

    transmission = variables.get("transmission")
    if transmission:
        memories.append({
            "text": f"User prefers {transmission} transmission.",
            "type": "preference",
            "importance": 0.6,
            "confidence": 0.7,
        })

    budget = variables.get("budget_max")
    currency = variables.get("currency", "")
    if budget:
        memories.append({
            "text": f"User budget is around {budget} {currency}.",
            "type": "constraint",
            "importance": 0.7,
            "confidence": 0.75,
        })

    service_needed = variables.get("service_needed")
    if service_needed:
        memories.append({
            "text": f"User is interested in service: {service_needed}.",
            "type": "preference",
            "importance": 0.7,
            "confidence": 0.75,
        })

    doctor_preference = variables.get("doctor_preference")
    if doctor_preference:
        memories.append({
            "text": f"User prefers doctor: {doctor_preference}.",
            "type": "preference",
            "importance": 0.8,
            "confidence": 0.8,
        })

    location_branch = variables.get("location_branch")
    if location_branch:
        memories.append({
            "text": f"User prefers branch/location: {location_branch}.",
            "type": "preference",
            "importance": 0.7,
            "confidence": 0.75,
        })

    appointment_time = variables.get("appointment_time")
    if appointment_time:
        memories.append({
            "text": f"User prefers appointments in the {appointment_time}.",
            "type": "preference",
            "importance": 0.55,
            "confidence": 0.65,
        })

    insurance_provider = variables.get("insurance_provider")
    if insurance_provider:
        memories.append({
            "text": f"User mentioned insurance preference/provider: {insurance_provider}.",
            "type": "preference",
            "importance": 0.55,
            "confidence": 0.65,
        })

    return memories


def decide_and_write_long_term_memories(
    assistant_id,
    user_id,
    conversation_id,
    summary,
    recent_messages,
    variables,
):
    if MOCK_MODE:
        memories = mock_memory_decision(variables)
    else:
        prompt = f"""
You are a long-term memory manager.

Decide what should be remembered about the user for future conversations.

Conversation summary:
{summary}

Recent messages:
{recent_messages}

Current variables:
{variables}

Remember only durable information:
- stable preferences
- preferred contact method
- recurring interests
- budget range
- preferred brand/service/doctor/branch
- preferred appointment time
- important constraints
- unresolved long-term needs

Do not remember:
- small talk
- one-time temporary details
- sensitive medical details unless necessary for booking workflow
- low-confidence assumptions

If the user changed their mind, remember only the latest preference.

Support English, Arabic, and Egyptian Arabic.

Return JSON only:
{{
  "memories": [
    {{
      "text": "User prefers WhatsApp instead of calls.",
      "type": "preference|constraint|goal|issue|other",
      "importance": 0.0,
      "confidence": 0.0
    }}
  ]
}}
"""
        result = chat_json(
            memory_model(),
            [{"role": "user", "content": prompt}],
            max_tokens=600,
        )
        memories = result.get("memories", [])

    written = []

    for memory in memories:
        text = memory.get("text", "").strip()
        memory_type = memory.get("type", "other")
        importance = float(memory.get("importance", 0.5))
        confidence = float(memory.get("confidence", 0.5))

        if not text or importance < 0.5 or confidence < 0.65:
            continue

        upsert_long_term_memory(
            assistant_id=assistant_id,
            user_id=user_id,
            memory_text=text,
            memory_type=memory_type,
            importance=importance,
            confidence=confidence,
        )

        write_memory(
            assistant_id=assistant_id,
            user_id=user_id,
            conversation_id=conversation_id,
            text=text,
            memory_type=memory_type,
            importance=importance,
            confidence=confidence,
        )

        written.append(memory)

    return written
