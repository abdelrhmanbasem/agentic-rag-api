import contextlib
import hashlib
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import psycopg

from app.config import (
    POSTGRES_HOST,
    POSTGRES_PORT,
    POSTGRES_DB,
    POSTGRES_USER,
    POSTGRES_PASSWORD,
    ESTIMATE_CHARS_PER_TOKEN,
)


# Process-local hint used for backward-compatible functions that currently
# receive only conversation_id. main.py calls ensure_conversation before the
# state helpers, so this lets db.py resolve assistant-scoped storage keys
# without changing old call sites.
_ASSISTANT_HINT_BY_CONVERSATION: Dict[str, str] = {}


def get_conn(autocommit: bool = False):
    return psycopg.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        autocommit=autocommit,
    )


def json_dumps(value: Any) -> str:
    return json.dumps(value if value is not None else {}, ensure_ascii=False)


def normalize_json(value: Any, default: Any = None) -> Any:
    if default is None:
        default = {}

    if value is None:
        return default

    if isinstance(value, (dict, list)):
        return value

    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return default

    return default


def safe_text(value: Any) -> str:
    return str(value or "").strip()


def safe_storage_key_part(value: Any, fallback: str = "default") -> str:
    text = safe_text(value)

    if not text:
        text = fallback

    text = re.sub(r"\s+", " ", text)
    return text


def make_conversation_key(assistant_id: str, conversation_id: str) -> str:
    assistant = safe_storage_key_part(assistant_id)
    conversation = safe_storage_key_part(conversation_id)

    raw = f"{assistant}::{conversation}"

    # Keep keys readable under normal conditions, but bounded.
    if len(raw) <= 512:
        return raw

    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return f"{assistant[:120]}::hash::{digest}"


def remember_assistant_hint(conversation_id: str, assistant_id: str) -> None:
    conversation = safe_text(conversation_id)
    assistant = safe_text(assistant_id)

    if conversation and assistant:
        _ASSISTANT_HINT_BY_CONVERSATION[conversation] = assistant


def get_assistant_hint(conversation_id: str) -> str:
    return _ASSISTANT_HINT_BY_CONVERSATION.get(safe_text(conversation_id), "")


def candidate_conversation_keys(
    conversation_id: str,
    assistant_id: Optional[str] = None,
    include_legacy: bool = True
) -> List[str]:
    conversation = safe_text(conversation_id)
    assistant = safe_text(assistant_id) or get_assistant_hint(conversation)

    candidates: List[str] = []

    if assistant and conversation:
        candidates.append(make_conversation_key(assistant, conversation))

    if include_legacy and conversation:
        candidates.append(conversation)

    output: List[str] = []
    for item in candidates:
        if item and item not in output:
            output.append(item)

    return output


def get_primary_conversation_key(assistant_id: str, conversation_id: str) -> str:
    return make_conversation_key(assistant_id, conversation_id)


def advisory_lock_id(assistant_id: str, conversation_id: str) -> int:
    raw = make_conversation_key(assistant_id, conversation_id)
    digest = hashlib.sha256(raw.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False)
    # Postgres advisory lock accepts signed BIGINT.
    if value >= 2**63:
        value -= 2**64
    return value


@contextlib.contextmanager
def conversation_transaction(
    assistant_id: str,
    conversation_id: str,
    lock: bool = True
):
    with get_conn(autocommit=False) as conn:
        with conn.cursor() as cur:
            if lock:
                cur.execute(
                    "SELECT pg_advisory_xact_lock(%s)",
                    (advisory_lock_id(assistant_id, conversation_id),)
                )
            yield conn, cur


def init_db():
    with get_conn(autocommit=False) as conn:
        with conn.cursor() as cur:
            cur.execute("""
            CREATE TABLE IF NOT EXISTS assistants (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                system_prompt TEXT NOT NULL,
                tone TEXT DEFAULT '',
                memory_policy TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            );
            """)

            cur.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                assistant_id TEXT NOT NULL,
                conversation_id TEXT,
                user_id TEXT NOT NULL,
                channel TEXT DEFAULT '',
                version BIGINT DEFAULT 0,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            );
            """)

            cur.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id BIGSERIAL PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                original_conversation_id TEXT,
                assistant_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                request_key TEXT DEFAULT '',
                message_id TEXT DEFAULT '',
                trace_id TEXT DEFAULT '',
                metadata JSONB NOT NULL DEFAULT '{}',
                created_at TIMESTAMP DEFAULT NOW()
            );
            """)

            cur.execute("""
            CREATE TABLE IF NOT EXISTS conversation_summaries (
                conversation_id TEXT PRIMARY KEY,
                original_conversation_id TEXT,
                assistant_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                summary TEXT NOT NULL DEFAULT '',
                message_count INT DEFAULT 0,
                updated_at TIMESTAMP DEFAULT NOW()
            );
            """)

            cur.execute("""
            CREATE TABLE IF NOT EXISTS assistant_variable_schemas (
                assistant_id TEXT PRIMARY KEY,
                schema JSONB NOT NULL,
                updated_at TIMESTAMP DEFAULT NOW()
            );
            """)

            cur.execute("""
            CREATE TABLE IF NOT EXISTS conversation_variables (
                conversation_id TEXT PRIMARY KEY,
                original_conversation_id TEXT,
                assistant_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                variables JSONB NOT NULL DEFAULT '{}',
                version BIGINT DEFAULT 0,
                updated_at TIMESTAMP DEFAULT NOW()
            );
            """)

            cur.execute("""
            CREATE TABLE IF NOT EXISTS conversation_states (
                conversation_id TEXT PRIMARY KEY,
                original_conversation_id TEXT NOT NULL,
                assistant_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                channel TEXT DEFAULT '',
                state JSONB NOT NULL DEFAULT '{}',
                variables JSONB NOT NULL DEFAULT '{}',
                summary TEXT NOT NULL DEFAULT '',
                message_count INT DEFAULT 0,
                version BIGINT DEFAULT 0,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            );
            """)

            cur.execute("""
            CREATE TABLE IF NOT EXISTS conversation_idempotency (
                assistant_id TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                request_key TEXT NOT NULL,
                trace_id TEXT DEFAULT '',
                final_answer TEXT DEFAULT '',
                response_payload JSONB NOT NULL DEFAULT '{}',
                trace JSONB NOT NULL DEFAULT '{}',
                state_version BIGINT DEFAULT 0,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW(),
                PRIMARY KEY (assistant_id, conversation_id, request_key)
            );
            """)

            cur.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_documents (
                assistant_id TEXT NOT NULL,
                document_id TEXT NOT NULL,
                title TEXT NOT NULL,
                metadata JSONB NOT NULL DEFAULT '{}',
                chunk_count INT DEFAULT 0,
                updated_at TIMESTAMP DEFAULT NOW(),
                PRIMARY KEY (assistant_id, document_id)
            );
            """)

            cur.execute("""
            CREATE TABLE IF NOT EXISTS long_term_memories (
                id BIGSERIAL PRIMARY KEY,
                assistant_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                memory_key TEXT NOT NULL,
                memory_text TEXT NOT NULL,
                memory_type TEXT DEFAULT 'other',
                importance NUMERIC DEFAULT 0.5,
                confidence NUMERIC DEFAULT 0.5,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW(),
                UNIQUE (assistant_id, user_id, memory_key)
            );
            """)

            cur.execute("""
            CREATE TABLE IF NOT EXISTS model_usage (
                id BIGSERIAL PRIMARY KEY,
                assistant_id TEXT,
                conversation_id TEXT,
                user_id TEXT,
                model TEXT,
                purpose TEXT,
                input_tokens INT DEFAULT 0,
                output_tokens INT DEFAULT 0,
                estimated_cost_usd NUMERIC DEFAULT 0,
                metadata JSONB NOT NULL DEFAULT '{}',
                created_at TIMESTAMP DEFAULT NOW()
            );
            """)

            cur.execute("""
            CREATE TABLE IF NOT EXISTS conversation_rag_cache (
                conversation_id TEXT PRIMARY KEY,
                original_conversation_id TEXT,
                assistant_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                query TEXT NOT NULL DEFAULT '',
                knowledge_payload JSONB NOT NULL DEFAULT '[]',
                compressed_payload JSONB NOT NULL DEFAULT '[]',
                updated_at TIMESTAMP DEFAULT NOW()
            );
            """)

            # Backward-compatible migrations for existing deployments.
            cur.execute("ALTER TABLE assistants ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP DEFAULT NOW();")
            cur.execute("ALTER TABLE conversations ADD COLUMN IF NOT EXISTS conversation_id TEXT;")
            cur.execute("ALTER TABLE conversations ADD COLUMN IF NOT EXISTS version BIGINT DEFAULT 0;")
            cur.execute("ALTER TABLE messages ADD COLUMN IF NOT EXISTS original_conversation_id TEXT;")
            cur.execute("ALTER TABLE messages ADD COLUMN IF NOT EXISTS request_key TEXT DEFAULT '';")
            cur.execute("ALTER TABLE messages ADD COLUMN IF NOT EXISTS message_id TEXT DEFAULT '';")
            cur.execute("ALTER TABLE messages ADD COLUMN IF NOT EXISTS trace_id TEXT DEFAULT '';")
            cur.execute("ALTER TABLE messages ADD COLUMN IF NOT EXISTS metadata JSONB NOT NULL DEFAULT '{}';")
            cur.execute("ALTER TABLE conversation_summaries ADD COLUMN IF NOT EXISTS original_conversation_id TEXT;")
            cur.execute("ALTER TABLE conversation_summaries ADD COLUMN IF NOT EXISTS message_count INT DEFAULT 0;")
            cur.execute("ALTER TABLE conversation_variables ADD COLUMN IF NOT EXISTS original_conversation_id TEXT;")
            cur.execute("ALTER TABLE conversation_variables ADD COLUMN IF NOT EXISTS version BIGINT DEFAULT 0;")
            cur.execute("ALTER TABLE model_usage ADD COLUMN IF NOT EXISTS metadata JSONB NOT NULL DEFAULT '{}';")
            cur.execute("ALTER TABLE conversation_rag_cache ADD COLUMN IF NOT EXISTS original_conversation_id TEXT;")

            cur.execute("""
            UPDATE conversations
            SET conversation_id = id
            WHERE conversation_id IS NULL OR conversation_id = '';
            """)
            cur.execute("""
            UPDATE messages
            SET original_conversation_id = conversation_id
            WHERE original_conversation_id IS NULL OR original_conversation_id = '';
            """)
            cur.execute("""
            UPDATE conversation_summaries
            SET original_conversation_id = conversation_id
            WHERE original_conversation_id IS NULL OR original_conversation_id = '';
            """)
            cur.execute("""
            UPDATE conversation_variables
            SET original_conversation_id = conversation_id
            WHERE original_conversation_id IS NULL OR original_conversation_id = '';
            """)
            cur.execute("""
            UPDATE conversation_rag_cache
            SET original_conversation_id = conversation_id
            WHERE original_conversation_id IS NULL OR original_conversation_id = '';
            """)

            cur.execute("CREATE INDEX IF NOT EXISTS idx_conversations_assistant_original ON conversations (assistant_id, conversation_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages (conversation_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_messages_assistant_conversation_created ON messages (assistant_id, conversation_id, created_at DESC);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_messages_original ON messages (assistant_id, original_conversation_id, created_at DESC);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_conversation_states_assistant_original ON conversation_states (assistant_id, original_conversation_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_conversation_variables_assistant_original ON conversation_variables (assistant_id, original_conversation_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_long_term_memories_user ON long_term_memories (assistant_id, user_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_rag_cache_conversation ON conversation_rag_cache (conversation_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_rag_cache_assistant_original ON conversation_rag_cache (assistant_id, original_conversation_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_idempotency_conversation ON conversation_idempotency (assistant_id, conversation_id);")


def estimate_tokens_from_text(text):
    if not text:
        return 0
    return max(1, int(len(str(text)) / ESTIMATE_CHARS_PER_TOKEN))


def estimate_tokens_from_obj(obj):
    try:
        return estimate_tokens_from_text(json.dumps(obj, ensure_ascii=False))
    except Exception:
        return estimate_tokens_from_text(str(obj))


def ensure_conversation(conversation_id, assistant_id, user_id, channel):
    remember_assistant_hint(conversation_id, assistant_id)
    key = get_primary_conversation_key(assistant_id, conversation_id)

    with conversation_transaction(assistant_id, conversation_id, lock=True) as (_conn, cur):
        cur.execute("""
            INSERT INTO conversations (id, assistant_id, conversation_id, user_id, channel, updated_at)
            VALUES (%s, %s, %s, %s, %s, NOW())
            ON CONFLICT (id)
            DO UPDATE SET
                assistant_id = EXCLUDED.assistant_id,
                conversation_id = EXCLUDED.conversation_id,
                user_id = EXCLUDED.user_id,
                channel = EXCLUDED.channel,
                updated_at = NOW(),
                version = conversations.version + 1
        """, (key, assistant_id, conversation_id, user_id, channel))


def resolve_existing_key(cur, conversation_id: str, assistant_id: Optional[str] = None) -> str:
    candidates = candidate_conversation_keys(conversation_id, assistant_id)

    for key in candidates:
        cur.execute("SELECT id FROM conversations WHERE id = %s", (key,))
        row = cur.fetchone()
        if row:
            return row[0]

    if assistant_id:
        cur.execute("""
            SELECT id
            FROM conversations
            WHERE assistant_id = %s AND conversation_id = %s
            ORDER BY updated_at DESC
            LIMIT 1
        """, (assistant_id, conversation_id))
        row = cur.fetchone()
        if row:
            return row[0]

    cur.execute("""
        SELECT id
        FROM conversations
        WHERE conversation_id = %s OR id = %s
        ORDER BY updated_at DESC
        LIMIT 1
    """, (conversation_id, conversation_id))
    row = cur.fetchone()

    return row[0] if row else (candidates[0] if candidates else conversation_id)


def save_message(
    conversation_id,
    assistant_id,
    user_id,
    role,
    content,
    request_key: str = "",
    message_id: str = "",
    trace_id: str = "",
    metadata: Optional[Dict[str, Any]] = None,
):
    remember_assistant_hint(conversation_id, assistant_id)
    key = get_primary_conversation_key(assistant_id, conversation_id)

    with conversation_transaction(assistant_id, conversation_id, lock=False) as (_conn, cur):
        cur.execute("""
            INSERT INTO messages
            (conversation_id, original_conversation_id, assistant_id, user_id, role, content, request_key, message_id, trace_id, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            key,
            conversation_id,
            assistant_id,
            user_id,
            role,
            content,
            request_key or "",
            message_id or "",
            trace_id or "",
            json_dumps(metadata or {}),
        ))


def count_messages(conversation_id, assistant_id: Optional[str] = None):
    with get_conn() as conn:
        with conn.cursor() as cur:
            key = resolve_existing_key(cur, conversation_id, assistant_id)
            cur.execute("""
                SELECT COUNT(*)
                FROM messages
                WHERE conversation_id = %s
            """, (key,))
            return cur.fetchone()[0]


def get_recent_messages(conversation_id, limit=10, assistant_id: Optional[str] = None):
    with get_conn() as conn:
        with conn.cursor() as cur:
            key = resolve_existing_key(cur, conversation_id, assistant_id)
            cur.execute("""
                SELECT role, content, request_key, message_id, trace_id, created_at
                FROM messages
                WHERE conversation_id = %s
                ORDER BY id DESC
                LIMIT %s
            """, (key, limit))
            rows = cur.fetchall()

    rows.reverse()
    return [
        {
            "role": r[0],
            "content": r[1],
            "request_key": r[2] or "",
            "message_id": r[3] or "",
            "trace_id": r[4] or "",
            "created_at": r[5].isoformat() if r[5] else None,
        }
        for r in rows
    ]


def get_or_create_assistant(assistant_id):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, name, system_prompt, tone, memory_policy
                FROM assistants
                WHERE id = %s
            """, (assistant_id,))
            row = cur.fetchone()

            if row:
                return {
                    "id": row[0],
                    "name": row[1],
                    "system_prompt": row[2],
                    "tone": row[3],
                    "memory_policy": row[4],
                }

            default_prompt = "You are a helpful AI assistant. Answer clearly and helpfully."
            default_tone = "clear, helpful, concise"
            default_memory_policy = "Remember stable user preferences, goals, constraints, and important facts."

            cur.execute("""
                INSERT INTO assistants (id, name, system_prompt, tone, memory_policy)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                assistant_id,
                assistant_id,
                default_prompt,
                default_tone,
                default_memory_policy,
            ))

            return {
                "id": assistant_id,
                "name": assistant_id,
                "system_prompt": default_prompt,
                "tone": default_tone,
                "memory_policy": default_memory_policy,
            }


def upsert_assistant(assistant_id, name, system_prompt, tone, memory_policy):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO assistants (id, name, system_prompt, tone, memory_policy, updated_at)
                VALUES (%s, %s, %s, %s, %s, NOW())
                ON CONFLICT (id)
                DO UPDATE SET
                    name = EXCLUDED.name,
                    system_prompt = EXCLUDED.system_prompt,
                    tone = EXCLUDED.tone,
                    memory_policy = EXCLUDED.memory_policy,
                    updated_at = NOW()
            """, (assistant_id, name, system_prompt, tone, memory_policy))


def save_schema(assistant_id, schema):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO assistant_variable_schemas (assistant_id, schema, updated_at)
                VALUES (%s, %s, NOW())
                ON CONFLICT (assistant_id)
                DO UPDATE SET schema = EXCLUDED.schema, updated_at = NOW()
            """, (assistant_id, json_dumps(schema or {})))


def get_schema(assistant_id):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT schema
                FROM assistant_variable_schemas
                WHERE assistant_id = %s
            """, (assistant_id,))
            row = cur.fetchone()
            return normalize_json(row[0], {}) if row else {}


def get_variables(conversation_id, assistant_id: Optional[str] = None):
    with get_conn() as conn:
        with conn.cursor() as cur:
            key = resolve_existing_key(cur, conversation_id, assistant_id)
            cur.execute("""
                SELECT variables
                FROM conversation_variables
                WHERE conversation_id = %s
            """, (key,))
            row = cur.fetchone()
            return normalize_json(row[0], {}) if row else {}


def save_variables(conversation_id, assistant_id, user_id, variables):
    remember_assistant_hint(conversation_id, assistant_id)
    key = get_primary_conversation_key(assistant_id, conversation_id)

    with conversation_transaction(assistant_id, conversation_id, lock=True) as (_conn, cur):
        cur.execute("""
            INSERT INTO conversation_variables
            (conversation_id, original_conversation_id, assistant_id, user_id, variables, version, updated_at)
            VALUES (%s, %s, %s, %s, %s, 1, NOW())
            ON CONFLICT (conversation_id)
            DO UPDATE SET
                assistant_id = EXCLUDED.assistant_id,
                original_conversation_id = EXCLUDED.original_conversation_id,
                user_id = EXCLUDED.user_id,
                variables = EXCLUDED.variables,
                version = conversation_variables.version + 1,
                updated_at = NOW()
        """, (key, conversation_id, assistant_id, user_id, json_dumps(variables or {})))


def get_summary(conversation_id, assistant_id: Optional[str] = None):
    with get_conn() as conn:
        with conn.cursor() as cur:
            key = resolve_existing_key(cur, conversation_id, assistant_id)
            cur.execute("""
                SELECT summary
                FROM conversation_summaries
                WHERE conversation_id = %s
            """, (key,))
            row = cur.fetchone()
            return row[0] if row else ""


def save_summary(conversation_id, assistant_id, user_id, summary):
    remember_assistant_hint(conversation_id, assistant_id)
    key = get_primary_conversation_key(assistant_id, conversation_id)
    message_count = count_messages(conversation_id, assistant_id=assistant_id)

    with conversation_transaction(assistant_id, conversation_id, lock=False) as (_conn, cur):
        cur.execute("""
            INSERT INTO conversation_summaries
            (conversation_id, original_conversation_id, assistant_id, user_id, summary, message_count, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, NOW())
            ON CONFLICT (conversation_id)
            DO UPDATE SET
                assistant_id = EXCLUDED.assistant_id,
                original_conversation_id = EXCLUDED.original_conversation_id,
                user_id = EXCLUDED.user_id,
                summary = EXCLUDED.summary,
                message_count = EXCLUDED.message_count,
                updated_at = NOW()
        """, (key, conversation_id, assistant_id, user_id, summary or "", message_count))


def get_summary_message_count(conversation_id, assistant_id: Optional[str] = None):
    with get_conn() as conn:
        with conn.cursor() as cur:
            key = resolve_existing_key(cur, conversation_id, assistant_id)
            cur.execute("""
                SELECT message_count
                FROM conversation_summaries
                WHERE conversation_id = %s
            """, (key,))
            row = cur.fetchone()
            return row[0] if row else 0


def load_conversation_state(conversation_id, assistant_id: Optional[str] = None):
    assistant_hint = assistant_id or get_assistant_hint(conversation_id)

    with get_conn() as conn:
        with conn.cursor() as cur:
            candidate_keys = candidate_conversation_keys(conversation_id, assistant_hint)

            # Prefer new state rows.
            for key in candidate_keys:
                cur.execute("""
                    SELECT conversation_id, original_conversation_id, assistant_id, user_id, channel,
                           state, variables, summary, message_count, version, updated_at
                    FROM conversation_states
                    WHERE conversation_id = %s
                    LIMIT 1
                """, (key,))
                row = cur.fetchone()

                if row:
                    remember_assistant_hint(row[1], row[2])
                    return {
                        "conversation_key": row[0],
                        "conversation_id": row[1],
                        "assistant_id": row[2],
                        "user_id": row[3],
                        "channel": row[4],
                        "state": normalize_json(row[5], {}),
                        "variables": normalize_json(row[6], {}),
                        "summary": row[7] or "",
                        "message_count": row[8] or 0,
                        "version": row[9] or 0,
                        "updated_at": row[10].isoformat() if row[10] else None,
                    }

            # If assistant hint exists, look by original id.
            if assistant_hint:
                cur.execute("""
                    SELECT conversation_id, original_conversation_id, assistant_id, user_id, channel,
                           state, variables, summary, message_count, version, updated_at
                    FROM conversation_states
                    WHERE assistant_id = %s AND original_conversation_id = %s
                    ORDER BY updated_at DESC
                    LIMIT 1
                """, (assistant_hint, conversation_id))
                row = cur.fetchone()
                if row:
                    return {
                        "conversation_key": row[0],
                        "conversation_id": row[1],
                        "assistant_id": row[2],
                        "user_id": row[3],
                        "channel": row[4],
                        "state": normalize_json(row[5], {}),
                        "variables": normalize_json(row[6], {}),
                        "summary": row[7] or "",
                        "message_count": row[8] or 0,
                        "version": row[9] or 0,
                        "updated_at": row[10].isoformat() if row[10] else None,
                    }

            # Legacy fallback from split tables.
            key = resolve_existing_key(cur, conversation_id, assistant_hint)
            cur.execute("""
                SELECT id, conversation_id, assistant_id, user_id, channel, version, updated_at
                FROM conversations
                WHERE id = %s
                LIMIT 1
            """, (key,))
            conv = cur.fetchone()

            if not conv:
                return None

            cur.execute("""
                SELECT variables, version, updated_at
                FROM conversation_variables
                WHERE conversation_id = %s
            """, (key,))
            variables_row = cur.fetchone()

            cur.execute("""
                SELECT summary, message_count, updated_at
                FROM conversation_summaries
                WHERE conversation_id = %s
            """, (key,))
            summary_row = cur.fetchone()

            return {
                "conversation_key": conv[0],
                "conversation_id": conv[1] or conversation_id,
                "assistant_id": conv[2],
                "user_id": conv[3],
                "channel": conv[4],
                "state": {
                    "messages": get_recent_messages(conversation_id, limit=80, assistant_id=conv[2]),
                    "traces": [],
                    "processed_requests": [],
                },
                "variables": normalize_json(variables_row[0], {}) if variables_row else {},
                "summary": summary_row[0] if summary_row else "",
                "message_count": summary_row[1] if summary_row else count_messages(conversation_id, assistant_id=conv[2]),
                "version": max(conv[5] or 0, variables_row[1] if variables_row else 0),
                "updated_at": conv[6].isoformat() if conv[6] else None,
            }


def save_conversation_state(
    conversation_id,
    assistant_id,
    user_id,
    channel,
    state,
    variables,
    summary="",
    message_count=0,
    expected_version: Optional[int] = None,
):
    remember_assistant_hint(conversation_id, assistant_id)
    key = get_primary_conversation_key(assistant_id, conversation_id)

    with conversation_transaction(assistant_id, conversation_id, lock=True) as (_conn, cur):
        if expected_version is not None:
            cur.execute("""
                SELECT version
                FROM conversation_states
                WHERE conversation_id = %s
            """, (key,))
            row = cur.fetchone()
            if row and int(row[0] or 0) != int(expected_version):
                raise RuntimeError(
                    f"Conversation state version mismatch: expected {expected_version}, found {row[0]}"
                )

        cur.execute("""
            INSERT INTO conversations (id, assistant_id, conversation_id, user_id, channel, version, updated_at)
            VALUES (%s, %s, %s, %s, %s, 1, NOW())
            ON CONFLICT (id)
            DO UPDATE SET
                assistant_id = EXCLUDED.assistant_id,
                conversation_id = EXCLUDED.conversation_id,
                user_id = EXCLUDED.user_id,
                channel = EXCLUDED.channel,
                version = conversations.version + 1,
                updated_at = NOW()
        """, (key, assistant_id, conversation_id, user_id, channel))

        cur.execute("""
            INSERT INTO conversation_states
            (conversation_id, original_conversation_id, assistant_id, user_id, channel, state, variables, summary, message_count, version, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 1, NOW())
            ON CONFLICT (conversation_id)
            DO UPDATE SET
                original_conversation_id = EXCLUDED.original_conversation_id,
                assistant_id = EXCLUDED.assistant_id,
                user_id = EXCLUDED.user_id,
                channel = EXCLUDED.channel,
                state = EXCLUDED.state,
                variables = EXCLUDED.variables,
                summary = EXCLUDED.summary,
                message_count = EXCLUDED.message_count,
                version = conversation_states.version + 1,
                updated_at = NOW()
        """, (
            key,
            conversation_id,
            assistant_id,
            user_id,
            channel,
            json_dumps(state or {}),
            json_dumps(variables or {}),
            summary or "",
            int(message_count or 0),
        ))

        cur.execute("""
            INSERT INTO conversation_variables
            (conversation_id, original_conversation_id, assistant_id, user_id, variables, version, updated_at)
            VALUES (%s, %s, %s, %s, %s, 1, NOW())
            ON CONFLICT (conversation_id)
            DO UPDATE SET
                original_conversation_id = EXCLUDED.original_conversation_id,
                assistant_id = EXCLUDED.assistant_id,
                user_id = EXCLUDED.user_id,
                variables = EXCLUDED.variables,
                version = conversation_variables.version + 1,
                updated_at = NOW()
        """, (key, conversation_id, assistant_id, user_id, json_dumps(variables or {})))

        cur.execute("""
            INSERT INTO conversation_summaries
            (conversation_id, original_conversation_id, assistant_id, user_id, summary, message_count, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, NOW())
            ON CONFLICT (conversation_id)
            DO UPDATE SET
                original_conversation_id = EXCLUDED.original_conversation_id,
                assistant_id = EXCLUDED.assistant_id,
                user_id = EXCLUDED.user_id,
                summary = EXCLUDED.summary,
                message_count = EXCLUDED.message_count,
                updated_at = NOW()
        """, (key, conversation_id, assistant_id, user_id, summary or "", int(message_count or 0)))


def clear_conversation_data(conversation_id, assistant_id: Optional[str] = None):
    assistant_hint = assistant_id or get_assistant_hint(conversation_id)

    with get_conn() as conn:
        with conn.cursor() as cur:
            keys = candidate_conversation_keys(conversation_id, assistant_hint)

            if assistant_hint:
                cur.execute("""
                    SELECT id FROM conversations
                    WHERE assistant_id = %s AND conversation_id = %s
                """, (assistant_hint, conversation_id))
                keys.extend([row[0] for row in cur.fetchall()])
            else:
                cur.execute("""
                    SELECT id FROM conversations
                    WHERE conversation_id = %s OR id = %s
                """, (conversation_id, conversation_id))
                keys.extend([row[0] for row in cur.fetchall()])

            keys = list(dict.fromkeys([key for key in keys if key]))

            for key in keys:
                cur.execute("DELETE FROM messages WHERE conversation_id = %s", (key,))
                cur.execute("DELETE FROM conversation_states WHERE conversation_id = %s", (key,))
                cur.execute("DELETE FROM conversation_variables WHERE conversation_id = %s", (key,))
                cur.execute("DELETE FROM conversation_summaries WHERE conversation_id = %s", (key,))
                cur.execute("DELETE FROM conversation_rag_cache WHERE conversation_id = %s", (key,))
                cur.execute("DELETE FROM conversations WHERE id = %s", (key,))

            if assistant_hint:
                cur.execute("""
                    DELETE FROM conversation_idempotency
                    WHERE assistant_id = %s AND conversation_id = %s
                """, (assistant_hint, conversation_id))
            else:
                cur.execute("""
                    DELETE FROM conversation_idempotency
                    WHERE conversation_id = %s
                """, (conversation_id,))

    if safe_text(conversation_id) in _ASSISTANT_HINT_BY_CONVERSATION:
        _ASSISTANT_HINT_BY_CONVERSATION.pop(safe_text(conversation_id), None)


def save_idempotency_record(
    assistant_id: str,
    conversation_id: str,
    request_key: str,
    final_answer: str = "",
    response_payload: Optional[Dict[str, Any]] = None,
    trace: Optional[Dict[str, Any]] = None,
    trace_id: str = "",
    state_version: int = 0,
):
    if not assistant_id or not conversation_id or not request_key:
        return

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO conversation_idempotency
                (assistant_id, conversation_id, request_key, trace_id, final_answer, response_payload, trace, state_version, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT (assistant_id, conversation_id, request_key)
                DO UPDATE SET
                    trace_id = EXCLUDED.trace_id,
                    final_answer = EXCLUDED.final_answer,
                    response_payload = EXCLUDED.response_payload,
                    trace = EXCLUDED.trace,
                    state_version = EXCLUDED.state_version,
                    updated_at = NOW()
            """, (
                assistant_id,
                conversation_id,
                request_key,
                trace_id or "",
                final_answer or "",
                json_dumps(response_payload or {}),
                json_dumps(trace or {}),
                int(state_version or 0),
            ))


def get_idempotency_record(assistant_id: str, conversation_id: str, request_key: str):
    if not assistant_id or not conversation_id or not request_key:
        return None

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT trace_id, final_answer, response_payload, trace, state_version, updated_at
                FROM conversation_idempotency
                WHERE assistant_id = %s
                  AND conversation_id = %s
                  AND request_key = %s
            """, (assistant_id, conversation_id, request_key))
            row = cur.fetchone()

    if not row:
        return None

    return {
        "trace_id": row[0] or "",
        "final_answer": row[1] or "",
        "response_payload": normalize_json(row[2], {}),
        "trace": normalize_json(row[3], {}),
        "state_version": row[4] or 0,
        "updated_at": row[5].isoformat() if row[5] else None,
    }


def upsert_knowledge_document(assistant_id, document_id, title, metadata, chunk_count):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO knowledge_documents
                (assistant_id, document_id, title, metadata, chunk_count, updated_at)
                VALUES (%s, %s, %s, %s, %s, NOW())
                ON CONFLICT (assistant_id, document_id)
                DO UPDATE SET
                    title = EXCLUDED.title,
                    metadata = EXCLUDED.metadata,
                    chunk_count = EXCLUDED.chunk_count,
                    updated_at = NOW()
            """, (
                assistant_id,
                document_id,
                title,
                json_dumps(metadata or {}),
                chunk_count,
            ))


def list_knowledge_documents(assistant_id):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT document_id, title, metadata, chunk_count, updated_at
                FROM knowledge_documents
                WHERE assistant_id = %s
                ORDER BY updated_at DESC
            """, (assistant_id,))
            rows = cur.fetchall()

    return [
        {
            "document_id": r[0],
            "title": r[1],
            "metadata": normalize_json(r[2], {}),
            "chunk_count": r[3],
            "updated_at": r[4].isoformat() if r[4] else None,
        }
        for r in rows
    ]


def make_memory_key(memory_type, memory_text):
    normalized = " ".join(str(memory_text or "").lower().strip().split())
    material = f"{memory_type}:{normalized}"
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]
    return f"{memory_type}:{normalized[:120]}:{digest}"


def upsert_long_term_memory(assistant_id, user_id, memory_text, memory_type, importance, confidence):
    memory_key = make_memory_key(memory_type, memory_text)

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO long_term_memories
                (assistant_id, user_id, memory_key, memory_text, memory_type, importance, confidence, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT (assistant_id, user_id, memory_key)
                DO UPDATE SET
                    memory_text = EXCLUDED.memory_text,
                    memory_type = EXCLUDED.memory_type,
                    importance = GREATEST(long_term_memories.importance, EXCLUDED.importance),
                    confidence = GREATEST(long_term_memories.confidence, EXCLUDED.confidence),
                    updated_at = NOW()
                RETURNING memory_key
            """, (
                assistant_id,
                user_id,
                memory_key,
                memory_text,
                memory_type,
                importance,
                confidence,
            ))
            return cur.fetchone()[0]


def list_long_term_memories(assistant_id, user_id):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT memory_text, memory_type, importance, confidence, updated_at
                FROM long_term_memories
                WHERE assistant_id = %s AND user_id = %s
                ORDER BY importance DESC, updated_at DESC
                LIMIT 50
            """, (assistant_id, user_id))
            rows = cur.fetchall()

    return [
        {
            "text": r[0],
            "type": r[1],
            "importance": float(r[2]),
            "confidence": float(r[3]),
            "updated_at": r[4].isoformat() if r[4] else None,
        }
        for r in rows
    ]


def save_rag_cache(conversation_id, assistant_id, user_id, query, knowledge_payload, compressed_payload):
    remember_assistant_hint(conversation_id, assistant_id)
    key = get_primary_conversation_key(assistant_id, conversation_id)

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO conversation_rag_cache
                (conversation_id, original_conversation_id, assistant_id, user_id, query, knowledge_payload, compressed_payload, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT (conversation_id)
                DO UPDATE SET
                    original_conversation_id = EXCLUDED.original_conversation_id,
                    assistant_id = EXCLUDED.assistant_id,
                    user_id = EXCLUDED.user_id,
                    query = EXCLUDED.query,
                    knowledge_payload = EXCLUDED.knowledge_payload,
                    compressed_payload = EXCLUDED.compressed_payload,
                    updated_at = NOW()
            """, (
                key,
                conversation_id,
                assistant_id,
                user_id,
                query,
                json_dumps(knowledge_payload or []),
                json_dumps(compressed_payload or []),
            ))


def get_rag_cache(conversation_id, max_age_minutes=30, assistant_id: Optional[str] = None):
    with get_conn() as conn:
        with conn.cursor() as cur:
            keys = candidate_conversation_keys(conversation_id, assistant_id)
            for key in keys:
                cur.execute("""
                    SELECT query, knowledge_payload, compressed_payload, updated_at
                    FROM conversation_rag_cache
                    WHERE conversation_id = %s
                    AND updated_at >= NOW() - (%s || ' minutes')::interval
                    LIMIT 1
                """, (key, str(max_age_minutes)))
                row = cur.fetchone()

                if row:
                    return {
                        "query": row[0],
                        "knowledge_payload": normalize_json(row[1], []),
                        "compressed_payload": normalize_json(row[2], []),
                        "updated_at": row[3].isoformat() if row[3] else None,
                    }

    return None


def log_model_usage(
    assistant_id,
    conversation_id,
    user_id,
    model,
    purpose,
    input_tokens=0,
    output_tokens=0,
    estimated_cost_usd=0,
    metadata=None,
):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO model_usage
                (assistant_id, conversation_id, user_id, model, purpose, input_tokens, output_tokens, estimated_cost_usd, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                assistant_id,
                conversation_id,
                user_id,
                model,
                purpose,
                input_tokens,
                output_tokens,
                estimated_cost_usd,
                json_dumps(metadata or {}),
            ))


def log_estimated_usage(
    assistant_id,
    conversation_id,
    user_id,
    model,
    purpose,
    input_obj=None,
    output_text="",
    metadata=None,
):
    input_tokens = estimate_tokens_from_obj(input_obj or {})
    output_tokens = estimate_tokens_from_text(output_text or "")

    log_model_usage(
        assistant_id=assistant_id,
        conversation_id=conversation_id,
        user_id=user_id,
        model=model,
        purpose=purpose,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_cost_usd=0,
        metadata=metadata or {},
    )
