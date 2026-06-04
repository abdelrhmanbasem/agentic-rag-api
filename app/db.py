import json
import psycopg
from typing import Any, Dict
from app.config import (
    POSTGRES_HOST,
    POSTGRES_PORT,
    POSTGRES_DB,
    POSTGRES_USER,
    POSTGRES_PASSWORD,
    ESTIMATE_CHARS_PER_TOKEN,
)
from app.defaults import DEFAULT_AGENT_CONFIG


def get_conn():
    return psycopg.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        autocommit=True,
    )


def init_db():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
            CREATE TABLE IF NOT EXISTS assistants (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                system_prompt TEXT NOT NULL,
                agent_config JSONB NOT NULL DEFAULT '{}',
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            );
            """)

            cur.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                assistant_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                channel TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
            );
            """)

            cur.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id BIGSERIAL PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                assistant_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT NOW()
            );
            """)

            cur.execute("""
            CREATE TABLE IF NOT EXISTS conversation_summaries (
                conversation_id TEXT PRIMARY KEY,
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
                assistant_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                variables JSONB NOT NULL DEFAULT '{}',
                updated_at TIMESTAMP DEFAULT NOW()
            );
            """)

            cur.execute("""
            CREATE TABLE IF NOT EXISTS conversations_state (
                conversation_id TEXT PRIMARY KEY,
                assistant_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                channel TEXT DEFAULT '',
                state JSONB NOT NULL DEFAULT '{}',
                variables JSONB NOT NULL DEFAULT '{}',
                summary TEXT NOT NULL DEFAULT '',
                message_count INT DEFAULT 0,
                created_at TIMESTAMP DEFAULT NOW(),
                updated_at TIMESTAMP DEFAULT NOW()
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

            cur.execute("ALTER TABLE assistants ADD COLUMN IF NOT EXISTS agent_config JSONB NOT NULL DEFAULT '{}';")
            cur.execute("ALTER TABLE assistants ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP DEFAULT NOW();")
            cur.execute("ALTER TABLE conversation_summaries ADD COLUMN IF NOT EXISTS message_count INT DEFAULT 0;")
            cur.execute("ALTER TABLE model_usage ADD COLUMN IF NOT EXISTS metadata JSONB NOT NULL DEFAULT '{}';")

            cur.execute("ALTER TABLE conversations_state ADD COLUMN IF NOT EXISTS channel TEXT DEFAULT '';")
            cur.execute("ALTER TABLE conversations_state ADD COLUMN IF NOT EXISTS state JSONB NOT NULL DEFAULT '{}';")
            cur.execute("ALTER TABLE conversations_state ADD COLUMN IF NOT EXISTS variables JSONB NOT NULL DEFAULT '{}';")
            cur.execute("ALTER TABLE conversations_state ADD COLUMN IF NOT EXISTS summary TEXT NOT NULL DEFAULT '';")
            cur.execute("ALTER TABLE conversations_state ADD COLUMN IF NOT EXISTS message_count INT DEFAULT 0;")
            cur.execute("ALTER TABLE conversations_state ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT NOW();")
            cur.execute("ALTER TABLE conversations_state ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP DEFAULT NOW();")

            cur.execute("CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages (conversation_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_messages_conversation_created ON messages (conversation_id, created_at DESC);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_long_term_memories_user ON long_term_memories (assistant_id, user_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_conversations_state_assistant_user ON conversations_state (assistant_id, user_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_conversations_state_updated ON conversations_state (updated_at DESC);")


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base or {})
    for key, value in (override or {}).items():
        if isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def normalize_agent_config(config: Dict[str, Any]) -> Dict[str, Any]:
    return deep_merge(DEFAULT_AGENT_CONFIG, config or {})


def safe_json_obj(value: Any, default: Any = None) -> Any:
    if default is None:
        default = {}

    if value is None:
        return default

    if isinstance(value, (dict, list)):
        return value

    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if parsed is not None else default
        except Exception:
            return default

    return default


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
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO conversations (id, assistant_id, user_id, channel, updated_at)
                VALUES (%s, %s, %s, %s, NOW())
                ON CONFLICT (id)
                DO UPDATE SET
                    assistant_id = EXCLUDED.assistant_id,
                    user_id = EXCLUDED.user_id,
                    channel = EXCLUDED.channel,
                    updated_at = NOW()
            """, (conversation_id, assistant_id, user_id, channel))


def save_message(conversation_id, assistant_id, user_id, role, content):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO messages (conversation_id, assistant_id, user_id, role, content)
                VALUES (%s, %s, %s, %s, %s)
            """, (conversation_id, assistant_id, user_id, role, content))


def count_messages(conversation_id):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM messages WHERE conversation_id = %s", (conversation_id,))
            return cur.fetchone()[0]


def get_recent_messages(conversation_id, limit=10):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT role, content
                FROM messages
                WHERE conversation_id = %s
                ORDER BY id DESC
                LIMIT %s
            """, (conversation_id, limit))
            rows = cur.fetchall()
    rows.reverse()
    return [{"role": r[0], "content": r[1]} for r in rows]


def upsert_assistant(assistant_id, name, system_prompt, agent_config=None):
    config = normalize_agent_config(agent_config or {})
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO assistants (id, name, system_prompt, agent_config, updated_at)
                VALUES (%s, %s, %s, %s, NOW())
                ON CONFLICT (id)
                DO UPDATE SET
                    name = EXCLUDED.name,
                    system_prompt = EXCLUDED.system_prompt,
                    agent_config = EXCLUDED.agent_config,
                    updated_at = NOW()
            """, (assistant_id, name, system_prompt, json.dumps(config, ensure_ascii=False)))


def get_or_create_assistant(assistant_id):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, name, system_prompt, agent_config
                FROM assistants
                WHERE id = %s
            """, (assistant_id,))
            row = cur.fetchone()

            if row:
                return {
                    "id": row[0],
                    "name": row[1],
                    "system_prompt": row[2],
                    "agent_config": normalize_agent_config(row[3] or {}),
                }

            default_prompt = "You are a helpful, grounded, human-like AI assistant."
            config = normalize_agent_config({})
            cur.execute("""
                INSERT INTO assistants (id, name, system_prompt, agent_config)
                VALUES (%s, %s, %s, %s)
            """, (assistant_id, assistant_id, default_prompt, json.dumps(config, ensure_ascii=False)))

            return {
                "id": assistant_id,
                "name": assistant_id,
                "system_prompt": default_prompt,
                "agent_config": config,
            }


def save_schema(assistant_id, schema):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO assistant_variable_schemas (assistant_id, schema, updated_at)
                VALUES (%s, %s, NOW())
                ON CONFLICT (assistant_id)
                DO UPDATE SET schema = EXCLUDED.schema, updated_at = NOW()
            """, (assistant_id, json.dumps(schema or {}, ensure_ascii=False)))


def get_schema(assistant_id):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT schema FROM assistant_variable_schemas WHERE assistant_id = %s", (assistant_id,))
            row = cur.fetchone()
            return row[0] if row else {}


def get_variables(conversation_id):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT variables FROM conversation_variables WHERE conversation_id = %s", (conversation_id,))
            row = cur.fetchone()
            return row[0] if row else {}


def save_variables(conversation_id, assistant_id, user_id, variables):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO conversation_variables (conversation_id, assistant_id, user_id, variables, updated_at)
                VALUES (%s, %s, %s, %s, NOW())
                ON CONFLICT (conversation_id)
                DO UPDATE SET
                    assistant_id = EXCLUDED.assistant_id,
                    user_id = EXCLUDED.user_id,
                    variables = EXCLUDED.variables,
                    updated_at = NOW()
            """, (conversation_id, assistant_id, user_id, json.dumps(variables or {}, ensure_ascii=False)))


def load_conversation_state(conversation_id):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT conversation_id, assistant_id, user_id, channel, state, variables, summary, message_count, updated_at
                FROM conversations_state
                WHERE conversation_id = %s
            """, (conversation_id,))
            row = cur.fetchone()

    if not row:
        return {}

    return {
        "conversation_id": row[0],
        "assistant_id": row[1],
        "user_id": row[2],
        "channel": row[3] or "",
        "state": safe_json_obj(row[4], {}),
        "variables": safe_json_obj(row[5], {}),
        "summary": row[6] or "",
        "message_count": int(row[7] or 0),
        "updated_at": row[8].isoformat() if row[8] else None,
    }


def save_conversation_state(
    conversation_id,
    assistant_id,
    user_id,
    channel="",
    state=None,
    variables=None,
    summary="",
    message_count=None
):
    state_obj = state if isinstance(state, dict) else {}
    variables_obj = variables if isinstance(variables, dict) else {}

    if message_count is None:
        try:
            message_count = count_messages(conversation_id)
        except Exception:
            message_count = 0

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO conversations_state
                (conversation_id, assistant_id, user_id, channel, state, variables, summary, message_count, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
                ON CONFLICT (conversation_id)
                DO UPDATE SET
                    assistant_id = EXCLUDED.assistant_id,
                    user_id = EXCLUDED.user_id,
                    channel = EXCLUDED.channel,
                    state = EXCLUDED.state,
                    variables = EXCLUDED.variables,
                    summary = EXCLUDED.summary,
                    message_count = EXCLUDED.message_count,
                    updated_at = NOW()
            """, (
                conversation_id,
                assistant_id,
                user_id,
                channel or "",
                json.dumps(state_obj, ensure_ascii=False),
                json.dumps(variables_obj, ensure_ascii=False),
                summary or "",
                int(message_count or 0),
            ))


def patch_conversation_state(
    conversation_id,
    assistant_id,
    user_id,
    channel="",
    state_patch=None,
    variables_patch=None,
    summary=None,
    message_count=None
):
    existing = load_conversation_state(conversation_id)

    existing_state = existing.get("state", {}) if isinstance(existing, dict) else {}
    existing_variables = existing.get("variables", {}) if isinstance(existing, dict) else {}

    merged_state = deep_merge(existing_state, state_patch or {})
    merged_variables = deep_merge(existing_variables, variables_patch or {})

    final_summary = existing.get("summary", "") if isinstance(existing, dict) else ""
    if summary is not None:
        final_summary = summary

    final_message_count = existing.get("message_count", 0) if isinstance(existing, dict) else 0
    if message_count is not None:
        final_message_count = message_count

    save_conversation_state(
        conversation_id=conversation_id,
        assistant_id=assistant_id,
        user_id=user_id,
        channel=channel or existing.get("channel", "") if isinstance(existing, dict) else channel,
        state=merged_state,
        variables=merged_variables,
        summary=final_summary,
        message_count=final_message_count
    )

    return {
        "conversation_id": conversation_id,
        "assistant_id": assistant_id,
        "user_id": user_id,
        "channel": channel or existing.get("channel", "") if isinstance(existing, dict) else channel,
        "state": merged_state,
        "variables": merged_variables,
        "summary": final_summary,
        "message_count": final_message_count,
    }


def delete_conversation_state(conversation_id):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM conversations_state WHERE conversation_id = %s", (conversation_id,))


def clear_conversation_data(conversation_id):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM messages WHERE conversation_id = %s", (conversation_id,))
            cur.execute("DELETE FROM conversation_summaries WHERE conversation_id = %s", (conversation_id,))
            cur.execute("DELETE FROM conversation_variables WHERE conversation_id = %s", (conversation_id,))
            cur.execute("DELETE FROM conversations_state WHERE conversation_id = %s", (conversation_id,))
            cur.execute("DELETE FROM conversations WHERE id = %s", (conversation_id,))


def get_summary(conversation_id):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT summary FROM conversation_summaries WHERE conversation_id = %s", (conversation_id,))
            row = cur.fetchone()
            return row[0] if row else ""


def save_summary(conversation_id, assistant_id, user_id, summary):
    message_count = count_messages(conversation_id)
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO conversation_summaries (conversation_id, assistant_id, user_id, summary, message_count, updated_at)
                VALUES (%s, %s, %s, %s, %s, NOW())
                ON CONFLICT (conversation_id)
                DO UPDATE SET
                    assistant_id = EXCLUDED.assistant_id,
                    user_id = EXCLUDED.user_id,
                    summary = EXCLUDED.summary,
                    message_count = EXCLUDED.message_count,
                    updated_at = NOW()
            """, (conversation_id, assistant_id, user_id, summary or "", message_count))


def get_summary_message_count(conversation_id):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT message_count FROM conversation_summaries WHERE conversation_id = %s", (conversation_id,))
            row = cur.fetchone()
            return row[0] if row else 0


def upsert_knowledge_document(assistant_id, document_id, title, metadata, chunk_count):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO knowledge_documents (assistant_id, document_id, title, metadata, chunk_count, updated_at)
                VALUES (%s, %s, %s, %s, %s, NOW())
                ON CONFLICT (assistant_id, document_id)
                DO UPDATE SET title = EXCLUDED.title, metadata = EXCLUDED.metadata, chunk_count = EXCLUDED.chunk_count, updated_at = NOW()
            """, (assistant_id, document_id, title, json.dumps(metadata or {}, ensure_ascii=False), chunk_count))


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
            "metadata": r[2],
            "chunk_count": r[3],
            "updated_at": r[4].isoformat() if r[4] else None,
        }
        for r in rows
    ]


def make_memory_key(memory_type, memory_text):
    normalized = " ".join((memory_text or "").lower().strip().split())
    return f"{memory_type}:{normalized[:180]}"


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
            """, (assistant_id, user_id, memory_key, memory_text, memory_type, importance, confidence))
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


def log_model_usage(assistant_id, conversation_id, user_id, model, purpose, input_tokens=0, output_tokens=0, estimated_cost_usd=0, metadata=None):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO model_usage
                (assistant_id, conversation_id, user_id, model, purpose, input_tokens, output_tokens, estimated_cost_usd, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                assistant_id, conversation_id, user_id, model, purpose,
                input_tokens, output_tokens, estimated_cost_usd,
                json.dumps(metadata or {}, ensure_ascii=False),
            ))
