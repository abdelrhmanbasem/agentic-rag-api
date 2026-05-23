import json
import psycopg
from app.config import (
    POSTGRES_HOST,
    POSTGRES_PORT,
    POSTGRES_DB,
    POSTGRES_USER,
    POSTGRES_PASSWORD,
)


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
                tone TEXT DEFAULT '',
                memory_policy TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT NOW()
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


def save_message(conversation_id, assistant_id, user_id, role, content):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO messages (conversation_id, assistant_id, user_id, role, content)
                VALUES (%s, %s, %s, %s, %s)
            """, (conversation_id, assistant_id, user_id, role, content))


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


def ensure_conversation(conversation_id, assistant_id, user_id, channel):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO conversations (id, assistant_id, user_id, channel, updated_at)
                VALUES (%s, %s, %s, %s, NOW())
                ON CONFLICT (id)
                DO UPDATE SET updated_at = NOW()
            """, (conversation_id, assistant_id, user_id, channel))


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
                INSERT INTO assistants (id, name, system_prompt, tone, memory_policy)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (id)
                DO UPDATE SET
                    name = EXCLUDED.name,
                    system_prompt = EXCLUDED.system_prompt,
                    tone = EXCLUDED.tone,
                    memory_policy = EXCLUDED.memory_policy
            """, (assistant_id, name, system_prompt, tone, memory_policy))


def save_schema(assistant_id, schema):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO assistant_variable_schemas (assistant_id, schema, updated_at)
                VALUES (%s, %s, NOW())
                ON CONFLICT (assistant_id)
                DO UPDATE SET schema = EXCLUDED.schema, updated_at = NOW()
            """, (assistant_id, json.dumps(schema)))


def get_schema(assistant_id):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT schema
                FROM assistant_variable_schemas
                WHERE assistant_id = %s
            """, (assistant_id,))
            row = cur.fetchone()
            return row[0] if row else {}


def get_variables(conversation_id):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT variables
                FROM conversation_variables
                WHERE conversation_id = %s
            """, (conversation_id,))
            row = cur.fetchone()
            return row[0] if row else {}


def save_variables(conversation_id, assistant_id, user_id, variables):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO conversation_variables
                (conversation_id, assistant_id, user_id, variables, updated_at)
                VALUES (%s, %s, %s, %s, NOW())
                ON CONFLICT (conversation_id)
                DO UPDATE SET variables = EXCLUDED.variables, updated_at = NOW()
            """, (conversation_id, assistant_id, user_id, json.dumps(variables)))


def get_summary(conversation_id):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT summary
                FROM conversation_summaries
                WHERE conversation_id = %s
            """, (conversation_id,))
            row = cur.fetchone()
            return row[0] if row else ""


def save_summary(conversation_id, assistant_id, user_id, summary):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO conversation_summaries
                (conversation_id, assistant_id, user_id, summary, updated_at)
                VALUES (%s, %s, %s, %s, NOW())
                ON CONFLICT (conversation_id)
                DO UPDATE SET summary = EXCLUDED.summary, updated_at = NOW()
            """, (conversation_id, assistant_id, user_id, summary))
