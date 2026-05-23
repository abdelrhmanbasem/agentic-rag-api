import os
from typing import Dict, Any

import psycopg
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI(title="Agentic RAG API")

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL_ROUTER = os.getenv("MODEL_ROUTER", "gpt-4.1-nano")
MODEL_NORMAL = os.getenv("MODEL_NORMAL", "gpt-4.1-mini")
MODEL_STRONG = os.getenv("MODEL_STRONG", "gpt-4.1")


class ChatRequest(BaseModel):
    assistant_id: str
    user_id: str
    conversation_id: str
    message: str
    channel: str = "n8n"
    metadata: Dict[str, Any] = {}


class AssistantRequest(BaseModel):
    assistant_id: str
    name: str
    system_prompt: str
    tone: str = "clear, helpful, concise"
    memory_policy: str = "Remember stable preferences, goals, constraints, and important facts."


def check_auth(x_api_key: str):
    if x_api_key != os.getenv("APP_SECRET"):
        raise HTTPException(status_code=401, detail="Unauthorized")


def db():
    return psycopg.connect(
        host=os.getenv("POSTGRES_HOST", "postgres"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        dbname=os.getenv("POSTGRES_DB", "rag_db"),
        user=os.getenv("POSTGRES_USER", "rag_user"),
        password=os.getenv("POSTGRES_PASSWORD"),
        autocommit=True,
    )


def init_db():
    with db() as conn:
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


@app.on_event("startup")
def startup():
    init_db()


@app.get("/health")
def health():
    return {"status": "ok"}


def save_message(conversation_id, assistant_id, user_id, role, content):
    with db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO messages (conversation_id, assistant_id, user_id, role, content)
                VALUES (%s, %s, %s, %s, %s)
            """, (conversation_id, assistant_id, user_id, role, content))


def get_recent_messages(conversation_id, limit=10):
    with db() as conn:
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


def get_or_create_assistant(assistant_id):
    with db() as conn:
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
                    "memory_policy": row[4]
                }

            default_prompt = "You are a helpful AI assistant. Answer clearly and helpfully."
            cur.execute("""
                INSERT INTO assistants (id, name, system_prompt, tone, memory_policy)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                assistant_id,
                assistant_id,
                default_prompt,
                "clear, helpful, concise",
                "Remember stable preferences, goals, constraints, and important facts."
            ))

            return {
                "id": assistant_id,
                "name": assistant_id,
                "system_prompt": default_prompt,
                "tone": "clear, helpful, concise",
                "memory_policy": "Remember stable preferences, goals, constraints, and important facts."
            }


def ensure_conversation(conversation_id, assistant_id, user_id, channel):
    with db() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO conversations (id, assistant_id, user_id, channel, updated_at)
                VALUES (%s, %s, %s, %s, NOW())
                ON CONFLICT (id)
                DO UPDATE SET updated_at = NOW()
            """, (conversation_id, assistant_id, user_id, channel))


def choose_model(message):
    lowered = message.lower()
    angry_words = ["angry", "refund", "complaint", "unacceptable", "lawsuit", "terrible"]

    if len(message) < 25:
        return MODEL_ROUTER, "cheap"

    if any(word in lowered for word in angry_words):
        return MODEL_STRONG, "strong"

    return MODEL_NORMAL, "normal"


def generate_answer(model, assistant, recent_messages):
    system_prompt = f"""
{assistant["system_prompt"]}

Tone:
{assistant["tone"]}

Memory policy:
{assistant["memory_policy"]}

Rules:
- Use the recent conversation context.
- Be concise and useful.
- Do not mention internal systems.
"""

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(recent_messages)

    response = openai_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.4,
        max_tokens=600,
    )

    return response.choices[0].message.content


@app.post("/assistants")
def create_assistant(req: AssistantRequest, x_api_key: str = Header(default="")):
    check_auth(x_api_key)

    with db() as conn:
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
            """, (
                req.assistant_id,
                req.name,
                req.system_prompt,
                req.tone,
                req.memory_policy
            ))

    return {"status": "saved", "assistant_id": req.assistant_id}


@app.post("/chat")
def chat(req: ChatRequest, x_api_key: str = Header(default="")):
    check_auth(x_api_key)

    assistant = get_or_create_assistant(req.assistant_id)
    ensure_conversation(req.conversation_id, req.assistant_id, req.user_id, req.channel)

    save_message(req.conversation_id, req.assistant_id, req.user_id, "user", req.message)

    recent_messages = get_recent_messages(req.conversation_id, limit=10)

    model, tier = choose_model(req.message)
    answer = generate_answer(model, assistant, recent_messages)

    save_message(req.conversation_id, req.assistant_id, req.user_id, "assistant", answer)

    return {
        "answer": answer,
        "assistant_id": req.assistant_id,
        "conversation_id": req.conversation_id,
        "model_used": model,
        "model_tier": tier,
        "memory_saved": True
    }
