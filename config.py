import os
from zoneinfo import ZoneInfo

ENV = os.getenv("ENV", "development").lower()
MOCK_MODE = os.getenv("MOCK_MODE", "true").lower() == "true"

APP_SECRET = os.getenv("APP_SECRET", "")

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgres")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "rag_db")
POSTGRES_USER = os.getenv("POSTGRES_USER", "rag_user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

MODEL_PLANNER = os.getenv("MODEL_PLANNER", "gpt-4o")
MODEL_SUBAGENT = os.getenv("MODEL_SUBAGENT", "gpt-4o-mini")
MODEL_RESPONSE = os.getenv("MODEL_RESPONSE", "gpt-4o")
MODEL_EXTRACTION = os.getenv("MODEL_EXTRACTION", "gpt-4o-mini")
MODEL_MEMORY = os.getenv("MODEL_MEMORY", "gpt-4o-mini")
MODEL_QUALITY = os.getenv("MODEL_QUALITY", "gpt-4o-mini")

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", "1536"))

APP_TIMEZONE = os.getenv("APP_TIMEZONE", "Africa/Cairo")
try:
    TZ = ZoneInfo(APP_TIMEZONE)
except Exception:
    TZ = ZoneInfo("Africa/Cairo")

RECENT_MESSAGES_LIMIT = int(os.getenv("RECENT_MESSAGES_LIMIT", "14"))
SUMMARY_TRIGGER_MESSAGE_COUNT = int(os.getenv("SUMMARY_TRIGGER_MESSAGE_COUNT", "8"))

KNOWLEDGE_TOP_K = int(os.getenv("KNOWLEDGE_TOP_K", "8"))
MEMORY_TOP_K = int(os.getenv("MEMORY_TOP_K", "6"))

RAG_MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.25"))
MEMORY_MIN_SCORE = float(os.getenv("MEMORY_MIN_SCORE", "0.35"))

KNOWLEDGE_COMPRESS_MAX_CHARS = int(os.getenv("KNOWLEDGE_COMPRESS_MAX_CHARS", "1000"))

MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "750"))
QUALITY_GUARD_ENABLED = os.getenv("QUALITY_GUARD_ENABLED", "true").lower() == "true"

ESTIMATE_CHARS_PER_TOKEN = int(os.getenv("ESTIMATE_CHARS_PER_TOKEN", "4"))


def validate_runtime_config() -> None:
    if ENV in {"production", "prod"} and not MOCK_MODE:
        missing = []
        if not APP_SECRET:
            missing.append("APP_SECRET")
        if not OPENAI_API_KEY:
            missing.append("OPENAI_API_KEY")
        if not POSTGRES_PASSWORD:
            missing.append("POSTGRES_PASSWORD")
        if missing:
            raise RuntimeError(f"Missing required production environment variables: {', '.join(missing)}")
