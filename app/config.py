import os

APP_SECRET = os.getenv("APP_SECRET", "")
MOCK_MODE = os.getenv("MOCK_MODE", "true").lower() == "true"

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgres")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "rag_db")
POSTGRES_USER = os.getenv("POSTGRES_USER", "rag_user")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

MODEL_ROUTER = os.getenv("MODEL_ROUTER", "gpt-4o-mini")
MODEL_MEMORY = os.getenv("MODEL_MEMORY", "gpt-4o-mini")
MODEL_EXTRACTION = os.getenv("MODEL_EXTRACTION", "gpt-4o-mini")
MODEL_NORMAL = os.getenv("MODEL_NORMAL", "gpt-4o-mini")
MODEL_STRONG = os.getenv("MODEL_STRONG", "gpt-4o-mini")

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", "1536"))

RECENT_MESSAGES_LIMIT = int(os.getenv("RECENT_MESSAGES_LIMIT", "10"))
SUMMARY_TRIGGER_MESSAGE_COUNT = int(os.getenv("SUMMARY_TRIGGER_MESSAGE_COUNT", "6"))
KNOWLEDGE_TOP_K = int(os.getenv("KNOWLEDGE_TOP_K", "4"))
MEMORY_TOP_K = int(os.getenv("MEMORY_TOP_K", "5"))

MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "700"))

RAG_MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.0"))
MEMORY_MIN_SCORE = float(os.getenv("MEMORY_MIN_SCORE", "0.0"))

RAG_CACHE_ENABLED = os.getenv("RAG_CACHE_ENABLED", "true").lower() == "true"
RAG_CACHE_MAX_AGE_MINUTES = int(os.getenv("RAG_CACHE_MAX_AGE_MINUTES", "30"))
RAG_CACHE_MAX_ITEMS = int(os.getenv("RAG_CACHE_MAX_ITEMS", "4"))

KNOWLEDGE_COMPRESS_ENABLED = os.getenv("KNOWLEDGE_COMPRESS_ENABLED", "true").lower() == "true"
KNOWLEDGE_COMPRESS_MAX_CHARS = int(os.getenv("KNOWLEDGE_COMPRESS_MAX_CHARS", "700"))

ESTIMATE_CHARS_PER_TOKEN = int(os.getenv("ESTIMATE_CHARS_PER_TOKEN", "4"))
