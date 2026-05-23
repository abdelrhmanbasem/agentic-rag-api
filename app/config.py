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
