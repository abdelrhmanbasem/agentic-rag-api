import os
from zoneinfo import ZoneInfo


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)

    if value is None:
        return default

    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)

    if value is None or str(value).strip() == "":
        return default

    try:
        return int(value)
    except Exception:
        return default


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)

    if value is None or str(value).strip() == "":
        return default

    try:
        return float(value)
    except Exception:
        return default


ENV = os.getenv("ENV", "development").strip().lower()

# Important for production:
# Default to real mode. Mock mode must be explicitly enabled.
MOCK_MODE = env_bool("MOCK_MODE", default=False)

APP_SECRET = os.getenv("APP_SECRET", os.getenv("API_KEY", "")).strip()

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgres").strip()
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432").strip()
POSTGRES_DB = os.getenv("POSTGRES_DB", "rag_db").strip()
POSTGRES_USER = os.getenv("POSTGRES_USER", "rag_user").strip()
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "").strip()

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333").strip()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "").strip()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

# LangGraph model roles:
# - planner/manifest decides intent, subagent, tool need, and next move
# - subagent produces private structured guidance only
# - response writes the final user-facing answer
# - quality validates hallucination, tone, and required facts
MODEL_PLANNER = os.getenv("MODEL_PLANNER", "gpt-4o").strip()
MODEL_SUBAGENT = os.getenv("MODEL_SUBAGENT", "gpt-4o-mini").strip()
MODEL_RESPONSE = os.getenv("MODEL_RESPONSE", "gpt-4o").strip()
MODEL_EXTRACTION = os.getenv("MODEL_EXTRACTION", "gpt-4o-mini").strip()
MODEL_MEMORY = os.getenv("MODEL_MEMORY", "gpt-4o-mini").strip()
MODEL_QUALITY = os.getenv("MODEL_QUALITY", "gpt-4o-mini").strip()

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small").strip()
VECTOR_SIZE = env_int("VECTOR_SIZE", 1536)

APP_TIMEZONE = os.getenv("APP_TIMEZONE", "Africa/Cairo").strip()

try:
    TZ = ZoneInfo(APP_TIMEZONE)
except Exception:
    TZ = ZoneInfo("Africa/Cairo")

RECENT_MESSAGES_LIMIT = env_int("RECENT_MESSAGES_LIMIT", 14)
SUMMARY_TRIGGER_MESSAGE_COUNT = env_int("SUMMARY_TRIGGER_MESSAGE_COUNT", 8)

KNOWLEDGE_TOP_K = env_int("KNOWLEDGE_TOP_K", 8)
MEMORY_TOP_K = env_int("MEMORY_TOP_K", 6)

RAG_MIN_SCORE = env_float("RAG_MIN_SCORE", 0.25)
MEMORY_MIN_SCORE = env_float("MEMORY_MIN_SCORE", 0.35)

KNOWLEDGE_COMPRESS_MAX_CHARS = env_int("KNOWLEDGE_COMPRESS_MAX_CHARS", 1000)

# Expert architecture Batch 5:
# Character-based chunking gives more predictable approximate token budgets,
# especially for Arabic and mixed Arabic/English documents.
CHUNK_CHARS = env_int("CHUNK_CHARS", 2600)
CHUNK_OVERLAP_CHARS = env_int("CHUNK_OVERLAP_CHARS", 300)

# Safety bounds so bad env values do not create huge/empty chunks.
if CHUNK_CHARS < 300:
    CHUNK_CHARS = 300

if CHUNK_CHARS > 12000:
    CHUNK_CHARS = 12000

if CHUNK_OVERLAP_CHARS < 0:
    CHUNK_OVERLAP_CHARS = 0

if CHUNK_OVERLAP_CHARS > CHUNK_CHARS // 2:
    CHUNK_OVERLAP_CHARS = CHUNK_CHARS // 2

MAX_OUTPUT_TOKENS = env_int("MAX_OUTPUT_TOKENS", 750)

# In the expert architecture, quality guard should be enabled by default.
QUALITY_GUARD_ENABLED = env_bool("QUALITY_GUARD_ENABLED", default=True)

ESTIMATE_CHARS_PER_TOKEN = env_int("ESTIMATE_CHARS_PER_TOKEN", 4)


def is_production() -> bool:
    return ENV in {"production", "prod"}


def validate_runtime_config() -> None:
    missing = []

    if not APP_SECRET:
        missing.append("APP_SECRET")

    if not OPENAI_API_KEY and not MOCK_MODE:
        missing.append("OPENAI_API_KEY")

    if not QDRANT_URL:
        missing.append("QDRANT_URL")

    if VECTOR_SIZE <= 0:
        missing.append("VECTOR_SIZE")

    if CHUNK_CHARS <= 0:
        missing.append("CHUNK_CHARS")

    if CHUNK_OVERLAP_CHARS < 0:
        missing.append("CHUNK_OVERLAP_CHARS")

    if CHUNK_OVERLAP_CHARS >= CHUNK_CHARS:
        raise RuntimeError("CHUNK_OVERLAP_CHARS must be smaller than CHUNK_CHARS.")

    if is_production():
        if MOCK_MODE:
            raise RuntimeError("MOCK_MODE must be false in production.")

        if not POSTGRES_PASSWORD:
            missing.append("POSTGRES_PASSWORD")

        if not QUALITY_GUARD_ENABLED:
            raise RuntimeError("QUALITY_GUARD_ENABLED must be true in production.")

    if missing:
        raise RuntimeError(
            "Missing required runtime environment variables: "
            + ", ".join(sorted(set(missing)))
        )


def runtime_config_summary() -> dict:
    """
    Safe non-secret summary for debug/health diagnostics.
    Do not include secret values.
    """
    return {
        "env": ENV,
        "mock_mode": MOCK_MODE,
        "qdrant_url": QDRANT_URL,
        "embed_model": EMBED_MODEL,
        "vector_size": VECTOR_SIZE,
        "model_planner": MODEL_PLANNER,
        "model_subagent": MODEL_SUBAGENT,
        "model_response": MODEL_RESPONSE,
        "model_quality": MODEL_QUALITY,
        "quality_guard_enabled": QUALITY_GUARD_ENABLED,
        "timezone": APP_TIMEZONE,
        "knowledge_top_k": KNOWLEDGE_TOP_K,
        "memory_top_k": MEMORY_TOP_K,
        "rag_min_score": RAG_MIN_SCORE,
        "memory_min_score": MEMORY_MIN_SCORE,
        "knowledge_compress_max_chars": KNOWLEDGE_COMPRESS_MAX_CHARS,
        "chunk_chars": CHUNK_CHARS,
        "chunk_overlap_chars": CHUNK_OVERLAP_CHARS,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "estimate_chars_per_token": ESTIMATE_CHARS_PER_TOKEN,
        "has_app_secret": bool(APP_SECRET),
        "has_openai_api_key": bool(OPENAI_API_KEY),
        "has_postgres_password": bool(POSTGRES_PASSWORD)
    }
