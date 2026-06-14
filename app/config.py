import os
from zoneinfo import ZoneInfo


# Architecture batch: 6.34-runtime-controls-no-hardcoding
# Architecture patch: 6.45-code-expert-runtime-controls-no-hardcoding


def env_str(name: str, default: str = "") -> str:
    value = os.getenv(name)

    if value is None:
        return default

    return str(value).strip()


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


def clamp_int(value: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = minimum

    return max(minimum, min(parsed, maximum))


def clamp_float(value: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except Exception:
        parsed = minimum

    return max(minimum, min(parsed, maximum))


ENV = env_str("ENV", "development").lower()

# Important for production:
# Default to real mode. Mock mode must be explicitly enabled.
MOCK_MODE = env_bool("MOCK_MODE", default=False)

# API/runtime identity is configurable so main.py does not need hardcoded titles.
API_TITLE = env_str("API_TITLE", "Modular Agentic LangGraph API")
API_SERVICE_NAME = env_str("API_SERVICE_NAME", "modular-agentic-langgraph-api")

APP_SECRET = env_str("APP_SECRET", env_str("API_KEY", ""))

DATA_DIR = env_str("DATA_DIR", "/app/data")
CONFIG_DIR = env_str("CONFIG_DIR", "/app/configs")

POSTGRES_HOST = env_str("POSTGRES_HOST", "postgres")
POSTGRES_PORT = env_str("POSTGRES_PORT", "5432")
POSTGRES_DB = env_str("POSTGRES_DB", "rag_db")
POSTGRES_USER = env_str("POSTGRES_USER", "rag_user")
POSTGRES_PASSWORD = env_str("POSTGRES_PASSWORD", "")

QDRANT_URL = env_str("QDRANT_URL", "http://qdrant:6333")
QDRANT_API_KEY = env_str("QDRANT_API_KEY", "")

OPENAI_API_KEY = env_str("OPENAI_API_KEY", "")

# Root rule #4: all chat/LLM role defaults must use GPT mini models only.
# gpt-4o-mini is the safest/current cost-efficient default used across the app.
DEFAULT_GPT_MINI_MODEL = env_str("DEFAULT_GPT_MINI_MODEL", "gpt-4o-mini")

# LangGraph model roles:
# - planner/manifest decides intent, subagent, tool need, and next move
# - subagent produces private structured guidance only
# - response writes the final user-facing answer
# - extraction performs focused semantic variable extraction from field descriptions
# - memory summarizes/writes useful long-term facts
# - quality validates hallucination, tone, and required facts
MODEL_PLANNER = env_str("MODEL_PLANNER", DEFAULT_GPT_MINI_MODEL)
MODEL_SUBAGENT = env_str("MODEL_SUBAGENT", DEFAULT_GPT_MINI_MODEL)
MODEL_RESPONSE = env_str("MODEL_RESPONSE", DEFAULT_GPT_MINI_MODEL)
# Optional deployment-wide default for low-risk/simple response routing.
# Per-assistant routing decisions still belong in domain_bundle.json.
MODEL_RESPONSE_SIMPLE = env_str("MODEL_RESPONSE_SIMPLE", DEFAULT_GPT_MINI_MODEL)
MODEL_EXTRACTION = env_str("MODEL_EXTRACTION", DEFAULT_GPT_MINI_MODEL)
MODEL_MEMORY = env_str("MODEL_MEMORY", DEFAULT_GPT_MINI_MODEL)
MODEL_QUALITY = env_str("MODEL_QUALITY", DEFAULT_GPT_MINI_MODEL)

EMBED_MODEL = env_str("EMBED_MODEL", "text-embedding-3-small")
VECTOR_SIZE = env_int("VECTOR_SIZE", 1536)

DEFAULT_TIMEZONE = env_str("DEFAULT_TIMEZONE", "Africa/Cairo")
APP_TIMEZONE = env_str("APP_TIMEZONE", DEFAULT_TIMEZONE)

try:
    TZ = ZoneInfo(APP_TIMEZONE)
except Exception:
    TZ = ZoneInfo(DEFAULT_TIMEZONE)


def is_gpt_mini_model(model_name: str) -> bool:
    """Return true only for GPT mini chat/LLM model IDs.

    Accepts IDs like gpt-4o-mini, gpt-4.1-mini, or openai/gpt-4o-mini.
    Embedding models are intentionally validated separately.
    """
    value = str(model_name or "").strip().lower()

    if value.startswith("openai/"):
        value = value.split("/", 1)[1]

    return value.startswith("gpt") and "mini" in value


def validate_gpt_mini_models_only() -> None:
    """Enforce root rule #4 for every configured chat/LLM role."""
    model_values = {
        "DEFAULT_GPT_MINI_MODEL": DEFAULT_GPT_MINI_MODEL,
        "MODEL_PLANNER": MODEL_PLANNER,
        "MODEL_SUBAGENT": MODEL_SUBAGENT,
        "MODEL_RESPONSE": MODEL_RESPONSE,
        "MODEL_RESPONSE_SIMPLE": MODEL_RESPONSE_SIMPLE,
        "MODEL_EXTRACTION": MODEL_EXTRACTION,
        "MODEL_MEMORY": MODEL_MEMORY,
        "MODEL_QUALITY": MODEL_QUALITY,
    }

    invalid = [
        f"{name}={value}"
        for name, value in model_values.items()
        if not is_gpt_mini_model(value)
    ]

    if invalid:
        raise RuntimeError(
            "Root rule #4 violation: all chat/LLM models must be GPT mini models only. Invalid: "
            + ", ".join(invalid)
        )

RECENT_MESSAGES_LIMIT = clamp_int(env_int("RECENT_MESSAGES_LIMIT", 14), 1, 200)
SUMMARY_TRIGGER_MESSAGE_COUNT = clamp_int(env_int("SUMMARY_TRIGGER_MESSAGE_COUNT", 8), 2, 200)

KNOWLEDGE_TOP_K = clamp_int(env_int("KNOWLEDGE_TOP_K", 8), 1, 50)
MEMORY_TOP_K = clamp_int(env_int("MEMORY_TOP_K", 6), 1, 50)

RAG_MIN_SCORE = clamp_float(env_float("RAG_MIN_SCORE", 0.25), 0.0, 1.0)
MEMORY_MIN_SCORE = clamp_float(env_float("MEMORY_MIN_SCORE", 0.35), 0.0, 1.0)

KNOWLEDGE_COMPRESS_MAX_CHARS = clamp_int(
    env_int("KNOWLEDGE_COMPRESS_MAX_CHARS", 1000),
    200,
    20000
)

# Character-based chunking gives predictable approximate token budgets,
# especially for Arabic and mixed Arabic/English documents.
CHUNK_CHARS = clamp_int(env_int("CHUNK_CHARS", 2600), 300, 12000)
CHUNK_OVERLAP_CHARS = env_int("CHUNK_OVERLAP_CHARS", 300)

if CHUNK_OVERLAP_CHARS < 0:
    CHUNK_OVERLAP_CHARS = 0

if CHUNK_OVERLAP_CHARS > CHUNK_CHARS // 2:
    CHUNK_OVERLAP_CHARS = CHUNK_CHARS // 2

# Token caps are role-specific so graph.py does not need hidden hardcoded budgets.
# Backward compatible: MAX_OUTPUT_TOKENS remains the default response cap.
MAX_OUTPUT_TOKENS = clamp_int(env_int("MAX_OUTPUT_TOKENS", 750), 64, 8000)
MAX_RESPONSE_TOKENS = clamp_int(
    env_int("MAX_RESPONSE_TOKENS", MAX_OUTPUT_TOKENS),
    64,
    8000
)
MAX_PLANNER_TOKENS = clamp_int(env_int("MAX_PLANNER_TOKENS", 1600), 256, 8000)
MAX_SUBAGENT_TOKENS = clamp_int(env_int("MAX_SUBAGENT_TOKENS", 1200), 256, 8000)
# Private raw subagent reasoning cap. This is separate from MAX_SUBAGENT_TOKENS
# because the raw JSON/analysis path should stay compact by default.
MAX_SUBAGENT_REASONING_TOKENS = clamp_int(
    env_int("MAX_SUBAGENT_REASONING_TOKENS", 500),
    128,
    2000
)
MAX_EXTRACTION_TOKENS = clamp_int(env_int("MAX_EXTRACTION_TOKENS", 700), 128, 4000)
MAX_MEMORY_TOKENS = clamp_int(env_int("MAX_MEMORY_TOKENS", 800), 128, 4000)
MAX_QUALITY_TOKENS = clamp_int(env_int("MAX_QUALITY_TOKENS", 600), 128, 4000)

# Global feature controls. Assistant-specific behavior should still live in
# domain_bundle.json; these values are deployment-wide kill switches/bounds.
QUALITY_GUARD_ENABLED = env_bool("QUALITY_GUARD_ENABLED", default=True)
SEMANTIC_EXTRACTION_GLOBAL_ENABLED = env_bool(
    "SEMANTIC_EXTRACTION_GLOBAL_ENABLED",
    default=True
)
SEMANTIC_EXTRACTION_MIN_CONFIDENCE = clamp_float(
    env_float("SEMANTIC_EXTRACTION_MIN_CONFIDENCE", 0.72),
    0.0,
    1.0
)
SEMANTIC_EXTRACTION_MAX_FIELDS = clamp_int(
    env_int("SEMANTIC_EXTRACTION_MAX_FIELDS", 8),
    1,
    50
)
SEMANTIC_EXTRACTION_MAX_WORKERS = clamp_int(
    env_int("SEMANTIC_EXTRACTION_MAX_WORKERS", 4),
    1,
    16
)
SEMANTIC_EXTRACTION_TIMEOUT_SECONDS = clamp_float(
    env_float("SEMANTIC_EXTRACTION_TIMEOUT_SECONDS", 20.0),
    1.0,
    120.0
)

# Response/history controls used by graph nodes.
SIMPLE_RESPONSE_HISTORY_LIMIT = clamp_int(
    env_int("SIMPLE_RESPONSE_HISTORY_LIMIT", 8),
    1,
    50
)
MANIFEST_HISTORY_LIMIT = clamp_int(
    env_int("MANIFEST_HISTORY_LIMIT", RECENT_MESSAGES_LIMIT),
    1,
    100
)

# Estimation helper for compression/token budgeting.
ESTIMATE_CHARS_PER_TOKEN = clamp_int(env_int("ESTIMATE_CHARS_PER_TOKEN", 4), 1, 12)

# Deployment-wide safety bounds for code-expert optimization features.
# Assistant/domain-specific behavior and phrases still belong in domain_bundle.json.
RESPONSE_MODEL_ROUTING_GLOBAL_ENABLED = env_bool(
    "RESPONSE_MODEL_ROUTING_GLOBAL_ENABLED",
    default=True
)


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

    if not POSTGRES_HOST:
        missing.append("POSTGRES_HOST")

    if not POSTGRES_DB:
        missing.append("POSTGRES_DB")

    if not POSTGRES_USER:
        missing.append("POSTGRES_USER")

    try:
        parsed_port = int(POSTGRES_PORT)
        if parsed_port <= 0:
            missing.append("POSTGRES_PORT")
    except Exception:
        missing.append("POSTGRES_PORT")

    if VECTOR_SIZE <= 0:
        missing.append("VECTOR_SIZE")

    if CHUNK_CHARS <= 0:
        missing.append("CHUNK_CHARS")

    if CHUNK_OVERLAP_CHARS < 0:
        missing.append("CHUNK_OVERLAP_CHARS")

    if CHUNK_OVERLAP_CHARS >= CHUNK_CHARS:
        raise RuntimeError("CHUNK_OVERLAP_CHARS must be smaller than CHUNK_CHARS.")

    model_values = {
        "MODEL_PLANNER": MODEL_PLANNER,
        "MODEL_SUBAGENT": MODEL_SUBAGENT,
        "MODEL_RESPONSE": MODEL_RESPONSE,
        "MODEL_RESPONSE_SIMPLE": MODEL_RESPONSE_SIMPLE,
        "MODEL_EXTRACTION": MODEL_EXTRACTION,
        "MODEL_MEMORY": MODEL_MEMORY,
        "MODEL_QUALITY": MODEL_QUALITY,
        "EMBED_MODEL": EMBED_MODEL,
    }

    for name, value in model_values.items():
        if not value:
            missing.append(name)

    # EMBED_MODEL can remain an embedding model; all chat/LLM roles must be GPT mini only.
    validate_gpt_mini_models_only()

    token_values = {
        "MAX_OUTPUT_TOKENS": MAX_OUTPUT_TOKENS,
        "MAX_RESPONSE_TOKENS": MAX_RESPONSE_TOKENS,
        "MAX_PLANNER_TOKENS": MAX_PLANNER_TOKENS,
        "MAX_SUBAGENT_TOKENS": MAX_SUBAGENT_TOKENS,
        "MAX_SUBAGENT_REASONING_TOKENS": MAX_SUBAGENT_REASONING_TOKENS,
        "MAX_EXTRACTION_TOKENS": MAX_EXTRACTION_TOKENS,
        "MAX_MEMORY_TOKENS": MAX_MEMORY_TOKENS,
        "MAX_QUALITY_TOKENS": MAX_QUALITY_TOKENS,
    }

    for name, value in token_values.items():
        if value <= 0:
            missing.append(name)

    if not 0 <= SEMANTIC_EXTRACTION_MIN_CONFIDENCE <= 1:
        missing.append("SEMANTIC_EXTRACTION_MIN_CONFIDENCE")

    if is_production():
        if MOCK_MODE:
            raise RuntimeError("MOCK_MODE must be false in production.")

        if not POSTGRES_PASSWORD:
            missing.append("POSTGRES_PASSWORD")

        if not QUALITY_GUARD_ENABLED:
            raise RuntimeError("QUALITY_GUARD_ENABLED must be true in production.")

    if missing:
        raise RuntimeError(
            "Missing or invalid required runtime environment variables: "
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
        "api_title": API_TITLE,
        "api_service_name": API_SERVICE_NAME,
        "data_dir": DATA_DIR,
        "config_dir": CONFIG_DIR,
        "qdrant_url": QDRANT_URL,
        "embed_model": EMBED_MODEL,
        "vector_size": VECTOR_SIZE,
        "default_gpt_mini_model": DEFAULT_GPT_MINI_MODEL,
        "model_planner": MODEL_PLANNER,
        "model_subagent": MODEL_SUBAGENT,
        "model_response": MODEL_RESPONSE,
        "model_response_simple": MODEL_RESPONSE_SIMPLE,
        "model_extraction": MODEL_EXTRACTION,
        "model_memory": MODEL_MEMORY,
        "model_quality": MODEL_QUALITY,
        "quality_guard_enabled": QUALITY_GUARD_ENABLED,
        "semantic_extraction_global_enabled": SEMANTIC_EXTRACTION_GLOBAL_ENABLED,
        "timezone": APP_TIMEZONE,
        "knowledge_top_k": KNOWLEDGE_TOP_K,
        "memory_top_k": MEMORY_TOP_K,
        "rag_min_score": RAG_MIN_SCORE,
        "memory_min_score": MEMORY_MIN_SCORE,
        "knowledge_compress_max_chars": KNOWLEDGE_COMPRESS_MAX_CHARS,
        "chunk_chars": CHUNK_CHARS,
        "chunk_overlap_chars": CHUNK_OVERLAP_CHARS,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "max_response_tokens": MAX_RESPONSE_TOKENS,
        "max_planner_tokens": MAX_PLANNER_TOKENS,
        "max_subagent_tokens": MAX_SUBAGENT_TOKENS,
        "max_subagent_reasoning_tokens": MAX_SUBAGENT_REASONING_TOKENS,
        "max_extraction_tokens": MAX_EXTRACTION_TOKENS,
        "max_memory_tokens": MAX_MEMORY_TOKENS,
        "max_quality_tokens": MAX_QUALITY_TOKENS,
        "semantic_extraction_min_confidence": SEMANTIC_EXTRACTION_MIN_CONFIDENCE,
        "semantic_extraction_max_fields": SEMANTIC_EXTRACTION_MAX_FIELDS,
        "semantic_extraction_max_workers": SEMANTIC_EXTRACTION_MAX_WORKERS,
        "semantic_extraction_timeout_seconds": SEMANTIC_EXTRACTION_TIMEOUT_SECONDS,
        "simple_response_history_limit": SIMPLE_RESPONSE_HISTORY_LIMIT,
        "manifest_history_limit": MANIFEST_HISTORY_LIMIT,
        "estimate_chars_per_token": ESTIMATE_CHARS_PER_TOKEN,
        "response_model_routing_global_enabled": RESPONSE_MODEL_ROUTING_GLOBAL_ENABLED,
        "has_app_secret": bool(APP_SECRET),
        "has_openai_api_key": bool(OPENAI_API_KEY),
        "has_qdrant_api_key": bool(QDRANT_API_KEY),
        "has_postgres_password": bool(POSTGRES_PASSWORD)
    }
