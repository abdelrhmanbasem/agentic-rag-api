import hashlib
import math
import os
import re
import time
import uuid
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app.config import (
    MOCK_MODE,
    OPENAI_API_KEY,
    QDRANT_URL,
    QDRANT_API_KEY,
    EMBED_MODEL,
    VECTOR_SIZE,
    RAG_MIN_SCORE,
    MEMORY_MIN_SCORE,
    KNOWLEDGE_COMPRESS_MAX_CHARS,
    CHUNK_CHARS,
    CHUNK_OVERLAP_CHARS,
)


qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None)
_openai_client = None

_EMBED_CACHE: "OrderedDict[str, List[float]]" = OrderedDict()
_COLLECTIONS_READY = False


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)

    if value is None:
        return default

    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


EMBED_MAX_CHARS = env_int("EMBED_MAX_CHARS", 12000)
EMBED_RETRY_ATTEMPTS = max(env_int("EMBED_RETRY_ATTEMPTS", 3), 1)
EMBED_RETRY_SLEEP_SECONDS = max(env_float("EMBED_RETRY_SLEEP_SECONDS", 0.4), 0.0)
EMBED_CACHE_SIZE = max(env_int("EMBED_CACHE_SIZE", 512), 0)

RAG_SEARCH_MULTIPLIER = max(env_int("RAG_SEARCH_MULTIPLIER", 4), 1)
RAG_MIN_VECTOR_CANDIDATES = max(env_int("RAG_MIN_VECTOR_CANDIDATES", 12), 1)
RAG_LEXICAL_WEIGHT = max(env_float("RAG_LEXICAL_WEIGHT", 0.18), 0.0)
RAG_TITLE_WEIGHT = max(env_float("RAG_TITLE_WEIGHT", 0.05), 0.0)
RAG_RECENCY_WEIGHT = max(env_float("RAG_RECENCY_WEIGHT", 0.0), 0.0)

MEMORY_SEARCH_MULTIPLIER = max(env_int("MEMORY_SEARCH_MULTIPLIER", 3), 1)
MEMORY_IMPORTANCE_WEIGHT = max(env_float("MEMORY_IMPORTANCE_WEIGHT", 0.15), 0.0)
MEMORY_CONFIDENCE_WEIGHT = max(env_float("MEMORY_CONFIDENCE_WEIGHT", 0.15), 0.0)
MEMORY_MAX_TEXT_CHARS = max(env_int("MEMORY_MAX_TEXT_CHARS", 2000), 100)

QDRANT_WAIT = env_bool("QDRANT_WAIT", True)
RAG_FAIL_CLOSED = env_bool("RAG_FAIL_CLOSED", True)


def get_openai_client():
    global _openai_client

    if _openai_client is None:
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)

    return _openai_client


def stable_hash(value: Any) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def truncate_for_embedding(text: str) -> str:
    value = str(text or "").strip()

    if len(value) <= EMBED_MAX_CHARS:
        return value

    head = EMBED_MAX_CHARS // 2
    tail = EMBED_MAX_CHARS - head
    return value[:head].rstrip() + "\n...[embedding middle trimmed]...\n" + value[-tail:].lstrip()


def embed_cache_key(text: str) -> str:
    return stable_hash(f"{EMBED_MODEL}|{VECTOR_SIZE}|{text}")


def get_cached_embedding(text: str) -> Optional[List[float]]:
    if EMBED_CACHE_SIZE <= 0:
        return None

    key = embed_cache_key(text)

    if key not in _EMBED_CACHE:
        return None

    value = _EMBED_CACHE.pop(key)
    _EMBED_CACHE[key] = value
    return list(value)


def set_cached_embedding(text: str, vector: List[float]) -> None:
    if EMBED_CACHE_SIZE <= 0:
        return

    key = embed_cache_key(text)
    _EMBED_CACHE[key] = list(vector)

    while len(_EMBED_CACHE) > EMBED_CACHE_SIZE:
        _EMBED_CACHE.popitem(last=False)


def normalize_vector(vector: List[float]) -> List[float]:
    if not isinstance(vector, list):
        return mock_embed("")

    values = []

    for value in vector:
        try:
            values.append(float(value))
        except Exception:
            values.append(0.0)

    if len(values) != VECTOR_SIZE:
        if len(values) > VECTOR_SIZE:
            values = values[:VECTOR_SIZE]
        else:
            values.extend([0.0] * (VECTOR_SIZE - len(values)))

    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    return [v / norm for v in values]


def mock_embed(text):
    digest = hashlib.sha256((text or "").encode("utf-8")).digest()
    values = []

    while len(values) < VECTOR_SIZE:
        for byte in digest:
            values.append((byte / 255.0) - 0.5)

            if len(values) >= VECTOR_SIZE:
                break

        digest = hashlib.sha256(digest).digest()

    return normalize_vector(values)


def embed(text):
    text = truncate_for_embedding(text or "")

    cached = get_cached_embedding(text)
    if cached is not None:
        return cached

    if MOCK_MODE:
        vector = mock_embed(text)
        set_cached_embedding(text, vector)
        return vector

    last_error: Optional[Exception] = None

    for attempt in range(1, EMBED_RETRY_ATTEMPTS + 1):
        try:
            client = get_openai_client()
            response = client.embeddings.create(model=EMBED_MODEL, input=text)
            vector = normalize_vector(response.data[0].embedding)
            set_cached_embedding(text, vector)
            return vector
        except Exception as exc:
            last_error = exc

            if attempt >= EMBED_RETRY_ATTEMPTS:
                break

            if EMBED_RETRY_SLEEP_SECONDS > 0:
                time.sleep(EMBED_RETRY_SLEEP_SECONDS * attempt)

    if RAG_FAIL_CLOSED:
        raise RuntimeError(f"Embedding failed: {type(last_error).__name__}: {last_error}")

    vector = mock_embed(text)
    set_cached_embedding(text, vector)
    return vector


def ensure_qdrant():
    global _COLLECTIONS_READY

    if _COLLECTIONS_READY:
        return

    existing = {c.name for c in qdrant.get_collections().collections}

    if "knowledge" not in existing:
        qdrant.create_collection(
            collection_name="knowledge",
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )

    if "memories" not in existing:
        qdrant.create_collection(
            collection_name="memories",
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )

    _COLLECTIONS_READY = True


def safe_ensure_qdrant() -> bool:
    try:
        ensure_qdrant()
        return True
    except Exception:
        return False


def normalize_for_lexical(text: str) -> str:
    value = str(text or "").lower()
    value = re.sub(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]", "", value)
    value = value.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    value = value.replace("ة", "ه").replace("ى", "ي")
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def clean_words(text):
    normalized = normalize_for_lexical(text)
    return set(
        w.lower()
        for w in re.findall(r"[\w\u0600-\u06FF]+", normalized or "")
        if len(w) > 1
    )


def normalize_document_text(text: str) -> str:
    """
    Normalize text before chunking.

    Keep paragraph boundaries, but remove noisy repeated whitespace.
    This works better for Arabic and mixed Arabic/English documents than
    word-count chunking because character count gives more predictable token size.
    """
    value = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()

    if not value:
        return ""

    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)

    return value.strip()


def split_long_text_by_chars(text: str, max_chars: int, overlap_chars: int) -> List[str]:
    """
    Split a long text block by characters with overlap.

    Prefer ending chunks at natural boundaries near the target size:
    paragraph break, newline, sentence punctuation, then space.
    """
    text = str(text or "").strip()

    if not text:
        return []

    if len(text) <= max_chars:
        return [text]

    max_chars = max(300, int(max_chars or 2600))
    overlap_chars = max(0, min(int(overlap_chars or 300), max_chars // 2))

    chunks: List[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        hard_end = min(start + max_chars, text_len)

        if hard_end >= text_len:
            piece = text[start:text_len].strip()
            if piece:
                chunks.append(piece)
            break

        window = text[start:hard_end]
        split_at = find_best_split_index(window)

        if split_at <= 0:
            split_at = len(window)

        end = start + split_at
        piece = text[start:end].strip()

        if piece:
            chunks.append(piece)

        next_start = max(end - overlap_chars, start + 1)

        if next_start <= start:
            next_start = end

        start = next_start

    return chunks


def find_best_split_index(window: str) -> int:
    """
    Find a natural split point inside a chunk window.
    Returns an index relative to window.
    """
    if not window:
        return 0

    min_index = max(1, int(len(window) * 0.55))

    boundary_patterns = [
        "\n\n",
        "\n",
        "۔ ",
        "؟ ",
        "! ",
        ". ",
        "؛ ",
        "، ",
        ", ",
        " ",
    ]

    for boundary in boundary_patterns:
        index = window.rfind(boundary, min_index)

        if index != -1:
            return index + len(boundary)

    return len(window)


def chunk_text(text, chunk_size=None, overlap=None):
    """
    Character-based RAG chunking.

    Defaults:
    - CHUNK_CHARS from config/env, default intended around 2600 chars
    - CHUNK_OVERLAP_CHARS from config/env, default intended around 300 chars
    """
    normalized = normalize_document_text(text)

    if not normalized:
        return []

    max_chars = int(chunk_size or CHUNK_CHARS or 2600)
    overlap_chars = int(overlap or CHUNK_OVERLAP_CHARS or 300)

    max_chars = max(300, max_chars)
    overlap_chars = max(0, min(overlap_chars, max_chars // 2))

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", normalized) if p.strip()]

    if not paragraphs:
        return split_long_text_by_chars(normalized, max_chars, overlap_chars)

    chunks: List[str] = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if current.strip():
                chunks.append(current.strip())
                current = ""

            chunks.extend(
                split_long_text_by_chars(
                    text=paragraph,
                    max_chars=max_chars,
                    overlap_chars=overlap_chars
                )
            )
            continue

        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph

        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current.strip():
            chunks.append(current.strip())

        current = paragraph

    if current.strip():
        chunks.append(current.strip())

    if overlap_chars > 0 and len(chunks) > 1:
        chunks = add_chunk_overlap(chunks, overlap_chars, max_chars)

    return [chunk for chunk in chunks if chunk.strip()]


def add_chunk_overlap(chunks: List[str], overlap_chars: int, max_chars: int) -> List[str]:
    """
    Add previous-tail overlap to each chunk after the first, while keeping chunks bounded.
    """
    if not chunks:
        return []

    output = [chunks[0]]

    for index in range(1, len(chunks)):
        previous_tail = chunks[index - 1][-overlap_chars:].strip()
        current = chunks[index].strip()

        if previous_tail:
            combined = f"{previous_tail}\n\n{current}".strip()
        else:
            combined = current

        if len(combined) > max_chars + overlap_chars:
            combined = combined[-(max_chars + overlap_chars):].strip()

        output.append(combined)

    return output


def document_filter(assistant_id, document_id):
    return Filter(
        must=[
            FieldCondition(key="assistant_id", match=MatchValue(value=assistant_id)),
            FieldCondition(key="document_id", match=MatchValue(value=document_id)),
        ]
    )


def assistant_filter(assistant_id):
    return Filter(
        must=[
            FieldCondition(key="assistant_id", match=MatchValue(value=assistant_id))
        ]
    )


def memory_filter(assistant_id, user_id):
    return Filter(
        must=[
            FieldCondition(key="assistant_id", match=MatchValue(value=assistant_id)),
            FieldCondition(key="user_id", match=MatchValue(value=user_id)),
        ]
    )


def query_collection(collection_name, query_vector, query_filter, limit):
    if hasattr(qdrant, "query_points"):
        response = qdrant.query_points(
            collection_name=collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
        )
        return response.points

    return qdrant.search(
        collection_name=collection_name,
        query_vector=query_vector,
        query_filter=query_filter,
        limit=limit,
    )


def result_score(result):
    return float(getattr(result, "score", 0.0) or 0.0)


def point_payload(result) -> Dict[str, Any]:
    payload = getattr(result, "payload", {}) or {}
    return dict(payload) if isinstance(payload, dict) else {}


def delete_document_chunks(assistant_id, document_id):
    ensure_qdrant()

    qdrant.delete(
        collection_name="knowledge",
        points_selector=FilterSelector(filter=document_filter(assistant_id, document_id)),
        wait=QDRANT_WAIT,
    )


def deterministic_point_id(assistant_id: str, collection: str, *parts: Any) -> str:
    material = "|".join([str(assistant_id), collection] + [str(part) for part in parts])
    return str(uuid.uuid5(uuid.NAMESPACE_URL, material))


def safe_metadata(metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(metadata, dict):
        return {}

    output: Dict[str, Any] = {}

    for key, value in metadata.items():
        key_text = str(key or "").strip()

        if not key_text:
            continue

        if isinstance(value, (str, int, float, bool)) or value is None:
            output[key_text] = value
        elif isinstance(value, list):
            output[key_text] = [
                item for item in value
                if isinstance(item, (str, int, float, bool)) or item is None
            ]
        elif isinstance(value, dict):
            output[key_text] = {
                str(k): v
                for k, v in value.items()
                if isinstance(v, (str, int, float, bool)) or v is None
            }
        else:
            output[key_text] = str(value)

    return output


def ingest_document(assistant_id, document_id, title, text, metadata=None):
    ensure_qdrant()

    if not assistant_id or not document_id:
        raise ValueError("assistant_id and document_id are required")

    metadata = safe_metadata(metadata or {})
    normalized_text = normalize_document_text(text)

    if not normalized_text:
        delete_document_chunks(assistant_id, document_id)
        return 0

    chunks = chunk_text(normalized_text)

    delete_document_chunks(assistant_id, document_id)

    points = []

    for index, chunk in enumerate(chunks):
        chunk_hash = stable_hash(chunk)
        points.append(
            PointStruct(
                id=deterministic_point_id(assistant_id, "knowledge", document_id, index, chunk_hash[:16]),
                vector=embed(chunk),
                payload={
                    "assistant_id": assistant_id,
                    "document_id": document_id,
                    "title": title,
                    "chunk_index": index,
                    "text": chunk,
                    "text_hash": chunk_hash,
                    "metadata": metadata,
                    "chunk_chars": len(chunk),
                    "chunking_strategy": "character",
                    "ingested_at_ms": int(time.time() * 1000),
                },
            )
        )

    if points:
        qdrant.upsert(collection_name="knowledge", points=points, wait=QDRANT_WAIT)

    return len(chunks)


def lexical_overlap_score(item: Dict[str, Any], query: str) -> float:
    query_words = clean_words(query)
    item_words = clean_words((item.get("title") or "") + " " + (item.get("text") or ""))

    if not query_words or not item_words:
        return 0.0

    overlap = len(query_words.intersection(item_words))
    return overlap / max(1, len(query_words))


def title_overlap_score(item: Dict[str, Any], query: str) -> float:
    query_words = clean_words(query)
    title_words = clean_words(item.get("title") or "")

    if not query_words or not title_words:
        return 0.0

    return len(query_words.intersection(title_words)) / max(1, len(query_words))


def recency_score(item: Dict[str, Any]) -> float:
    metadata = item.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    timestamp = (
        item.get("updated_at_ms")
        or item.get("created_at_ms")
        or item.get("ingested_at_ms")
        or metadata.get("updated_at_ms")
        or metadata.get("created_at_ms")
        or metadata.get("ingested_at_ms")
    )

    try:
        timestamp = float(timestamp)
    except Exception:
        return 0.0

    if timestamp <= 0:
        return 0.0

    age_days = max((time.time() * 1000 - timestamp) / (1000 * 60 * 60 * 24), 0)
    return 1.0 / (1.0 + age_days / 30.0)


def rerank_knowledge_payload(payload: Dict[str, Any], query: str) -> Dict[str, Any]:
    vector_score = float(payload.get("score", 0.0) or 0.0)
    lexical = lexical_overlap_score(payload, query)
    title = title_overlap_score(payload, query)
    recent = recency_score(payload)

    payload["lexical_overlap"] = lexical
    payload["title_overlap"] = title
    payload["recency_score"] = recent
    payload["combined_score"] = (
        vector_score
        + RAG_LEXICAL_WEIGHT * lexical
        + RAG_TITLE_WEIGHT * title
        + RAG_RECENCY_WEIGHT * recent
    )

    return payload


def dedupe_payload_key(payload: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        payload.get("assistant_id"),
        payload.get("document_id"),
        payload.get("chunk_index"),
        payload.get("text_hash") or stable_hash(payload.get("text", "")),
    )


def normalize_queries(query: Union[str, List[str]]) -> List[str]:
    if isinstance(query, list):
        raw = query
    else:
        raw = [query]

    output: List[str] = []
    seen = set()

    for item in raw:
        text = str(item or "").strip()

        if not text:
            continue

        key = normalize_for_lexical(text)
        if key in seen:
            continue

        seen.add(key)
        output.append(text)

    return output


def search_knowledge(assistant_id, query, limit=6):
    if not assistant_id or not query:
        return []

    if not safe_ensure_qdrant():
        return []

    queries = normalize_queries(query)

    if not queries:
        return []

    fetch_limit = max(int(limit or 6) * RAG_SEARCH_MULTIPLIER, RAG_MIN_VECTOR_CANDIDATES)
    payloads: List[Dict[str, Any]] = []
    seen = set()

    for single_query in queries:
        try:
            query_vector = embed(single_query)
            vector_results = query_collection(
                collection_name="knowledge",
                query_vector=query_vector,
                query_filter=assistant_filter(assistant_id),
                limit=fetch_limit,
            )
        except Exception:
            continue

        for r in vector_results:
            score = result_score(r)

            if score < RAG_MIN_SCORE:
                continue

            payload = point_payload(r)
            payload["score"] = score
            payload["query"] = single_query
            payload = rerank_knowledge_payload(payload, single_query)

            dedupe_key = dedupe_payload_key(payload)

            if dedupe_key in seen:
                continue

            seen.add(dedupe_key)
            payloads.append(payload)

    payloads.sort(
        key=lambda item: (
            float(item.get("combined_score", item.get("score", 0.0)) or 0.0),
            float(item.get("score", 0.0) or 0.0),
            float(item.get("lexical_overlap", 0.0) or 0.0),
        ),
        reverse=True,
    )

    return payloads[:int(limit or 6)]


def split_sentences(text: str) -> List[str]:
    value = str(text or "").strip()

    if not value:
        return []

    parts = re.split(r"(?<=[.!؟?؛])\s+|\n+", value)
    return [part.strip() for part in parts if part.strip()]


def compress_single_knowledge_item(item, query):
    if not isinstance(item, dict):
        return {}

    text = item.get("text", "")

    if len(text) <= KNOWLEDGE_COMPRESS_MAX_CHARS:
        return item

    query_words = clean_words(query)
    raw_sentences = split_sentences(text)

    scored = []

    for index, sentence in enumerate(raw_sentences):
        sentence = sentence.strip()

        if not sentence:
            continue

        words = clean_words(sentence)
        overlap = len(words.intersection(query_words))

        scored.append({
            "index": index,
            "score": overlap,
            "sentence": sentence,
        })

    scored.sort(key=lambda item_: (item_["score"], -item_["index"]), reverse=True)

    selected = []
    total = 0

    for item_ in scored:
        sentence = item_["sentence"]
        score = item_["score"]

        if score <= 0 and selected:
            continue

        if total + len(sentence) > KNOWLEDGE_COMPRESS_MAX_CHARS:
            continue

        selected.append(item_)
        total += len(sentence)

        if total >= KNOWLEDGE_COMPRESS_MAX_CHARS:
            break

    selected.sort(key=lambda item_: item_["index"])

    compressed_text = " ".join(item_["sentence"] for item_ in selected).strip()

    if not compressed_text:
        compressed_text = text[:KNOWLEDGE_COMPRESS_MAX_CHARS]

    compressed = dict(item)
    compressed["text"] = compressed_text
    compressed["compressed"] = True
    compressed["original_text_chars"] = len(text)

    return compressed


def compress_knowledge(knowledge_items, query):
    if not isinstance(knowledge_items, list):
        return []

    query_text = " ".join(normalize_queries(query)) if isinstance(query, list) else str(query or "")

    compressed_items: List[Dict[str, Any]] = []
    seen = set()

    for item in knowledge_items:
        compressed = compress_single_knowledge_item(item, query_text)

        if not compressed:
            continue

        key = dedupe_payload_key(compressed)
        if key in seen:
            continue

        seen.add(key)
        compressed_items.append(compressed)

    return compressed_items


def write_memory(
    assistant_id,
    user_id,
    conversation_id,
    text,
    memory_type="preference",
    importance=0.5,
    confidence=0.5
):
    if not assistant_id or not user_id:
        return

    text = str(text or "").strip()

    if not text:
        return

    if not safe_ensure_qdrant():
        return

    memory_type = str(memory_type or "memory").strip() or "memory"
    text_to_store = text[:MEMORY_MAX_TEXT_CHARS]

    try:
        qdrant.upsert(
            collection_name="memories",
            points=[
                PointStruct(
                    id=deterministic_point_id(
                        assistant_id,
                        "memories",
                        user_id,
                        conversation_id,
                        memory_type,
                        stable_hash(text_to_store)[:16],
                    ),
                    vector=embed(text_to_store),
                    payload={
                        "assistant_id": assistant_id,
                        "user_id": user_id,
                        "conversation_id": conversation_id,
                        "text": text_to_store,
                        "text_hash": stable_hash(text_to_store),
                        "type": memory_type,
                        "importance": float(importance or 0.5),
                        "confidence": float(confidence or 0.5),
                        "created_at_ms": int(time.time() * 1000),
                    },
                )
            ],
            wait=QDRANT_WAIT,
        )
    except Exception:
        return


def rerank_memory_payload(payload: Dict[str, Any], query: str) -> Dict[str, Any]:
    vector_score = float(payload.get("score", 0.0) or 0.0)
    lexical = lexical_overlap_score(payload, query)
    importance = max(min(float(payload.get("importance", 0.5) or 0.5), 1.0), 0.0)
    confidence = max(min(float(payload.get("confidence", 0.5) or 0.5), 1.0), 0.0)

    payload["lexical_overlap"] = lexical
    payload["combined_score"] = (
        vector_score
        + RAG_LEXICAL_WEIGHT * lexical
        + MEMORY_IMPORTANCE_WEIGHT * importance
        + MEMORY_CONFIDENCE_WEIGHT * confidence
    )

    return payload


def search_memories(assistant_id, user_id, query, limit=5):
    if not assistant_id or not user_id or not query:
        return []

    if not safe_ensure_qdrant():
        return []

    queries = normalize_queries(query)

    if not queries:
        return []

    fetch_limit = max(int(limit or 5) * MEMORY_SEARCH_MULTIPLIER, int(limit or 5))
    payloads: List[Dict[str, Any]] = []
    seen = set()

    for single_query in queries:
        try:
            results = query_collection(
                collection_name="memories",
                query_vector=embed(single_query),
                query_filter=memory_filter(assistant_id, user_id),
                limit=fetch_limit,
            )
        except Exception:
            continue

        for r in results:
            score = result_score(r)

            if score < MEMORY_MIN_SCORE:
                continue

            payload = point_payload(r)
            payload["score"] = score
            payload["query"] = single_query
            payload = rerank_memory_payload(payload, single_query)

            dedupe_key = (
                payload.get("assistant_id"),
                payload.get("user_id"),
                payload.get("conversation_id"),
                payload.get("type"),
                payload.get("text_hash") or stable_hash(payload.get("text", "")),
            )

            if dedupe_key in seen:
                continue

            seen.add(dedupe_key)
            payloads.append(payload)

    payloads.sort(
        key=lambda item: (
            float(item.get("combined_score", item.get("score", 0.0)) or 0.0),
            float(item.get("importance", 0.5) or 0.5),
            float(item.get("confidence", 0.5) or 0.5),
            float(item.get("score", 0.0) or 0.0),
        ),
        reverse=True,
    )

    return payloads[:int(limit or 5)]
