import hashlib
import math
import re
import uuid
from typing import Dict, List, Any

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
)


qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None)
_openai_client = None


def get_openai_client():
    global _openai_client

    if _openai_client is None:
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)

    return _openai_client


def mock_embed(text):
    digest = hashlib.sha256((text or "").encode("utf-8")).digest()
    values = []

    while len(values) < VECTOR_SIZE:
        for byte in digest:
            values.append((byte / 255.0) - 0.5)

            if len(values) >= VECTOR_SIZE:
                break

        digest = hashlib.sha256(digest).digest()

    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    return [v / norm for v in values]


def embed(text):
    text = text or ""

    if MOCK_MODE:
        return mock_embed(text)

    client = get_openai_client()
    response = client.embeddings.create(model=EMBED_MODEL, input=text)
    return response.data[0].embedding


def ensure_qdrant():
    existing = [c.name for c in qdrant.get_collections().collections]

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


def safe_ensure_qdrant() -> bool:
    try:
        ensure_qdrant()
        return True
    except Exception:
        return False


def clean_words(text):
    return set(
        w.lower()
        for w in re.findall(r"[\w\u0600-\u06FF]+", text or "")
        if len(w) > 1
    )


def chunk_text(text, chunk_size=650, overlap=90):
    text = (text or "").replace("\r\n", "\n").strip()

    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks = []

    for paragraph in paragraphs:
        words = paragraph.split()

        if len(words) <= chunk_size:
            chunks.append(paragraph)
            continue

        i = 0
        step = max(1, chunk_size - overlap)

        while i < len(words):
            piece = " ".join(words[i:i + chunk_size]).strip()

            if piece:
                chunks.append(piece)

            i += step

    return chunks


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


def delete_document_chunks(assistant_id, document_id):
    ensure_qdrant()

    qdrant.delete(
        collection_name="knowledge",
        points_selector=FilterSelector(filter=document_filter(assistant_id, document_id)),
        wait=True,
    )


def ingest_document(assistant_id, document_id, title, text, metadata=None):
    ensure_qdrant()

    metadata = metadata or {}
    chunks = chunk_text(text)

    delete_document_chunks(assistant_id, document_id)

    points = []

    for index, chunk in enumerate(chunks):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embed(chunk),
                payload={
                    "assistant_id": assistant_id,
                    "document_id": document_id,
                    "title": title,
                    "chunk_index": index,
                    "text": chunk,
                    "metadata": metadata,
                },
            )
        )

    if points:
        qdrant.upsert(collection_name="knowledge", points=points, wait=True)

    return len(chunks)


def lexical_overlap_score(item: Dict[str, Any], query: str) -> float:
    query_words = clean_words(query)
    item_words = clean_words((item.get("title") or "") + " " + (item.get("text") or ""))

    if not query_words or not item_words:
        return 0.0

    overlap = len(query_words.intersection(item_words))
    return overlap / max(1, len(query_words))


def search_knowledge(assistant_id, query, limit=6):
    if not assistant_id or not query:
        return []

    if not safe_ensure_qdrant():
        return []

    try:
        query_vector = embed(query)
        vector_results = query_collection(
            collection_name="knowledge",
            query_vector=query_vector,
            query_filter=assistant_filter(assistant_id),
            limit=max(limit, 12),
        )
    except Exception:
        return []

    payloads = []
    seen = set()

    for r in vector_results:
        score = result_score(r)

        if score >= RAG_MIN_SCORE:
            payload = dict(r.payload or {})
            payload["score"] = score
            payload["lexical_overlap"] = lexical_overlap_score(payload, query)

            dedupe_key = (
                payload.get("assistant_id"),
                payload.get("document_id"),
                payload.get("chunk_index"),
                payload.get("text"),
            )

            if dedupe_key not in seen:
                seen.add(dedupe_key)
                payloads.append(payload)

    payloads.sort(
        key=lambda item: (
            float(item.get("score", 0.0)),
            float(item.get("lexical_overlap", 0.0)),
        ),
        reverse=True,
    )

    return payloads[:limit]


def compress_single_knowledge_item(item, query):
    if not isinstance(item, dict):
        return {}

    text = item.get("text", "")

    if len(text) <= KNOWLEDGE_COMPRESS_MAX_CHARS:
        return item

    query_words = clean_words(query)
    raw_sentences = re.split(r"(?<=[.!؟])\s+", text)

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

    scored.sort(key=lambda item_: item_["score"], reverse=True)

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

    # Critical architecture fix:
    # relevance chooses which sentences to keep, but final context keeps original document order.
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

    return [
        compressed
        for compressed in [
            compress_single_knowledge_item(item, query)
            for item in knowledge_items
        ]
        if compressed
    ]


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

    if not (text or "").strip():
        return

    if not safe_ensure_qdrant():
        return

    try:
        qdrant.upsert(
            collection_name="memories",
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embed(text),
                    payload={
                        "assistant_id": assistant_id,
                        "user_id": user_id,
                        "conversation_id": conversation_id,
                        "text": text,
                        "type": memory_type,
                        "importance": importance,
                        "confidence": confidence,
                    },
                )
            ],
            wait=True,
        )
    except Exception:
        return


def search_memories(assistant_id, user_id, query, limit=5):
    if not assistant_id or not user_id or not query:
        return []

    if not safe_ensure_qdrant():
        return []

    try:
        results = query_collection(
            collection_name="memories",
            query_vector=embed(query),
            query_filter=memory_filter(assistant_id, user_id),
            limit=limit,
        )
    except Exception:
        return []

    payloads = []

    for r in results:
        score = result_score(r)

        if score >= MEMORY_MIN_SCORE:
            payload = dict(r.payload or {})
            payload["score"] = score
            payloads.append(payload)

    payloads.sort(
        key=lambda item: (
            float(item.get("importance", 0.5)),
            float(item.get("confidence", 0.5)),
            float(item.get("score", 0.0)),
        ),
        reverse=True,
    )

    return payloads[:limit]
