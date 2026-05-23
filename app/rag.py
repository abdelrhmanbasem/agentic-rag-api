import hashlib
import math
import re
import uuid

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
    KNOWLEDGE_COMPRESS_ENABLED,
    KNOWLEDGE_COMPRESS_MAX_CHARS,
)

qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
_openai_client = None


def get_openai_client():
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


def mock_embed(text):
    digest = hashlib.sha256(text.encode("utf-8")).digest()
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
    if MOCK_MODE:
        return mock_embed(text)

    client = get_openai_client()
    response = client.embeddings.create(
        model=EMBED_MODEL,
        input=text,
    )
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


def smart_split_text(text):
    text = text.replace("\r\n", "\n").strip()

    lines = [line.strip() for line in text.split("\n") if line.strip()]

    if len(lines) > 1:
        chunks = []
        buffer = []

        for line in lines:
            buffer.append(line)
            joined = " ".join(buffer)

            if len(joined) >= 350 or re.search(r"[.!؟]$", line):
                chunks.append(joined.strip())
                buffer = []

        if buffer:
            chunks.append(" ".join(buffer).strip())

        return chunks

    sentence_parts = re.split(r"(?<=[.!؟])\s+", text)
    sentence_parts = [s.strip() for s in sentence_parts if s.strip()]

    if len(sentence_parts) > 1:
        return sentence_parts

    return [text] if text else []


def chunk_text(text, chunk_size=700, overlap=100):
    smart_chunks = smart_split_text(text)

    final_chunks = []
    for chunk in smart_chunks:
        words = chunk.split()

        if len(words) <= chunk_size:
            final_chunks.append(chunk)
            continue

        i = 0
        while i < len(words):
            piece = " ".join(words[i:i + chunk_size])
            if piece.strip():
                final_chunks.append(piece)
            i += chunk_size - overlap

    return final_chunks


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
            FieldCondition(key="assistant_id", match=MatchValue(value=assistant_id)),
        ]
    )


def memory_filter(assistant_id, user_id):
    return Filter(
        must=[
            FieldCondition(key="assistant_id", match=MatchValue(value=assistant_id)),
            FieldCondition(key="user_id", match=MatchValue(value=user_id)),
        ]
    )


def delete_document_chunks(assistant_id, document_id):
    qdrant.delete(
        collection_name="knowledge",
        points_selector=FilterSelector(
            filter=document_filter(assistant_id, document_id)
        ),
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
        qdrant.upsert(
            collection_name="knowledge",
            points=points,
            wait=True,
        )

    return len(chunks)


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


def search_knowledge(assistant_id, query, limit=4):
    ensure_qdrant()

    results = query_collection(
        collection_name="knowledge",
        query_vector=embed(query),
        query_filter=assistant_filter(assistant_id),
        limit=limit,
    )

    payloads = []
    for r in results:
        score = result_score(r)
        if score >= RAG_MIN_SCORE:
            payload = dict(r.payload or {})
            payload["score"] = score
            payloads.append(payload)

    return payloads


def clean_words(text):
    return set(
        w.lower()
        for w in re.findall(r"[\w\u0600-\u06FF]+", text)
        if len(w) > 1
    )


def compress_single_knowledge_item(item, query):
    text = item.get("text", "")
    query_words = clean_words(query)

    if not KNOWLEDGE_COMPRESS_ENABLED:
        return item

    if len(text) <= KNOWLEDGE_COMPRESS_MAX_CHARS:
        return item

    sentences = re.split(r"(?<=[.!؟])\s+", text)
    scored = []

    for sentence in sentences:
        words = clean_words(sentence)
        overlap = len(words.intersection(query_words))
        scored.append((overlap, sentence))

    scored.sort(key=lambda x: x[0], reverse=True)

    selected = []
    total = 0

    for score, sentence in scored:
        if score <= 0 and selected:
            continue

        if total + len(sentence) > KNOWLEDGE_COMPRESS_MAX_CHARS:
            continue

        selected.append(sentence)
        total += len(sentence)

        if total >= KNOWLEDGE_COMPRESS_MAX_CHARS:
            break

    compressed_text = " ".join(selected).strip()

    if not compressed_text:
        compressed_text = text[:KNOWLEDGE_COMPRESS_MAX_CHARS]

    compressed = dict(item)
    compressed["original_text_chars"] = len(text)
    compressed["text"] = compressed_text
    compressed["compressed"] = True

    return compressed


def compress_knowledge(knowledge_items, query):
    return [
        compress_single_knowledge_item(item, query)
        for item in knowledge_items
    ]


def write_memory(
    assistant_id,
    user_id,
    conversation_id,
    text,
    memory_type="preference",
    importance=0.5,
    confidence=0.5,
):
    ensure_qdrant()

    if not text.strip():
        return

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


def search_memories(assistant_id, user_id, query, limit=5):
    ensure_qdrant()

    results = query_collection(
        collection_name="memories",
        query_vector=embed(query),
        query_filter=memory_filter(assistant_id, user_id),
        limit=limit,
    )

    payloads = []
    for r in results:
        score = result_score(r)
        if score >= MEMORY_MIN_SCORE:
            payload = dict(r.payload or {})
            payload["score"] = score
            payloads.append(payload)

    return payloads
