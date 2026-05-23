import hashlib
import math
import uuid

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
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


def chunk_text(text, chunk_size=700, overlap=100):
    words = text.split()
    chunks = []
    i = 0

    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        i += chunk_size - overlap

    return chunks


def delete_document_chunks(assistant_id, document_id):
    qdrant.delete(
        collection_name="knowledge",
        points_selector=Filter(
            must=[
                FieldCondition(key="assistant_id", match=MatchValue(value=assistant_id)),
                FieldCondition(key="document_id", match=MatchValue(value=document_id)),
            ]
        ),
    )


def ingest_document(assistant_id, document_id, title, text):
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
                },
            )
        )

    if points:
        qdrant.upsert(collection_name="knowledge", points=points)

    return len(chunks)


def search_knowledge(assistant_id, query, limit=4):
    try:
        results = qdrant.search(
            collection_name="knowledge",
            query_vector=embed(query),
            query_filter=Filter(
                must=[
                    FieldCondition(key="assistant_id", match=MatchValue(value=assistant_id))
                ]
            ),
            limit=limit,
        )
        return [r.payload for r in results]
    except Exception:
        return []


def write_memory(assistant_id, user_id, conversation_id, text, memory_type="preference", importance=0.5, confidence=0.5):
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
    )


def search_memories(assistant_id, user_id, query, limit=5):
    try:
        results = qdrant.search(
            collection_name="memories",
            query_vector=embed(query),
            query_filter=Filter(
                must=[
                    FieldCondition(key="assistant_id", match=MatchValue(value=assistant_id)),
                    FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                ]
            ),
            limit=limit,
        )
        return [r.payload for r in results]
    except Exception:
        return []
