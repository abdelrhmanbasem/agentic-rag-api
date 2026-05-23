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


def clean_words(text):
    return set(
        w.lower()
        for w in re.findall(r"[\w\u0600-\u06FF]+", text or "")
        if len(w) > 1
    )


def parse_budget_from_query(query_text):
    query_text = (query_text or "").lower()

    patterns = [
        r"under\s+(\d+(?:\.\d+)?)\s*(million|m|k|thousand)?",
        r"up to\s+(\d+(?:\.\d+)?)\s*(million|m|k|thousand)?",
        r"budget\s+(\d+(?:\.\d+)?)\s*(million|m|k|thousand)?",
        r"لحد\s+(\d+(?:\.\d+)?)\s*(مليون|الف|ألف)?",
        r"حدود\s+(\d+(?:\.\d+)?)\s*(مليون|الف|ألف)?",
    ]

    for pattern in patterns:
        match = re.search(pattern, query_text)
        if not match:
            continue

        number = float(match.group(1))
        unit = match.group(2)

        if unit in ["million", "m", "مليون"]:
            number *= 1000000
        elif unit in ["k", "thousand", "الف", "ألف"]:
            number *= 1000

        return int(number)

    return None


def extract_prices_from_text(text):
    text = (text or "").lower()
    prices = []

    for match in re.findall(r"(\d{5,})\s*egp", text):
        try:
            prices.append(int(match))
        except Exception:
            pass

    for match in re.findall(r"(\d{5,})\s*جنيه", text):
        try:
            prices.append(int(match))
        except Exception:
            pass

    return prices


def rerank_knowledge_items(items, query):
    query_text = (query or "").lower()
    query_words = clean_words(query_text)
    budget = parse_budget_from_query(query_text)

    requested_brands = []

    brand_aliases = {
        "bmw": ["bmw", "بي ام", "بي ام دبليو"],
        "mercedes": ["mercedes", "مرسيدس"],
        "hyundai": ["hyundai", "هيونداي"],
        "toyota": ["toyota", "تويوتا"],
        "kia": ["kia", "كيا"],
        "nissan": ["nissan", "نيسان"],
        "audi": ["audi", "اودي"],
    }

    for normalized_brand, aliases in brand_aliases.items():
        if any(alias in query_text for alias in aliases):
            requested_brands.append(normalized_brand)

    def item_score(item):
        text = (item.get("text") or "").lower()
        title = (item.get("title") or "").lower()
        combined = text + " " + title
        words = clean_words(combined)

        score = 0

        # Keep Qdrant score as a weak signal.
        score += float(item.get("score", 0.0) or 0.0) * 5

        # Keyword overlap.
        score += len(words.intersection(query_words)) * 10

        # Strong brand boosts.
        for brand, aliases in brand_aliases.items():
            brand_in_query = brand in requested_brands
            brand_in_item = any(alias in combined for alias in aliases)

            if brand_in_query and brand_in_item:
                score += 120

            if requested_brands and not brand_in_query and brand_in_item:
                score -= 60

        # Budget awareness.
        if budget:
            prices = extract_prices_from_text(combined)

            if prices:
                best_price = min(prices)

                if best_price <= budget:
                    score += 90
                else:
                    score -= 90

        # Helpful matching terms.
        if "automatic" in query_text and "automatic" in combined:
            score += 30

        if "manual" in query_text and "manual" in combined:
            score += 30

        if "اوتوماتيك" in query_text and "اوتوماتيك" in combined:
            score += 30

        if "مانيوال" in query_text and "مانيوال" in combined:
            score += 30

        return score

    return sorted(items, key=item_score, reverse=True)


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

    return rerank_knowledge_items(payloads, query)


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
