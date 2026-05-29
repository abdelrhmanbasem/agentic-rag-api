# app/premium_retrieval.py
# Multi-query retrieval for adaptive premium mode.
#
# This keeps your current Qdrant + structured inventory setup,
# but retrieves broader evidence only on important turns.

from typing import Dict, Any, List
from app.rag import search_knowledge, search_memories
from app.structured_inventory import search_structured_inventory, inventory_items_to_knowledge


def compact_item(item: Dict[str, Any], max_chars: int = 900) -> Dict[str, Any]:
    item = dict(item or {})
    text = item.get("text") or ""

    if len(text) > max_chars:
        item["text"] = text[:max_chars].rstrip() + "..."

    return item


def dedupe_knowledge(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    output = []

    for item in items or []:
        key = (
            item.get("assistant_id"),
            item.get("document_id"),
            item.get("chunk_index"),
            item.get("text"),
        )

        if key in seen:
            continue

        seen.add(key)
        output.append(item)

    return output


def build_retrieval_queries(
    *,
    user_message: str,
    variables: Dict[str, Any],
    mode: str,
) -> List[str]:
    variables = variables or {}

    queries = [user_message]

    model = (
        variables.get("matched_car_model")
        or (variables.get("selected_item") or {}).get("model")
        if isinstance(variables.get("selected_item"), dict)
        else None
    )

    brand = variables.get("car_brand")
    budget = variables.get("budget_max")
    transmission = variables.get("transmission")
    condition = variables.get("car_condition")

    parts = []
    if brand:
        parts.append(str(brand))
    if model:
        parts.append(str(model))
    if condition:
        parts.append(str(condition))
    if transmission:
        parts.append(str(transmission))
    if budget:
        parts.append(f"under {budget}")

    if parts:
        queries.append(" ".join(parts))

    if model:
        queries.append(f"{model} price km year transmission availability")

    if brand or budget:
        queries.append(f"alternatives for {brand or 'car'} under {budget or 'user budget'}")

    if mode in ["premium_sales", "deep_advisor", "careful_strong"]:
        queries.append("warranty inspection financing installment negotiation policy")
        queries.append("available inventory price mileage condition")

    clean = []
    for q in queries:
        q = (q or "").strip()
        if q and q not in clean:
            clean.append(q)

    return clean[:6]


def retrieve_premium_evidence(
    *,
    assistant_id: str,
    user_id: str,
    user_message: str,
    variables: Dict[str, Any],
    mode: str,
    knowledge_limit_per_query: int = 8,
    memory_limit: int = 8,
) -> Dict[str, Any]:
    variables = variables or {}

    queries = build_retrieval_queries(
        user_message=user_message,
        variables=variables,
        mode=mode,
    )

    knowledge: List[Dict[str, Any]] = []

    structured_items = search_structured_inventory(
        assistant_id=assistant_id,
        query_variables=variables,
        limit=8,
        raw_query=user_message,
    )

    if structured_items:
        knowledge.extend(inventory_items_to_knowledge(structured_items))

    for query in queries:
        hits = search_knowledge(
            assistant_id=assistant_id,
            query=query,
            limit=knowledge_limit_per_query,
        )
        knowledge.extend(hits or [])

    knowledge = dedupe_knowledge(knowledge)
    knowledge = sorted(knowledge, key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True)
    knowledge = [compact_item(item) for item in knowledge[:20]]

    memories = search_memories(
        assistant_id=assistant_id,
        user_id=user_id,
        query=user_message,
        limit=memory_limit,
    )

    memories = [compact_item(item, max_chars=500) for item in memories or []]

    return {
        "queries": queries,
        "knowledge": knowledge,
        "memories": memories,
        "knowledge_source": "premium_multi_retrieval" if knowledge else "none",
    }
