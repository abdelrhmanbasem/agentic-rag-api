import json
from typing import Any, Dict


def estimate_text_tokens(text: str) -> int:
    text = text or ""
    if not text:
        return 0
    arabic_chars = sum(1 for ch in text if "\u0600" <= ch <= "\u06FF")
    total_chars = len(text)
    arabic_ratio = arabic_chars / total_chars if total_chars else 0
    return max(1, int(total_chars / 3.2)) if arabic_ratio > 0.3 else max(1, int(total_chars / 4))


def safe_json_dumps(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str)
    except Exception:
        return str(obj)


def estimate_object_tokens(obj: Any) -> int:
    return estimate_text_tokens(safe_json_dumps(obj))


def build_token_usage_report(
    *,
    model_used: str,
    model_tier: str,
    answer_mode: str,
    input_obj: Dict[str, Any],
    output_text: str,
    knowledge_source: str = "none",
    rag_cache_hit: bool = False,
    notes: str = "",
) -> Dict[str, Any]:
    input_tokens = estimate_object_tokens(input_obj)
    output_tokens = estimate_text_tokens(output_text)
    total_tokens = input_tokens + output_tokens
    return {
        "model_used": model_used or "unknown",
        "model_tier": model_tier or "unknown",
        "answer_mode": answer_mode or "unknown",
        "input_tokens_estimate": input_tokens,
        "output_tokens_estimate": output_tokens,
        "total_tokens_estimate": total_tokens,
        "estimated_cost_usd": round(total_tokens * 0.00000025, 8),
        "knowledge_source": knowledge_source,
        "rag_cache_hit": bool(rag_cache_hit),
        "is_estimate": True,
        "notes": notes or "Estimated locally. Exact usage is logged from LangChain callback in main.py.",
    }
