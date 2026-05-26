# app/token_usage.py
# Lightweight per-message token usage estimator.
# Works for all assistants and all routes.
# This does NOT call GPT. It estimates tokens locally.

import json
from typing import Any, Dict


def estimate_text_tokens(text: str) -> int:
    """
    Rough token estimator.

    English average: ~4 chars/token.
    Arabic can vary, so this intentionally uses a slightly conservative estimate.

    This is not exact OpenAI billing usage.
    It is useful for comparing turns and detecting token-heavy paths.
    """
    text = text or ""

    if not text:
        return 0

    arabic_chars = sum(1 for ch in text if "\u0600" <= ch <= "\u06FF")
    total_chars = len(text)

    if total_chars == 0:
        return 0

    arabic_ratio = arabic_chars / total_chars

    if arabic_ratio > 0.3:
        return max(1, int(total_chars / 3.2))

    return max(1, int(total_chars / 4))


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
    """
    Returns a per-turn token estimate.

    For model_used='none', GPT tokens are zero.
    This is what we want for state/no_llm/fast_path turns.
    """
    model_used = model_used or "none"
    model_tier = model_tier or "unknown"
    answer_mode = answer_mode or "unknown"

    zero_token_tiers = {
        "state",
        "fluid_state",
        "fast_path",
        "no_llm",
    }

    if model_used == "none" or model_tier in zero_token_tiers:
        input_tokens = 0
        output_tokens = 0
        total_tokens = 0
        estimated_cost_usd = 0.0
    else:
        input_tokens = estimate_object_tokens(input_obj)
        output_tokens = estimate_text_tokens(output_text)
        total_tokens = input_tokens + output_tokens

        # Conservative placeholder for mini-class models.
        # Real cost depends on the exact model and current pricing.
        estimated_cost_usd = round(total_tokens * 0.00000025, 8)

    return {
        "model_used": model_used,
        "model_tier": model_tier,
        "answer_mode": answer_mode,
        "input_tokens_estimate": input_tokens,
        "output_tokens_estimate": output_tokens,
        "total_tokens_estimate": total_tokens,
        "estimated_cost_usd": estimated_cost_usd,
        "knowledge_source": knowledge_source,
        "rag_cache_hit": bool(rag_cache_hit),
        "is_estimate": True,
        "notes": notes or "Estimated locally. Exact OpenAI usage requires llm.py usage tracking.",
    }
