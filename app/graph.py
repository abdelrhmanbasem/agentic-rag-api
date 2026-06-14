# Architecture patch: 6.63-configured-legacy-tool-argument-fallback-and-gpt-mini-only-no-hardcoding
# Architecture patch: 6.70-persist-post-tool-required-input-continuation-no-hardcoding-graph
from typing import TypedDict, Annotated, Sequence, Dict, Any, List, Optional

# Architecture batch: 6.36-manifest-history-limit-no-hardcoding-graph
# Architecture patch: 6.39-previous-manifest-summary-no-hardcoding-graph
# Architecture patch: 6.42-breathtaking-smartness-runtime-no-hardcoding-graph
# Architecture patch: 6.44-semantic-detail-safety-no-hardcoding-graph
# Architecture patch: 6.45-code-expert-cost-smartness-no-hardcoding-graph
# Architecture patch: 6.46-root-regression-fixes-no-hardcoding-graph
# Architecture patch: 6.47-runtime-detail-safety-no-hardcoding-graph
# Architecture patch: 6.49-runtime-debug-state-and-closing-route-no-hardcoding-graph
# Architecture patch: 6.50-configured-completed-closing-direct-response-no-hardcoding-graph
# Architecture patch: 6.51-help-gated-pattern-extraction-and-closing-id-skip-no-hardcoding-graph
# Architecture patch: 6.52-configured-answer-draft-id-append-skip-no-hardcoding-graph
# Architecture patch: 6.53-enforce-answer-safety-id-skip-no-hardcoding-graph
# Architecture patch: 6.54-active-flow-stage-union-and-emotion-mirror-no-hardcoding-graph
# Architecture patch: 6.55-mirror-variable-changes-to-debug-state-no-hardcoding-graph
# Architecture patch: 6.57-configured-smart-clarification-fallback-no-hardcoding-graph
# Architecture patch: 6.58-explicit-smartness-policy-enabled-no-hardcoding-graph
# Architecture patch: 6.59-configured-booking-executor-lock-recovery-no-hardcoding-graph
# Architecture patch: 6.61-code-expert-completion-no-hardcoding-graph
# Architecture patch: 6.62-code-expert-final-gap-closure-no-hardcoding-graph
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import operator
import os
import re

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END

from app.config import (
    KNOWLEDGE_TOP_K,
    MEMORY_TOP_K,
    MODEL_PLANNER,
    MODEL_SUBAGENT,
    MODEL_RESPONSE,
    MODEL_QUALITY,
    MODEL_EXTRACTION,
    OPENAI_API_KEY,
    MAX_OUTPUT_TOKENS,
    MAX_RESPONSE_TOKENS,
    MAX_PLANNER_TOKENS,
    MAX_SUBAGENT_TOKENS,
    MAX_EXTRACTION_TOKENS,
    MAX_MEMORY_TOKENS,
    MAX_QUALITY_TOKENS,
    QUALITY_GUARD_ENABLED,
    SEMANTIC_EXTRACTION_GLOBAL_ENABLED,
    SEMANTIC_EXTRACTION_MIN_CONFIDENCE,
    SEMANTIC_EXTRACTION_MAX_FIELDS,
    SEMANTIC_EXTRACTION_MAX_WORKERS,
    SEMANTIC_EXTRACTION_TIMEOUT_SECONDS,
    SIMPLE_RESPONSE_HISTORY_LIMIT,
    MANIFEST_HISTORY_LIMIT,
)
from app.rag import search_knowledge, compress_knowledge, search_memories
from app.db import get_rag_cache, save_rag_cache
from app.tool_runner import ToolRunner
from app.subagents.base import (
    SubagentContext,
    apply_variable_patch as apply_subagent_variable_patch,
    apply_tool_update_rules,
    get_subagent_variable_scope,
    get_subagent_config,
    matches_any,
    deep_get,
    render_template,
)
from app.subagents.booking_subagent import BookingSubagent
from app.subagents.location_subagent import LocationSubagent
from app.subagents.lookup_subagent import LookupSubagent
from app.subagents.troubleshooting_subagent import TroubleshootingSubagent
from app.subagents.handoff_subagent import HandoffSubagent


class AgentState(TypedDict, total=False):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    assistant_id: str
    user_id: str
    conversation_id: str

    variables: Dict[str, Any]
    summary: str
    system_prompt: str
    agent_config: Dict[str, Any]
    language_instruction: str
    schema: Dict[str, Any]
    tool_result: Dict[str, Any]
    multi_tool_results: List[Dict[str, Any]]

    manifest: Dict[str, Any]
    previous_manifest_summary: str
    best_guess_clarification: Dict[str, Any]
    emotion_history: List[str]
    emotion_trajectory: str
    last_offered_options: Dict[str, Any]
    funnel_stage: str
    opener_context: str
    variable_changes_this_turn: Annotated[List[Dict[str, Any]], operator.add]
    stuck_signals: Dict[str, int]
    stuck_pattern: Dict[str, Any]
    proactive_surface_items: List[Dict[str, Any]]
    smart_inferences: List[str]
    memories_raw: List[Dict[str, Any]]
    failure_recovery_context: Dict[str, Any]
    progressive_display_context: Dict[str, Any]
    multi_intents: List[Dict[str, Any]]
    parallel_tool_requests: List[Dict[str, Any]]
    knowledge_queries: List[str]
    multi_knowledge: List[Dict[str, Any]]
    response_synthesis: Dict[str, Any]
    planner: Dict[str, Any]

    selected_subagent: Dict[str, Any]
    subagent_analysis: Dict[str, Any]

    knowledge: str
    knowledge_items: List[Dict[str, Any]]
    memories: str

    final_answer: str
    quality: Dict[str, Any]
    memory_writer: Dict[str, Any]


def llm(model: str, temperature: float = 0.0, max_tokens: Optional[int] = None):
    kwargs = {
        "model": model,
        "temperature": temperature,
        "api_key": OPENAI_API_KEY,
    }
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    return ChatOpenAI(**kwargs)


class SubagentAnalysis(BaseModel):
    understanding: str = ""
    user_goal: str = ""
    detected_intents: List[str] = Field(default_factory=list)
    facts_to_use: List[str] = Field(default_factory=list)
    facts_missing: List[str] = Field(default_factory=list)
    variable_updates_to_consider: Dict[str, Any] = Field(default_factory=dict)
    next_best_step: str = ""
    response_constraints: List[str] = Field(default_factory=list)
    should_chain: bool = False
    recommended_chained_subagent_id: str = ""
    confidence: float = 0.7


class QualityDecision(BaseModel):
    pass_check: bool
    revised_answer: str = ""
    issues: List[str] = Field(default_factory=list)
    energy_ok: bool = True
    energy_issue: str = ""


def graph_env_int(name: str, default: int, minimum: int = 1, maximum: int = 20000) -> int:
    """
    Read a generic integer runtime control from the environment.

    This keeps graph-level token controls configurable before config.py has been
    updated to export them, while avoiding tenant/domain behavior in Python.
    """
    try:
        value = int(os.getenv(name, str(default)))
    except Exception:
        value = default

    return max(minimum, min(maximum, value))


SUBAGENT_REASONING_MAX_TOKENS = graph_env_int(
    "MAX_SUBAGENT_REASONING_TOKENS",
    default=500,
    minimum=128,
    maximum=2000,
)


manifest_llm = llm(MODEL_PLANNER, temperature=0, max_tokens=MAX_PLANNER_TOKENS).bind(
    response_format={"type": "json_object"}
)
subagent_llm = llm(MODEL_SUBAGENT, temperature=0.15).with_structured_output(SubagentAnalysis)
subagent_llm_raw = llm(MODEL_SUBAGENT, temperature=0.15, max_tokens=SUBAGENT_REASONING_MAX_TOKENS)
response_llm = llm(MODEL_RESPONSE, temperature=0.3, max_tokens=MAX_RESPONSE_TOKENS)
quality_llm = llm(MODEL_QUALITY, temperature=0, max_tokens=MAX_QUALITY_TOKENS).with_structured_output(QualityDecision)
semantic_extraction_llm = llm(MODEL_EXTRACTION, temperature=0, max_tokens=MAX_EXTRACTION_TOKENS).bind(
    response_format={"type": "json_object"}
)


SUBAGENT_EXECUTORS = {
    "booking": BookingSubagent(),
    "booking_advisor": BookingSubagent(),
    "location": LocationSubagent(),
    "branch_locator": LocationSubagent(),
    "lookup": LookupSubagent(),
    "general_lookup": LookupSubagent(),
    "troubleshooting": TroubleshootingSubagent(),
    "diagnosis_advisor": TroubleshootingSubagent(),
    "handoff": HandoffSubagent(),
    "human_handoff": HandoffSubagent(),
}


# Source-of-truth protections are assistant-config driven.
# Configure these per assistant in domain_bundle.json:
# - source_of_truth_variables
# - source_of_truth_prefixes
# - source_of_truth_include_tool_result


def last_user_message(state: AgentState) -> str:
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            return msg.content
    if state.get("messages"):
        return state["messages"][-1].content
    return ""


def clip_text(value: Any, max_chars: int = 1200) -> str:
    text = "" if value is None else str(value)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n...[trimmed]"


def clip_text_head_tail(value: Any, max_chars: int = 900, head_ratio: float = 0.55) -> str:
    text = "" if value is None else str(value)
    if len(text) <= max_chars:
        return text

    head_chars = int(max_chars * head_ratio)
    tail_chars = max_chars - head_chars - 24

    return (
        text[:head_chars].rstrip()
        + "\n...[middle trimmed]...\n"
        + text[-tail_chars:].lstrip()
    )


def safe_json(value: Any, max_chars: Optional[int] = None) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False)
    except Exception:
        text = str(value)
    if max_chars:
        return clip_text(text, max_chars)
    return text


def parse_manifest_response(content: Any) -> Dict[str, Any]:
    text = content.content if hasattr(content, "content") else str(content)

    try:
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            raise ValueError("Manifest JSON root must be an object.")
        return parsed
    except Exception as exc:
        raise ValueError(
            f"Could not parse manifest JSON: {exc}. Raw content: {clip_text(text, 1000)}"
        )


def normalize_json_manifest(manifest: Dict[str, Any]) -> Dict[str, Any]:
    defaults = {
        "user_intent": "",
        "selected_subagent_id": "",
        "chained_subagent_id": "",
        "chained_subagent_reason": "",
        "detected_intents": [],
        "multi_intents": [],
        "parallel_tool_requests": [],
        "knowledge_queries": [],
        "response_synthesis": {},
        "multi_intent_execution_mode": "",
        "conversation_stage": "",
        "workflow_stage": "",
        "customer_emotion": "",
        "emotion_trajectory": "",
        "funnel_stage": "",
        "best_guess_clarification": {
            "hypothesis": "",
            "hypothesis_confidence": 0.0,
            "ask_confirm": False,
        },
        "user_expectation": "",
        "risk_level": "low",
        "confidence": 0.7,
        "simple_response_mode": False,
        "simple_response_reason": "",
        "needs_knowledge": False,
        "needs_memory": False,
        "needs_tool": False,
        "requested_tool_name": "",
        "tool_request_payload": {},
        "missing_tool_inputs": [],
        "needs_subagent_reasoning": False,
        "needs_quality_guard": False,
        "needs_style_repair": False,
        "needs_full_manifest": False,
        "extracted_updates": {},
        "extracted_deletions": [],
        "response_style": "",
        "reply_length": "",
        "should_ask_question": False,
        "question_goal": "",
        "should_offer_next_action": False,
        "response_brief": {
            "tone": "",
            "language": "",
            "reply_length": "",
            "must_do": [],
            "must_not_do": [],
            "next_move": "",
        },
        "response_strategy": "",
        "reasoning_summary": "",
        "manifest_profile_used": "",
    }

    out = dict(defaults)
    out.update(manifest or {})

    if not isinstance(out.get("tool_request_payload"), dict):
        out["tool_request_payload"] = {}

    if not isinstance(out.get("extracted_updates"), dict):
        out["extracted_updates"] = {}

    if not isinstance(out.get("missing_tool_inputs"), list):
        out["missing_tool_inputs"] = []

    if not isinstance(out.get("extracted_deletions"), list):
        out["extracted_deletions"] = []

    if not isinstance(out.get("detected_intents"), list):
        out["detected_intents"] = []

    if not isinstance(out.get("multi_intents"), list):
        out["multi_intents"] = []

    out["multi_intents"] = [
        item for item in out.get("multi_intents", [])
        if isinstance(item, dict)
    ]

    if not isinstance(out.get("parallel_tool_requests"), list):
        out["parallel_tool_requests"] = []

    out["parallel_tool_requests"] = [
        item for item in out.get("parallel_tool_requests", [])
        if isinstance(item, dict)
    ]

    if not isinstance(out.get("knowledge_queries"), list):
        out["knowledge_queries"] = []

    out["knowledge_queries"] = [
        str(item).strip()
        for item in out.get("knowledge_queries", [])
        if str(item or "").strip()
    ]

    if not isinstance(out.get("response_synthesis"), dict):
        out["response_synthesis"] = {}

    out["selected_subagent_id"] = unify_subagent_id(out.get("selected_subagent_id", ""))
    out["chained_subagent_id"] = unify_subagent_id(out.get("chained_subagent_id", ""))

    if out.get("chained_subagent_id") == out.get("selected_subagent_id"):
        out["chained_subagent_id"] = ""

    if not isinstance(out.get("response_brief"), dict):
        out["response_brief"] = defaults["response_brief"]

    brief = dict(defaults["response_brief"])
    brief.update(out["response_brief"])

    if not isinstance(brief.get("must_do"), list):
        brief["must_do"] = []

    if not isinstance(brief.get("must_not_do"), list):
        brief["must_not_do"] = []

    out["response_brief"] = brief

    if not isinstance(out.get("best_guess_clarification"), dict):
        out["best_guess_clarification"] = {
            "hypothesis": "",
            "hypothesis_confidence": 0.0,
            "ask_confirm": False,
        }
    else:
        clarification = dict(out.get("best_guess_clarification") or {})
        try:
            clarification["hypothesis_confidence"] = float(
                clarification.get("hypothesis_confidence", 0.0) or 0.0
            )
        except Exception:
            clarification["hypothesis_confidence"] = 0.0
        clarification["hypothesis"] = str(clarification.get("hypothesis") or "").strip()
        clarification["ask_confirm"] = bool(clarification.get("ask_confirm", False))
        out["best_guess_clarification"] = clarification

    try:
        out["confidence"] = float(out.get("confidence", 0.7) or 0.7)
    except Exception:
        out["confidence"] = 0.7

    if out.get("risk_level") not in ["low", "medium", "high"]:
        out["risk_level"] = "low"

    return out



def as_string_list(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    output: List[str] = []
    for item in value:
        text = str(item or "").strip()
        if text:
            output.append(text)
    return output


def get_config_bool(config: Dict[str, Any], path: str, default: bool = False) -> bool:
    value = get_config_path_value(config, path, default)
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def get_source_of_truth_variables(agent_config: Dict[str, Any]) -> set:
    """
    Multi-tenant source-of-truth variable protection.

    This intentionally reads from assistant config instead of hardcoded booking
    fields so a hotel/support/sales assistant can define its own protected
    operational state in domain_bundle.json.
    """
    values = set(as_string_list(agent_config.get("source_of_truth_variables", [])))

    schema_config = agent_config.get("schema") or agent_config.get("variables") or {}
    if isinstance(schema_config, dict):
        schema_fields = get_schema_fields(schema_config)
        for key, meta in schema_fields.items():
            if not isinstance(meta, dict):
                continue
            if meta.get("source_of_truth") is True or meta.get("operational_state") is True:
                key_text = str(key or "").strip()
                if key_text:
                    values.add(key_text)

    return values


def get_source_of_truth_prefixes(agent_config: Dict[str, Any]) -> tuple:
    """
    Multi-tenant prefix protection for operational namespaces.
    Configure per assistant with source_of_truth_prefixes.
    """
    return tuple(as_string_list(agent_config.get("source_of_truth_prefixes", [])))


def is_source_of_truth_path(path: str, agent_config: Dict[str, Any]) -> bool:
    path_text = str(path or "").strip()

    if not path_text:
        return False

    exact = get_source_of_truth_variables(agent_config)
    prefixes = get_source_of_truth_prefixes(agent_config)

    if path_text in exact:
        return True

    for prefix in prefixes:
        prefix_text = str(prefix or "").strip()
        if not prefix_text:
            continue
        if path_text == prefix_text or path_text.startswith(prefix_text + "."):
            return True

    return False


def filter_manifest_updates(
    updates: Dict[str, Any],
    agent_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    The manifest LLM may extract soft user info, but it must not directly write
    source-of-truth operational state. Tool/subagent execution owns those fields.

    Source-of-truth paths are multi-tenant and config-driven through
    source_of_truth_variables/source_of_truth_prefixes.
    """
    agent_config = agent_config or {}

    if not isinstance(updates, dict):
        return {}

    allowed: Dict[str, Any] = {}

    for key, value in updates.items():
        key_text = str(key or "").strip()

        if not key_text:
            continue

        if is_source_of_truth_path(key_text, agent_config):
            continue

        allowed[key_text] = value

    return allowed

def filter_manifest_deletions(
    deletions: List[str],
    agent_config: Optional[Dict[str, Any]] = None
) -> List[str]:
    """
    The manifest LLM may request deletion of soft state, but cannot delete
    source-of-truth operational state.

    Source-of-truth paths are multi-tenant and config-driven through
    source_of_truth_variables/source_of_truth_prefixes.
    """
    agent_config = agent_config or {}

    if not isinstance(deletions, list):
        return []

    allowed: List[str] = []

    for item in deletions:
        path = str(item or "").strip()

        if not path:
            continue

        if is_source_of_truth_path(path, agent_config):
            continue

        allowed.append(path)

    return allowed

def get_schema_fields(schema: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(schema, dict):
        return {}

    nested = schema.get("schema")
    if isinstance(nested, dict):
        return nested

    variables = schema.get("variables")
    if isinstance(variables, dict):
        return variables

    return schema


def schema_priority_rank(value: Dict[str, Any]) -> int:
    if not isinstance(value, dict):
        return 99

    priority = str(value.get("priority", "low")).lower().strip()

    if priority == "high":
        return 0
    if priority == "medium":
        return 1
    if priority == "low":
        return 2
    return 3


def summarize_schema_fields(
    schema: Dict[str, Any],
    max_fields: int = 50,
    profile: str = "short",
) -> Dict[str, Any]:
    fields = get_schema_fields(schema)
    if not fields:
        return {}

    profile = (profile or "short").lower().strip()

    if profile == "short":
        max_fields = min(max_fields, 32)
        desc_chars = 95
        allowed_keys = ["type", "description", "enum", "items", "required", "priority"]
    else:
        max_fields = min(max_fields, 70)
        desc_chars = 160
        allowed_keys = ["type", "description", "enum", "items", "required", "priority"]

    ordered_items = sorted(
        fields.items(),
        key=lambda item: (schema_priority_rank(item[1]), item[0]),
    )

    compact = {}

    for idx, (key, value) in enumerate(ordered_items):
        if idx >= max_fields:
            break

        if isinstance(value, dict):
            compact[key] = {
                k: clip_text(v, desc_chars) if isinstance(v, str) else v
                for k, v in value.items()
                if k in allowed_keys
            }
        else:
            compact[key] = clip_text(value, desc_chars)

    return compact


def compact_variables(
    variables: Dict[str, Any],
    schema: Optional[Dict[str, Any]] = None,
    max_items: int = 50,
) -> Dict[str, Any]:
    if not variables:
        return {}

    compact = {}
    schema_fields = get_schema_fields(schema or {})
    schema_keys = list(schema_fields.keys())

    for key in schema_keys:
        if key in variables and variables[key] not in [None, "", [], {}]:
            compact[key] = variables[key]
        if len(compact) >= max_items:
            return compact

    for key, value in variables.items():
        if key in compact:
            continue
        if value in [None, "", [], {}]:
            continue
        compact[key] = value
        if len(compact) >= max_items:
            break

    return compact


def get_subagent_config_list(agent_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw = agent_config.get("subagents") or []

    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, dict)]

    if isinstance(raw, dict):
        output = []
        for key, value in raw.items():
            if not isinstance(value, dict):
                continue
            item = dict(value)
            item.setdefault("id", key)
            item.setdefault("name", key)
            item.setdefault("when_to_use", item.get("description", ""))
            output.append(item)
        return output

    return []


def get_subagent_by_id(agent_config: Dict[str, Any], subagent_id: str) -> Dict[str, Any]:
    subagents = get_subagent_config_list(agent_config)

    if not subagents:
        return {
            "id": "general",
            "name": "General",
            "goal": "Help the user naturally and accurately.",
            "instructions": "Help the user naturally and accurately.",
            "allowed_actions": ["answer", "ask_follow_up"],
        }

    for subagent in subagents:
        if subagent.get("id") == subagent_id:
            return subagent

    return subagents[0]


def compact_subagents_for_manifest(agent_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    compact = []

    for subagent in get_subagent_config_list(agent_config):
        compact.append({
            "id": subagent.get("id", ""),
            "name": subagent.get("name", ""),
            "when_to_use": clip_text(subagent.get("when_to_use", ""), 280),
            "goal": clip_text(subagent.get("goal", ""), 220),
            "allowed_actions": subagent.get("allowed_actions", []),
        })

    return compact


def compact_tool_catalog(agent_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    tools = []

    for tool in agent_config.get("tool_catalog") or []:
        if not isinstance(tool, dict):
            continue
        tools.append({
            "name": tool.get("name", ""),
            "description": clip_text(tool.get("description", ""), 220),
            "required_inputs": tool.get("required_inputs", []),
            "result_fields": tool.get("result_fields", []),
            "source_of_truth": tool.get("source_of_truth", False),
            "result_policy": clip_text(tool.get("result_policy", ""), 260),
        })

    for tool in agent_config.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        tools.append({
            "name": tool.get("name", ""),
            "description": clip_text(tool.get("description", ""), 220),
            "operations": tool.get("operations", {}),
            "source_of_truth": True,
        })

    return tools


def unified_manifest_card(
    agent_config: Dict[str, Any],
    schema: Dict[str, Any],
    profile: str = "short",
) -> Dict[str, Any]:
    profile = (profile or "short").lower().strip()

    if profile == "full":
        return {
            "profile": "full",
            "assistant_goal": clip_text_head_tail(agent_config.get("assistant_goal", ""), 650),
            "conversation_style": clip_text_head_tail(agent_config.get("conversation_style", ""), 550),
            "language_policy": clip_text_head_tail(agent_config.get("language_policy", ""), 420),
            "routing_policy": clip_text_head_tail(agent_config.get("routing_policy", ""), 1200),
            "grounding_policy": clip_text_head_tail(agent_config.get("grounding_policy", ""), 900),
            "response_rules": (agent_config.get("response_rules") or [])[:20],
            "subagents": compact_subagents_for_manifest(agent_config),
            "tools": compact_tool_catalog(agent_config),
            "variable_schema": summarize_schema_fields(schema, max_fields=70, profile="full"),
        }

    return {
        "profile": "short",
        "assistant_goal": clip_text_head_tail(agent_config.get("assistant_goal", ""), 360),
        "conversation_style": clip_text_head_tail(agent_config.get("conversation_style", ""), 320),
        "language_policy": clip_text_head_tail(agent_config.get("language_policy", ""), 260),
        "routing_policy": clip_text_head_tail(agent_config.get("routing_policy", ""), 650),
        "grounding_policy": clip_text_head_tail(agent_config.get("grounding_policy", ""), 480),
        "response_rules": (agent_config.get("response_rules") or [])[:14],
        "subagents": compact_subagents_for_manifest(agent_config),
        "tools": compact_tool_catalog(agent_config),
        "variable_schema": summarize_schema_fields(schema, max_fields=36, profile="short"),
    }


def compact_agent_context(agent_config: Dict[str, Any], selected_subagent: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "assistant_goal": clip_text_head_tail(agent_config.get("assistant_goal", ""), 550),
        "conversation_style": clip_text_head_tail(agent_config.get("conversation_style", ""), 450),
        "persona": agent_config.get("persona", {}) if isinstance(agent_config.get("persona", {}), dict) else {},
        "proactive_nudge": agent_config.get("proactive_nudge", {}) if isinstance(agent_config.get("proactive_nudge", {}), dict) else {},
        "language_policy": clip_text_head_tail(agent_config.get("language_policy", ""), 320),
        "grounding_policy": clip_text_head_tail(agent_config.get("grounding_policy", ""), 650),
        "response_rules": (agent_config.get("response_rules") or [])[:14],
        "selected_subagent": {
            "id": selected_subagent.get("id", ""),
            "name": selected_subagent.get("name", ""),
            "goal": clip_text_head_tail(selected_subagent.get("goal", ""), 380),
            "instructions": clip_text_head_tail(selected_subagent.get("instructions", ""), 850),
            "allowed_actions": selected_subagent.get("allowed_actions", []),
        },
    }

def compact_manifest(manifest: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "user_intent": manifest.get("user_intent", ""),
        "selected_subagent_id": manifest.get("selected_subagent_id", ""),
        "chained_subagent_id": manifest.get("chained_subagent_id", ""),
        "chained_subagent_reason": manifest.get("chained_subagent_reason", ""),
        "detected_intents": manifest.get("detected_intents", [])[:8],
        "multi_intents": manifest.get("multi_intents", [])[:6],
        "parallel_tool_requests": manifest.get("parallel_tool_requests", [])[:6],
        "knowledge_queries": manifest.get("knowledge_queries", [])[:6],
        "response_synthesis": manifest.get("response_synthesis", {}),
        "multi_intent_execution_mode": manifest.get("multi_intent_execution_mode", ""),
        "conversation_stage": manifest.get("conversation_stage", ""),
        "workflow_stage": manifest.get("workflow_stage", ""),
        "customer_emotion": manifest.get("customer_emotion", ""),
        "emotion_trajectory": manifest.get("emotion_trajectory", ""),
        "funnel_stage": manifest.get("funnel_stage", ""),
        "best_guess_clarification": manifest.get("best_guess_clarification", {}),
        "user_expectation": manifest.get("user_expectation", ""),
        "risk_level": manifest.get("risk_level", "low"),
        "confidence": manifest.get("confidence", 0.0),
        "simple_response_mode": manifest.get("simple_response_mode", False),
        "simple_response_reason": manifest.get("simple_response_reason", ""),
        "needs_knowledge": manifest.get("needs_knowledge", False),
        "needs_memory": manifest.get("needs_memory", False),
        "needs_tool": manifest.get("needs_tool", False),
        "requested_tool_name": manifest.get("requested_tool_name", ""),
        "tool_request_payload": manifest.get("tool_request_payload", {}),
        "missing_tool_inputs": manifest.get("missing_tool_inputs", [])[:10],
        "needs_subagent_reasoning": manifest.get("needs_subagent_reasoning", False),
        "needs_quality_guard": manifest.get("needs_quality_guard", False),
        "needs_style_repair": manifest.get("needs_style_repair", False),
        "needs_full_manifest": manifest.get("needs_full_manifest", False),
        "manifest_profile_used": manifest.get("manifest_profile_used", ""),
        "response_style": manifest.get("response_style", ""),
        "reply_length": manifest.get("reply_length", ""),
        "should_ask_question": manifest.get("should_ask_question", False),
        "question_goal": manifest.get("question_goal", ""),
        "should_offer_next_action": manifest.get("should_offer_next_action", False),
        "response_strategy": clip_text(manifest.get("response_strategy", ""), 520),
        "response_brief": manifest.get("response_brief", {}),
        "manifest_error": manifest.get("manifest_error", ""),
        "reasoning_summary": manifest.get("reasoning_summary", ""),
    }


def graph_config_int(
    config: Dict[str, Any],
    path: str,
    default: int,
    minimum: int = 1,
    maximum: int = 20000,
) -> int:
    try:
        value = get_config_path_value(config or {}, path, default)
        number = int(value)
    except Exception:
        number = default

    return max(minimum, min(maximum, number))


def graph_dotted_get(obj: Dict[str, Any], path: str, default: Any = None) -> Any:
    current: Any = obj

    for part in str(path or "").split("."):
        part = part.strip()

        if not part:
            continue

        if isinstance(current, dict):
            current = current.get(part)
        else:
            return default

        if current is None:
            return default

    return current


def delete_dotted_path(data: Dict[str, Any], path: str) -> Dict[str, Any]:
    """
    Delete a dotted path from a dict copy.

    Generic prompt-compaction utility. It does not know any domain variable
    names; paths are supplied by assistant configuration.
    """
    if not isinstance(data, dict):
        return {}

    path_text = str(path or "").strip()
    if not path_text:
        return data

    parts = [part for part in path_text.split(".") if part]
    if not parts:
        return data

    current: Any = data

    for part in parts[:-1]:
        if not isinstance(current, dict):
            return data
        if part not in current:
            return data
        current = current.get(part)

    if isinstance(current, dict):
        current.pop(parts[-1], None)

    return data


def configured_response_compaction(agent_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Read response prompt compaction config.

    Supported locations:
    - assistant.response_compaction
    - assistant.response_context_compaction
    - assistant.smartness.response_compaction

    Behavior remains tenant-configured; Python only applies generic path rules.
    """
    if not isinstance(agent_config, dict):
        return {}

    for key in ["response_compaction", "response_context_compaction"]:
        value = agent_config.get(key)
        if isinstance(value, dict):
            if get_config_bool({"policy": value}, "policy.enabled", True) is False:
                return {}
            return value

    smartness = agent_config.get("smartness", {})
    if isinstance(smartness, dict):
        value = smartness.get("response_compaction")
        if isinstance(value, dict):
            if get_config_bool({"policy": value}, "policy.enabled", True) is False:
                return {}
            return value

    return {}


def compact_variables_for_response(
    variables: Dict[str, Any],
    schema: Optional[Dict[str, Any]],
    tool_result: Optional[Dict[str, Any]],
    agent_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compact variables for response prompts and remove paths duplicated by the
    current tool_result.

    This implements the code-expert token fix without embedding any domain
    fields. The assistant config declares which variable paths are redundant
    when specific tool_result paths are present.
    """
    cfg = configured_response_compaction(agent_config)
    max_items = graph_config_int(
        {"response_compaction": cfg},
        "response_compaction.max_variable_items",
        default=50,
        minimum=1,
        maximum=200,
    )

    compact = compact_variables(variables or {}, schema or {}, max_items=max_items)

    if not isinstance(compact, dict) or not compact:
        return {}

    if not isinstance(tool_result, dict):
        tool_result = {}

    explicit_excludes = cfg.get("exclude_variable_paths", [])
    if isinstance(explicit_excludes, list):
        for path in explicit_excludes:
            delete_dotted_path(compact, str(path or ""))

    duplicate_rules = cfg.get("exclude_variable_paths_when_tool_result_has", [])
    if not isinstance(duplicate_rules, list):
        duplicate_rules = []

    for rule in duplicate_rules:
        if not isinstance(rule, dict):
            continue

        result_path = str(
            rule.get("tool_result_path")
            or rule.get("result_path")
            or ""
        ).strip()

        if not result_path:
            continue

        result_value = graph_dotted_get(tool_result, result_path, None)

        if result_value in [None, "", [], {}]:
            continue

        variable_paths = rule.get("variable_paths", [])
        if isinstance(variable_paths, str):
            variable_paths = [variable_paths]

        if not isinstance(variable_paths, list):
            continue

        for variable_path in variable_paths:
            delete_dotted_path(compact, str(variable_path or ""))

    return compact


def configured_response_model_routing(agent_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Read generic response model routing config.

    Supported locations:
    - assistant.response_model_routing
    - assistant.model_routing.response
    - assistant.smartness.response_model_routing
    """
    if not isinstance(agent_config, dict):
        return {}

    value = agent_config.get("response_model_routing")
    if isinstance(value, dict):
        return value

    model_routing = agent_config.get("model_routing")
    if isinstance(model_routing, dict):
        response_value = model_routing.get("response")
        if isinstance(response_value, dict):
            return response_value

    smartness = agent_config.get("smartness")
    if isinstance(smartness, dict):
        value = smartness.get("response_model_routing")
        if isinstance(value, dict):
            return value

    return {}


def get_response_model(
    state: AgentState,
    tool_result: Optional[Dict[str, Any]] = None,
    *,
    simple_response: bool = False,
) -> str:
    """
    Route low-risk/simple responses to a configured cheaper model while keeping
    grounded, risky, tool, memory, and knowledge responses on the default model.

    The routing policy is config-driven and disabled by default until the bundle
    opts in.
    """
    agent_config = state.get("agent_config", {}) or {}
    cfg = configured_response_model_routing(agent_config)

    if not cfg.get("enabled", False):
        return MODEL_RESPONSE

    manifest = state.get("manifest", {}) or {}
    tool_result = tool_result if isinstance(tool_result, dict) else (state.get("tool_result", {}) or {})

    default_model = str(cfg.get("default_model") or cfg.get("fallback_model") or MODEL_RESPONSE).strip() or MODEL_RESPONSE
    simple_model = str(cfg.get("simple_model") or cfg.get("low_risk_model") or default_model).strip() or default_model

    never = cfg.get("never_use_simple_model_when", {})
    if not isinstance(never, dict):
        never = {}

    if never.get("tool_result", True) and tool_result:
        return default_model

    if never.get("knowledge", True):
        knowledge = str(state.get("knowledge") or "")
        if knowledge.strip() and "NO_CONFIDENT_KNOWLEDGE_FOUND" not in knowledge:
            return default_model

    if never.get("memory", True):
        memories = str(state.get("memories") or "")
        if memories.strip() and "No relevant memories" not in memories:
            return default_model

    if never.get("needs_tool", True) and manifest.get("needs_tool"):
        return default_model

    if never.get("needs_knowledge", True) and manifest.get("needs_knowledge"):
        return default_model

    if never.get("needs_memory", True) and manifest.get("needs_memory"):
        return default_model

    blocked_risks = never.get("risk_levels", ["medium", "high"])
    if isinstance(blocked_risks, list):
        risk = str(manifest.get("risk_level") or "low").strip().lower()
        if risk in {str(item or "").strip().lower() for item in blocked_risks}:
            return default_model

    try:
        min_confidence = float(cfg.get("simple_min_confidence", 0.8) or 0.8)
    except Exception:
        min_confidence = 0.8

    if (simple_response or manifest.get("simple_response_mode")) and manifest_confidence(manifest) >= min_confidence:
        return simple_model

    return default_model


def get_manifest_retry_policy(agent_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Configurable retry policy for short -> full manifest escalation.
    """
    if not isinstance(agent_config, dict):
        return {}

    for key in ["manifest_retry_policy", "planner_retry_policy"]:
        value = agent_config.get(key)
        if isinstance(value, dict):
            return value

    context = get_manifest_context_config(agent_config)
    value = context.get("retry_policy") if isinstance(context, dict) else None
    if isinstance(value, dict):
        return value

    return {}


def should_retry_full_manifest_for_stripped_updates(
    prepared_updates: Dict[str, Any],
    filtered_updates: Dict[str, Any],
    agent_config: Dict[str, Any],
) -> bool:
    """
    Retry the full manifest only when the short planner tried to write a
    meaningful amount of protected/source-of-truth state.

    Thresholds are config-driven to prevent an aggressive token-cost regression.
    """
    if not prepared_updates:
        return False

    retry_policy = get_manifest_retry_policy(agent_config)

    try:
        ratio_threshold = float(
            retry_policy.get("source_of_truth_strip_ratio_threshold", 0.6)
            or 0.6
        )
    except Exception:
        ratio_threshold = 0.6

    min_stripped_updates = graph_config_int(
        {"manifest_retry_policy": retry_policy},
        "manifest_retry_policy.min_stripped_updates_for_retry",
        default=3,
        minimum=1,
        maximum=100,
    )

    stripped_count = max(len(prepared_updates) - len(filtered_updates), 0)
    strip_ratio = stripped_count / max(len(prepared_updates), 1)

    return strip_ratio > ratio_threshold and stripped_count >= min_stripped_updates


def build_response_guidance_block(
    agent_config: Dict[str, Any],
    tool_result: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Return exactly one response-guidance block for the response prompt.

    Code-expert token fix: do not send both the full layered rules and the full
    template policy when only one is relevant.
    """
    template_policy = compact_template_response_policy(agent_config, tool_result)
    cfg = configured_response_compaction(agent_config)
    prefer_template_when_available = bool(cfg.get("prefer_template_policy_when_available", True))

    if template_policy and prefer_template_when_available and isinstance(tool_result, dict) and tool_result:
        return {
            "mode": "template_response_policy",
            "policy": template_policy,
        }

    return {
        "mode": "layered_response_rules",
        "policy": build_layered_response_rules(agent_config),
    }


def get_manifest_context_config(agent_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return config for planner continuity.

    Supported config:
    assistant.manifest_context or assistant.planner_context:
      previous_manifest_summary_enabled: bool
      previous_manifest_summary_max_chars: int
      previous_manifest_summary_fields: list[str]

    This is intentionally assistant-level config and contains no domain-specific
    workflow, booking, or field names.
    """
    for key in ["manifest_context", "planner_context"]:
        value = (agent_config or {}).get(key)
        if isinstance(value, dict):
            return value

    return {}


def previous_manifest_summary_enabled(agent_config: Dict[str, Any]) -> bool:
    config = get_manifest_context_config(agent_config)

    if "previous_manifest_summary_enabled" not in config:
        return True

    value = config.get("previous_manifest_summary_enabled")

    if isinstance(value, bool):
        return value

    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def summarize_manifest_for_next_turn(
    manifest: Dict[str, Any],
    agent_config: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Build a compact, configurable summary of the manifest for the next turn.

    The goal is conversation momentum: the next manifest can see what it decided
    last turn, what workflow/stage it believed it was in, and what next move it
    had planned. This avoids reclassifying every turn in isolation.
    """
    agent_config = agent_config or {}

    if not previous_manifest_summary_enabled(agent_config):
        return ""

    if not isinstance(manifest, dict) or not manifest:
        return ""

    config = get_manifest_context_config(agent_config)
    max_chars = graph_config_int(
        {"manifest_context": config},
        "manifest_context.previous_manifest_summary_max_chars",
        default=600,
        minimum=200,
        maximum=6000,
    )

    configured_fields = config.get("previous_manifest_summary_fields", [])

    if isinstance(configured_fields, list) and configured_fields:
        compact: Dict[str, Any] = {}

        for path in configured_fields:
            path_text = str(path or "").strip()

            if not path_text:
                continue

            value = graph_dotted_get(manifest, path_text, None)

            if value in [None, "", [], {}]:
                continue

            compact[path_text] = value
    else:
        compact = compact_manifest(manifest)

    if not compact:
        return ""

    return safe_json(compact, max_chars=max_chars)


def previous_manifest_summary_from_state(state: AgentState) -> str:
    """
    Read the previous manifest summary from persisted state when available.

    If an older state only persisted the prior manifest object, derive a summary
    from that object. The unified_manifest_node stores the current summary back
    into previous_manifest_summary for the next graph invocation.
    """
    agent_config = state.get("agent_config", {}) or {}

    if not previous_manifest_summary_enabled(agent_config):
        return ""

    config = get_manifest_context_config(agent_config)
    max_chars = graph_config_int(
        {"manifest_context": config},
        "manifest_context.previous_manifest_summary_max_chars",
        default=600,
        minimum=200,
        maximum=6000,
    )

    stored = str(state.get("previous_manifest_summary") or "").strip()

    if stored:
        return clip_text(stored, max_chars)

    prior_manifest = state.get("manifest", {}) or {}

    if isinstance(prior_manifest, dict) and prior_manifest:
        return summarize_manifest_for_next_turn(prior_manifest, agent_config)

    return ""


def compact_analysis(analysis: Dict[str, Any]) -> Dict[str, Any]:
    if not analysis:
        return {}

    return {
        "understanding": clip_text(analysis.get("understanding", ""), 450),
        "user_goal": clip_text(analysis.get("user_goal", ""), 280),
        "detected_intents": analysis.get("detected_intents", [])[:8],
        "facts_to_use": analysis.get("facts_to_use", [])[:8],
        "facts_missing": analysis.get("facts_missing", [])[:8],
        "variable_updates_to_consider": analysis.get("variable_updates_to_consider", {}),
        "next_best_step": clip_text(analysis.get("next_best_step", ""), 360),
        "response_constraints": analysis.get("response_constraints", [])[:10],
        "should_chain": analysis.get("should_chain", False),
        "recommended_chained_subagent_id": analysis.get("recommended_chained_subagent_id", ""),
        "confidence": analysis.get("confidence", 0.0),
    }


def compact_knowledge_for_final(knowledge: str, max_chars: int = 1400) -> str:
    if not knowledge:
        return "No knowledge retrieved."
    return clip_text(knowledge, max_chars)


def compact_memories_for_final(memories: str, max_chars: int = 500) -> str:
    if not memories:
        return "No relevant memories retrieved."
    return clip_text(memories, max_chars)


def manifest_confidence(manifest: Dict[str, Any]) -> float:
    try:
        return float(manifest.get("confidence", 0) or 0)
    except Exception:
        return 0.0


def tool_result_exists(state: AgentState) -> bool:
    result = state.get("tool_result")
    return isinstance(result, dict) and bool(result)


def manifest_has_parallel_tool_requests(manifest: Dict[str, Any]) -> bool:
    requests = manifest.get("parallel_tool_requests", [])
    return isinstance(requests, list) and any(isinstance(item, dict) for item in requests)


def manifest_has_knowledge_queries(manifest: Dict[str, Any]) -> bool:
    queries = manifest.get("knowledge_queries", [])
    return isinstance(queries, list) and any(str(item or "").strip() for item in queries)


def manifest_has_multi_intents(manifest: Dict[str, Any]) -> bool:
    intents = manifest.get("multi_intents", [])
    return isinstance(intents, list) and any(isinstance(item, dict) for item in intents)


def should_use_simple_response(state: AgentState) -> bool:
    manifest = state.get("manifest", {}) or {}

    if active_deterministic_flow_subagent_id_from_state(state):
        return False

    if manifest.get("needs_tool") or manifest_has_parallel_tool_requests(manifest):
        return False
    if manifest.get("needs_knowledge") or manifest_has_knowledge_queries(manifest):
        return False
    if manifest.get("needs_memory"):
        return False
    if manifest.get("risk_level") in ["high", "medium"]:
        return False
    if manifest_confidence(manifest) < 0.72:
        return False
    if tool_result_exists(state):
        return False

    return bool(manifest.get("simple_response_mode"))


def should_run_quality_guard(state: AgentState) -> bool:
    if not QUALITY_GUARD_ENABLED:
        return False

    manifest = state.get("manifest", {}) or {}
    tool_result = state.get("tool_result", {}) or {}
    knowledge = state.get("knowledge", "") or ""
    answer = state.get("final_answer", "") or ""
    agent_config = state.get("agent_config", {}) or {}

    policy = {}
    if isinstance(agent_config, dict):
        raw_policy = agent_config.get("quality_guard_policy")
        if isinstance(raw_policy, dict):
            policy = raw_policy
        else:
            smartness = agent_config.get("smartness", {})
            if isinstance(smartness, dict) and isinstance(smartness.get("quality_guard_policy"), dict):
                policy = smartness.get("quality_guard_policy")

    if policy and get_config_bool({"policy": policy}, "policy.enabled", True) is False:
        return False

    def policy_bool(key: str, default: bool = True) -> bool:
        value = policy.get(key, default)
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}

    long_answer_chars = graph_config_int(
        {"quality_guard_policy": policy},
        "quality_guard_policy.long_answer_chars",
        default=700,
        minimum=120,
        maximum=8000,
    )

    if manifest.get("needs_quality_guard"):
        return True
    if policy_bool("run_on_style_repair", True) and manifest.get("needs_style_repair"):
        return True
    if policy_bool("run_on_tool_result", True) and tool_result:
        return True
    if policy_bool("run_on_multi_tool_results", True) and (state.get("multi_tool_results") or state.get("multi_knowledge")):
        return True
    if policy_bool("run_on_tool_intent", True) and (manifest.get("needs_tool") or manifest_has_parallel_tool_requests(manifest)):
        return True

    risk_levels = policy.get("risk_levels", ["medium", "high"])
    if isinstance(risk_levels, list):
        risk = str(manifest.get("risk_level") or "low").strip().lower()
        if risk in {str(item or "").strip().lower() for item in risk_levels}:
            return True

    if policy_bool("run_on_knowledge", True) and "NO_CONFIDENT_KNOWLEDGE_FOUND" not in knowledge and knowledge.strip():
        return True
    if policy_bool("run_on_long_answer", True) and len(answer) > long_answer_chars:
        return True

    return False


def build_planner_compat(manifest: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "user_intent": manifest.get("user_intent", ""),
        "selected_subagent_id": manifest.get("selected_subagent_id", ""),
        "chained_subagent_id": manifest.get("chained_subagent_id", ""),
        "chained_subagent_reason": manifest.get("chained_subagent_reason", ""),
        "detected_intents": manifest.get("detected_intents", []),
        "multi_intents": manifest.get("multi_intents", []),
        "parallel_tool_requests": manifest.get("parallel_tool_requests", []),
        "knowledge_queries": manifest.get("knowledge_queries", []),
        "response_synthesis": manifest.get("response_synthesis", {}),
        "multi_intent_execution_mode": manifest.get("multi_intent_execution_mode", ""),
        "conversation_stage": manifest.get("conversation_stage", ""),
        "workflow_stage": manifest.get("workflow_stage", ""),
        "customer_emotion": manifest.get("customer_emotion", ""),
        "emotion_trajectory": manifest.get("emotion_trajectory", ""),
        "funnel_stage": manifest.get("funnel_stage", ""),
        "best_guess_clarification": manifest.get("best_guess_clarification", {}),
        "needs_knowledge": manifest.get("needs_knowledge", False),
        "needs_memory": manifest.get("needs_memory", False),
        "needs_tool": manifest.get("needs_tool", False),
        "requested_tool_name": manifest.get("requested_tool_name", ""),
        "risk_level": manifest.get("risk_level", "low"),
        "confidence": manifest.get("confidence", 0.0),
        "response_strategy": manifest.get("response_strategy", ""),
        "simple_response_mode": manifest.get("simple_response_mode", False),
        "needs_subagent_reasoning": manifest.get("needs_subagent_reasoning", False),
        "needs_quality_guard": manifest.get("needs_quality_guard", False),
        "needs_style_repair": manifest.get("needs_style_repair", False),
        "needs_full_manifest": manifest.get("needs_full_manifest", False),
        "manifest_profile_used": manifest.get("manifest_profile_used", ""),
        "tool_request_payload": manifest.get("tool_request_payload", {}),
        "missing_tool_inputs": manifest.get("missing_tool_inputs", []),
        "manifest_error": manifest.get("manifest_error", ""),
        "reasoning_summary": manifest.get("reasoning_summary", ""),
    }


def unify_subagent_id(selected_id: str) -> str:
    aliases = {
        "booking_advisor": "booking",
        "branch_locator": "location",
        "diagnosis_advisor": "troubleshooting",
        "general_support": "general",
        "human_handoff": "handoff",
    }
    selected_id = str(selected_id or "").strip()
    return aliases.get(selected_id, selected_id)


def append_unique(values: List[str], additions: List[str]) -> List[str]:
    output: List[str] = []

    for item in values or []:
        text = str(item or "").strip()
        if text and text not in output:
            output.append(text)

    for item in additions or []:
        text = str(item or "").strip()
        if text and text not in output:
            output.append(text)

    return output


def has_known_branch(
    variables: Dict[str, Any],
    agent_config: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Return whether a location/branch-like context is already known.

    The checked paths are configured per assistant. If an assistant does not
    declare known_branch_paths, this helper returns False instead of assuming a
    service-center field schema.
    """
    agent_config = agent_config or {}
    paths = as_string_list(agent_config.get("known_branch_paths", []))

    return any(
        is_present(deep_get(variables, path, ""))
        for path in paths
        if str(path or "").strip()
    )


def get_config_path_value(config: Dict[str, Any], path: str, default: Any = None) -> Any:
    current: Any = config

    for part in str(path or "").split("."):
        if not part:
            continue

        if isinstance(current, dict):
            current = current.get(part)
        else:
            return default

        if current is None:
            return default

    return current


def collect_configured_phrases(
    agent_config: Dict[str, Any],
    paths: List[str]
) -> List[str]:
    phrases: List[str] = []

    for path in paths or []:
        value = get_config_path_value(agent_config, str(path), [])

        if isinstance(value, list):
            phrases.extend([str(item) for item in value if str(item or "").strip()])
        elif isinstance(value, str) and value.strip():
            phrases.append(value)

    return phrases


def has_visit_intent_context(
    variables: Dict[str, Any],
    guardrail_config: Dict[str, Any]
) -> bool:
    context_paths = guardrail_config.get("context_paths", [
        "troubleshooting.stage",
        "troubleshooting.inspection_recommended",
        "service_needed",
    ])

    context_stage_values = {
        str(value).strip()
        for value in guardrail_config.get("context_stage_values", ["active", "complete"])
        if str(value).strip()
    }

    for path in context_paths:
        value = deep_get(variables, str(path), None)

        if value in [None, "", [], {}]:
            continue

        if isinstance(value, bool) and value:
            return True

        value_text = str(value).strip()

        if value_text and not context_stage_values:
            return True

        if value_text in context_stage_values:
            return True

        if str(path) not in {"troubleshooting.stage"} and value_text:
            return True

    return False


def has_configured_visit_intent(
    message: str,
    agent_config: Dict[str, Any],
    variables: Dict[str, Any]
) -> bool:
    normalization = agent_config.get("normalization", {}) or {}
    guardrail_config = (
        agent_config.get("routing_guardrails", {})
        .get("visit_intent", {})
    )

    if not isinstance(guardrail_config, dict) or not guardrail_config.get("enabled", False):
        subagents = agent_config.get("subagents", {}) or {}
        booking_config = subagents.get("booking", {}) or {}
        troubleshooting_config = subagents.get("troubleshooting", {}) or {}

        fallback_phrases: List[str] = []
        fallback_phrases.extend(booking_config.get("trigger_phrases", []) or [])
        fallback_phrases.extend(troubleshooting_config.get("direct_booking_phrases", []) or [])

        return bool(
            matches_any(message, fallback_phrases, normalization)
            and has_visit_intent_context(variables, {})
        )

    strong_phrases = guardrail_config.get("strong_phrases", []) or []
    if matches_any(message, strong_phrases, normalization):
        return True

    contextual_paths = guardrail_config.get("contextual_phrases_from", []) or []
    contextual_phrases = collect_configured_phrases(agent_config, contextual_paths)

    if not matches_any(message, contextual_phrases, normalization):
        return False

    if guardrail_config.get("requires_context_for_contextual_phrases", True):
        return has_visit_intent_context(variables, guardrail_config)

    return True


def apply_routing_guardrails(
    manifest: Dict[str, Any],
    message: str,
    agent_config: Dict[str, Any],
    variables: Dict[str, Any]
) -> Dict[str, Any]:
    patched = dict(manifest or {})

    if not has_configured_visit_intent(message, agent_config, variables):
        return patched

    known_branch = has_known_branch(variables, agent_config)

    patched["simple_response_mode"] = False
    patched["needs_tool"] = True
    patched["needs_subagent_reasoning"] = False
    patched["needs_quality_guard"] = True
    patched["needs_style_repair"] = True
    patched["user_intent"] = "visit_or_booking_after_troubleshooting"
    patched["conversation_stage"] = "visit_intent_after_diagnostics"

    extracted_updates = patched.get("extracted_updates", {})
    if not isinstance(extracted_updates, dict):
        extracted_updates = {}

    if known_branch:
        patched["selected_subagent_id"] = "booking"
        patched["workflow_stage"] = "booking_information_collection"
        patched["response_strategy"] = (
            "The user agreed to inspect/check the car after troubleshooting. "
            "Do not continue diagnostics. Continue toward visit booking. "
            "A branch is already known from variables, so ask only for the preferred day/date if missing. "
            "Do not ask for location again unless branch data is missing."
        )
        next_move = "Move to booking flow using the known branch."
        must_do = [
            "acknowledge the user wants to inspect/check the car",
            "do not ask more diagnostic questions",
            "use the known branch if present",
            "ask for preferred date if date is missing"
        ]
    else:
        patched["selected_subagent_id"] = "location"
        patched["workflow_stage"] = "nearest_branch_location_needed"
        patched["response_strategy"] = (
            "The user agreed to inspect/check the car after troubleshooting. "
            "Do not continue diagnostics. No branch is known yet, so ask naturally for the user's area/location "
            "to find the nearest suitable branch before checking appointments."
        )
        extracted_updates["location.intent"] = "nearest_branch"
        next_move = "Ask for area/location to find the nearest branch."
        must_do = [
            "acknowledge the user wants to inspect/check the car",
            "do not ask more diagnostic questions",
            "ask for the user's area/location",
            "do not ask for appointment date before branch is known"
        ]

    patched["extracted_updates"] = extracted_updates

    guardrail_cfg = agent_config.get("routing_guardrails", {})
    visit_guardrail_cfg = (
        guardrail_cfg.get("visit_intent_after_diagnostics", {})
        if isinstance(guardrail_cfg, dict)
        else {}
    )
    if not isinstance(visit_guardrail_cfg, dict):
        visit_guardrail_cfg = {}

    brief_tone = str(
        visit_guardrail_cfg.get("brief_tone")
        or agent_config.get("language_policy", "")
        or ""
    ).strip() or "warm and natural"

    brief_language = str(
        visit_guardrail_cfg.get("brief_language")
        or ""
    ).strip() or "same as user"

    configured_must_not_do = visit_guardrail_cfg.get("brief_must_not_do", [])
    if not isinstance(configured_must_not_do, list) or not configured_must_not_do:
        configured_must_not_do = agent_config.get("routing_guardrail_default_must_not_do", [])
    if not isinstance(configured_must_not_do, list):
        configured_must_not_do = []

    brief = patched.get("response_brief")
    if not isinstance(brief, dict):
        brief = {}

    brief["tone"] = brief_tone
    brief["language"] = brief_language
    brief["reply_length"] = str(visit_guardrail_cfg.get("brief_reply_length") or "short")
    brief["next_move"] = next_move
    brief["must_do"] = append_unique(brief.get("must_do", []), must_do)
    brief["must_not_do"] = append_unique(
        brief.get("must_not_do", []),
        [str(item) for item in configured_must_not_do if str(item or "").strip()],
    )

    patched["response_brief"] = brief

    return patched


def flatten_update_paths(
    updates: Dict[str, Any],
    prefix: str = ""
) -> Dict[str, Any]:
    """
    Convert nested update objects into dotted-path updates.

    This prevents a manifest/subagent update like {"customer_profile": {"name": "..."}}
    from replacing the entire existing customer_profile object and deleting sibling
    fields such as phone or plate_number.
    """
    output: Dict[str, Any] = {}

    if not isinstance(updates, dict):
        return output

    for key, value in updates.items():
        key_text = str(key or "").strip()

        if not key_text:
            continue

        path = f"{prefix}.{key_text}" if prefix else key_text

        if isinstance(value, dict):
            nested = flatten_update_paths(value, path)
            if nested:
                output.update(nested)
            elif value not in [None, "", [], {}]:
                output[path] = value
        else:
            output[path] = value

    return output


def apply_configured_manifest_aliases(
    flat_updates: Dict[str, Any],
    agent_config: Dict[str, Any],
    schema: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Promote extracted aliases using config/schema only.

    Supported sources:
    - assistant.manifest_update_aliases:
      [{"source_path": "...", "target_path": "..."}]
    - schema variables with alias_for / alias_for_path / target_path
    - nested schema properties with alias_for / alias_for_path / target_path

    No domain-specific alias names are embedded here.
    """
    output = dict(flat_updates or {})

    if not isinstance(output, dict):
        return {}

    configured_aliases = agent_config.get("manifest_update_aliases", [])
    if isinstance(configured_aliases, list):
        for item in configured_aliases:
            if not isinstance(item, dict):
                continue

            source_path = str(item.get("source_path") or "").strip()
            target_path = str(item.get("target_path") or "").strip()

            if not source_path or not target_path:
                continue

            if source_path in output and output.get(source_path) not in [None, "", [], {}]:
                output.setdefault(target_path, output.get(source_path))

    fields = get_schema_fields(schema or {})

    if isinstance(fields, dict):
        for key, meta in fields.items():
            if not isinstance(meta, dict):
                continue

            source_key = str(key or "").strip()
            if not source_key:
                continue

            for alias_key in ["alias_for_path", "target_path", "alias_for"]:
                target = str(meta.get(alias_key) or "").strip()
                if not target:
                    continue

                target_path = target if "." in target else target

                if source_key in output and output.get(source_key) not in [None, "", [], {}]:
                    output.setdefault(target_path, output.get(source_key))

            properties = meta.get("properties", {})
            if not isinstance(properties, dict):
                continue

            for prop_key, prop_meta in properties.items():
                if not isinstance(prop_meta, dict):
                    continue

                source_path = f"{source_key}.{str(prop_key or '').strip()}"

                if source_path not in output or output.get(source_path) in [None, "", [], {}]:
                    continue

                for alias_key in ["alias_for_path", "target_path", "alias_for"]:
                    target = str(prop_meta.get(alias_key) or "").strip()
                    if not target:
                        continue

                    target_path = target if "." in target else f"{source_key}.{target}"
                    output.setdefault(target_path, output.get(source_path))

    return output



def merge_variables_intelligently(
    existing: Dict[str, Any],
    incoming: Dict[str, Any],
    deletions: List[str],
    agent_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Smart manifest variable merge.

    Rules:
    - Existing values are never lost unless the user explicitly changed/deleted them.
    - Empty incoming values never overwrite non-empty existing values.
    - Source-of-truth paths are protected and remain owned by tools/subagents.
    - Deletions only apply to non-source-of-truth paths.
    """
    result = json.loads(json.dumps(existing or {}, ensure_ascii=False))

    for path in deletions or []:
        path_text = str(path or "").strip()

        if not path_text:
            continue

        if is_source_of_truth_path(path_text, agent_config):
            continue

        parts = [part for part in path_text.split(".") if part]
        if not parts:
            continue

        current: Any = result

        for part in parts[:-1]:
            if not isinstance(current, dict):
                current = None
                break
            current = current.get(part)

        if isinstance(current, dict) and parts[-1] in current:
            del current[parts[-1]]

    for key, value in (incoming or {}).items():
        key_text = str(key or "").strip()

        if not key_text:
            continue

        if is_source_of_truth_path(key_text, agent_config):
            continue

        existing_value = deep_get(result, key_text)

        if existing_value not in [None, "", [], {}] and value in [None, "", [], {}]:
            continue

        if value in [None, "", [], {}] and existing_value in [None, "", [], {}]:
            continue

        parts = [part for part in key_text.split(".") if part]
        if not parts:
            continue

        current: Any = result

        for part in parts[:-1]:
            if not isinstance(current, dict):
                current = None
                break

            if part not in current or not isinstance(current.get(part), dict):
                current[part] = {}

            current = current[part]

        if not isinstance(current, dict):
            continue

        final_key = parts[-1]
        existing_leaf = current.get(final_key)

        if isinstance(existing_leaf, dict) and isinstance(value, dict):
            merged = dict(existing_leaf)
            for child_key, child_value in value.items():
                if child_value in [None, "", [], {}] and merged.get(child_key) not in [None, "", [], {}]:
                    continue
                if child_value not in [None, "", [], {}]:
                    merged[child_key] = child_value
            current[final_key] = merged
        else:
            current[final_key] = value

    return result


def prepare_manifest_extracted_updates(
    updates: Dict[str, Any],
    agent_config: Dict[str, Any],
    schema: Dict[str, Any]
) -> Dict[str, Any]:
    flat = flatten_update_paths(updates or {})
    return apply_configured_manifest_aliases(
        flat_updates=flat,
        agent_config=agent_config or {},
        schema=schema or {},
    )


def prepare_variable_updates_for_patch(updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Subagent results may return either patch-style dotted paths or a full state
    object. Flattening gives apply_variable_patch merge semantics instead of
    replacing nested objects.
    """
    return flatten_update_paths(updates or {})




# ── SEMANTIC VARIABLE EXTRACTION NODE ────────────────────────────────────────

def get_semantic_extraction_config(agent_config: Dict[str, Any]) -> Dict[str, Any]:
    cfg = agent_config.get("semantic_variable_extraction", {})
    return cfg if isinstance(cfg, dict) else {}


def semantic_extraction_is_enabled(agent_config: Dict[str, Any]) -> bool:
    if not SEMANTIC_EXTRACTION_GLOBAL_ENABLED:
        return False

    cfg = get_semantic_extraction_config(agent_config)
    return bool(cfg.get("enabled", False))


def field_is_required_now(
    field: Dict[str, Any],
    variables: Dict[str, Any],
    manifest: Dict[str, Any],
    agent_config: Dict[str, Any],
) -> bool:
    """
    Evaluate whether a semantic extraction field is required now.

    Zero hardcoding:
    - field target path comes from field.target_path
    - stage path comes from field.required_when_stage_path
    - stage values come from field.required_when_stages
    - variable predicates come from field.required_when_paths
    - manifest predicates come from field.required_when_manifest
    """
    if not isinstance(field, dict):
        return False

    target_path = str(field.get("target_path") or "").strip()
    if not target_path:
        return False

    existing = deep_get(variables, target_path)
    if existing not in [None, "", [], {}]:
        return False

    if field.get("always_required", False):
        return True

    conditions_met: List[bool] = []

    stage_path = str(field.get("required_when_stage_path") or "").strip()
    required_stages = field.get("required_when_stages", [])

    if stage_path and isinstance(required_stages, list) and required_stages:
        current_stage = str(deep_get(variables, stage_path) or "").strip()
        stage_values = {
            str(stage_value or "").strip()
            for stage_value in required_stages
            if str(stage_value or "").strip()
        }
        if current_stage:
            conditions_met.append(current_stage in stage_values)

    path_conditions = field.get("required_when_paths", [])
    if isinstance(path_conditions, list):
        for condition in path_conditions:
            if not isinstance(condition, dict):
                continue

            check_path = str(condition.get("path") or "").strip()
            if not check_path:
                continue

            value = deep_get(variables, check_path)

            if condition.get("must_be_present", False):
                conditions_met.append(value not in [None, "", [], {}])

            if condition.get("must_be_absent", False):
                conditions_met.append(value in [None, "", [], {}])

            equals_values = condition.get("equals", None)
            if equals_values is not None:
                if not isinstance(equals_values, list):
                    equals_values = [equals_values]
                normalized_value = str(value or "").strip().lower()
                allowed = {
                    str(item or "").strip().lower()
                    for item in equals_values
                }
                conditions_met.append(normalized_value in allowed)

            not_in_values = condition.get("not_in", [])
            if isinstance(not_in_values, list) and not_in_values:
                normalized_value = str(value or "").strip().lower()
                blocked = {
                    str(item or "").strip().lower()
                    for item in not_in_values
                }
                conditions_met.append(normalized_value not in blocked)

    manifest_conditions = field.get("required_when_manifest", {})
    if isinstance(manifest_conditions, dict):
        for manifest_key, expected_value in manifest_conditions.items():
            key_text = str(manifest_key or "").strip()
            if not key_text:
                continue

            actual = manifest.get(key_text)

            if isinstance(expected_value, list):
                conditions_met.append(actual in expected_value)
            else:
                conditions_met.append(actual == expected_value)

    if not conditions_met:
        return True

    return all(conditions_met)


def build_extraction_prompt(
    field: Dict[str, Any],
    message: str,
    variables: Dict[str, Any],
    agent_config: Dict[str, Any],
) -> str:
    normalization = agent_config.get("normalization", {}) or {}
    digit_map = normalization.get("digit_map", {}) if isinstance(normalization, dict) else {}

    digit_map_desc = ""
    if isinstance(digit_map, dict) and digit_map:
        digit_map_desc = (
            "Digit normalization: apply this mapping → "
            + ", ".join(f"{key}→{value}" for key, value in list(digit_map.items())[:16])
        )

    examples_text = ""
    examples = field.get("examples", [])
    if isinstance(examples, list) and examples:
        lines: List[str] = []
        for example in examples[:4]:
            if not isinstance(example, dict):
                continue
            user_message = str(example.get("user_message") or "").strip()
            value = str(example.get("value") or "").strip()
            if user_message and value:
                lines.append(f'  User said: "{user_message}" → Value: "{value}"')
        if lines:
            examples_text = "Examples:\n" + "\n".join(lines)

    compact_vars = compact_variables(variables or {}, max_items=12)

    return (
        "Extract one specific field from the user message.\n\n"
        f"Field ID: {field.get('id', '')}\n"
        f"Description: {field.get('description', '')}\n"
        f"Output format: {field.get('output_format', '')}\n"
        f"Validation: {field.get('validation_description', '')}\n"
        f"{digit_map_desc}\n"
        f"{examples_text}\n\n"
        f'User message: "{message}"\n\n'
        "Known variables (context only; do not repeat these):\n"
        f"{json.dumps(compact_vars, ensure_ascii=False)}\n\n"
        "Return JSON only with exactly two keys:\n"
        '  "found": true if the value is clearly present, false otherwise\n'
        '  "value": the extracted value in the specified output format, or "" if not found\n'
        "Do not invent, guess, infer absent values, or include prose."
    )


def run_single_field_extraction(
    field: Dict[str, Any],
    message: str,
    variables: Dict[str, Any],
    agent_config: Dict[str, Any],
) -> Optional[str]:
    prompt_text = build_extraction_prompt(
        field=field,
        message=message,
        variables=variables,
        agent_config=agent_config,
    )

    system_msg = str(
        field.get("extractor_system_prompt")
        or agent_config.get("semantic_extraction_system_prompt")
        or "You are a precise field extractor. Return only valid JSON."
    ).strip()

    try:
        response = semantic_extraction_llm.invoke([
            SystemMessage(content=system_msg),
            HumanMessage(content=prompt_text),
        ])
        raw = response.content if hasattr(response, "content") else str(response)
        data = json.loads(raw)

        if not isinstance(data, dict):
            return None

        found = bool(data.get("found", False))
        value = str(data.get("value") or "").strip()

        if found and value:
            return value

    except Exception:
        return None

    return None


def semantic_value_present_in_latest_message(value: Any, message: str, agent_config: Dict[str, Any]) -> bool:
    normalization = agent_config.get("normalization", {}) or {}
    digit_map = normalization.get("digit_map", {}) if isinstance(normalization, dict) else {}
    value_text = graph_normalize_digits(str(value or "").strip(), digit_map)
    message_text = graph_normalize_digits(str(message or "").strip(), digit_map)

    if not value_text:
        return False

    normalized_value = normalize_label_value(value_text)
    normalized_message = normalize_label_value(message_text)
    if normalized_value and normalized_value in normalized_message:
        return True

    value_digits = re.sub(r"\D+", "", value_text)
    if value_digits and value_digits in re.sub(r"\D+", "", message_text):
        return True

    return False


def sanitize_semantic_extracted_value(
    field: Dict[str, Any],
    value: Any,
    message: str,
    agent_config: Dict[str, Any],
) -> str:
    """
    Validate and clean semantic extraction output using field-level config.

    This prevents the extractor from copying values from known variables/context
    instead of the latest user message. It is generic: every regex, cleanup
    pattern, and presence requirement is configured per field in domain_bundle.
    """
    if not isinstance(field, dict):
        return ""

    text = str(value or "").strip()
    if not text:
        return ""

    normalization = agent_config.get("normalization", {}) or {}
    digit_map = normalization.get("digit_map", {}) if isinstance(normalization, dict) else {}
    text = graph_normalize_digits(text, digit_map)

    cleanup_patterns = field.get("cleanup_patterns", [])
    if isinstance(cleanup_patterns, list):
        for pattern in cleanup_patterns:
            pattern_text = str(pattern or "").strip()
            if not pattern_text:
                continue
            try:
                text = re.sub(pattern_text, " ", text, flags=re.IGNORECASE).strip()
            except re.error:
                continue

    text = re.sub(r"[\s،,.!?؟:;]+$", "", text)
    text = re.sub(r"^[\s:：\-–—ـ]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return ""

    reject_value_sources = (
        field.get("reject_value_sources")
        or field.get("reject_if_value_sources")
        or field.get("blocked_value_sources")
        or []
    )
    if reject_value_sources and value_matches_configured_sources(text, agent_config, reject_value_sources):
        return ""

    if field.get("value_must_appear_in_message") is True:
        if not semantic_value_present_in_latest_message(text, message, agent_config):
            return ""

    try:
        min_digits = int(field.get("min_digit_count", field.get("min_digits", 0)) or 0)
    except Exception:
        min_digits = 0
    try:
        min_letters = int(field.get("min_letter_count", field.get("min_letters", 0)) or 0)
    except Exception:
        min_letters = 0

    if min_digits > 0 and len(re.findall(r"\d", text)) < min_digits:
        return ""
    if min_letters > 0 and len(re.findall(r"[A-Za-z\u0600-\u06FF]", text)) < min_letters:
        return ""

    reject_regexes = field.get("reject_if_regex", [])
    if isinstance(reject_regexes, str):
        reject_regexes = [reject_regexes]
    if isinstance(reject_regexes, list):
        for pattern in reject_regexes:
            pattern_text = str(pattern or "").strip()
            if not pattern_text:
                continue
            try:
                if re.search(pattern_text, text):
                    return ""
            except re.error:
                return ""

    accept_regex = str(field.get("accept_if_regex") or "").strip()
    if accept_regex:
        try:
            if not re.search(accept_regex, text):
                return ""
        except re.error:
            return ""

    try:
        min_length = int(field.get("min_length", 1) or 1)
    except Exception:
        min_length = 1
    try:
        max_length = int(field.get("max_length", 500) or 500)
    except Exception:
        max_length = 500

    if len(text) < min_length or len(text) > max_length:
        return ""

    return text


def build_last_offered_options_from_variables(
    variables: Dict[str, Any],
    tool_result: Dict[str, Any],
    agent_config: Dict[str, Any]
) -> Dict[str, Any]:
    cfg = get_smartness_config(agent_config, "last_offered_options")
    if not cfg.get("enabled", False) or not isinstance(variables, dict):
        return {}

    rules = cfg.get("result_rules", [])
    if not isinstance(rules, list):
        return {}

    answer_draft = ""
    if isinstance(tool_result, dict):
        answer_draft = str(tool_result.get("answer_draft") or "").strip()

    for rule in rules:
        if not isinstance(rule, dict) or rule.get("enabled", True) is False:
            continue

        result_path = str(rule.get("result_path") or "").strip()
        if not result_path:
            continue

        value = deep_get(variables, result_path, None)
        if value in [None, "", [], {}]:
            continue

        if isinstance(value, list):
            count = len(value)
            first_value = value[0] if count == 1 else None
        else:
            count = 1
            first_value = value

        return {
            "type": str(rule.get("type") or result_path),
            "count": count,
            "value": first_value,
            "source_path": result_path,
            "answer_draft": answer_draft,
        }

    return {}


# ── SEMANTIC EXTRACTION SAFETY GATES (CONFIG-DRIVEN) ────────────────────────

def regex_list_matches(text: str, patterns: Any) -> bool:
    if isinstance(patterns, str):
        patterns = [patterns]
    if not isinstance(patterns, list):
        return False
    for pattern in patterns:
        pattern_text = str(pattern or "").strip()
        if not pattern_text:
            continue
        try:
            if re.search(pattern_text, str(text or ""), flags=re.IGNORECASE):
                return True
        except re.error:
            continue
    return False


def semantic_sources_match(message: str, agent_config: Dict[str, Any], sources: Any) -> bool:
    if not sources:
        return False
    if isinstance(sources, dict):
        sources = [sources]
    if not isinstance(sources, list):
        return False
    return message_matches_configured_sources(message, agent_config, sources)


def semantic_extraction_message_blocked(
    cfg: Dict[str, Any],
    message: str,
    agent_config: Dict[str, Any],
) -> bool:
    """
    Config-driven global extraction gate.

    Use this for turns like hold/delay/help/control messages where semantic LLM
    extraction is more likely to copy context than extract a new user value.
    """
    if not isinstance(cfg, dict):
        return False
    if semantic_sources_match(message, agent_config, cfg.get("blocked_message_sources")):
        return True
    if semantic_sources_match(message, agent_config, cfg.get("skip_when_message_sources")):
        return True
    if regex_list_matches(message, cfg.get("blocked_message_regex")):
        return True
    if regex_list_matches(message, cfg.get("skip_when_message_regex")):
        return True
    return False


def semantic_field_message_allowed(
    field: Dict[str, Any],
    message: str,
    agent_config: Dict[str, Any],
) -> bool:
    """
    Field-level latest-message guardrails.

    Python owns only generic mechanics. The domain bundle defines which phrase
    sources or regexes should allow/block each configured field.
    """
    if not isinstance(field, dict):
        return False
    if semantic_sources_match(message, agent_config, field.get("blocked_message_sources")):
        return False
    if semantic_sources_match(message, agent_config, field.get("skip_when_message_sources")):
        return False
    if regex_list_matches(message, field.get("blocked_message_regex")):
        return False
    if regex_list_matches(message, field.get("skip_when_message_regex")):
        return False

    required_sources = field.get("required_message_sources")
    if required_sources and not semantic_sources_match(message, agent_config, required_sources):
        return False

    required_regex = field.get("required_message_regex")
    if required_regex and not regex_list_matches(message, required_regex):
        return False

    any_sources = field.get("required_any_message_sources")
    any_regex = field.get("required_any_message_regex")
    if any_sources or any_regex:
        if not semantic_sources_match(message, agent_config, any_sources) and not regex_list_matches(message, any_regex):
            return False

    return True


def semantic_field_context_conditions_met(
    field: Dict[str, Any],
    variables: Dict[str, Any],
    manifest: Dict[str, Any],
) -> bool:
    """
    Reuse the same configured activation predicates as field_is_required_now,
    but do not treat an existing value as a reason to stop. This allows a later
    explicit user correction to repair a bad/partial prior extraction.
    """
    if not isinstance(field, dict):
        return False

    if field.get("always_required", False):
        return True

    conditions_met: List[bool] = []

    stage_path = str(field.get("required_when_stage_path") or "").strip()
    required_stages = field.get("required_when_stages", [])
    if stage_path and isinstance(required_stages, list) and required_stages:
        current_stage = str(deep_get(variables, stage_path) or "").strip()
        stage_values = {str(item or "").strip() for item in required_stages if str(item or "").strip()}
        conditions_met.append(bool(current_stage and current_stage in stage_values))

    path_conditions = field.get("required_when_paths", [])
    if isinstance(path_conditions, list):
        for condition in path_conditions:
            if not isinstance(condition, dict):
                continue
            check_path = str(condition.get("path") or "").strip()
            if not check_path:
                continue
            value = deep_get(variables, check_path)
            if condition.get("must_be_present", False):
                conditions_met.append(value not in [None, "", [], {}])
            if condition.get("must_be_absent", False):
                conditions_met.append(value in [None, "", [], {}])
            if "equals" in condition:
                expected = condition.get("equals")
                if isinstance(expected, list):
                    conditions_met.append(value in expected)
                else:
                    conditions_met.append(value == expected)
            not_in_values = condition.get("not_in", [])
            if isinstance(not_in_values, list) and not_in_values:
                normalized_value = str(value or "").strip().lower()
                blocked = {str(item or "").strip().lower() for item in not_in_values}
                conditions_met.append(normalized_value not in blocked)

    manifest_conditions = field.get("required_when_manifest", {})
    if isinstance(manifest_conditions, dict):
        for manifest_key, expected_value in manifest_conditions.items():
            key_text = str(manifest_key or "").strip()
            if not key_text:
                continue
            actual = manifest.get(key_text)
            if isinstance(expected_value, list):
                conditions_met.append(actual in expected_value)
            else:
                conditions_met.append(actual == expected_value)

    if not conditions_met:
        return True
    return all(conditions_met)


def semantic_field_should_run(
    field: Dict[str, Any],
    variables: Dict[str, Any],
    manifest: Dict[str, Any],
    message: str,
    agent_config: Dict[str, Any],
) -> bool:
    target_path = str((field or {}).get("target_path") or "").strip()
    if not target_path:
        return False
    if not semantic_field_message_allowed(field, message, agent_config):
        return False

    existing = deep_get(variables, target_path, None)
    if existing in [None, "", [], {}]:
        return field_is_required_now(field, variables, manifest, agent_config)

    if not bool(field.get("allow_update_from_latest_message") or field.get("allow_correction_from_latest_message")):
        return False

    return semantic_field_context_conditions_met(field, variables, manifest)


def value_matches_configured_sources(
    value: str,
    agent_config: Dict[str, Any],
    sources: Any,
) -> bool:
    return semantic_sources_match(str(value or ""), agent_config, sources)


def semantic_extraction_node(state: AgentState):
    """
    Config-driven semantic variable extraction.

    Runs only when semantic_variable_extraction.enabled=true. Each configured
    field owns its own target path, conditions, prompt wording, examples,
    validation description, and ask-if-missing text.
    """
    agent_config = state.get("agent_config", {}) or {}

    if not semantic_extraction_is_enabled(agent_config):
        return {}

    cfg = get_semantic_extraction_config(agent_config)
    fields = cfg.get("fields", [])

    if not isinstance(fields, list) or not fields:
        return {}

    variables = state.get("variables", {}) or {}
    manifest = state.get("manifest", {}) or {}
    message = last_user_message(state)

    if not str(message or "").strip():
        return {}

    if semantic_extraction_message_blocked(cfg, message, agent_config):
        return {}

    required_fields = [
        field
        for field in fields
        if isinstance(field, dict)
        and semantic_field_should_run(field, variables, manifest, message, agent_config)
    ]

    if not required_fields:
        return {}

    extracted_updates: Dict[str, Any] = {}
    missing_asks: List[str] = []

    batch_values = run_batch_semantic_extraction(
        required_fields=required_fields,
        message=message,
        variables=variables,
        agent_config=agent_config,
    )
    if batch_values:
        field_by_id = {str(field.get("id") or ""): field for field in required_fields if isinstance(field, dict)}
        for field_id, value in batch_values.items():
            field = field_by_id.get(str(field_id))
            if not field:
                continue
            target = str(field.get("target_path") or "").strip()
            if target and value:
                clean_value = sanitize_semantic_extracted_value(field, value, message, agent_config)
                if clean_value:
                    extracted_updates[target] = clean_value

    try:
        max_workers = int(cfg.get("max_parallel_extractions", 3) or 3)
    except Exception:
        max_workers = 3
    max_workers = max(1, min(max_workers, len(required_fields)))

    def extract_one(field: Dict[str, Any]):
        target = str(field.get("target_path") or "").strip()
        ask = str(field.get("ask_if_missing") or "").strip()

        if not target:
            return field, target, None, ask

        if target in extracted_updates:
            return field, target, extracted_updates.get(target), ask

        value = run_single_field_extraction(
            field=field,
            message=message,
            variables=variables,
            agent_config=agent_config,
        )

        return field, target, value, ask

    if len(required_fields) > 1 and max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {
                executor.submit(extract_one, field): field
                for field in required_fields
            }
            for future in as_completed(future_map):
                try:
                    field, target, value, ask = future.result(timeout=20)
                    if value and target:
                        clean_value = sanitize_semantic_extracted_value(field, value, message, agent_config)
                        if clean_value:
                            extracted_updates[target] = clean_value
                        elif ask:
                            missing_asks.append(ask)
                    elif ask:
                        missing_asks.append(ask)
                except Exception:
                    continue
    else:
        for field in required_fields:
            field, target, value, ask = extract_one(field)
            if value and target:
                clean_value = sanitize_semantic_extracted_value(field, value, message, agent_config)
                if clean_value:
                    extracted_updates[target] = clean_value
                elif ask:
                    missing_asks.append(ask)
            elif ask:
                missing_asks.append(ask)

    if not extracted_updates and not missing_asks:
        return {}

    result: Dict[str, Any] = {}

    if extracted_updates:
        # Semantic extraction is a configured executor step, not a free-form
        # manifest update. It must be allowed to write configured target paths
        # even when those paths are protected from the manifest by
        # source_of_truth_variables/source_of_truth_prefixes.
        extracted_variables = apply_subagent_variable_patch(
            variables,
            prepare_variable_updates_for_patch(extracted_updates),
            [],
            assistant_config=agent_config,
        )
        extracted_variables = validate_and_heal_variables(extracted_variables, state.get("schema", {}) or {}, agent_config)
        result["variables"] = extracted_variables
        result["variable_changes_this_turn"] = compute_variable_changes(variables, extracted_variables, agent_config)

    if missing_asks and not extracted_updates:
        first_field = required_fields[0]
        manifest_patch = dict(manifest)
        brief = dict(manifest_patch.get("response_brief", {}) or {})
        if not str(brief.get("next_move") or "").strip():
            brief["next_move"] = missing_asks[0]

        manifest_patch["response_brief"] = brief
        manifest_patch["should_ask_question"] = True
        manifest_patch["question_goal"] = str(
            first_field.get("question_goal")
            or f"collect_{first_field.get('id', 'field')}"
        )
        result["manifest"] = manifest_patch

    return result


def should_route_to_semantic_extraction(state: AgentState) -> bool:
    agent_config = state.get("agent_config", {}) or {}

    if not semantic_extraction_is_enabled(agent_config):
        return False

    manifest = state.get("manifest", {}) or {}

    return bool(
        manifest.get("needs_tool")
        or manifest_has_parallel_tool_requests(manifest)
        or active_deterministic_flow_subagent_id_from_state(state)
    )



def graph_normalize_digits(text: str, digit_map: Dict[str, str]) -> str:
    return "".join(digit_map.get(ch, ch) for ch in str(text or ""))


def graph_path_parts(path: str) -> List[str]:
    text = str(path or "").strip()
    if text.startswith("variables."):
        text = text[len("variables."):]
    return [part for part in text.split(".") if part]


def graph_strip_variables_prefix(path: str) -> str:
    text = str(path or "").strip()
    if text.startswith("variables."):
        return text[len("variables."):]
    return text


def graph_required_path_missing(variables: Dict[str, Any], path: str) -> bool:
    return deep_get(variables, graph_strip_variables_prefix(path), None) in [None, "", [], {}]


def graph_extract_pending_required_details_from_patterns(
    agent_config: Dict[str, Any],
    variables: Dict[str, Any],
    message: str,
) -> Dict[str, Any]:
    """
    Deterministically pre-extract configured missing required details before the
    booking executor runs.

    This is intentionally generic:
    - uses booking.required_before_create to know missing paths
    - uses booking.extraction_patterns to capture values
    - does not know field names, labels, Arabic phrases, or car-specific terms
    """
    booking_config = get_subagent_config(agent_config, "booking")
    if not isinstance(booking_config, dict) or not booking_config.get("enabled", False):
        return variables

    pending_path = str(booking_config.get("pending_booking_path") or "booking.pending")
    pending = deep_get(variables, pending_path, None)
    if not isinstance(pending, dict) or not pending:
        return variables

    patterns = booking_config.get("extraction_patterns", [])
    required_paths = booking_config.get("required_before_create", [])

    if not isinstance(patterns, list) or not isinstance(required_paths, list):
        return variables

    missing_paths = [
        graph_strip_variables_prefix(path)
        for path in required_paths
        if str(path or "").strip() and graph_required_path_missing(variables, str(path))
    ]

    if not missing_paths:
        return variables

    normalization = agent_config.get("normalization", {}) or {}
    digit_map = normalization.get("digit_map", {}) if isinstance(normalization, dict) else {}
    raw_message = str(message or "")
    digit_message = graph_normalize_digits(raw_message, digit_map)

    # v6.51: deterministic graph pattern extraction must obey the global
    # configured semantic-extraction message gate too. This keeps help,
    # hold/delay, and control-only turns from being captured as customer
    # details before the deterministic booking executor can answer them.
    try:
        semantic_cfg_for_message_gate = get_semantic_extraction_config(agent_config)
        if semantic_extraction_message_blocked(semantic_cfg_for_message_gate, raw_message, agent_config):
            return variables
    except Exception:
        pass

    # v6.47: pattern extraction must obey the same configured field gates as
    # semantic extraction. This prevents disabled/broad regexes or control turns
    # from writing protected customer-detail fields before the deterministic
    # booking executor runs. Field names/markers stay in domain_bundle.json.
    semantic_field_by_target: Dict[str, Dict[str, Any]] = {}
    try:
        semantic_cfg = get_semantic_extraction_config(agent_config)
        for configured_field in semantic_cfg.get("fields", []) or []:
            if not isinstance(configured_field, dict):
                continue
            target_path = graph_strip_variables_prefix(str(configured_field.get("target_path") or "").strip())
            if target_path:
                semantic_field_by_target[target_path] = configured_field
    except Exception:
        semantic_field_by_target = {}

    updates: Dict[str, Any] = {}

    for missing_path in missing_paths:
        for item in patterns:
            if not isinstance(item, dict):
                continue
            if item.get("enabled", True) is False:
                continue

            target = graph_strip_variables_prefix(
                str(
                    item.get("variable")
                    or item.get("path")
                    or item.get("target_path")
                    or ""
                ).strip()
            )
            when_missing = graph_strip_variables_prefix(str(item.get("when_missing") or "").strip())

            if target != missing_path and when_missing != missing_path:
                continue

            field_config = semantic_field_by_target.get(missing_path, {})
            if field_config and not semantic_field_message_allowed(field_config, raw_message, agent_config):
                continue

            pattern_text = str(item.get("regex") or item.get("pattern") or "").strip()
            if not pattern_text:
                continue

            try:
                group_index = int(item.get("group", 1))
            except Exception:
                group_index = 1

            candidates: List[str] = []
            for haystack in [raw_message, digit_message]:
                if not haystack:
                    continue
                try:
                    match = re.search(pattern_text, haystack, flags=re.IGNORECASE)
                except re.error:
                    continue

                if not match:
                    continue

                try:
                    candidate = match.group(group_index).strip() if match.groups() else match.group(0).strip()
                except Exception:
                    candidate = match.group(0).strip()

                candidate = graph_normalize_digits(candidate, digit_map)
                candidate = re.sub(r"[،,.!?؟:;]+", " ", candidate)
                candidate = re.sub(r"\s+", " ", candidate).strip()

                if candidate:
                    candidates.append(candidate)

            if not candidates:
                continue

            candidates.sort(key=lambda value: (len(value), bool(re.search(r"\d", value))), reverse=True)
            value = candidates[0]
            if field_config:
                value = sanitize_semantic_extracted_value(field_config, value, raw_message, agent_config)
            if not value:
                continue
            updates[missing_path] = value

            if missing_path.startswith("customer_profile."):
                field_name = missing_path.split(".", 1)[1].strip()
                if field_name:
                    updates.setdefault(f"booking.customer_profile.{field_name}", value)
                    updates.setdefault(f"booking.pending.customer_profile.{field_name}", value)

            break

    if not updates:
        return variables

    return apply_subagent_variable_patch(
        variables,
        updates,
        [],
        assistant_config=agent_config,
    )


def configured_active_booking_stage_values(
    agent_config: Dict[str, Any],
    booking_config: Dict[str, Any],
) -> List[str]:
    """
    Collect all configured booking-flow stages that should be owned by the
    deterministic booking executor while a pending booking exists.

    This is intentionally config-driven: the graph does not know domain phrases
    or field names. It reads stage values from booking config and routing
    guardrails so confirmation, slot-selection, and customer-detail continuations
    stay in the executor instead of falling back to a simple response.
    """
    stages: List[str] = []

    def add_many(value: Any) -> None:
        if isinstance(value, list):
            for item in value:
                text = str(item or "").strip()
                if text:
                    stages.append(text)

    add_many(booking_config.get("extraction_active_stages", []))
    add_many(booking_config.get("active_request_stages", []))

    stages_map = booking_config.get("stages", {})
    if isinstance(stages_map, dict):
        for value in stages_map.values():
            text = str(value or "").strip()
            if text:
                stages.append(text)

    # Common bundle shape for active booking routing. The graph only reads the
    # configured values; it does not hardcode what those stages mean.
    active_route_rule = (
        agent_config.get("routing_guardrails", {})
        .get("active_booking_customer_detail_routing", {})
    )
    if isinstance(active_route_rule, dict):
        add_many(active_route_rule.get("active_stage_values", []))

    # Allow future tenants to expose additional stage lists in config without
    # needing a graph code branch per domain.
    add_many(booking_config.get("executor_owned_stages", []))
    add_many(booking_config.get("pending_booking_executor_stages", []))

    return append_unique([], stages)


def booking_pending_requires_executor(
    agent_config: Dict[str, Any],
    variables: Dict[str, Any],
    message: str = "",
) -> bool:
    """
    Config-driven booking execution lock.

    A pending booking is the strongest signal, but it is not the only safe
    signal. Some older or partially-migrated states can preserve the booking
    stage/confirmation/root appointment fields while dropping the nested
    pending object. In that case the booking executor must still own the turn
    so it can rebuild pending state and persist customer details.

    The recovery paths are configured in the assistant bundle; Python does not
    embed tenant phrases, fields, dates, slots, or test data.
    """
    booking_config = get_subagent_config(agent_config, "booking")
    if not isinstance(booking_config, dict) or not booking_config.get("enabled", False):
        return False

    # Do not keep forcing booking after a successful booking id/status exists.
    completion = booking_config.get("booking_completion", {})
    if not isinstance(completion, dict):
        completion = {}

    id_paths = completion.get("id_paths", ["visit_id"])
    if not isinstance(id_paths, list):
        id_paths = ["visit_id"]

    for path in id_paths:
        if deep_get(variables, str(path or ""), None) not in [None, "", [], {}]:
            return False

    status_paths = completion.get("status_paths", ["booking_status"])
    if not isinstance(status_paths, list):
        status_paths = ["booking_status"]

    completed_statuses = {
        str(item or "").strip().lower()
        for item in completion.get("completed_statuses", ["confirmed", "booked"])
        if str(item or "").strip()
    }

    for path in status_paths:
        value = deep_get(variables, str(path or ""), None)
        if str(value or "").strip().lower() in completed_statuses:
            return False

    pending_path = str(booking_config.get("pending_booking_path") or "booking.pending")
    pending = deep_get(variables, pending_path, None)
    pending_exists = isinstance(pending, dict) and bool(pending)

    stage_path = str(booking_config.get("stage_path") or "booking.stage")
    stage = str(deep_get(variables, stage_path, "") or "").strip()

    active_stages = configured_active_booking_stage_values(
        agent_config=agent_config,
        booking_config=booking_config,
    )

    active_stage_set = {str(item or "").strip() for item in active_stages if str(item or "").strip()}
    stage_requires_executor = bool(stage and (not active_stage_set or stage in active_stage_set))

    if pending_exists and stage_requires_executor:
        return True

    recovery = booking_config.get("executor_lock_recovery", {})
    if not isinstance(recovery, dict):
        recovery = {}

    recovery_enabled = recovery.get("enabled", True) is not False

    if recovery_enabled and stage_requires_executor:
        context_paths = recovery.get("context_paths", [])
        if not isinstance(context_paths, list) or not context_paths:
            context_paths = [
                booking_config.get("pending_booking_path", "booking.pending"),
                booking_config.get("appointment_date_path", "appointment_date"),
                booking_config.get("date_text_path", "date_text"),
                "appointment_time",
                "selected_branch",
                "location_branch",
                "nearest_branch",
            ]

        has_recoverable_context = any(
            deep_get(variables, str(path or ""), None) not in [None, "", [], {}]
            for path in context_paths
        )

        confirmation_paths = recovery.get("confirmation_paths", [])
        if not isinstance(confirmation_paths, list) or not confirmation_paths:
            confirmation_paths = ["customer_confirmed_booking"]

        has_confirmation_context = any(
            deep_get(variables, str(path or ""), None) is True
            for path in confirmation_paths
        )

        if has_recoverable_context or has_confirmation_context:
            return True

    normalization = agent_config.get("normalization", {}) or {}
    detail_marker_paths = (
        agent_config.get("routing_guardrails", {})
        .get("active_booking_customer_detail_routing", {})
        .get("detail_marker_paths", [])
    )
    if isinstance(detail_marker_paths, list):
        markers = collect_configured_phrases_deep(agent_config, detail_marker_paths)
        if markers and matches_any(message, markers, normalization):
            return bool(pending_exists or stage_requires_executor)

    return False


def collect_strings_deep(value: Any) -> List[str]:
    """
    Recursively collect strings from a config value.

    This is needed because some config paths point to lists of field objects,
    not plain phrase arrays. The graph should still be able to read configured
    markers without hardcoding any field names.
    """
    output: List[str] = []

    if isinstance(value, str):
        text = value.strip()
        if text:
            output.append(text)
        return output

    if isinstance(value, list):
        for item in value:
            output.extend(collect_strings_deep(item))
        return output

    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).endswith("_markers") or str(key) in {
                "markers",
                "phrases",
                "detail_signal_phrases",
                "trigger_phrases",
                "strong_phrases",
            }:
                output.extend(collect_strings_deep(item))
            elif isinstance(item, (list, dict)):
                output.extend(collect_strings_deep(item))
        return output

    return output


def collect_configured_phrases_deep(
    agent_config: Dict[str, Any],
    paths: List[str]
) -> List[str]:
    phrases: List[str] = []

    for path in paths or []:
        value = get_config_path_value(agent_config, str(path), [])
        phrases.extend(collect_strings_deep(value))

    return append_unique([], phrases)




def completed_flow_closing_subagent_id(
    agent_config: Dict[str, Any],
    variables: Dict[str, Any],
    message: str,
) -> str:
    """
    Config-driven routing for short closing/thanks messages after a flow is
    already complete. This lets the deterministic executor return the configured
    short closing label instead of the response LLM repeating operational facts.
    """
    guardrails = agent_config.get("routing_guardrails", {})
    if not isinstance(guardrails, dict):
        return ""

    rule = guardrails.get("completed_flow_closing") or guardrails.get("completed_flow_closing_routing")
    if not isinstance(rule, dict) or rule.get("enabled", False) is False:
        return ""

    target = unify_subagent_id(str(rule.get("target_subagent_id") or rule.get("subagent_id") or "").strip())
    if not target:
        return ""

    id_paths = rule.get("completion_id_paths", [])
    if not isinstance(id_paths, list):
        id_paths = []

    status_paths = rule.get("completion_status_paths", [])
    if not isinstance(status_paths, list):
        status_paths = []

    completed_statuses = {
        str(item or "").strip().lower()
        for item in rule.get("completed_statuses", []) or []
        if str(item or "").strip()
    }

    completed = False
    for path in id_paths:
        if deep_get(variables, str(path or ""), None) not in [None, "", [], {}]:
            completed = True
            break

    if not completed:
        for path in status_paths:
            value = str(deep_get(variables, str(path or ""), "") or "").strip().lower()
            if value and (not completed_statuses or value in completed_statuses):
                completed = True
                break

    if not completed:
        return ""

    sources = rule.get("message_sources") or rule.get("closing_message_sources") or []
    regexes = rule.get("message_regex") or rule.get("closing_message_regex") or []

    if sources and semantic_sources_match(message, agent_config, sources):
        return target
    if regexes and regex_list_matches(message, regexes):
        return target

    return ""

def get_active_configured_flow_subagent_id(
    agent_config: Dict[str, Any],
    variables: Dict[str, Any],
    manifest: Optional[Dict[str, Any]] = None,
    message: str = ""
) -> str:
    """
    Config-driven deterministic-flow lock.

    Any routing_guardrails entry can force a target subagent while configured
    state paths have active values. This lets deterministic executors handle
    state transitions even when the manifest says needs_tool=false.
    """
    guardrails = agent_config.get("routing_guardrails", {})
    if not isinstance(guardrails, dict):
        return ""

    closing_target = completed_flow_closing_subagent_id(
        agent_config=agent_config,
        variables=variables,
        message=message,
    )
    if closing_target:
        return closing_target

    continuation_target = get_active_post_tool_required_input_continuation_subagent_id(
        agent_config=agent_config,
        variables=variables,
    )
    if continuation_target:
        return continuation_target

    if booking_pending_requires_executor(
        agent_config=agent_config,
        variables=variables,
        message=message,
    ):
        return "booking"

    normalization = agent_config.get("normalization", {}) or {}
    manifest = manifest or {}

    for _, rule in guardrails.items():
        if not isinstance(rule, dict) or not rule.get("enabled", False):
            continue

        target_subagent = str(rule.get("target_subagent_id") or "").strip()
        if not target_subagent:
            continue

        stage_paths = rule.get("active_stage_paths", [])
        stage_values = {
            str(item or "").strip()
            for item in rule.get("active_stage_values", []) or []
            if str(item or "").strip()
        }

        if not isinstance(stage_paths, list) or not stage_paths:
            continue

        active = False

        for path in stage_paths:
            value = deep_get(variables, str(path or "").strip(), "")
            value_text = str(value or "").strip()

            if not value_text:
                continue

            if not stage_values or value_text in stage_values:
                active = True
                break

        if not active:
            continue

        must_not_select = {
            str(item or "").strip()
            for item in rule.get("must_not_select_subagents", []) or []
            if str(item or "").strip()
        }

        selected = str(manifest.get("selected_subagent_id") or "").strip()
        if selected in must_not_select:
            return target_subagent

        detail_marker_paths = rule.get("detail_marker_paths", [])
        if isinstance(detail_marker_paths, list) and detail_marker_paths:
            markers = collect_configured_phrases(agent_config, detail_marker_paths)
            if markers and matches_any(message, markers, normalization):
                return target_subagent

        if rule.get("force_for_active_stage", True):
            return target_subagent

    return ""


def apply_active_deterministic_flow_guardrails(
    manifest: Dict[str, Any],
    message: str,
    agent_config: Dict[str, Any],
    variables: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Patch the manifest when config says a deterministic subagent owns the active
    flow. This is not a tool-only concept; the executor may need to extract,
    merge, ask the next missing detail, or finalize the operation.
    """
    target = get_active_configured_flow_subagent_id(
        agent_config=agent_config,
        variables=variables,
        manifest=manifest,
        message=message,
    )

    if not target:
        return manifest

    patched = dict(manifest or {})
    patched["selected_subagent_id"] = unify_subagent_id(target)
    patched["simple_response_mode"] = False
    patched["needs_tool"] = True
    patched["needs_subagent_reasoning"] = False
    patched["needs_quality_guard"] = True

    brief = patched.get("response_brief")
    if not isinstance(brief, dict):
        brief = {}

    brief["must_do"] = append_unique(brief.get("must_do", []), [
        "run the configured deterministic executor for this active flow",
        "preserve existing variables when adding new user details",
        "ask only for the next missing detail if something is still missing",
    ])
    brief["must_not_do"] = append_unique(brief.get("must_not_do", []), [
        "do not answer with simple response while this active flow is incomplete",
        "do not claim an external action was completed without a successful tool result",
    ])

    patched["response_brief"] = brief
    return patched


def active_deterministic_flow_subagent_id_from_state(state: AgentState) -> str:
    return get_active_configured_flow_subagent_id(
        agent_config=state.get("agent_config", {}) or {},
        variables=state.get("variables", {}) or {},
        manifest=state.get("manifest", {}) or {},
        message=last_user_message(state),
    )




def collect_subagent_configured_phrases(
    agent_config: Dict[str, Any],
    subagent_id: str,
    paths: List[str],
) -> List[str]:
    """
    Collect phrase lists from a configured subagent.

    This is intentionally config-driven: the caller supplies config paths and the
    phrase values live in domain_bundle.json, not in graph.py.
    """
    subagent_id = unify_subagent_id(str(subagent_id or "").strip())
    config = get_subagent_config(agent_config, subagent_id)
    phrases: List[str] = []

    for path in paths or []:
        value = get_config_path_value(config, str(path or "").strip(), [])

        if isinstance(value, list):
            phrases.extend([
                str(item)
                for item in value
                if str(item or "").strip()
            ])
        elif isinstance(value, str) and value.strip():
            phrases.append(value)

    return phrases


def message_matches_phrase_source(
    message: str,
    agent_config: Dict[str, Any],
    source: Dict[str, Any],
) -> bool:
    """
    Match latest user message against a configured phrase source.

    Supported source shapes:
    - {"subagent_id": "booking", "paths": ["availability_request_terms"]}
    - {"paths": ["assistant_level.path"]}
    - {"phrases": ["..."]}

    No user-facing or domain phrases are embedded here.
    """
    if not isinstance(source, dict):
        return False

    normalization = agent_config.get("normalization", {}) or {}
    phrases: List[str] = []

    direct_phrases = source.get("phrases", [])
    if isinstance(direct_phrases, list):
        phrases.extend([
            str(item)
            for item in direct_phrases
            if str(item or "").strip()
        ])

    paths = source.get("paths", [])
    if isinstance(paths, str):
        paths = [paths]
    if not isinstance(paths, list):
        paths = []

    subagent_id = str(source.get("subagent_id") or source.get("subagent") or "").strip()

    if subagent_id:
        phrases.extend(
            collect_subagent_configured_phrases(
                agent_config=agent_config,
                subagent_id=subagent_id,
                paths=[str(path) for path in paths],
            )
        )
    else:
        phrases.extend(collect_configured_phrases(agent_config, [str(path) for path in paths]))

    if not phrases:
        return False

    return matches_any(message, phrases, normalization)


def manifest_intent_texts(manifest: Dict[str, Any]) -> List[str]:
    texts: List[str] = []

    for key in [
        "user_intent",
        "conversation_stage",
        "workflow_stage",
        "response_strategy",
        "chained_subagent_reason",
    ]:
        value = str((manifest or {}).get(key) or "").strip()
        if value:
            texts.append(value)

    for item in (manifest or {}).get("detected_intents", []) or []:
        text = str(item or "").strip()
        if text:
            texts.append(text)

    for item in (manifest or {}).get("multi_intents", []) or []:
        if not isinstance(item, dict):
            continue

        for key in [
            "intent_id",
            "intent_type",
            "user_goal",
            "response_role",
            "selected_subagent_id",
            "requested_tool_name",
        ]:
            value = str(item.get(key) or "").strip()
            if value:
                texts.append(value)

    return texts


def manifest_matches_any_intent_label(
    manifest: Dict[str, Any],
    labels: List[str],
    normalization: Dict[str, Any],
) -> bool:
    if not isinstance(labels, list) or not labels:
        return True

    normalized_labels = [
        normalize_label
        for normalize_label in [
            re.sub(r"\s+", " ", str(label or "").strip().lower())
            for label in labels
        ]
        if normalize_label
    ]

    if not normalized_labels:
        return True

    for text in manifest_intent_texts(manifest):
        normalized_text = re.sub(r"\s+", " ", str(text or "").strip().lower())
        if not normalized_text:
            continue

        for label in normalized_labels:
            if label == normalized_text or label in normalized_text or normalized_text in label:
                return True

    return False


def message_has_date_for_subagent(
    message: str,
    agent_config: Dict[str, Any],
    subagent_id: str,
) -> bool:
    """
    Ask the configured subagent date extractor whether this message contains a
    date expression. This avoids graph.py carrying domain/date phrase lists.
    """
    subagent_id = unify_subagent_id(str(subagent_id or "").strip())
    executor = SUBAGENT_EXECUTORS.get(subagent_id)
    config = get_subagent_config(agent_config, subagent_id)
    normalization = agent_config.get("normalization", {}) or {}

    if executor and hasattr(executor, "extract_date_text"):
        try:
            date_text = executor.extract_date_text(message, config, normalization)
            return bool(str(date_text or "").strip())
        except Exception:
            pass

    date_sources = agent_config.get("date_detection_phrase_sources", [])
    if isinstance(date_sources, list):
        for source in date_sources:
            if isinstance(source, dict) and message_matches_phrase_source(message, agent_config, source):
                return True

    return False


def get_dependent_intent_chain_rules(agent_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Return configured dependent-intent chain rules.

    Preferred config keys:
    - dependent_intent_chains
    - sequential_intent_chains

    If the bundle has not yet been updated, derive one safe rule from configured
    booking/location subagent phrase lists. The derived rule contains no branch,
    slot, date, or user phrase values; it only says:
    selected location flow + booking availability phrase + booking date extractor
    should run booking after location in the same graph turn.
    """
    raw = agent_config.get("dependent_intent_chains")
    if not isinstance(raw, list):
        raw = agent_config.get("sequential_intent_chains")

    rules = [item for item in (raw or []) if isinstance(item, dict)] if isinstance(raw, list) else []

    if rules:
        return rules

    derived_enabled = get_config_bool(
        agent_config,
        "routing_guardrails.derive_dependent_chains_from_subagent_config",
        True,
    )

    if not derived_enabled:
        return []

    location_config = get_subagent_config(agent_config, "location")
    booking_config = get_subagent_config(agent_config, "booking")

    if not location_config or not booking_config:
        return []

    return [{
        "id": "derived_location_then_booking_availability",
        "enabled": True,
        "primary_subagent_id": "location",
        "chained_subagent_id": "booking",
        "only_when_selected_subagent_is_primary": True,
        "requires_empty_chained_subagent": True,
        "requires_date_from_subagent": "booking",
        "requires_any_message_phrase_sources": [
            {
                "subagent_id": "booking",
                "paths": [
                    "availability_request_terms",
                    "trigger_phrases",
                ],
            }
        ],
        "set_needs_tool": True,
        "set_quality_guard": True,
        "set_style_repair": True,
        "response_strategy_append": (
            "The latest user turn combines a configured location lookup with a configured "
            "availability request. Run the primary subagent first, then run the chained "
            "subagent in the same turn using the updated variables. Do not answer with a "
            "waiting/acknowledgement message before the chained result is available."
        ),
        "response_brief": {
            "next_move": "Run the dependent chained flow in the same turn.",
            "must_do": [
                "execute the configured chained subagent after the primary subagent",
                "use the primary subagent result as input to the chained subagent",
                "answer only after the chained result is available",
            ],
            "must_not_do": [
                "do not send a waiting message when the chained tool can run now",
                "do not ask the user to repeat information already present in the same message",
            ],
        },
    }]


def dependent_chain_rule_matches(
    rule: Dict[str, Any],
    manifest: Dict[str, Any],
    message: str,
    agent_config: Dict[str, Any],
    variables: Dict[str, Any],
) -> bool:
    if not isinstance(rule, dict) or rule.get("enabled", True) is False:
        return False

    primary_id = unify_subagent_id(str(rule.get("primary_subagent_id") or "").strip())
    chained_id = unify_subagent_id(str(rule.get("chained_subagent_id") or "").strip())

    if not primary_id or not chained_id:
        return False

    selected_id = unify_subagent_id(str((manifest or {}).get("selected_subagent_id") or "").strip())
    existing_chain = unify_subagent_id(str((manifest or {}).get("chained_subagent_id") or "").strip())

    if rule.get("requires_empty_chained_subagent", True) and existing_chain:
        return False

    if rule.get("only_when_selected_subagent_is_primary", True) and selected_id != primary_id:
        return False

    blocked_stages = rule.get("blocked_stage_values", [])
    stage_paths = rule.get("stage_paths", [])
    if isinstance(blocked_stages, list) and isinstance(stage_paths, list):
        blocked_stage_set = {str(item or "").strip() for item in blocked_stages if str(item or "").strip()}
        for path in stage_paths:
            value = str(deep_get(variables, str(path or "").strip(), "") or "").strip()
            if value and value in blocked_stage_set:
                return False

    normalization = agent_config.get("normalization", {}) or {}

    intent_labels = rule.get("requires_any_intent_labels", [])
    if isinstance(intent_labels, list) and intent_labels:
        if not manifest_matches_any_intent_label(manifest, intent_labels, normalization):
            return False

    all_sources = rule.get("requires_all_message_phrase_sources", [])
    if isinstance(all_sources, list):
        for source in all_sources:
            if not isinstance(source, dict):
                continue
            if not message_matches_phrase_source(message, agent_config, source):
                return False

    any_sources = rule.get("requires_any_message_phrase_sources", [])
    if isinstance(any_sources, list) and any_sources:
        if not any(
            isinstance(source, dict) and message_matches_phrase_source(message, agent_config, source)
            for source in any_sources
        ):
            return False

    date_subagent = str(rule.get("requires_date_from_subagent") or "").strip()
    if date_subagent and not message_has_date_for_subagent(message, agent_config, date_subagent):
        return False

    return True


def apply_dependent_intent_chain_guardrails(
    manifest: Dict[str, Any],
    message: str,
    agent_config: Dict[str, Any],
    variables: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Config-driven deterministic repair for sequential multi-intent turns.

    Example class of turn:
      "Find the nearest branch and show available appointments tomorrow."

    If the LLM planner misses the dependency unless the user adds a connector
    word like "there", this guard repairs the manifest so the graph runs:
      primary_subagent -> chained_subagent
    in the same turn.

    Domain phrases remain in config/subagent configs; this function only uses
    configured phrase sources and configured subagent date extraction.
    """
    patched = dict(manifest or {})

    for rule in get_dependent_intent_chain_rules(agent_config):
        if not dependent_chain_rule_matches(
            rule=rule,
            manifest=patched,
            message=message,
            agent_config=agent_config,
            variables=variables,
        ):
            continue

        primary_id = unify_subagent_id(str(rule.get("primary_subagent_id") or "").strip())
        chained_id = unify_subagent_id(str(rule.get("chained_subagent_id") or "").strip())

        patched["selected_subagent_id"] = primary_id
        patched["chained_subagent_id"] = chained_id
        patched["chained_subagent_reason"] = str(
            rule.get("reason")
            or "Configured dependent intent chain matched."
        )

        if rule.get("set_needs_tool", True):
            patched["needs_tool"] = True

        patched["simple_response_mode"] = False
        patched["needs_subagent_reasoning"] = bool(rule.get("set_subagent_reasoning", False))
        patched["needs_quality_guard"] = bool(rule.get("set_quality_guard", True))
        patched["needs_style_repair"] = bool(rule.get("set_style_repair", True))

        strategy_append = str(rule.get("response_strategy_append") or "").strip()
        if strategy_append:
            previous_strategy = str(patched.get("response_strategy") or "").strip()
            patched["response_strategy"] = (
                f"{previous_strategy}\n{strategy_append}".strip()
                if previous_strategy
                else strategy_append
            )

        rule_brief = rule.get("response_brief", {})
        if isinstance(rule_brief, dict):
            brief = patched.get("response_brief")
            if not isinstance(brief, dict):
                brief = {}

            for key in ["tone", "language", "reply_length", "next_move"]:
                value = str(rule_brief.get(key) or "").strip()
                if value:
                    brief[key] = value

            brief["must_do"] = append_unique(
                brief.get("must_do", []),
                [str(item) for item in rule_brief.get("must_do", []) or []],
            )
            brief["must_not_do"] = append_unique(
                brief.get("must_not_do", []),
                [str(item) for item in rule_brief.get("must_not_do", []) or []],
            )
            patched["response_brief"] = brief

        synthesis = patched.get("response_synthesis")
        if not isinstance(synthesis, dict):
            synthesis = {}

        synthesis["dependent_chain_applied"] = {
            "rule_id": str(rule.get("id") or ""),
            "primary_subagent_id": primary_id,
            "chained_subagent_id": chained_id,
        }
        patched["response_synthesis"] = synthesis

        return patched

    return patched




# ── BREATHTAKING SMARTNESS RUNTIME HELPERS (CONFIG-DRIVEN) ──────────────────

def get_smartness_config(agent_config: Dict[str, Any], key: str, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Read smartness-layer config without domain hardcoding.

    Preferred locations:
    - assistant.smartness.<key>
    - assistant.<key>

    Python owns generic mechanics only. Phrases, thresholds, answer variants,
    field names, and policies belong in domain_bundle.json.
    """
    default = default or {}
    smartness = agent_config.get("smartness", {}) if isinstance(agent_config.get("smartness"), dict) else {}
    value = smartness.get(key)
    if isinstance(value, dict):
        return value
    value = agent_config.get(key)
    if isinstance(value, dict):
        return value
    return dict(default)


def smartness_enabled(agent_config: Dict[str, Any], key: str, default: bool = True) -> bool:
    cfg = get_smartness_config(agent_config, key)
    if "enabled" not in cfg:
        return default
    value = cfg.get("enabled")
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def normalize_label_value(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def get_nested_config(agent_config: Dict[str, Any], key: str) -> Any:
    smartness = agent_config.get("smartness", {}) if isinstance(agent_config.get("smartness"), dict) else {}
    if key in smartness:
        return smartness.get(key)
    return agent_config.get(key)


def compute_reply_length_from_message(message: str, agent_config: Dict[str, Any]) -> str:
    cfg = get_smartness_config(agent_config, "message_length_mirroring")
    thresholds = cfg.get("thresholds", {}) if isinstance(cfg.get("thresholds"), dict) else {}
    labels = cfg.get("labels", {}) if isinstance(cfg.get("labels"), dict) else {}

    try:
        very_short_max = int(thresholds.get("very_short_max_chars", 20) or 20)
        short_max = int(thresholds.get("short_max_chars", 60) or 60)
        medium_max = int(thresholds.get("medium_max_chars", 180) or 180)
    except Exception:
        very_short_max, short_max, medium_max = 20, 60, 180

    char_count = len(str(message or "").strip())

    if char_count < very_short_max:
        return str(labels.get("very_short") or "very_short")
    if char_count < short_max:
        return str(labels.get("short") or "short")
    if char_count < medium_max:
        return str(labels.get("medium") or "medium")
    return str(labels.get("detailed") or "detailed")


def apply_length_mirroring(manifest: Dict[str, Any], message: str, agent_config: Dict[str, Any]) -> Dict[str, Any]:
    if not smartness_enabled(agent_config, "message_length_mirroring", default=True):
        return manifest

    patched = dict(manifest or {})
    brief = dict(patched.get("response_brief", {}) or {})

    explicit_paths = get_smartness_config(agent_config, "message_length_mirroring").get("explicit_length_paths", [])
    explicit_values = []
    if isinstance(explicit_paths, list):
        for path in explicit_paths:
            value = graph_dotted_get(patched, str(path), None)
            if value not in [None, "", [], {}]:
                explicit_values.append(value)

    if not str(brief.get("reply_length") or "").strip() and not explicit_values:
        mirrored = compute_reply_length_from_message(message, agent_config)
        brief["reply_length"] = mirrored
        patched["reply_length"] = mirrored

    patched["response_brief"] = brief
    return patched


def normalize_emotion_history(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    output: List[str] = []
    for item in value:
        text = normalize_label_value(item)
        if text:
            output.append(text)
    return output


def compute_emotion_trajectory(current_emotion: str, prior_history: List[str], agent_config: Dict[str, Any]) -> str:
    cfg = get_smartness_config(agent_config, "emotion_arc_tracking")
    negative = {normalize_label_value(item) for item in cfg.get("negative_emotions", []) if normalize_label_value(item)}
    positive = {normalize_label_value(item) for item in cfg.get("positive_emotions", []) if normalize_label_value(item)}

    if not negative:
        negative = {"frustrated", "urgent", "skeptical", "angry", "upset", "confused"}
    if not positive:
        positive = {"excited", "happy", "satisfied"}

    current = normalize_label_value(current_emotion) or "neutral"
    window_size = graph_config_int({"emotion_arc_tracking": cfg}, "emotion_arc_tracking.window_size", 3, 2, 12)
    recent = prior_history[-window_size:]

    if not recent:
        if current in negative:
            return "escalating"
        if current in positive:
            return "positive_momentum"
        return "stable"

    was_negative = any(item in negative for item in recent)
    last_was_negative = bool(recent and recent[-1] in negative)
    is_negative = current in negative
    is_positive = current in positive

    if was_negative and not is_negative:
        return "de-escalating"
    if not was_negative and is_negative:
        return "escalating"
    if last_was_negative and is_negative:
        return "persistently_frustrated"
    if is_positive:
        return "positive_momentum"
    return "stable"


def update_emotion_history(current_emotion: str, prior_history: List[str], agent_config: Dict[str, Any]) -> List[str]:
    cfg = get_smartness_config(agent_config, "emotion_arc_tracking")
    max_items = graph_config_int({"emotion_arc_tracking": cfg}, "emotion_arc_tracking.max_history", 8, 2, 50)
    emotion = normalize_label_value(current_emotion) or "neutral"
    return (prior_history + [emotion])[-max_items:]


def apply_emotion_arc_to_manifest(manifest: Dict[str, Any], prior_history: List[str], agent_config: Dict[str, Any]) -> Dict[str, Any]:
    if not smartness_enabled(agent_config, "emotion_arc_tracking", default=True):
        return manifest

    patched = dict(manifest or {})
    trajectory = compute_emotion_trajectory(
        current_emotion=str(patched.get("customer_emotion") or "neutral"),
        prior_history=prior_history,
        agent_config=agent_config,
    )
    patched["emotion_trajectory"] = trajectory

    guidance = agent_config.get("emotion_arc_guidance", {})
    if not isinstance(guidance, dict):
        guidance = {}
    instruction = str(guidance.get(trajectory) or "").strip()
    if instruction:
        brief = dict(patched.get("response_brief", {}) or {})
        brief["must_do"] = append_unique(brief.get("must_do", []), [instruction])
        patched["response_brief"] = brief

    return patched


def message_matches_configured_sources(message: str, agent_config: Dict[str, Any], sources: Any) -> bool:
    if isinstance(sources, dict):
        sources = [sources]
    if not isinstance(sources, list):
        return False
    return any(
        isinstance(source, dict) and message_matches_phrase_source(message, agent_config, source)
        for source in sources
    )


def apply_hesitation_detection(manifest: Dict[str, Any], message: str, agent_config: Dict[str, Any]) -> Dict[str, Any]:
    cfg = get_smartness_config(agent_config, "hesitation_detection")
    if not cfg.get("enabled", False):
        return manifest

    sources = cfg.get("signal_sources") or cfg.get("signals") or []
    direct_phrases = cfg.get("phrases", [])
    if isinstance(direct_phrases, list) and direct_phrases:
        sources = list(sources if isinstance(sources, list) else []) + [{"phrases": direct_phrases}]

    if not message_matches_configured_sources(message, agent_config, sources):
        return manifest

    patched = dict(manifest or {})
    patched["customer_emotion"] = str(cfg.get("emotion_label") or "undecided")
    patched["should_offer_next_action"] = bool(cfg.get("should_offer_next_action", False))

    brief = dict(patched.get("response_brief", {}) or {})
    if cfg.get("tone"):
        brief["tone"] = str(cfg.get("tone"))
    brief["must_do"] = append_unique(
        brief.get("must_do", []),
        [str(item) for item in cfg.get("must_do", []) or [] if str(item or "").strip()],
    )
    brief["must_not_do"] = append_unique(
        brief.get("must_not_do", []),
        [str(item) for item in cfg.get("must_not_do", []) or [] if str(item or "").strip()],
    )
    next_move = str(cfg.get("next_move") or "").strip()
    if next_move:
        brief["next_move"] = next_move
    patched["response_brief"] = brief
    return patched


def build_configured_smart_clarification_hypothesis(manifest: Dict[str, Any], cfg: Dict[str, Any]) -> str:
    fallback_cfg = cfg.get("fallback_hypothesis", {})
    if not isinstance(fallback_cfg, dict) or fallback_cfg.get("enabled", False) is not True:
        return ""

    try:
        min_chars = int(fallback_cfg.get("min_chars", 1) or 1)
    except Exception:
        min_chars = 1

    try:
        max_chars = int(fallback_cfg.get("max_chars", 180) or 180)
    except Exception:
        max_chars = 180

    source_paths = fallback_cfg.get("source_paths", [])
    if not isinstance(source_paths, list):
        source_paths = []

    values: Dict[str, Any] = {}
    for path in source_paths:
        path_text = str(path or "").strip()
        if not path_text:
            continue

        value = deep_get(manifest, path_text)
        if value in [None, "", [], {}]:
            value = deep_get({"manifest": manifest}, path_text)

        if isinstance(value, (dict, list)) or value in [None, "", [], {}]:
            continue

        clean_value = re.sub(r"\s+", " ", str(value or "").strip())
        if clean_value:
            values[path_text] = clean_value

    templates = fallback_cfg.get("templates", [])
    if not isinstance(templates, list):
        templates = []

    render_context = {
        "manifest": manifest,
        "values": values,
    }

    candidates: List[str] = []

    for template in templates:
        rendered = render_template(str(template or ""), render_context).strip()
        if rendered:
            candidates.append(rendered)

    for path in source_paths:
        path_text = str(path or "").strip()
        if path_text in values:
            candidates.append(str(values.get(path_text) or ""))

    for candidate in candidates:
        cleaned = re.sub(r"\s+", " ", str(candidate or "").strip())
        if len(cleaned) < min_chars:
            continue
        if max_chars > 0 and len(cleaned) > max_chars:
            cleaned = cleaned[:max_chars].strip()
        if cleaned:
            return cleaned

    return ""


def apply_smart_clarification_policy(manifest: Dict[str, Any], agent_config: Dict[str, Any]) -> Dict[str, Any]:
    cfg = get_smartness_config(agent_config, "smart_clarification")
    if not cfg.get("enabled", True):
        return manifest

    patched = dict(manifest or {})
    clarification = patched.get("best_guess_clarification")
    if not isinstance(clarification, dict):
        clarification = {}

    threshold = float(cfg.get("confidence_threshold", 0.80) or 0.80)
    try:
        confidence = float(patched.get("confidence", 0.0) or 0.0)
    except Exception:
        confidence = 0.0

    try:
        min_hypothesis_confidence = float(cfg.get("hypothesis_min_confidence", 0.0) or 0.0)
    except Exception:
        min_hypothesis_confidence = 0.0

    hypothesis = str(clarification.get("hypothesis") or "").strip()
    ask_confirm = bool(clarification.get("ask_confirm", False))

    if (
        not hypothesis
        and confidence < threshold
        and confidence >= min_hypothesis_confidence
    ):
        hypothesis = build_configured_smart_clarification_hypothesis(patched, cfg)

    if hypothesis and confidence < threshold and not bool(patched.get("needs_tool", False)):
        ask_confirm = True

    clarification["hypothesis"] = hypothesis
    clarification["ask_confirm"] = ask_confirm
    if not clarification.get("hypothesis_confidence"):
        clarification["hypothesis_confidence"] = confidence
    patched["best_guess_clarification"] = clarification
    return patched


def build_last_offered_options_from_tool_result(tool_result: Dict[str, Any], agent_config: Dict[str, Any]) -> Dict[str, Any]:
    cfg = get_smartness_config(agent_config, "last_offered_options")
    if not cfg.get("enabled", False) or not isinstance(tool_result, dict):
        return {}

    rules = cfg.get("result_rules", [])
    if not isinstance(rules, list):
        return {}

    for rule in rules:
        if not isinstance(rule, dict) or rule.get("enabled", True) is False:
            continue
        labels = rule.get("answer_draft_labels", [])
        if isinstance(labels, list) and labels:
            if str(tool_result.get("answer_draft") or "") not in {str(item) for item in labels}:
                continue
        result_path = str(rule.get("result_path") or "").strip()
        if not result_path:
            continue
        value = graph_dotted_get(tool_result, result_path, None)
        if value in [None, "", [], {}]:
            continue
        if isinstance(value, list):
            count = len(value)
            first_value = value[0] if count == 1 else None
        else:
            count = 1
            first_value = value
        return {
            "type": str(rule.get("type") or result_path),
            "count": count,
            "value": first_value,
            "source_path": result_path,
            "answer_draft": tool_result.get("answer_draft", ""),
        }
    return {}


def apply_implicit_confirmation_guardrail(manifest: Dict[str, Any], message: str, state: AgentState) -> Dict[str, Any]:
    agent_config = state.get("agent_config", {}) or {}
    cfg = get_smartness_config(agent_config, "implicit_confirmation")
    if not cfg.get("enabled", False):
        return manifest

    offered = state.get("last_offered_options", {}) or {}
    if not isinstance(offered, dict) or int(offered.get("count") or 0) != 1:
        return manifest

    sources = cfg.get("affirmation_sources") or []
    direct_phrases = cfg.get("affirmation_phrases", [])
    if isinstance(direct_phrases, list) and direct_phrases:
        sources = list(sources if isinstance(sources, list) else []) + [{"phrases": direct_phrases}]
    if not message_matches_configured_sources(message, agent_config, sources):
        return manifest

    patched = dict(manifest or {})
    type_map = cfg.get("type_to_subagent", {}) if isinstance(cfg.get("type_to_subagent"), dict) else {}
    target = str(type_map.get(str(offered.get("type") or "")) or cfg.get("default_subagent_id") or "").strip()
    if target:
        patched["selected_subagent_id"] = unify_subagent_id(target)
    patched["simple_response_mode"] = False
    patched["needs_tool"] = bool(cfg.get("set_needs_tool", True))
    patched["conversation_stage"] = str(cfg.get("conversation_stage") or "implicit_confirmation")
    patched["workflow_stage"] = str(cfg.get("workflow_stage") or patched.get("workflow_stage") or "implicit_confirmation")
    response_synthesis = patched.get("response_synthesis") if isinstance(patched.get("response_synthesis"), dict) else {}
    response_synthesis["implicit_confirmation"] = offered
    patched["response_synthesis"] = response_synthesis
    return patched


def apply_funnel_stage_policy(manifest: Dict[str, Any], agent_config: Dict[str, Any]) -> Dict[str, Any]:
    cfg = get_smartness_config(agent_config, "funnel_awareness")
    if not cfg.get("enabled", True):
        return manifest
    patched = dict(manifest or {})
    if not str(patched.get("funnel_stage") or "").strip():
        default_stage = str(cfg.get("default_stage") or "").strip()
        if default_stage:
            patched["funnel_stage"] = default_stage
    guidance = cfg.get("stage_guidance", {}) if isinstance(cfg.get("stage_guidance"), dict) else {}
    stage = str(patched.get("funnel_stage") or "").strip()
    stage_guidance = guidance.get(stage) if stage else None
    if isinstance(stage_guidance, dict):
        brief = dict(patched.get("response_brief", {}) or {})
        for key in ["tone", "reply_length", "next_move"]:
            value = str(stage_guidance.get(key) or "").strip()
            if value and not str(brief.get(key) or "").strip():
                brief[key] = value
        brief["must_do"] = append_unique(brief.get("must_do", []), [str(item) for item in stage_guidance.get("must_do", []) or []])
        brief["must_not_do"] = append_unique(brief.get("must_not_do", []), [str(item) for item in stage_guidance.get("must_not_do", []) or []])
        patched["response_brief"] = brief
    return patched


def deep_flatten_values(value: Any, prefix: str = "") -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key or "").strip()
            if not key_text:
                continue
            path = f"{prefix}.{key_text}" if prefix else key_text
            output.update(deep_flatten_values(child, path))
    else:
        output[prefix] = value
    return output


def compute_variable_changes(existing: Dict[str, Any], updated: Dict[str, Any], agent_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    cfg = get_smartness_config(agent_config, "contradiction_acknowledgment")
    if cfg.get("enabled", True) is False:
        return []
    max_changes = graph_config_int({"contradiction_acknowledgment": cfg}, "contradiction_acknowledgment.max_changes", 8, 1, 50)
    old_flat = deep_flatten_values(existing or {})
    new_flat = deep_flatten_values(updated or {})
    include_paths = set(as_string_list(cfg.get("include_paths", [])))
    exclude_paths = set(as_string_list(cfg.get("exclude_paths", [])))
    changes: List[Dict[str, Any]] = []
    for path, new_value in new_flat.items():
        if not path or path in exclude_paths:
            continue
        if include_paths and path not in include_paths:
            continue
        old_value = old_flat.get(path)
        if old_value in [None, "", [], {}]:
            continue
        if new_value in [None, "", [], {}]:
            continue
        if old_value == new_value:
            continue
        changes.append({"path": path, "from": clip_text(old_value, 80), "to": clip_text(new_value, 80)})
        if len(changes) >= max_changes:
            break
    return changes


def build_opener_context(state: AgentState) -> str:
    agent_config = state.get("agent_config", {}) or {}
    cfg = get_smartness_config(agent_config, "opener_context")
    if cfg.get("enabled", True) is False:
        return ""
    variables = state.get("variables", {}) or {}
    memories = state.get("memories", "") or ""
    summary = state.get("summary", "") or ""
    rules = cfg.get("rules", [])
    if isinstance(rules, list):
        for rule in rules:
            if not isinstance(rule, dict):
                continue
            label = str(rule.get("label") or "").strip()
            if not label:
                continue
            any_paths = rule.get("any_variable_paths", [])
            if isinstance(any_paths, list) and any(deep_get(variables, str(path), None) not in [None, "", [], {}] for path in any_paths):
                return label
            contains = str(rule.get("summary_contains") or "").strip().lower()
            if contains and contains in str(summary).lower():
                return label
            if rule.get("requires_memory") and str(memories).strip():
                return label
    if str(memories).strip():
        return str(cfg.get("returning_known_user_label") or "returning_known_user")
    if str(summary).strip():
        return str(cfg.get("returning_with_context_label") or "returning_with_context")
    return str(cfg.get("new_user_label") or "new_user")


def configured_stuck_progress_paths(agent_config: Dict[str, Any]) -> List[str]:
    """
    Return configured variable paths that represent user-visible progress.

    This is assistant-configured. Python only supplies generic mechanics, so
    future assistants can define their own progress paths without code changes.
    """
    cfg = get_smartness_config(agent_config, "stuck_pattern_detection")
    paths = as_string_list(
        cfg.get("required_progress_paths")
        or cfg.get("progress_paths")
        or cfg.get("variable_progress_paths")
        or []
    )
    return [str(path or "").strip() for path in paths if str(path or "").strip()]


def compute_variable_progress_events(
    existing: Dict[str, Any],
    updated: Dict[str, Any],
    agent_config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Detect progress events, including first-time captures and corrections.

    compute_variable_changes intentionally focuses on corrections to existing
    values. Stuck detection needs a broader signal: a previously missing
    configured progress field becoming available is also progress.
    """
    cfg = get_smartness_config(agent_config, "stuck_pattern_detection")
    if cfg.get("enabled", False) is False:
        return []

    progress_paths = configured_stuck_progress_paths(agent_config)
    if not progress_paths:
        return []

    max_events = graph_config_int(
        {"stuck_pattern_detection": cfg},
        "stuck_pattern_detection.max_progress_events",
        12,
        1,
        100,
    )

    events: List[Dict[str, Any]] = []
    for path in progress_paths:
        old_value = deep_get(existing or {}, path, None)
        new_value = deep_get(updated or {}, path, None)
        if new_value in [None, "", [], {}]:
            continue
        if old_value == new_value:
            continue
        event_type = "filled" if old_value in [None, "", [], {}] else "changed"
        events.append({
            "path": path,
            "type": event_type,
            "from": clip_text(old_value, 80) if old_value not in [None, "", [], {}] else "",
            "to": clip_text(new_value, 80),
        })
        if len(events) >= max_events:
            break
    return events


def missing_configured_progress_paths(variables: Dict[str, Any], agent_config: Dict[str, Any]) -> List[str]:
    paths = configured_stuck_progress_paths(agent_config)
    return [path for path in paths if deep_get(variables or {}, path, None) in [None, "", [], {}]]


def detect_stuck_pattern(
    state: AgentState,
    manifest: Dict[str, Any],
    variables_after: Optional[Dict[str, Any]] = None,
    stuck_signals: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """
    Detect generic stuck patterns:
    1. repeated intent without configured variable progress
    2. variable stall: configured progress paths remain missing for N turns

    All thresholds and paths are assistant-configured.
    """
    agent_config = state.get("agent_config", {}) or {}
    cfg = get_smartness_config(agent_config, "stuck_pattern_detection")
    if cfg.get("enabled", False) is False:
        return {}

    signals = stuck_signals if isinstance(stuck_signals, dict) else (state.get("stuck_signals", {}) or {})
    if not isinstance(signals, dict):
        signals = {}

    variables_now = variables_after if isinstance(variables_after, dict) else (state.get("variables", {}) or {})

    variable_stall_enabled = cfg.get("variable_stall_enabled", True) is not False
    if variable_stall_enabled:
        stall_key = str(cfg.get("variable_stall_signal_key") or "__variable_stall__")
        stall_count = int(signals.get(stall_key, 0) or 0)
        stall_threshold = graph_config_int(
            {"stuck_pattern_detection": cfg},
            "stuck_pattern_detection.variable_stall_turns",
            graph_config_int({"stuck_pattern_detection": cfg}, "stuck_pattern_detection.threshold", 2, 1, 20),
            1,
            20,
        )
        missing = missing_configured_progress_paths(variables_now, agent_config)
        if missing and stall_count >= stall_threshold:
            return {
                "pattern": "variable_stall",
                "missing_paths": missing,
                "turns_without_progress": stall_count,
                "suggested_approach": str(cfg.get("variable_stall_suggestion") or cfg.get("response_policy", {}).get("variable_stall_suggestion") or ""),
            }

    intent = str(manifest.get("user_intent") or manifest.get("conversation_stage") or "").strip()
    if not intent:
        return {}

    try:
        count = int(signals.get(intent, 0) or 0)
    except Exception:
        count = 0

    threshold = graph_config_int({"stuck_pattern_detection": cfg}, "stuck_pattern_detection.threshold", 2, 1, 20)
    if count >= threshold:
        return {
            "pattern": "repeated_intent_without_progress",
            "intent": intent,
            "count": count,
            "suggested_approach": str(cfg.get("repeated_intent_suggestion") or cfg.get("response_policy", {}).get("repeated_intent_suggestion") or ""),
        }
    return {}


def update_stuck_signals(state: AgentState, manifest: Dict[str, Any], variables_before: Dict[str, Any], variables_after: Dict[str, Any]) -> Dict[str, int]:
    agent_config = state.get("agent_config", {}) or {}
    cfg = get_smartness_config(agent_config, "stuck_pattern_detection")
    if cfg.get("enabled", False) is False:
        return state.get("stuck_signals", {}) or {}

    existing = state.get("stuck_signals", {}) or {}
    if not isinstance(existing, dict):
        existing = {}

    result = {str(k): int(v or 0) for k, v in existing.items() if str(k).strip()}
    intent = str(manifest.get("user_intent") or manifest.get("conversation_stage") or "").strip()

    progress_events = compute_variable_progress_events(variables_before, variables_after, agent_config)
    if not progress_events:
        progress_events = compute_variable_changes(variables_before, variables_after, agent_config)

    made_progress = bool(progress_events)

    if intent:
        if made_progress:
            result[intent] = 0
        else:
            result[intent] = int(result.get(intent, 0) or 0) + 1

    stall_key = str(cfg.get("variable_stall_signal_key") or "__variable_stall__")
    missing = missing_configured_progress_paths(variables_after, agent_config)
    if made_progress or not missing:
        result[stall_key] = 0
    else:
        result[stall_key] = int(result.get(stall_key, 0) or 0) + 1

    max_keys = graph_config_int({"stuck_pattern_detection": cfg}, "stuck_pattern_detection.max_keys", 20, 1, 100)
    return dict(list(result.items())[-max_keys:])

def build_proactive_surface_items(knowledge_items: List[Dict[str, Any]], agent_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    cfg = get_smartness_config(agent_config, "proactive_surface")
    if not cfg.get("enabled", False) or not isinstance(knowledge_items, list):
        return []
    triggers = cfg.get("trigger_keywords", [])
    if not isinstance(triggers, list) or not triggers:
        return []
    normalization = agent_config.get("normalization", {}) or {}
    min_score = float(cfg.get("min_score", 0.55) or 0.55)
    max_items = graph_config_int({"proactive_surface": cfg}, "proactive_surface.max_items", 3, 1, 12)
    items: List[Dict[str, Any]] = []
    for item in knowledge_items:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "")
        try:
            score = float(item.get("score", 0.0) or 0.0)
        except Exception:
            score = 0.0
        if score < min_score:
            continue
        for trigger in triggers:
            if not isinstance(trigger, dict):
                continue
            keywords = trigger.get("keywords", [])
            if isinstance(keywords, list) and matches_any(text, keywords, normalization):
                items.append({
                    "label": str(trigger.get("label") or ""),
                    "text": clip_text(text, int(cfg.get("item_max_chars", 220) or 220)),
                    "source": item.get("title", ""),
                    "score": score,
                })
                break
        if len(items) >= max_items:
            break
    return items


def build_failure_recovery_context(tool_result: Dict[str, Any], agent_config: Dict[str, Any]) -> Dict[str, Any]:
    cfg = get_smartness_config(agent_config, "tool_failure_recovery")
    if not cfg.get("enabled", False) or not isinstance(tool_result, dict):
        return {}
    if tool_result.get("ok") is not False:
        return {}
    operation = str(tool_result.get("operation") or tool_result.get("answer_draft") or tool_result.get("subagent") or "").strip()
    rules = cfg.get("rules", {}) if isinstance(cfg.get("rules"), dict) else {}
    rule = rules.get(operation) if operation else None
    if not isinstance(rule, dict):
        rule = cfg.get("default", {}) if isinstance(cfg.get("default"), dict) else {}
    if not rule:
        return {}
    return {
        "operation": operation,
        "suggest_action": rule.get("suggest_action", ""),
        "must_do": rule.get("response_brief_must_do", []),
        "must_not_do": rule.get("response_brief_must_not_do", []),
        "error_summary": clip_text(tool_result.get("error", ""), 180),
    }


def build_progressive_display_context(state: AgentState) -> Dict[str, Any]:
    agent_config = state.get("agent_config", {}) or {}
    cfg = get_smartness_config(agent_config, "progressive_display")
    if not cfg.get("enabled", False):
        return {}
    variables = state.get("variables", {}) or {}
    tool_result = state.get("tool_result", {}) or {}
    source_paths = cfg.get("source_paths", [])
    if not isinstance(source_paths, list):
        source_paths = []
    items = []
    for path in source_paths:
        value = graph_dotted_get(tool_result, str(path), None)
        if value in [None, "", [], {}]:
            value = deep_get(variables, str(path), None)
        if isinstance(value, list) and value:
            items = value
            break
    if not items:
        return {}
    display_count = graph_config_int({"progressive_display": cfg}, "progressive_display.display_count", 3, 1, 20)
    return {
        "total_count": len(items),
        "display_items": items[:display_count],
        "has_more": len(items) > display_count,
        "policy": cfg.get("response_policy", {}),
    }


def validate_and_heal_variables(variables: Dict[str, Any], schema: Dict[str, Any], agent_config: Dict[str, Any]) -> Dict[str, Any]:
    cfg = get_smartness_config(agent_config, "variable_schema_validation")
    if not cfg.get("enabled", False):
        return variables
    fields = get_schema_fields(schema or {})
    if not isinstance(fields, dict):
        return variables
    healed = dict(variables or {})
    rules = cfg.get("rules", {}) if isinstance(cfg.get("rules"), dict) else {}
    for path, meta in fields.items():
        path_text = str(path or "").strip()
        if not path_text or not isinstance(meta, dict):
            continue
        value = deep_get(healed, path_text, None)
        if value in [None, "", [], {}]:
            continue
        validator = str(meta.get("validator") or meta.get("type") or "").strip()
        rule = rules.get(validator) if validator else None
        if not isinstance(rule, dict):
            continue
        pattern = str(rule.get("regex") or "").strip()
        if pattern:
            try:
                if not re.search(pattern, str(value)):
                    healed = apply_subagent_variable_patch(healed, {path_text: None}, [], assistant_config=agent_config)
            except re.error:
                continue
    return healed


def apply_memory_to_variable_bridge(memories: List[Dict[str, Any]], variables: Dict[str, Any], agent_config: Dict[str, Any]) -> Dict[str, Any]:
    cfg = get_smartness_config(agent_config, "memory_to_variable_bridge")
    rules = cfg.get("rules") if isinstance(cfg, dict) else None
    if rules is None and isinstance(get_nested_config(agent_config, "memory_to_variable_bridge"), list):
        rules = get_nested_config(agent_config, "memory_to_variable_bridge")
    if not isinstance(rules, list) or not isinstance(memories, list):
        return variables
    updates: Dict[str, Any] = {}
    for memory in memories:
        if not isinstance(memory, dict):
            continue
        text = str(memory.get("text") or "").lower()
        memory_type = str(memory.get("type") or "")
        try:
            confidence = float(memory.get("confidence", 0.0) or 0.0)
        except Exception:
            confidence = 0.0
        for rule in rules:
            if not isinstance(rule, dict) or rule.get("enabled", True) is False:
                continue
            min_conf = float(rule.get("min_confidence", 0.7) or 0.7)
            if confidence < min_conf:
                continue
            required_type = str(rule.get("memory_type") or "").strip()
            if required_type and memory_type != required_type:
                continue
            contains = str(rule.get("memory_contains") or "").strip().lower()
            if contains and contains not in text:
                continue
            target = str(rule.get("set_variable") or rule.get("target_path") or "").strip()
            if not target:
                continue
            if bool(rule.get("only_if_missing", True)) and deep_get(variables, target, None) not in [None, "", [], {}]:
                continue
            if "set_value" in rule:
                updates[target] = rule.get("set_value")
            else:
                source_path = str(rule.get("memory_value_path") or "").strip()
                if source_path:
                    value = graph_dotted_get(memory, source_path, None)
                    if value not in [None, "", [], {}]:
                        updates[target] = value
    if not updates:
        return variables
    return apply_subagent_variable_patch(variables, updates, [], assistant_config=agent_config)


def run_batch_semantic_extraction(required_fields: List[Dict[str, Any]], message: str, variables: Dict[str, Any], agent_config: Dict[str, Any]) -> Dict[str, str]:
    cfg = get_semantic_extraction_config(agent_config)
    batch_cfg = cfg.get("batch_extraction", {}) if isinstance(cfg.get("batch_extraction"), dict) else {}
    if not batch_cfg.get("enabled", False) or len(required_fields) <= 1:
        return {}
    max_fields = graph_config_int({"batch_extraction": batch_cfg}, "batch_extraction.max_fields", 6, 2, 20)
    selected = required_fields[:max_fields]
    lines = []
    for field in selected:
        field_id = str(field.get("id") or field.get("target_path") or "").strip()
        if not field_id:
            continue
        lines.append(f"- {field_id}: {field.get('description', '')} | format: {field.get('output_format', '')}")
    if not lines:
        return {}
    prompt_text = (
        "Extract all configured fields from the user message in one pass.\n"
        "Return JSON only. Keys must be field ids. Use empty string for missing fields.\n\n"
        f"Fields:\n{chr(10).join(lines)}\n\n"
        f"User message: {message}\n\n"
        f"Known variables: {safe_json(compact_variables(variables, max_items=12), max_chars=1400)}"
    )
    system_msg = str(batch_cfg.get("system_prompt") or "You are a precise multi-field extractor. Return JSON only.")
    try:
        response = semantic_extraction_llm.invoke([
            SystemMessage(content=system_msg),
            HumanMessage(content=prompt_text),
        ])
        raw = response.content if hasattr(response, "content") else str(response)
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            return {}
        return {str(k): str(v or "").strip() for k, v in parsed.items() if str(v or "").strip()}
    except Exception:
        return {}


def dedupe_variable_change_events(events: List[Dict[str, Any]], agent_config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    cfg = get_smartness_config(agent_config or {}, "contradiction_acknowledgment")
    max_changes = graph_config_int({"contradiction_acknowledgment": cfg}, "contradiction_acknowledgment.max_changes", 12, 1, 100)
    output: List[Dict[str, Any]] = []
    seen = set()
    for event in events or []:
        if not isinstance(event, dict) or not event:
            continue
        path = str(event.get("path") or "").strip()
        key = (path, str(event.get("from") or ""), str(event.get("to") or ""), str(event.get("type") or ""))
        if not path or key in seen:
            continue
        seen.add(key)
        output.append(dict(event))
        if len(output) >= max_changes:
            break
    return output


def smart_inference_node(state: AgentState):
    agent_config = state.get("agent_config", {}) or {}
    cfg = get_smartness_config(agent_config, "smart_inference")
    if cfg.get("enabled", True) is False:
        return {}
    rules = cfg.get("rules") if isinstance(cfg, dict) else None
    if rules is None:
        raw = get_nested_config(agent_config, "smart_inference_rules")
        rules = raw if isinstance(raw, list) else []
    if not isinstance(rules, list) or not rules:
        return {}
    variables = state.get("variables", {}) or {}
    tool_result = state.get("tool_result", {}) or {}
    updates: Dict[str, Any] = {}
    notes: List[str] = []
    for rule in rules:
        if not isinstance(rule, dict) or rule.get("enabled", True) is False:
            continue
        target = str(rule.get("target_path") or rule.get("set_variable") or "").strip()
        if not target:
            continue
        if deep_get(variables, target, None) not in [None, "", [], {}] and rule.get("overwrite", False) is not True:
            continue
        value = None
        source_path = str(rule.get("from_variable_path") or rule.get("source_path") or "").strip()
        if source_path:
            value = deep_get(variables, source_path, None)
        result_path = str(rule.get("from_tool_result_path") or "").strip()
        if result_path:
            value = graph_dotted_get(tool_result, result_path, None)
        if value in [None, "", [], {}]:
            continue
        if rule.get("only_if_single_result") and isinstance(value, list):
            if len(value) != 1:
                continue
            value = value[0]
        updates[target] = value
        note = str(rule.get("note") or f"Inferred {target} from configured rule").strip()
        if note:
            notes.append(note)
    if not updates:
        return {}
    updated_variables = apply_subagent_variable_patch(variables, updates, [], assistant_config=agent_config)
    changes = compute_variable_changes(variables, updated_variables, agent_config)
    progress = compute_variable_progress_events(variables, updated_variables, agent_config)
    all_events = changes or progress
    if all_events:
        updated_variables = mirror_runtime_metadata_into_variables(
            updated_variables,
            variable_changes=dedupe_variable_change_events((state.get("variable_changes_this_turn", []) or []) + all_events, agent_config),
            agent_config=agent_config,
        )
    result = {"variables": updated_variables, "smart_inferences": notes}
    if all_events:
        result["variable_changes_this_turn"] = all_events
    return result


def attach_smartness_to_tool_update(
    result: Dict[str, Any],
    previous_variables: Dict[str, Any],
    agent_config: Dict[str, Any],
    existing_variable_changes: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    if not isinstance(result, dict):
        return result

    updated = dict(result)
    variables_after = updated.get("variables") if isinstance(updated.get("variables"), dict) else previous_variables
    existing_changes = existing_variable_changes if isinstance(existing_variable_changes, list) else []
    changes = compute_variable_changes(previous_variables, variables_after, agent_config)
    progress = compute_variable_progress_events(previous_variables, variables_after, agent_config)
    emitted_events = changes or progress

    if emitted_events:
        merged_events = dedupe_variable_change_events(existing_changes + emitted_events, agent_config)
        updated["variable_changes_this_turn"] = emitted_events
        if isinstance(variables_after, dict):
            variables_after = mirror_runtime_metadata_into_variables(
                variables_after,
                variable_changes=merged_events,
                agent_config=agent_config,
            )
            updated["variables"] = variables_after

    tool_result = updated.get("tool_result")
    if isinstance(tool_result, dict):
        offered = build_last_offered_options_from_tool_result(tool_result, agent_config)
        if not offered:
            offered = build_last_offered_options_from_variables(variables_after, tool_result, agent_config)
        if offered:
            updated["last_offered_options"] = offered
            if isinstance(variables_after, dict):
                variables_after = mirror_runtime_metadata_into_variables(
                    variables_after,
                    last_offered_options=offered,
                    agent_config=agent_config,
                )
                updated["variables"] = variables_after

        recovery = build_failure_recovery_context(tool_result, agent_config)
        if recovery:
            updated["failure_recovery_context"] = recovery

        pseudo_state: AgentState = {
            "variables": variables_after if isinstance(variables_after, dict) else previous_variables,
            "tool_result": tool_result,
            "agent_config": agent_config,
        }
        progressive = build_progressive_display_context(pseudo_state)
        if progressive:
            updated["progressive_display_context"] = progressive

    return updated

def derive_last_offered_options_from_state(state: AgentState) -> Dict[str, Any]:
    existing = state.get("last_offered_options", {}) or {}
    if isinstance(existing, dict) and existing:
        return existing

    agent_config = state.get("agent_config", {}) or {}
    variables = state.get("variables", {}) or {}
    tool_result = state.get("tool_result", {}) or {}

    if isinstance(tool_result, dict):
        offered = build_last_offered_options_from_tool_result(tool_result, agent_config)
        if offered:
            return offered

    offered = build_last_offered_options_from_variables(variables, tool_result, agent_config)
    if offered:
        return offered

    for item in reversed(state.get("multi_tool_results", []) or []):
        if not isinstance(item, dict):
            continue
        offered = build_last_offered_options_from_tool_result(item, agent_config)
        if offered:
            return offered

    return {}


def mirror_runtime_metadata_into_variables(
    variables: Dict[str, Any],
    *,
    manifest: Optional[Dict[str, Any]] = None,
    funnel_stage: str = "",
    last_offered_options: Optional[Dict[str, Any]] = None,
    emotion_history: Optional[List[str]] = None,
    emotion_trajectory: str = "",
    variable_changes: Optional[List[Dict[str, Any]]] = None,
    agent_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Mirror graph-level runtime metadata into the persisted variables object.

    The API debug payload exposes variables as state_after, so generic runtime
    metadata that regression/debug tooling needs must be mirrored there as
    ordinary state too. This helper is domain-neutral: it only mirrors generic
    graph concepts already present in AgentState, not assistant-specific fields.
    """
    if not isinstance(variables, dict):
        variables = {}

    patched = json.loads(json.dumps(variables, ensure_ascii=False))
    config = agent_config or {}

    if isinstance(manifest, dict) and manifest:
        manifest_copy = apply_funnel_stage_policy(dict(manifest), config)
        patched["manifest"] = compact_manifest(manifest_copy)
        stage = str(funnel_stage or manifest_copy.get("funnel_stage") or "").strip()
        if stage:
            patched["funnel_stage"] = stage

    if isinstance(last_offered_options, dict) and last_offered_options:
        patched["last_offered_options"] = last_offered_options

    if isinstance(emotion_history, list) and emotion_history:
        patched["emotion_history"] = [str(item) for item in emotion_history if str(item or "").strip()]

    trajectory = str(emotion_trajectory or "").strip()
    if trajectory:
        patched["emotion_trajectory"] = trajectory

    if isinstance(variable_changes, list) and variable_changes:
        clean_changes: List[Dict[str, Any]] = []
        for change in variable_changes:
            if isinstance(change, dict) and change:
                clean_changes.append(dict(change))
        if clean_changes:
            patched["variable_changes_this_turn"] = clean_changes

    return patched

def unified_manifest_node(state: AgentState):
    message = last_user_message(state)
    agent_config = state.get("agent_config", {}) or {}
    variables = state.get("variables", {}) or {}
    schema = state.get("schema", {}) or {}
    tool_result = state.get("tool_result", {}) or {}
    previous_manifest_summary = previous_manifest_summary_from_state(state)
    prior_emotion_history = normalize_emotion_history(state.get("emotion_history", []))
    opener_context = build_opener_context(state)
    last_offered_options = state.get("last_offered_options", {}) or {}
    stuck_pattern = state.get("stuck_pattern", {}) or {}

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            # IDENTITY
            "You are the Unified Manifest — the intelligence core of a "
            "configurable multi-tenant agentic assistant. "
            "Return ONLY a single valid JSON object. No markdown. No prose. "
            "\n\n"
            # HOW TO THINK
            "THINK LIKE A BRILLIANT HUMAN FIRST. Before classifying anything:\n"
            "1. Read the full message and conversation context as a human would.\n"
            "2. Understand what the person MEANS, not just what they literally typed.\n"
            "   Examples:\n"
            "   - 'بكرة الصبح' means tomorrow morning — extract date AND time preference.\n"
            "   - 'الفرع القريب مني' means find nearest branch — location intent.\n"
            "   - 'غير الموعد' means change an existing appointment — not a new independent booking.\n"
            "   - 'تمام اتفقنا' means confirmed — move to the confirmation/next step.\n"
            "   - 'لا معلش' or 'actually no' means cancelled or changed mind — update accordingly.\n"
            "3. Check previous_manifest_summary to understand the flow momentum from last turn: what was selected, what stage was active, what next move was planned, and whether the current message continues, confirms, corrects, or cancels that flow.\n"
            "4. Infer implicit context: if a user said their name, phone, branch, or preference earlier and it is in variables, it is already known. Do not ask for it again.\n"
            "5. Detect the FULL intent, including things the user implied but did not say explicitly.\n"
            "\n\n"
            # VARIABLE INTELLIGENCE
            "VARIABLE AWARENESS:\n"
            "Read current_variables carefully before deciding anything.\n"
            "- If a value is already present in variables, do NOT ask for it again.\n"
            "- If the user provides a new value that differs from variables, update it as soft info only when allowed.\n"
            "- If the user confirms an existing value, keep it.\n"
            "- If the user says 'change X to Y', set extracted_updates.X = Y and delete stale derived soft state if allowed.\n"
            "- Never clear a variable unless the user explicitly said to remove/change it.\n"
            "- missing_tool_inputs should ONLY contain fields genuinely absent from both the message AND current_variables.\n"
            "\n\n"
            # MULTI-INTENT
            "MULTI-INTENT DETECTION:\n"
            "Scan the full message for multiple needs. Common patterns:\n"
            "- Action + question: 'book me + how long does it take?'\n"
            "- Sequential dependency: 'nearest branch + available slots'\n"
            "- Independent parallel: 'price of oil change + price of brakes'\n"
            "When multi-intent: set multi_intent=true, populate multi_intents, use selected_subagent_id for the first executor, and chained_subagent_id when the second genuinely depends on the first output.\n"
            "NEVER parallelize alternatives from a change of mind; the latest preference wins.\n"
            "\n\n"
            # CHANGE OF MIND
            "CHANGE OF MIND HANDLING:\n"
            "When the user changes a previous choice, the LATEST explicit preference wins.\n"
            "- Set extracted_updates with the new soft value.\n"
            "- Set extracted_deletions only for stale non-source-of-truth soft state.\n"
            "- Never run both old and new choices in parallel.\n"
            "- Do not mention the old choice unless helpful for clarity.\n"
            "\n\n"
            # PROACTIVE INTELLIGENCE
            "PROACTIVE INTELLIGENCE:\n"
            "After every tool result, think: what is the most useful NEXT step?\n"
            "- Slots shown → ask which slot works for them.\n"
            "- Branch found → offer/check available times if the user requested availability or asked what to do next.\n"
            "- Troubleshooting complete → offer to book a visit if appropriate.\n"
            "- Booking confirmed → close warmly; no more questions needed.\n"
            "Set should_offer_next_action=true and response_brief.next_move accordingly when useful.\n"
            "\n\n"
            # EMOTION AND TONE
            "CUSTOMER EMOTION AND SMARTNESS SIGNALS:\n"
            "Detect tone using the assistant config guidance, then set customer_emotion. Use emotion_history and emotion_trajectory from context when present; do not reset the user's mood every turn.\n"
            "If configured hesitation signals match, switch into advisor mode: help the user decide instead of pushing completion.\n"
            "If the user seems to confirm one previously offered option and last_offered_options shows exactly one option, treat it as implicit confirmation according to config.\n"
            "If confidence is below the configured smart_clarification threshold but you have a plausible hypothesis, set best_guess_clarification.ask_confirm=true with a concise hypothesis instead of asking an open-ended clarification.\n"
            "Set funnel_stage when the config or context supports it: awareness, consideration, intent, confirmation, or post_conversion.\n"
            "\n\n"
            # GROUNDING
            "GROUNDING:\n"
            "extracted_updates may ONLY contain soft user info such as name, phone, stated location, service preference, or customer-provided detail fields. "
            "NEVER write to source-of-truth operational fields such as available slots, booking_status, visit_id, branch confirmed by tool, or tool-owned appointment results. "
            "Those are owned by executors/tools. Never claim tool results, slots, prices, or IDs not in tool_result/variables/knowledge.\n"
            "\n\n"
            # OUTPUT
            "OUTPUT: Valid JSON with exactly these top-level keys:\n"
            "user_intent, selected_subagent_id, chained_subagent_id, chained_subagent_reason, detected_intents, multi_intents, parallel_tool_requests, knowledge_queries, response_synthesis, "
            "multi_intent_execution_mode, conversation_stage, workflow_stage, customer_emotion, emotion_trajectory, funnel_stage, best_guess_clarification, user_expectation, risk_level, confidence, simple_response_mode, simple_response_reason, "
            "needs_knowledge, needs_memory, needs_tool, requested_tool_name, tool_request_payload, missing_tool_inputs, needs_subagent_reasoning, needs_quality_guard, needs_style_repair, "
            "needs_full_manifest, extracted_updates, extracted_deletions, response_style, reply_length, should_ask_question, question_goal, should_offer_next_action, response_brief, "
            "response_strategy, reasoning_summary.\n"
            "response_brief must have: tone, language, reply_length, must_do, must_not_do, next_move.",
        ),
        (
            "user",
            "=== ASSISTANT MANIFEST CARD ===\n{manifest_card}\n\n"
            "=== PREVIOUS TURN MANIFEST SUMMARY — WHAT YOU DECIDED LAST TURN ===\n"
            "{previous_manifest_summary}\n\n"
            "=== SMARTNESS CONTEXT ===\n"
            "Emotion history: {emotion_history}\n"
            "Opener context: {opener_context}\n"
            "Last offered options: {last_offered_options}\n"
            "Stuck pattern: {stuck_pattern}\n\n"
            "=== CONVERSATION SUMMARY (what happened before) ===\n{summary}\n\n"
            "=== WHAT IS ALREADY KNOWN — DO NOT ASK FOR THESE AGAIN ===\n"
            "{variables}\n\n"
            "=== LAST TOOL RESULT ===\n{tool_result}\n\n"
            "=== LATEST USER MESSAGE ===\n{message}\n\n"
            "Think step by step:\n"
            "1. What does the user MEAN (not just what they literally typed)?\n"
            "2. What is already known from the variables above?\n"
            "3. What did you decide last turn, and does this message continue, confirm, correct, or cancel that flow?\n"
            "4. What is the minimum needed to move forward?\n"
            "5. Which executor and tools are needed?\n"
            "6. What should missing_tool_inputs contain? "
            "   (Only fields absent from BOTH the message AND the known variables.)\n"
            "Then return the manifest JSON.",
        ),
    ])

    def invoke_manifest(profile: str) -> Dict[str, Any]:
        manifest_card_max = 5200 if profile == "short" else 8400

        decision = (prompt | manifest_llm).invoke({
            "manifest_card": safe_json(
                unified_manifest_card(agent_config, schema, profile=profile),
                max_chars=manifest_card_max,
            ),
            "previous_manifest_summary": clip_text(
                previous_manifest_summary,
                graph_config_int(
                    {"manifest_context": get_manifest_context_config(agent_config)},
                    "manifest_context.previous_manifest_summary_max_chars",
                    default=600,
                    minimum=200,
                    maximum=6000,
                ),
            ) or "No previous manifest summary.",
            "emotion_history": safe_json(prior_emotion_history[-8:], max_chars=500),
            "opener_context": opener_context or "",
            "last_offered_options": safe_json(last_offered_options, max_chars=900),
            "stuck_pattern": safe_json(stuck_pattern, max_chars=700),
            "summary": clip_text(state.get("summary", ""), 650),
            "variables": safe_json(compact_variables(variables, schema), max_chars=2200),
            "tool_result": safe_json(tool_result, max_chars=2600),
            "message": message,
        })

        parsed = parse_manifest_response(decision)
        parsed = normalize_json_manifest(parsed)
        parsed["selected_subagent_id"] = unify_subagent_id(parsed.get("selected_subagent_id", ""))
        parsed = apply_routing_guardrails(
            manifest=parsed,
            message=message,
            agent_config=agent_config,
            variables=variables,
        )
        parsed = apply_active_deterministic_flow_guardrails(
            manifest=parsed,
            message=message,
            agent_config=agent_config,
            variables=variables,
        )
        parsed = apply_dependent_intent_chain_guardrails(
            manifest=parsed,
            message=message,
            agent_config=agent_config,
            variables=variables,
        )
        parsed = apply_hesitation_detection(parsed, message, agent_config)
        parsed = apply_implicit_confirmation_guardrail(parsed, message, state)
        parsed = apply_smart_clarification_policy(parsed, agent_config)
        parsed = apply_funnel_stage_policy(parsed, agent_config)
        parsed = apply_length_mirroring(parsed, message, agent_config)
        parsed = apply_emotion_arc_to_manifest(parsed, prior_emotion_history, agent_config)
        parsed["selected_subagent_id"] = unify_subagent_id(parsed.get("selected_subagent_id", ""))
        parsed["chained_subagent_id"] = unify_subagent_id(parsed.get("chained_subagent_id", ""))
        if parsed.get("chained_subagent_id") == parsed.get("selected_subagent_id"):
            parsed["chained_subagent_id"] = ""
        parsed["manifest_profile_used"] = profile
        return parsed

    try:
        manifest = invoke_manifest("short")

        known_subagent_ids = {
            s.get("id")
            for s in get_subagent_config_list(agent_config)
            if isinstance(s, dict) and s.get("id")
        }

        selected_id = manifest.get("selected_subagent_id", "")
        chained_id = manifest.get("chained_subagent_id", "")
        selected_missing = bool(known_subagent_ids) and selected_id and selected_id not in known_subagent_ids
        chained_missing = bool(known_subagent_ids) and chained_id and chained_id not in known_subagent_ids

        _raw_parallel = manifest.get("parallel_tool_requests", [])
        _parallel_declared_but_broken = (
            isinstance(_raw_parallel, list)
            and len(_raw_parallel) > 0
            and not normalize_parallel_tool_requests(manifest)
        )

        _raw_updates = manifest.get("extracted_updates", {}) or {}
        _prepared_retry_updates = prepare_manifest_extracted_updates(
            _raw_updates,
            agent_config,
            schema,
        )
        _filtered_retry_updates = filter_manifest_updates(
            _prepared_retry_updates,
            agent_config,
        )
        _update_strip_ratio = (
            (len(_prepared_retry_updates) - len(_filtered_retry_updates)) / max(len(_prepared_retry_updates), 1)
            if _prepared_retry_updates
            else 0.0
        )

        should_retry_full = (
            bool(manifest.get("needs_full_manifest"))
            or manifest_confidence(manifest) < 0.65
            or selected_missing
            or chained_missing
            or (manifest.get("needs_tool") and not manifest.get("requested_tool_name") and not manifest.get("selected_subagent_id"))
            or (manifest.get("chained_subagent_id") and not manifest.get("selected_subagent_id"))
            or _parallel_declared_but_broken
            or (manifest_has_multi_intents(manifest) and not manifest.get("selected_subagent_id") and not _raw_parallel)
            or should_retry_full_manifest_for_stripped_updates(_prepared_retry_updates, _filtered_retry_updates, agent_config)
        )

        if should_retry_full:
            manifest = invoke_manifest("full")

    except Exception as exc:
        subagents = get_subagent_config_list(agent_config) or [{"id": "general"}]
        error_text = f"{type(exc).__name__}: {exc}"

        manifest = {
            "user_intent": "unknown",
            "selected_subagent_id": subagents[0].get("id", "general"),
            "chained_subagent_id": "",
            "chained_subagent_reason": "",
            "detected_intents": [],
            "multi_intents": [],
            "parallel_tool_requests": [],
            "knowledge_queries": [],
            "response_synthesis": {},
            "multi_intent_execution_mode": "",
            "conversation_stage": "unknown",
            "workflow_stage": "unknown",
            "customer_emotion": "unknown",
            "user_expectation": "",
            "risk_level": "medium",
            "confidence": 0.4,
            "simple_response_mode": False,
            "simple_response_reason": "",
            "needs_knowledge": True,
            "needs_memory": True,
            "needs_tool": False,
            "requested_tool_name": "",
            "tool_request_payload": {},
            "missing_tool_inputs": [],
            "needs_subagent_reasoning": True,
            "needs_quality_guard": True,
            "needs_style_repair": False,
            "needs_full_manifest": True,
            "extracted_updates": {},
            "extracted_deletions": [],
            "response_style": "",
            "reply_length": "",
            "should_ask_question": True,
            "question_goal": "",
            "should_offer_next_action": False,
            "response_brief": {
                "tone": "careful and helpful",
                "language": "",
                "reply_length": "",
                "must_do": ["answer carefully"],
                "must_not_do": ["do not invent facts"],
                "next_move": "",
            },
            "response_strategy": "Answer carefully and do not invent facts.",
            "reasoning_summary": f"Fallback manifest due to error: {error_text}",
            "manifest_error": error_text,
            "manifest_profile_used": "fallback",
        }

        manifest = apply_routing_guardrails(
            manifest=manifest,
            message=message,
            agent_config=agent_config,
            variables=variables,
        )
        manifest = apply_active_deterministic_flow_guardrails(
            manifest=manifest,
            message=message,
            agent_config=agent_config,
            variables=variables,
        )
        manifest = apply_dependent_intent_chain_guardrails(
            manifest=manifest,
            message=message,
            agent_config=agent_config,
            variables=variables,
        )
        manifest["selected_subagent_id"] = unify_subagent_id(
            manifest.get("selected_subagent_id", "")
        )
        manifest["chained_subagent_id"] = unify_subagent_id(
            manifest.get("chained_subagent_id", "")
        )
        if manifest.get("chained_subagent_id") == manifest.get("selected_subagent_id"):
            manifest["chained_subagent_id"] = ""

    manifest = apply_funnel_stage_policy(manifest, agent_config)
    selected_subagent = get_subagent_by_id(agent_config, manifest.get("selected_subagent_id", ""))

    prepared_manifest_updates = prepare_manifest_extracted_updates(
        manifest.get("extracted_updates", {}) or {},
        agent_config,
        schema,
    )

    safe_manifest_updates = filter_manifest_updates(
        prepared_manifest_updates,
        agent_config,
    )

    safe_manifest_deletions = filter_manifest_deletions(
        manifest.get("extracted_deletions", []) or [],
        agent_config,
    )

    updated_variables = merge_variables_intelligently(
        existing=variables,
        incoming=safe_manifest_updates,
        deletions=safe_manifest_deletions,
        agent_config=agent_config,
    )
    updated_variables = validate_and_heal_variables(updated_variables, schema, agent_config)
    variable_changes = compute_variable_changes(variables, updated_variables, agent_config)
    new_emotion_history = update_emotion_history(
        current_emotion=str(manifest.get("customer_emotion") or "neutral"),
        prior_history=prior_emotion_history,
        agent_config=agent_config,
    )
    updated_variables = mirror_runtime_metadata_into_variables(
        updated_variables,
        manifest=manifest,
        funnel_stage=str(manifest.get("funnel_stage") or ""),
        emotion_history=new_emotion_history,
        emotion_trajectory=str(manifest.get("emotion_trajectory") or ""),
        variable_changes=variable_changes,
        agent_config=agent_config,
    )
    new_stuck_signals = update_stuck_signals(state, manifest, variables, updated_variables)
    new_stuck_pattern = detect_stuck_pattern(
        state,
        manifest,
        variables_after=updated_variables,
        stuck_signals=new_stuck_signals,
    )

    return {
        "manifest": manifest,
        "planner": build_planner_compat(manifest),
        "selected_subagent": selected_subagent,
        "variables": updated_variables,
        "emotion_history": new_emotion_history,
        "emotion_trajectory": manifest.get("emotion_trajectory", ""),
        "funnel_stage": manifest.get("funnel_stage", ""),
        "best_guess_clarification": manifest.get("best_guess_clarification", {}),
        "opener_context": opener_context,
        "variable_changes_this_turn": variable_changes,
        "stuck_signals": new_stuck_signals,
        "stuck_pattern": new_stuck_pattern,
        "multi_intents": manifest.get("multi_intents", []),
        "parallel_tool_requests": manifest.get("parallel_tool_requests", []),
        "knowledge_queries": manifest.get("knowledge_queries", []),
        "response_synthesis": manifest.get("response_synthesis", {}),
        "previous_manifest_summary": summarize_manifest_for_next_turn(manifest, agent_config),
    }


def retrieve_memory_and_knowledge_node(state: AgentState):
    """
    Run independent memory and knowledge retrieval in parallel when both are
    requested by the manifest. Both branches use the same immutable incoming
    state; resulting variables are merged config-safely.
    """
    with ThreadPoolExecutor(max_workers=2) as executor:
        memory_future = executor.submit(retrieve_memory_node, dict(state))
        knowledge_future = executor.submit(retrieve_knowledge_node, dict(state))

        try:
            memory_result = memory_future.result(timeout=20)
        except Exception:
            memory_result = {"memories": "", "memories_raw": []}

        try:
            knowledge_result = knowledge_future.result(timeout=20)
        except Exception:
            knowledge_result = {
                "knowledge": "NO_CONFIDENT_KNOWLEDGE_FOUND",
                "knowledge_items": [],
                "multi_knowledge": [],
            }

    if not isinstance(memory_result, dict):
        memory_result = {"memories": "", "memories_raw": []}
    if not isinstance(knowledge_result, dict):
        knowledge_result = {"knowledge": "NO_CONFIDENT_KNOWLEDGE_FOUND", "knowledge_items": [], "multi_knowledge": []}

    merged: Dict[str, Any] = dict(knowledge_result)
    merged.update({k: v for k, v in memory_result.items() if k != "variables"})

    original_variables = state.get("variables", {}) or {}
    memory_variables = memory_result.get("variables") if isinstance(memory_result.get("variables"), dict) else original_variables
    knowledge_variables = knowledge_result.get("variables") if isinstance(knowledge_result.get("variables"), dict) else None

    variables = memory_variables
    if isinstance(knowledge_variables, dict) and knowledge_variables != original_variables:
        variables = merge_variables_intelligently(
            existing=memory_variables,
            incoming=knowledge_variables,
            deletions=[],
            agent_config=state.get("agent_config", {}) or {},
        )

    if variables != original_variables:
        merged["variables"] = variables
        changes = compute_variable_changes(original_variables, variables, state.get("agent_config", {}) or {})
        progress = compute_variable_progress_events(original_variables, variables, state.get("agent_config", {}) or {})
        events = changes or progress
        if events:
            merged["variable_changes_this_turn"] = events

    if not merged.get("proactive_surface_items"):
        merged["proactive_surface_items"] = build_proactive_surface_items(
            merged.get("knowledge_items", []) or [],
            state.get("agent_config", {}) or {},
        )

    return merged


def decide_after_manifest(state: AgentState) -> str:
    manifest = state.get("manifest", {}) or {}

    if active_deterministic_flow_subagent_id_from_state(state):
        return "tool_execution"

    if should_use_simple_response(state):
        return "simple_response"

    needs_memory = bool(manifest.get("needs_memory"))
    needs_knowledge = bool(manifest.get("needs_knowledge") or manifest_has_knowledge_queries(manifest))

    if needs_memory and needs_knowledge:
        return "retrieve_memory_and_knowledge"

    if needs_memory:
        return "retrieve_memory"

    if needs_knowledge:
        return "retrieve_knowledge"

    if should_route_to_semantic_extraction(state):
        return "semantic_extraction"

    if manifest_has_parallel_tool_requests(manifest):
        return "tool_execution"

    if manifest.get("needs_tool"):
        return "tool_execution"

    if manifest.get("needs_subagent_reasoning"):
        return "subagent_reasoning"

    return "response"



def retrieve_memory_node(state: AgentState):
    manifest = state.get("manifest", {}) or {}

    if manifest.get("needs_memory") is False:
        return {"memories": ""}

    message = last_user_message(state)

    try:
        memories = search_memories(
            state["assistant_id"],
            state["user_id"],
            message,
            limit=MEMORY_TOP_K,
        )
    except Exception:
        memories = []

    if not memories:
        return {"memories": "", "memories_raw": []}

    bridged_variables = apply_memory_to_variable_bridge(
        memories=memories,
        variables=state.get("variables", {}) or {},
        agent_config=state.get("agent_config", {}) or {},
    )

    text = "\n".join([
        f"- {m.get('text', '')} (type={m.get('type', 'other')}, confidence={m.get('confidence', '')})"
        for m in memories
        if m.get("text")
    ])

    result = {"memories": text, "memories_raw": memories}
    if bridged_variables != (state.get("variables", {}) or {}):
        result["variables"] = bridged_variables
        result["variable_changes_this_turn"] = compute_variable_changes(
            state.get("variables", {}) or {},
            bridged_variables,
            state.get("agent_config", {}) or {},
        )
    return result


def decide_after_memory(state: AgentState) -> str:
    manifest = state.get("manifest", {}) or {}

    if manifest.get("needs_knowledge") or manifest_has_knowledge_queries(manifest):
        return "retrieve_knowledge"

    if should_run_proactive_surface_check(state):
        return "proactive_surface"

    if should_route_to_semantic_extraction(state):
        return "semantic_extraction"

    if active_deterministic_flow_subagent_id_from_state(state):
        return "tool_execution"

    if manifest.get("needs_tool") or manifest_has_parallel_tool_requests(manifest):
        return "tool_execution"

    if manifest.get("needs_subagent_reasoning"):
        return "subagent_reasoning"

    return "response"



def dedupe_strings(values: List[str], limit: int = 6) -> List[str]:
    output: List[str] = []
    seen = set()

    for value in values or []:
        text = str(value or "").strip()
        if not text:
            continue

        key = re.sub(r"\s+", " ", text.lower()).strip()
        if key in seen:
            continue

        seen.add(key)
        output.append(text)

        if len(output) >= limit:
            break

    return output


def build_knowledge_queries(
    message: str,
    manifest: Dict[str, Any],
    variables: Dict[str, Any],
    schema: Dict[str, Any],
    agent_config: Dict[str, Any]
) -> List[str]:
    retrieval_config = agent_config.get("knowledge_retrieval", {})
    if not isinstance(retrieval_config, dict):
        retrieval_config = {}

    max_queries = int(retrieval_config.get("max_queries", 4) or 4)
    queries: List[str] = []

    if isinstance(manifest.get("knowledge_queries"), list):
        queries.extend([str(item) for item in manifest.get("knowledge_queries", [])])

    for intent in manifest.get("multi_intents", []) or []:
        if not isinstance(intent, dict):
            continue

        for key in ["knowledge_query", "query", "user_goal", "intent_type"]:
            value = str(intent.get(key) or "").strip()
            if value:
                queries.append(value)
                break

    default_query = " ".join([
        message,
        manifest.get("user_intent", ""),
        manifest.get("response_strategy", ""),
        safe_json(compact_variables(variables, schema), max_chars=1000),
    ]).strip()

    if default_query:
        queries.append(default_query)

    return dedupe_strings(queries, limit=max_queries)


def knowledge_items_key(item: Dict[str, Any]) -> str:
    if not isinstance(item, dict):
        return ""

    return "|".join([
        str(item.get("title") or "").strip().lower(),
        str(item.get("text") or "").strip().lower()[:240],
    ])


def retrieve_knowledge_node(state: AgentState):
    message = last_user_message(state)
    manifest = state.get("manifest", {}) or {}
    variables = state.get("variables", {}) or {}
    schema = state.get("schema", {}) or {}
    agent_config = state.get("agent_config", {}) or {}
    assistant_id = state.get("assistant_id", "")
    conversation_id = state.get("conversation_id", "")
    user_id = state.get("user_id", "")

    queries = build_knowledge_queries(
        message=message,
        manifest=manifest,
        variables=variables,
        schema=schema,
        agent_config=agent_config,
    )

    if not queries:
        return {
            "knowledge": "NO_CONFIDENT_KNOWLEDGE_FOUND",
            "knowledge_items": [],
            "knowledge_queries": [],
            "multi_knowledge": [],
        }

    retrieval_config = agent_config.get("knowledge_retrieval", {})
    if not isinstance(retrieval_config, dict):
        retrieval_config = {}

    cache_enabled = bool(retrieval_config.get("cache_enabled", True))
    try:
        cache_max_age = int(retrieval_config.get("cache_max_age_minutes", 20) or 20)
    except Exception:
        cache_max_age = 20

    if cache_enabled and conversation_id:
        try:
            cached = get_rag_cache(
                conversation_id=conversation_id,
                max_age_minutes=cache_max_age,
                assistant_id=assistant_id,
            )
            if cached and isinstance(cached.get("compressed_payload"), list) and cached["compressed_payload"]:
                compressed_items = cached["compressed_payload"]
                lines = []
                for item in compressed_items:
                    if not isinstance(item, dict):
                        continue
                    title = item.get("title", "Untitled")
                    score = float(item.get("score", 0.0) or 0.0)
                    item_text = item.get("text", "")
                    lines.append(f"- Source: {title} | Score: {score:.3f}\n  Content: {item_text}")

                return {
                    "knowledge": "\n".join(lines) if lines else "NO_CONFIDENT_KNOWLEDGE_FOUND",
                    "knowledge_items": compressed_items,
                    "knowledge_queries": queries,
                    "multi_knowledge": [{
                        "query": cached.get("query", queries[0]),
                        "knowledge": "\n".join(lines),
                        "items": compressed_items,
                        "cache_hit": True,
                    }],
                    "proactive_surface_items": build_proactive_surface_items(compressed_items, agent_config),
                }
        except Exception:
            pass

    per_query_limit = int(retrieval_config.get("per_query_top_k", KNOWLEDGE_TOP_K) or KNOWLEDGE_TOP_K)
    max_total_items = int(retrieval_config.get("max_total_items", max(KNOWLEDGE_TOP_K, len(queries) * 2)) or KNOWLEDGE_TOP_K)

    multi_knowledge: List[Dict[str, Any]] = []
    combined_items: List[Dict[str, Any]] = []
    seen = set()
    errors: List[str] = []

    for query in queries:
        try:
            raw = search_knowledge(assistant_id, query, limit=per_query_limit)
            compressed = compress_knowledge(raw, query)
        except Exception as exc:
            errors.append(f"{query}: {type(exc).__name__}: {exc}")
            compressed = []

        if not compressed:
            multi_knowledge.append({
                "query": query,
                "knowledge": "NO_CONFIDENT_KNOWLEDGE_FOUND",
                "items": [],
            })
            continue

        query_lines = []
        for item in compressed:
            if not isinstance(item, dict):
                continue

            key = knowledge_items_key(item)
            if key and key not in seen:
                seen.add(key)
                combined_items.append(item)

            title = item.get("title", "Untitled")
            score = float(item.get("score", 0.0) or 0.0)
            item_text = item.get("text", "")
            query_lines.append(f"- Source: {title} | Score: {score:.3f}\n  Content: {item_text}")

        multi_knowledge.append({
            "query": query,
            "knowledge": "\n".join(query_lines),
            "items": compressed,
        })

    combined_items = combined_items[:max_total_items]

    if not combined_items:
        error_text = f" Retrieval errors: {'; '.join(errors)}" if errors else ""
        return {
            "knowledge": f"NO_CONFIDENT_KNOWLEDGE_FOUND.{error_text}".strip(),
            "knowledge_items": [],
            "knowledge_queries": queries,
            "multi_knowledge": multi_knowledge,
        }

    lines = []
    for item in combined_items:
        title = item.get("title", "Untitled")
        score = float(item.get("score", 0.0) or 0.0)
        item_text = item.get("text", "")
        lines.append(f"- Source: {title} | Score: {score:.3f}\n  Content: {item_text}")

    if cache_enabled and conversation_id and combined_items:
        try:
            save_rag_cache(
                conversation_id=conversation_id,
                assistant_id=assistant_id,
                user_id=user_id,
                query=queries[0],
                knowledge_payload=[item.get("text", "") for item in combined_items if isinstance(item, dict)],
                compressed_payload=combined_items,
            )
        except Exception:
            pass

    return {
        "knowledge": "\n".join(lines),
        "knowledge_items": combined_items,
        "knowledge_queries": queries,
        "multi_knowledge": multi_knowledge,
        "proactive_surface_items": build_proactive_surface_items(combined_items, agent_config),
    }


def should_run_proactive_surface_check(state: AgentState) -> bool:
    agent_config = state.get("agent_config", {}) or {}
    cfg = get_smartness_config(agent_config, "proactive_surface")
    if cfg.get("enabled", False) is False:
        return False
    if state.get("proactive_surface_items"):
        return False
    return bool(state.get("knowledge_items"))


def decide_after_knowledge(state: AgentState) -> str:
    manifest = state.get("manifest", {}) or {}

    if should_run_proactive_surface_check(state):
        return "proactive_surface"

    if should_route_to_semantic_extraction(state):
        return "semantic_extraction"

    if active_deterministic_flow_subagent_id_from_state(state):
        return "tool_execution"

    if manifest.get("needs_tool") or manifest_has_parallel_tool_requests(manifest):
        return "tool_execution"

    if manifest.get("needs_subagent_reasoning"):
        return "subagent_reasoning"

    return "response"


def decide_after_proactive_surface(state: AgentState) -> str:
    manifest = state.get("manifest", {}) or {}

    if should_route_to_semantic_extraction(state):
        return "semantic_extraction"

    if active_deterministic_flow_subagent_id_from_state(state):
        return "tool_execution"

    if manifest.get("needs_tool") or manifest_has_parallel_tool_requests(manifest):
        return "tool_execution"

    if manifest.get("needs_subagent_reasoning"):
        return "subagent_reasoning"

    return "response"


def proactive_surface_check_node(state: AgentState):
    items = build_proactive_surface_items(
        state.get("knowledge_items", []) or [],
        state.get("agent_config", {}) or {},
    )
    if not items:
        return {}
    return {"proactive_surface_items": items}



def subagent_history_from_messages(messages: Sequence[BaseMessage]) -> List[Dict[str, str]]:
    output: List[Dict[str, str]] = []

    for item in messages[-MANIFEST_HISTORY_LIMIT:]:
        if isinstance(item, HumanMessage):
            output.append({"role": "user", "content": item.content})
        elif isinstance(item, AIMessage):
            output.append({"role": "assistant", "content": item.content})

    return output


def normalize_tool_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {}

    if "arguments" in payload or "operation" in payload:
        return payload

    operation = payload.get("op") or payload.get("name") or payload.get("action") or ""
    arguments = payload.get("args") or payload.get("payload") or {}

    if not isinstance(arguments, dict):
        arguments = {}

    return {
        "operation": operation,
        "arguments": arguments,
    }


def is_present(value: Any) -> bool:
    return value not in [None, "", [], {}]


def first_present(*values: Any) -> Any:
    for value in values:
        if is_present(value):
            return value
    return ""


def get_known_branch_from_variables(
    variables: Dict[str, Any],
    agent_config: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Return the first configured location/branch-like value.

    Uses assistant.known_branch_paths, so future assistants can use their own
    location/property/store paths without Python changes. If no paths are
    configured, no branch value is guessed.
    """
    agent_config = agent_config or {}
    paths = as_string_list(agent_config.get("known_branch_paths", []))

    for path in paths:
        clean_path = str(path or "").strip()
        if not clean_path:
            continue
        value = deep_get(variables, clean_path, "")
        if is_present(value):
            return str(value).strip()

    return ""


def normalize_direct_tool_arguments(
    operation: str,
    arguments: Dict[str, Any],
    variables: Dict[str, Any],
    agent_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Auto-inject variable values into direct tool arguments.

    Primary path is config-driven:
    - assistant.tool_argument_injection_rules
    - assistant.tool_pending_object_rules

    The fallback branch preserves old behavior only when no injection rules are configured.
    """
    agent_config = agent_config or {}
    op = str(operation or "").strip()
    args = dict(arguments or {})

    injection_rules = agent_config.get("tool_argument_injection_rules", [])
    applied_config_rule = False

    if isinstance(injection_rules, list) and injection_rules:
        for rule in injection_rules:
            if not isinstance(rule, dict):
                continue

            rule_operations = rule.get("operations", [])
            if rule_operations == "*" or rule_operations == ["*"]:
                operation_matches = True
            elif isinstance(rule_operations, list):
                operation_matches = op in [str(item) for item in rule_operations]
            else:
                operation_matches = False

            if not operation_matches:
                continue

            applied_config_rule = True

            for injection in rule.get("inject", []):
                if not isinstance(injection, dict):
                    continue

                arg_name = str(injection.get("arg") or "").strip()
                if not arg_name:
                    continue

                if is_present(args.get(arg_name)):
                    continue

                if "literal" in injection:
                    value = injection.get("literal")
                    if is_present(value) or value is False:
                        args[arg_name] = value
                    continue

                paths = injection.get("from_paths", [])
                if not isinstance(paths, list):
                    paths = []

                value = first_present(*[
                    deep_get(variables, str(path or "").strip(), "")
                    for path in paths
                    if str(path or "").strip()
                ])

                if is_present(value):
                    args[arg_name] = value

    pending_rules = agent_config.get("tool_pending_object_rules", [])
    if isinstance(pending_rules, list) and pending_rules:
        for rule in pending_rules:
            if not isinstance(rule, dict):
                continue

            rule_operations = rule.get("operations", [])
            if rule_operations == "*" or rule_operations == ["*"]:
                operation_matches = True
            elif isinstance(rule_operations, list):
                operation_matches = op in [str(item) for item in rule_operations]
            else:
                operation_matches = False

            if not operation_matches:
                continue

            applied_config_rule = True

            pending_path = str(rule.get("pending_path") or "").strip()
            fields = rule.get("fields", [])
            if not pending_path or not isinstance(fields, list):
                continue

            pending = deep_get(variables, pending_path, {})
            if not isinstance(pending, dict):
                continue

            for field in fields:
                field_str = str(field or "").strip()
                if field_str and not is_present(args.get(field_str)) and is_present(pending.get(field_str)):
                    args[field_str] = pending.get(field_str)

    if applied_config_rule:
        return args

    legacy_groups = agent_config.get("legacy_tool_argument_operation_groups", {})
    legacy_fields = agent_config.get("legacy_tool_argument_field_config", {})
    if not isinstance(legacy_groups, dict) or legacy_groups.get("enabled") is not True:
        return args
    if not isinstance(legacy_fields, dict) or legacy_fields.get("enabled") is not True:
        return args

    location_and_date_ops = set(as_string_list(legacy_groups.get("location_and_date_ops", [])))
    time_ops = set(as_string_list(legacy_groups.get("time_ops", [])))
    create_ops = set(as_string_list(legacy_groups.get("create_ops", [])))

    if op in location_and_date_ops:
        branch_arg = str(legacy_fields.get("branch_arg") or "").strip()
        if branch_arg:
            branch = first_present(
                args.get(branch_arg),
                get_known_branch_from_variables(variables, agent_config),
            )
            if is_present(branch):
                args[branch_arg] = branch

        date_arg = str(legacy_fields.get("date_arg") or "").strip()
        date_paths = as_string_list(legacy_fields.get("date_paths", []))
        date_argument_aliases = as_string_list(legacy_fields.get("date_argument_aliases", []))
        if date_arg:
            date_value = first_present(
                args.get(date_arg),
                *[args.get(alias) for alias in date_argument_aliases if alias],
                *[deep_get(variables, path, "") for path in date_paths if path],
            )
            if is_present(date_value):
                args[date_arg] = date_value

        date_text_arg = str(legacy_fields.get("date_text_arg") or "").strip()
        date_text_paths = as_string_list(legacy_fields.get("date_text_paths", []))
        if date_text_arg:
            date_text = first_present(
                args.get(date_text_arg),
                *[deep_get(variables, path, "") for path in date_text_paths if path],
            )
            if is_present(date_text):
                args[date_text_arg] = date_text

    if op in time_ops:
        time_arg = str(legacy_fields.get("time_arg") or "").strip()
        time_paths = as_string_list(legacy_fields.get("time_paths", []))
        if time_arg:
            time_value = first_present(
                args.get(time_arg),
                *[deep_get(variables, path, "") for path in time_paths if path],
            )
            if is_present(time_value):
                args[time_arg] = time_value

    if op in create_ops:
        pending_path = str(legacy_fields.get("pending_path") or "").strip()
        pending_fields = as_string_list(legacy_fields.get("pending_fields", []))
        pending = deep_get(variables, pending_path, {}) if pending_path else {}
        if isinstance(pending, dict):
            for field in pending_fields:
                field_name = str(field or "").strip()
                if field_name and not is_present(args.get(field_name)) and is_present(pending.get(field_name)):
                    args[field_name] = pending.get(field_name)

        profile_path = str(legacy_fields.get("profile_path") or "").strip()
        profile_field_map = legacy_fields.get("profile_field_map", {})
        profile = deep_get(variables, profile_path, {}) if profile_path else {}
        if isinstance(profile, dict) and isinstance(profile_field_map, dict):
            for arg_name, source_path in profile_field_map.items():
                arg_name = str(arg_name or "").strip()
                source_path = str(source_path or "").strip()
                if not arg_name or not source_path or is_present(args.get(arg_name)):
                    continue
                value = deep_get(profile, source_path, "") if "." in source_path else profile.get(source_path)
                if is_present(value):
                    args[arg_name] = value

        confirmation_arg = str(legacy_fields.get("confirmation_arg") or "").strip()
        confirmation_paths = as_string_list(legacy_fields.get("confirmation_paths", []))
        if confirmation_arg and not is_present(args.get(confirmation_arg)):
            confirmed = first_present(*[
                deep_get(variables, path, "")
                for path in confirmation_paths
                if path
            ])
            if is_present(confirmed) or confirmed is False:
                args[confirmation_arg] = confirmed

    return args


def get_tool_operation_required_fields(
    agent_config: Dict[str, Any],
    tool_name: str,
    operation: str
) -> List[str]:
    required: List[str] = []

    for tool in agent_config.get("tools") or []:
        if not isinstance(tool, dict):
            continue

        if str(tool.get("name", "")) != str(tool_name):
            continue

        operation_config = (tool.get("operations") or {}).get(operation, {})

        if isinstance(operation_config, dict):
            required = [
                str(item)
                for item in operation_config.get("required", []) or []
                if str(item or "").strip()
            ]

        break

    overrides = agent_config.get("tool_required_field_overrides") or {}

    if isinstance(overrides, dict):
        for field in overrides.get(str(operation), []) or []:
            field_text = str(field or "").strip()
            if field_text and field_text not in required:
                required.append(field_text)

    return required


def missing_required_tool_inputs(
    operation: str,
    arguments: Dict[str, Any],
    required_fields: List[str]
) -> List[str]:
    missing: List[str] = []

    for field in required_fields:
        value = arguments.get(field)

        if field == "date":
            value = arguments.get("date")

        if field == "customer_confirmed_booking":
            if value is not True:
                missing.append(field)
            continue

        if not is_present(value):
            missing.append(field)

    return missing


def blocked_tool_call_result(
    tool_name: str,
    operation: str,
    arguments: Dict[str, Any],
    missing_inputs: List[str],
    agent_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    agent_config = agent_config or {}

    blocked_config = agent_config.get("blocked_tool_call_policy", {})
    if not isinstance(blocked_config, dict):
        blocked_config = {}

    answer_draft_map = blocked_config.get("answer_draft_by_missing_input", {})
    if not isinstance(answer_draft_map, dict):
        answer_draft_map = {}

    next_move_map = blocked_config.get("next_move_by_missing_input", {})
    if not isinstance(next_move_map, dict):
        next_move_map = {}

    default_answer_draft = str(blocked_config.get("default_answer_draft") or "MISSING_REQUIRED_FIELDS").strip()
    default_next_move = str(blocked_config.get("default_next_move") or "Ask for the next missing detail before calling the tool.").strip()

    answer_draft = default_answer_draft
    next_move = default_next_move

    for missing_input in missing_inputs:
        key = str(missing_input or "").strip()
        if key in answer_draft_map and str(answer_draft_map.get(key) or "").strip():
            answer_draft = str(answer_draft_map.get(key) or "").strip()
            break

    for missing_input in missing_inputs:
        key = str(missing_input or "").strip()
        if key in next_move_map and str(next_move_map.get(key) or "").strip():
            next_move = str(next_move_map.get(key) or "").strip()
            break

    must_do = blocked_config.get("must_do", [])
    must_not_do = blocked_config.get("must_not_do", [])

    if not isinstance(must_do, list):
        must_do = []

    if not isinstance(must_not_do, list):
        must_not_do = []

    return {
        "ok": True,
        "blocked_tool_call": True,
        "action": "ask_user",
        "tool_name": tool_name,
        "operation": operation,
        "arguments": arguments,
        "missing_inputs": missing_inputs,
        "answer_draft": answer_draft,
        "notes": "Tool call was safely blocked because required inputs were missing.",
        "response_brief": {
            "tone": str(blocked_config.get("tone") or "natural and helpful"),
            "language": str(blocked_config.get("language") or "same as user"),
            "reply_length": str(blocked_config.get("reply_length") or "short"),
            "next_move": next_move,
            "must_do": [str(item) for item in must_do if str(item or "").strip()],
            "must_not_do": [str(item) for item in must_not_do if str(item or "").strip()],
        }
    }

def validate_direct_tool_request(
    tool_name: str,
    operation: str,
    arguments: Dict[str, Any],
    variables: Dict[str, Any],
    agent_config: Dict[str, Any]
) -> Dict[str, Any]:
    normalized_arguments = normalize_direct_tool_arguments(
        operation=operation,
        arguments=arguments,
        variables=variables,
        agent_config=agent_config,
    )

    required_fields = get_tool_operation_required_fields(
        agent_config=agent_config,
        tool_name=tool_name,
        operation=operation,
    )

    missing_inputs = missing_required_tool_inputs(
        operation=operation,
        arguments=normalized_arguments,
        required_fields=required_fields,
    )

    if not missing_inputs:
        return {
            "ok": True,
            "arguments": normalized_arguments,
            "missing_inputs": [],
            "variable_updates": {},
            "tool_result": {},
        }

    variable_updates: Dict[str, Any] = {}

    missing_input_updates = agent_config.get("missing_input_variable_updates", {})
    if isinstance(missing_input_updates, dict):
        op_updates = missing_input_updates.get(operation, {})
        wildcard_updates = missing_input_updates.get("*", {})

        for update_map in [wildcard_updates, op_updates]:
            if not isinstance(update_map, dict):
                continue

            for missing_field in missing_inputs:
                field_updates = update_map.get(str(missing_field), {})
                if isinstance(field_updates, dict):
                    variable_updates.update(field_updates)

    if not variable_updates and operation == "list_available_slots" and "date" in missing_inputs:
        booking_config = get_subagent_config(agent_config, "booking")
        stage_path = str(booking_config.get("stage_path") or "booking.stage")
        awaiting_date_value = str(
            (booking_config.get("stages") or {}).get("awaiting_date") or "awaiting_date"
        )
        variable_updates[stage_path] = awaiting_date_value

    return {
        "ok": False,
        "arguments": normalized_arguments,
        "missing_inputs": missing_inputs,
        "variable_updates": variable_updates,
        "tool_result": blocked_tool_call_result(
            tool_name=tool_name,
            operation=operation,
            arguments=normalized_arguments,
            missing_inputs=missing_inputs,
            agent_config=agent_config,
        ),
    }



def subagent_observations_are_ok(observations: List[Dict[str, Any]]) -> bool:
    if not isinstance(observations, list):
        return True

    for obs in observations:
        if not isinstance(obs, dict):
            continue

        obs_result = obs.get("result")

        if isinstance(obs_result, dict) and obs_result.get("ok") is False:
            return False

    return True



def normalize_parallel_tool_requests(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_requests = manifest.get("parallel_tool_requests", [])
    if not isinstance(raw_requests, list):
        return []

    output: List[Dict[str, Any]] = []
    seen_call_keys: set = set()

    for index, item in enumerate(raw_requests):
        if not isinstance(item, dict):
            continue

        tool_name = str(item.get("tool_name") or item.get("requested_tool_name") or "").strip()
        payload = normalize_tool_payload(
            item.get("tool_request_payload")
            or item.get("payload")
            or {
                "operation": item.get("operation", ""),
                "arguments": item.get("arguments", item.get("args", {})),
            }
        )

        operation = str(payload.get("operation") or item.get("operation") or "").strip()
        arguments = payload.get("arguments", {}) or {}

        if not isinstance(arguments, dict):
            arguments = {}

        if not tool_name or not operation:
            continue

        try:
            args_key = json.dumps(arguments, sort_keys=True, ensure_ascii=False)
        except Exception:
            args_key = str(sorted(arguments.items()))

        call_key = f"{tool_name}::{operation}::{args_key}"
        if call_key in seen_call_keys:
            continue
        seen_call_keys.add(call_key)

        output.append({
            "request_id": str(item.get("request_id") or item.get("intent_id") or f"request_{index + 1}"),
            "intent_id": str(item.get("intent_id") or ""),
            "tool_name": tool_name,
            "operation": operation,
            "arguments": arguments,
            "purpose": str(item.get("purpose") or ""),
            "can_run_in_parallel": item.get("can_run_in_parallel", True),
        })

    return output


def should_execute_parallel_tool_requests(
    manifest: Dict[str, Any],
    selected_id: str,
    agent_config: Dict[str, Any],
) -> bool:
    requests = normalize_parallel_tool_requests(manifest)
    if not requests:
        return False

    mode = str(manifest.get("multi_intent_execution_mode") or "").strip().lower()
    if mode in {"parallel_tools", "parallel", "independent_tools"}:
        return True

    config = agent_config.get("multi_intent_execution", {})
    if not isinstance(config, dict):
        config = {}

    if config.get("execute_parallel_tool_requests_with_subagents") is True:
        return True

    return not bool(selected_id)


def aggregate_multi_tool_action(results: List[Dict[str, Any]]) -> str:
    actions = {
        str(item.get("action") or "").strip()
        for item in results
        if isinstance(item, dict) and str(item.get("action") or "").strip()
    }

    if "ask_user" in actions:
        return "ask_user"
    if "handoff" in actions:
        return "handoff"
    return "reply"


def run_parallel_direct_tool_requests(
    requests: List[Dict[str, Any]],
    variables: Dict[str, Any],
    agent_config: Dict[str, Any],
    tool_runner: ToolRunner,
) -> Dict[str, Any]:
    """
    Execute independent direct tool requests.

    Requests marked can_run_in_parallel=True run through ThreadPoolExecutor.
    Requests marked false are executed sequentially after the parallel batch.
    State updates from parallel calls are merged after completion.
    """
    parallel_reqs = [r for r in requests if r.get("can_run_in_parallel", True)]
    sequential_reqs = [r for r in requests if not r.get("can_run_in_parallel", True)]

    results: List[Dict[str, Any]] = []
    observations: List[Dict[str, Any]] = []
    updated_variables = dict(variables or {})

    try:
        max_workers = int(agent_config.get("max_parallel_tool_workers", 4) or 4)
    except Exception:
        max_workers = 4

    max_workers = max(1, min(len(parallel_reqs) or 1, max_workers))

    def execute_one(request: Dict[str, Any], vars_snapshot: Dict[str, Any]):
        tool_name = request.get("tool_name", "")
        operation = request.get("operation", "")
        arguments = request.get("arguments", {}) or {}

        validation = validate_direct_tool_request(
            tool_name=tool_name,
            operation=operation,
            arguments=arguments,
            variables=vars_snapshot,
            agent_config=agent_config,
        )
        normalized_arguments = validation.get("arguments", arguments)

        if not validation.get("ok", False):
            result = validation.get("tool_result", {}) or {}
            result.update({
                "request_id": request.get("request_id", ""),
                "intent_id": request.get("intent_id", ""),
                "tool_name": tool_name,
                "operation": operation,
                "arguments": normalized_arguments,
                "purpose": request.get("purpose", ""),
            })
            obs = {
                "request_id": request.get("request_id", ""),
                "intent_id": request.get("intent_id", ""),
                "tool_name": tool_name,
                "operation": operation,
                "arguments": normalized_arguments,
                "result": result,
            }
            return result, obs, validation.get("variable_updates", {}) or {}

        try:
            raw_result = tool_runner.call(
                tool_name=tool_name,
                operation=operation,
                arguments=normalized_arguments,
            )
        except Exception as exc:
            raw_result = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

        variable_updates: Dict[str, Any] = {}
        try:
            new_variables = apply_tool_update_rules(
                assistant_config=agent_config,
                variables=vars_snapshot,
                operation=operation,
                arguments=normalized_arguments,
                result=raw_result,
            )
            if isinstance(new_variables, dict):
                variable_updates = {
                    key: value
                    for key, value in new_variables.items()
                    if vars_snapshot.get(key) != value
                }
        except Exception:
            variable_updates = {}

        result = {
            **(raw_result if isinstance(raw_result, dict) else {"result": raw_result}),
            "request_id": request.get("request_id", ""),
            "intent_id": request.get("intent_id", ""),
            "tool_name": tool_name,
            "operation": operation,
            "arguments": normalized_arguments,
            "purpose": request.get("purpose", ""),
        }

        obs = {
            "request_id": request.get("request_id", ""),
            "intent_id": request.get("intent_id", ""),
            "tool_name": tool_name,
            "operation": operation,
            "arguments": normalized_arguments,
            "result": raw_result,
        }

        return result, obs, variable_updates

    if parallel_reqs and max_workers > 1:
        vars_snapshot = dict(updated_variables)
        futures_map = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for req in parallel_reqs:
                future = executor.submit(execute_one, req, vars_snapshot)
                futures_map[future] = req

            for future in as_completed(futures_map):
                req = futures_map[future]
                try:
                    result, obs, var_updates = future.result(timeout=35)
                    results.append(result)
                    observations.append(obs)
                    if isinstance(var_updates, dict):
                        updated_variables.update(var_updates)
                except Exception as exc:
                    error_result = {
                        "ok": False,
                        "error": f"Parallel execution error: {type(exc).__name__}: {exc}",
                        "request_id": req.get("request_id", ""),
                        "intent_id": req.get("intent_id", ""),
                        "tool_name": req.get("tool_name", ""),
                        "operation": req.get("operation", ""),
                    }
                    results.append(error_result)
                    observations.append({
                        "request_id": req.get("request_id", ""),
                        "intent_id": req.get("intent_id", ""),
                        "tool_name": req.get("tool_name", ""),
                        "operation": req.get("operation", ""),
                        "arguments": req.get("arguments", {}) or {},
                        "result": error_result,
                    })
    elif parallel_reqs:
        for req in parallel_reqs:
            result, obs, var_updates = execute_one(req, updated_variables)
            results.append(result)
            observations.append(obs)
            if isinstance(var_updates, dict):
                updated_variables.update(var_updates)

    for req in sequential_reqs:
        result, obs, var_updates = execute_one(req, updated_variables)
        results.append(result)
        observations.append(obs)
        if isinstance(var_updates, dict):
            updated_variables.update(var_updates)

    ok = all(not (isinstance(item, dict) and item.get("ok") is False) for item in results)

    return {
        "variables": updated_variables,
        "multi_tool_results": results,
        "tool_result": {
            "ok": ok,
            "multi_tool_results": results,
            "parallel_tool_requests": requests,
            "observations": observations,
            "action": aggregate_multi_tool_action(results),
            "answer_draft": "MULTI_TOOL_RESULTS",
            "notes": f"Executed {len(results)} tool request(s). Parallel: {len(parallel_reqs)}, Sequential: {len(sequential_reqs)}.",
            "tool_calls_used": len(results),
        },
    }



def get_post_subagent_chain_rules(agent_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Return config-driven post-subagent chain rules.

    Purpose:
    Sometimes the manifest selects only the first step of a dependent workflow
    (for example, a location lookup) even though the same user turn also asks
    for a downstream action that becomes executable after the first subagent
    updates variables. This rule family allows graph.py to run the downstream
    configured subagent in the same turn after the primary result.

    Preferred config keys:
    - post_subagent_chain_rules
    - post_execution_chain_rules
    - dependent_subagent_chains

    The derived compatibility rule uses configured subagent availability only;
    it does not embed user-facing phrases, branch names, dates, slots, or
    business wording. It can be disabled with:
      routing_guardrails.derive_post_subagent_chains_from_subagent_config=false
    """
    raw = agent_config.get("post_subagent_chain_rules")
    if not isinstance(raw, list):
        raw = agent_config.get("post_execution_chain_rules")
    if not isinstance(raw, list):
        raw = agent_config.get("dependent_subagent_chains")

    rules = [item for item in (raw or []) if isinstance(item, dict)] if isinstance(raw, list) else []
    if rules:
        return rules

    derived_enabled = get_config_bool(
        agent_config,
        "routing_guardrails.derive_post_subagent_chains_from_subagent_config",
        True,
    )

    if not derived_enabled:
        return []

    location_config = get_subagent_config(agent_config, "location")
    booking_config = get_subagent_config(agent_config, "booking")

    if not location_config or not booking_config:
        return []

    return [{
        "id": "derived_post_location_booking_chain",
        "enabled": True,
        "source_subagent_id": "location",
        "target_subagent_id": "booking",
        "only_when_target_not_already_run": True,
        "only_when_no_configured_chain_executed": True,
        "requires_known_branch": True,
        "requires_date_from_subagent": "booking",
        "target_must_handle": True,
        "response_strategy_append": (
            "After the configured location subagent resolves a branch, the same user turn "
            "still contains a configured date expression. Give the configured booking "
            "subagent a chance to handle the downstream availability/booking step in the "
            "same turn. Use the target subagent result only if it handles the turn."
        ),
    }]


def post_subagent_chain_rule_matches(
    rule: Dict[str, Any],
    source_subagent_id: str,
    already_run: set,
    manifest: Dict[str, Any],
    message: str,
    agent_config: Dict[str, Any],
    variables: Dict[str, Any],
    configured_chain_executed: bool,
) -> bool:
    if not isinstance(rule, dict) or rule.get("enabled", True) is False:
        return False

    source_id = unify_subagent_id(str(rule.get("source_subagent_id") or rule.get("primary_subagent_id") or "").strip())
    target_id = unify_subagent_id(str(rule.get("target_subagent_id") or rule.get("chained_subagent_id") or "").strip())

    if not source_id or not target_id:
        return False

    if unify_subagent_id(source_subagent_id) != source_id:
        return False

    if rule.get("only_when_target_not_already_run", True) and target_id in already_run:
        return False

    if rule.get("only_when_no_configured_chain_executed", True) and configured_chain_executed:
        return False

    if target_id not in SUBAGENT_EXECUTORS:
        return False

    target_config = get_subagent_config(agent_config, target_id)
    if isinstance(target_config, dict) and target_config.get("enabled", True) is False:
        return False

    if rule.get("requires_known_branch", False) and not has_known_branch(variables, agent_config):
        return False

    required_variable_paths = rule.get("required_variable_paths", [])
    if isinstance(required_variable_paths, list):
        for path in required_variable_paths:
            if not str(path or "").strip():
                continue
            if not is_present(deep_get(variables, str(path).strip(), "")):
                return False

    blocked_stage_values = {
        str(item or "").strip()
        for item in (rule.get("blocked_stage_values", []) or [])
        if str(item or "").strip()
    }
    stage_paths = rule.get("stage_paths", [])
    if blocked_stage_values and isinstance(stage_paths, list):
        for path in stage_paths:
            value = str(deep_get(variables, str(path or "").strip(), "") or "").strip()
            if value and value in blocked_stage_values:
                return False

    intent_labels = rule.get("requires_any_intent_labels", [])
    normalization = agent_config.get("normalization", {}) or {}
    if isinstance(intent_labels, list) and intent_labels:
        if not manifest_matches_any_intent_label(manifest, intent_labels, normalization):
            return False

    all_sources = rule.get("requires_all_message_phrase_sources", [])
    if isinstance(all_sources, list):
        for source in all_sources:
            if not isinstance(source, dict):
                continue
            if not message_matches_phrase_source(message, agent_config, source):
                return False

    any_sources = rule.get("requires_any_message_phrase_sources", [])
    if isinstance(any_sources, list) and any_sources:
        if not any(
            isinstance(source, dict) and message_matches_phrase_source(message, agent_config, source)
            for source in any_sources
        ):
            return False

    date_subagent = str(rule.get("requires_date_from_subagent") or "").strip()
    if date_subagent and not message_has_date_for_subagent(message, agent_config, date_subagent):
        return False

    return True


def append_post_chain_strategy_to_manifest(
    manifest: Dict[str, Any],
    rule: Dict[str, Any],
    source_subagent_id: str,
    target_subagent_id: str,
) -> Dict[str, Any]:
    patched = dict(manifest or {})

    strategy_append = str(rule.get("response_strategy_append") or "").strip()
    if strategy_append:
        previous_strategy = str(patched.get("response_strategy") or "").strip()
        patched["response_strategy"] = (
            f"{previous_strategy}\n{strategy_append}".strip()
            if previous_strategy
            else strategy_append
        )

    synthesis = patched.get("response_synthesis")
    if not isinstance(synthesis, dict):
        synthesis = {}

    applied = synthesis.get("post_subagent_chain_applied")
    if not isinstance(applied, list):
        applied = []

    applied.append({
        "rule_id": str(rule.get("id") or ""),
        "source_subagent_id": source_subagent_id,
        "target_subagent_id": target_subagent_id,
    })

    synthesis["post_subagent_chain_applied"] = applied
    patched["response_synthesis"] = synthesis

    return patched



def get_post_tool_required_input_continuation_rules(agent_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Return config-driven rules for post-tool/subagent required-input checks.

    Purpose:
    A first executor can complete successfully and make a downstream action
    partially possible, while a required downstream input is still missing. In
    that case the graph should not stop with the first executor's generic
    answer. It should ask only for the missing downstream input using configured
    operation policies.

    Preferred config keys:
    - post_tool_required_input_continuation_rules
    - post_required_input_continuation_rules
    - post_execution_required_input_rules

    The compatibility rule is derived only from configured subagent/tool
    metadata. Domain phrases, field names, dates, branches, and user examples
    stay in domain_bundle.json.
    """
    raw = agent_config.get("post_tool_required_input_continuation_rules")
    if not isinstance(raw, list):
        raw = agent_config.get("post_required_input_continuation_rules")
    if not isinstance(raw, list):
        raw = agent_config.get("post_execution_required_input_rules")

    rules = [item for item in (raw or []) if isinstance(item, dict)] if isinstance(raw, list) else []
    if rules:
        return rules

    derived_enabled = get_config_bool(
        agent_config,
        "routing_guardrails.derive_post_tool_required_input_continuation_from_subagent_config",
        True,
    )
    if not derived_enabled:
        return []

    location_config = get_subagent_config(agent_config, "location")
    booking_config = get_subagent_config(agent_config, "booking")
    if not location_config or not booking_config:
        return []

    operations = booking_config.get("operations", {}) if isinstance(booking_config.get("operations"), dict) else {}
    target_operation = str(operations.get("list_slots") or "").strip()
    if not target_operation:
        return []

    target_tool_name = str(booking_config.get("tool_name") or "").strip()
    if not target_tool_name:
        return []

    return [{
        "id": "derived_post_location_booking_required_inputs",
        "enabled": True,
        "source_subagent_id": "location",
        "target_subagent_id": "booking",
        "target_tool_name": target_tool_name,
        "target_operation": target_operation,
        "target_argument_mapping_key": "list_slots_arguments",
        "only_when_target_not_already_run": True,
        "only_when_no_configured_chain_executed": True,
        "requires_known_branch": True,
        "requires_missing_inputs": True,
        "requires_any_message_phrase_sources": [
            {
                "subagent_id": "booking",
                "paths": [
                    "availability_request_terms",
                    "trigger_phrases",
                ],
            }
        ],
        "response_strategy_append": (
            "After the configured primary executor completes, the same user turn still "
            "contains a configured downstream operation request. Validate the downstream "
            "operation from configured required inputs. If inputs are missing, ask only "
            "for those missing inputs and do not stop with the primary executor answer."
        ),
        "response_brief": {
            "must_do": [
                "use the completed primary executor result as context",
                "ask only for the configured missing downstream input",
                "do not claim the downstream operation was executed",
            ],
            "must_not_do": [
                "do not stop with only the primary executor result when a downstream input is missing",
                "do not invent downstream tool results",
            ],
        },
    }]


def post_tool_required_input_rule_matches(
    rule: Dict[str, Any],
    source_subagent_id: str,
    already_run: set,
    manifest: Dict[str, Any],
    message: str,
    agent_config: Dict[str, Any],
    variables: Dict[str, Any],
    configured_chain_executed: bool,
) -> bool:
    if not isinstance(rule, dict) or rule.get("enabled", True) is False:
        return False

    source_id = unify_subagent_id(str(rule.get("source_subagent_id") or rule.get("primary_subagent_id") or "").strip())
    target_id = unify_subagent_id(str(rule.get("target_subagent_id") or rule.get("chained_subagent_id") or "").strip())

    if not source_id or not target_id:
        return False
    if unify_subagent_id(source_subagent_id) != source_id:
        return False
    if rule.get("only_when_target_not_already_run", True) and target_id in already_run:
        return False
    if rule.get("only_when_no_configured_chain_executed", True) and configured_chain_executed:
        return False
    if target_id not in SUBAGENT_EXECUTORS:
        return False

    target_config = get_subagent_config(agent_config, target_id)
    if isinstance(target_config, dict) and target_config.get("enabled", True) is False:
        return False

    if rule.get("requires_known_branch", False) and not has_known_branch(variables, agent_config):
        return False

    required_variable_paths = rule.get("required_variable_paths", [])
    if isinstance(required_variable_paths, list):
        for path in required_variable_paths:
            path_text = str(path or "").strip()
            if path_text and not is_present(deep_get(variables, path_text, "")):
                return False

    blocked_stage_values = {
        str(item or "").strip()
        for item in (rule.get("blocked_stage_values", []) or [])
        if str(item or "").strip()
    }
    stage_paths = rule.get("stage_paths", [])
    if blocked_stage_values and isinstance(stage_paths, list):
        for path in stage_paths:
            value = str(deep_get(variables, str(path or "").strip(), "") or "").strip()
            if value and value in blocked_stage_values:
                return False

    normalization = agent_config.get("normalization", {}) or {}
    intent_labels = rule.get("requires_any_intent_labels", [])
    if isinstance(intent_labels, list) and intent_labels:
        if not manifest_matches_any_intent_label(manifest, intent_labels, normalization):
            return False

    all_sources = rule.get("requires_all_message_phrase_sources", [])
    if isinstance(all_sources, list):
        for source in all_sources:
            if not isinstance(source, dict):
                continue
            if not message_matches_phrase_source(message, agent_config, source):
                return False

    any_sources = rule.get("requires_any_message_phrase_sources", [])
    if isinstance(any_sources, list) and any_sources:
        if not any(
            isinstance(source, dict) and message_matches_phrase_source(message, agent_config, source)
            for source in any_sources
        ):
            return False

    return True


def find_tool_name_for_operation(agent_config: Dict[str, Any], operation: str) -> str:
    op = str(operation or "").strip()
    if not op:
        return ""

    for tool in agent_config.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        operations = tool.get("operations") or {}
        if isinstance(operations, dict) and op in operations:
            return str(tool.get("name") or "").strip()

    return ""


def get_tool_operation_config(agent_config: Dict[str, Any], tool_name: str, operation: str) -> Dict[str, Any]:
    for tool in agent_config.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        if tool_name and str(tool.get("name") or "") != str(tool_name):
            continue
        operations = tool.get("operations") or {}
        if isinstance(operations, dict) and isinstance(operations.get(operation), dict):
            return operations.get(operation) or {}
    return {}


def resolve_argument_mapping_value(source: Any, variables: Dict[str, Any]) -> Any:
    if not isinstance(source, str):
        return source

    source_text = source.strip()
    if not source_text:
        return ""

    if source_text.startswith("variables."):
        return deep_get(variables, source_text[len("variables."):], "")

    if source_text.startswith("literal:"):
        return source_text[len("literal:"):]

    return deep_get(variables, source_text, "")


def build_post_tool_continuation_arguments(
    rule: Dict[str, Any],
    target_config: Dict[str, Any],
    operation: str,
    variables: Dict[str, Any],
) -> Dict[str, Any]:
    args: Dict[str, Any] = {}

    explicit_args = rule.get("target_arguments") or rule.get("arguments")
    if isinstance(explicit_args, dict):
        args.update(explicit_args)

    mapping = rule.get("target_argument_mapping") or rule.get("argument_mapping")
    mapping_key = str(rule.get("target_argument_mapping_key") or rule.get("argument_mapping_key") or "").strip()

    if not isinstance(mapping, dict) and mapping_key and isinstance(target_config, dict):
        configured_mapping = target_config.get(mapping_key)
        if isinstance(configured_mapping, dict):
            mapping = configured_mapping

    operation_argument_maps = target_config.get("operation_arguments") if isinstance(target_config, dict) else None
    if not isinstance(mapping, dict) and isinstance(operation_argument_maps, dict):
        configured_mapping = operation_argument_maps.get(operation)
        if isinstance(configured_mapping, dict):
            mapping = configured_mapping

    if isinstance(mapping, dict):
        for arg_name, source in mapping.items():
            arg_text = str(arg_name or "").strip()
            if not arg_text or is_present(args.get(arg_text)):
                continue
            value = resolve_argument_mapping_value(source, variables)
            if is_present(value) or value is False:
                args[arg_text] = value

    return args


def get_operation_missing_input_response_brief(
    agent_config: Dict[str, Any],
    operation: str,
    missing_inputs: List[str],
) -> Dict[str, Any]:
    operation = str(operation or "").strip()
    if not operation:
        return {}

    candidates: List[Dict[str, Any]] = []

    # Common guardrail location used by service-center and future assistants.
    guardrails = agent_config.get("routing_guardrails", {})
    if isinstance(guardrails, dict):
        for guardrail in guardrails.values():
            if not isinstance(guardrail, dict):
                continue
            operations = guardrail.get("operations")
            if not isinstance(operations, dict):
                continue
            op_cfg = operations.get(operation)
            if isinstance(op_cfg, dict):
                candidates.append(op_cfg)

    # Tool operation config can also carry missing-input response briefs.
    tool_name = find_tool_name_for_operation(agent_config, operation)
    op_cfg = get_tool_operation_config(agent_config, tool_name, operation)
    if op_cfg:
        candidates.append(op_cfg)

    for missing_input in missing_inputs or []:
        key = str(missing_input or "").strip()
        if not key:
            continue
        specific_keys = [
            f"missing_{key}_response_brief",
            f"missing_{key}_brief",
        ]
        for candidate in candidates:
            for brief_key in specific_keys:
                brief = candidate.get(brief_key)
                if isinstance(brief, dict) and brief:
                    return brief
            for map_key in ["missing_input_response_briefs", "response_brief_by_missing_input", "missing_input_briefs"]:
                brief_map = candidate.get(map_key)
                if isinstance(brief_map, dict) and isinstance(brief_map.get(key), dict):
                    return brief_map.get(key) or {}

    for candidate in candidates:
        brief = candidate.get("missing_required_response_brief") or candidate.get("missing_inputs_response_brief")
        if isinstance(brief, dict) and brief:
            return brief

    return {}




def configured_post_tool_continuation_state_paths(agent_config: Dict[str, Any]) -> List[str]:
    """
    Generic internal continuation-state paths.

    These paths are not domain concepts; they store a paused downstream action
    after a prior executor completed and a configured required input was still
    missing. Future assistants may override the path list in config.
    """
    if not isinstance(agent_config, dict):
        return ["post_tool_required_input_continuation"]

    cfg = agent_config.get("post_tool_required_input_continuation_state")
    if not isinstance(cfg, dict):
        cfg = agent_config.get("post_required_input_continuation_state")
    if not isinstance(cfg, dict):
        cfg = {}

    paths = as_string_list(cfg.get("state_paths", []))
    if paths:
        return paths

    single_path = str(cfg.get("state_path") or "").strip()
    if single_path:
        return [single_path]

    return ["post_tool_required_input_continuation"]


def build_post_tool_continuation_state_payload(
    rule: Dict[str, Any],
    source_subagent_id: str,
    target_subagent_id: str,
    tool_name: str,
    operation: str,
    arguments: Dict[str, Any],
    missing_inputs: List[str],
) -> Dict[str, Any]:
    return {
        "enabled": True,
        "rule_id": str(rule.get("id") or ""),
        "source_subagent_id": source_subagent_id,
        "target_subagent_id": target_subagent_id,
        "tool_name": tool_name,
        "operation": operation,
        "arguments": dict(arguments or {}),
        "missing_inputs": [str(item) for item in missing_inputs or [] if str(item or "").strip()],
    }


def get_active_post_tool_required_input_continuation_subagent_id(
    agent_config: Dict[str, Any],
    variables: Dict[str, Any],
) -> str:
    """
    Route a follow-up turn back into the downstream executor that previously
    paused because required inputs were missing.

    The stored continuation is generic and config/path driven. It is intentionally
    not tied to service centers, bookings, dates, branches, or any specific
    assistant. The target executor is used only while the configured operation
    still has missing inputs after normal argument injection from current state.
    """
    if not isinstance(variables, dict):
        return ""

    for state_path in configured_post_tool_continuation_state_paths(agent_config):
        state_value = deep_get(variables, str(state_path or "").strip(), None)
        if not isinstance(state_value, dict) or state_value.get("enabled") is False:
            continue

        target_id = unify_subagent_id(str(state_value.get("target_subagent_id") or "").strip())
        if not target_id or target_id not in SUBAGENT_EXECUTORS:
            continue

        target_config = get_subagent_config(agent_config, target_id)
        if isinstance(target_config, dict) and target_config.get("enabled", True) is False:
            continue

        operation = str(state_value.get("operation") or "").strip()
        tool_name = str(state_value.get("tool_name") or "").strip()
        arguments = state_value.get("arguments") if isinstance(state_value.get("arguments"), dict) else {}

        if not operation:
            return target_id
        if not tool_name:
            tool_name = find_tool_name_for_operation(agent_config, operation)
        if not tool_name:
            return target_id

        validation = validate_direct_tool_request(
            tool_name=tool_name,
            operation=operation,
            arguments=arguments,
            variables=variables,
            agent_config=agent_config,
        )

        if validation.get("missing_inputs"):
            return target_id

    return ""


def merge_response_briefs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base or {}) if isinstance(base, dict) else {}
    if not isinstance(override, dict):
        return result

    for key in ["tone", "language", "reply_length", "next_move"]:
        value = override.get(key)
        if value not in [None, "", [], {}]:
            result[key] = value

    result["must_do"] = append_unique(
        result.get("must_do", []),
        [str(item) for item in override.get("must_do", []) or [] if str(item or "").strip()],
    )
    result["must_not_do"] = append_unique(
        result.get("must_not_do", []),
        [str(item) for item in override.get("must_not_do", []) or [] if str(item or "").strip()],
    )
    return result


def build_post_tool_missing_input_continuation(
    rule: Dict[str, Any],
    source_subagent_id: str,
    target_subagent_id: str,
    agent_config: Dict[str, Any],
    variables: Dict[str, Any],
    manifest: Dict[str, Any],
) -> Dict[str, Any]:
    target_config = get_subagent_config(agent_config, target_subagent_id)
    operation = str(rule.get("target_operation") or rule.get("operation") or "").strip()
    if not operation and isinstance(target_config, dict):
        operations = target_config.get("operations") if isinstance(target_config.get("operations"), dict) else {}
        operation = str(operations.get(str(rule.get("target_operation_key") or "")) or "").strip()
    if not operation:
        return {}

    tool_name = str(rule.get("target_tool_name") or rule.get("tool_name") or "").strip()
    if not tool_name and isinstance(target_config, dict):
        tool_name = str(target_config.get("tool_name") or "").strip()
    if not tool_name:
        tool_name = find_tool_name_for_operation(agent_config, operation)
    if not tool_name:
        return {}

    initial_arguments = build_post_tool_continuation_arguments(
        rule=rule,
        target_config=target_config if isinstance(target_config, dict) else {},
        operation=operation,
        variables=variables,
    )

    validation = validate_direct_tool_request(
        tool_name=tool_name,
        operation=operation,
        arguments=initial_arguments,
        variables=variables,
        agent_config=agent_config,
    )

    missing_inputs = validation.get("missing_inputs", []) or []
    if not missing_inputs:
        return {}

    if rule.get("requires_missing_inputs", True) is False:
        return {}

    variable_updates = dict(validation.get("variable_updates", {}) or {})

    continuation_state = build_post_tool_continuation_state_payload(
        rule=rule,
        source_subagent_id=source_subagent_id,
        target_subagent_id=target_subagent_id,
        tool_name=tool_name,
        operation=operation,
        arguments=validation.get("arguments", initial_arguments),
        missing_inputs=missing_inputs,
    )

    continuation_state_path = str(rule.get("continuation_state_path") or "").strip()
    if not continuation_state_path:
        continuation_state_path = configured_post_tool_continuation_state_paths(agent_config)[0]
    if continuation_state_path:
        variable_updates[continuation_state_path] = continuation_state

    updated_variables = apply_subagent_variable_patch(
        variables,
        variable_updates,
        [],
        assistant_config=agent_config,
    )

    tool_result = dict(validation.get("tool_result", {}) or {})
    op_brief = get_operation_missing_input_response_brief(agent_config, operation, missing_inputs)
    if op_brief:
        tool_result["response_brief"] = merge_response_briefs(tool_result.get("response_brief", {}) or {}, op_brief)

    rule_brief = rule.get("response_brief")
    if isinstance(rule_brief, dict) and rule_brief:
        tool_result["response_brief"] = merge_response_briefs(tool_result.get("response_brief", {}) or {}, rule_brief)

    tool_result.update({
        "ok": True,
        "blocked_tool_call": True,
        "post_tool_required_input_continuation": True,
        "source_subagent_id": source_subagent_id,
        "target_subagent_id": target_subagent_id,
        "tool_name": tool_name,
        "operation": operation,
        "arguments": validation.get("arguments", initial_arguments),
        "missing_inputs": missing_inputs,
        "answer_draft": str(tool_result.get("answer_draft") or "MISSING_REQUIRED_FIELDS"),
        "action": str(tool_result.get("action") or "ask_user"),
        "notes": str(
            tool_result.get("notes")
            or "Downstream operation was safely paused because configured required inputs are missing."
        ),
    })

    observation = {
        "subagent": target_subagent_id,
        "tool_name": tool_name,
        "operation": operation,
        "arguments": tool_result.get("arguments", {}),
        "result": tool_result,
    }

    executed_result = {
        "intent_id": str(rule.get("id") or "post_required_input_continuation"),
        "subagent": target_subagent_id,
        "action": tool_result.get("action", "ask_user"),
        "answer_draft": tool_result.get("answer_draft", "MISSING_REQUIRED_FIELDS"),
        "notes": tool_result.get("notes", ""),
        "tool_calls_used": 0,
        "observations": [observation],
        "response_brief": tool_result.get("response_brief", {}) if isinstance(tool_result.get("response_brief"), dict) else {},
        "target_tool_result": tool_result,
    }

    return {
        "variables": updated_variables,
        "tool_result": tool_result,
        "executed_result": executed_result,
        "observation": observation,
    }


def tool_execution_node(state: AgentState):
    manifest = state.get("manifest", {}) or {}
    agent_config = state.get("agent_config", {}) or {}
    variables = state.get("variables", {}) or {}
    schema = state.get("schema", {}) or {}
    message = last_user_message(state)

    variables = graph_extract_pending_required_details_from_patterns(
        agent_config=agent_config,
        variables=variables,
        message=message,
    )

    selected_id = unify_subagent_id(manifest.get("selected_subagent_id", ""))
    chained_id = unify_subagent_id(manifest.get("chained_subagent_id", ""))

    if booking_pending_requires_executor(
        agent_config=agent_config,
        variables=variables,
        message=message,
    ):
        selected_id = "booking"
        chained_id = ""

    tool_runner = ToolRunner(agent_config)
    all_observations: List[Dict[str, Any]] = []
    parallel_requests = normalize_parallel_tool_requests(manifest)

    if should_execute_parallel_tool_requests(
        manifest=manifest,
        selected_id=selected_id,
        agent_config=agent_config,
    ):
        return attach_smartness_to_tool_update(
            run_parallel_direct_tool_requests(
                requests=parallel_requests,
                variables=variables,
                agent_config=agent_config,
                tool_runner=tool_runner,
            ),
            previous_variables=variables,
            agent_config=agent_config,
            existing_variable_changes=state.get("variable_changes_this_turn", []) or [],
        )

    def run_executor(exec_id: str, vars_in: Dict[str, Any]):
        exec_id = unify_subagent_id(exec_id)
        executor = SUBAGENT_EXECUTORS.get(exec_id)

        if not executor:
            return None, vars_in, []

        scoped_vars = get_subagent_variable_scope(
            assistant_config=agent_config,
            subagent_name=getattr(executor, "name", exec_id),
            variables=vars_in,
        )

        context = SubagentContext(
            assistant_config=agent_config,
            schema=schema,
            variables=scoped_vars,
            user_message=message,
            history=subagent_history_from_messages(state.get("messages", [])),
            tool_runner=tool_runner,
            observations=[],
            max_tool_calls=int(agent_config.get("max_tool_calls", 4)),
        )

        try:
            result = executor.run(context)
        except Exception as exc:
            return None, vars_in, [{
                "subagent": exec_id,
                "error": f"{type(exc).__name__}: {exc}",
            }]

        if not result.handled:
            return None, vars_in, result.observations or []

        updated_variables = apply_subagent_variable_patch(
            vars_in,
            prepare_variable_updates_for_patch(result.variable_updates or {}),
            result.clear_variables or [],
            assistant_config=agent_config,
        )

        return result, updated_variables, result.observations or []

    executed_results: List[Dict[str, Any]] = []
    already_run = set()
    primary_result = None
    chained_result = None

    configured_chain_executed = False

    if selected_id:
        primary_result, variables, obs1 = run_executor(selected_id, variables)
        all_observations.extend(obs1)

        if primary_result and primary_result.handled:
            already_run.add(selected_id)
            executed_results.append({
                "intent_id": "primary",
                "subagent": selected_id,
                "answer_draft": primary_result.answer,
                "action": primary_result.action,
                "notes": primary_result.notes,
                "tool_calls_used": primary_result.tool_calls_used or 0,
                "observations": obs1,
            })

            if chained_id and chained_id != selected_id:
                chained_result, variables, obs2 = run_executor(chained_id, variables)
                all_observations.extend(obs2)

                if chained_result and chained_result.handled:
                    configured_chain_executed = True
                    already_run.add(chained_id)
                    executed_results.append({
                        "intent_id": "chained",
                        "subagent": chained_id,
                        "answer_draft": chained_result.answer,
                        "action": chained_result.action,
                        "notes": chained_result.notes,
                        "tool_calls_used": chained_result.tool_calls_used or 0,
                        "observations": obs2,
                    })

            for continuation_rule in get_post_tool_required_input_continuation_rules(agent_config):
                if not post_tool_required_input_rule_matches(
                    rule=continuation_rule,
                    source_subagent_id=selected_id,
                    already_run=already_run,
                    manifest=manifest,
                    message=message,
                    agent_config=agent_config,
                    variables=variables,
                    configured_chain_executed=configured_chain_executed,
                ):
                    continue

                target_id = unify_subagent_id(
                    str(
                        continuation_rule.get("target_subagent_id")
                        or continuation_rule.get("chained_subagent_id")
                        or ""
                    ).strip()
                )

                if not target_id or target_id in already_run:
                    continue

                continuation = build_post_tool_missing_input_continuation(
                    rule=continuation_rule,
                    source_subagent_id=selected_id,
                    target_subagent_id=target_id,
                    agent_config=agent_config,
                    variables=variables,
                    manifest=manifest,
                )

                if not continuation:
                    continue

                variables = continuation.get("variables", variables)
                observation = continuation.get("observation")
                if isinstance(observation, dict):
                    all_observations.append(observation)

                executed = continuation.get("executed_result")
                if isinstance(executed, dict):
                    executed_results.append(executed)

                already_run.add(target_id)
                manifest = append_post_chain_strategy_to_manifest(
                    manifest=manifest,
                    rule=continuation_rule,
                    source_subagent_id=selected_id,
                    target_subagent_id=target_id,
                )
                break

            for post_rule in get_post_subagent_chain_rules(agent_config):
                if not post_subagent_chain_rule_matches(
                    rule=post_rule,
                    source_subagent_id=selected_id,
                    already_run=already_run,
                    manifest=manifest,
                    message=message,
                    agent_config=agent_config,
                    variables=variables,
                    configured_chain_executed=configured_chain_executed,
                ):
                    continue

                target_id = unify_subagent_id(
                    str(
                        post_rule.get("target_subagent_id")
                        or post_rule.get("chained_subagent_id")
                        or ""
                    ).strip()
                )

                if not target_id or target_id in already_run:
                    continue

                post_result, variables, post_obs = run_executor(target_id, variables)
                all_observations.extend(post_obs)

                target_must_handle = post_rule.get("target_must_handle", True)
                if not post_result or not post_result.handled:
                    if target_must_handle:
                        continue
                    already_run.add(target_id)
                    continue

                already_run.add(target_id)
                executed_results.append({
                    "intent_id": str(post_rule.get("id") or "post_chain"),
                    "subagent": target_id,
                    "answer_draft": post_result.answer,
                    "action": post_result.action,
                    "notes": post_result.notes,
                    "tool_calls_used": post_result.tool_calls_used or 0,
                    "observations": post_obs,
                })

                manifest = append_post_chain_strategy_to_manifest(
                    manifest=manifest,
                    rule=post_rule,
                    source_subagent_id=selected_id,
                    target_subagent_id=target_id,
                )

    multi_intents_list = manifest.get("multi_intents", [])
    if not isinstance(multi_intents_list, list):
        multi_intents_list = []

    for intent in multi_intents_list:
        if not isinstance(intent, dict):
            continue

        intent_subagent_id = unify_subagent_id(
            str(intent.get("selected_subagent_id") or "").strip()
        )

        if not intent_subagent_id:
            continue
        if intent_subagent_id in already_run:
            continue
        if intent_subagent_id not in SUBAGENT_EXECUTORS:
            continue
        if not intent.get("needs_tool", True):
            continue

        intent_result, variables, intent_obs = run_executor(intent_subagent_id, variables)
        all_observations.extend(intent_obs)

        if intent_result and intent_result.handled:
            already_run.add(intent_subagent_id)
            executed_results.append({
                "intent_id": intent.get("intent_id", ""),
                "subagent": intent_subagent_id,
                "action": intent_result.action,
                "answer_draft": intent_result.answer,
                "notes": intent_result.notes,
                "tool_calls_used": intent_result.tool_calls_used or 0,
                "observations": intent_obs,
            })

    if executed_results:
        total_calls = sum(int(r.get("tool_calls_used", 0) or 0) for r in executed_results)
        primary_answer = executed_results[-1].get("answer_draft", "MULTI_INTENT_RESULTS")

        tool_result = {
            "ok": subagent_observations_are_ok(all_observations),
            "subagent": "+".join(sorted(already_run)),
            "multi_intent": len(executed_results) > 1,
            "action": aggregate_multi_tool_action(executed_results),
            "answer_draft": primary_answer,
            "observations": all_observations,
            "tool_calls_used": total_calls,
            "notes": f"Executed {len(executed_results)} subagent intent(s).",
        }

        latest_executed = executed_results[-1] if executed_results else {}
        if isinstance(latest_executed, dict):
            latest_response_brief = latest_executed.get("response_brief")
            if isinstance(latest_response_brief, dict) and latest_response_brief:
                tool_result["response_brief"] = latest_response_brief

            latest_target_tool_result = latest_executed.get("target_tool_result")
            if isinstance(latest_target_tool_result, dict) and latest_target_tool_result:
                tool_result["target_tool_result"] = latest_target_tool_result
                if latest_target_tool_result.get("blocked_tool_call"):
                    tool_result["blocked_tool_call"] = True
                    tool_result["missing_inputs"] = latest_target_tool_result.get("missing_inputs", [])
                    tool_result["operation"] = latest_target_tool_result.get("operation", "")
                    tool_result["tool_name"] = latest_target_tool_result.get("tool_name", "")

        if primary_result and chained_result and chained_result.handled:
            tool_result.update({
                "primary_subagent": selected_id,
                "chained_subagent": chained_id,
                "chained": True,
                "primary_answer_draft": primary_result.answer,
                "primary_notes": primary_result.notes,
                "chained_notes": chained_result.notes,
                "notes": f"Executed chained subagents plus {max(len(executed_results) - 2, 0)} extra multi_intents.",
            })

        return attach_smartness_to_tool_update({
            "variables": variables,
            "tool_result": tool_result,
            "multi_tool_results": executed_results,
            "manifest": manifest,
            "response_synthesis": manifest.get("response_synthesis", {}),
        }, previous_variables=state.get("variables", {}) or {}, agent_config=agent_config, existing_variable_changes=state.get("variable_changes_this_turn", []) or [])

    if selected_id and all_observations:
        return attach_smartness_to_tool_update({
            "variables": variables,
            "tool_result": {
                "ok": False,
                "subagent": selected_id,
                "operation": selected_id,
                "error": "Subagent execution failed or was not handled.",
                "action": "reply",
                "observations": all_observations,
            }
        }, previous_variables=state.get("variables", {}) or {}, agent_config=agent_config, existing_variable_changes=state.get("variable_changes_this_turn", []) or [])

    tool_name = manifest.get("requested_tool_name", "")
    payload = normalize_tool_payload(manifest.get("tool_request_payload", {}) or {})
    operation = payload.get("operation", "")
    arguments = payload.get("arguments", {}) or {}

    if tool_name and operation:
        validation = validate_direct_tool_request(
            tool_name=tool_name,
            operation=operation,
            arguments=arguments,
            variables=variables,
            agent_config=agent_config,
        )

        arguments = validation.get("arguments", arguments)

        if not validation.get("ok", False):
            updated_variables = apply_subagent_variable_patch(
                variables,
                validation.get("variable_updates", {}) or {},
                [],
                assistant_config=agent_config,
            )

            return attach_smartness_to_tool_update({
                "variables": updated_variables,
                "tool_result": validation.get("tool_result", {}),
            }, previous_variables=variables, agent_config=agent_config, existing_variable_changes=state.get("variable_changes_this_turn", []) or [])

        try:
            raw_result = tool_runner.call(
                tool_name=tool_name,
                operation=operation,
                arguments=arguments,
            )
        except Exception as exc:
            raw_result = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

        updated_variables = apply_tool_update_rules(
            assistant_config=agent_config,
            variables=variables,
            operation=operation,
            arguments=arguments,
            result=raw_result,
        )

        enriched_result = {
            **raw_result,
            "tool_name": tool_name,
            "operation": operation,
            "arguments": arguments,
        }

        return attach_smartness_to_tool_update({
            "variables": updated_variables,
            "tool_result": enriched_result,
            "multi_tool_results": [enriched_result],
        }, previous_variables=variables, agent_config=agent_config, existing_variable_changes=state.get("variable_changes_this_turn", []) or [])

    return attach_smartness_to_tool_update({
        "variables": variables,
        "tool_result": {
            "ok": False,
            "operation": str(operation or tool_name or selected_id or ""),
            "error": "Tool requested but no executable subagent/tool operation was available.",
            "selected_subagent_id": selected_id,
            "chained_subagent_id": chained_id,
            "requested_tool_name": tool_name,
            "tool_request_payload": payload,
        }
    }, previous_variables=state.get("variables", {}) or {}, agent_config=agent_config, existing_variable_changes=state.get("variable_changes_this_turn", []) or [])


def should_skip_subagent_reasoning_after_tool(state: AgentState) -> bool:
    """
    Skip the private subagent reasoning LLM when a deterministic executor already
    produced a clean, high-confidence result.

    Config-driven cost fix:
    assistant.subagent_reasoning_policy.skip_on_clean_executor_result
    assistant.subagent_reasoning_policy.min_manifest_confidence
    assistant.subagent_reasoning_policy.clean_actions
    """
    agent_config = state.get("agent_config", {}) or {}
    policy = agent_config.get("subagent_reasoning_policy", {})
    if not isinstance(policy, dict):
        policy = {}

    if policy.get("skip_on_clean_executor_result", True) is False:
        return False

    manifest = state.get("manifest", {}) or {}
    tool_result = state.get("tool_result", {}) or {}

    if not isinstance(tool_result, dict) or not tool_result:
        return False

    if state.get("multi_tool_results"):
        return False

    if tool_result.get("ok") is False:
        return False

    if tool_result.get("error"):
        return False

    answer_draft = str(tool_result.get("answer_draft") or "").strip()
    if not answer_draft:
        return False

    try:
        min_confidence = float(policy.get("min_manifest_confidence", 0.85) or 0.85)
    except Exception:
        min_confidence = 0.85

    if manifest_confidence(manifest) < min_confidence:
        return False

    clean_actions = policy.get("clean_actions", ["reply", "ask_user"])
    if isinstance(clean_actions, str):
        clean_actions = [clean_actions]
    if not isinstance(clean_actions, list):
        clean_actions = ["reply", "ask_user"]

    action = str(tool_result.get("action") or "").strip()
    if action and action not in {str(item or "").strip() for item in clean_actions}:
        return False

    return True


def decide_after_tool(state: AgentState) -> str:
    manifest = state.get("manifest", {}) or {}

    if manifest.get("needs_subagent_reasoning") and not should_skip_subagent_reasoning_after_tool(state):
        return "subagent_reasoning"

    return "response"


def subagent_reasoning_node(state: AgentState):
    message = last_user_message(state)
    subagent = state.get("selected_subagent", {}) or {}
    manifest = state.get("manifest", {}) or {}
    schema = state.get("schema", {}) or {}

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are the selected subagent's private analysis brain inside a configurable multi-tenant assistant engine. "
            "Deliberate internally, then end your response with one valid JSON object only. "
            "The JSON object must contain exactly these keys: understanding, user_goal, detected_intents, facts_to_use, facts_missing, "
            "variable_updates_to_consider, next_best_step, response_constraints, should_chain, recommended_chained_subagent_id, confidence. "
            "Do not write the final user-facing reply. "
            "Do not expose hidden reasoning, chain-of-thought, markdown, or prose outside the JSON. "
            "Prepare concise guidance for the final response. "
            "Use the manifest response brief as the main direction. "
            "Use only provided context, variables, memory, knowledge, tool result, and subagent instructions. "
            "Do not invent facts.",
        ),
        (
            "user",
            "Selected subagent:\n{subagent}\n\n"
            "Unified manifest:\n{manifest}\n\n"
            "Multi-intent context:\n{multi_intent_context}\n\n"
            "Assistant context:\n{context}\n\n"
            "Conversation summary:\n{summary}\n\n"
            "Variables:\n{variables}\n\n"
            "Memory:\n{memories}\n\n"
            "Knowledge:\n{knowledge}\n\n"
            "Tool result:\n{tool_result}\n\n"
            "Latest message:\n{message}",
        ),
    ])

    try:
        raw_response = (prompt | subagent_llm_raw).invoke({
            "subagent": safe_json({
                "id": subagent.get("id", ""),
                "name": subagent.get("name", ""),
                "goal": clip_text_head_tail(subagent.get("goal", ""), 380),
                "instructions": clip_text_head_tail(subagent.get("instructions", ""), 850),
                "allowed_actions": subagent.get("allowed_actions", []),
            }),
            "manifest": safe_json(compact_manifest(manifest), max_chars=2400),
            "multi_intent_context": safe_json({
                "multi_intents": state.get("multi_intents", []) or manifest.get("multi_intents", []),
                "response_synthesis": state.get("response_synthesis", {}) or manifest.get("response_synthesis", {}),
                "parallel_tool_requests": state.get("parallel_tool_requests", []) or manifest.get("parallel_tool_requests", []),
                "knowledge_queries": state.get("knowledge_queries", []) or manifest.get("knowledge_queries", []),
            }, max_chars=1800),
            "context": safe_json(compact_agent_context(state.get("agent_config", {}) or {}, subagent), max_chars=2800),
            "summary": clip_text(state.get("summary", ""), 520),
            "variables": safe_json(compact_variables(state.get("variables", {}) or {}, schema), max_chars=2200),
            "memories": compact_memories_for_final(state.get("memories", "")),
            "knowledge": compact_knowledge_for_final(state.get("knowledge", "No knowledge retrieved."), 1200),
            "tool_result": safe_json(state.get("tool_result", {}) or {}, max_chars=2800),
            "message": message,
        })

        raw_text = raw_response.content if hasattr(raw_response, "content") else str(raw_response)
        json_match = re.search(r"\{[\s\S]*\}", raw_text)
        data: Dict[str, Any] = {}

        if json_match:
            try:
                parsed = json.loads(json_match.group())
                if isinstance(parsed, dict):
                    data = parsed
            except Exception:
                data = {}

        def safe_str(value: Any) -> str:
            return str(value or "").strip()

        def safe_list_of_str(value: Any) -> List[str]:
            if not isinstance(value, list):
                return []
            return [str(item) for item in value if str(item or "").strip()]

        try:
            confidence = float(data.get("confidence", 0.7) or 0.7)
        except Exception:
            confidence = 0.7

        return {
            "subagent_analysis": {
                "understanding": safe_str(data.get("understanding")),
                "user_goal": safe_str(data.get("user_goal")),
                "detected_intents": safe_list_of_str(data.get("detected_intents", [])),
                "facts_to_use": safe_list_of_str(data.get("facts_to_use", [])),
                "facts_missing": safe_list_of_str(data.get("facts_missing", [])),
                "variable_updates_to_consider": data.get("variable_updates_to_consider", {}) if isinstance(data.get("variable_updates_to_consider"), dict) else {},
                "next_best_step": safe_str(data.get("next_best_step")),
                "response_constraints": safe_list_of_str(data.get("response_constraints", [])),
                "should_chain": bool(data.get("should_chain", False)),
                "recommended_chained_subagent_id": safe_str(data.get("recommended_chained_subagent_id")),
                "confidence": confidence,
            }
        }

    except Exception as exc:
        return {
            "subagent_analysis": {
                "understanding": "Subagent analysis failed.",
                "user_goal": "",
                "detected_intents": [],
                "facts_to_use": [],
                "facts_missing": [],
                "variable_updates_to_consider": {},
                "next_best_step": "Answer carefully and avoid unsupported facts.",
                "response_constraints": [f"Subagent error: {exc}"],
                "should_chain": False,
                "recommended_chained_subagent_id": "",
                "confidence": 0.4,
            }
        }


def get_template_policy_for_answer_draft(
    agent_config: Dict[str, Any],
    answer_draft: str
) -> Dict[str, Any]:
    policies = agent_config.get("template_response_policy", {}) or {}

    if not isinstance(policies, dict):
        return {}

    policy = policies.get(str(answer_draft or "").strip(), {})

    return policy if isinstance(policy, dict) else {}


def render_policy_template_answer(state: AgentState) -> str:
    """
    Render a deterministic configured response for high-risk operational states.

    This is config-driven:
    - template_response_policy[answer_draft].response_template
    - template_response_policy[answer_draft].deterministic_template
    - template_response_policy[answer_draft].safe_template

    It prevents the response LLM from rewriting grounded operational facts such
    as selected slot time, branch, date, available slots, missing fields, or IDs.
    """
    agent_config = state.get("agent_config", {}) or {}
    tool_result = state.get("tool_result", {}) or {}
    variables = state.get("variables", {}) or {}
    manifest = state.get("manifest", {}) or {}

    if not isinstance(tool_result, dict) or not tool_result:
        return ""

    answer_draft = str(tool_result.get("answer_draft") or "").strip()

    if not answer_draft:
        return ""

    policy = get_template_policy_for_answer_draft(agent_config, answer_draft)

    if not policy:
        return ""

    template = str(
        policy.get("response_template")
        or policy.get("deterministic_template")
        or policy.get("safe_template")
        or ""
    ).strip()

    if not template:
        safe_examples = policy.get("safe_examples", []) if isinstance(policy, dict) else []
        if isinstance(safe_examples, list):
            for example in safe_examples:
                example_text = str(example or "").strip()
                if example_text:
                    template = example_text
                    break

    if not template:
        return ""

    rendered = render_template(template, {
        "variables": variables,
        "tool_result": tool_result,
        "manifest": manifest,
        "latest_user_message": last_user_message(state),
        "answer_draft": answer_draft,
    }).strip()

    return rendered


def should_use_policy_template_answer(state: AgentState) -> bool:
    """
    Decide whether graph should bypass the response LLM and use the configured
    deterministic template for this answer_draft.

    This is intentionally generic. Domain-specific labels live in the bundle.
    """
    agent_config = state.get("agent_config", {}) or {}
    tool_result = state.get("tool_result", {}) or {}

    if not isinstance(tool_result, dict) or not tool_result:
        return False

    answer_draft = str(tool_result.get("answer_draft") or "").strip()

    if not answer_draft:
        return False

    policy = get_template_policy_for_answer_draft(agent_config, answer_draft)

    if not policy:
        return False

    if policy.get("force_template_answer") is True:
        return True

    if policy.get("deterministic_response") is True:
        return True

    if policy.get("must_use_grounded_template") is True:
        return True

    deterministic_labels = agent_config.get("deterministic_answer_draft_labels", [])
    if isinstance(deterministic_labels, list) and answer_draft in deterministic_labels:
        return True

    return False


def maybe_policy_template_answer(state: AgentState) -> str:
    if not should_use_policy_template_answer(state):
        return ""

    return render_policy_template_answer(state)


def completed_flow_closing_response_from_config(state: AgentState) -> str:
    """
    Render a configured short closing response after a completed flow.

    This is a generic safety bypass for completed-flow thanks/closing turns. It
    reads completion paths, closing message sources, answer labels, and response
    text from assistant config, then avoids sending completed operational facts
    back through the response LLM where IDs/details can be repeated.
    """
    agent_config = state.get("agent_config", {}) or {}
    variables = state.get("variables", {}) or {}
    message = last_user_message(state)

    target = completed_flow_closing_subagent_id(
        agent_config=agent_config,
        variables=variables,
        message=message,
    )
    if not target:
        return ""

    guardrails = agent_config.get("routing_guardrails", {})
    if not isinstance(guardrails, dict):
        return ""

    rule = guardrails.get("completed_flow_closing") or guardrails.get("completed_flow_closing_routing")
    if not isinstance(rule, dict) or rule.get("enabled", False) is False:
        return ""

    template = str(
        rule.get("response_template")
        or rule.get("deterministic_template")
        or rule.get("safe_template")
        or ""
    ).strip()

    answer_draft = str(
        rule.get("answer_draft")
        or rule.get("template_key")
        or rule.get("answer_label")
        or ""
    ).strip()

    if not template and answer_draft:
        policy = get_template_policy_for_answer_draft(agent_config, answer_draft)
        if isinstance(policy, dict):
            template = str(
                policy.get("response_template")
                or policy.get("deterministic_template")
                or policy.get("safe_template")
                or ""
            ).strip()
            if not template:
                examples = policy.get("safe_examples", [])
                if isinstance(examples, list):
                    for example in examples:
                        example_text = str(example or "").strip()
                        if example_text:
                            template = example_text
                            break

    if not template:
        return ""

    rendered = render_template(template, {
        "variables": variables,
        "manifest": state.get("manifest", {}) or {},
        "tool_result": state.get("tool_result", {}) or {},
        "latest_user_message": message,
        "answer_draft": answer_draft,
        "target_subagent_id": target,
    }).strip()

    return rendered



def get_response_temperature(manifest: Dict[str, Any], tool_result: Dict[str, Any]) -> float:
    """
    Dynamic response creativity/precision.

    Defaults are config-independent and generic:
    - tool/data/action replies stay precise
    - simple low-risk replies can be warmer
    Per-assistant overrides can be supplied in response_temperature.
    """
    temperature_config = {}
    # manifest may carry a copy through response_brief in future; keep this helper pure.

    risk = str(manifest.get("risk_level") or "low").lower().strip()
    action = str((tool_result or {}).get("action") or "").lower().strip()
    stage = str(manifest.get("conversation_stage") or "").lower().strip()
    has_tool_result = isinstance(tool_result, dict) and bool(tool_result)

    if risk == "high":
        return 0.1

    if has_tool_result:
        return 0.15

    if action == "ask_user":
        return 0.25

    if "slot" in stage or "lookup" in stage:
        return 0.2

    if manifest.get("simple_response_mode"):
        return 0.6

    return 0.45


def get_response_energy_instruction(agent_config: Dict[str, Any], manifest: Dict[str, Any]) -> str:
    emotion = str(manifest.get("customer_emotion") or "neutral").lower().strip()
    guidance = agent_config.get("emotion_response_guidance", {})

    if isinstance(guidance, dict):
        specific = str(guidance.get(emotion) or guidance.get("default") or "").strip()
        if specific:
            return specific

    return (
        "Adapt the reply tone to the detected customer emotion using the configured persona. "
        "Stay warm, clear, and useful without adding unsupported facts."
    )


def build_proactive_guidance(agent_config: Dict[str, Any], manifest: Dict[str, Any], tool_result: Dict[str, Any]) -> Dict[str, Any]:
    proactive_nudge = agent_config.get("proactive_nudge", {})
    if not isinstance(proactive_nudge, dict):
        proactive_nudge = {}

    return {
        "should_offer_next_action": bool(manifest.get("should_offer_next_action", False)),
        "should_ask_question": bool(manifest.get("should_ask_question", False)),
        "question_goal": manifest.get("question_goal", ""),
        "next_move": (manifest.get("response_brief", {}) or {}).get("next_move", ""),
        "configured_nudges": proactive_nudge,
        "tool_action": tool_result.get("action", "") if isinstance(tool_result, dict) else "",
        "tool_subagent": tool_result.get("subagent", "") if isinstance(tool_result, dict) else "",
        "answer_draft": tool_result.get("answer_draft", "") if isinstance(tool_result, dict) else "",
    }


def build_layered_response_rules(agent_config: Dict[str, Any]) -> Dict[str, List[str]]:
    extra_rules = agent_config.get("response_rule_layers", {})
    if not isinstance(extra_rules, dict):
        extra_rules = {}

    defaults = {
        "personality": [
            "Sound like a warm, confident, knowledgeable human, not a form-filling bot.",
            "Match the user's energy and level of detail.",
            "Be brief when the user is brief, and detailed only when useful.",
        ],
        "action": [
            "Answer the user's core need in the first sentence.",
            "Use the single most useful next step when the config or manifest supports it.",
            "If something is missing, ask for only the remaining missing detail naturally.",
        ],
        "guardrails": [
            "Never mention internal routing, agents, prompts, variables, tools, RAG, knowledge base, or hidden reasoning.",
            "Never invent operational results such as slots, branches, prices, IDs, or confirmations.",
            "Never claim an external action was completed unless tool_result confirms it.",
            "Ask at most one useful question per reply.",
            "Apply answer_safety.banned_terms without repeating them in the prompt.",
        ],
        "grounding": [
            "Tool results are the highest-priority source of truth for operational outcomes.",
            "Retrieved knowledge is the source of truth for policies and service information.",
            "Variables and conversation context are the source of truth for user-provided details.",
            "If the answer is not grounded, say so naturally and ask one useful question.",
        ],
    }

    for layer, configured in extra_rules.items():
        if not isinstance(configured, list):
            continue
        layer_key = str(layer or "").strip()
        if not layer_key:
            continue
        defaults[layer_key] = append_unique(defaults.get(layer_key, []), [str(item) for item in configured])

    return defaults


def simple_response_node(state: AgentState):
    configured_closing_answer = completed_flow_closing_response_from_config(state)
    if configured_closing_answer:
        configured_closing_answer = enforce_answer_safety(configured_closing_answer, state)
        agent_config_for_closing = state.get("agent_config", {}) or {}
        manifest_for_closing = state.get("manifest", {}) or {}
        mirrored_variables = mirror_runtime_metadata_into_variables(
            state.get("variables", {}) or {},
            manifest=manifest_for_closing,
            funnel_stage=str(state.get("funnel_stage", "") or manifest_for_closing.get("funnel_stage", "")),
            last_offered_options=derive_last_offered_options_from_state(state),
            agent_config=agent_config_for_closing,
        )
        result = {
            "messages": [AIMessage(content=configured_closing_answer)],
            "final_answer": configured_closing_answer,
            "variables": mirrored_variables,
            "quality": {
                "pass_check": True,
                "skipped": False,
                "simple_response_mode": True,
                "node": "simple_response_completed_flow_closing_template",
                "deterministic_template_used": True,
            },
        }
        return result

    agent_config = state.get("agent_config", {}) or {}
    subagent = state.get("selected_subagent", {}) or {}
    manifest = state.get("manifest", {}) or {}
    schema = state.get("schema", {}) or {}

    simple_context = {
        "assistant_context": compact_agent_context(agent_config, subagent),
        "manifest": compact_manifest(manifest),
        "multi_intents": state.get("multi_intents", []) or manifest.get("multi_intents", []),
        "response_synthesis": state.get("response_synthesis", {}) or manifest.get("response_synthesis", {}),
        "response_brief": manifest.get("response_brief", {}),
        "proactive_guidance": build_proactive_guidance(agent_config, manifest, {}),
        "tone_guidance": get_response_energy_instruction(agent_config, manifest),
        "best_guess_clarification": manifest.get("best_guess_clarification", {}) or {},
        "emotion_trajectory": state.get("emotion_trajectory", "") or manifest.get("emotion_trajectory", ""),
        "funnel_stage": state.get("funnel_stage", "") or manifest.get("funnel_stage", ""),
        "opener_context": state.get("opener_context", ""),
        "variable_changes_this_turn": state.get("variable_changes_this_turn", []) or [],
        "proactive_surface_items": state.get("proactive_surface_items", []) or [],
        "variables": compact_variables_for_response(state.get("variables", {}) or {}, schema, {}, agent_config),
        "answer_safety": agent_config.get("answer_safety", {}) or {},
    }

    response_rules = build_layered_response_rules(agent_config)

    system_instruction = f"""
{clip_text_head_tail(state.get('system_prompt', ''), 900)}

You are the voice of this configurable assistant. Generate a simple low-risk user-facing reply. If the user asks multiple low-risk questions, answer them naturally in one message. If the user corrected themselves or changed preference, follow the latest preference and do not dwell on the discarded one.

Use only this compact config-driven context:
{safe_json(simple_context, max_chars=5200)}

How to reply — follow in this priority order:
{safe_json(response_rules)}

Tone guidance for this message:
{get_response_energy_instruction(agent_config, manifest)}

Smartness instructions:
- Match the configured reply_length; very_short means 1-2 short sentences maximum.
- If a best-guess clarification exists, confirm the hypothesis naturally instead of asking an open question.
- If a change was detected, acknowledge it briefly.

{state.get('language_instruction', '')}
"""

    messages = [SystemMessage(content=system_instruction)] + list(state["messages"][-SIMPLE_RESPONSE_HISTORY_LIMIT:])
    dynamic_response_llm = llm(
        get_response_model(state, {}, simple_response=True),
        temperature=get_response_temperature(manifest, {}),
        max_tokens=MAX_OUTPUT_TOKENS,
    )
    response = dynamic_response_llm.invoke(messages)
    answer = response.content if hasattr(response, "content") else str(response)
    answer = enforce_answer_safety(answer, state)

    mirrored_variables = mirror_runtime_metadata_into_variables(
        state.get("variables", {}) or {},
        manifest=manifest,
        funnel_stage=str(state.get("funnel_stage", "") or manifest.get("funnel_stage", "")),
        last_offered_options=derive_last_offered_options_from_state(state),
        agent_config=agent_config,
    )

    return {
        "messages": [AIMessage(content=answer)],
        "final_answer": answer,
        "variables": mirrored_variables,
        "quality": {
            "pass_check": True,
            "skipped": False,
            "simple_response_mode": True,
            "node": "simple_response_pre_quality_guard",
        },
    }

def response_node(state: AgentState):
    derived_last_offered_options = derive_last_offered_options_from_state(state)
    configured_closing_answer = completed_flow_closing_response_from_config(state)

    if configured_closing_answer:
        configured_closing_answer = enforce_answer_safety(configured_closing_answer, state)
        result = {
            "messages": [AIMessage(content=configured_closing_answer)],
            "final_answer": configured_closing_answer,
            "quality": {
                "node": "response_completed_flow_closing_template",
                "pre_quality_guard": True,
                "deterministic_template_used": True,
            },
        }
        mirrored_variables = mirror_runtime_metadata_into_variables(
            state.get("variables", {}) or {},
            manifest=state.get("manifest", {}) or {},
            funnel_stage=str(state.get("funnel_stage", "") or (state.get("manifest", {}) or {}).get("funnel_stage", "")),
            last_offered_options=derived_last_offered_options,
            agent_config=state.get("agent_config", {}) or {},
        )
        if mirrored_variables != (state.get("variables", {}) or {}):
            result["variables"] = mirrored_variables
        if derived_last_offered_options:
            result["last_offered_options"] = derived_last_offered_options
        return result

    deterministic_answer = maybe_policy_template_answer(state)

    if deterministic_answer:
        deterministic_answer = enforce_answer_safety(deterministic_answer, state)
        result = {
            "messages": [AIMessage(content=deterministic_answer)],
            "final_answer": deterministic_answer,
            "quality": {
                "node": "response_policy_template",
                "pre_quality_guard": True,
                "deterministic_template_used": True,
            },
        }
        mirrored_variables = mirror_runtime_metadata_into_variables(
            state.get("variables", {}) or {},
            manifest=state.get("manifest", {}) or {},
            funnel_stage=str(state.get("funnel_stage", "") or (state.get("manifest", {}) or {}).get("funnel_stage", "")),
            last_offered_options=derived_last_offered_options,
            agent_config=state.get("agent_config", {}) or {},
        )
        if mirrored_variables != (state.get("variables", {}) or {}):
            result["variables"] = mirrored_variables
        if derived_last_offered_options:
            result["last_offered_options"] = derived_last_offered_options
        return result

    agent_config = state.get("agent_config", {}) or {}
    subagent = state.get("selected_subagent", {}) or {}
    manifest = state.get("manifest", {}) or {}
    analysis = state.get("subagent_analysis", {}) or {}
    variables = state.get("variables", {}) or {}
    schema = state.get("schema", {}) or {}
    knowledge = state.get("knowledge", "No knowledge retrieved.")
    memories = state.get("memories", "No relevant memories retrieved.")
    tool_result = state.get("tool_result", {}) or {}

    effective_manifest = compact_manifest(manifest)
    tool_response_brief = tool_result.get("response_brief") if isinstance(tool_result, dict) else None

    if isinstance(tool_response_brief, dict) and tool_response_brief:
        effective_manifest["response_brief"] = tool_response_brief

    compact_response_variables = compact_variables_for_response(
        variables=variables,
        schema=schema,
        tool_result=tool_result,
        agent_config=agent_config,
    )
    response_guidance = build_response_guidance_block(agent_config, tool_result)

    persona = agent_config.get("persona", {}) or {}
    persona_desc = ""

    if isinstance(persona, dict) and persona:
        persona_desc = "\n".join([
            f"- Character: {persona.get('character', '')}",
            f"- Voice: {persona.get('voice', '')}",
            f"- When selling: {persona.get('energy_when_selling', '')}",
            f"- When helping: {persona.get('energy_when_helping', '')}",
            f"- When closing: {persona.get('energy_when_closing', '')}",
        ])

    user_need_summary = " | ".join(filter(None, [
        str(manifest.get("user_intent") or ""),
        str(manifest.get("conversation_stage") or ""),
        str(manifest.get("response_strategy") or "")[:200],
    ]))

    facts_available: List[str] = []

    if tool_result and isinstance(tool_result, dict) and tool_result.get("ok"):
        facts_available.append(
            f"Tool succeeded: {tool_result.get('operation', tool_result.get('subagent', 'tool'))}"
        )
        if tool_result.get("answer_draft"):
            facts_available.append(f"Executor draft: {str(tool_result.get('answer_draft', ''))[:300]}")

    if knowledge and "NO_CONFIDENT_KNOWLEDGE_FOUND" not in str(knowledge):
        facts_available.append(f"Retrieved knowledge available ({len(str(knowledge))} chars)")

    if memories and "No relevant memories" not in str(memories):
        facts_available.append("User memory available")

    multi_results = (state.get("multi_tool_results", []) or (tool_result.get("multi_tool_results", []) if isinstance(tool_result, dict) else [])) or []
    synthesis_instruction = ""

    if len(multi_results) > 1:
        intent_summaries: List[str] = []

        for result in multi_results:
            if not isinstance(result, dict):
                continue

            label = result.get("intent_id") or result.get("subagent") or "intent"
            draft = str(result.get("answer_draft") or result.get("message") or "")[:200]
            intent_summaries.append(f"  [{label}]: {draft}")

        if intent_summaries:
            synthesis_instruction = (
                "This turn resolved multiple intents. Synthesize all results into ONE coherent response. "
                "Do not list them mechanically:\n"
                + "\n".join(intent_summaries)
            )

    system_instruction = f"""
{clip_text_head_tail(state.get('system_prompt', ''), 1200)}

You are the voice of this assistant. Write the exact reply the user will read.
Think like a brilliant, warm human expert — not a form-filling bot.

=== WHO YOU ARE ===
Goal: {clip_text(agent_config.get('assistant_goal', ''), 300)}
Style: {clip_text(agent_config.get('conversation_style', ''), 300)}
{f"Persona:{chr(10)}{persona_desc}" if persona_desc else ""}

=== WHAT THE USER NEEDS RIGHT NOW ===
{user_need_summary}

Emotion detected: {manifest.get('customer_emotion', 'neutral')}
Emotion trajectory: {state.get('emotion_trajectory', '') or manifest.get('emotion_trajectory', '')}
Funnel stage: {state.get('funnel_stage', '') or manifest.get('funnel_stage', '')}
Tone guidance: {get_response_energy_instruction(agent_config, manifest)}

Smart clarification: {safe_json(manifest.get('best_guess_clarification', {}) or {}, max_chars=500)}
Variable changes this turn: {safe_json(state.get('variable_changes_this_turn', []) or [], max_chars=700)}
Proactive info to weave in naturally if relevant: {safe_json(state.get('proactive_surface_items', []) or [], max_chars=700)}
Failure recovery context: {safe_json(build_failure_recovery_context(tool_result, agent_config) or state.get('failure_recovery_context', {}) or {}, max_chars=700)}
Progressive display context: {safe_json(state.get('progressive_display_context', {}) or build_progressive_display_context(state), max_chars=900)}

=== WHAT YOU KNOW AS FACTS ===
Variables (user's known state):
{safe_json(compact_response_variables, max_chars=1800)}

{f"Tool result: {safe_json(tool_result, max_chars=1400)}" if tool_result else "No tool result this turn."}

{f"Knowledge:{chr(10)}{compact_knowledge_for_final(knowledge, 1200)}" if knowledge and 'NO_CONFIDENT_KNOWLEDGE_FOUND' not in str(knowledge) else "No knowledge retrieved."}

{f"User memory:{chr(10)}{compact_memories_for_final(memories)}" if memories and 'No relevant memories' not in str(memories) else ""}

{f"Conversation context:{chr(10)}{clip_text(state.get('summary', ''), 400)}" if state.get('summary') else ""}

{synthesis_instruction}

=== HOW TO REPLY ===
1. PERSONALITY: Sound human, warm, confident. Match the user's energy level.
2. ACTION: Answer the core need first. Then offer the single most useful next step.
3. VARIABLES: Everything in "Variables" is already known — never ask for it again.
4. PROACTIVE: {(manifest.get('response_brief', {}) or {}).get('next_move', 'Help the user move forward naturally.')}
5. GUARDRAILS: Never mention internal routing, agents, RAG, variables, or tools.
   Never invent slots, branches, prices, IDs, or confirmations.
   Ask at most ONE question per reply.
   If best_guess_clarification.ask_confirm=true, lead with the configured hypothesis as a natural yes/no confirmation, not an open-ended question.
   If variable_changes_this_turn is present, acknowledge the change briefly before moving forward.
   If proactive_surface_items are present, mention only items that are directly helpful and do it naturally, not as a separate dump.
   Respect the reply_length target from response_brief, especially very_short for brief user messages.

Must do: {safe_json((manifest.get('response_brief', {}) or {}).get('must_do', []))}
Must not do: {safe_json((manifest.get('response_brief', {}) or {}).get('must_not_do', []))}

Response guidance mode: {response_guidance.get('mode', 'layered_response_rules')}
{safe_json(response_guidance.get('policy', {}), max_chars=1800)}

{state.get('language_instruction', '')}
"""

    messages = [SystemMessage(content=system_instruction)] + list(state["messages"][-SIMPLE_RESPONSE_HISTORY_LIMIT:])
    dynamic_response_llm = llm(
        get_response_model(state, tool_result, simple_response=False),
        temperature=get_response_temperature(manifest, tool_result),
        max_tokens=MAX_OUTPUT_TOKENS,
    )
    response = dynamic_response_llm.invoke(messages)
    answer = response.content if hasattr(response, "content") else str(response)
    answer = enforce_answer_safety(answer, state)

    result = {
        "messages": [AIMessage(content=answer)],
        "final_answer": answer,
        "quality": {
            "node": "response_pre_quality_guard",
            "pre_quality_guard": True,
        },
    }
    mirrored_variables = mirror_runtime_metadata_into_variables(
        variables,
        manifest=manifest,
        funnel_stage=str(state.get("funnel_stage", "") or manifest.get("funnel_stage", "")),
        last_offered_options=derived_last_offered_options,
        agent_config=agent_config,
    )
    if mirrored_variables != variables:
        result["variables"] = mirrored_variables
    if derived_last_offered_options:
        result["last_offered_options"] = derived_last_offered_options
    return result

def should_skip_confirmed_record_id_append(
    state: AgentState,
    answer: str,
    agent_config: Dict[str, Any],
    safety_config: Dict[str, Any],
) -> bool:
    """
    Decide whether the confirmed-record ID appender should be suppressed.

    The appender is useful for newly confirmed actions, but completed-flow
    closings and other configured answer drafts must remain short. This helper
    is fully config-driven: it reads answer-draft labels, template policy flags,
    and completed-flow closing routing config instead of embedding domain labels
    or phrases in Python.
    """
    tool_result = state.get("tool_result", {}) or {}
    answer_draft = ""
    if isinstance(tool_result, dict):
        answer_draft = str(tool_result.get("answer_draft") or "").strip()

    skip_label_keys = [
        "do_not_append_visit_id_for_answer_drafts",
        "do_not_append_record_id_for_answer_drafts",
        "do_not_append_confirmed_record_id_for_answer_drafts",
        "closing_answer_draft_labels",
        "short_closing_answer_draft_labels",
    ]

    skip_labels: List[str] = []
    for key in skip_label_keys:
        skip_labels = append_unique(skip_labels, as_string_list(safety_config.get(key, [])))

    if answer_draft and answer_draft in set(skip_labels):
        return True

    if bool(safety_config.get("do_not_append_confirmed_record_id_on_closing", False)):
        closing_labels = set(as_string_list(safety_config.get("closing_answer_draft_labels", [])))
        if answer_draft and answer_draft in closing_labels:
            return True

    if answer_draft:
        policy = get_template_policy_for_answer_draft(agent_config, answer_draft)
        if isinstance(policy, dict):
            policy_skip = bool(
                policy.get("suppress_confirmed_record_id_append", False)
                or policy.get("do_not_append_confirmed_record_id", False)
                or policy.get("do_not_append_record_id", False)
                or policy.get("do_not_append_visit_id", False)
            )
            if policy_skip:
                return True

            if policy.get("deterministic_response") is True and policy.get("force_template_answer") is True:
                no_id_for_deterministic = bool(
                    policy.get("completed_flow_short_closing", False)
                    or policy.get("state") in {"completed_flow_short_closing", "short_closing"}
                )
                if no_id_for_deterministic:
                    return True

    # Existing completed-flow routing check remains as a generic fallback for
    # configs that suppress IDs by completed state/message source rather than
    # by answer-draft label.
    closing_target = completed_flow_closing_subagent_id(
        agent_config=agent_config,
        variables=state.get("variables", {}) or {},
        message=last_user_message(state),
    )
    if closing_target:
        guardrails = agent_config.get("routing_guardrails", {}) or {}
        closing_rule = {}
        if isinstance(guardrails, dict):
            raw_closing_rule = guardrails.get("completed_flow_closing") or guardrails.get("completed_flow_closing_routing")
            if isinstance(raw_closing_rule, dict):
                closing_rule = raw_closing_rule
        suppress_id_append = closing_rule.get("suppress_confirmed_record_id_append", True)
        if suppress_id_append is not False:
            return True

    return False

def pre_response_guardrail_node(state: AgentState):
    """
    Lightweight config-driven post-response guardrail.

    Currently it can append the configured confirmed record ID line when an
    action has just been confirmed and the response omitted that ID.
    """
    answer = str(state.get("final_answer", "") or "").strip()

    if not answer:
        return {}

    agent_config = state.get("agent_config", {}) or {}
    safety_config = agent_config.get("answer_safety", {}) or {}
    if not isinstance(safety_config, dict):
        safety_config = {}

    append_id = bool(
        safety_config.get("append_visit_id_on_confirmed_booking", False)
        or safety_config.get("append_record_id_on_confirmed_action", False)
    )

    if not append_id:
        return {}

    if should_skip_confirmed_record_id_append(state, answer, agent_config, safety_config):
        return {}

    if not create_booking_confirmed(state):
        return {}

    record_id = extract_visit_id_from_state(state)

    if not record_id or record_id in answer:
        return {}

    id_label = str(
        safety_config.get("visit_id_label")
        or safety_config.get("record_id_label")
        or ""
    ).strip()
    id_format = str(
        safety_config.get("visit_id_format")
        or safety_config.get("record_id_format")
        or "{label}: {id}"
    ).strip()

    if id_label:
        id_line = id_format.replace("{label}", id_label).replace("{id}", record_id)
    else:
        id_line = record_id

    updated_answer = f"{answer}\n{id_line}".strip()

    return {
        "final_answer": updated_answer,
        "messages": [AIMessage(content=updated_answer)],
    }


def extract_visit_id_from_state(state: AgentState) -> str:
    agent_config = state.get("agent_config", {}) or {}
    variables = state.get("variables", {}) or {}
    tool_result = state.get("tool_result", {}) or {}

    id_paths = as_string_list(
        agent_config.get("confirmed_record_id_paths", [])
        or agent_config.get("visit_id_paths", [])
        or agent_config.get("confirmation_id_paths", [])
    )

    for path in id_paths:
        value = str(deep_get(variables, path) or "").strip()
        if value:
            return value

    if isinstance(tool_result, dict):
        for path in id_paths:
            value = str(deep_get(tool_result, path) or "").strip()
            if value:
                return value

        direct_fields = as_string_list(agent_config.get("confirmed_record_id_fields", []))
        for field in direct_fields:
            value = str(tool_result.get(field) or "").strip()
            if value:
                return value

        observations = tool_result.get("observations")
        if isinstance(observations, list):
            for obs in observations:
                if not isinstance(obs, dict):
                    continue
                result = obs.get("result") or {}
                if not isinstance(result, dict):
                    continue

                for path in id_paths:
                    value = str(deep_get(result, path) or "").strip()
                    if value:
                        return value

                for field in direct_fields:
                    value = str(result.get(field) or "").strip()
                    if value:
                        return value

    return ""

def create_booking_confirmed(state: AgentState) -> bool:
    agent_config = state.get("agent_config", {}) or {}
    tool_result = state.get("tool_result", {}) or {}
    variables = state.get("variables", {}) or {}

    confirmed_statuses = as_string_list(
        agent_config.get("booking_confirmed_statuses", [])
        or agent_config.get("confirmed_statuses", [])
    )
    confirmed_status_set = {status.lower() for status in confirmed_statuses if status}

    status_paths = as_string_list(
        agent_config.get("booking_status_paths", [])
        or [agent_config.get("booking_status_path", "")]
    )

    for status_path in status_paths:
        status = str(deep_get(variables, status_path) or "").lower().strip()
        if status and status in confirmed_status_set:
            return True

    confirmed_operation = str(
        agent_config.get("booking_confirmed_operation")
        or agent_config.get("confirmed_operation")
        or ""
    ).strip()

    operation_ok_paths = as_string_list(agent_config.get("confirmed_operation_ok_paths", []))

    if isinstance(tool_result, dict):
        if confirmed_operation and tool_result.get("operation") == confirmed_operation and tool_result.get("ok") is True:
            return True

        for path in operation_ok_paths:
            value = deep_get(tool_result, path)
            if value is True:
                return True

        observations = tool_result.get("observations")
        if isinstance(observations, list):
            for obs in observations:
                if not isinstance(obs, dict):
                    continue

                result = obs.get("result") or {}

                if confirmed_operation and obs.get("operation") == confirmed_operation:
                    if isinstance(result, dict) and result.get("ok") is True:
                        return True

                for path in operation_ok_paths:
                    value = deep_get(result if isinstance(result, dict) else obs, path)
                    if value is True:
                        return True

    return False

def compact_template_response_policy(
    agent_config: Dict[str, Any],
    tool_result: Optional[Dict[str, Any]] = None,
    max_policies: int = 8,
) -> Dict[str, Any]:
    policies = agent_config.get("template_response_policy", {}) or {}

    if not isinstance(policies, dict):
        return {}

    selected_keys: List[str] = []

    answer_draft = ""
    if isinstance(tool_result, dict):
        answer_draft = str(tool_result.get("answer_draft") or "").strip()

    if answer_draft and answer_draft in policies:
        selected_keys.append(answer_draft)

    priority_keys = as_string_list(agent_config.get("template_policy_priority_keys", []))
    for key in priority_keys:
        if key in policies and key not in selected_keys:
            selected_keys.append(key)
        if len(selected_keys) >= max_policies:
            break

    for key in policies:
        if key not in selected_keys:
            selected_keys.append(key)
        if len(selected_keys) >= max_policies:
            break

    output: Dict[str, Any] = {}

    for key in selected_keys[:max_policies]:
        policy = policies.get(key)
        if not isinstance(policy, dict):
            continue

        output[key] = {
            "state": policy.get("state", ""),
            "must_do": policy.get("must_do", [])[:8] if isinstance(policy.get("must_do"), list) else [],
            "must_not_do": policy.get("must_not_do", [])[:8] if isinstance(policy.get("must_not_do"), list) else [],
            "safe_examples": policy.get("safe_examples", [])[:3] if isinstance(policy.get("safe_examples"), list) else [],
            "banned_examples": policy.get("banned_examples", [])[:3] if isinstance(policy.get("banned_examples"), list) else [],
        }

    return output

def configured_booking_template_labels(
    agent_config: Dict[str, Any],
    key_prefixes: Optional[List[str]] = None,
    value_prefixes: Optional[List[str]] = None,
) -> List[str]:
    """
    Backward-compatible helper for booking template labels.

    Uses get_subagent_config so it works for both dict-format and list-format
    subagent configs. The prefixes are caller/config supplied; no assistant
    wording is embedded here.
    """
    booking_config = get_subagent_config(agent_config, "booking")
    templates = booking_config.get("templates", {}) if isinstance(booking_config, dict) else {}

    if not isinstance(templates, dict):
        return []

    output: List[str] = []
    key_prefixes = key_prefixes or []
    value_prefixes = value_prefixes or []

    for key, value in templates.items():
        key_text = str(key or "")
        value_text = str(value or "")

        key_match = any(key_text.startswith(prefix) for prefix in key_prefixes)
        value_match = any(value_text.startswith(prefix) for prefix in value_prefixes)

        if (key_match or value_match) and value_text and value_text not in output:
            output.append(value_text)

    return output

def get_tool_answer_draft(state: AgentState) -> str:
    tool_result = state.get("tool_result", {}) or {}

    if not isinstance(tool_result, dict):
        return ""

    answer_draft = str(tool_result.get("answer_draft") or "").strip()

    if answer_draft:
        return answer_draft

    primary_answer = str(tool_result.get("primary_answer_draft") or "").strip()
    if primary_answer:
        return primary_answer

    return ""


def get_tool_action(state: AgentState) -> str:
    tool_result = state.get("tool_result", {}) or {}

    if not isinstance(tool_result, dict):
        return ""

    return str(tool_result.get("action") or "").strip()


def get_booking_stage(state: AgentState) -> str:
    variables = state.get("variables", {}) or {}
    agent_config = state.get("agent_config", {}) or {}
    booking_config = get_subagent_config(agent_config, "booking")
    stage_path = str(booking_config.get("stage_path") or "booking.stage")
    return str(deep_get(variables, stage_path, "") or "").strip()


def safe_example_for_answer_draft(state: AgentState, answer_draft: str = "") -> str:
    agent_config = state.get("agent_config", {}) or {}
    answer_draft = str(answer_draft or get_tool_answer_draft(state) or "").strip()

    if not answer_draft:
        return ""

    policies = agent_config.get("template_response_policy", {}) or {}

    if not isinstance(policies, dict):
        return ""

    policy = policies.get(answer_draft, {})

    if not isinstance(policy, dict):
        return ""

    safe_examples = policy.get("safe_examples", [])

    if not isinstance(safe_examples, list):
        return ""

    for example in safe_examples:
        text = str(example or "").strip()
        if text:
            return text

    return ""


def answer_contains_any_configured_terms(answer: str, terms: Any) -> bool:
    if not isinstance(terms, list):
        return False

    text = str(answer or "")

    for term in terms:
        term_text = str(term or "").strip()
        if term_text and term_text in text:
            return True

    return False


def missing_or_pending_answer_draft_labels(agent_config: Dict[str, Any]) -> List[str]:
    configured = as_string_list(agent_config.get("missing_or_pending_answer_draft_labels", []))
    if configured:
        return configured

    policy_config = agent_config.get("template_response_policy_detection", {})
    if not isinstance(policy_config, dict):
        policy_config = {}

    key_prefixes = as_string_list(policy_config.get("missing_key_prefixes", []))
    value_prefixes = as_string_list(policy_config.get("missing_value_prefixes", []))

    labels = configured_booking_template_labels(
        agent_config,
        key_prefixes=key_prefixes,
        value_prefixes=value_prefixes,
    )

    policies = agent_config.get("template_response_policy", {}) or {}
    if isinstance(policies, dict):
        for key, policy in policies.items():
            if not isinstance(policy, dict):
                continue

            state = str(policy.get("state") or "").lower()
            is_missing = bool(policy.get("missing_fields_policy") is True)

            if state and any(token in state for token in as_string_list(policy_config.get("missing_state_tokens", []))):
                is_missing = True

            if is_missing:
                key_text = str(key or "").strip()
                if key_text and key_text not in labels:
                    labels.append(key_text)

    return labels

def should_use_safe_template_answer(answer: str, state: AgentState) -> bool:
    agent_config = state.get("agent_config", {}) or {}
    safety_config = agent_config.get("answer_safety", {}) or {}
    answer_draft = get_tool_answer_draft(state)
    action = get_tool_action(state)
    booking_stage = get_booking_stage(state)
    confirmed = create_booking_confirmed(state)

    if not answer_draft:
        return False

    if confirmed:
        return False

    missing_labels = missing_or_pending_answer_draft_labels(agent_config)

    if answer_draft in missing_labels:
        return True

    booking_config_inner = get_subagent_config(agent_config, "booking")
    safe_template_active_stages = booking_config_inner.get("extraction_active_stages", [])
    if not isinstance(safe_template_active_stages, list):
        safe_template_active_stages = []
    safe_template_stage_set = {
        str(stage_value or "").strip()
        for stage_value in safe_template_active_stages
        if str(stage_value or "").strip()
    }

    if action == "ask_user" and (not safe_template_stage_set or booking_stage in safe_template_stage_set):
        banned_groups = []
        banned_groups.extend(safety_config.get("banned_pending_booking_terms", []) or [])
        banned_groups.extend(safety_config.get("banned_unconfirmed_booking_terms", []) or [])
        banned_groups.extend(safety_config.get("banned_slot_preselection_phrases", []) or [])

        if answer_contains_any_configured_terms(answer, banned_groups):
            return True

    if action == "ask_user" and safe_example_for_answer_draft(state, answer_draft):
        banned_groups = []
        banned_groups.extend(safety_config.get("banned_pending_booking_terms", []) or [])
        banned_groups.extend(safety_config.get("banned_unconfirmed_booking_terms", []) or [])

        if answer_contains_any_configured_terms(answer, banned_groups):
            return True

    return False


def render_configured_visit_id_append(
    text: str,
    visit_id: str,
    state: AgentState
) -> str:
    agent_config = state.get("agent_config", {}) or {}
    safety_config = agent_config.get("answer_safety", {}) or {}

    append_template = str(
        safety_config.get("visit_id_append_template")
        or safety_config.get("visit_id_template")
        or ""
    ).strip()

    if append_template:
        addition = append_template.replace("{visit_id}", visit_id).strip()
    else:
        addition = visit_id

    if not addition or addition in text:
        return text

    return f"{text}\n{addition}".strip()

def enforce_answer_safety(answer: str, state: AgentState) -> str:
    text = str(answer or "").strip()
    agent_config = state.get("agent_config", {}) or {}
    safety_config = agent_config.get("answer_safety", {}) or {}

    if should_use_safe_template_answer(text, state):
        safe_answer = safe_example_for_answer_draft(state)
        if safe_answer:
            text = safe_answer

    banned_terms = safety_config.get("banned_terms", [])

    if not isinstance(banned_terms, list):
        banned_terms = []

    for term in banned_terms:
        term_text = str(term or "").strip()
        if term_text:
            text = text.replace(term_text, "").strip()

    if not create_booking_confirmed(state):
        answer_draft = get_tool_answer_draft(state)
        safe_answer = safe_example_for_answer_draft(state, answer_draft)

        blocked_terms: List[str] = []
        for key in [
            "banned_pending_booking_terms",
            "banned_unconfirmed_booking_terms",
            "banned_slot_preselection_phrases",
        ]:
            values = safety_config.get(key, [])
            if isinstance(values, list):
                blocked_terms.extend([str(item or "") for item in values])

        if safe_answer and answer_contains_any_configured_terms(text, blocked_terms):
            text = safe_answer

    policy_answer = render_policy_template_answer(state)
    if should_use_policy_template_answer(state) and policy_answer:
        text = policy_answer

    text = re.sub(r"\s{2,}", " ", text).strip()

    append_visit_id = safety_config.get("append_visit_id_on_confirmed_booking", True)
    visit_id = extract_visit_id_from_state(state)

    # 6.53: enforce_answer_safety is the final safety pass and can run after
    # pre_response_guardrail_node. Reuse the same config-driven skip helper here
    # so short completed-flow closings and configured answer-draft labels do not
    # receive a confirmed record id later in the pipeline.
    skip_confirmed_record_id = should_skip_confirmed_record_id_append(
        state,
        text,
        agent_config,
        safety_config,
    )

    if (
        append_visit_id
        and not skip_confirmed_record_id
        and create_booking_confirmed(state)
        and visit_id
        and visit_id not in text
    ):
        text = render_configured_visit_id_append(text, visit_id, state)

    return text

def quality_guard_node(state: AgentState):
    if not should_run_quality_guard(state):
        final_answer = enforce_answer_safety(state.get("final_answer", ""), state)

        return {
            "final_answer": final_answer,
            "quality": {
                "pass_check": True,
                "skipped": True,
                "node": "quality_guard_skipped",
            }
        }

    answer = state.get("final_answer", "")
    latest_user = last_user_message(state)
    agent_config = state.get("agent_config", {}) or {}
    manifest = state.get("manifest", {}) or {}
    known_variable_labels = ", ".join(
        as_string_list(agent_config.get("quality_guard_known_variable_labels", []))
    )
    if not known_variable_labels:
        known_variable_labels = "known user and workflow variables"

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are the quality guardian of a configurable multi-tenant assistant. "
            "Check the assistant's answer. Run these checks in order:\n\n"
            "1. CORRECTNESS — Any hallucinated facts, invented slots, IDs, branch names, "
            "prices, or confirmations not in the provided context? Rewrite if yes.\n\n"
            "2. SAFETY — Any banned terms, or claiming a confirmed action when "
            "tool_result does not confirm it? Rewrite if yes.\n\n"
            "3. LANGUAGE — Wrong language, or exposes internal jargon "
            "(RAG, variables, routing, agent, tool, JSON)? Rewrite if yes.\n\n"
            "4. ENERGY — Sounds like a scripted bot, repeats the user's question, "
            "starts with 'Sure/Certainly/Of course', uses overly formal language, "
            "or lists facts mechanically instead of speaking naturally? Rewrite if yes.\n\n"
            "5. VARIABLE AWARENESS — Asks for something already in the known variables "
            f"({known_variable_labels})? "
            "Rewrite to use the known value instead of asking again.\n\n"
            "6. ONE QUESTION — Does the response ask more than one question, even "
            "phrased indirectly? Rewrite to ask only the single most important thing. "
            "The second question can wait for the next turn.\n\n"
            "If all six pass: pass_check=true, revised_answer=''. "
            "If any fail: pass_check=false, revised_answer=the corrected answer. "
            "Keep revised_answer in the same language as the original. "
            "Preserve all grounded facts when rewriting. "
            "If tool_result.action is ask_user, never rewrite into completed-action wording. "
            "If an ID is required by the confirmed action policy, do not remove it.",
        ),
        (
            "user",
            "Latest message:\n{latest_user}\n\n"
            "Manifest:\n{manifest}\n\n"
            "Context:\n{context}\n\n"
            "Analysis:\n{analysis}\n\n"
            "Knowledge:\n{knowledge}\n\n"
            "Tool result:\n{tool_result}\n\n"
            "Variables:\n{variables}\n\n"
            "Answer:\n{answer}",
        ),
    ])

    try:
        decision = (prompt | quality_llm).invoke({
            "latest_user": latest_user,
            "manifest": safe_json(compact_manifest(manifest), max_chars=2600),
            "context": safe_json({
                **compact_agent_context(
                    agent_config,
                    state.get("selected_subagent", {}) or {},
                ),
                "answer_safety": agent_config.get("answer_safety", {}) or {},
                "template_response_policy": compact_template_response_policy(
                    agent_config,
                    state.get("tool_result", {}) or {},
                ),
                "multi_tool_results": state.get("multi_tool_results", []) or [],
                "multi_knowledge": state.get("multi_knowledge", []) or [],
                "response_synthesis": state.get("response_synthesis", {}) or {},
                "proactive_guidance": build_proactive_guidance(
                    agent_config,
                    manifest,
                    state.get("tool_result", {}) or {},
                ),
                "tone_guidance": get_response_energy_instruction(agent_config, manifest),
            }, max_chars=4200),
            "analysis": safe_json(compact_analysis(state.get("subagent_analysis", {}) or {}), max_chars=1800),
            "knowledge": clip_text(state.get("knowledge", ""), 1200),
            "tool_result": safe_json(state.get("tool_result", {}) or {}, max_chars=2800),
            "variables": safe_json(compact_variables(state.get("variables", {}) or {}, state.get("schema", {}) or {}), max_chars=2400),
            "answer": answer,
        })

        data = decision.model_dump()

        should_rewrite_for_energy = (
            getattr(decision, "energy_ok", True) is False
            and str(getattr(decision, "revised_answer", "") or "").strip()
        )

        if (not decision.pass_check or should_rewrite_for_energy) and decision.revised_answer.strip():
            revised = enforce_answer_safety(decision.revised_answer.strip(), state)
            data["node"] = "quality_guard_revised"
            return {
                "messages": [AIMessage(content=revised)],
                "final_answer": revised,
                "quality": data,
            }

        final_answer = enforce_answer_safety(answer, state)
        data["node"] = "quality_guard_passed"
        return {
            "final_answer": final_answer,
            "quality": data,
        }

    except Exception as exc:
        final_answer = enforce_answer_safety(answer, state)
        return {
            "final_answer": final_answer,
            "quality": {
                "pass_check": True,
                "guard_error": str(exc),
                "node": "quality_guard_error",
            }
        }

def compact_graph_messages_for_memory(messages: Sequence[BaseMessage], limit: int = 14) -> List[Dict[str, str]]:
    """
    Convert current graph messages into compact role/content dictionaries for memory maintenance.
    This avoids depending only on DB reads because the current turn may not be saved until after app_graph.invoke().
    """
    output: List[Dict[str, str]] = []

    for item in list(messages or [])[-limit:]:
        role = ""
        content = ""

        if isinstance(item, HumanMessage):
            role = "user"
            content = str(item.content or "")
        elif isinstance(item, AIMessage):
            role = "assistant"
            content = str(item.content or "")
        elif isinstance(item, SystemMessage):
            continue
        elif hasattr(item, "content"):
            role = str(getattr(item, "type", "") or "message")
            content = str(getattr(item, "content", "") or "")

        content = content.strip()

        if role and content:
            output.append({
                "role": role,
                "content": clip_text(content, 1600)
            })

    return output


def memory_writer_node(state: AgentState):
    """
    Best-effort post-turn memory maintenance node.

    Expert architecture goal:
    - run after quality_guard_node
    - never block or change the user-facing answer
    - update rolling summary and durable memories conservatively
    - return only compact operational metadata for debug traces
    - never expose or store chain-of-thought
    """
    try:
        from app.memory import run_memory_maintenance_best_effort

        result = run_memory_maintenance_best_effort(
            assistant_id=state.get("assistant_id", ""),
            user_id=state.get("user_id", ""),
            conversation_id=state.get("conversation_id", ""),
            variables=state.get("variables", {}) or {},
            agent_config=state.get("agent_config", {}) or {},
            recent_messages=compact_graph_messages_for_memory(state.get("messages", []), limit=14),
            existing_summary=state.get("summary", "") or "",
        )

        if not isinstance(result, dict):
            result = {
                "ok": False,
                "skipped": True,
                "reason": "invalid_memory_writer_result",
            }

        response: Dict[str, Any] = {
            "memory_writer": result
        }

        derived_last_offered_options = derive_last_offered_options_from_state(state)
        if derived_last_offered_options:
            response["last_offered_options"] = derived_last_offered_options

        manifest = state.get("manifest", {}) or {}
        patched_manifest = apply_funnel_stage_policy(manifest, state.get("agent_config", {}) or {})
        if patched_manifest != manifest:
            response["manifest"] = patched_manifest
            if str(patched_manifest.get("funnel_stage") or "").strip():
                response["funnel_stage"] = str(patched_manifest.get("funnel_stage") or "").strip()

        mirrored_variables = mirror_runtime_metadata_into_variables(
            state.get("variables", {}) or {},
            manifest=patched_manifest if isinstance(patched_manifest, dict) else manifest,
            funnel_stage=str((patched_manifest or {}).get("funnel_stage") or state.get("funnel_stage", "")),
            last_offered_options=derived_last_offered_options,
            agent_config=state.get("agent_config", {}) or {},
        )
        if mirrored_variables != (state.get("variables", {}) or {}):
            response["variables"] = mirrored_variables

        return response

    except Exception as exc:
        return {
            "memory_writer": {
                "ok": False,
                "skipped": True,
                "reason": "memory_writer_node_error",
                "error": f"{type(exc).__name__}: {exc}",
            }
        }


workflow = StateGraph(AgentState)

workflow.add_node("manifest_node", unified_manifest_node)
workflow.add_node("retrieve_memory_node", retrieve_memory_node)
workflow.add_node("retrieve_knowledge_node", retrieve_knowledge_node)
workflow.add_node("retrieve_memory_and_knowledge_node", retrieve_memory_and_knowledge_node)
workflow.add_node("proactive_surface_check_node", proactive_surface_check_node)
workflow.add_node("tool_execution_node", tool_execution_node)
workflow.add_node("semantic_extraction_node", semantic_extraction_node)
workflow.add_node("smart_inference_node", smart_inference_node)
workflow.add_node("subagent_reasoning_node", subagent_reasoning_node)
workflow.add_node("simple_response_node", simple_response_node)
workflow.add_node("response_node", response_node)
workflow.add_node("quality_guard_node", quality_guard_node)
workflow.add_node("pre_response_guardrail_node", pre_response_guardrail_node)
workflow.add_node("memory_writer_node", memory_writer_node)

workflow.set_entry_point("manifest_node")

workflow.add_conditional_edges(
    "manifest_node",
    decide_after_manifest,
    {
        "simple_response": "simple_response_node",
        "retrieve_memory": "retrieve_memory_node",
        "retrieve_knowledge": "retrieve_knowledge_node",
        "retrieve_memory_and_knowledge": "retrieve_memory_and_knowledge_node",
        "tool_execution": "tool_execution_node",
        "semantic_extraction": "semantic_extraction_node",
        "subagent_reasoning": "subagent_reasoning_node",
        "response": "response_node",
    },
)

workflow.add_conditional_edges(
    "retrieve_memory_node",
    decide_after_memory,
    {
        "retrieve_knowledge": "retrieve_knowledge_node",
        "proactive_surface": "proactive_surface_check_node",
        "tool_execution": "tool_execution_node",
        "semantic_extraction": "semantic_extraction_node",
        "subagent_reasoning": "subagent_reasoning_node",
        "response": "response_node",
    },
)

workflow.add_conditional_edges(
    "retrieve_knowledge_node",
    decide_after_knowledge,
    {
        "proactive_surface": "proactive_surface_check_node",
        "tool_execution": "tool_execution_node",
        "semantic_extraction": "semantic_extraction_node",
        "subagent_reasoning": "subagent_reasoning_node",
        "response": "response_node",
    },
)

workflow.add_conditional_edges(
    "retrieve_memory_and_knowledge_node",
    decide_after_knowledge,
    {
        "proactive_surface": "proactive_surface_check_node",
        "tool_execution": "tool_execution_node",
        "semantic_extraction": "semantic_extraction_node",
        "subagent_reasoning": "subagent_reasoning_node",
        "response": "response_node",
    },
)

workflow.add_conditional_edges(
    "proactive_surface_check_node",
    decide_after_proactive_surface,
    {
        "tool_execution": "tool_execution_node",
        "semantic_extraction": "semantic_extraction_node",
        "subagent_reasoning": "subagent_reasoning_node",
        "response": "response_node",
    },
)

workflow.add_edge("semantic_extraction_node", "smart_inference_node")
workflow.add_edge("smart_inference_node", "tool_execution_node")

workflow.add_conditional_edges(
    "tool_execution_node",
    decide_after_tool,
    {
        "subagent_reasoning": "subagent_reasoning_node",
        "response": "response_node",
    },
)

workflow.add_edge("subagent_reasoning_node", "response_node")
workflow.add_edge("response_node", "pre_response_guardrail_node")
workflow.add_edge("simple_response_node", "pre_response_guardrail_node")
workflow.add_edge("pre_response_guardrail_node", "quality_guard_node")
workflow.add_edge("quality_guard_node", "memory_writer_node")
workflow.add_edge("memory_writer_node", END)

app_graph = workflow.compile()
