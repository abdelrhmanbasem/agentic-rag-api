from typing import TypedDict, Annotated, Sequence, Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import operator
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
    OPENAI_API_KEY,
    MAX_OUTPUT_TOKENS,
    MAX_QUALITY_TOKENS,
    QUALITY_GUARD_ENABLED,
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


manifest_llm = llm(MODEL_PLANNER, temperature=0, max_tokens=1600).bind(
    response_format={"type": "json_object"}
)
subagent_llm = llm(MODEL_SUBAGENT, temperature=0.15).with_structured_output(SubagentAnalysis)
subagent_llm_raw = llm(MODEL_SUBAGENT, temperature=0.15, max_tokens=700)
response_llm = llm(MODEL_RESPONSE, temperature=0.45, max_tokens=MAX_OUTPUT_TOKENS)
quality_llm = llm(MODEL_QUALITY, temperature=0, max_tokens=MAX_QUALITY_TOKENS).with_structured_output(QualityDecision)


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

    if manifest.get("needs_quality_guard"):
        return True
    if manifest.get("needs_style_repair"):
        return True
    if tool_result:
        return True
    if state.get("multi_tool_results") or state.get("multi_knowledge"):
        return True
    if manifest.get("needs_tool") or manifest_has_parallel_tool_requests(manifest):
        return True
    if manifest.get("risk_level") in ["high", "medium"]:
        return True
    if "NO_CONFIDENT_KNOWLEDGE_FOUND" not in knowledge and knowledge.strip():
        return True
    if len(answer) > 700:
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


def has_known_branch(variables: Dict[str, Any]) -> bool:
    return bool(
        deep_get(variables, "selected_branch", "")
        or deep_get(variables, "location_branch", "")
        or deep_get(variables, "nearest_branch", "")
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

    known_branch = has_known_branch(variables)

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

    brief = patched.get("response_brief")
    if not isinstance(brief, dict):
        brief = {}

    brief["tone"] = "natural professional Egyptian Arabic"
    brief["language"] = "Egyptian Arabic"
    brief["reply_length"] = "short"
    brief["next_move"] = next_move
    brief["must_do"] = append_unique(brief.get("must_do", []), must_do)
    brief["must_not_do"] = append_unique(brief.get("must_not_do", []), [
        "do not continue troubleshooting",
        "do not ask what type of sound again",
        "do not invent branch or slot",
        "do not use formal MSA wording"
    ])

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



def unified_manifest_node(state: AgentState):
    message = last_user_message(state)
    agent_config = state.get("agent_config", {}) or {}
    variables = state.get("variables", {}) or {}
    schema = state.get("schema", {}) or {}
    tool_result = state.get("tool_result", {}) or {}

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are the Unified Manifest brain of a configurable multi-tenant agentic assistant engine. "
            "Return ONLY valid JSON. No markdown. No prose. "
            "You decide the next move; you never write the final user-facing reply. "
            "All domain behavior must come from the assistant manifest card, current variables, conversation summary, tool results, schema, and latest user message. "
            "Do not use hidden business rules or domain assumptions outside the manifest card. "
            "Detect the user's full intent, including multiple independent or sequential intents in one message. "
            "Handle hesitation and change-of-mind naturally: when the latest message corrects or replaces an earlier preference in the same turn or recent context, the latest explicit preference wins. Do not run superseded alternatives in parallel unless the user clearly asks to compare them. "
            "Use detected_intents for short intent labels, and multi_intents for structured intent objects. Each multi_intents item may include: intent_id, intent_type, user_goal, selected_subagent_id, needs_tool, requested_tool_name, tool_request_payload, needs_knowledge, knowledge_query, depends_on, priority, missing_inputs, and response_role. "
            "For sequential dependencies, use selected_subagent_id plus chained_subagent_id. Example: nearest branch then availability should run location first and booking second when booking needs location output. "
            "For independent direct tool actions that can safely run from current variables, use parallel_tool_requests. Each request should include: request_id, intent_id, tool_name, operation, arguments, purpose, and can_run_in_parallel=true. "
            "Only use parallel_tool_requests for independent non-conflicting actions. Never parallelize two alternatives where the user corrected their mind, such as changing date, branch, slot, quantity, product, or service preference. "
            "Do not put deterministic subagent-owned booking/location/troubleshooting flows into parallel_tool_requests unless the manifest card says direct tool calls are safe. Prefer selected_subagent_id/chained_subagent_id for stateful workflows. "
            "For knowledge needs, set needs_knowledge=true and provide knowledge_queries when multiple distinct retrieval questions are useful. "
            "If a tool needs missing inputs, set needs_tool=false and list missing_tool_inputs; the final response should ask naturally for only the missing input. "
            "If a tool can be called safely with current variables/message, set needs_tool=true and provide requested_tool_name plus tool_request_payload or parallel_tool_requests. "
            "Never claim tool results, availability, prices, IDs, branches, or external facts unless already present in tool_result, variables, conversation context, or retrieved knowledge. "
            "For booking/availability/nearest-branch/branch-list actions, prefer needs_tool=true when inputs are available. "
            "For symptom/problem reports, do not force a booking immediately; response_brief should guide a diagnostic response first unless user asks to book. "
            "The manifest may extract soft user info only. Source-of-truth operational state must come from tool/subagent execution, not manifest extracted_updates. "
            "response_synthesis should describe how the final response should combine multiple results without inventing facts. "
            "The JSON object must use exactly these top-level keys: "
            "user_intent, selected_subagent_id, chained_subagent_id, chained_subagent_reason, detected_intents, multi_intents, parallel_tool_requests, knowledge_queries, response_synthesis, multi_intent_execution_mode, conversation_stage, workflow_stage, customer_emotion, user_expectation, risk_level, confidence, "
            "simple_response_mode, simple_response_reason, needs_knowledge, needs_memory, needs_tool, requested_tool_name, tool_request_payload, missing_tool_inputs, "
            "needs_subagent_reasoning, needs_quality_guard, needs_style_repair, needs_full_manifest, extracted_updates, extracted_deletions, response_style, reply_length, "
            "should_ask_question, question_goal, should_offer_next_action, response_brief, response_strategy, reasoning_summary. "
            "response_brief must be an object with keys: tone, language, reply_length, must_do, must_not_do, next_move.",
        ),
        (
            "user",
            "Assistant manifest card:\n{manifest_card}\n\n"
            "Conversation summary:\n{summary}\n\n"
            "Current variables:\n{variables}\n\n"
            "Tool result if any:\n{tool_result}\n\n"
            "Latest user message:\n{message}\n\n"
            "Return the manifest JSON. selected_subagent_id must be one configured subagent id when possible. "
            "Use multi_intents, knowledge_queries, and parallel_tool_requests only when they add real value and are safe.",
        ),
    ])

    def invoke_manifest(profile: str) -> Dict[str, Any]:
        manifest_card_max = 5200 if profile == "short" else 8400

        decision = (prompt | manifest_llm).invoke({
            "manifest_card": safe_json(
                unified_manifest_card(agent_config, schema, profile=profile),
                max_chars=manifest_card_max,
            ),
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
        parsed["selected_subagent_id"] = unify_subagent_id(parsed.get("selected_subagent_id", ""))
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

        should_retry_full = (
            bool(manifest.get("needs_full_manifest"))
            or manifest_confidence(manifest) < 0.65
            or selected_missing
            or chained_missing
            or (manifest.get("needs_tool") and not manifest.get("requested_tool_name") and not manifest.get("selected_subagent_id"))
            or (manifest.get("chained_subagent_id") and not manifest.get("selected_subagent_id"))
            or _parallel_declared_but_broken
            or (manifest_has_multi_intents(manifest) and not manifest.get("selected_subagent_id") and not _raw_parallel)
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
        manifest["selected_subagent_id"] = unify_subagent_id(
            manifest.get("selected_subagent_id", "")
        )

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

    updated_variables = apply_subagent_variable_patch(
        variables,
        safe_manifest_updates,
        safe_manifest_deletions,
        assistant_config=agent_config,
    )

    return {
        "manifest": manifest,
        "planner": build_planner_compat(manifest),
        "selected_subagent": selected_subagent,
        "variables": updated_variables,
        "multi_intents": manifest.get("multi_intents", []),
        "parallel_tool_requests": manifest.get("parallel_tool_requests", []),
        "knowledge_queries": manifest.get("knowledge_queries", []),
        "response_synthesis": manifest.get("response_synthesis", {}),
    }


def decide_after_manifest(state: AgentState) -> str:
    manifest = state.get("manifest", {}) or {}

    if active_deterministic_flow_subagent_id_from_state(state):
        return "tool_execution"

    if manifest_has_parallel_tool_requests(manifest):
        return "tool_execution"

    if should_use_simple_response(state):
        return "simple_response"

    if manifest.get("needs_memory"):
        return "retrieve_memory"

    if manifest.get("needs_knowledge") or manifest_has_knowledge_queries(manifest):
        return "retrieve_knowledge"

    if manifest.get("needs_tool") or manifest_has_parallel_tool_requests(manifest):
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
        return {"memories": ""}

    text = "\n".join([
        f"- {m.get('text', '')} (type={m.get('type', 'other')}, confidence={m.get('confidence', '')})"
        for m in memories
        if m.get("text")
    ])

    return {"memories": text}


def decide_after_memory(state: AgentState) -> str:
    manifest = state.get("manifest", {}) or {}

    if active_deterministic_flow_subagent_id_from_state(state):
        return "tool_execution"

    if manifest.get("needs_knowledge") or manifest_has_knowledge_queries(manifest):
        return "retrieve_knowledge"

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
    }


def decide_after_knowledge(state: AgentState) -> str:
    manifest = state.get("manifest", {}) or {}

    if active_deterministic_flow_subagent_id_from_state(state):
        return "tool_execution"

    if manifest.get("needs_tool") or manifest_has_parallel_tool_requests(manifest):
        return "tool_execution"

    if manifest.get("needs_subagent_reasoning"):
        return "subagent_reasoning"

    return "response"


def subagent_history_from_messages(messages: Sequence[BaseMessage]) -> List[Dict[str, str]]:
    output: List[Dict[str, str]] = []

    for item in messages[-12:]:
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


def get_known_branch_from_variables(variables: Dict[str, Any]) -> str:
    return str(first_present(
        deep_get(variables, "selected_branch", ""),
        deep_get(variables, "location_branch", ""),
        deep_get(variables, "nearest_branch", ""),
    ) or "").strip()


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

    if op in {"list_available_slots", "check_availability", "create_booking"}:
        branch = first_present(
            args.get("branch"),
            get_known_branch_from_variables(variables),
        )

        if branch:
            args["branch"] = branch

        date_value = first_present(
            args.get("date"),
            args.get("appointment_date"),
            deep_get(variables, "appointment_date", ""),
            deep_get(variables, "date", ""),
        )

        if date_value:
            args["date"] = date_value

        date_text = first_present(
            args.get("date_text"),
            deep_get(variables, "date_text", ""),
        )

        if date_text:
            args["date_text"] = date_text

    if op in {"check_availability", "create_booking"}:
        time_value = first_present(
            args.get("time"),
            deep_get(variables, "appointment_time", ""),
            deep_get(variables, "booking.pending.time", ""),
        )

        if time_value:
            args["time"] = time_value

    if op == "create_booking":
        pending = deep_get(variables, "booking.pending", {})
        if isinstance(pending, dict):
            for key in ["branch", "date", "date_text", "time", "section"]:
                if not is_present(args.get(key)) and is_present(pending.get(key)):
                    args[key] = pending.get(key)

        profile = deep_get(variables, "customer_profile", {})
        if isinstance(profile, dict):
            if not is_present(args.get("full_name")) and is_present(profile.get("full_name")):
                args["full_name"] = profile.get("full_name")
            if not is_present(args.get("phone")) and is_present(profile.get("phone")):
                args["phone"] = profile.get("phone")
            if not is_present(args.get("plate_number")) and is_present(profile.get("plate_number")):
                args["plate_number"] = profile.get("plate_number")

        if not is_present(args.get("customer_confirmed_booking")):
            confirmed = deep_get(variables, "customer_confirmed_booking", None)
            if confirmed is not None:
                args["customer_confirmed_booking"] = confirmed

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


def tool_execution_node(state: AgentState):
    manifest = state.get("manifest", {}) or {}
    agent_config = state.get("agent_config", {}) or {}
    variables = state.get("variables", {}) or {}
    schema = state.get("schema", {}) or {}
    message = last_user_message(state)

    selected_id = unify_subagent_id(manifest.get("selected_subagent_id", ""))
    chained_id = unify_subagent_id(manifest.get("chained_subagent_id", ""))

    tool_runner = ToolRunner(agent_config)
    all_observations: List[Dict[str, Any]] = []
    parallel_requests = normalize_parallel_tool_requests(manifest)

    if should_execute_parallel_tool_requests(
        manifest=manifest,
        selected_id=selected_id,
        agent_config=agent_config,
    ):
        return run_parallel_direct_tool_requests(
            requests=parallel_requests,
            variables=variables,
            agent_config=agent_config,
            tool_runner=tool_runner,
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

        return {
            "variables": variables,
            "tool_result": tool_result,
            "multi_tool_results": executed_results,
        }

    if selected_id and all_observations:
        return {
            "tool_result": {
                "ok": False,
                "subagent": selected_id,
                "error": "Subagent execution failed or was not handled.",
                "action": "reply",
                "observations": all_observations,
            }
        }

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

            return {
                "variables": updated_variables,
                "tool_result": validation.get("tool_result", {}),
            }

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

        return {
            "variables": updated_variables,
            "tool_result": enriched_result,
            "multi_tool_results": [enriched_result],
        }

    return {
        "tool_result": {
            "ok": False,
            "error": "Tool requested but no executable subagent/tool operation was available.",
            "selected_subagent_id": selected_id,
            "chained_subagent_id": chained_id,
            "requested_tool_name": tool_name,
            "tool_request_payload": payload,
        }
    }


def decide_after_tool(state: AgentState) -> str:
    manifest = state.get("manifest", {}) or {}

    if manifest.get("needs_subagent_reasoning"):
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
        "variables": compact_variables(state.get("variables", {}) or {}, schema, max_items=18),
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

{state.get('language_instruction', '')}
"""

    messages = [SystemMessage(content=system_instruction)] + list(state["messages"][-4:])
    dynamic_response_llm = llm(
        MODEL_RESPONSE,
        temperature=get_response_temperature(manifest, {}),
        max_tokens=MAX_OUTPUT_TOKENS,
    )
    response = dynamic_response_llm.invoke(messages)
    answer = response.content if hasattr(response, "content") else str(response)
    answer = enforce_answer_safety(answer, state)

    return {
        "messages": [AIMessage(content=answer)],
        "final_answer": answer,
        "quality": {
            "pass_check": True,
            "skipped": False,
            "simple_response_mode": True,
            "node": "simple_response_pre_quality_guard",
        },
    }

def response_node(state: AgentState):
    deterministic_answer = maybe_policy_template_answer(state)

    if deterministic_answer:
        deterministic_answer = enforce_answer_safety(deterministic_answer, state)
        return {
            "messages": [AIMessage(content=deterministic_answer)],
            "final_answer": deterministic_answer,
            "quality": {
                "node": "response_policy_template",
                "pre_quality_guard": True,
                "deterministic_template_used": True,
            },
        }

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

    response_context = {
        "assistant_context": compact_agent_context(agent_config, subagent),
        "manifest": effective_manifest,
        "private_analysis": compact_analysis(analysis),
        "summary": clip_text(state.get("summary", ""), 700),
        "variables": compact_variables(variables, schema),
        "memory": compact_memories_for_final(memories),
        "knowledge": compact_knowledge_for_final(knowledge),
        "multi_knowledge": state.get("multi_knowledge", []) or [],
        "knowledge_queries": state.get("knowledge_queries", []) or [],
        "tool_result": tool_result,
        "multi_tool_results": (
            state.get("multi_tool_results", [])
            or (tool_result.get("multi_tool_results", []) if isinstance(tool_result, dict) else [])
        ),
        "response_synthesis": state.get("response_synthesis", {}) or manifest.get("response_synthesis", {}),
        "multi_intents": state.get("multi_intents", []) or manifest.get("multi_intents", []),
        "answer_safety": agent_config.get("answer_safety", {}) or {},
        "template_response_policy": compact_template_response_policy(agent_config, tool_result),
        "proactive_guidance": build_proactive_guidance(agent_config, manifest, tool_result),
        "tone_guidance": get_response_energy_instruction(agent_config, manifest),
    }

    response_rules = build_layered_response_rules(agent_config)

    system_instruction = f"""
{clip_text_head_tail(state.get('system_prompt', ''), 1200)}

You are the voice of this assistant. You are generating the exact reply the user will read.

If the context contains multi_intents, multi_tool_results, or multi_knowledge, synthesize them into one coherent answer. Keep each intent/result grounded. Do not merge facts from different tool results unless the context explicitly supports it. If one intent is completed and another is missing input, state the completed result briefly and ask only for the missing input. If the user hesitated or changed their mind, reflect only the latest selected preference and do not mention discarded alternatives unless helpful for clarity.

Context you have available:
{safe_json(response_context, max_chars=9200)}

How to reply — follow in this priority order:
{safe_json(response_rules)}

Customer emotion detected:
{manifest.get('customer_emotion', 'neutral')}

Tone guidance for this specific message:
{get_response_energy_instruction(agent_config, manifest)}

Expected response style:
{manifest.get('response_style', '')}

Reply length:
{manifest.get('reply_length', '')}

{state.get('language_instruction', '')}
"""

    messages = [SystemMessage(content=system_instruction)] + list(state["messages"][-8:])
    dynamic_response_llm = llm(
        MODEL_RESPONSE,
        temperature=get_response_temperature(manifest, tool_result),
        max_tokens=MAX_OUTPUT_TOKENS,
    )
    response = dynamic_response_llm.invoke(messages)
    answer = response.content if hasattr(response, "content") else str(response)
    answer = enforce_answer_safety(answer, state)

    return {
        "messages": [AIMessage(content=answer)],
        "final_answer": answer,
        "quality": {
            "node": "response_pre_quality_guard",
            "pre_quality_guard": True,
        },
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
    return str(deep_get(variables, "booking.stage", "") or "").strip()


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

    if action == "ask_user" and booking_stage in {"awaiting_customer_details", "awaiting_confirmation"}:
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

    if append_visit_id and create_booking_confirmed(state) and visit_id and visit_id not in text:
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

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Quality check the answer for a configurable assistant. "
            "Check in this order: correctness, safety, language, and energy. "
            "Correctness means no unsupported facts, no invented operational results, and no fake confirmations. "
            "Safety means obey configured answer_safety and template_response_policy. "
            "Language means the reply follows the user's language and assistant language policy. "
            "Energy means the reply sounds like the configured persona: natural, confident, helpful, and not robotic. "
            "If correctness or safety fails, rewrite it. "
            "If only energy fails, rewrite it to sound more natural while preserving every grounded fact. "
            "If tool_result.action is ask_user or tool_result.answer_draft is a missing-field label, never rewrite it into completed-action wording. "
            "If an ID is required by confirmed action policy, do not remove it.",
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

        return {
            "memory_writer": result
        }

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
workflow.add_node("tool_execution_node", tool_execution_node)
workflow.add_node("subagent_reasoning_node", subagent_reasoning_node)
workflow.add_node("simple_response_node", simple_response_node)
workflow.add_node("response_node", response_node)
workflow.add_node("quality_guard_node", quality_guard_node)
workflow.add_node("memory_writer_node", memory_writer_node)

workflow.set_entry_point("manifest_node")

workflow.add_conditional_edges(
    "manifest_node",
    decide_after_manifest,
    {
        "simple_response": "simple_response_node",
        "retrieve_memory": "retrieve_memory_node",
        "retrieve_knowledge": "retrieve_knowledge_node",
        "tool_execution": "tool_execution_node",
        "subagent_reasoning": "subagent_reasoning_node",
        "response": "response_node",
    },
)

workflow.add_conditional_edges(
    "retrieve_memory_node",
    decide_after_memory,
    {
        "retrieve_knowledge": "retrieve_knowledge_node",
        "tool_execution": "tool_execution_node",
        "subagent_reasoning": "subagent_reasoning_node",
        "response": "response_node",
    },
)

workflow.add_conditional_edges(
    "retrieve_knowledge_node",
    decide_after_knowledge,
    {
        "tool_execution": "tool_execution_node",
        "subagent_reasoning": "subagent_reasoning_node",
        "response": "response_node",
    },
)

workflow.add_conditional_edges(
    "tool_execution_node",
    decide_after_tool,
    {
        "subagent_reasoning": "subagent_reasoning_node",
        "response": "response_node",
    },
)

workflow.add_edge("subagent_reasoning_node", "response_node")
workflow.add_edge("response_node", "quality_guard_node")
workflow.add_edge("simple_response_node", "quality_guard_node")
workflow.add_edge("quality_guard_node", "memory_writer_node")
workflow.add_edge("memory_writer_node", END)

app_graph = workflow.compile()
