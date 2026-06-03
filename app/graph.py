from typing import TypedDict, Annotated, Sequence, Dict, Any, List, Optional
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
    QUALITY_GUARD_ENABLED,
)
from app.rag import search_knowledge, compress_knowledge, search_memories
from app.tool_runner import ToolRunner
from app.subagents.base import (
    SubagentContext,
    apply_variable_patch as apply_subagent_variable_patch,
    apply_tool_update_rules,
    get_subagent_variable_scope,
    matches_any,
    deep_get,
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

    manifest: Dict[str, Any]
    planner: Dict[str, Any]

    selected_subagent: Dict[str, Any]
    subagent_analysis: Dict[str, Any]

    knowledge: str
    knowledge_items: List[Dict[str, Any]]
    memories: str

    final_answer: str
    quality: Dict[str, Any]


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
    facts_to_use: List[str] = Field(default_factory=list)
    facts_missing: List[str] = Field(default_factory=list)
    next_best_step: str = ""
    response_constraints: List[str] = Field(default_factory=list)
    confidence: float = 0.7


class QualityDecision(BaseModel):
    pass_check: bool
    revised_answer: str = ""
    issues: List[str] = Field(default_factory=list)


manifest_llm = llm(MODEL_PLANNER, temperature=0, max_tokens=900).bind(
    response_format={"type": "json_object"}
)
subagent_llm = llm(MODEL_SUBAGENT, temperature=0.15).with_structured_output(SubagentAnalysis)
response_llm = llm(MODEL_RESPONSE, temperature=0.45, max_tokens=MAX_OUTPUT_TOKENS)
quality_llm = llm(MODEL_QUALITY, temperature=0).with_structured_output(QualityDecision)


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


SOURCE_OF_TRUTH_EXACT_VARIABLES = {
    "visit_id",
    "booking_status",
    "available_slots",
    "available_slots_text",
    "available_branches",
    "available_branches_text",
    "slots_found",
    "appointment_date",
    "appointment_time",
    "selected_branch",
    "nearest_branch",
    "location_branch",
    "customer_confirmed_booking",
}

SOURCE_OF_TRUTH_PREFIXES = (
    "booking",
    "booking.",
    "tool_result",
    "exact_slot",
    "nearest_slots",
    "unavailable_reason",
)


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


def filter_manifest_updates(updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    The manifest LLM may extract soft user info, but it must not directly write
    source-of-truth operational state. Tool/subagent execution owns those fields.
    """
    if not isinstance(updates, dict):
        return {}

    allowed: Dict[str, Any] = {}

    for key, value in updates.items():
        key_text = str(key or "").strip()

        if not key_text:
            continue

        if key_text in SOURCE_OF_TRUTH_EXACT_VARIABLES:
            continue

        if any(
            key_text == prefix or key_text.startswith(prefix + ".")
            for prefix in SOURCE_OF_TRUTH_PREFIXES
        ):
            continue

        allowed[key_text] = value

    return allowed


def filter_manifest_deletions(deletions: List[str]) -> List[str]:
    """
    The manifest LLM may request deletion of soft state, but cannot delete
    source-of-truth operational state.
    """
    if not isinstance(deletions, list):
        return []

    allowed: List[str] = []

    for item in deletions:
        path = str(item or "").strip()

        if not path:
            continue

        if path in SOURCE_OF_TRUTH_EXACT_VARIABLES:
            continue

        if any(
            path == prefix or path.startswith(prefix + ".")
            for prefix in SOURCE_OF_TRUTH_PREFIXES
        ):
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
        "facts_to_use": analysis.get("facts_to_use", [])[:8],
        "facts_missing": analysis.get("facts_missing", [])[:8],
        "next_best_step": clip_text(analysis.get("next_best_step", ""), 360),
        "response_constraints": analysis.get("response_constraints", [])[:10],
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


def should_use_simple_response(state: AgentState) -> bool:
    manifest = state.get("manifest", {}) or {}

    if manifest.get("needs_tool"):
        return False
    if manifest.get("needs_knowledge"):
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
    if manifest.get("needs_tool"):
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


def has_configured_visit_intent(
    message: str,
    agent_config: Dict[str, Any],
    variables: Dict[str, Any]
) -> bool:
    normalization = agent_config.get("normalization", {}) or {}
    subagents = agent_config.get("subagents", {}) or {}

    booking_config = subagents.get("booking", {}) or {}
    troubleshooting_config = subagents.get("troubleshooting", {}) or {}

    visit_phrases: List[str] = []
    visit_phrases.extend(booking_config.get("trigger_phrases", []) or [])
    visit_phrases.extend(troubleshooting_config.get("direct_booking_phrases", []) or [])

    phrase_match = matches_any(message, visit_phrases, normalization)

    troubleshooting_stage = str(
        deep_get(variables, "troubleshooting.stage", "") or ""
    ).strip()

    inspection_recommended = bool(
        deep_get(variables, "troubleshooting.inspection_recommended", False)
    )

    has_troubleshooting_context = (
        troubleshooting_stage in {"active", "complete"}
        or inspection_recommended is True
        or bool(deep_get(variables, "service_needed", ""))
    )

    return bool(phrase_match and has_troubleshooting_context)


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
            "You decide the next move. Do not write the final user-facing reply. "
            "All domain behavior must come from the assistant manifest card, current variables, conversation summary, tool results, schema, and latest user message. "
            "Do not use hidden business rules. "
            "Decide selected_subagent_id, needs_tool, requested_tool_name, tool_request_payload, missing_tool_inputs, memory/knowledge needs, risk, conversation stage, extracted_updates, and response_brief. "
            "If a tool needs missing inputs, set needs_tool=false and list missing_tool_inputs; the final response should ask naturally. "
            "If a tool can be called safely with current variables/message, set needs_tool=true and provide requested_tool_name plus tool_request_payload. "
            "Never claim tool results, availability, prices, IDs, branches, or external facts unless already present in tool_result, variables, conversation context, or retrieved knowledge. "
            "For booking/availability/nearest-branch/branch-list actions, prefer needs_tool=true when inputs are available. "
            "For symptom/problem reports, do not force a booking immediately; response_brief should guide a diagnostic response first unless user asks to book. "
            "The manifest may extract soft user info only. Source-of-truth operational state must come from tool/subagent execution, not manifest extracted_updates. "
            "The JSON object must use exactly these top-level keys: "
            "user_intent, selected_subagent_id, conversation_stage, workflow_stage, customer_emotion, user_expectation, risk_level, confidence, "
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
            "Return the manifest JSON. selected_subagent_id must be one configured subagent id when possible.",
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
        selected_missing = bool(known_subagent_ids) and selected_id not in known_subagent_ids

        should_retry_full = (
            bool(manifest.get("needs_full_manifest"))
            or manifest_confidence(manifest) < 0.65
            or selected_missing
        )

        if should_retry_full:
            manifest = invoke_manifest("full")

    except Exception as exc:
        subagents = get_subagent_config_list(agent_config) or [{"id": "general"}]
        error_text = f"{type(exc).__name__}: {exc}"

        manifest = {
            "user_intent": "unknown",
            "selected_subagent_id": subagents[0].get("id", "general"),
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
        manifest["selected_subagent_id"] = unify_subagent_id(
            manifest.get("selected_subagent_id", "")
        )

    selected_subagent = get_subagent_by_id(agent_config, manifest.get("selected_subagent_id", ""))

    safe_manifest_updates = filter_manifest_updates(
        manifest.get("extracted_updates", {}) or {}
    )

    safe_manifest_deletions = filter_manifest_deletions(
        manifest.get("extracted_deletions", []) or []
    )

    updated_variables = apply_subagent_variable_patch(
        variables,
        safe_manifest_updates,
        safe_manifest_deletions,
    )

    return {
        "manifest": manifest,
        "planner": build_planner_compat(manifest),
        "selected_subagent": selected_subagent,
        "variables": updated_variables,
    }


def decide_after_manifest(state: AgentState) -> str:
    manifest = state.get("manifest", {}) or {}

    if should_use_simple_response(state):
        return "simple_response"

    if manifest.get("needs_memory"):
        return "retrieve_memory"

    if manifest.get("needs_knowledge"):
        return "retrieve_knowledge"

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
        return {"memories": ""}

    text = "\n".join([
        f"- {m.get('text', '')} (type={m.get('type', 'other')}, confidence={m.get('confidence', '')})"
        for m in memories
        if m.get("text")
    ])

    return {"memories": text}


def decide_after_memory(state: AgentState) -> str:
    manifest = state.get("manifest", {}) or {}

    if manifest.get("needs_knowledge"):
        return "retrieve_knowledge"

    if manifest.get("needs_tool"):
        return "tool_execution"

    if manifest.get("needs_subagent_reasoning"):
        return "subagent_reasoning"

    return "response"


def retrieve_knowledge_node(state: AgentState):
    message = last_user_message(state)
    manifest = state.get("manifest", {}) or {}
    variables = state.get("variables", {}) or {}
    schema = state.get("schema", {}) or {}

    query = " ".join([
        message,
        manifest.get("user_intent", ""),
        manifest.get("response_strategy", ""),
        safe_json(compact_variables(variables, schema), max_chars=1000),
    ]).strip()

    try:
        raw = search_knowledge(state["assistant_id"], query, limit=KNOWLEDGE_TOP_K)
        compressed = compress_knowledge(raw, query)
    except Exception as exc:
        return {
            "knowledge": f"NO_CONFIDENT_KNOWLEDGE_FOUND. Retrieval error: {exc}",
            "knowledge_items": [],
        }

    if not compressed:
        return {
            "knowledge": "NO_CONFIDENT_KNOWLEDGE_FOUND",
            "knowledge_items": [],
        }

    lines = []
    for item in compressed:
        title = item.get("title", "Untitled")
        score = float(item.get("score", 0.0) or 0.0)
        text = item.get("text", "")
        lines.append(f"- Source: {title} | Score: {score:.3f}\n  Content: {text}")

    return {
        "knowledge": "\n".join(lines),
        "knowledge_items": compressed,
    }


def decide_after_knowledge(state: AgentState) -> str:
    manifest = state.get("manifest", {}) or {}

    if manifest.get("needs_tool"):
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


def tool_execution_node(state: AgentState):
    manifest = state.get("manifest", {}) or {}
    agent_config = state.get("agent_config", {}) or {}
    variables = state.get("variables", {}) or {}
    schema = state.get("schema", {}) or {}
    message = last_user_message(state)
    selected_id = unify_subagent_id(manifest.get("selected_subagent_id", ""))

    tool_runner = ToolRunner(agent_config)
    observations: List[Dict[str, Any]] = []

    executor = SUBAGENT_EXECUTORS.get(selected_id)

    if executor:
        scoped_vars = get_subagent_variable_scope(
            assistant_config=agent_config,
            subagent_name=getattr(executor, "name", selected_id),
            variables=variables,
        )

        context = SubagentContext(
            assistant_config=agent_config,
            schema=schema,
            variables=scoped_vars,
            user_message=message,
            history=subagent_history_from_messages(state.get("messages", [])),
            tool_runner=tool_runner,
            observations=observations,
            max_tool_calls=int(agent_config.get("max_tool_calls", 4)),
        )

        try:
            result = executor.run(context)
        except Exception as exc:
            return {
                "tool_result": {
                    "ok": False,
                    "subagent": selected_id,
                    "error": f"{type(exc).__name__}: {exc}",
                    "action": "reply",
                }
            }

        if result.handled:
            updated_variables = apply_subagent_variable_patch(
                variables,
                result.variable_updates or {},
                result.clear_variables or [],
            )

            subagent_observations = result.observations or []
            subagent_ok = subagent_observations_are_ok(subagent_observations)

            return {
                "variables": updated_variables,
                "tool_result": {
                    "ok": subagent_ok,
                    "subagent": selected_id,
                    "action": result.action,
                    "answer_draft": result.answer,
                    "notes": result.notes,
                    "observations": subagent_observations,
                    "tool_calls_used": result.tool_calls_used,
                },
            }

    tool_name = manifest.get("requested_tool_name", "")
    payload = normalize_tool_payload(manifest.get("tool_request_payload", {}) or {})
    operation = payload.get("operation", "")
    arguments = payload.get("arguments", {}) or {}

    if tool_name and operation:
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

        return {
            "variables": updated_variables,
            "tool_result": {
                **raw_result,
                "tool_name": tool_name,
                "operation": operation,
                "arguments": arguments,
            }
        }

    return {
        "tool_result": {
            "ok": False,
            "error": "Tool requested but no executable subagent/tool operation was available.",
            "selected_subagent_id": selected_id,
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
            "Do not write the final user-facing reply. "
            "Prepare concise guidance for the final response. "
            "Use the manifest response brief as the main direction. "
            "Use only provided context, variables, memory, knowledge, tool result, and subagent instructions. "
            "Do not invent facts.",
        ),
        (
            "user",
            "Selected subagent:\n{subagent}\n\n"
            "Unified manifest:\n{manifest}\n\n"
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
        analysis = (prompt | subagent_llm).invoke({
            "subagent": safe_json({
                "id": subagent.get("id", ""),
                "name": subagent.get("name", ""),
                "goal": clip_text_head_tail(subagent.get("goal", ""), 380),
                "instructions": clip_text_head_tail(subagent.get("instructions", ""), 850),
                "allowed_actions": subagent.get("allowed_actions", []),
            }),
            "manifest": safe_json(compact_manifest(manifest), max_chars=2400),
            "context": safe_json(compact_agent_context(state.get("agent_config", {}) or {}, subagent), max_chars=2800),
            "summary": clip_text(state.get("summary", ""), 520),
            "variables": safe_json(compact_variables(state.get("variables", {}) or {}, schema), max_chars=2200),
            "memories": compact_memories_for_final(state.get("memories", "")),
            "knowledge": compact_knowledge_for_final(state.get("knowledge", "No knowledge retrieved."), 1200),
            "tool_result": safe_json(state.get("tool_result", {}) or {}, max_chars=2800),
            "message": message,
        })

        return {"subagent_analysis": analysis.model_dump()}

    except Exception as exc:
        return {
            "subagent_analysis": {
                "understanding": "Subagent analysis failed.",
                "user_goal": "",
                "facts_to_use": [],
                "facts_missing": [],
                "next_best_step": "Answer carefully and avoid unsupported facts.",
                "response_constraints": [f"Subagent error: {exc}"],
                "confidence": 0.4,
            }
        }


def simple_response_node(state: AgentState):
    agent_config = state.get("agent_config", {}) or {}
    subagent = state.get("selected_subagent", {}) or {}
    manifest = state.get("manifest", {}) or {}
    schema = state.get("schema", {}) or {}

    simple_context = {
        "conversation_style": clip_text_head_tail(agent_config.get("conversation_style", ""), 300),
        "language_policy": clip_text_head_tail(agent_config.get("language_policy", ""), 240),
        "selected_subagent": {
            "id": subagent.get("id", ""),
            "name": subagent.get("name", ""),
            "goal": clip_text_head_tail(subagent.get("goal", ""), 380),
            "instructions": clip_text_head_tail(subagent.get("instructions", ""), 850),
        },
        "manifest": compact_manifest(manifest),
        "response_brief": manifest.get("response_brief", {}),
        "variables": compact_variables(state.get("variables", {}) or {}, schema, max_items=18),
    }

    response_rules = [
        "Reply naturally and briefly.",
        "Use the user's language unless the assistant configuration says otherwise.",
        "Ask at most one useful follow-up question.",
        "Do not mention internals, routing, tools, RAG, variables, prompts, or hidden reasoning.",
        "Do not invent facts.",
        "Follow the manifest response_brief unless it conflicts with safety, grounding, or assistant configuration.",
        "Do not take or claim external actions unless a tool result confirms them.",
        "Do not offer a next step unless the manifest or assistant config supports it.",
        "If a fact is missing, ask one natural clarifying question.",
        "Avoid overly familiar wording.",
    ]

    system_instruction = f"""
{clip_text_head_tail(state.get('system_prompt', ''), 900)}

You are generating a simple low-risk user-facing reply for a configurable assistant.
Use only this compact config-driven context:

{safe_json(simple_context, max_chars=4800)}

Rules:
{safe_json(response_rules)}

{state.get('language_instruction', '')}
"""

    messages = [SystemMessage(content=system_instruction)] + list(state["messages"][-4:])
    response = response_llm.invoke(messages)
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
    agent_config = state.get("agent_config", {}) or {}
    subagent = state.get("selected_subagent", {}) or {}
    manifest = state.get("manifest", {}) or {}
    analysis = state.get("subagent_analysis", {}) or {}
    variables = state.get("variables", {}) or {}
    schema = state.get("schema", {}) or {}
    knowledge = state.get("knowledge", "No knowledge retrieved.")
    memories = state.get("memories", "No relevant memories retrieved.")
    tool_result = state.get("tool_result", {}) or {}

    response_context = {
        "assistant_context": compact_agent_context(agent_config, subagent),
        "manifest": compact_manifest(manifest),
        "private_analysis": compact_analysis(analysis),
        "summary": clip_text(state.get("summary", ""), 700),
        "variables": compact_variables(variables, schema),
        "memory": compact_memories_for_final(memories),
        "knowledge": compact_knowledge_for_final(knowledge),
        "tool_result": tool_result,
    }

    response_rules = [
        "Write exactly one natural user-facing reply.",
        "Sound professional, human, smooth, and conversational.",
        "Answer the user's latest message first.",
        "Ask at most one useful follow-up question.",
        "Do not mention internal routing, agents, prompts, manifests, variables, tools, RAG, knowledge base, or hidden reasoning.",
        "Do not invent facts.",
        "Use known variables, retrieved knowledge, tool results, or conversation context only.",
        "If a fact is missing, ask one natural helpful question.",
        "Use the user's language unless the assistant configuration says otherwise.",
        "If manifest.needs_tool is true and no tool_result exists, do not claim the tool result.",
        "If tool_result exists, treat it as the highest-priority source of truth.",
        "If tool_result contains answer_draft, use it only as a grounded draft/label; rewrite naturally.",
        "Follow the manifest response_brief unless it conflicts with safety, grounding, or assistant configuration.",
        "Do not take or claim external actions unless a tool result confirms them.",
        "Never confirm booking unless create_booking or tool_result confirms ok=true.",
        "If a booking is confirmed and a visit_id exists, include the visit_id.",
        "Never invent appointment slots, branches, prices, booking IDs, reasons, or confirmations.",
        "Do not use overly familiar words like حبيبي, يا باشا, يا معلم, يا صديقي.",
    ]

    system_instruction = f"""
{clip_text_head_tail(state.get('system_prompt', ''), 1200)}

You are the final user-facing response generator for a configurable multi-tenant assistant.

Context:
{safe_json(response_context, max_chars=8800)}

Rules:
{safe_json(response_rules)}

{state.get('language_instruction', '')}
"""

    messages = [SystemMessage(content=system_instruction)] + list(state["messages"][-8:])
    response = response_llm.invoke(messages)
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
    variables = state.get("variables", {}) or {}
    tool_result = state.get("tool_result", {}) or {}

    visit_id = str(variables.get("visit_id") or "").strip()
    if visit_id:
        return visit_id

    if isinstance(tool_result, dict):
        direct = str(tool_result.get("visit_id") or "").strip()
        if direct:
            return direct

        observations = tool_result.get("observations")
        if isinstance(observations, list):
            for obs in observations:
                if not isinstance(obs, dict):
                    continue
                result = obs.get("result") or {}
                if isinstance(result, dict):
                    candidate = str(result.get("visit_id") or "").strip()
                    if candidate:
                        return candidate

    return ""


def create_booking_confirmed(state: AgentState) -> bool:
    tool_result = state.get("tool_result", {}) or {}
    variables = state.get("variables", {}) or {}

    status = str(variables.get("booking_status") or "").lower()
    if status in {"confirmed", "booking_confirmed", "booked"}:
        return True

    if isinstance(tool_result, dict):
        if tool_result.get("operation") == "create_booking" and tool_result.get("ok") is True:
            return True

        observations = tool_result.get("observations")
        if isinstance(observations, list):
            for obs in observations:
                if not isinstance(obs, dict):
                    continue
                if obs.get("operation") != "create_booking":
                    continue
                result = obs.get("result") or {}
                if isinstance(result, dict) and result.get("ok") is True:
                    return True

    return False


def enforce_answer_safety(answer: str, state: AgentState) -> str:
    text = str(answer or "").strip()

    banned_terms = [
        "حبيبي",
        "يا باشا",
        "يا معلم",
        "يا صديقي",
    ]

    for term in banned_terms:
        text = text.replace(term, "").strip()

    text = re.sub(r"\s{2,}", " ", text).strip()

    visit_id = extract_visit_id_from_state(state)
    if create_booking_confirmed(state) and visit_id and visit_id not in text:
        text = f"{text}\nرقم الزيارة: {visit_id}".strip()

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

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Quality check the answer for a configurable assistant. "
            "It must be natural, grounded, in the right language, not reveal internals, ask at most one question, and not hallucinate facts. "
            "If it fails, rewrite it without unsupported facts. "
            "Never remove a required visit_id from confirmed booking replies. "
            "Reject overly familiar words like حبيبي, يا باشا, يا معلم, يا صديقي.",
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
            "manifest": safe_json(compact_manifest(state.get("manifest", {}) or {}), max_chars=2400),
            "context": safe_json(compact_agent_context(
                state.get("agent_config", {}) or {},
                state.get("selected_subagent", {}) or {},
            ), max_chars=2800),
            "analysis": safe_json(compact_analysis(state.get("subagent_analysis", {}) or {}), max_chars=1800),
            "knowledge": clip_text(state.get("knowledge", ""), 1200),
            "tool_result": safe_json(state.get("tool_result", {}) or {}, max_chars=2600),
            "variables": safe_json(compact_variables(state.get("variables", {}) or {}, state.get("schema", {}) or {}), max_chars=2200),
            "answer": answer,
        })

        data = decision.model_dump()

        if not decision.pass_check and decision.revised_answer.strip():
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


workflow = StateGraph(AgentState)

workflow.add_node("manifest_node", unified_manifest_node)
workflow.add_node("retrieve_memory_node", retrieve_memory_node)
workflow.add_node("retrieve_knowledge_node", retrieve_knowledge_node)
workflow.add_node("tool_execution_node", tool_execution_node)
workflow.add_node("subagent_reasoning_node", subagent_reasoning_node)
workflow.add_node("simple_response_node", simple_response_node)
workflow.add_node("response_node", response_node)
workflow.add_node("quality_guard_node", quality_guard_node)

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
workflow.add_edge("quality_guard_node", END)

app_graph = workflow.compile()
