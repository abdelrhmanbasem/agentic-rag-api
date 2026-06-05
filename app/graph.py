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
        "chained_subagent_id": "",
        "chained_subagent_reason": "",
        "detected_intents": [],
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
        "chained_subagent_id": manifest.get("chained_subagent_id", ""),
        "chained_subagent_reason": manifest.get("chained_subagent_reason", ""),
        "detected_intents": manifest.get("detected_intents", [])[:8],
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
        "chained_subagent_id": manifest.get("chained_subagent_id", ""),
        "chained_subagent_reason": manifest.get("chained_subagent_reason", ""),
        "detected_intents": manifest.get("detected_intents", []),
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
            "Decide selected_subagent_id, optional chained_subagent_id, needs_tool, requested_tool_name, tool_request_payload, missing_tool_inputs, memory/knowledge needs, risk, conversation stage, detected_intents, extracted_updates, and response_brief. "
            "If the latest message clearly contains two sequential needs where the second depends on the first, set selected_subagent_id to the first executor and chained_subagent_id to the second executor. "
            "Example: nearest branch plus availability in one message should run location first and booking second. Only chain when the second executor genuinely needs variables from the first executor. "
            "If a tool needs missing inputs, set needs_tool=false and list missing_tool_inputs; the final response should ask naturally. "
            "If a tool can be called safely with current variables/message, set needs_tool=true and provide requested_tool_name plus tool_request_payload. "
            "Never claim tool results, availability, prices, IDs, branches, or external facts unless already present in tool_result, variables, conversation context, or retrieved knowledge. "
            "For booking/availability/nearest-branch/branch-list actions, prefer needs_tool=true when inputs are available. "
            "For symptom/problem reports, do not force a booking immediately; response_brief should guide a diagnostic response first unless user asks to book. "
            "The manifest may extract soft user info only. Source-of-truth operational state must come from tool/subagent execution, not manifest extracted_updates. "
            "The JSON object must use exactly these top-level keys: "
            "user_intent, selected_subagent_id, chained_subagent_id, chained_subagent_reason, detected_intents, conversation_stage, workflow_stage, customer_emotion, user_expectation, risk_level, confidence, "
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
        chained_id = manifest.get("chained_subagent_id", "")
        selected_missing = bool(known_subagent_ids) and selected_id and selected_id not in known_subagent_ids
        chained_missing = bool(known_subagent_ids) and chained_id and chained_id not in known_subagent_ids

        should_retry_full = (
            bool(manifest.get("needs_full_manifest"))
            or manifest_confidence(manifest) < 0.65
            or selected_missing
            or chained_missing
            or (manifest.get("needs_tool") and not manifest.get("requested_tool_name") and not manifest.get("selected_subagent_id"))
            or (manifest.get("chained_subagent_id") and not manifest.get("selected_subagent_id"))
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
    variables: Dict[str, Any]
) -> Dict[str, Any]:
    op = str(operation or "").strip()
    args = dict(arguments or {})

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
    missing_inputs: List[str]
) -> Dict[str, Any]:
    if "date" in missing_inputs:
        answer_draft = "BOOKING_MISSING_DATE"
        next_move = "Ask the customer which day/date they want before checking appointment availability."
    elif "branch" in missing_inputs:
        answer_draft = "BOOKING_MISSING_BRANCH"
        next_move = "Ask the customer which branch or area they want before checking appointment availability."
    else:
        answer_draft = "BOOKING_MISSING_FIELDS"
        next_move = "Ask for the next missing booking detail before calling the tool."

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
            "tone": "natural professional Egyptian Arabic",
            "language": "same as user",
            "reply_length": "short",
            "next_move": next_move,
            "must_do": [
                "ask for the missing input",
                "do not call availability tools without branch and normalized date",
                "ask only one question"
            ],
            "must_not_do": [
                "do not say there are no available slots",
                "do not invent slots",
                "do not invent dates",
                "do not claim a tool result was checked"
            ]
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

    if operation == "list_available_slots" and "date" in missing_inputs:
        variable_updates["booking.stage"] = "awaiting_date"

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
            result.variable_updates or {},
            result.clear_variables or [],
        )

        return result, updated_variables, result.observations or []

    primary_result = None

    if selected_id:
        primary_result, variables, obs1 = run_executor(selected_id, variables)
        all_observations.extend(obs1)

    if primary_result and primary_result.handled:
        if chained_id and chained_id != selected_id:
            chained_result, variables, obs2 = run_executor(chained_id, variables)
            all_observations.extend(obs2)

            if chained_result and chained_result.handled:
                return {
                    "variables": variables,
                    "tool_result": {
                        "ok": subagent_observations_are_ok(all_observations),
                        "subagent": f"{selected_id}+{chained_id}",
                        "primary_subagent": selected_id,
                        "chained_subagent": chained_id,
                        "chained": True,
                        "action": chained_result.action,
                        "answer_draft": chained_result.answer,
                        "primary_answer_draft": primary_result.answer,
                        "notes": "Executed chained subagents sequentially.",
                        "primary_notes": primary_result.notes,
                        "chained_notes": chained_result.notes,
                        "observations": all_observations,
                        "tool_calls_used": (primary_result.tool_calls_used or 0) + (chained_result.tool_calls_used or 0),
                    },
                }

        return {
            "variables": variables,
            "tool_result": {
                "ok": subagent_observations_are_ok(all_observations),
                "subagent": selected_id,
                "action": primary_result.action,
                "answer_draft": primary_result.answer,
                "notes": primary_result.notes,
                "observations": all_observations,
                "tool_calls_used": primary_result.tool_calls_used,
            },
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
            "Reason privately, but return only the structured SubagentAnalysis fields. Do not expose chain-of-thought. "
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
        "tool_result": tool_result,
        "answer_safety": agent_config.get("answer_safety", {}) or {},
        "template_response_policy": compact_template_response_policy(agent_config, tool_result),
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
        "If template_response_policy contains a policy for tool_result.answer_draft, obey its must_do, must_not_do, and safe_examples.",
        "If tool_result.action is ask_user or answer_draft represents missing fields, do not use wording that implies an external action was completed.",
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

    common_keys = [
        "BOOKING_CONFIRMED",
        "BOOKING_COMPLETED_CLOSING",
        "BOOKING_ALREADY_CONFIRMED",
        "BOOKING_MISSING_FIELDS",
        "BOOKING_MISSING_FULL_NAME",
        "BOOKING_MISSING_PLATE_NUMBER",
        "BOOKING_MISSING_NAME_AND_PLATE",
        "BOOKING_CONFIRM_SELECTED_SLOT",
        "BOOKING_SLOTS_FOUND",
    ]

    for key in common_keys:
        if key in policies and key not in selected_keys:
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
    subagents = agent_config.get("subagents", {}) or {}
    booking_config = {}

    if isinstance(subagents, dict):
        booking_config = subagents.get("booking", {}) or {}

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
    labels = configured_booking_template_labels(
        agent_config,
        key_prefixes=["missing_"],
        value_prefixes=["BOOKING_MISSING"],
    )

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

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Quality check the answer for a configurable assistant. "
            "It must be natural, grounded, in the right language, not reveal internals, ask at most one question, and not hallucinate facts. "
            "If it fails, rewrite it without unsupported facts. "
            "Obey the configured answer_safety and template_response_policy. "
            "If tool_result.action is ask_user or tool_result.answer_draft is a missing-field label, never rewrite it into completed-action wording. "
            "Never remove a required visit_id from confirmed booking replies. "
            "Reject any overly familiar words listed in the assistant safety config.",
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
            "context": safe_json({
                **compact_agent_context(
                    state.get("agent_config", {}) or {},
                    state.get("selected_subagent", {}) or {},
                ),
                "answer_safety": (state.get("agent_config", {}) or {}).get("answer_safety", {}) or {},
                "template_response_policy": compact_template_response_policy(
                    state.get("agent_config", {}) or {},
                    state.get("tool_result", {}) or {},
                ),
            }, max_chars=3600),
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
