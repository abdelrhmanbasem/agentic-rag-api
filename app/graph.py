from typing import TypedDict, Annotated, Sequence, Dict, Any, List, Optional
import json
import operator

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
from app.variables import apply_variable_patch


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


# JSON mode is more robust for a generic multi-tenant manifest than strict Pydantic structured output.
manifest_llm = llm(MODEL_PLANNER, temperature=0).bind(response_format={"type": "json_object"})
subagent_llm = llm(MODEL_SUBAGENT, temperature=0.2).with_structured_output(SubagentAnalysis)
response_llm = llm(MODEL_RESPONSE, temperature=0.55, max_tokens=MAX_OUTPUT_TOKENS)
quality_llm = llm(MODEL_QUALITY, temperature=0).with_structured_output(QualityDecision)


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


def safe_json(value: Any, max_chars: Optional[int] = None) -> str:
    text = json.dumps(value, ensure_ascii=False)
    if max_chars:
        return clip_text(text, max_chars)
    return text


def parse_json_object(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value

    if not isinstance(value, str):
        return {}

    try:
        parsed = json.loads(value or "{}")
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


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


def get_schema_fields(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Supports both shapes:
    1. {"field_name": {...}}
    2. {"schema": {"field_name": {...}}}
    """
    if not isinstance(schema, dict):
        return {}

    nested = schema.get("schema")
    if isinstance(nested, dict):
        return nested

    return schema


def summarize_schema_fields(schema: Dict[str, Any], max_fields: int = 50) -> Dict[str, Any]:
    """
    Generic schema compaction.
    No assistant-specific fields are hardcoded.
    The engine trusts whatever schema was uploaded for this assistant.
    """
    fields = get_schema_fields(schema)
    if not fields:
        return {}

    compact = {}
    for idx, (key, value) in enumerate(fields.items()):
        if idx >= max_fields:
            break

        if isinstance(value, dict):
            compact[key] = {
                k: clip_text(v, 140) if isinstance(v, str) else v
                for k, v in value.items()
                if k in ["type", "description", "enum", "items", "required"]
            }
        else:
            compact[key] = clip_text(value, 140)

    return compact


def compact_variables(
    variables: Dict[str, Any],
    schema: Optional[Dict[str, Any]] = None,
    max_items: int = 40,
) -> Dict[str, Any]:
    """
    Generic variable compaction.
    Prioritizes variables defined by the assistant schema, then keeps other non-empty variables.
    No business-specific keys are hardcoded.
    """
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


def get_subagent_by_id(agent_config: Dict[str, Any], subagent_id: str) -> Dict[str, Any]:
    subagents = agent_config.get("subagents") or []

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

    for subagent in agent_config.get("subagents") or []:
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
        tools.append({
            "name": tool.get("name", ""),
            "description": clip_text(tool.get("description", ""), 220),
            "required_inputs": tool.get("required_inputs", []),
            "result_fields": tool.get("result_fields", []),
            "source_of_truth": tool.get("source_of_truth", False),
            "result_policy": clip_text(tool.get("result_policy", ""), 260),
        })

    return tools


def unified_manifest_card(agent_config: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generic assistant manifest card.
    Everything assistant-specific comes from /assistants and /schemas.
    """
    return {
        "assistant_goal": clip_text(agent_config.get("assistant_goal", ""), 320),
        "conversation_style": clip_text(agent_config.get("conversation_style", ""), 280),
        "language_policy": clip_text(agent_config.get("language_policy", ""), 220),
        "routing_policy": clip_text(agent_config.get("routing_policy", ""), 650),
        "grounding_policy": clip_text(agent_config.get("grounding_policy", ""), 420),
        "response_rules": (agent_config.get("response_rules") or [])[:10],
        "subagents": compact_subagents_for_manifest(agent_config),
        "tools": compact_tool_catalog(agent_config),
        "variable_schema": summarize_schema_fields(schema, max_fields=50),
    }


def compact_agent_context(agent_config: Dict[str, Any], selected_subagent: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "assistant_goal": clip_text(agent_config.get("assistant_goal", ""), 400),
        "conversation_style": clip_text(agent_config.get("conversation_style", ""), 320),
        "language_policy": clip_text(agent_config.get("language_policy", ""), 220),
        "grounding_policy": clip_text(agent_config.get("grounding_policy", ""), 500),
        "response_rules": (agent_config.get("response_rules") or [])[:10],
        "selected_subagent": {
            "id": selected_subagent.get("id", ""),
            "name": selected_subagent.get("name", ""),
            "goal": clip_text(selected_subagent.get("goal", ""), 280),
            "instructions": clip_text(selected_subagent.get("instructions", ""), 650),
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
        "response_style": manifest.get("response_style", ""),
        "reply_length": manifest.get("reply_length", ""),
        "should_ask_question": manifest.get("should_ask_question", False),
        "question_goal": manifest.get("question_goal", ""),
        "should_offer_next_action": manifest.get("should_offer_next_action", False),
        "response_strategy": clip_text(manifest.get("response_strategy", ""), 480),
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
        "facts_to_use": analysis.get("facts_to_use", [])[:6],
        "facts_missing": analysis.get("facts_missing", [])[:6],
        "next_best_step": clip_text(analysis.get("next_best_step", ""), 360),
        "response_constraints": analysis.get("response_constraints", [])[:8],
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


def should_use_simple_response(state: AgentState) -> bool:
    manifest = state.get("manifest", {}) or {}

    if not manifest.get("simple_response_mode"):
        return False

    if manifest.get("needs_tool"):
        return False

    if manifest.get("needs_knowledge"):
        return False

    if manifest.get("needs_memory"):
        return False

    if manifest.get("needs_subagent_reasoning"):
        return False

    if manifest.get("risk_level") in ["high", "medium"]:
        return False

    # Safe lower threshold: other gates already block risky, tool, KB, memory, and complex cases.
    if manifest_confidence(manifest) < 0.7:
        return False

    if state.get("tool_result"):
        return False

    return True


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
    """
    Keeps compatibility with existing /chat response code that expects final_state['planner'].
    Generic fields only.
    """
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
        "tool_request_payload": manifest.get("tool_request_payload", {}),
        "missing_tool_inputs": manifest.get("missing_tool_inputs", []),
        "manifest_error": manifest.get("manifest_error", ""),
        "reasoning_summary": manifest.get("reasoning_summary", ""),
    }


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
            "Return ONLY a valid JSON object. No markdown. No prose. "
            "Do not leave confidence at the default value. Use confidence 0.85-0.98 when the route and next step are clear. Use 0.6-0.75 only when uncertain. "
            "Do not write the final user-facing answer. "
            "Decide routing, variable updates, memory/knowledge/tool needs, risk, conversation stage, response brief, and whether simple response mode is appropriate. "
            "Do not use hidden business rules. Use only the assistant manifest card, current variables, summary, tool result, and latest user message. "
            "All domain behavior must come from the assistant config, subagents, tool catalog, variable schema, retrieved knowledge if requested, and conversation context. "
            "Extract only clear durable variable updates from the latest message using the provided variable schema. Do not infer aggressively. "
            "Use simple_response_mode only for low-risk messages that can be answered well from config/context without memory, knowledge, tools, subagent reasoning, or quality guard. "
            "Do not use simple_response_mode when the message requires business facts, policy facts, tool results, external actions, high/medium risk handling, uncertain facts, or multi-step reasoning. "
            "Never claim tool results, availability, prices, policies, IDs, or external facts unless they are present in tool_result, variables, conversation context, or retrieved knowledge. "
            "The JSON object must use exactly these top-level keys: "
            "user_intent, selected_subagent_id, conversation_stage, workflow_stage, customer_emotion, user_expectation, risk_level, confidence, "
            "simple_response_mode, simple_response_reason, needs_knowledge, needs_memory, needs_tool, requested_tool_name, tool_request_payload, missing_tool_inputs, "
            "needs_subagent_reasoning, needs_quality_guard, needs_style_repair, extracted_updates, extracted_deletions, response_style, reply_length, "
            "should_ask_question, question_goal, should_offer_next_action, response_brief, response_strategy, reasoning_summary. "
            "response_brief must be an object with keys: tone, language, reply_length, must_do, must_not_do, next_move. "
            "extracted_updates and tool_request_payload must be JSON objects. extracted_deletions and missing_tool_inputs must be arrays. "
            "Create a response_brief that guides the response LLM: tone, language, reply_length, must_do, must_not_do, and next_move. "
            "For simple symptom or problem reports, response_brief.must_do should include: acknowledge the issue naturally, mention likely causes only if supported by assistant config or common diagnostic reasoning, then ask one specific follow-up.",
        ),
        (
            "user",
            "Assistant manifest card:\n{manifest_card}\n\n"
            "Conversation summary:\n{summary}\n\n"
            "Current variables:\n{variables}\n\n"
            "Tool result if any:\n{tool_result}\n\n"
            "Latest user message:\n{message}\n\n"
            "Return the manifest JSON. selected_subagent_id must be one of the configured subagent ids.",
        ),
    ])

    try:
        decision = (prompt | manifest_llm).invoke({
            "manifest_card": safe_json(unified_manifest_card(agent_config, schema), max_chars=6200),
            "summary": clip_text(state.get("summary", ""), 500),
            "variables": safe_json(compact_variables(variables, schema), max_chars=1800),
            "tool_result": safe_json(tool_result, max_chars=2200),
            "message": message,
        })
        manifest = parse_manifest_response(decision)
        manifest = normalize_json_manifest(manifest)

    except Exception as exc:
        subagents = agent_config.get("subagents") or [{"id": "general"}]
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
        }

    selected_subagent = get_subagent_by_id(agent_config, manifest.get("selected_subagent_id", ""))

    updated_variables = apply_variable_patch(
        variables,
        manifest.get("extracted_updates", {}) or {},
        manifest.get("extracted_deletions", []) or [],
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
                "goal": clip_text(subagent.get("goal", ""), 260),
                "instructions": clip_text(subagent.get("instructions", ""), 520),
                "allowed_actions": subagent.get("allowed_actions", []),
            }),
            "manifest": safe_json(compact_manifest(manifest), max_chars=2200),
            "context": safe_json(compact_agent_context(state.get("agent_config", {}) or {}, subagent), max_chars=2200),
            "summary": clip_text(state.get("summary", ""), 520),
            "variables": safe_json(compact_variables(state.get("variables", {}) or {}, schema), max_chars=1800),
            "memories": compact_memories_for_final(state.get("memories", "")),
            "knowledge": compact_knowledge_for_final(state.get("knowledge", "No knowledge retrieved."), 1000),
            "tool_result": safe_json(state.get("tool_result", {}) or {}, max_chars=2200),
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
        "conversation_style": clip_text(agent_config.get("conversation_style", ""), 260),
        "language_policy": clip_text(agent_config.get("language_policy", ""), 180),
        "selected_subagent": {
            "id": subagent.get("id", ""),
            "name": subagent.get("name", ""),
            "goal": clip_text(subagent.get("goal", ""), 260),
            "instructions": clip_text(subagent.get("instructions", ""), 520),
        },
        "manifest": compact_manifest(manifest),
        "response_brief": manifest.get("response_brief", {}),
        "variables": compact_variables(state.get("variables", {}) or {}, schema),
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
    ]

    system_instruction = f"""
{clip_text(state.get('system_prompt', ''), 750)}

You are generating a simple low-risk user-facing reply for a configurable assistant.
Use only this compact config-driven context:

{safe_json(simple_context, max_chars=4500)}

Rules:
{safe_json(response_rules)}

{state.get('language_instruction', '')}
"""

    messages = [SystemMessage(content=system_instruction)] + list(state["messages"][-4:])

    response = response_llm.invoke(messages)
    answer = response.content if hasattr(response, "content") else str(response)

    return {
        "messages": [AIMessage(content=answer)],
        "final_answer": answer,
        "quality": {
            "pass_check": True,
            "skipped": True,
            "simple_response_mode": True,
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
        "summary": clip_text(state.get("summary", ""), 600),
        "variables": compact_variables(variables, schema),
        "memory": compact_memories_for_final(memories),
        "knowledge": compact_knowledge_for_final(knowledge),
        "tool_result": tool_result,
    }

    response_rules = [
        "Sound human, smooth, warm, and conversational.",
        "Answer the user's latest message first.",
        "Ask at most one useful follow-up question.",
        "Do not mention internal routing, agents, prompts, variables, tools, RAG, knowledge base, or hidden reasoning.",
        "Do not invent facts.",
        "Use known variables, retrieved knowledge, tool results, or conversation context.",
        "If a fact is missing, ask one natural helpful question.",
        "Use the user's language unless the assistant configuration says otherwise.",
        "If manifest.needs_tool is true and no tool_result exists, do not claim the tool result.",
        "If tool_result exists, treat it as the highest-priority source of truth.",
        "Follow the manifest response_brief unless it conflicts with safety, grounding, or assistant configuration.",
        "Do not take or claim external actions unless a tool result confirms them.",
    ]

    system_instruction = f"""
{clip_text(state.get('system_prompt', ''), 950)}

You are the final user-facing response generator for a configurable multi-tenant assistant.

Context:
{safe_json(response_context, max_chars=7000)}

Rules:
{safe_json(response_rules)}

{state.get('language_instruction', '')}
"""

    messages = [SystemMessage(content=system_instruction)] + list(state["messages"][-8:])

    response = response_llm.invoke(messages)
    answer = response.content if hasattr(response, "content") else str(response)

    return {
        "messages": [AIMessage(content=answer)],
        "final_answer": answer,
    }


def quality_guard_node(state: AgentState):
    if not should_run_quality_guard(state):
        return {"quality": {"pass_check": True, "skipped": True}}

    answer = state.get("final_answer", "")
    knowledge = state.get("knowledge", "")
    latest_user = last_user_message(state)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Quality check the answer for a configurable assistant. "
            "It must be natural, grounded, in the right language, not reveal internals, ask at most one question, and not hallucinate facts. "
            "If it fails, rewrite it without unsupported facts.",
        ),
        (
            "user",
            "Latest message:\n{latest_user}\n\n"
            "Manifest:\n{manifest}\n\n"
            "Context:\n{context}\n\n"
            "Analysis:\n{analysis}\n\n"
            "Knowledge:\n{knowledge}\n\n"
            "Answer:\n{answer}",
        ),
    ])

    try:
        decision = (prompt | quality_llm).invoke({
            "latest_user": latest_user,
            "manifest": safe_json(compact_manifest(state.get("manifest", {}) or {}), max_chars=2200),
            "context": safe_json(compact_agent_context(
                state.get("agent_config", {}) or {},
                state.get("selected_subagent", {}) or {},
            ), max_chars=2200),
            "analysis": safe_json(compact_analysis(state.get("subagent_analysis", {}) or {}), max_chars=1800),
            "knowledge": clip_text(knowledge, 1000),
            "answer": answer,
        })

        data = decision.model_dump()

        if not decision.pass_check and decision.revised_answer.strip():
            revised = decision.revised_answer.strip()
            return {
                "messages": [AIMessage(content=revised)],
                "final_answer": revised,
                "quality": data,
            }

        return {"quality": data}

    except Exception as exc:
        return {
            "quality": {
                "pass_check": True,
                "guard_error": str(exc),
            }
        }


workflow = StateGraph(AgentState)

workflow.add_node("manifest", unified_manifest_node)
workflow.add_node("retrieve_memory", retrieve_memory_node)
workflow.add_node("retrieve_knowledge", retrieve_knowledge_node)
workflow.add_node("subagent_reasoning", subagent_reasoning_node)
workflow.add_node("simple_response", simple_response_node)
workflow.add_node("response", response_node)
workflow.add_node("quality_guard", quality_guard_node)

workflow.set_entry_point("manifest")

workflow.add_conditional_edges(
    "manifest",
    decide_after_manifest,
    {
        "simple_response": "simple_response",
        "retrieve_memory": "retrieve_memory",
        "retrieve_knowledge": "retrieve_knowledge",
        "subagent_reasoning": "subagent_reasoning",
        "response": "response",
    },
)

workflow.add_conditional_edges(
    "retrieve_memory",
    decide_after_memory,
    {
        "retrieve_knowledge": "retrieve_knowledge",
        "subagent_reasoning": "subagent_reasoning",
        "response": "response",
    },
)

workflow.add_conditional_edges(
    "retrieve_knowledge",
    decide_after_knowledge,
    {
        "subagent_reasoning": "subagent_reasoning",
        "response": "response",
    },
)

workflow.add_edge("subagent_reasoning", "response")
workflow.add_edge("response", "quality_guard")
workflow.add_edge("quality_guard", END)
workflow.add_edge("simple_response", END)

app_graph = workflow.compile()
