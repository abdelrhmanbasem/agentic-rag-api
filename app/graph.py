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
    MODEL_EXTRACTION,
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


class VariableExtraction(BaseModel):
    updates: Dict[str, Any] = Field(default_factory=dict)
    deletions: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.7)
    reasoning_summary: str = ""


class PlanDecision(BaseModel):
    user_intent: str
    selected_subagent_id: str
    needs_knowledge: bool
    needs_memory: bool
    needs_tool: bool
    requested_tool_name: str = ""
    should_extract_variables: bool = False
    extraction_reason: str = ""
    simple_response_mode: bool = False
    simple_response_reason: str = ""
    should_answer_now: bool = True
    missing_or_uncertain_info: List[str] = Field(default_factory=list)
    risk_level: str = Field(description="low|medium|high")
    response_strategy: str
    confidence: float
    reasoning_summary: str


class SubagentAnalysis(BaseModel):
    understanding: str
    user_goal: str
    facts_to_use: List[str] = Field(default_factory=list)
    facts_missing: List[str] = Field(default_factory=list)
    next_best_step: str
    response_constraints: List[str] = Field(default_factory=list)
    confidence: float


class QualityDecision(BaseModel):
    pass_check: bool
    revised_answer: str = ""
    issues: List[str] = Field(default_factory=list)


extractor_llm = llm(MODEL_EXTRACTION, temperature=0).with_structured_output(VariableExtraction)
planner_llm = llm(MODEL_PLANNER, temperature=0).with_structured_output(PlanDecision)
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


def compact_subagents_for_planner(agent_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    compact = []

    for subagent in agent_config.get("subagents") or []:
        compact.append({
            "id": subagent.get("id", ""),
            "name": subagent.get("name", ""),
            "when_to_use": clip_text(subagent.get("when_to_use", ""), 260),
            "goal": clip_text(subagent.get("goal", ""), 180),
            "allowed_actions": subagent.get("allowed_actions", []),
        })

    return compact


def ultra_compact_routing_card(agent_config: Dict[str, Any]) -> Dict[str, Any]:
    tools = []
    for tool in agent_config.get("tool_catalog") or []:
        tools.append({
            "name": tool.get("name", ""),
            "description": clip_text(tool.get("description", ""), 160),
            "required_inputs": tool.get("required_inputs", []),
        })

    return {
        "goal": clip_text(agent_config.get("assistant_goal", ""), 260),
        "style": clip_text(agent_config.get("conversation_style", ""), 220),
        "language": clip_text(agent_config.get("language_policy", ""), 160),
        "routing": clip_text(agent_config.get("routing_policy", ""), 520),
        "grounding": clip_text(agent_config.get("grounding_policy", ""), 320),
        "subagents": compact_subagents_for_planner(agent_config),
        "tools": tools,
    }


def compact_agent_context(agent_config: Dict[str, Any], selected_subagent: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "assistant_goal": clip_text(agent_config.get("assistant_goal", ""), 360),
        "conversation_style": clip_text(agent_config.get("conversation_style", ""), 260),
        "language_policy": clip_text(agent_config.get("language_policy", ""), 180),
        "grounding_policy": clip_text(agent_config.get("grounding_policy", ""), 420),
        "response_rules": (agent_config.get("response_rules") or [])[:8],
        "selected_subagent": {
            "id": selected_subagent.get("id", ""),
            "name": selected_subagent.get("name", ""),
            "goal": clip_text(selected_subagent.get("goal", ""), 260),
            "instructions": clip_text(selected_subagent.get("instructions", ""), 520),
            "allowed_actions": selected_subagent.get("allowed_actions", []),
        },
    }


def compact_schema_for_extraction(schema: Dict[str, Any]) -> Dict[str, Any]:
    if not schema:
        return {}

    important_keys = [
        "intent",
        "issue_description",
        "symptoms",
        "recommended_section",
        "service_needed",
        "customer_facing_section",
        "customer_agreed_to_visit",
        "booking_stage",
        "location_branch",
        "user_area",
        "appointment_date",
        "appointment_time",
        "slot_status",
        "booking_status",
        "unavailable_reason",
        "nearest_slots_text",
        "customer_full_name",
        "plate_digits",
        "phone_number",
        "phone_confirmed",
        "visit_id",
    ]

    compact = {}
    for key in important_keys:
        if key in schema:
            compact[key] = schema[key]

    return compact or schema


def compact_variables(variables: Dict[str, Any]) -> Dict[str, Any]:
    if not variables:
        return {}

    important_keys = [
        "intent",
        "workflow_stage",
        "conversation_stage",
        "diagnostic_stage",
        "issue_description",
        "symptoms",
        "known_facts",
        "recommended_section",
        "service_needed",
        "customer_facing_section",
        "next_service_action",
        "customer_agreed_to_visit",
        "booking_stage",
        "location_branch",
        "user_area",
        "appointment_date",
        "appointment_time",
        "slot_status",
        "booking_status",
        "unavailable_reason",
        "nearest_slots_text",
        "customer_full_name",
        "plate_digits",
        "phone_number",
        "phone_confirmed",
        "visit_id",
    ]

    compact = {}
    for key in important_keys:
        if key in variables and variables[key] not in [None, "", [], {}]:
            compact[key] = variables[key]

    for key, value in variables.items():
        if key in compact:
            continue
        if len(compact) >= 24:
            break
        if value not in [None, "", [], {}]:
            compact[key] = value

    return compact


def compact_planner(planner: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "user_intent": planner.get("user_intent", ""),
        "selected_subagent_id": planner.get("selected_subagent_id", ""),
        "needs_knowledge": planner.get("needs_knowledge", False),
        "needs_memory": planner.get("needs_memory", False),
        "needs_tool": planner.get("needs_tool", False),
        "requested_tool_name": planner.get("requested_tool_name", ""),
        "should_extract_variables": planner.get("should_extract_variables", False),
        "extraction_reason": clip_text(planner.get("extraction_reason", ""), 220),
        "simple_response_mode": planner.get("simple_response_mode", False),
        "simple_response_reason": clip_text(planner.get("simple_response_reason", ""), 220),
        "missing_or_uncertain_info": planner.get("missing_or_uncertain_info", [])[:6],
        "risk_level": planner.get("risk_level", "low"),
        "response_strategy": clip_text(planner.get("response_strategy", ""), 420),
        "confidence": planner.get("confidence", 0.0),
    }


def compact_analysis(analysis: Dict[str, Any]) -> Dict[str, Any]:
    if not analysis:
        return {}

    return {
        "understanding": clip_text(analysis.get("understanding", ""), 420),
        "user_goal": clip_text(analysis.get("user_goal", ""), 260),
        "facts_to_use": analysis.get("facts_to_use", [])[:5],
        "facts_missing": analysis.get("facts_missing", [])[:5],
        "next_best_step": clip_text(analysis.get("next_best_step", ""), 320),
        "response_constraints": analysis.get("response_constraints", [])[:6],
        "confidence": analysis.get("confidence", 0.0),
    }


def compact_knowledge_for_final(knowledge: str, max_chars: int = 1300) -> str:
    if not knowledge:
        return "No knowledge retrieved."
    return clip_text(knowledge, max_chars)


def compact_memories_for_final(memories: str, max_chars: int = 450) -> str:
    if not memories:
        return "No relevant memories retrieved."
    return clip_text(memories, max_chars)


def planner_confidence(planner: Dict[str, Any]) -> float:
    try:
        return float(planner.get("confidence", 0) or 0)
    except Exception:
        return 0.0


def should_skip_subagent_reasoning(state: AgentState) -> bool:
    planner = state.get("planner", {}) or {}

    if planner.get("needs_tool"):
        return False

    if planner.get("needs_knowledge"):
        return False

    if planner.get("risk_level") in ["high", "medium"]:
        return False

    if planner_confidence(planner) < 0.75:
        return False

    return True


def should_run_quality_guard(state: AgentState) -> bool:
    if not QUALITY_GUARD_ENABLED:
        return False

    planner = state.get("planner", {}) or {}
    tool_result = state.get("tool_result", {}) or {}
    knowledge = state.get("knowledge", "") or ""
    answer = state.get("final_answer", "") or ""

    if tool_result:
        return True

    if planner.get("needs_tool"):
        return True

    if planner.get("risk_level") in ["high", "medium"]:
        return True

    if "NO_CONFIDENT_KNOWLEDGE_FOUND" not in knowledge and knowledge.strip():
        return True

    if len(answer) > 700:
        return True

    return False


def can_use_simple_response(state: AgentState) -> bool:
    planner = state.get("planner", {}) or {}

    if not planner.get("simple_response_mode"):
        return False

    if planner.get("needs_tool"):
        return False

    if planner.get("needs_knowledge"):
        return False

    if planner.get("needs_memory"):
        return False

    if planner.get("should_extract_variables"):
        return False

    if planner.get("risk_level") in ["high", "medium"]:
        return False

    if planner_confidence(planner) < 0.8:
        return False

    if state.get("tool_result"):
        return False

    return True


def extract_variables_node(state: AgentState):
    message = last_user_message(state)
    variables = state.get("variables", {}) or {}
    schema = compact_schema_for_extraction(state.get("schema", {}) or {})

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Extract only clear durable variables from the latest message. "
            "Do not infer aggressively. Do not invent. "
            "If there are no useful updates, return empty updates. "
            "Support English and Egyptian Arabic.",
        ),
        (
            "user",
            "Schema:\n{schema}\n\n"
            "Current variables:\n{variables}\n\n"
            "Latest message:\n{message}",
        ),
    ])

    try:
        result = (prompt | extractor_llm).invoke({
            "schema": json.dumps(schema, ensure_ascii=False),
            "variables": json.dumps(compact_variables(variables), ensure_ascii=False),
            "message": message,
        })
        updated = apply_variable_patch(variables, result.updates, result.deletions)
        return {"variables": updated}
    except Exception as exc:
        return {"variables": variables, "planner": {"extract_error": str(exc)}}


def planner_node(state: AgentState):
    message = last_user_message(state)
    agent_config = state.get("agent_config", {}) or {}
    variables = state.get("variables", {}) or {}
    tool_result = state.get("tool_result", {}) or {}

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are the routing and planning brain of a configurable agentic assistant. "
            "Choose the best subagent and decide whether memory, knowledge, tool use, variable extraction, or simple response mode is needed. "
            "Use only the provided routing card, current variables, summary, tool result, and latest message. "
            "Do not use hardcoded business rules. "
            "Prefer smooth human conversation, but avoid unnecessary expensive steps. "
            "Be selective: simple greetings, acknowledgements, and first-step symptom messages usually do not need memory, knowledge, tools, or extraction unless they include durable details. "
            "Use simple_response_mode=true only when the message can be answered well from the selected subagent instructions, current context, and recent conversation without memory, KB, tools, extraction, or subagent reasoning. "
            "Do not use simple_response_mode for booking, availability, tool results, branch facts, prices, policies, high/medium risk, unclear facts, or multi-step reasoning. "
            "Do not over-escalate safety. Normal symptom descriptions should not become emergencies unless the routing card says so or the user clearly describes danger. "
            "Variable extraction should be true only when the user likely provided or changed durable info such as name, phone, date, time, branch, service choice, confirmation, cancellation, correction, or booking detail.",
        ),
        (
            "user",
            "Routing card:\n{routing_card}\n\n"
            "Conversation summary:\n{summary}\n\n"
            "Known variables:\n{variables}\n\n"
            "Tool result if any:\n{tool_result}\n\n"
            "Latest user message:\n{message}\n\n"
            "Return the plan. selected_subagent_id must be one of the provided subagent ids.",
        ),
    ])

    try:
        decision = (prompt | planner_llm).invoke({
            "routing_card": json.dumps(ultra_compact_routing_card(agent_config), ensure_ascii=False),
            "summary": clip_text(state.get("summary", ""), 420),
            "variables": json.dumps(compact_variables(variables), ensure_ascii=False),
            "tool_result": json.dumps(tool_result, ensure_ascii=False),
            "message": message,
        })
        planner = decision.model_dump()
    except Exception as exc:
        subagents = agent_config.get("subagents") or [{"id": "general"}]
        planner = {
            "user_intent": "unknown",
            "selected_subagent_id": subagents[0].get("id", "general"),
            "needs_knowledge": True,
            "needs_memory": True,
            "needs_tool": False,
            "requested_tool_name": "",
            "should_extract_variables": False,
            "extraction_reason": "",
            "simple_response_mode": False,
            "simple_response_reason": "",
            "should_answer_now": True,
            "missing_or_uncertain_info": [],
            "risk_level": "medium",
            "response_strategy": "Answer carefully and do not invent facts.",
            "confidence": 0.4,
            "reasoning_summary": f"Fallback planner due to error: {exc}",
        }

    selected_subagent = get_subagent_by_id(agent_config, planner.get("selected_subagent_id", ""))
    return {"planner": planner, "selected_subagent": selected_subagent}


def decide_after_planner_simple_or_extract(state: AgentState) -> str:
    if can_use_simple_response(state):
        return "simple_response"

    planner = state.get("planner", {}) or {}
    if planner.get("should_extract_variables"):
        return "extract_variables"

    return "retrieve_memory"


def planner_refine_node(state: AgentState):
    return planner_node(state)


def retrieve_memory_node(state: AgentState):
    planner = state.get("planner", {}) or {}
    if planner.get("needs_memory") is False:
        return {"memories": ""}

    message = last_user_message(state)
    try:
        memories = search_memories(state["assistant_id"], state["user_id"], message, limit=MEMORY_TOP_K)
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
    planner = state.get("planner", {}) or {}

    if can_use_simple_response(state):
        return "simple_response"

    if planner.get("needs_knowledge"):
        return "retrieve_knowledge"

    if should_skip_subagent_reasoning(state):
        return "response"

    return "subagent_reasoning"


def retrieve_knowledge_node(state: AgentState):
    message = last_user_message(state)
    planner = state.get("planner", {}) or {}
    variables = state.get("variables", {}) or {}

    query = " ".join([
        message,
        planner.get("user_intent", ""),
        planner.get("response_strategy", ""),
        json.dumps(compact_variables(variables), ensure_ascii=False),
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
        return {"knowledge": "NO_CONFIDENT_KNOWLEDGE_FOUND", "knowledge_items": []}

    lines = []
    for item in compressed:
        title = item.get("title", "Untitled")
        score = float(item.get("score", 0.0) or 0.0)
        text = item.get("text", "")
        lines.append(f"- Source: {title} | Score: {score:.3f}\n  Content: {text}")

    return {"knowledge": "\n".join(lines), "knowledge_items": compressed}


def decide_after_knowledge(state: AgentState) -> str:
    if should_skip_subagent_reasoning(state):
        return "response"
    return "subagent_reasoning"


def subagent_reasoning_node(state: AgentState):
    message = last_user_message(state)
    subagent = state.get("selected_subagent", {}) or {}
    planner = state.get("planner", {}) or {}

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are the selected subagent's private analysis brain. "
            "Do not write the final user-facing reply. "
            "Prepare concise guidance for the final response. "
            "Be grounded and identify missing facts instead of inventing them.",
        ),
        (
            "user",
            "Selected subagent:\n{subagent}\n\n"
            "Planner:\n{planner}\n\n"
            "Context:\n{context}\n\n"
            "Summary:\n{summary}\n\n"
            "Variables:\n{variables}\n\n"
            "Memory:\n{memories}\n\n"
            "Knowledge:\n{knowledge}\n\n"
            "Tool result:\n{tool_result}\n\n"
            "Latest message:\n{message}",
        ),
    ])

    try:
        analysis = (prompt | subagent_llm).invoke({
            "subagent": json.dumps({
                "id": subagent.get("id", ""),
                "name": subagent.get("name", ""),
                "goal": clip_text(subagent.get("goal", ""), 220),
                "instructions": clip_text(subagent.get("instructions", ""), 420),
                "allowed_actions": subagent.get("allowed_actions", []),
            }, ensure_ascii=False),
            "planner": json.dumps(compact_planner(planner), ensure_ascii=False),
            "context": json.dumps(compact_agent_context(state.get("agent_config", {}) or {}, subagent), ensure_ascii=False),
            "summary": clip_text(state.get("summary", ""), 420),
            "variables": json.dumps(compact_variables(state.get("variables", {}) or {}), ensure_ascii=False),
            "memories": compact_memories_for_final(state.get("memories", "")),
            "knowledge": compact_knowledge_for_final(state.get("knowledge", "No knowledge retrieved."), 900),
            "tool_result": json.dumps(state.get("tool_result", {}) or {}, ensure_ascii=False),
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
    planner = state.get("planner", {}) or {}
    variables = compact_variables(state.get("variables", {}) or {})

    simple_context = {
        "style": clip_text(agent_config.get("conversation_style", ""), 220),
        "language": clip_text(agent_config.get("language_policy", ""), 160),
        "selected_subagent": {
            "id": subagent.get("id", ""),
            "name": subagent.get("name", ""),
            "goal": clip_text(subagent.get("goal", ""), 220),
            "instructions": clip_text(subagent.get("instructions", ""), 420),
        },
        "planner": compact_planner(planner),
        "variables": variables,
    }

    response_rules = [
        "Reply naturally and briefly.",
        "Use the customer's language. If Egyptian Arabic, reply in natural Egyptian Arabic.",
        "Ask at most one useful follow-up question.",
        "Do not mention internals, routing, tools, RAG, variables, or prompts.",
        "Do not invent business facts.",
        "For first symptom messages, do not offer booking yet unless the user explicitly asks to book, visit, inspect, or schedule.",
        "Do not tell the customer to stop driving unless there is clear immediate danger.",
        "For normal brake squeak, brake whistle, or brake noise without weak braking or brake failure, respond diagnostically, not as an emergency.",
        "Avoid process phrases like: نوجهك للقسم المناسب, عشان نحدد المشكلة بشكل أدق, يرجى, سأقوم, هل ترغب.",
    ]

    system_instruction = f"""
{clip_text(state.get('system_prompt', ''), 650)}

You are generating a simple low-risk user-facing reply.
Use only this compact config-driven context:

{json.dumps(simple_context, ensure_ascii=False)}

Rules:
{json.dumps(response_rules, ensure_ascii=False)}

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
    planner = state.get("planner", {}) or {}
    analysis = state.get("subagent_analysis", {}) or {}
    variables = state.get("variables", {}) or {}
    knowledge = state.get("knowledge", "No knowledge retrieved.")
    memories = state.get("memories", "No relevant memories retrieved.")
    tool_result = state.get("tool_result", {}) or {}

    compact_context = compact_agent_context(agent_config, subagent)
    compact_vars = compact_variables(variables)
    compact_plan = compact_planner(planner)
    compact_private_analysis = compact_analysis(analysis)
    compact_knowledge = compact_knowledge_for_final(knowledge)
    compact_memory = compact_memories_for_final(memories)

    response_rules = [
        "Sound human, smooth, warm, and conversational.",
        "Answer the user's latest message first.",
        "Ask at most one useful follow-up question.",
        "Do not mention internal routing, agents, prompts, variables, tools, RAG, or knowledge base.",
        "Do not invent business facts. Use known variables, retrieved knowledge, tool results, or conversation context.",
        "If a fact is missing, say it naturally and ask one helpful question.",
        "If the user used Egyptian Arabic, reply in natural Egyptian Arabic.",
        "If planner.needs_tool is true and no tool_result exists, do not claim the tool result.",
        "If tool_result exists, it is the highest-priority source of truth.",
        "Do not tell the customer to stop driving unless there is clear immediate danger.",
        "Normal brake squeak, brake whistle, or brake noise without weak braking or brake failure should be diagnostic, not emergency.",
        "For first symptom messages, do not offer booking yet. Ask one diagnostic follow-up first unless the user explicitly asks to book, inspect, visit, or schedule.",
    ]

    system_instruction = f"""
{clip_text(state.get('system_prompt', ''), 900)}

You are the final user-facing response generator.

Compact context:
{json.dumps(compact_context, ensure_ascii=False)}

Plan:
{json.dumps(compact_plan, ensure_ascii=False)}

Private analysis:
{json.dumps(compact_private_analysis, ensure_ascii=False)}

Summary:
{clip_text(state.get('summary', ''), 520)}

Variables:
{json.dumps(compact_vars, ensure_ascii=False)}

Memory:
{compact_memory}

Knowledge:
{compact_knowledge}

Tool result:
{json.dumps(tool_result, ensure_ascii=False)}

{state.get('language_instruction', '')}

Rules:
{json.dumps(response_rules, ensure_ascii=False)}
"""

    messages = [SystemMessage(content=system_instruction)] + list(state["messages"][-8:])

    response = response_llm.invoke(messages)
    answer = response.content if hasattr(response, "content") else str(response)
    return {"messages": [AIMessage(content=answer)], "final_answer": answer}


def quality_guard_node(state: AgentState):
    if not should_run_quality_guard(state):
        return {"quality": {"pass_check": True, "skipped": True}}

    answer = state.get("final_answer", "")
    knowledge = state.get("knowledge", "")
    latest_user = last_user_message(state)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Quality check the answer. It must be natural, grounded, in the right language, "
            "not reveal internals, ask at most one question, and not hallucinate business facts. "
            "If it fails, rewrite it without unsupported facts.",
        ),
        (
            "user",
            "Latest message:\n{latest_user}\n\n"
            "Context:\n{context}\n\n"
            "Planner:\n{planner}\n\n"
            "Analysis:\n{analysis}\n\n"
            "Knowledge:\n{knowledge}\n\n"
            "Answer:\n{answer}",
        ),
    ])

    try:
        decision = (prompt | quality_llm).invoke({
            "latest_user": latest_user,
            "context": json.dumps(compact_agent_context(
                state.get("agent_config", {}) or {},
                state.get("selected_subagent", {}) or {},
            ), ensure_ascii=False),
            "planner": json.dumps(compact_planner(state.get("planner", {}) or {}), ensure_ascii=False),
            "analysis": json.dumps(compact_analysis(state.get("subagent_analysis", {}) or {}), ensure_ascii=False),
            "knowledge": clip_text(knowledge, 900),
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
        return {"quality": {"pass_check": True, "guard_error": str(exc)}}


workflow = StateGraph(AgentState)

workflow.add_node("planner", planner_node)
workflow.add_node("extract_variables", extract_variables_node)
workflow.add_node("planner_refine", planner_refine_node)
workflow.add_node("retrieve_memory", retrieve_memory_node)
workflow.add_node("retrieve_knowledge", retrieve_knowledge_node)
workflow.add_node("subagent_reasoning", subagent_reasoning_node)
workflow.add_node("response", response_node)
workflow.add_node("simple_response", simple_response_node)
workflow.add_node("quality_guard", quality_guard_node)

workflow.set_entry_point("planner")

workflow.add_conditional_edges(
    "planner",
    decide_after_planner_simple_or_extract,
    {
        "simple_response": "simple_response",
        "extract_variables": "extract_variables",
        "retrieve_memory": "retrieve_memory",
    },
)

workflow.add_edge("extract_variables", "planner_refine")
workflow.add_edge("planner_refine", "retrieve_memory")

workflow.add_conditional_edges(
    "retrieve_memory",
    decide_after_memory,
    {
        "simple_response": "simple_response",
        "retrieve_knowledge": "retrieve_knowledge",
        "subagent_reasoning": "subagent_reasoning",
        "response": "response",
    },
)

workflow.add_conditional_edges(
    "retrieve_knowledge",
    decide_after_knowledge,
    {
        "response": "response",
        "subagent_reasoning": "subagent_reasoning",
    },
)

workflow.add_edge("subagent_reasoning", "response")
workflow.add_edge("response", "quality_guard")
workflow.add_edge("quality_guard", END)
workflow.add_edge("simple_response", END)

app_graph = workflow.compile()
