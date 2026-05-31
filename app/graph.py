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
    kwargs = {"model": model, "temperature": temperature, "api_key": OPENAI_API_KEY}
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


def format_subagents(agent_config: Dict[str, Any]) -> str:
    subagents = agent_config.get("subagents") or []
    return json.dumps(subagents, ensure_ascii=False, indent=2)


def get_subagent_by_id(agent_config: Dict[str, Any], subagent_id: str) -> Dict[str, Any]:
    subagents = agent_config.get("subagents") or []
    if not subagents:
        return {
            "id": "general",
            "name": "General",
            "instructions": "Help the user naturally and accurately.",
        }

    for subagent in subagents:
        if subagent.get("id") == subagent_id:
            return subagent

    return subagents[0]


def extract_variables_node(state: AgentState):
    message = last_user_message(state)
    variables = state.get("variables", {}) or {}
    schema = state.get("schema", {}) or {}
    agent_config = state.get("agent_config", {}) or {}

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are the variable extraction brain of a configurable agentic assistant. "
            "Extract only what the user clearly stated, implied with high confidence, or corrected. "
            "Use the assistant schema and config. Do not invent. "
            "If the user changes their mind, update the value. If they remove a value, place the key in deletions. "
            "Support English and Egyptian Arabic.",
        ),
        (
            "user",
            "Assistant config:\n{agent_config}\n\n"
            "Variable schema:\n{schema}\n\n"
            "Current variables:\n{variables}\n\n"
            "Latest user message:\n{message}",
        ),
    ])

    try:
        result = (prompt | extractor_llm).invoke({
            "agent_config": json.dumps(agent_config, ensure_ascii=False),
            "schema": json.dumps(schema, ensure_ascii=False),
            "variables": json.dumps(variables, ensure_ascii=False),
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
            "You are the main reasoning brain of a configurable Agentic RAG system. "
            "Think internally, analyze the user's goal, deduce what is needed, choose the best subagent from the provided config, "
            "and decide whether memory, knowledge search, or tool usage is needed. "
            "Do not use fixed business rules. Use only the assistant config, subagent descriptions, current context, and your reasoning. "
            "Prefer smooth human conversation over token saving. "
            "Avoid hallucination by requesting knowledge when factual business info may be needed. "
            "If the latest message is a tool result, select the correct tool-result handler subagent and answer from the tool result.",
        ),
        (
            "user",
            "System prompt:\n{system_prompt}\n\n"
            "Assistant config:\n{agent_config}\n\n"
            "Available subagents:\n{subagents}\n\n"
            "Conversation summary:\n{summary}\n\n"
            "Known variables:\n{variables}\n\n"
            "Latest tool result, if any:\n{tool_result}\n\n"
            "Latest user message:\n{message}\n\n"
            "Return a decision. The selected_subagent_id MUST be one of the configured subagent ids.",
        ),
    ])

    try:
        decision = (prompt | planner_llm).invoke({
            "system_prompt": state.get("system_prompt", ""),
            "agent_config": json.dumps(agent_config, ensure_ascii=False),
            "subagents": format_subagents(agent_config),
            "summary": state.get("summary", ""),
            "variables": json.dumps(variables, ensure_ascii=False),
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
            "should_answer_now": True,
            "missing_or_uncertain_info": [],
            "risk_level": "medium",
            "response_strategy": "Answer carefully and do not invent facts.",
            "confidence": 0.4,
            "reasoning_summary": f"Fallback planner due to error: {exc}",
        }

    selected_subagent = get_subagent_by_id(agent_config, planner.get("selected_subagent_id", ""))
    return {"planner": planner, "selected_subagent": selected_subagent}


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
    if planner.get("needs_knowledge"):
        return "retrieve_knowledge"
    return "subagent_reasoning"


def retrieve_knowledge_node(state: AgentState):
    message = last_user_message(state)
    planner = state.get("planner", {}) or {}
    variables = state.get("variables", {}) or {}

    query = " ".join([
        message,
        planner.get("user_intent", ""),
        planner.get("response_strategy", ""),
        json.dumps(variables, ensure_ascii=False),
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


def subagent_reasoning_node(state: AgentState):
    message = last_user_message(state)
    subagent = state.get("selected_subagent", {}) or {}
    planner = state.get("planner", {}) or {}

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are the selected subagent's private analysis brain. "
            "Do not write the final user-facing reply. "
            "Analyze the situation according to the selected subagent config and prepare structured guidance for the final response LLM. "
            "Be grounded. Identify missing facts instead of inventing them. "
            "If a tool is needed, explain exactly why and which inputs are still missing.",
        ),
        (
            "user",
            "Selected subagent:\n{subagent}\n\n"
            "Planner decision:\n{planner}\n\n"
            "Assistant config:\n{agent_config}\n\n"
            "Conversation summary:\n{summary}\n\n"
            "Known variables:\n{variables}\n\n"
            "Relevant memories:\n{memories}\n\n"
            "Retrieved knowledge:\n{knowledge}\n\n"
            "Tool result:\n{tool_result}\n\n"
            "Latest user message:\n{message}",
        ),
    ])

    try:
        analysis = (prompt | subagent_llm).invoke({
            "subagent": json.dumps(subagent, ensure_ascii=False),
            "planner": json.dumps(planner, ensure_ascii=False),
            "agent_config": json.dumps(state.get("agent_config", {}), ensure_ascii=False),
            "summary": state.get("summary", ""),
            "variables": json.dumps(state.get("variables", {}), ensure_ascii=False),
            "memories": state.get("memories", ""),
            "knowledge": state.get("knowledge", "No knowledge retrieved."),
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


def response_node(state: AgentState):
    agent_config = state.get("agent_config", {}) or {}
    subagent = state.get("selected_subagent", {}) or {}
    planner = state.get("planner", {}) or {}
    analysis = state.get("subagent_analysis", {}) or {}
    variables = state.get("variables", {}) or {}
    knowledge = state.get("knowledge", "No knowledge retrieved.")
    memories = state.get("memories", "No relevant memories retrieved.")
    tool_result = state.get("tool_result", {}) or {}

    system_instruction = f"""
{state.get('system_prompt', '')}

You are the final response generator.
The previous nodes reasoned, routed, retrieved, and analyzed. Your job is to generate the actual user-facing reply.

Assistant configuration:
{json.dumps(agent_config, ensure_ascii=False)}

Selected subagent:
{json.dumps(subagent, ensure_ascii=False)}

Planner decision:
{json.dumps(planner, ensure_ascii=False)}

Subagent private analysis:
{json.dumps(analysis, ensure_ascii=False)}

Conversation summary:
{state.get('summary', '')}

Known variables:
{json.dumps(variables, ensure_ascii=False)}

Relevant long-term memories:
<memory>
{memories}
</memory>

Retrieved knowledge:
<knowledge>
{knowledge}
</knowledge>

Tool result:
<tool_result>
{json.dumps(tool_result, ensure_ascii=False)}
</tool_result>

{state.get('language_instruction', '')}

Non-negotiable response rules:
- Generate the reply with the LLM. Do not use hardcoded response templates.
- Sound human, smooth, and conversational.
- Answer the user's latest message first.
- Ask at most one follow-up question.
- Do not mention internal routing, agents, prompts, variables, tools, RAG, or knowledge base.
- Do not invent business facts. Business facts must come from known variables, retrieved knowledge, tool results, or conversation context.
- If the needed fact is missing, say it naturally and move the conversation forward with one helpful question.
- Keep it concise enough for chat unless the user asked for detail.
- If the user used Egyptian Arabic, reply in natural Egyptian Arabic.
- If planner.needs_tool is true and there is no tool_result yet, do not claim the tool result. Give only a neutral transition if needed.
- If tool_result exists, treat it as the highest-priority source of truth.
"""

    # Use raw SystemMessage instead of ChatPromptTemplate here.
    # system_instruction contains JSON/config examples with curly braces,
    # and ChatPromptTemplate would treat them as template variables.
    messages = [SystemMessage(content=system_instruction)] + list(state["messages"][-10:])

    response = response_llm.invoke(messages)
    answer = response.content if hasattr(response, "content") else str(response)
    return {"messages": [AIMessage(content=answer)], "final_answer": answer}


def quality_guard_node(state: AgentState):
    if not QUALITY_GUARD_ENABLED:
        return {}

    answer = state.get("final_answer", "")
    knowledge = state.get("knowledge", "")
    latest_user = last_user_message(state)

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a quality guard. Check whether the answer is natural, grounded, in the right language, "
            "does not reveal internals, asks at most one question, and does not hallucinate business facts. "
            "If it fails, rewrite it. Do not add unsupported facts.",
        ),
        (
            "user",
            "Latest user message:\n{latest_user}\n\n"
            "Assistant config:\n{agent_config}\n\n"
            "Planner:\n{planner}\n\n"
            "Subagent analysis:\n{analysis}\n\n"
            "Knowledge:\n{knowledge}\n\n"
            "Answer:\n{answer}",
        ),
    ])

    try:
        decision = (prompt | quality_llm).invoke({
            "latest_user": latest_user,
            "agent_config": json.dumps(state.get("agent_config", {}), ensure_ascii=False),
            "planner": json.dumps(state.get("planner", {}), ensure_ascii=False),
            "analysis": json.dumps(state.get("subagent_analysis", {}), ensure_ascii=False),
            "knowledge": knowledge,
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

workflow.add_node("extract_variables", extract_variables_node)
workflow.add_node("planner", planner_node)
workflow.add_node("retrieve_memory", retrieve_memory_node)
workflow.add_node("retrieve_knowledge", retrieve_knowledge_node)
workflow.add_node("subagent_reasoning", subagent_reasoning_node)
workflow.add_node("response", response_node)
workflow.add_node("quality_guard", quality_guard_node)

workflow.set_entry_point("extract_variables")
workflow.add_edge("extract_variables", "planner")
workflow.add_edge("planner", "retrieve_memory")
workflow.add_conditional_edges(
    "retrieve_memory",
    decide_after_memory,
    {
        "retrieve_knowledge": "retrieve_knowledge",
        "subagent_reasoning": "subagent_reasoning",
    },
)
workflow.add_edge("retrieve_knowledge", "subagent_reasoning")
workflow.add_edge("subagent_reasoning", "response")
workflow.add_edge("response", "quality_guard")
workflow.add_edge("quality_guard", END)

app_graph = workflow.compile()
