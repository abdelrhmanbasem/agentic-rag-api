# app/graph.py
from typing import TypedDict, Annotated, Sequence, Dict, Any, List
import json
import operator

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END

# --- Core Utility & Configuration Imports ---
from app.config import KNOWLEDGE_TOP_K, MODEL_ROUTER, MODEL_NORMAL, MODEL_STRONG, OPENAI_API_KEY
from app.rag import search_knowledge, compress_knowledge
from app.variables import apply_variable_patch

# --- Intelligence & Strategy Imports ---
from app.domain_playbooks import build_playbook_prompt
from app.datetime_location_extractor import extract_datetime_location_patch
from app.booking_subagent import extract_user_area, recommend_nearest_branches_for_area

# ==========================================
# 1. DEFINE STATE ARCHITECTURE
# ==========================================
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    assistant_id: str
    user_id: str
    conversation_id: str
    variables: Dict[str, Any]
    summary: str
    knowledge: str
    system_prompt: str
    tone: str
    language_instruction: str
    next_step: str

# ==========================================
# 2. THE TOKEN-SAVING ADAPTIVE ROUTER
# ==========================================
class RouteDecision(BaseModel):
    step: str = Field(description="Must be one of: 'rag_agent', 'chat_agent'")
    reason: str = Field(description="Brief logic explaining why this path was chosen.")

# Router runs on your cheap/fast model tier
router_llm = ChatOpenAI(
    model=MODEL_ROUTER, 
    temperature=0, 
    api_key=OPENAI_API_KEY
).with_structured_output(RouteDecision)

def adaptive_router_node(state: AgentState):
    """
    Evaluates the semantic context of the conversation to determine if
    external factual data or catalog matching is necessary.
    """
    last_message = state["messages"][-1].content
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are the advanced pre-router for a premium business virtual assistant. "
                   "Analyze the user's latest input and determine if it requires specific factual records, "
                   "inventory details, property listings, pricing data, or technical metrics from our Knowledge Base.\n\n"
                   "- Choose 'rag_agent' if they ask about items, specs, features, specific costs, or database listings.\n"
                   "- Choose 'chat_agent' if they are providing details, greeting you, confirming an action, or making casual chat.\n"),
        ("user", "Conversation Context Summary: {summary}\nKnown Details: {variables}\nUser Message: {message}")
    ])
    
    decision = (prompt | router_llm).invoke({
        "summary": state.get("summary", ""),
        "variables": json.dumps(state.get("variables", {})),
        "message": last_message
    })
    
    return {"next_step": decision.step}

# ==========================================
# 3. KNOWLEDGE DISCOVERY (RAG) NODE
# ==========================================
def dynamic_rag_node(state: AgentState):
    """Queries your Qdrant database using your native semantic compression routines."""
    last_message = state["messages"][-1].content
    
    # Run structural knowledge lookup
    raw_knowledge = search_knowledge(state["assistant_id"], last_message, limit=KNOWLEDGE_TOP_K)
    compressed = compress_knowledge(raw_knowledge, last_message)
    
    if compressed:
        knowledge_text = "\n".join([f"- {item.get('text', '')}" for item in compressed])
    else:
        knowledge_text = "No matching items found in the current inventory vector space."
        
    return {"knowledge": knowledge_text, "next_step": "chat_agent"}

# ==========================================
# 4. DATA EXTRACTION & MAIN GENERATOR NODE
# ==========================================
class VariableExtraction(BaseModel):
    updates: Dict[str, Any] = Field(default_factory=dict, description="Key-value profile elements updated from the text.")
    deletions: List[str] = Field(default_factory=list, description="Profile fields to drop if user changed their mind.")

extractor_llm = ChatOpenAI(
    model=MODEL_NORMAL, 
    temperature=0, 
    api_key=OPENAI_API_KEY
).with_structured_output(VariableExtraction)

# Main chat brain utilizes your top-tier conversational model
generator_llm = ChatOpenAI(
    model=MODEL_STRONG, 
    temperature=0.55, # Optimized balance for human warmth without hallucinating facts
    api_key=OPENAI_API_KEY
)

def ultimate_brain_node(state: AgentState):
    """
    Combines zero-token deterministic extraction, geographical math, secondary LLM extraction,
    and business playbook enforcement to create a highly natural, genius-level conversation.
    """
    last_message = state["messages"][-1].content
    current_variables = state.get("variables", {}) or {}
    
    workflow_type = current_variables.get("workflow", "general")
    
    # --- Step A: Zero-Token Stealth Date/Time Extraction ---
    stealth_updates = extract_datetime_location_patch(last_message, current_variables, workflow_type)
    if stealth_updates:
        current_variables.update(stealth_updates)

    # --- Step B: Zero-Token Geographical Intelligence ---
    # Automatically calculate the nearest branch if they mention a neighborhood!
    user_area = extract_user_area(last_message)
    if user_area:
        current_variables["user_area"] = user_area
        if not current_variables.get("location_branch"):
            nearest_branches = recommend_nearest_branches_for_area(user_area, limit=1)
            if nearest_branches:
                current_variables["location_branch"] = nearest_branches[0]["branch"]
                current_variables["_nearest_branch_ar_hint"] = nearest_branches[0]["branch_ar"]

    # --- Step C: Secondary LLM Extraction (For arbitrary slot filling) ---
    extraction_prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract any profile context or variables relevant to this business interaction from the user's latest text. "
                   "Update tracking metrics when users state clear preferences, item matches, constraints, or updates."),
        ("user", "Current Profile State: {variables}\nUser message: {message}")
    ])
    try:
        llm_extracted = (extraction_prompt | extractor_llm).invoke({
            "variables": json.dumps(current_variables),
            "message": last_message
        })
        updated_vars = apply_variable_patch(current_variables, llm_extracted.updates, llm_extracted.deletions)
    except Exception:
        updated_vars = current_variables # Safe fallback

    # --- Step D: Build Playbook Strategy Prompt ---
    tone_profile = state.get("tone", "helpful_operator")
    playbook_strategy = build_playbook_prompt(workflow_type, tone_profile)

    # --- Step E: Conversational Synthesis ---
    system_instruction = f"""
{state.get('system_prompt', '')}

{playbook_strategy}

{state.get('language_instruction', '')}

Operational Real-Time Context:
- Summary of events so far: {state.get('summary', '')}
- Active Profile Metadata: {updated_vars}

Retrieved Knowledge Base Entries:
<knowledge>
{state.get('knowledge', 'No external records pulled for this conversation turn.')}
</knowledge>

CRITICAL EXECUTION POLICIES:
1. Speak beautifully, empathetically, and naturally. If replying in Egyptian Arabic, sound fully colloquial (عامية مصرية), warm, professional, and authentic—never sound like a rigid textbook translation.
2. If '_nearest_branch_ar_hint' is in the metadata, naturally suggest this branch to the user since we mathematically determined it is closest to them.
3. Use retrieved factual database properties seamlessly. Weave prices, car names, or doctor names into organic sentences instead of repeating text chunks mechanically.
4. If essential context details are missing from the Profile Metadata to complete the playbook's Primary CTA, gently guide the user to share them one specific thing at a time as part of an elegant conversation loop.
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_instruction),
        ("placeholder", "{messages}")
    ])
    
    # Pass historical window context (Last 6 messages) to optimize attention and cost
    chat_chain = prompt | generator_llm
    
    # Strip internal hints before saving back to the DB to keep state clean
    vars_to_save = dict(updated_vars)
    vars_to_save.pop("_nearest_branch_ar_hint", None)
    
    response = chat_chain.invoke({"messages": state["messages"][-6:]})
    
    return {"messages": [response], "variables": vars_to_save}

# ==========================================
# 5. COMPILE AND EXPOSE THE WORKFLOW GRAPH
# ==========================================
workflow = StateGraph(AgentState)

# Add Node Processing Units
workflow.add_node("router", adaptive_router_node)
workflow.add_node("rag_agent", dynamic_rag_node)
workflow.add_node("chat_agent", ultimate_brain_node)

# Flow Settings
workflow.set_entry_point("router")

workflow.add_conditional_edges(
    "router",
    lambda state: state["next_step"],
    {
        "rag_agent": "rag_agent",
        "chat_agent": "chat_agent"
    }
)

workflow.add_edge("rag_agent", "chat_agent")
workflow.add_edge("chat_agent", END)

app_graph = workflow.compile()
