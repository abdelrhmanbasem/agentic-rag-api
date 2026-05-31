# app/graph.py
from typing import TypedDict, Annotated, Sequence, Dict, Any, List
import json

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import operator

# Core utility imports directly from your project architecture
from app.config import KNOWLEDGE_TOP_K, MODEL_ROUTER, MODEL_NORMAL, MODEL_STRONG, OPENAI_API_KEY
from app.rag import search_knowledge, compress_knowledge
from app.variables import apply_variable_patch

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

# Run router on cheap/fast model as requested
router_llm = ChatOpenAI(
    model=MODEL_ROUTER, 
    temperature=0, 
    api_key=OPENAI_API_KEY
).with_structured_output(RouteDecision)

def adaptive_router_node(state: AgentState):
    """
    Replaces intelligence_modes.py by using an LLM to evaluate 
    if RAG knowledge is needed based on real semantic context.
    """
    last_message = state["messages"][-1].content
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are the advanced pre-router for a premium business virtual assistant. "
                   "Analyze the user's latest input and determine if it requires specific factual records, "
                   "inventory details, real estate listings, pricing data, or technical information from our Knowledge Base.\n\n"
                   "- Choose 'rag_agent' if they ask about items, details, specific features, specs, costs, or catalog lookups.\n"
                   "- Choose 'chat_agent' if they are giving info, greeting you, confirming an appointment, or making casual chat.\n"),
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
    """Queries your Qdrant instance smoothly using existing RAG compress routines."""
    last_message = state["messages"][-1].content
    
    # Utilizing your existing robust search + semantic compression pipelines
    raw_knowledge = search_knowledge(state["assistant_id"], last_message, limit=KNOWLEDGE_TOP_K)
    compressed = compress_knowledge(raw_knowledge, last_message)
    
    knowledge_text = ""
    if compressed:
        knowledge_text = "\n".join([f"- {item.get('text', '')}" for item in compressed])
    else:
        knowledge_text = "No matching items found in the current inventory vector space."
        
    return {"knowledge": knowledge_text, "next_step": "chat_agent"}

# ==========================================
# 4. VARIABLE EXTRACTION & CHAT BRAIN NODE
# ==========================================
class VariableExtraction(BaseModel):
    updates: Dict[str, Any] = Field(default_factory=dict, description="Key-value pairs extracted from user message.")
    deletions: List[str] = Field(default_factory=list, description="List of variable keys to drop.")

extractor_llm = ChatOpenAI(
    model=MODEL_NORMAL, 
    temperature=0, 
    api_key=OPENAI_API_KEY
).with_structured_output(VariableExtraction)

# The core brain uses your high-tier model for human conversational beauty
generator_llm = ChatOpenAI(
    model=MODEL_STRONG, 
    temperature=0.55, # Tuned for highly natural, non-robotic flow
    api_key=OPENAI_API_KEY
)

def ultimate_brain_node(state: AgentState):
    """
    Extracts slots/variables inline and builds highly tailored, 
    human-like contextual responses in the correct language.
    """
    last_message = state["messages"][-1].content
    
    # --- Part A: Variable Extraction (Token Saving Inline step) ---
    extraction_prompt = ChatPromptTemplate.from_messages([
        ("system", "Extract any updated values or slots relevant to this business scenario from the user's latest text. "
                   "Update values if the user clarifies details like budget, preferences, date, or contact details."),
        ("user", "Current Variables: {variables}\nUser message: {message}")
    ])
    try:
        extracted = (extraction_prompt | extractor_llm).invoke({
            "variables": json.dumps(state["variables"]),
            "message": last_message
        })
        updated_vars = apply_variable_patch(state["variables"], extracted.updates, extracted.deletions)
    except Exception:
        updated_vars = state["variables"] # Safe fallback

    # --- Part B: Fluid Generation ---
    system_instruction = f"""
{state['system_prompt']}

Tone & Personality Style: {state['tone']}

{state['language_instruction']}

Operational Context:
- Summary of events so far: {state['summary']}
- Tracked Metadata Profiles: {updated_vars}

Retrieved System Knowledge base matches:
<knowledge>
{state.get('knowledge', 'No external records pulled for this conversation turn.')}
</knowledge>

CRITICAL EXECUTION POLICIES:
1. Speak beautifully, empathetically, and natively. If replying in Egyptian Arabic, sound fully colloquial (عامية مصرية), professional, yet lively.
2. Use retrieved factual database elements natively. Do not quote strings mechanically—weave them into a smooth sentence structure.
3. If crucial context data is missing from the Profile to take action, gracefully guide the user to share it as part of an organic conversation flow.
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_instruction),
        ("placeholder", "{messages}")
    ])
    
    # Pass historical window context (Last 6 interactions) to retain conversational fluidity 
    chat_chain = prompt | generator_llm
    response = chat_chain.invoke({"messages": state["messages"][-6:]})
    
    return {"messages": [response], "variables": updated_vars}

# ==========================================
# 5. GRAPH ENGINE COMPILED ARCHITECTURE
# ==========================================
from langgraph.graph import StateGraph, END

workflow = StateGraph(AgentState)

workflow.add_node("router", adaptive_router_node)
workflow.add_node("rag_agent", dynamic_rag_node)
workflow.add_node("chat_agent", ultimate_brain_node)

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
