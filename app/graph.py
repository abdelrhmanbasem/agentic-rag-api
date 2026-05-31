# app/graph.py
from typing import TypedDict, Annotated, Sequence, Dict, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
import operator

# Import your existing utilities (adjust imports as needed based on your file structure)
from app.rag import search_knowledge, compress_knowledge
from app.config import KNOWLEDGE_TOP_K
from app.variables import extract_variables

# 1. Define the State
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    assistant_id: str
    user_id: str
    conversation_id: str
    variables: Dict[str, Any]
    summary: str
    knowledge: str
    next_step: str
    system_prompt: str
    tone: str
    language_instruction: str

# 2. Define the Cheap Router (Token Saver)
class RouteDecision(BaseModel):
    step: str = Field(description="Must be one of: 'rag_agent', 'booking_agent', 'chat_agent'")

router_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(RouteDecision)

def router_node(state: AgentState):
    """Analyzes the latest message and routes it cheaply."""
    last_message = state["messages"][-1].content
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a smart router. If the user asks about car inventory, prices, or technical details, route to 'rag_agent'. "
                   "If they want to book a viewing or appointment, route to 'booking_agent'. "
                   "Otherwise, route to 'chat_agent'."),
        ("user", "{message}")
    ])
    
    decision = (prompt | router_llm).invoke({"message": last_message})
    return {"next_step": decision.step}

# 3. Define the RAG Node
def rag_node(state: AgentState):
    """Fetches knowledge based on the user's query."""
    last_message = state["messages"][-1].content
    
    # Using your existing RAG logic!
    raw_knowledge = search_knowledge(state["assistant_id"], last_message, limit=KNOWLEDGE_TOP_K)
    compressed = compress_knowledge(raw_knowledge, last_message)
    
    # Format knowledge into a string for the generator
    knowledge_text = "\n".join([item.get("text", "") for item in compressed]) if compressed else "No specific inventory found."
    
    return {"knowledge": knowledge_text, "next_step": "generator"}

# 4. Define the Human-Like Generator Node
generator_llm = ChatOpenAI(model="gpt-4o", temperature=0.6) # Temp 0.6 for smooth, natural conversation

def generator_node(state: AgentState):
    """The brain that crafts the final human-like response."""
    
    system_prompt = f"""
    {state['system_prompt']}
    
    Tone: {state['tone']}
    
    {state['language_instruction']}
    
    Context Overview:
    - Conversation Summary: {state['summary']}
    - Known Variables: {state['variables']}
    
    Knowledge Base Retrieval:
    <knowledge>
    {state.get('knowledge', 'No additional knowledge retrieved.')}
    </knowledge>
    
    Instructions:
    1. Answer the user smoothly and naturally. Be conversational, not robotic.
    2. Use the Knowledge Base if facts/prices are needed. Do not hallucinate cars or prices.
    3. If variables are missing for their goal (like budget or date), naturally ask them in conversation.
    """
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        # We pass the last few messages to maintain human-like flow
        ("placeholder", "{messages}") 
    ])
    
    chain = prompt | generator_llm
    response = chain.invoke({"messages": state["messages"][-5:]}) # Only pass last 5 to save tokens
    
    return {"messages": [response]}

# 5. Build the Graph
from langgraph.graph import StateGraph, END

workflow = StateGraph(AgentState)

workflow.add_node("router", router_node)
workflow.add_node("rag_agent", rag_node)
workflow.add_node("generator", generator_node)
# Note: You can add your booking_node here later wrapping run_booking_subagent

workflow.set_entry_point("router")

# Add conditional edges from router
workflow.add_conditional_edges(
    "router",
    lambda x: x["next_step"],
    {
        "rag_agent": "rag_agent",
        "chat_agent": "generator",
        "booking_agent": "generator" # Routing to generator for now until we wrap the booking agent
    }
)

workflow.add_edge("rag_agent", "generator")
workflow.add_edge("generator", END)

app_graph = workflow.compile()
