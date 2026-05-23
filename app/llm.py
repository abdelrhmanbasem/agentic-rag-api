import json
from openai import OpenAI
from app.config import (
    MOCK_MODE,
    OPENAI_API_KEY,
    MODEL_ROUTER,
    MODEL_NORMAL,
    MODEL_STRONG,
    MODEL_EXTRACTION,
    MODEL_MEMORY,
)

_client = None


def get_client():
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


def chat_text(model, messages, max_tokens=600):
    if MOCK_MODE:
        return "Mock response: I understood your request and updated the conversation variables."

    client = get_client()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.4,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def chat_json(model, messages, max_tokens=500):
    if MOCK_MODE:
        return {}

    client = get_client()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )

    return json.loads(response.choices[0].message.content)


def memory_model():
    return MODEL_MEMORY


def extraction_model():
    return MODEL_EXTRACTION


def mock_route_message(user_message, variables):
    text = user_message.lower().strip()

    route = {
        "intent_hint": variables.get("intent", "general_question") if variables else "general_question",
        "needs_rag": False,
        "needs_memory": True,
        "needs_variable_extraction": True,
        "selected_model_tier": "normal",
        "answer_mode": "generate",
        "risk_score": 0.1,
        "complexity_score": 0.3,
        "reason": "Mock router decision.",
    }

    if len(text) <= 3 or text in ["hi", "hello", "hey", "thanks", "thank you", "ok", "okay", "yes", "no"]:
        route.update({
            "needs_rag": False,
            "needs_memory": False,
            "needs_variable_extraction": False,
            "selected_model_tier": "cheap",
            "answer_mode": "no_llm",
            "risk_score": 0.0,
            "complexity_score": 0.05,
            "reason": "Very simple short message.",
        })
        return route

    rag_keywords = [
        "do you have",
        "available",
        "availability",
        "price",
        "cost",
        "how much",
        "policy",
        "service",
        "doctor",
        "branch",
        "opening hours",
        "insurance",
        "warranty",
        "refund",
        "inventory",
        "bmw",
        "mercedes",
        "clinic",
        "appointment",
        "booking",
    ]

    if any(keyword in text for keyword in rag_keywords):
        route["needs_rag"] = True
        route["reason"] = "Business or knowledge-base information may be needed."

    risk_keywords = [
        "angry",
        "complaint",
        "unacceptable",
        "lawsuit",
        "emergency",
        "urgent",
        "severe pain",
        "bleeding",
        "chest pain",
        "report you",
        "cancel",
    ]

    if any(keyword in text for keyword in risk_keywords):
        route.update({
            "selected_model_tier": "strong",
            "answer_mode": "generate",
            "risk_score": 0.8,
            "complexity_score": 0.7,
            "needs_rag": True,
            "needs_memory": True,
            "needs_variable_extraction": True,
            "reason": "Risk, urgency, complaint, or escalation keywords detected.",
        })

    return route


def gpt_route_message(assistant, summary, variables, recent_messages, user_message):
    prompt = f"""
You are the router for a low-cost Agentic RAG system.

Choose which steps are needed and which model tier should answer.

Assistant:
{assistant}

Summary:
{summary}

Current variables:
{variables}

Recent messages:
{recent_messages[-4:]}

Latest user message:
{user_message}

Rules:
- Skip RAG for greetings, thanks, yes/no, phone number only, or simple corrections.
- Use RAG when business/client knowledge is needed.
- Use memory when prior preferences may matter.
- Use cheap for simple messages.
- Use normal for most business conversations.
- Use strong only for risk, complaints, urgency, complex reasoning, or low confidence.
- Use answer_mode "no_llm" only when a deterministic short answer is enough.
- Use answer_mode "generate" when a natural assistant reply is needed.

Return JSON only:
{{
  "intent_hint": "string",
  "needs_rag": true,
  "needs_memory": true,
  "needs_variable_extraction": true,
  "selected_model_tier": "cheap|normal|strong",
  "answer_mode": "no_llm|generate",
  "risk_score": 0.0,
  "complexity_score": 0.0,
  "reason": "short reason"
}}
"""

    result = chat_json(
        MODEL_ROUTER,
        [{"role": "user", "content": prompt}],
        max_tokens=400,
    )

    return {
        "intent_hint": result.get("intent_hint", "general_question"),
        "needs_rag": bool(result.get("needs_rag", True)),
        "needs_memory": bool(result.get("needs_memory", True)),
        "needs_variable_extraction": bool(result.get("needs_variable_extraction", True)),
        "selected_model_tier": result.get("selected_model_tier", "normal"),
        "answer_mode": result.get("answer_mode", "generate"),
        "risk_score": float(result.get("risk_score", 0.2)),
        "complexity_score": float(result.get("complexity_score", 0.4)),
        "reason": result.get("reason", ""),
    }


def route_message(assistant, summary, variables, recent_messages, user_message):
    if MOCK_MODE:
        return mock_route_message(user_message, variables)

    return gpt_route_message(assistant, summary, variables, recent_messages, user_message)


def model_for_tier(tier):
    if tier == "cheap":
        return MODEL_ROUTER

    if tier == "strong":
        return MODEL_STRONG

    return MODEL_NORMAL
