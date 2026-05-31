import json
from openai import OpenAI
from app.config import MOCK_MODE, OPENAI_API_KEY, MODEL_MEMORY


_client = None


def get_client():
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


def chat_text(model, messages, max_tokens=600, temperature=0.4):
    if MOCK_MODE:
        return "Mock response: I understood and responded using the configured assistant prompt."
    client = get_client()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
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
