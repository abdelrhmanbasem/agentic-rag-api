# app/variables.py
# Stripped down for LangGraph architecture. 
# Extraction logic is now handled natively via Pydantic in graph.py.

def apply_variable_patch(existing, updates, deletions):
    """Safely merges new LLM variable updates and handles deletions."""
    result = dict(existing or {})

    for key in deletions or []:
        if key in result:
            del result[key]

    for key, value in (updates or {}).items():
        if value is not None:
            result[key] = value

    return result
