from typing import Any, Dict, List


def apply_variable_patch(existing: Dict[str, Any], updates: Dict[str, Any], deletions: List[str]):
    result = dict(existing or {})

    for key in deletions or []:
        if key in result:
            del result[key]

    for key, value in (updates or {}).items():
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        result[key] = value

    return result
