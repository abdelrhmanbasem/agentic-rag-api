import re
from dataclasses import dataclass, field
from typing import Any, Dict, List


def deep_get(data: Dict[str, Any], path: str, default: Any = None) -> Any:
    if not path:
        return default

    current: Any = data

    for part in str(path).split("."):
        if not isinstance(current, dict):
            return default

        current = current.get(part)

    return current if current is not None else default


def deep_set(data: Dict[str, Any], path: str, value: Any) -> Dict[str, Any]:
    if not path:
        return data

    target = data
    parts = str(path).split(".")

    for part in parts[:-1]:
        if part not in target or not isinstance(target[part], dict):
            target[part] = {}

        target = target[part]

    target[parts[-1]] = value
    return data


def deep_delete(data: Dict[str, Any], path: str) -> Dict[str, Any]:
    if not path:
        return data

    target = data
    parts = str(path).split(".")

    for part in parts[:-1]:
        if not isinstance(target, dict) or part not in target:
            return data

        target = target[part]

    if isinstance(target, dict):
        target.pop(parts[-1], None)

    return data


def deep_merge(base: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base or {})

    for key, value in (incoming or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        elif value not in [None, "", [], {}]:
            merged[key] = value

    return merged


def compact_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(data, dict):
        return {}

    return {
        key: value
        for key, value in data.items()
        if value not in [None, "", [], {}]
    }


def normalize_digits(text: str, digit_map: Dict[str, str]) -> str:
    return "".join(digit_map.get(ch, ch) for ch in str(text or ""))


def normalize_text(text: str, normalization_config: Dict[str, Any]) -> str:
    digit_map = normalization_config.get("digit_map", {})
    replacements = normalization_config.get("replacements", {})

    output = normalize_digits(str(text or ""), digit_map).strip().lower()

    if isinstance(replacements, dict):
        for old, new in replacements.items():
            output = output.replace(str(old), str(new))

    output = re.sub(r"\s+", " ", output)
    return output.strip()


def matches_any(message: str, phrases: List[str], normalization_config: Dict[str, Any]) -> bool:
    normalized = normalize_text(message, normalization_config)

    for phrase in phrases or []:
        normalized_phrase = normalize_text(str(phrase), normalization_config)

        if normalized_phrase and normalized_phrase in normalized:
            return True

    return False


def render_template(template: str, context: Dict[str, Any]) -> str:
    if not isinstance(template, str):
        return ""

    pattern = re.compile(r"{{\s*([^}]+)\s*}}")

    def replace(match: re.Match) -> str:
        path = match.group(1).strip()
        value = deep_get(context, path, "")
        return "" if value is None else str(value)

    return pattern.sub(replace, template)


def apply_variable_patch(
    variables: Dict[str, Any],
    updates: Dict[str, Any],
    clear: List[str]
) -> Dict[str, Any]:
    patched = dict(variables or {})

    if isinstance(clear, list):
        for path in clear:
            if isinstance(path, str) and path:
                deep_delete(patched, path)

    if isinstance(updates, dict):
        for path, value in updates.items():
            if value not in [None, "", [], {}]:
                deep_set(patched, path, value)

    return patched


def pick_variable_scope(variables: Dict[str, Any], include_paths: List[str]) -> Dict[str, Any]:
    if not include_paths:
        return dict(variables or {})

    scoped: Dict[str, Any] = {}

    for path in include_paths:
        if path == "*":
            return dict(variables or {})

        value = deep_get(variables, path)

        if value not in [None, "", [], {}]:
            deep_set(scoped, path, value)

    return scoped


def get_subagent_variable_scope(
    assistant_config: Dict[str, Any],
    subagent_name: str,
    variables: Dict[str, Any]
) -> Dict[str, Any]:
    subagent_config = (
        assistant_config
        .get("subagents", {})
        .get(subagent_name, {})
    )

    scope = subagent_config.get("variable_scope", {})

    if not isinstance(scope, dict):
        return dict(variables or {})

    include_paths = scope.get("include", [])

    if include_paths == "*" or include_paths == ["*"]:
        return dict(variables or {})

    if not isinstance(include_paths, list):
        return dict(variables or {})

    return pick_variable_scope(variables, include_paths)


def build_object_from_mapping(mapping: Dict[str, str], context: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}

    if not isinstance(mapping, dict):
        return result

    for target, source in mapping.items():
        value = deep_get(context, str(source), "")

        if value not in [None, "", [], {}]:
            deep_set(result, str(target), value)

    return result


def build_updates_from_mapping(mapping: Dict[str, str], context: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}

    if not isinstance(mapping, dict):
        return result

    for target, source in mapping.items():
        value = deep_get(context, str(source), "")

        if value not in [None, "", [], {}]:
            result[str(target)] = value

    return result


def extract_by_patterns(
    message: str,
    patterns: List[Dict[str, Any]],
    variables: Dict[str, Any],
    normalization_config: Dict[str, Any]
) -> Dict[str, Any]:
    updates: Dict[str, Any] = {}

    if not isinstance(patterns, list):
        return updates

    raw_message = normalize_digits(message, normalization_config.get("digit_map", {}))

    for item in patterns:
        if not isinstance(item, dict):
            continue

        variable = item.get("variable")
        regex = item.get("regex")
        group = int(item.get("group", 1))
        when_missing = item.get("when_missing")

        if when_missing:
            current_value = deep_get(variables, str(when_missing))

            if current_value not in [None, "", [], {}]:
                continue

        if not variable or not regex:
            continue

        try:
            match = re.search(regex, raw_message, flags=re.IGNORECASE)
        except re.error:
            continue

        if match:
            value = match.group(group).strip()

            if value:
                updates[str(variable)] = value
                deep_set(variables, str(variable), value)

    return updates


def get_missing_paths(paths: List[str], variables: Dict[str, Any]) -> List[str]:
    missing: List[str] = []

    for path in paths or []:
        value = deep_get({"variables": variables}, path)

        if value in [None, "", [], {}]:
            missing.append(path)

    return missing


def format_missing_fields(
    missing: List[str],
    labels: Dict[str, str]
) -> str:
    readable = []

    for path in missing:
        readable.append(str(labels.get(path, path)))

    return " و ".join(readable)


def apply_tool_update_rules(
    assistant_config: Dict[str, Any],
    variables: Dict[str, Any],
    operation: str,
    arguments: Dict[str, Any],
    result: Dict[str, Any]
) -> Dict[str, Any]:
    tool_rules = (
        assistant_config.get("tool_update_rules")
        or assistant_config.get("state_engine", {}).get("tool_update_rules")
        or {}
    )

    if not isinstance(tool_rules, dict):
        return variables

    rule = tool_rules.get(operation)

    if not isinstance(rule, dict):
        return variables

    context = {
        "variables": variables,
        "arguments": arguments or {},
        "result": result or {}
    }

    patched = dict(variables or {})

    clear = rule.get("clear", [])

    if isinstance(clear, list):
        for path in clear:
            if isinstance(path, str):
                deep_delete(patched, path)

    set_mapping = rule.get("set", {})

    if isinstance(set_mapping, dict):
        updates = build_updates_from_mapping(set_mapping, context)
        patched = apply_variable_patch(patched, updates, [])

    context["variables"] = patched

    conditional = rule.get("conditional", [])

    if isinstance(conditional, list):
        for item in conditional:
            if not isinstance(item, dict):
                continue

            when = item.get("when", {})

            if not isinstance(when, dict):
                continue

            path = str(when.get("path", ""))
            expected = when.get("equals")
            actual = deep_get(context, path)

            if actual == expected:
                c_clear = item.get("clear", [])

                if isinstance(c_clear, list):
                    for path_to_clear in c_clear:
                        if isinstance(path_to_clear, str):
                            deep_delete(patched, path_to_clear)

                c_set = item.get("set", {})

                if isinstance(c_set, dict):
                    updates = build_updates_from_mapping(c_set, context)
                    patched = apply_variable_patch(patched, updates, [])

                context["variables"] = patched

    return patched


@dataclass
class SubagentContext:
    assistant_config: Dict[str, Any]
    schema: Dict[str, Any]
    variables: Dict[str, Any]
    user_message: str
    history: List[Dict[str, str]]
    tool_runner: Any
    observations: List[Dict[str, Any]]
    max_tool_calls: int = 4


@dataclass
class SubagentResult:
    handled: bool = False
    action: str = "reply"
    answer: str = ""
    variable_updates: Dict[str, Any] = field(default_factory=dict)
    clear_variables: List[str] = field(default_factory=list)
    observations: List[Dict[str, Any]] = field(default_factory=list)
    selected_subagent: str = ""
    tool_calls_used: int = 0
    notes: str = ""
