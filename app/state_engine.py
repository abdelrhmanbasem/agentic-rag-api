import re
from typing import Any, Dict, List, Optional


def get_nested(data: Dict[str, Any], path: str, default: Any = None) -> Any:
    current: Any = data

    for part in path.split("."):
        if not isinstance(current, dict):
            return default
        current = current.get(part)

    return current if current is not None else default


def set_nested(data: Dict[str, Any], path: str, value: Any) -> Dict[str, Any]:
    target = data

    parts = path.split(".")
    for part in parts[:-1]:
        if part not in target or not isinstance(target[part], dict):
            target[part] = {}
        target = target[part]

    target[parts[-1]] = value
    return data


def delete_nested(data: Dict[str, Any], path: str) -> Dict[str, Any]:
    target = data
    parts = path.split(".")

    for part in parts[:-1]:
        if not isinstance(target, dict) or part not in target:
            return data
        target = target[part]

    if isinstance(target, dict):
        target.pop(parts[-1], None)

    return data


def compact_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(data, dict):
        return {}

    out: Dict[str, Any] = {}

    for key, value in data.items():
        if value is None:
            continue
        if value == "":
            continue
        if value == []:
            continue
        if value == {}:
            continue
        out[key] = value

    return out


def normalize_digits(text: str, digit_map: Dict[str, str]) -> str:
    return "".join(digit_map.get(ch, ch) for ch in str(text or ""))


def normalize_text(text: str, normalization_config: Dict[str, Any]) -> str:
    digit_map = normalization_config.get("digit_map", {})
    replacements = normalization_config.get("replacements", {})

    output = normalize_digits(text, digit_map).strip().lower()

    if isinstance(replacements, dict):
        for old, new in replacements.items():
            output = output.replace(str(old), str(new))

    output = re.sub(r"\s+", " ", output)
    return output.strip()


def message_matches_any(message: str, phrases: List[str], normalization_config: Dict[str, Any]) -> bool:
    normalized_message = normalize_text(message, normalization_config)

    for phrase in phrases:
        normalized_phrase = normalize_text(str(phrase), normalization_config)
        if normalized_phrase and normalized_phrase in normalized_message:
            return True

    return False


def get_state_config(assistant_config: Dict[str, Any]) -> Dict[str, Any]:
    config = assistant_config.get("state_engine", {})
    return config if isinstance(config, dict) else {}


def get_normalization_config(assistant_config: Dict[str, Any]) -> Dict[str, Any]:
    config = get_state_config(assistant_config).get("normalization", {})
    return config if isinstance(config, dict) else {}


def apply_variable_patch(
    variables: Dict[str, Any],
    variable_updates: Dict[str, Any],
    clear_variables: List[str]
) -> Dict[str, Any]:
    patched = dict(variables or {})

    if isinstance(clear_variables, list):
        for key in clear_variables:
            if isinstance(key, str) and key:
                delete_nested(patched, key)

    if isinstance(variable_updates, dict):
        for key, value in variable_updates.items():
            if value is None or value == "":
                continue
            if isinstance(key, str) and key:
                set_nested(patched, key, value)

    return patched


def render_template(template: str, context: Dict[str, Any]) -> str:
    if not isinstance(template, str):
        return ""

    result = template

    pattern = re.compile(r"{{\s*([^}]+)\s*}}")

    def replace(match: re.Match) -> str:
        path = match.group(1).strip()
        value = get_nested(context, path, "")
        return "" if value is None else str(value)

    result = pattern.sub(replace, result)
    return result


def extract_by_patterns(message: str, patterns: List[Dict[str, Any]], normalization_config: Dict[str, Any]) -> Dict[str, Any]:
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

    return updates


def normalize_time(value: Any, assistant_config: Dict[str, Any]) -> str:
    config = get_state_config(assistant_config).get("time_normalization", {})
    if not isinstance(config, dict):
        config = {}

    normalization_config = get_normalization_config(assistant_config)
    text = normalize_digits(str(value or ""), normalization_config.get("digit_map", {})).strip().lower()

    replacements = config.get("replacements", {})
    if isinstance(replacements, dict):
        for old, new in replacements.items():
            text = text.replace(str(old), str(new))

    regex = config.get("regex", r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?")

    try:
        match = re.search(regex, text, flags=re.IGNORECASE)
    except re.error:
        match = None

    if not match:
        return ""

    hour = int(match.group(1))
    minute = int(match.group(2) or "0")
    suffix = match.group(3) or ""

    if suffix == "pm" and hour < 12:
        hour += 12

    if suffix == "am" and hour == 12:
        hour = 0

    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        return ""

    return f"{hour:02d}:{minute:02d}"


def resolve_list_item_from_message(
    message: str,
    items: List[Any],
    assistant_config: Dict[str, Any],
    resolver_config: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    if not isinstance(items, list) or not items:
        return None

    normalization_config = get_normalization_config(assistant_config)
    normalized_message = normalize_text(message, normalization_config)

    ordinal_map = resolver_config.get("ordinal_map", {})
    if isinstance(ordinal_map, dict):
        for phrase, index in ordinal_map.items():
            normalized_phrase = normalize_text(str(phrase), normalization_config)
            if normalized_phrase and normalized_phrase in normalized_message:
                try:
                    i = int(index)
                except Exception:
                    continue

                if 0 <= i < len(items) and isinstance(items[i], dict):
                    return items[i]

    time_field = resolver_config.get("time_field")
    if time_field:
        requested_time = normalize_time(message, assistant_config)
        if requested_time:
            for item in items:
                if not isinstance(item, dict):
                    continue

                item_time = normalize_time(get_nested(item, str(time_field), ""), assistant_config)
                if item_time == requested_time:
                    return item

            requested_hour = requested_time.split(":")[0]
            for item in items:
                if not isinstance(item, dict):
                    continue

                item_time = normalize_time(get_nested(item, str(time_field), ""), assistant_config)
                if item_time and item_time.split(":")[0] == requested_hour:
                    return item

    return None


def build_context(
    variables: Dict[str, Any],
    user_message: str,
    tool_result: Optional[Dict[str, Any]] = None,
    arguments: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    return {
        "variables": variables or {},
        "message": user_message or "",
        "tool_result": tool_result or {},
        "arguments": arguments or {}
    }


def evaluate_condition(condition: Dict[str, Any], context: Dict[str, Any], assistant_config: Dict[str, Any]) -> bool:
    if not isinstance(condition, dict):
        return False

    kind = condition.get("type")

    if kind == "state_equals":
        path = condition.get("path", "")
        expected = condition.get("value")
        return get_nested(context, f"variables.{path}") == expected

    if kind == "state_exists":
        path = condition.get("path", "")
        value = get_nested(context, f"variables.{path}")
        return value not in [None, "", [], {}]

    if kind == "message_matches_any":
        phrases = condition.get("phrases", [])
        if not isinstance(phrases, list):
            phrases = []
        return message_matches_any(context.get("message", ""), phrases, get_normalization_config(assistant_config))

    if kind == "message_extracts_list_item":
        items_path = condition.get("items_path", "")
        resolver = condition.get("resolver", {})
        items = get_nested(context, f"variables.{items_path}", [])
        return resolve_list_item_from_message(context.get("message", ""), items, assistant_config, resolver) is not None

    return False


def all_conditions_match(conditions: List[Dict[str, Any]], context: Dict[str, Any], assistant_config: Dict[str, Any]) -> bool:
    if not isinstance(conditions, list):
        return False

    for condition in conditions:
        if not evaluate_condition(condition, context, assistant_config):
            return False

    return True


def apply_mapping(mapping: Dict[str, str], context: Dict[str, Any]) -> Dict[str, Any]:
    updates: Dict[str, Any] = {}

    if not isinstance(mapping, dict):
        return updates

    for target, source in mapping.items():
        value = get_nested(context, str(source), "")
        if value not in [None, "", [], {}]:
            updates[str(target)] = value

    return updates


def build_object_from_mapping(mapping: Dict[str, str], context: Dict[str, Any]) -> Dict[str, Any]:
    obj: Dict[str, Any] = {}

    if not isinstance(mapping, dict):
        return obj

    for target, source in mapping.items():
        value = get_nested(context, str(source), "")
        if value not in [None, "", [], {}]:
            set_nested(obj, str(target), value)

    return obj


def get_missing_required_fields(
    required_fields: List[str],
    variables: Dict[str, Any],
    extra_context: Dict[str, Any]
) -> List[str]:
    missing: List[str] = []

    combined = {
        "variables": variables or {},
        **(extra_context or {})
    }

    if not isinstance(required_fields, list):
        return missing

    for field in required_fields:
        if not isinstance(field, str):
            continue

        value = get_nested(combined, field)

        if value in [None, "", [], {}]:
            missing.append(field)

    return missing


def run_configured_state_rules(
    assistant_config: Dict[str, Any],
    variables: Dict[str, Any],
    user_message: str
) -> Optional[Dict[str, Any]]:
    state_config = get_state_config(assistant_config)
    rules = state_config.get("pre_llm_rules", [])

    if not isinstance(rules, list):
        return None

    context = build_context(variables, user_message)

    extraction_patterns = state_config.get("extraction_patterns", [])
    extracted = extract_by_patterns(user_message, extraction_patterns, get_normalization_config(assistant_config))

    if extracted:
        variables = apply_variable_patch(variables, extracted, [])
        context = build_context(variables, user_message)

    for rule in rules:
        if not isinstance(rule, dict):
            continue

        conditions = rule.get("when", [])
        if not all_conditions_match(conditions, context, assistant_config):
            continue

        variable_updates = dict(rule.get("variable_updates", {}) or {})
        clear_variables = list(rule.get("clear_variables", []) or [])

        list_item_config = rule.get("resolve_list_item")
        resolved_item = None

        if isinstance(list_item_config, dict):
            items_path = list_item_config.get("items_path", "")
            item_variable = list_item_config.get("save_as", "")
            resolver = list_item_config.get("resolver", {})

            items = get_nested(variables, str(items_path), [])
            resolved_item = resolve_list_item_from_message(user_message, items, assistant_config, resolver)

            if resolved_item and item_variable:
                variable_updates[str(item_variable)] = resolved_item

                item_mapping = list_item_config.get("map_to_variables", {})
                context_with_item = {
                    **context,
                    "resolved_item": resolved_item
                }
                variable_updates.update(apply_mapping(item_mapping, context_with_item))

        object_mapping = rule.get("build_object")
        if isinstance(object_mapping, dict):
            target = object_mapping.get("target")
            mapping = object_mapping.get("mapping", {})
            if target:
                variable_updates[str(target)] = build_object_from_mapping(mapping, {
                    **context,
                    "resolved_item": resolved_item or {}
                })

        required_fields = rule.get("required_fields", [])
        missing_fields = get_missing_required_fields(required_fields, apply_variable_patch(variables, variable_updates, []), {
            "resolved_item": resolved_item or {}
        })

        if missing_fields:
            missing_labels = []
            labels = state_config.get("field_labels", {})
            if not isinstance(labels, dict):
                labels = {}

            for field in missing_fields:
                missing_labels.append(str(labels.get(field, field)))

            template = rule.get("missing_template") or state_config.get("default_missing_template") or "{{missing_fields}}"
            answer = render_template(template, {
                "missing_fields": " و ".join(missing_labels),
                "variables": apply_variable_patch(variables, variable_updates, [])
            })

            return {
                "action": "ask_user",
                "answer": answer,
                "variable_updates": variable_updates,
                "clear_variables": clear_variables,
                "deterministic": True,
                "rule_id": rule.get("id", "")
            }

        action = rule.get("action", "ask_user")

        if action == "call_tool":
            tool_name = rule.get("tool_name") or state_config.get("default_tool_name", "")
            operation = rule.get("operation") or ""
            arguments_mapping = rule.get("arguments", {})

            patched_variables = apply_variable_patch(variables, variable_updates, clear_variables)
            args = build_object_from_mapping(arguments_mapping, {
                "variables": patched_variables,
                "resolved_item": resolved_item or {},
                "message": user_message
            })

            return {
                "action": "call_tool",
                "tool_name": tool_name,
                "operation": operation,
                "arguments": compact_dict(args),
                "variable_updates": variable_updates,
                "clear_variables": clear_variables,
                "deterministic": True,
                "rule_id": rule.get("id", "")
            }

        template = rule.get("answer_template", "")
        answer = render_template(template, {
            "variables": apply_variable_patch(variables, variable_updates, clear_variables),
            "resolved_item": resolved_item or {},
            "message": user_message
        })

        return {
            "action": action,
            "answer": answer,
            "variable_updates": variable_updates,
            "clear_variables": clear_variables,
            "deterministic": True,
            "rule_id": rule.get("id", "")
        }

    if extracted:
        return {
            "action": "continue",
            "variable_updates": extracted,
            "clear_variables": [],
            "deterministic": True,
            "rule_id": "extraction_only"
        }

    return None


def apply_tool_update_rules(
    assistant_config: Dict[str, Any],
    variables: Dict[str, Any],
    tool_name: str,
    operation: str,
    arguments: Dict[str, Any],
    result: Dict[str, Any]
) -> Dict[str, Any]:
    variables = dict(variables or {})

    state_config = get_state_config(assistant_config)
    tool_rules = state_config.get("tool_update_rules", {})

    if not isinstance(tool_rules, dict):
        return variables

    rule = tool_rules.get(operation)
    if not isinstance(rule, dict):
        return variables

    context = {
        "variables": variables,
        "tool_name": tool_name,
        "operation": operation,
        "arguments": arguments or {},
        "result": result or {}
    }

    clear_variables = rule.get("clear", [])
    if isinstance(clear_variables, list):
        for item in clear_variables:
            if isinstance(item, str):
                delete_nested(variables, item)

    set_mapping = rule.get("set", {})
    if isinstance(set_mapping, dict):
        updates = apply_mapping(set_mapping, context)
        variables = apply_variable_patch(variables, updates, [])

    conditional_rules = rule.get("conditional", [])
    if isinstance(conditional_rules, list):
        for conditional in conditional_rules:
            if not isinstance(conditional, dict):
                continue

            when = conditional.get("when", {})
            if not isinstance(when, dict):
                continue

            source = when.get("path", "")
            expected = when.get("equals")
            actual = get_nested(context, str(source))

            if actual == expected:
                conditional_clear = conditional.get("clear", [])
                if isinstance(conditional_clear, list):
                    for item in conditional_clear:
                        if isinstance(item, str):
                            delete_nested(variables, item)

                conditional_set = conditional.get("set", {})
                if isinstance(conditional_set, dict):
                    updates = apply_mapping(conditional_set, context)
                    variables = apply_variable_patch(variables, updates, [])

    return variables
