import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# Domain-specific source-of-truth paths must live in domain_bundle.json.
# These empty defaults are kept only for backward compatibility with imports.
SOURCE_OF_TRUTH_EXACT_VARIABLES = set()
SOURCE_OF_TRUTH_PREFIXES = tuple()


def as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def unique_list(values: List[Any]) -> List[Any]:
    output: List[Any] = []

    for value in values or []:
        if value not in output:
            output.append(value)

    return output


def get_config_path_value(config: Dict[str, Any], path: str, default: Any = None) -> Any:
    if not isinstance(config, dict):
        return default

    current: Any = config

    for part in str(path or "").split("."):
        if not part:
            continue

        if not isinstance(current, dict):
            return default

        if part not in current:
            return default

        current = current.get(part)

    return current if current is not None else default


def collect_source_of_truth_exact_variables(assistant_config: Optional[Dict[str, Any]] = None) -> List[str]:
    assistant_config = assistant_config if isinstance(assistant_config, dict) else {}
    values: List[str] = []

    sources = [
        assistant_config.get("source_of_truth_variables"),
        get_config_path_value(assistant_config, "source_of_truth.exact_variables"),
        get_config_path_value(assistant_config, "source_of_truth.variables"),
        get_config_path_value(assistant_config, "state_engine.source_of_truth_variables"),
        get_config_path_value(assistant_config, "state_engine.source_of_truth.exact_variables"),
        list(SOURCE_OF_TRUTH_EXACT_VARIABLES),
    ]

    for source in sources:
        if isinstance(source, list):
            values.extend([str(item) for item in source if str(item or "").strip()])

    schema_vars = get_config_path_value(assistant_config, "schema.variables", {})
    if isinstance(schema_vars, dict):
        for key, cfg in schema_vars.items():
            if isinstance(cfg, dict) and (cfg.get("source_of_truth") is True or cfg.get("operational_state") is True):
                values.append(str(key))

    return unique_list(values)


def collect_source_of_truth_prefixes(assistant_config: Optional[Dict[str, Any]] = None) -> List[str]:
    assistant_config = assistant_config if isinstance(assistant_config, dict) else {}
    values: List[str] = []

    sources = [
        assistant_config.get("source_of_truth_prefixes"),
        get_config_path_value(assistant_config, "source_of_truth.prefixes"),
        get_config_path_value(assistant_config, "state_engine.source_of_truth_prefixes"),
        get_config_path_value(assistant_config, "state_engine.source_of_truth.prefixes"),
        list(SOURCE_OF_TRUTH_PREFIXES),
    ]

    for source in sources:
        if isinstance(source, list):
            values.extend([str(item) for item in source if str(item or "").strip()])

    return unique_list(values)


def path_matches_policy_list(path: str, patterns: Optional[List[str]]) -> bool:
    if not isinstance(patterns, list) or not patterns:
        return False

    path_text = str(path or "").strip()

    for item in patterns:
        pattern = str(item or "").strip()

        if not pattern:
            continue

        if pattern == path_text:
            return True

        if pattern.endswith(".*"):
            prefix = pattern[:-2]
            if path_text == prefix or path_text.startswith(prefix + "."):
                return True

        if pattern.endswith(".") and path_text.startswith(pattern):
            return True

    return False



def is_empty(value: Any) -> bool:
    return value in [None, "", [], {}]


def deep_get(data: Dict[str, Any], path: str, default: Any = None) -> Any:
    if not path:
        return default

    current: Any = data

    for part in str(path).split("."):
        if not isinstance(current, dict):
            return default

        if part not in current:
            return default

        current = current.get(part)

    return current if current is not None else default


def deep_set(data: Dict[str, Any], path: str, value: Any) -> Dict[str, Any]:
    if not path:
        return data

    target = data
    parts = str(path).split(".")

    for part in parts[:-1]:
        if part not in target or not isinstance(target.get(part), dict):
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

    if not isinstance(incoming, dict):
        return merged

    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        elif not is_empty(value):
            merged[key] = value

    return merged


def compact_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(data, dict):
        return {}

    return {
        key: value
        for key, value in data.items()
        if not is_empty(value)
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
    """
    Rendering is used for internal draft labels / fallback hints only.
    Final customer-facing wording belongs to graph.response_node.
    """
    if not isinstance(template, str):
        return ""

    pattern = re.compile(r"{{\s*([^}]+)\s*}}")

    def replace(match: re.Match) -> str:
        path = match.group(1).strip()
        value = deep_get(context, path, "")
        return "" if value is None else str(value)

    return pattern.sub(replace, template)


def path_is_source_of_truth(path: str, assistant_config: Optional[Dict[str, Any]] = None) -> bool:
    path = str(path or "").strip()

    if not path:
        return False

    if path in collect_source_of_truth_exact_variables(assistant_config):
        return True

    for prefix in collect_source_of_truth_prefixes(assistant_config):
        prefix_text = str(prefix or "").strip()

        if not prefix_text:
            continue

        if path == prefix_text or path.startswith(prefix_text + "."):
            return True

    return False


def filter_updates_by_policy(
    updates: Dict[str, Any],
    *,
    allow_source_of_truth: bool = True,
    allow_paths: Optional[List[str]] = None,
    deny_paths: Optional[List[str]] = None,
    assistant_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    if not isinstance(updates, dict):
        return {}

    allow_paths = allow_paths or []
    deny_paths = deny_paths or []

    filtered: Dict[str, Any] = {}

    for path, value in updates.items():
        path_text = str(path or "").strip()

        if not path_text:
            continue

        if is_empty(value):
            continue

        if deny_paths and path_matches_policy_list(path_text, deny_paths):
            continue

        if allow_paths and not path_matches_policy_list(path_text, allow_paths):
            continue

        if not allow_source_of_truth and path_is_source_of_truth(path_text, assistant_config):
            continue

        filtered[path_text] = value

    return filtered


def filter_clear_by_policy(
    clear: List[str],
    *,
    allow_source_of_truth: bool = True,
    allow_paths: Optional[List[str]] = None,
    deny_paths: Optional[List[str]] = None,
    assistant_config: Optional[Dict[str, Any]] = None
) -> List[str]:
    if not isinstance(clear, list):
        return []

    allow_paths = allow_paths or []
    deny_paths = deny_paths or []

    filtered: List[str] = []

    for path in clear:
        path_text = str(path or "").strip()

        if not path_text:
            continue

        if deny_paths and path_matches_policy_list(path_text, deny_paths):
            continue

        if allow_paths and not path_matches_policy_list(path_text, allow_paths):
            continue

        if not allow_source_of_truth and path_is_source_of_truth(path_text, assistant_config):
            continue

        filtered.append(path_text)

    return filtered


def apply_variable_patch(
    variables: Dict[str, Any],
    updates: Dict[str, Any],
    clear: List[str],
    *,
    allow_source_of_truth: bool = True,
    allow_paths: Optional[List[str]] = None,
    deny_paths: Optional[List[str]] = None,
    assistant_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    patched = dict(variables or {})

    safe_clear = filter_clear_by_policy(
        clear,
        allow_source_of_truth=allow_source_of_truth,
        allow_paths=allow_paths,
        deny_paths=deny_paths,
        assistant_config=assistant_config
    )

    for path in safe_clear:
        deep_delete(patched, path)

    safe_updates = filter_updates_by_policy(
        updates,
        allow_source_of_truth=allow_source_of_truth,
        allow_paths=allow_paths,
        deny_paths=deny_paths,
        assistant_config=assistant_config
    )

    for path, value in safe_updates.items():
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

        if not is_empty(value):
            deep_set(scoped, path, value)

    return scoped


def get_subagent_aliases(
    assistant_config: Dict[str, Any],
    subagent_name: str
) -> List[str]:
    if not isinstance(assistant_config, dict):
        return []

    aliases_config = assistant_config.get("subagent_aliases", {})
    if not isinstance(aliases_config, dict):
        aliases_config = get_config_path_value(assistant_config, "routing.subagent_aliases", {})

    target = str(subagent_name or "").strip()
    aliases: List[str] = [target] if target else []

    if isinstance(aliases_config, dict):
        configured = aliases_config.get(target, [])
        if isinstance(configured, list):
            aliases.extend([str(item) for item in configured if str(item or "").strip()])
        elif isinstance(configured, str) and configured.strip():
            aliases.append(configured)

    return unique_list(aliases)


def get_subagent_config(
    assistant_config: Dict[str, Any],
    subagent_name: str,
    default: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Return a subagent configuration by id/name/key/aliases, supporting both
    dict and list domain_bundle formats. Aliases are config-driven through
    assistant_config.subagent_aliases.
    """
    if not isinstance(assistant_config, dict):
        return dict(default or {})

    target = str(subagent_name or "").strip()
    if not target:
        return dict(default or {})

    normalization = assistant_config.get("normalization", {}) or {}
    normalized_targets = {
        normalize_text(candidate, normalization)
        for candidate in get_subagent_aliases(assistant_config, target)
        if str(candidate or "").strip()
    }

    raw_subagents = assistant_config.get("subagents", {})

    if isinstance(raw_subagents, dict):
        for key, value in raw_subagents.items():
            if not isinstance(value, dict):
                continue

            candidates = [
                str(key or ""),
                str(value.get("id", "") or ""),
                str(value.get("name", "") or ""),
                str(value.get("key", "") or ""),
            ]

            aliases = value.get("aliases", [])
            if isinstance(aliases, list):
                candidates.extend([str(item) for item in aliases if str(item or "").strip()])

            for candidate in candidates:
                if normalize_text(candidate, normalization) in normalized_targets:
                    config = dict(value)
                    config.setdefault("id", str(value.get("id") or key or target))
                    config.setdefault("name", str(value.get("name") or key or target))
                    return config

        return dict(default or {})

    if isinstance(raw_subagents, list):
        for item in raw_subagents:
            if not isinstance(item, dict):
                continue

            candidates = [
                str(item.get("id", "") or ""),
                str(item.get("name", "") or ""),
                str(item.get("key", "") or ""),
            ]

            aliases = item.get("aliases", [])
            if isinstance(aliases, list):
                candidates.extend([str(alias) for alias in aliases if str(alias or "").strip()])

            for candidate in candidates:
                if normalize_text(candidate, normalization) in normalized_targets:
                    config = dict(item)
                    config.setdefault("id", target)
                    config.setdefault("name", target)
                    return config

    return dict(default or {})


def get_subagent_config_list(assistant_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Return all subagent configs as a normalized list, regardless of whether the
    domain bundle stores subagents as a dict or list.
    """
    if not isinstance(assistant_config, dict):
        return []

    raw_subagents = assistant_config.get("subagents", {})

    if isinstance(raw_subagents, list):
        return [dict(item) for item in raw_subagents if isinstance(item, dict)]

    if isinstance(raw_subagents, dict):
        configs: List[Dict[str, Any]] = []

        for key, value in raw_subagents.items():
            if not isinstance(value, dict):
                continue

            config = dict(value)
            config.setdefault("id", str(key))
            config.setdefault("name", str(key))
            configs.append(config)

        return configs

    return []


def get_subagent_variable_scope(
    assistant_config: Dict[str, Any],
    subagent_name: str,
    variables: Dict[str, Any]
) -> Dict[str, Any]:
    subagent_config = get_subagent_config(
        assistant_config=assistant_config,
        subagent_name=subagent_name,
        default={}
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

        if not is_empty(value):
            deep_set(result, str(target), value)

    return result


def build_updates_from_mapping(mapping: Dict[str, str], context: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}

    if not isinstance(mapping, dict):
        return result

    for target, source in mapping.items():
        value = deep_get(context, str(source), "")

        if not is_empty(value):
            result[str(target)] = value

    return result


def extract_by_patterns(
    message: str,
    patterns: List[Dict[str, Any]],
    variables: Dict[str, Any],
    normalization_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Deterministic extraction helper.

    Important: this returns updates only. It does not mutate the input variables
    as a side effect, so the graph/state layer remains easier to reason about.
    """
    updates: Dict[str, Any] = {}

    if not isinstance(patterns, list):
        return updates

    raw_message = normalize_digits(message, normalization_config.get("digit_map", {}))

    working_variables = dict(variables or {})

    for item in patterns:
        if not isinstance(item, dict):
            continue

        if item.get("enabled", True) is False:
            continue

        variable = item.get("variable")
        regex = item.get("regex")
        group = int(item.get("group", 1))
        when_missing = item.get("when_missing")

        if when_missing:
            current_value = deep_get(working_variables, str(when_missing))

            if not is_empty(current_value):
                continue

        when_path = str(item.get("when_path") or "").strip()
        if when_path:
            current = deep_get(working_variables, when_path)
            if "when_equals" in item and current != item.get("when_equals"):
                continue
            if item.get("when_exists") is True and is_empty(current):
                continue
            if item.get("when_exists") is False and not is_empty(current):
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
                deep_set(working_variables, str(variable), value)

    return updates


def get_missing_paths(paths: List[str], variables: Dict[str, Any]) -> List[str]:
    missing: List[str] = []

    for path in paths or []:
        value = deep_get({"variables": variables}, path)

        if is_empty(value):
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


def evaluate_condition(condition: Dict[str, Any], context: Dict[str, Any]) -> bool:
    if not isinstance(condition, dict):
        return False

    path = str(condition.get("path", "") or "").strip()
    actual = deep_get(context, path) if path else None

    if condition.get("exists") is True and is_empty(actual):
        return False

    if condition.get("exists") is False and not is_empty(actual):
        return False

    if "equals" in condition and actual != condition.get("equals"):
        return False

    if "not_equals" in condition and actual == condition.get("not_equals"):
        return False

    allowed_values = condition.get("in")
    if isinstance(allowed_values, list) and actual not in allowed_values:
        return False

    disallowed_values = condition.get("not_in")
    if isinstance(disallowed_values, list) and actual in disallowed_values:
        return False

    if condition.get("truthy") is True and not bool(actual):
        return False

    if condition.get("falsy") is True and bool(actual):
        return False

    return True


def get_tool_update_policy(
    assistant_config: Dict[str, Any],
    operation: str,
    rule: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Source-of-truth state should normally be changed only by tool update rules.
    This helper lets the domain bundle explicitly restrict or relax paths per operation.
    """
    state_engine = assistant_config.get("state_engine", {}) if isinstance(assistant_config, dict) else {}
    policies = state_engine.get("tool_update_policy", {}) if isinstance(state_engine, dict) else {}

    operation_policy = policies.get(operation, {}) if isinstance(policies, dict) else {}

    if not isinstance(operation_policy, dict):
        operation_policy = {}

    rule_policy = rule.get("policy", {}) if isinstance(rule, dict) else {}

    if not isinstance(rule_policy, dict):
        rule_policy = {}

    merged = dict(operation_policy)
    merged.update(rule_policy)

    return merged


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

    policy = get_tool_update_policy(
        assistant_config=assistant_config,
        operation=operation,
        rule=rule
    )

    allow_paths = policy.get("allow_paths")
    deny_paths = policy.get("deny_paths")
    allow_source_of_truth = bool(policy.get("allow_source_of_truth", True))

    if not isinstance(allow_paths, list):
        allow_paths = None

    if not isinstance(deny_paths, list):
        deny_paths = None

    clear = rule.get("clear", [])

    safe_clear = filter_clear_by_policy(
        clear,
        allow_source_of_truth=allow_source_of_truth,
        allow_paths=allow_paths,
        deny_paths=deny_paths,
        assistant_config=assistant_config
    )

    for path in safe_clear:
        deep_delete(patched, path)

    set_mapping = rule.get("set", {})

    if isinstance(set_mapping, dict):
        updates = build_updates_from_mapping(set_mapping, context)
        patched = apply_variable_patch(
            patched,
            updates,
            [],
            allow_source_of_truth=allow_source_of_truth,
            allow_paths=allow_paths,
            deny_paths=deny_paths,
            assistant_config=assistant_config
        )

    context["variables"] = patched

    conditional = rule.get("conditional", [])

    if isinstance(conditional, list):
        for item in conditional:
            if not isinstance(item, dict):
                continue

            when = item.get("when", {})

            if not isinstance(when, dict):
                continue

            if evaluate_condition(when, context):
                c_clear = item.get("clear", [])

                safe_c_clear = filter_clear_by_policy(
                    c_clear,
                    allow_source_of_truth=allow_source_of_truth,
                    allow_paths=allow_paths,
                    deny_paths=deny_paths
                )

                for path_to_clear in safe_c_clear:
                    deep_delete(patched, path_to_clear)

                c_set = item.get("set", {})

                if isinstance(c_set, dict):
                    updates = build_updates_from_mapping(c_set, context)
                    patched = apply_variable_patch(
                        patched,
                        updates,
                        [],
                        allow_source_of_truth=allow_source_of_truth,
                        allow_paths=allow_paths,
                        deny_paths=deny_paths
                    )

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

    # Optional graph/debug metadata. These do not affect old callers.
    result_type: str = ""
    reply_label: str = ""
    facts: Dict[str, Any] = field(default_factory=dict)
    missing_fields: List[str] = field(default_factory=list)
