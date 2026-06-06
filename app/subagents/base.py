import copy
import fnmatch
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Architecture batch: 6.20-deep-merge-persistent-variables-no-hardcoding


# Domain-specific source-of-truth paths must live in domain_bundle.json.
# These empty defaults are kept only for backward compatibility with imports.
SOURCE_OF_TRUTH_EXACT_VARIABLES = set()
SOURCE_OF_TRUTH_PREFIXES = tuple()

MISSING = object()


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

        if isinstance(current, dict):
            if part not in current:
                return default
            current = current.get(part)
            continue

        if isinstance(current, list) and part.isdigit():
            index = int(part)
            if index < 0 or index >= len(current):
                return default
            current = current[index]
            continue

        return default

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
        values.extend(collect_schema_source_paths(schema_vars))

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

        if pattern in ["*", ".*"]:
            return True

        if pattern.startswith("re:"):
            try:
                if re.search(pattern[3:], path_text):
                    return True
            except re.error:
                continue

        if pattern == path_text:
            return True

        if pattern.endswith(".*"):
            prefix = pattern[:-2]
            if path_text == prefix or path_text.startswith(prefix + "."):
                return True

        if pattern.endswith(".") and path_text.startswith(pattern):
            return True

        if any(ch in pattern for ch in ["*", "?", "["]):
            if fnmatch.fnmatch(path_text, pattern):
                return True

    return False



def path_parts(path: str) -> List[str]:
    return [part for part in str(path or "").split(".") if part]


def is_missing(value: Any) -> bool:
    return value is MISSING


def deep_has(data: Any, path: str) -> bool:
    return deep_get(data, path, MISSING) is not MISSING


def flatten_update_paths(
    updates: Dict[str, Any],
    prefix: str = "",
    *,
    keep_empty_dicts: bool = False
) -> Dict[str, Any]:
    output: Dict[str, Any] = {}

    if not isinstance(updates, dict):
        return output

    for key, value in updates.items():
        key_text = str(key or "").strip()

        if not key_text:
            continue

        path = f"{prefix}.{key_text}" if prefix else key_text

        if isinstance(value, dict) and "." not in key_text:
            nested = flatten_update_paths(value, path, keep_empty_dicts=keep_empty_dicts)
            if nested:
                output.update(nested)
            elif keep_empty_dicts:
                output[path] = {}
        else:
            output[path] = value

    return output


def collect_schema_source_paths(schema_vars: Any, prefix: str = "") -> List[str]:
    paths: List[str] = []

    if not isinstance(schema_vars, dict):
        return paths

    for key, cfg in schema_vars.items():
        key_text = str(key or "").strip()
        if not key_text:
            continue

        path = f"{prefix}.{key_text}" if prefix else key_text

        if isinstance(cfg, dict):
            if cfg.get("source_of_truth") is True or cfg.get("operational_state") is True:
                paths.append(path)

            properties = cfg.get("properties")
            if isinstance(properties, dict):
                paths.extend(collect_schema_source_paths(properties, path))

    return paths


def is_empty(value: Any) -> bool:
    return value in [None, "", [], {}]


def deep_get(data: Any, path: str, default: Any = None) -> Any:
    if not path:
        return default

    current: Any = data

    for part in path_parts(path):
        if isinstance(current, dict):
            if part not in current:
                return default
            current = current.get(part)
            continue

        if isinstance(current, list) and part.isdigit():
            index = int(part)
            if index < 0 or index >= len(current):
                return default
            current = current[index]
            continue

        return default

    return current if current is not None else default


def deep_set(
    data: Dict[str, Any],
    path: str,
    value: Any,
    *,
    merge_dicts: bool = False,
    copy_value: bool = True
) -> Dict[str, Any]:
    if not path:
        return data

    if not isinstance(data, dict):
        return data

    target: Any = data
    parts = path_parts(path)

    if not parts:
        return data

    for part in parts[:-1]:
        if isinstance(target, dict):
            if part not in target or not isinstance(target.get(part), dict):
                target[part] = {}
            target = target[part]
            continue

        return data

    final_key = parts[-1]
    incoming = copy.deepcopy(value) if copy_value else value

    if (
        merge_dicts
        and isinstance(target, dict)
        and isinstance(target.get(final_key), dict)
        and isinstance(incoming, dict)
    ):
        target[final_key] = deep_merge(target.get(final_key, {}), incoming, allow_empty=True)
    elif isinstance(target, dict):
        target[final_key] = incoming

    return data


def deep_delete(data: Dict[str, Any], path: str, *, prune_empty: bool = False) -> Dict[str, Any]:
    if not path or not isinstance(data, dict):
        return data

    target: Any = data
    parts = path_parts(path)
    parents: List[Any] = []

    if not parts:
        return data

    for part in parts[:-1]:
        if not isinstance(target, dict) or part not in target:
            return data

        parents.append((target, part))
        target = target[part]

    if isinstance(target, dict):
        target.pop(parts[-1], None)

    if prune_empty:
        for parent, key in reversed(parents):
            child = parent.get(key)
            if child == {}:
                parent.pop(key, None)
            else:
                break

    return data


def deep_merge(
    base: Dict[str, Any],
    incoming: Dict[str, Any],
    *,
    allow_empty: bool = False,
    replace_lists: bool = True
) -> Dict[str, Any]:
    merged = copy.deepcopy(base or {})

    if not isinstance(incoming, dict):
        return merged

    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(
                merged[key],
                value,
                allow_empty=allow_empty,
                replace_lists=replace_lists
            )
        elif allow_empty or not is_empty(value):
            merged[key] = copy.deepcopy(value)

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
    normalization_config = normalization_config if isinstance(normalization_config, dict) else {}
    digit_map = normalization_config.get("digit_map", {})
    replacements = normalization_config.get("replacements", {})

    output = normalize_digits(str(text or ""), digit_map).strip().lower()

    if isinstance(replacements, dict):
        for old, new in replacements.items():
            output = output.replace(str(old), str(new))

    strip_diacritics = bool(normalization_config.get("strip_diacritics", True))
    if strip_diacritics:
        output = re.sub(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]", "", output)

    punctuation_replacements = normalization_config.get("punctuation_replacements", {})
    if isinstance(punctuation_replacements, dict):
        for old, new in punctuation_replacements.items():
            output = output.replace(str(old), str(new))

    output = re.sub(r"\s+", " ", output)
    return output.strip()


def matches_phrase(message: str, phrase: Any, normalization_config: Dict[str, Any]) -> bool:
    normalized = normalize_text(message, normalization_config)

    if isinstance(phrase, dict):
        exclude = phrase.get("exclude")
        if exclude and matches_any(message, as_list(exclude), normalization_config):
            return False

        if "regex" in phrase:
            pattern = str(phrase.get("regex") or "")
            if not pattern:
                return False
            try:
                return bool(re.search(pattern, str(message or ""), flags=re.IGNORECASE))
            except re.error:
                return False

        if "all" in phrase:
            return all(matches_phrase(message, item, normalization_config) for item in as_list(phrase.get("all")))

        if "any" in phrase:
            return any(matches_phrase(message, item, normalization_config) for item in as_list(phrase.get("any")))

        phrase_value = phrase.get("phrase", phrase.get("text", phrase.get("value", "")))
        normalized_phrase = normalize_text(str(phrase_value), normalization_config)
        mode = str(phrase.get("match", phrase.get("mode", "contains")) or "contains").strip().lower()

        if not normalized_phrase:
            return False

        if mode == "exact":
            return normalized == normalized_phrase
        if mode == "prefix":
            return normalized.startswith(normalized_phrase)
        if mode == "suffix":
            return normalized.endswith(normalized_phrase)
        if mode == "word":
            return bool(re.search(rf"(?<!\w){re.escape(normalized_phrase)}(?!\w)", normalized))

        return normalized_phrase in normalized

    normalized_phrase = normalize_text(str(phrase), normalization_config)
    return bool(normalized_phrase and normalized_phrase in normalized)


def matches_any(message: str, phrases: List[str], normalization_config: Dict[str, Any]) -> bool:
    for phrase in phrases or []:
        if matches_phrase(message, phrase, normalization_config):
            return True

    return False


def render_template(template: str, context: Dict[str, Any]) -> str:
    """
    Rendering is used for internal draft labels / fallback hints only.
    Final customer-facing wording belongs to graph.response_node.

    Supported placeholders:
    - {{path.to.value}}
    - {{path.to.value|fallback text}}
    - {{path.one||path.two||fallback text}} for first-present selection
    """
    if not isinstance(template, str):
        return ""

    pattern = re.compile(r"{{\s*([^}]+)\s*}}")

    def resolve_placeholder(expr: str) -> str:
        options = [part.strip() for part in str(expr or "").split("||") if part.strip()]

        if not options:
            return ""

        for option in options:
            path_and_default = [part.strip() for part in option.split("|", 1)]
            path = path_and_default[0]
            fallback = path_and_default[1] if len(path_and_default) > 1 else ""

            value = deep_get(context, path, MISSING)

            if value is not MISSING and value is not None:
                return str(value)

            if fallback:
                return fallback

        return ""

    def replace(match: re.Match) -> str:
        return resolve_placeholder(match.group(1).strip())

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
    assistant_config: Optional[Dict[str, Any]] = None,
    allow_empty_updates: bool = False,
    allow_empty_paths: Optional[List[str]] = None
) -> Dict[str, Any]:
    if not isinstance(updates, dict):
        return {}

    allow_paths = allow_paths or []
    deny_paths = deny_paths or []
    allow_empty_paths = allow_empty_paths or []

    filtered: Dict[str, Any] = {}

    for path, value in updates.items():
        path_text = str(path or "").strip()

        if not path_text:
            continue

        empty_allowed = allow_empty_updates or path_matches_policy_list(path_text, allow_empty_paths)

        if is_empty(value) and not empty_allowed:
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
        if isinstance(path, dict):
            path = path.get("path", "")

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

    return unique_list(filtered)


def apply_variable_patch(
    variables: Dict[str, Any],
    updates: Dict[str, Any],
    clear: List[str],
    *,
    allow_source_of_truth: bool = True,
    allow_paths: Optional[List[str]] = None,
    deny_paths: Optional[List[str]] = None,
    assistant_config: Optional[Dict[str, Any]] = None,
    allow_empty_updates: Optional[bool] = None,
    allow_empty_paths: Optional[List[str]] = None,
    flatten_nested_updates: Optional[bool] = None,
    prune_empty_on_clear: bool = False
) -> Dict[str, Any]:
    assistant_config = assistant_config if isinstance(assistant_config, dict) else {}
    state_engine = assistant_config.get("state_engine", {}) if isinstance(assistant_config, dict) else {}

    if not isinstance(state_engine, dict):
        state_engine = {}

    if allow_empty_updates is None:
        allow_empty_updates = bool(state_engine.get("allow_empty_updates", False))

    configured_empty_paths = state_engine.get("allow_empty_update_paths", [])
    if not isinstance(configured_empty_paths, list):
        configured_empty_paths = []

    allow_empty_paths = unique_list((allow_empty_paths or []) + configured_empty_paths)

    if flatten_nested_updates is None:
        flatten_nested_updates = bool(state_engine.get("flatten_nested_updates", True))

    patched = copy.deepcopy(variables or {})

    safe_clear = filter_clear_by_policy(
        clear,
        allow_source_of_truth=allow_source_of_truth,
        allow_paths=allow_paths,
        deny_paths=deny_paths,
        assistant_config=assistant_config
    )

    for path in safe_clear:
        deep_delete(patched, path, prune_empty=prune_empty_on_clear)

    update_payload = updates if isinstance(updates, dict) else {}

    if flatten_nested_updates:
        update_payload = flatten_update_paths(update_payload)

    safe_updates = filter_updates_by_policy(
        update_payload,
        allow_source_of_truth=allow_source_of_truth,
        allow_paths=allow_paths,
        deny_paths=deny_paths,
        assistant_config=assistant_config,
        allow_empty_updates=bool(allow_empty_updates),
        allow_empty_paths=allow_empty_paths
    )

    for path, value in safe_updates.items():
        existing_value = deep_get(patched, path)

        # Empty incoming values must never erase a non-empty existing value,
        # unless this exact path has been explicitly allowed by config.
        empty_allowed_for_path = (
            bool(allow_empty_updates)
            or path_matches_policy_list(path, allow_empty_paths)
        )

        if (
            is_empty(value)
            and not empty_allowed_for_path
            and existing_value not in [None, "", [], {}]
        ):
            continue

        # Deep-merge nested dictionaries instead of replacing the whole object.
        # This preserves sibling fields such as customer_profile.full_name when
        # customer_profile.phone is updated later.
        deep_set(patched, path, value, merge_dicts=True)

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


def resolve_mapping_value(source: Any, context: Dict[str, Any]) -> Any:
    if isinstance(source, dict):
        if "literal" in source:
            return source.get("literal")

        if "path" in source:
            return deep_get(context, str(source.get("path") or ""), MISSING)

        if "source" in source:
            return deep_get(context, str(source.get("source") or ""), MISSING)

        if "first_present" in source:
            for path in as_list(source.get("first_present")):
                value = deep_get(context, str(path or ""), MISSING)
                if value is not MISSING and not is_empty(value):
                    return value
            return MISSING

        if "template" in source:
            return render_template(str(source.get("template") or ""), context)

        if "concat" in source:
            separator = str(source.get("separator", "") or "")
            parts: List[str] = []
            for item in as_list(source.get("concat")):
                value = resolve_mapping_value(item, context)
                if value is not MISSING and not is_empty(value):
                    parts.append(str(value))
            return separator.join(parts)

        output: Dict[str, Any] = {}
        for key, value_source in source.items():
            value = resolve_mapping_value(value_source, context)
            if value is not MISSING and not is_empty(value):
                output[str(key)] = value
        return output if output else MISSING

    return deep_get(context, str(source), MISSING)


def build_object_from_mapping(
    mapping: Dict[str, Any],
    context: Dict[str, Any],
    *,
    allow_empty: bool = False
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}

    if not isinstance(mapping, dict):
        return result

    for target, source in mapping.items():
        target_text = str(target or "").strip()

        if not target_text:
            continue

        value = resolve_mapping_value(source, context)
        source_allows_empty = isinstance(source, dict) and bool(source.get("allow_empty", False))

        if value is not MISSING and (allow_empty or source_allows_empty or not is_empty(value)):
            deep_set(result, target_text, value)

    return result


def build_updates_from_mapping(
    mapping: Dict[str, Any],
    context: Dict[str, Any],
    *,
    allow_empty: bool = False
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}

    if not isinstance(mapping, dict):
        return result

    for target, source in mapping.items():
        target_text = str(target or "").strip()

        if not target_text:
            continue

        value = resolve_mapping_value(source, context)
        source_allows_empty = isinstance(source, dict) and bool(source.get("allow_empty", False))

        if value is not MISSING and (allow_empty or source_allows_empty or not is_empty(value)):
            result[target_text] = value

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
    working_variables = copy.deepcopy(variables or {})

    for item in patterns:
        if not isinstance(item, dict):
            continue

        if item.get("enabled", True) is False:
            continue

        variable = str(item.get("variable") or item.get("target") or "").strip()
        regex = str(item.get("regex") or "").strip()

        try:
            group = int(item.get("group", 1))
        except Exception:
            group = 1

        when_missing_values = as_list(item.get("when_missing"))
        if when_missing_values:
            should_skip = False
            for when_missing in when_missing_values:
                current_value = deep_get(working_variables, str(when_missing))
                if not is_empty(current_value):
                    should_skip = True
                    break
            if should_skip:
                continue

        when_path = str(item.get("when_path") or "").strip()
        if when_path:
            condition = {
                "path": when_path,
                "equals": item.get("when_equals") if "when_equals" in item else MISSING,
                "exists": item.get("when_exists") if "when_exists" in item else MISSING,
            }
            condition = {k: v for k, v in condition.items() if v is not MISSING}
            if condition and not evaluate_condition(condition, {"variables": working_variables, **working_variables}):
                continue

        conditions = item.get("conditions", [])
        if isinstance(conditions, list) and conditions:
            context = {
                "variables": working_variables,
                "message": raw_message,
                "normalization": normalization_config,
            }
            if not all(evaluate_condition(condition, context) for condition in conditions if isinstance(condition, dict)):
                continue

        if not variable or not regex:
            continue

        try:
            match = re.search(regex, raw_message, flags=re.IGNORECASE)
        except re.error:
            continue

        if not match:
            continue

        try:
            value = match.group(group).strip()
        except Exception:
            continue

        if value:
            updates[variable] = value
            deep_set(working_variables, variable, value)

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

    if "all" in condition:
        return all(evaluate_condition(item, context) for item in as_list(condition.get("all")) if isinstance(item, dict))

    if "any" in condition:
        return any(evaluate_condition(item, context) for item in as_list(condition.get("any")) if isinstance(item, dict))

    if "not" in condition and isinstance(condition.get("not"), dict):
        return not evaluate_condition(condition.get("not"), context)

    path = str(condition.get("path", "") or "").strip()
    actual = deep_get(context, path, MISSING) if path else context

    exists = actual is not MISSING and not is_empty(actual)

    if condition.get("exists") is True and not exists:
        return False

    if condition.get("exists") is False and exists:
        return False

    if condition.get("empty") is True and exists:
        return False

    if condition.get("not_empty") is True and not exists:
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

    if condition.get("truthy") is True and not bool(None if actual is MISSING else actual):
        return False

    if condition.get("falsy") is True and bool(None if actual is MISSING else actual):
        return False

    normalization = context.get("normalization", {}) if isinstance(context, dict) else {}

    if "equals_normalized" in condition:
        if normalize_text(str(actual if actual is not MISSING else ""), normalization) != normalize_text(str(condition.get("equals_normalized")), normalization):
            return False

    if "in_normalized" in condition and isinstance(condition.get("in_normalized"), list):
        normalized_actual = normalize_text(str(actual if actual is not MISSING else ""), normalization)
        normalized_allowed = {
            normalize_text(str(item), normalization)
            for item in condition.get("in_normalized", [])
        }
        if normalized_actual not in normalized_allowed:
            return False

    if "contains" in condition:
        if str(condition.get("contains")) not in str(actual if actual is not MISSING else ""):
            return False

    if "not_contains" in condition:
        if str(condition.get("not_contains")) in str(actual if actual is not MISSING else ""):
            return False

    if "regex" in condition:
        try:
            if not re.search(str(condition.get("regex") or ""), str(actual if actual is not MISSING else ""), flags=re.IGNORECASE):
                return False
        except re.error:
            return False

    if "matches_any" in condition:
        if not matches_any(str(actual if actual is not MISSING else ""), as_list(condition.get("matches_any")), normalization):
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
        "result": result or {},
        "assistant_config": assistant_config or {},
        "normalization": (assistant_config or {}).get("normalization", {})
    }

    patched = copy.deepcopy(variables or {})

    policy = get_tool_update_policy(
        assistant_config=assistant_config,
        operation=operation,
        rule=rule
    )

    allow_paths = policy.get("allow_paths")
    deny_paths = policy.get("deny_paths")
    allow_source_of_truth = bool(policy.get("allow_source_of_truth", True))
    allow_empty_updates = bool(policy.get("allow_empty_updates", True))
    allow_empty_paths = policy.get("allow_empty_paths", [])

    if not isinstance(allow_paths, list):
        allow_paths = None

    if not isinstance(deny_paths, list):
        deny_paths = None

    if not isinstance(allow_empty_paths, list):
        allow_empty_paths = []

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
        updates = build_updates_from_mapping(
            set_mapping,
            context,
            allow_empty=allow_empty_updates
        )
        patched = apply_variable_patch(
            patched,
            updates,
            [],
            allow_source_of_truth=allow_source_of_truth,
            allow_paths=allow_paths,
            deny_paths=deny_paths,
            assistant_config=assistant_config,
            allow_empty_updates=allow_empty_updates,
            allow_empty_paths=allow_empty_paths
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
                    deny_paths=deny_paths,
                    assistant_config=assistant_config
                )

                for path_to_clear in safe_c_clear:
                    deep_delete(patched, path_to_clear)

                c_set = item.get("set", {})

                if isinstance(c_set, dict):
                    updates = build_updates_from_mapping(
                        c_set,
                        context,
                        allow_empty=allow_empty_updates
                    )
                    patched = apply_variable_patch(
                        patched,
                        updates,
                        [],
                        allow_source_of_truth=allow_source_of_truth,
                        allow_paths=allow_paths,
                        deny_paths=deny_paths,
                        assistant_config=assistant_config,
                        allow_empty_updates=allow_empty_updates,
                        allow_empty_paths=allow_empty_paths
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
