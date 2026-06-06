import copy
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DATA_DIR = Path(os.getenv("DATA_DIR", "/app/data"))
CONFIG_DIR = Path(os.getenv("CONFIG_DIR", "/app/configs"))

ASSISTANTS_DIR = DATA_DIR / "assistants"
SCHEMAS_DIR = DATA_DIR / "schemas"

ASSISTANTS_DIR.mkdir(parents=True, exist_ok=True)
SCHEMAS_DIR.mkdir(parents=True, exist_ok=True)


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


def deep_copy_default(default: Any) -> Any:
    try:
        return copy.deepcopy(default)
    except Exception:
        return default


def safe_storage_id(value: str, fallback: str = "default") -> str:
    """
    Filesystem-safe assistant/config id.

    This prevents accidental path traversal while preserving normal assistant IDs.
    It is intentionally generic and not tied to any domain.
    """
    text = str(value or "").strip()

    if not text:
        return fallback

    text = re.sub(r"[^A-Za-z0-9_.@+-]+", "_", text)
    text = text.strip("._")

    return text or fallback


def safe_json_load(path: Path, default: Any) -> Any:
    if not path.exists():
        return deep_copy_default(default)

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return deep_copy_default(default)

    return data


def safe_json_load_with_warnings(path: Path, default: Any) -> Tuple[Any, List[str]]:
    if not path.exists():
        return deep_copy_default(default), [f"missing:{path}"]

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return deep_copy_default(default), [f"invalid_json:{path}:{type(exc).__name__}:{exc}"]

    return data, []


def atomic_json_write(path: Path, data: Any) -> None:
    """
    Optional helper used by admin tooling/tests.
    Loader itself is read-only, but keeping this generic helper here makes config
    writes safer if the app later exposes config save endpoints.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, indent=2)

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        delete=False,
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
    ) as tmp:
        tmp.write(payload)
        tmp_path = Path(tmp.name)

    tmp_path.replace(path)


def bundle_path(assistant_id: str) -> Path:
    return CONFIG_DIR / safe_storage_id(assistant_id) / "domain_bundle.json"


def assistant_path(assistant_id: str) -> Path:
    return ASSISTANTS_DIR / f"{safe_storage_id(assistant_id)}.json"


def schema_path(assistant_id: str) -> Path:
    return SCHEMAS_DIR / f"{safe_storage_id(assistant_id)}.json"


def deep_get(data: Any, path: str, default: Any = None) -> Any:
    current = data

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


def deep_set(data: Dict[str, Any], path: str, value: Any) -> Dict[str, Any]:
    if not path or not isinstance(data, dict):
        return data

    target = data
    parts = [part for part in str(path).split(".") if part]

    if not parts:
        return data

    for part in parts[:-1]:
        if part not in target or not isinstance(target.get(part), dict):
            target[part] = {}
        target = target[part]

    target[parts[-1]] = value
    return data


def deep_merge(base: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base or {})

    if not isinstance(incoming, dict):
        return merged

    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)

    return merged


def normalize_string_list(value: Any) -> List[str]:
    output: List[str] = []

    for item in as_list(value):
        text = str(item or "").strip()
        if text:
            output.append(text)

    return unique_list(output)


def normalize_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def normalize_int(value: Any, default: int, minimum: Optional[int] = None, maximum: Optional[int] = None) -> int:
    try:
        number = int(value)
    except Exception:
        number = default

    if minimum is not None:
        number = max(number, minimum)

    if maximum is not None:
        number = min(number, maximum)

    return number


def load_domain_bundle(assistant_id: str) -> Dict[str, Any]:
    data = safe_json_load(bundle_path(assistant_id), {})

    if isinstance(data, dict):
        return data

    return {}


def normalize_schema(assistant_id: str, schema: Any) -> Dict[str, Any]:
    if isinstance(schema, dict):
        normalized = copy.deepcopy(schema)
    else:
        normalized = {}

    normalized.setdefault("assistant_id", assistant_id)
    normalized.setdefault("variables", {})

    if not isinstance(normalized.get("variables"), dict):
        normalized["variables"] = {}

    # Support legacy schema shapes:
    # {"schema": {...}} or {"fields": {...}}.
    if not normalized["variables"]:
        if isinstance(normalized.get("schema"), dict):
            normalized["variables"] = copy.deepcopy(normalized.get("schema", {}))
        elif isinstance(normalized.get("fields"), dict):
            normalized["variables"] = copy.deepcopy(normalized.get("fields", {}))

    return normalized


def normalize_subagents(raw_subagents: Any) -> Dict[str, Dict[str, Any]]:
    """
    Supports both bundle formats:
    - "subagents": {"booking": {...}}
    - "subagents": [{"id": "booking", ...}]
    """
    normalized: Dict[str, Dict[str, Any]] = {}

    if isinstance(raw_subagents, dict):
        for key, value in raw_subagents.items():
            if not isinstance(value, dict):
                continue

            subagent_id = str(value.get("id") or key or "").strip()
            if not subagent_id:
                continue

            item = copy.deepcopy(value)
            item.setdefault("id", subagent_id)
            item.setdefault("name", subagent_id)
            normalized[subagent_id] = item

    elif isinstance(raw_subagents, list):
        for value in raw_subagents:
            if not isinstance(value, dict):
                continue

            subagent_id = str(value.get("id") or value.get("name") or value.get("key") or "").strip()
            if not subagent_id:
                continue

            item = copy.deepcopy(value)
            item.setdefault("id", subagent_id)
            item.setdefault("name", subagent_id)
            normalized[subagent_id] = item

    return normalized


def normalize_tool_operations(raw_operations: Any) -> Dict[str, Dict[str, Any]]:
    operations: Dict[str, Dict[str, Any]] = {}

    if isinstance(raw_operations, dict):
        for op_name, op_spec in raw_operations.items():
            op_name_text = str(op_name or "").strip()
            if not op_name_text:
                continue

            if isinstance(op_spec, dict):
                operations[op_name_text] = copy.deepcopy(op_spec)
            elif isinstance(op_spec, list):
                operations[op_name_text] = {"required": copy.deepcopy(op_spec)}
            else:
                operations[op_name_text] = {}

    elif isinstance(raw_operations, list):
        for op_spec in raw_operations:
            if not isinstance(op_spec, dict):
                continue

            op_name_text = str(op_spec.get("name") or op_spec.get("operation") or "").strip()
            if not op_name_text:
                continue

            item = copy.deepcopy(op_spec)
            item.pop("name", None)
            operations[op_name_text] = item

    return operations


def normalize_tools(raw_tools: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw_tools, list):
        return []

    normalized: List[Dict[str, Any]] = []

    for tool in raw_tools:
        if not isinstance(tool, dict):
            continue

        name = str(tool.get("name") or "").strip()
        if not name:
            continue

        item = copy.deepcopy(tool)
        item["name"] = name
        item["operations"] = normalize_tool_operations(item.get("operations", {}))
        normalized.append(item)

    return normalized


def collect_schema_source_of_truth_paths(schema: Dict[str, Any]) -> List[str]:
    variables = schema.get("variables", {})
    paths: List[str] = []

    def walk(fields: Dict[str, Any], prefix: str = "") -> None:
        if not isinstance(fields, dict):
            return

        for key, meta in fields.items():
            key_text = str(key or "").strip()
            if not key_text:
                continue

            path = f"{prefix}.{key_text}" if prefix else key_text

            if isinstance(meta, dict):
                if meta.get("source_of_truth") is True or meta.get("operational_state") is True:
                    paths.append(path)

                properties = meta.get("properties")
                if isinstance(properties, dict):
                    walk(properties, path)

    walk(variables)
    return unique_list(paths)


def normalize_source_of_truth(assistant: Dict[str, Any], schema: Dict[str, Any]) -> None:
    exact = normalize_string_list(assistant.get("source_of_truth_variables", []))
    exact.extend(normalize_string_list(deep_get(assistant, "source_of_truth.variables", [])))
    exact.extend(normalize_string_list(deep_get(assistant, "source_of_truth.exact_variables", [])))
    exact.extend(normalize_string_list(deep_get(assistant, "state_engine.source_of_truth_variables", [])))
    exact.extend(normalize_string_list(deep_get(assistant, "state_engine.source_of_truth.exact_variables", [])))
    exact.extend(collect_schema_source_of_truth_paths(schema))

    prefixes = normalize_string_list(assistant.get("source_of_truth_prefixes", []))
    prefixes.extend(normalize_string_list(deep_get(assistant, "source_of_truth.prefixes", [])))
    prefixes.extend(normalize_string_list(deep_get(assistant, "state_engine.source_of_truth_prefixes", [])))
    prefixes.extend(normalize_string_list(deep_get(assistant, "state_engine.source_of_truth.prefixes", [])))

    assistant["source_of_truth_variables"] = unique_list(exact)
    assistant["source_of_truth_prefixes"] = unique_list(prefixes)

    state_engine = assistant.setdefault("state_engine", {})
    if isinstance(state_engine, dict):
        state_engine["source_of_truth_variables"] = unique_list(
            normalize_string_list(state_engine.get("source_of_truth_variables", []))
            + assistant["source_of_truth_variables"]
        )
        state_engine["source_of_truth_prefixes"] = unique_list(
            normalize_string_list(state_engine.get("source_of_truth_prefixes", []))
            + assistant["source_of_truth_prefixes"]
        )


def ensure_tool_result_contract_defaults(assistant: Dict[str, Any]) -> None:
    """
    This only normalizes generic shape. It does not invent domain-specific
    required fields. Operation-specific contracts belong in domain_bundle.json.
    """
    for tool in assistant.get("tools", []) or []:
        if not isinstance(tool, dict):
            continue

        operations = tool.get("operations", {})
        if not isinstance(operations, dict):
            continue

        for operation, spec in operations.items():
            if not isinstance(spec, dict):
                continue

            spec.setdefault("required", spec.get("required", []))

            if not isinstance(spec.get("required"), list):
                spec["required"] = []

            if "required_result_fields" in spec and not isinstance(spec.get("required_result_fields"), list):
                spec["required_result_fields"] = []

            if "result_contract" in spec and not isinstance(spec.get("result_contract"), dict):
                spec["result_contract"] = {}


def normalize_template_response_policy(assistant: Dict[str, Any]) -> None:
    policy = assistant.get("template_response_policy", {})

    if not isinstance(policy, dict):
        assistant["template_response_policy"] = {}
        return

    normalized_policy: Dict[str, Dict[str, Any]] = {}

    for label, item in policy.items():
        label_text = str(label or "").strip()
        if not label_text:
            continue

        if isinstance(item, dict):
            normalized = copy.deepcopy(item)
        elif isinstance(item, str):
            normalized = {"response_template": item}
        else:
            normalized = {}

        for key in ["must_do", "must_not_do", "safe_examples", "banned_examples"]:
            if key in normalized and not isinstance(normalized.get(key), list):
                normalized[key] = as_list(normalized.get(key))

        normalized_policy[label_text] = normalized

    assistant["template_response_policy"] = normalized_policy


def normalize_routing_guardrails(assistant: Dict[str, Any]) -> None:
    guardrails = assistant.get("routing_guardrails", {})

    if not isinstance(guardrails, dict):
        assistant["routing_guardrails"] = {}
        return

    normalized: Dict[str, Dict[str, Any]] = {}

    for key, value in guardrails.items():
        key_text = str(key or "").strip()
        if not key_text or not isinstance(value, dict):
            continue

        item = copy.deepcopy(value)
        item.setdefault("enabled", True)

        normalized[key_text] = item

    assistant["routing_guardrails"] = normalized


def inject_generic_defaults(assistant: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Add safe generic defaults required by the hardened runtime.
    These are architectural defaults only, not domain behavior.
    """
    assistant.setdefault("normalization", {})
    if not isinstance(assistant.get("normalization"), dict):
        assistant["normalization"] = {}

    assistant.setdefault("state_engine", {})
    if not isinstance(assistant.get("state_engine"), dict):
        assistant["state_engine"] = {}

    assistant["state_engine"].setdefault("flatten_nested_updates", True)
    assistant["state_engine"].setdefault("allow_empty_updates", False)
    assistant["state_engine"].setdefault("allow_empty_update_paths", [])

    assistant.setdefault("request_handling", {})
    if not isinstance(assistant.get("request_handling"), dict):
        assistant["request_handling"] = {}

    request_handling = assistant["request_handling"]
    request_handling.setdefault("max_stored_messages", 80)
    request_handling.setdefault("max_stored_traces", 50)
    request_handling.setdefault("max_history_messages", 24)
    request_handling.setdefault("max_processed_requests", 120)
    request_handling.setdefault("conversation_lock_timeout_seconds", 30)
    request_handling.setdefault("max_message_chars", 12000)
    request_handling.setdefault("return_error_answer", True)
    request_handling.setdefault("idempotency_metadata_keys", [
        "idempotency_key",
        "message_id",
        "request_id"
    ])

    assistant.setdefault("request_variable_policy", {})
    if not isinstance(assistant.get("request_variable_policy"), dict):
        assistant["request_variable_policy"] = {}

    assistant["request_variable_policy"].setdefault("allow_source_of_truth", False)
    assistant["request_variable_policy"].setdefault("allow_empty_updates", False)

    assistant.setdefault("tool_runner", {})
    if not isinstance(assistant.get("tool_runner"), dict):
        assistant["tool_runner"] = {}

    assistant["tool_runner"].setdefault("raw_preview_chars", 600)
    assistant["tool_runner"].setdefault("allow_relaxed_json", False)

    assistant.setdefault("knowledge_retrieval", {})
    if not isinstance(assistant.get("knowledge_retrieval"), dict):
        assistant["knowledge_retrieval"] = {}

    assistant["knowledge_retrieval"].setdefault("max_queries", 4)
    assistant["knowledge_retrieval"].setdefault("per_query_top_k", 4)
    assistant["knowledge_retrieval"].setdefault("max_total_items", 8)

    assistant.setdefault("multi_intent_execution", {})
    if not isinstance(assistant.get("multi_intent_execution"), dict):
        assistant["multi_intent_execution"] = {}

    assistant["multi_intent_execution"].setdefault("execute_parallel_tool_requests_with_subagents", False)
    assistant["multi_intent_execution"].setdefault("latest_correction_wins", True)
    assistant["multi_intent_execution"].setdefault("never_parallelize_changed_preferences", True)

    assistant.setdefault("fallback_messages", {})
    if not isinstance(assistant.get("fallback_messages"), dict):
        assistant["fallback_messages"] = {}

    assistant["fallback_messages"].setdefault("empty_answer", "I need a little more detail to help.")
    assistant["fallback_messages"].setdefault("graph_error", "I had trouble processing that. Please try again.")
    assistant["fallback_messages"].setdefault("default_final", "I need a little more detail to help.")

    assistant.setdefault("tools", [])
    assistant["tools"] = normalize_tools(assistant.get("tools", []))

    assistant.setdefault("subagents", {})
    assistant["subagents"] = normalize_subagents(assistant.get("subagents", {}))

    normalize_template_response_policy(assistant)
    normalize_routing_guardrails(assistant)
    normalize_source_of_truth(assistant, schema)
    ensure_tool_result_contract_defaults(assistant)

    # Attach schema to assistant config so lower layers can read schema metadata
    # without needing a separate argument.
    assistant["schema"] = schema

    return assistant


def validate_assistant_config(assistant: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
    warnings: List[str] = []

    if not assistant.get("assistant_id"):
        warnings.append("assistant_id missing")

    if not isinstance(assistant.get("subagents"), dict):
        warnings.append("subagents normalized to empty dict")

    if not isinstance(assistant.get("tools"), list):
        warnings.append("tools normalized to empty list")

    for tool in assistant.get("tools", []) or []:
        if not isinstance(tool, dict):
            continue

        name = tool.get("name")
        if not name:
            warnings.append("tool without name ignored")
            continue

        if not tool.get("url") and str(tool.get("type", "http")).lower() == "http":
            warnings.append(f"tool {name} has no url")

        operations = tool.get("operations", {})
        if operations and not isinstance(operations, dict):
            warnings.append(f"tool {name} operations must be object/list")

        if isinstance(operations, dict):
            for op_name, spec in operations.items():
                if not isinstance(spec, dict):
                    warnings.append(f"tool {name}.{op_name} spec normalized to empty object")
                    continue

                required = spec.get("required", [])
                if required and not isinstance(required, list):
                    warnings.append(f"tool {name}.{op_name} required must be a list")

    variables = schema.get("variables", {})
    if not isinstance(variables, dict):
        warnings.append("schema.variables normalized to empty object")

    return warnings


def normalize_assistant(
    assistant_id: str,
    assistant: Any,
    playbook: Any = None,
    schema: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    if not isinstance(assistant, dict):
        return {}

    normalized = copy.deepcopy(assistant)
    normalized.setdefault("assistant_id", assistant_id)

    if isinstance(playbook, dict):
        normalized["domain_playbook"] = copy.deepcopy(playbook)

    schema = normalize_schema(assistant_id, schema or {})

    normalized = inject_generic_defaults(normalized, schema)

    warnings = validate_assistant_config(normalized, schema)
    normalized["_config_loader"] = {
        "version": "6.11-robust-config-loader",
        "warnings": warnings,
        "source_format": "domain_bundle_or_legacy_files"
    }

    return normalized


def load_assistant_and_schema(assistant_id: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    bundle = load_domain_bundle(assistant_id)

    if bundle:
        schema = normalize_schema(
            assistant_id=assistant_id,
            schema=bundle.get("schema", {})
        )

        assistant = normalize_assistant(
            assistant_id=assistant_id,
            assistant=bundle.get("assistant", {}),
            playbook=bundle.get("domain_playbook", {}),
            schema=schema
        )

        if assistant:
            return assistant, schema

    assistant_raw = safe_json_load(assistant_path(assistant_id), {})
    schema_raw = safe_json_load(schema_path(assistant_id), {})

    schema = normalize_schema(
        assistant_id=assistant_id,
        schema=schema_raw
    )

    assistant = normalize_assistant(
        assistant_id=assistant_id,
        assistant=assistant_raw,
        schema=schema
    )

    return assistant, schema


def get_config_source(assistant_id: str) -> Dict[str, Any]:
    domain_path = bundle_path(assistant_id)

    if domain_path.exists():
        bundle, warnings = safe_json_load_with_warnings(domain_path, {})
        assistant_exists = isinstance(bundle, dict) and isinstance(bundle.get("assistant"), dict) and bool(bundle.get("assistant"))
        schema_found = isinstance(bundle, dict) and isinstance(bundle.get("schema"), dict)

        assistant_config = {}
        schema = {}

        if isinstance(bundle, dict):
            schema = normalize_schema(assistant_id, bundle.get("schema", {}))
            assistant_config = normalize_assistant(
                assistant_id=assistant_id,
                assistant=bundle.get("assistant", {}),
                playbook=bundle.get("domain_playbook", {}),
                schema=schema
            )

        return {
            "source": "domain_bundle",
            "path": str(domain_path),
            "assistant_found": assistant_exists,
            "schema_found": schema_found,
            "warnings": warnings + deep_get(assistant_config, "_config_loader.warnings", []),
            "architecture_version": assistant_config.get("architecture_version", ""),
            "domain_playbook_version": deep_get(assistant_config, "domain_playbook.version", ""),
            "loader_version": deep_get(assistant_config, "_config_loader.version", "")
        }

    legacy_assistant_path = assistant_path(assistant_id)
    legacy_schema_path = schema_path(assistant_id)

    assistant_raw, assistant_warnings = safe_json_load_with_warnings(legacy_assistant_path, {})
    schema_raw, schema_warnings = safe_json_load_with_warnings(legacy_schema_path, {})

    schema = normalize_schema(assistant_id, schema_raw)
    assistant_config = normalize_assistant(
        assistant_id=assistant_id,
        assistant=assistant_raw,
        schema=schema
    )

    return {
        "source": "data_files",
        "assistant_path": str(legacy_assistant_path),
        "schema_path": str(legacy_schema_path),
        "assistant_found": legacy_assistant_path.exists() and isinstance(assistant_raw, dict) and bool(assistant_raw),
        "schema_found": legacy_schema_path.exists() and isinstance(schema_raw, dict),
        "warnings": assistant_warnings + schema_warnings + deep_get(assistant_config, "_config_loader.warnings", []),
        "architecture_version": assistant_config.get("architecture_version", ""),
        "loader_version": deep_get(assistant_config, "_config_loader.version", "")
    }
