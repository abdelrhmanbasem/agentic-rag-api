import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple


DATA_DIR = Path(os.getenv("DATA_DIR", "/app/data"))
CONFIG_DIR = Path(os.getenv("CONFIG_DIR", "/app/configs"))

ASSISTANTS_DIR = DATA_DIR / "assistants"
SCHEMAS_DIR = DATA_DIR / "schemas"


def safe_json_load(path: Path, default: Any) -> Any:
    if not path.exists():
        return default

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default

    return data


def bundle_path(assistant_id: str) -> Path:
    return CONFIG_DIR / assistant_id / "domain_bundle.json"


def assistant_path(assistant_id: str) -> Path:
    return ASSISTANTS_DIR / f"{assistant_id}.json"


def schema_path(assistant_id: str) -> Path:
    return SCHEMAS_DIR / f"{assistant_id}.json"


def load_domain_bundle(assistant_id: str) -> Dict[str, Any]:
    data = safe_json_load(bundle_path(assistant_id), {})

    if isinstance(data, dict):
        return data

    return {}


def normalize_schema(assistant_id: str, schema: Any) -> Dict[str, Any]:
    if isinstance(schema, dict):
        normalized = dict(schema)
    else:
        normalized = {}

    normalized.setdefault("assistant_id", assistant_id)
    normalized.setdefault("variables", {})

    if not isinstance(normalized.get("variables"), dict):
        normalized["variables"] = {}

    return normalized


def normalize_assistant(
    assistant_id: str,
    assistant: Any,
    playbook: Any = None
) -> Dict[str, Any]:
    if not isinstance(assistant, dict):
        return {}

    normalized = dict(assistant)
    normalized.setdefault("assistant_id", assistant_id)

    if isinstance(playbook, dict):
        normalized["domain_playbook"] = playbook

    return normalized


def load_assistant_and_schema(assistant_id: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    bundle = load_domain_bundle(assistant_id)

    if bundle:
        assistant = normalize_assistant(
            assistant_id=assistant_id,
            assistant=bundle.get("assistant", {}),
            playbook=bundle.get("domain_playbook", {})
        )

        schema = normalize_schema(
            assistant_id=assistant_id,
            schema=bundle.get("schema", {})
        )

        if assistant:
            return assistant, schema

    assistant = safe_json_load(assistant_path(assistant_id), {})
    schema = safe_json_load(schema_path(assistant_id), {})

    assistant = normalize_assistant(
        assistant_id=assistant_id,
        assistant=assistant
    )

    schema = normalize_schema(
        assistant_id=assistant_id,
        schema=schema
    )

    return assistant, schema


def get_config_source(assistant_id: str) -> Dict[str, Any]:
    domain_path = bundle_path(assistant_id)

    if domain_path.exists():
        bundle = load_domain_bundle(assistant_id)
        assistant_exists = isinstance(bundle.get("assistant"), dict) and bool(bundle.get("assistant"))

        return {
            "source": "domain_bundle",
            "path": str(domain_path),
            "assistant_found": assistant_exists,
            "schema_found": isinstance(bundle.get("schema"), dict)
        }

    legacy_assistant_path = assistant_path(assistant_id)
    legacy_schema_path = schema_path(assistant_id)

    return {
        "source": "data_files",
        "assistant_path": str(legacy_assistant_path),
        "schema_path": str(legacy_schema_path),
        "assistant_found": legacy_assistant_path.exists(),
        "schema_found": legacy_schema_path.exists()
    }
