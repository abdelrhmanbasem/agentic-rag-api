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
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def bundle_path(assistant_id: str) -> Path:
    return CONFIG_DIR / assistant_id / "domain_bundle.json"


def assistant_path(assistant_id: str) -> Path:
    return ASSISTANTS_DIR / f"{assistant_id}.json"


def schema_path(assistant_id: str) -> Path:
    return SCHEMAS_DIR / f"{assistant_id}.json"


def load_domain_bundle(assistant_id: str) -> Dict[str, Any]:
    return safe_json_load(bundle_path(assistant_id), {})


def load_assistant_and_schema(assistant_id: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    bundle = load_domain_bundle(assistant_id)

    if isinstance(bundle, dict) and bundle:
        assistant = bundle.get("assistant", {})
        schema = bundle.get("schema", {})
        playbook = bundle.get("domain_playbook", {})

        if isinstance(assistant, dict) and assistant:
            assistant = dict(assistant)
            assistant.setdefault("assistant_id", assistant_id)

            if isinstance(playbook, dict):
                assistant["domain_playbook"] = playbook

            if isinstance(schema, dict):
                schema = dict(schema)
                schema.setdefault("assistant_id", assistant_id)
            else:
                schema = {
                    "assistant_id": assistant_id,
                    "variables": {}
                }

            return assistant, schema

    assistant = safe_json_load(assistant_path(assistant_id), {})
    schema = safe_json_load(schema_path(assistant_id), {})

    return assistant, schema


def get_config_source(assistant_id: str) -> Dict[str, Any]:
    path = bundle_path(assistant_id)

    if path.exists():
        return {
            "source": "domain_bundle",
            "path": str(path)
        }

    return {
        "source": "data_files",
        "assistant_path": str(assistant_path(assistant_id)),
        "schema_path": str(schema_path(assistant_id))
    }
