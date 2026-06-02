import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

from app.subagents.base import compact_dict


class ToolRunner:
    def __init__(self, assistant_config: Dict[str, Any]) -> None:
        self.assistant_config = assistant_config
        self.tools = self.normalize_tools(assistant_config)

    @staticmethod
    def normalize_tools(assistant_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        tools = assistant_config.get("tools", [])

        if not isinstance(tools, list):
            return []

        normalized = []

        for tool in tools:
            if not isinstance(tool, dict):
                continue

            name = tool.get("name")
            if not name:
                continue

            operations = tool.get("operations", {})
            if isinstance(operations, list):
                operations = {
                    op.get("name"): op
                    for op in operations
                    if isinstance(op, dict) and op.get("name")
                }

            if not isinstance(operations, dict):
                operations = {}

            normalized.append({
                **tool,
                "operations": operations
            })

        return normalized

    def find_tool(self, tool_name: str) -> Optional[Dict[str, Any]]:
        for tool in self.tools:
            if tool.get("name") == tool_name:
                return tool

        return None

    def get_operation_spec(self, tool_name: str, operation: str) -> Dict[str, Any]:
        tool = self.find_tool(tool_name)

        if not tool:
            return {}

        operations = tool.get("operations", {})
        spec = operations.get(operation, {})

        return spec if isinstance(spec, dict) else {}

    def validate_required(self, tool_name: str, operation: str, arguments: Dict[str, Any]) -> List[str]:
        spec = self.get_operation_spec(tool_name, operation)
        required = spec.get("required", [])

        if not isinstance(required, list):
            return []

        missing = []

        for key in required:
            value = arguments.get(key)
            if value in [None, "", [], {}]:
                missing.append(key)

        return missing

    @staticmethod
    def render_value(value: Any, context: Dict[str, Any]) -> Any:
        if not isinstance(value, str):
            return value

        rendered = value

        for key, val in context.items():
            token = "{{" + key + "}}"
            if token in rendered:
                rendered = rendered.replace(token, str(val))

        return rendered

    def build_body(self, tool: Dict[str, Any], operation: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        template = tool.get("body_template")

        if isinstance(template, dict):
            context = {
                "operation": operation,
                **arguments,
                **os.environ
            }

            return {
                key: self.render_value(value, context)
                for key, value in template.items()
            }

        return {
            "operation": operation,
            **arguments
        }

    def call(self, tool_name: str, operation: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        arguments = compact_dict(arguments or {})

        tool = self.find_tool(tool_name)

        if not tool:
            return {
                "ok": False,
                "error_type": "tool_not_found",
                "error": "Tool not found",
                "tool_name": tool_name,
                "operation": operation
            }

        missing = self.validate_required(tool_name, operation, arguments)

        if missing:
            return {
                "ok": False,
                "error_type": "missing_tool_inputs",
                "error": "Missing required tool inputs",
                "tool_name": tool_name,
                "operation": operation,
                "missing": missing,
                "arguments": arguments
            }

        tool_type = str(tool.get("type", "http")).lower()

        if tool_type != "http":
            return {
                "ok": False,
                "error_type": "unsupported_tool_type",
                "error": f"Unsupported tool type: {tool_type}",
                "tool_name": tool_name,
                "operation": operation
            }

        return self.call_http(tool, operation, arguments)

    def call_http(self, tool: Dict[str, Any], operation: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        url = tool.get("url", "")

        if not url:
            return {
                "ok": False,
                "error_type": "tool_config_error",
                "error": "Tool URL is missing",
                "operation": operation
            }

        method = str(tool.get("method", "POST")).upper()
        timeout = int(tool.get("timeout_seconds", 30))

        headers = tool.get("headers", {})
        if not isinstance(headers, dict):
            headers = {}

        context = {
            "operation": operation,
            **arguments,
            **os.environ
        }

        rendered_headers = {
            str(key): str(self.render_value(value, context))
            for key, value in headers.items()
        }

        body = self.build_body(tool, operation, arguments)
        data = None

        if method in ["POST", "PUT", "PATCH"]:
            rendered_headers.setdefault("Content-Type", "application/json")
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")

        if method == "GET":
            query = urllib.parse.urlencode(body)
            separator = "&" if "?" in url else "?"
            url = f"{url}{separator}{query}"

        request = urllib.request.Request(
            url=url,
            data=data,
            headers=rendered_headers,
            method=method
        )

        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8", errors="replace")

                try:
                    parsed = json.loads(raw)
                except Exception:
                    parsed = {
                        "ok": True,
                        "raw_text": raw
                    }

                if isinstance(parsed, dict):
                    parsed.setdefault("ok", True)
                    parsed.setdefault("operation", operation)
                    return parsed

                return {
                    "ok": True,
                    "operation": operation,
                    "result": parsed
                }

        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            return {
                "ok": False,
                "error_type": "http_error",
                "operation": operation,
                "error": f"HTTP {exc.code}",
                "raw_text": raw
            }

        except Exception as exc:
            return {
                "ok": False,
                "error_type": "tool_runtime_error",
                "operation": operation,
                "error": str(exc)
            }
