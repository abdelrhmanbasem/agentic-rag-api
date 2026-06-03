import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from app.subagents.base import compact_dict


class ToolRunner:
    """
    Safe deterministic tool executor for the LangGraph architecture.

    Responsibilities:
    - execute only configured tools and operations
    - validate required inputs before execution
    - normalize success/error results
    - preserve source-of-truth tool metadata
    - never convert malformed/non-JSON tool responses into fake success

    Not responsible for:
    - deciding whether a tool should be called
    - writing final customer-facing replies
    """

    def __init__(self, assistant_config: Dict[str, Any]) -> None:
        self.assistant_config = assistant_config or {}
        self.tools = self.normalize_tools(self.assistant_config)

    @staticmethod
    def normalize_tools(assistant_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        tools = assistant_config.get("tools", [])

        if not isinstance(tools, list):
            return []

        normalized = []

        for tool in tools:
            if not isinstance(tool, dict):
                continue

            name = str(tool.get("name") or "").strip()

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
                "name": name,
                "operations": operations
            })

        return normalized

    def find_tool(self, tool_name: str) -> Optional[Dict[str, Any]]:
        name = str(tool_name or "").strip()

        for tool in self.tools:
            if tool.get("name") == name:
                return tool

        return None

    def get_operation_spec(self, tool_name: str, operation: str) -> Dict[str, Any]:
        tool = self.find_tool(tool_name)

        if not tool:
            return {}

        operations = tool.get("operations", {})
        spec = operations.get(operation, {})

        return spec if isinstance(spec, dict) else {}

    def validate_operation_exists(self, tool: Dict[str, Any], operation: str) -> bool:
        operations = tool.get("operations", {})

        if not isinstance(operations, dict):
            return False

        # If a tool has no explicit operation specs, keep backward compatibility.
        if not operations:
            return True

        return operation in operations

    def validate_required(self, tool_name: str, operation: str, arguments: Dict[str, Any]) -> List[str]:
        spec = self.get_operation_spec(tool_name, operation)
        required = spec.get("required", [])

        if not isinstance(required, list):
            return []

        missing = []

        for key in required:
            value = arguments.get(key)

            if value in [None, "", [], {}]:
                missing.append(str(key))

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
        tool_name = str(tool_name or "").strip()
        operation = str(operation or "").strip()
        arguments = compact_dict(arguments or {})

        if not tool_name:
            return self.error_result(
                error_type="tool_name_missing",
                error="Tool name is missing",
                tool_name=tool_name,
                operation=operation,
                arguments=arguments
            )

        if not operation:
            return self.error_result(
                error_type="operation_missing",
                error="Tool operation is missing",
                tool_name=tool_name,
                operation=operation,
                arguments=arguments
            )

        tool = self.find_tool(tool_name)

        if not tool:
            return self.error_result(
                error_type="tool_not_found",
                error="Tool not found",
                tool_name=tool_name,
                operation=operation,
                arguments=arguments
            )

        if not self.validate_operation_exists(tool, operation):
            return self.error_result(
                error_type="operation_not_configured",
                error="Tool operation is not configured",
                tool_name=tool_name,
                operation=operation,
                arguments=arguments
            )

        missing = self.validate_required(tool_name, operation, arguments)

        if missing:
            return self.error_result(
                error_type="missing_tool_inputs",
                error="Missing required tool inputs",
                tool_name=tool_name,
                operation=operation,
                arguments=arguments,
                extra={
                    "missing": missing
                }
            )

        tool_type = str(tool.get("type", "http")).lower().strip()

        if tool_type != "http":
            return self.error_result(
                error_type="unsupported_tool_type",
                error=f"Unsupported tool type: {tool_type}",
                tool_name=tool_name,
                operation=operation,
                arguments=arguments
            )

        return self.call_http(tool, operation, arguments)

    def call_http(self, tool: Dict[str, Any], operation: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        tool_name = str(tool.get("name") or "").strip()
        url = str(tool.get("url") or "").strip()

        if not url:
            return self.error_result(
                error_type="tool_config_error",
                error="Tool URL is missing",
                tool_name=tool_name,
                operation=operation,
                arguments=arguments
            )

        method = str(tool.get("method", "POST")).upper().strip()

        try:
            timeout = int(tool.get("timeout_seconds", 30))
        except Exception:
            timeout = 30

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
        final_url = url

        if method in ["POST", "PUT", "PATCH"]:
            rendered_headers.setdefault("Content-Type", "application/json")
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")

        elif method == "GET":
            query = urllib.parse.urlencode(body)
            separator = "&" if "?" in final_url else "?"
            final_url = f"{final_url}{separator}{query}"

        else:
            return self.error_result(
                error_type="unsupported_http_method",
                error=f"Unsupported HTTP method: {method}",
                tool_name=tool_name,
                operation=operation,
                arguments=arguments
            )

        request = urllib.request.Request(
            url=final_url,
            data=data,
            headers=rendered_headers,
            method=method
        )

        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                status_code = int(getattr(response, "status", 200) or 200)
                content_type = response.headers.get("Content-Type", "")
                raw = response.read().decode("utf-8", errors="replace")

                parsed, parse_error = self.parse_json_response(raw)

                if parse_error:
                    return self.error_result(
                        error_type="non_json_tool_response",
                        error="Tool returned a non-JSON response",
                        tool_name=tool_name,
                        operation=operation,
                        arguments=arguments,
                        extra={
                            "http_status": status_code,
                            "content_type": content_type,
                            "raw_text_preview": self.preview_raw(raw)
                        }
                    )

                normalized = self.normalize_success_result(
                    parsed=parsed,
                    tool_name=tool_name,
                    operation=operation,
                    arguments=arguments,
                    http_status=status_code,
                    content_type=content_type
                )

                return normalized

        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")

            return self.error_result(
                error_type="http_error",
                error=f"HTTP {exc.code}",
                tool_name=tool_name,
                operation=operation,
                arguments=arguments,
                extra={
                    "http_status": exc.code,
                    "raw_text_preview": self.preview_raw(raw)
                }
            )

        except urllib.error.URLError as exc:
            return self.error_result(
                error_type="network_error",
                error=str(exc.reason),
                tool_name=tool_name,
                operation=operation,
                arguments=arguments
            )

        except Exception as exc:
            return self.error_result(
                error_type="tool_runtime_error",
                error=f"{type(exc).__name__}: {exc}",
                tool_name=tool_name,
                operation=operation,
                arguments=arguments
            )

    @staticmethod
    def parse_json_response(raw: str) -> Tuple[Any, bool]:
        text = str(raw or "").strip()

        if not text:
            return None, True

        try:
            return json.loads(text), False
        except Exception:
            return None, True

    def normalize_success_result(
        self,
        parsed: Any,
        tool_name: str,
        operation: str,
        arguments: Dict[str, Any],
        http_status: int,
        content_type: str
    ) -> Dict[str, Any]:
        if isinstance(parsed, dict):
            result = dict(parsed)

            # Preserve explicit tool failure from source of truth.
            if result.get("ok") is False:
                result.setdefault("error_type", "tool_reported_failure")
            else:
                result.setdefault("ok", True)

            result.setdefault("tool_name", tool_name)
            result.setdefault("operation", operation)
            result.setdefault("arguments", arguments)
            result.setdefault("http_status", http_status)
            result.setdefault("content_type", content_type)

            return result

        return {
            "ok": True,
            "tool_name": tool_name,
            "operation": operation,
            "arguments": arguments,
            "http_status": http_status,
            "content_type": content_type,
            "result": parsed
        }

    @staticmethod
    def error_result(
        error_type: str,
        error: str,
        tool_name: str,
        operation: str,
        arguments: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        result = {
            "ok": False,
            "error_type": error_type,
            "error": error,
            "tool_name": tool_name,
            "operation": operation,
            "arguments": arguments or {}
        }

        if isinstance(extra, dict):
            result.update(extra)

        return result

    @staticmethod
    def preview_raw(raw: str, max_chars: int = 600) -> str:
        text = str(raw or "").strip()

        if len(text) <= max_chars:
            return text

        return text[:max_chars].rstrip() + "...[trimmed]"
