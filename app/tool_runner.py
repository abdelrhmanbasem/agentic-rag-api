import copy
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from app.subagents.base import compact_dict, deep_get


class TransientToolHTTPError(Exception):
    def __init__(self, status_code: int, raw: str = "") -> None:
        self.status_code = status_code
        self.raw = raw
        super().__init__(f"Transient HTTP {status_code}")


class TransientToolNetworkError(Exception):
    def __init__(self, reason: Any) -> None:
        self.reason = reason
        super().__init__(str(reason))


MISSING = object()


class ToolRunner:
    """
    Safe deterministic tool executor for the LangGraph architecture.

    Responsibilities:
    - execute only configured tools and operations
    - validate required inputs before execution
    - normalize success/error results
    - retry transient HTTP/network failures using config-driven policy
    - preserve source-of-truth tool metadata
    - support idempotency keys for unsafe write operations when configured
    - never convert malformed/non-JSON tool responses into fake success

    Not responsible for:
    - deciding whether a tool should be called
    - writing final customer-facing replies
    """

    DEFAULT_RETRYABLE_HTTP_STATUSES = [408, 425, 429, 500, 502, 503, 504]
    DEFAULT_RETRYABLE_NETWORK_MARKERS = [
        "timed out",
        "timeout",
        "temporarily unavailable",
        "temporary failure",
        "connection reset",
        "connection aborted",
        "connection refused",
        "remote end closed",
        "service unavailable",
    ]
    DEFAULT_SENSITIVE_KEYS = [
        "authorization",
        "api_key",
        "apikey",
        "x-api-key",
        "token",
        "access_token",
        "secret",
        "password",
        "cookie",
        "set-cookie",
    ]

    def __init__(self, assistant_config: Dict[str, Any]) -> None:
        self.assistant_config = assistant_config or {}
        self.runner_config = self.get_runner_config(self.assistant_config)
        self.tools = self.normalize_tools(self.assistant_config)

    @staticmethod
    def get_runner_config(assistant_config: Dict[str, Any]) -> Dict[str, Any]:
        config: Dict[str, Any] = {}

        if isinstance(assistant_config, dict):
            for key in ["tool_runner", "tool_execution", "tools_runtime"]:
                value = assistant_config.get(key)
                if isinstance(value, dict):
                    config.update(value)

        return config

    @staticmethod
    def deep_merge(base: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
        merged = copy.deepcopy(base or {})

        if not isinstance(incoming, dict):
            return merged

        for key, value in incoming.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = ToolRunner.deep_merge(merged[key], value)
            else:
                merged[key] = copy.deepcopy(value)

        return merged

    @staticmethod
    def as_list(value: Any) -> List[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        return [value]

    @staticmethod
    def is_missing_value(value: Any) -> bool:
        return value in [None, "", [], {}]

    @staticmethod
    def unique(values: List[Any]) -> List[Any]:
        output = []
        for value in values or []:
            if value not in output:
                output.append(value)
        return output

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
                    str(op.get("name") or op.get("operation") or "").strip(): op
                    for op in operations
                    if isinstance(op, dict)
                    and str(op.get("name") or op.get("operation") or "").strip()
                }

            if not isinstance(operations, dict):
                operations = {}

            normalized_operations: Dict[str, Any] = {}

            for op_name, op_spec in operations.items():
                op_name_text = str(op_name or "").strip()
                if not op_name_text:
                    continue

                if isinstance(op_spec, dict):
                    normalized_operations[op_name_text] = dict(op_spec)
                elif isinstance(op_spec, list):
                    normalized_operations[op_name_text] = {"required": op_spec}
                else:
                    normalized_operations[op_name_text] = {}

            aliases = tool.get("aliases", [])
            if isinstance(aliases, str):
                aliases = [aliases]
            if not isinstance(aliases, list):
                aliases = []

            normalized.append({
                **tool,
                "name": name,
                "aliases": [str(item).strip() for item in aliases if str(item or "").strip()],
                "operations": normalized_operations,
            })

        return normalized

    def find_tool(self, tool_name: str) -> Optional[Dict[str, Any]]:
        name = str(tool_name or "").strip()

        for tool in self.tools:
            names = [tool.get("name", "")]
            aliases = tool.get("aliases", [])
            if isinstance(aliases, list):
                names.extend([str(item) for item in aliases])

            if name in names:
                return tool

        return None

    def get_operation_spec(self, tool_name: str, operation: str) -> Dict[str, Any]:
        tool = self.find_tool(tool_name)

        if not tool:
            return {}

        operations = tool.get("operations", {})
        spec = operations.get(operation, {}) if isinstance(operations, dict) else {}

        return spec if isinstance(spec, dict) else {}

    def get_effective_operation_config(self, tool: Dict[str, Any], operation: str) -> Dict[str, Any]:
        operations = tool.get("operations", {})
        spec = operations.get(operation, {}) if isinstance(operations, dict) else {}
        spec = spec if isinstance(spec, dict) else {}

        inherited_keys = [
            "method",
            "url",
            "headers",
            "timeout_seconds",
            "retry",
            "retry_attempts",
            "retryable_http_statuses",
            "retryable_network_markers",
            "body_template",
            "query_template",
            "payload_mode",
            "parse_json",
            "allow_relaxed_json",
            "result_defaults",
            "success_defaults",
            "result_type",
            "result_kind",
            "result_contract",
            "required_result_fields",
            "sensitive_keys",
            "idempotency",
            "idempotency_enabled",
            "idempotency_key_fields",
            "idempotency_header",
            "allow_empty_arguments",
        ]

        effective = {key: copy.deepcopy(tool.get(key)) for key in inherited_keys if key in tool}
        effective = self.deep_merge(effective, spec)

        return effective

    def validate_operation_exists(self, tool: Dict[str, Any], operation: str) -> bool:
        operations = tool.get("operations", {})

        if not isinstance(operations, dict):
            return False

        # If a tool has no explicit operation specs, keep backward compatibility.
        if not operations:
            return True

        return operation in operations

    def get_argument_value(self, arguments: Dict[str, Any], path: str, default: Any = MISSING) -> Any:
        if not isinstance(arguments, dict):
            return default

        path_text = str(path or "").strip()

        if not path_text:
            return default

        if path_text in arguments:
            return arguments.get(path_text)

        return deep_get(arguments, path_text, default)

    @staticmethod
    def set_nested_argument(arguments: Dict[str, Any], path: str, value: Any) -> None:
        parts = [part for part in str(path or "").split(".") if part]
        if not parts:
            return

        target = arguments
        for part in parts[:-1]:
            if not isinstance(target.get(part), dict):
                target[part] = {}
            target = target[part]

        target[parts[-1]] = value

    def apply_argument_aliases(
        self,
        tool_name: str,
        operation: str,
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        spec = self.get_operation_spec(tool_name, operation)
        alias_config = spec.get("argument_aliases", {})

        if not isinstance(alias_config, dict):
            return arguments

        patched = copy.deepcopy(arguments or {})

        for canonical, aliases in alias_config.items():
            canonical_text = str(canonical or "").strip()

            if not canonical_text:
                continue

            current_value = self.get_argument_value(patched, canonical_text, MISSING)
            if current_value is not MISSING and not self.is_missing_value(current_value):
                continue

            for alias in self.as_list(aliases):
                value = self.get_argument_value(patched, str(alias or "").strip(), MISSING)
                if value is not MISSING and not self.is_missing_value(value):
                    self.set_nested_argument(patched, canonical_text, value)
                    break

        return patched

    def apply_argument_defaults(
        self,
        tool_name: str,
        operation: str,
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        spec = self.get_operation_spec(tool_name, operation)
        defaults = spec.get("argument_defaults", spec.get("defaults", spec.get("default_arguments", {})))

        if not isinstance(defaults, dict):
            return arguments

        patched = copy.deepcopy(arguments or {})

        for path, value in defaults.items():
            path_text = str(path or "").strip()
            if not path_text:
                continue

            current = self.get_argument_value(patched, path_text, MISSING)
            if current is MISSING or self.is_missing_value(current):
                self.set_nested_argument(patched, path_text, value)

        return patched

    def normalize_arguments(self, tool_name: str, operation: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        tool = self.find_tool(tool_name)
        effective = self.get_effective_operation_config(tool, operation) if tool else {}
        allow_empty_paths = effective.get("allow_empty_arguments", [])
        if not isinstance(allow_empty_paths, list):
            allow_empty_paths = []

        patched = copy.deepcopy(arguments or {})
        patched = self.apply_argument_aliases(tool_name, operation, patched)
        patched = self.apply_argument_defaults(tool_name, operation, patched)

        if allow_empty_paths:
            compacted = {}
            for key, value in patched.items():
                if not self.is_missing_value(value) or key in allow_empty_paths:
                    compacted[key] = value
            return compacted

        return compact_dict(patched)

    def validate_required_item(self, item: Any, arguments: Dict[str, Any]) -> List[str]:
        missing: List[str] = []

        if isinstance(item, str):
            value = self.get_argument_value(arguments, item, MISSING)
            if value is MISSING or self.is_missing_value(value):
                missing.append(item)
            return missing

        if not isinstance(item, dict):
            return missing

        if "any_of" in item:
            candidates = [str(v) for v in self.as_list(item.get("any_of")) if str(v or "").strip()]
            if not candidates:
                return missing

            present = False
            for candidate in candidates:
                value = self.get_argument_value(arguments, candidate, MISSING)
                if value is not MISSING and not self.is_missing_value(value):
                    present = True
                    break

            if not present:
                missing.append("|".join(candidates))
            return missing

        if "all_of" in item:
            for candidate in self.as_list(item.get("all_of")):
                missing.extend(self.validate_required_item(candidate, arguments))
            return missing

        key = str(item.get("name") or item.get("key") or item.get("path") or item.get("argument") or "").strip()
        aliases = [str(alias) for alias in self.as_list(item.get("aliases")) if str(alias or "").strip()]

        if not key and not aliases:
            return missing

        candidates = [key] if key else []
        candidates.extend(aliases)

        present = False
        for candidate in candidates:
            value = self.get_argument_value(arguments, candidate, MISSING)
            if value is not MISSING and not self.is_missing_value(value):
                present = True
                break

        if not present:
            missing.append(key or "|".join(aliases))

        return missing

    def validate_required(self, tool_name: str, operation: str, arguments: Dict[str, Any]) -> List[str]:
        spec = self.get_operation_spec(tool_name, operation)
        required = spec.get("required", [])

        if not isinstance(required, list):
            return []

        missing: List[str] = []

        for item in required:
            missing.extend(self.validate_required_item(item, arguments))

        return self.unique([item for item in missing if str(item or "").strip()])

    def render_value(self, value: Any, context: Dict[str, Any]) -> Any:
        if isinstance(value, dict):
            return {
                key: self.render_value(val, context)
                for key, val in value.items()
            }

        if isinstance(value, list):
            return [self.render_value(item, context) for item in value]

        if not isinstance(value, str):
            return value

        pattern = re.compile(r"{{\s*([^}]+)\s*}}")

        def replace(match: re.Match) -> str:
            path = match.group(1).strip()

            if path.startswith("env."):
                env_key = path[4:]
                return str(os.environ.get(env_key, ""))

            value_from_context = deep_get(context, path, MISSING)
            if value_from_context is MISSING:
                # Backward compatibility for {{operation}} and {{branch}} style.
                value_from_context = context.get(path, "")

            return "" if value_from_context is MISSING or value_from_context is None else str(value_from_context)

        return pattern.sub(replace, value)

    def build_body(
        self,
        tool: Dict[str, Any],
        operation: str,
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        effective = self.get_effective_operation_config(tool, operation)
        template = effective.get("body_template")

        context = {
            "operation": operation,
            "arguments": arguments,
            **arguments,
            "env": dict(os.environ),
        }

        if isinstance(template, dict):
            rendered = self.render_value(template, context)
            return rendered if isinstance(rendered, dict) else {}

        payload_mode = str(
            effective.get("payload_mode", tool.get("payload_mode", "operation_plus_arguments")) or ""
        ).strip()

        if payload_mode == "arguments_only":
            return dict(arguments or {})

        return {
            "operation": operation,
            **arguments
        }

    def build_query(
        self,
        tool: Dict[str, Any],
        operation: str,
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        effective = self.get_effective_operation_config(tool, operation)
        template = effective.get("query_template")

        if isinstance(template, dict):
            context = {
                "operation": operation,
                "arguments": arguments,
                **arguments,
                "env": dict(os.environ),
            }
            rendered = self.render_value(template, context)
            return rendered if isinstance(rendered, dict) else {}

        return self.build_body(tool, operation, arguments)

    def call(self, tool_name: str, operation: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        tool_name = str(tool_name or "").strip()
        operation = str(operation or "").strip()
        arguments = self.normalize_arguments(tool_name, operation, arguments or {})

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
                    "missing": missing,
                    "missing_inputs": missing,
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
        effective = self.get_effective_operation_config(tool, operation)
        url = str(effective.get("url") or tool.get("url") or "").strip()

        if not url:
            return self.error_result(
                error_type="tool_config_error",
                error="Tool URL is missing",
                tool_name=tool_name,
                operation=operation,
                arguments=arguments
            )

        method = str(effective.get("method", tool.get("method", "POST"))).upper().strip()

        try:
            timeout = int(effective.get("timeout_seconds", tool.get("timeout_seconds", 30)))
        except Exception:
            timeout = 30

        headers = self.build_headers(tool, operation, arguments)
        body = self.build_body(tool, operation, arguments)
        data = None
        final_url = url

        if method in ["POST", "PUT", "PATCH"]:
            headers.setdefault("Content-Type", "application/json")
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")

        elif method == "GET":
            query = urllib.parse.urlencode(compact_dict(self.build_query(tool, operation, arguments)), doseq=True)
            if query:
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
            headers=headers,
            method=method
        )

        started_at = time.time()

        try:
            status_code, content_type, raw, attempts = self.execute_http_with_retries(
                request=request,
                timeout=timeout,
                tool=tool,
                operation=operation,
            )

            parsed, parse_error = self.parse_json_response(
                raw=raw,
                tool=tool,
                operation=operation,
                content_type=content_type,
            )

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
                        "raw_text_preview": self.preview_raw(raw),
                        "attempts": attempts,
                        "duration_ms": self.duration_ms(started_at),
                    }
                )

            normalized = self.normalize_success_result(
                parsed=parsed,
                tool_name=tool_name,
                operation=operation,
                arguments=arguments,
                http_status=status_code,
                content_type=content_type,
                attempts=attempts,
                duration_ms=self.duration_ms(started_at),
            )

            return normalized

        except TransientToolHTTPError as exc:
            return self.error_result(
                error_type="http_error_after_retries",
                error=f"HTTP {exc.status_code} after retries",
                tool_name=tool_name,
                operation=operation,
                arguments=arguments,
                extra={
                    "http_status": exc.status_code,
                    "raw_text_preview": self.preview_raw(exc.raw),
                    "duration_ms": self.duration_ms(started_at),
                }
            )

        except TransientToolNetworkError as exc:
            return self.error_result(
                error_type="network_error_after_retries",
                error=str(exc.reason),
                tool_name=tool_name,
                operation=operation,
                arguments=arguments,
                extra={
                    "duration_ms": self.duration_ms(started_at),
                }
            )

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
                    "raw_text_preview": self.preview_raw(raw),
                    "duration_ms": self.duration_ms(started_at),
                }
            )

        except urllib.error.URLError as exc:
            return self.error_result(
                error_type="network_error",
                error=str(exc.reason),
                tool_name=tool_name,
                operation=operation,
                arguments=arguments,
                extra={
                    "duration_ms": self.duration_ms(started_at),
                }
            )

        except Exception as exc:
            return self.error_result(
                error_type="tool_runtime_error",
                error=f"{type(exc).__name__}: {exc}",
                tool_name=tool_name,
                operation=operation,
                arguments=arguments,
                extra={
                    "duration_ms": self.duration_ms(started_at),
                }
            )

    def build_headers(self, tool: Dict[str, Any], operation: str, arguments: Dict[str, Any]) -> Dict[str, str]:
        effective = self.get_effective_operation_config(tool, operation)
        headers = effective.get("headers", tool.get("headers", {}))

        if not isinstance(headers, dict):
            headers = {}

        context = {
            "operation": operation,
            "arguments": arguments,
            **arguments,
            "env": dict(os.environ),
        }

        rendered_headers = {
            str(key): str(self.render_value(value, context))
            for key, value in headers.items()
            if str(key or "").strip()
        }

        idempotency_key = self.build_idempotency_key(tool, operation, arguments)
        if idempotency_key:
            header_name = self.get_idempotency_header(tool, operation)
            if header_name:
                rendered_headers.setdefault(header_name, idempotency_key)

        return rendered_headers

    def get_retry_config(self, tool: Dict[str, Any], operation: str) -> Dict[str, Any]:
        effective = self.get_effective_operation_config(tool, operation)
        retry_config = effective.get("retry", {})

        if not isinstance(retry_config, dict):
            retry_config = {}

        attempts = retry_config.get("attempts", retry_config.get("max_attempts", effective.get("retry_attempts", 3)))
        try:
            attempts = int(attempts)
        except Exception:
            attempts = 3

        min_sleep = retry_config.get("min_seconds", retry_config.get("min", 0.5))
        max_sleep = retry_config.get("max_seconds", retry_config.get("max", 4))
        multiplier = retry_config.get("multiplier", 0.5)

        try:
            min_sleep = float(min_sleep)
        except Exception:
            min_sleep = 0.5

        try:
            max_sleep = float(max_sleep)
        except Exception:
            max_sleep = 4.0

        try:
            multiplier = float(multiplier)
        except Exception:
            multiplier = 0.5

        statuses = retry_config.get("http_statuses", effective.get("retryable_http_statuses", self.DEFAULT_RETRYABLE_HTTP_STATUSES))
        if not isinstance(statuses, list):
            statuses = self.DEFAULT_RETRYABLE_HTTP_STATUSES

        markers = retry_config.get("network_markers", effective.get("retryable_network_markers", self.DEFAULT_RETRYABLE_NETWORK_MARKERS))
        if not isinstance(markers, list):
            markers = self.DEFAULT_RETRYABLE_NETWORK_MARKERS

        safe_statuses = []
        for status in statuses:
            try:
                safe_statuses.append(int(status))
            except Exception:
                continue

        return {
            "attempts": max(attempts, 1),
            "min_seconds": max(min_sleep, 0.0),
            "max_seconds": max(max_sleep, 0.0),
            "multiplier": max(multiplier, 0.0),
            "http_statuses": safe_statuses,
            "network_markers": [str(item).lower() for item in markers if str(item or "").strip()],
        }

    def execute_http_with_retries(
        self,
        request: urllib.request.Request,
        timeout: int,
        tool: Dict[str, Any],
        operation: str,
    ) -> Tuple[int, str, str, int]:
        retry_config = self.get_retry_config(tool, operation)
        attempts = int(retry_config.get("attempts", 3))

        last_http_error: Optional[TransientToolHTTPError] = None
        last_network_error: Optional[TransientToolNetworkError] = None

        for attempt in range(1, attempts + 1):
            try:
                status_code, content_type, raw = self.execute_http_request_once(
                    request=request,
                    timeout=timeout,
                    retry_config=retry_config,
                )
                return status_code, content_type, raw, attempt

            except TransientToolHTTPError as exc:
                last_http_error = exc
                if attempt >= attempts:
                    raise

            except TransientToolNetworkError as exc:
                last_network_error = exc
                if attempt >= attempts:
                    raise

            sleep_seconds = min(
                float(retry_config.get("max_seconds", 4.0)),
                max(
                    float(retry_config.get("min_seconds", 0.5)),
                    float(retry_config.get("multiplier", 0.5)) * (2 ** (attempt - 1))
                )
            )

            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

        if last_http_error:
            raise last_http_error

        if last_network_error:
            raise last_network_error

        raise TransientToolNetworkError("unknown retry failure")

    def execute_http_request_once(
        self,
        request: urllib.request.Request,
        timeout: int,
        retry_config: Dict[str, Any]
    ) -> Tuple[int, str, str]:
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                status_code = int(getattr(response, "status", 200) or 200)
                content_type = response.headers.get("Content-Type", "")
                raw = response.read().decode("utf-8", errors="replace")

                return status_code, content_type, raw

        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            status_code = int(getattr(exc, "code", 0) or 0)

            if self.is_retryable_http_status(status_code, retry_config):
                raise TransientToolHTTPError(status_code=status_code, raw=raw) from exc

            raise

        except urllib.error.URLError as exc:
            if self.is_retryable_url_error(exc, retry_config):
                raise TransientToolNetworkError(reason=exc.reason) from exc

            raise

    def is_retryable_http_status(self, status_code: int, retry_config: Optional[Dict[str, Any]] = None) -> bool:
        retry_config = retry_config or {}
        statuses = retry_config.get("http_statuses", self.DEFAULT_RETRYABLE_HTTP_STATUSES)
        return status_code in set(statuses)

    def is_retryable_url_error(self, exc: urllib.error.URLError, retry_config: Optional[Dict[str, Any]] = None) -> bool:
        retry_config = retry_config or {}
        reason = getattr(exc, "reason", "")

        if reason is None:
            return True

        reason_text = str(reason).lower()
        markers = retry_config.get("network_markers", self.DEFAULT_RETRYABLE_NETWORK_MARKERS)

        return any(str(marker).lower() in reason_text for marker in markers)

    def parse_json_response(
        self,
        raw: str,
        tool: Optional[Dict[str, Any]] = None,
        operation: str = "",
        content_type: str = ""
    ) -> Tuple[Any, bool]:
        text = str(raw or "").strip().lstrip("\ufeff")

        if not text:
            return None, True

        try:
            return json.loads(text), False
        except Exception:
            pass

        tool = tool or {}
        effective = self.get_effective_operation_config(tool, operation) if tool else {}
        allow_relaxed = bool(effective.get("allow_relaxed_json", self.runner_config.get("allow_relaxed_json", False)))

        if not allow_relaxed:
            return None, True

        # Relaxed parse is still strict about producing JSON. It only trims
        # accidental wrapping text/code fences; it never fabricates a result.
        stripped = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        stripped = re.sub(r"```$", "", stripped).strip()

        start_candidates = [idx for idx in [stripped.find("{"), stripped.find("[")] if idx >= 0]
        if not start_candidates:
            return None, True

        start = min(start_candidates)
        end = max(stripped.rfind("}"), stripped.rfind("]"))

        if end < start:
            return None, True

        candidate = stripped[start:end + 1]

        try:
            return json.loads(candidate), False
        except Exception:
            return None, True

    def normalize_success_result(
        self,
        parsed: Any,
        tool_name: str,
        operation: str,
        arguments: Dict[str, Any],
        http_status: int,
        content_type: str,
        attempts: int = 1,
        duration_ms: int = 0,
    ) -> Dict[str, Any]:
        if isinstance(parsed, dict):
            result = dict(parsed)

            if result.get("ok") is False:
                result.setdefault("error_type", "tool_reported_failure")
            else:
                result.setdefault("ok", True)

        else:
            result = {
                "ok": True,
                "result": parsed
            }

        result.setdefault("tool_name", tool_name)
        result.setdefault("operation", operation)
        result.setdefault("arguments", arguments)
        result.setdefault("http_status", http_status)
        result.setdefault("content_type", content_type)
        result.setdefault("attempts", attempts)
        result.setdefault("duration_ms", duration_ms)

        spec = self.get_operation_spec(tool_name, operation)

        result_type = spec.get("result_type") or spec.get("result_kind")
        if result_type and not result.get("result_type"):
            result["result_type"] = result_type

        defaults = spec.get("result_defaults", spec.get("success_defaults", {}))
        if isinstance(defaults, dict):
            for key, value in defaults.items():
                result.setdefault(str(key), value)

        contract_error = self.validate_result_contract(
            result=result,
            tool_name=tool_name,
            operation=operation
        )

        if contract_error:
            return self.error_result(
                error_type="malformed_tool_result",
                error=contract_error,
                tool_name=tool_name,
                operation=operation,
                arguments=arguments,
                extra={
                    "http_status": http_status,
                    "content_type": content_type,
                    "result_preview": self.preview_raw(json.dumps(result, ensure_ascii=False)),
                    "attempts": attempts,
                    "duration_ms": duration_ms,
                }
            )

        return result

    def validate_result_contract(self, result: Dict[str, Any], tool_name: str, operation: str) -> str:
        spec = self.get_operation_spec(tool_name, operation)
        contract = spec.get("result_contract", {})
        required = spec.get("required_result_fields", [])

        if isinstance(contract, dict):
            contract_required = contract.get("required", [])
            if isinstance(contract_required, list):
                required = self.as_list(required) + contract_required

        if not isinstance(required, list):
            return ""

        missing: List[str] = []
        for path in required:
            path_text = str(path or "").strip()
            if not path_text:
                continue

            value = deep_get(result, path_text, MISSING)
            if value is MISSING or self.is_missing_value(value):
                missing.append(path_text)

        if missing:
            return f"Tool result missing required fields: {', '.join(missing)}"

        return ""

    def get_sensitive_keys(self) -> List[str]:
        configured = self.runner_config.get("sensitive_keys", [])
        if not isinstance(configured, list):
            configured = []

        return self.unique([
            str(item).lower()
            for item in self.DEFAULT_SENSITIVE_KEYS + configured
            if str(item or "").strip()
        ])

    def sanitize_value(self, key: str, value: Any) -> Any:
        key_l = str(key or "").lower()
        sensitive_keys = self.get_sensitive_keys()

        if any(marker in key_l for marker in sensitive_keys):
            return "[redacted]"

        if isinstance(value, dict):
            return {
                k: self.sanitize_value(str(k), v)
                for k, v in value.items()
            }

        if isinstance(value, list):
            return [self.sanitize_value(key, item) for item in value]

        return value

    def sanitize_arguments(self, arguments: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not isinstance(arguments, dict):
            return {}

        return {
            key: self.sanitize_value(str(key), value)
            for key, value in arguments.items()
        }

    def error_result(
        self,
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
            "arguments": self.sanitize_arguments(arguments or {}),
            "result_type": "tool_error",
        }

        if isinstance(extra, dict):
            result.update(extra)

        return result

    def preview_raw(self, raw: str, max_chars: Optional[int] = None) -> str:
        if max_chars is None:
            try:
                max_chars = int(self.runner_config.get("raw_preview_chars", 600))
            except Exception:
                max_chars = 600

        text = str(raw or "").strip()

        if len(text) <= max_chars:
            return text

        return text[:max_chars].rstrip() + "...[trimmed]"

    @staticmethod
    def duration_ms(started_at: float) -> int:
        return max(int((time.time() - started_at) * 1000), 0)

    def get_idempotency_header(self, tool: Dict[str, Any], operation: str) -> str:
        effective = self.get_effective_operation_config(tool, operation)
        header_name = effective.get("idempotency_header")

        if header_name:
            return str(header_name)

        idempotency = effective.get("idempotency", {})
        if isinstance(idempotency, dict) and idempotency.get("header"):
            return str(idempotency.get("header"))

        return ""

    def build_idempotency_key(self, tool: Dict[str, Any], operation: str, arguments: Dict[str, Any]) -> str:
        effective = self.get_effective_operation_config(tool, operation)
        idempotency = effective.get("idempotency", {})

        if not isinstance(idempotency, dict):
            idempotency = {}

        enabled = bool(idempotency.get("enabled", False) or effective.get("idempotency_enabled", False))
        if not enabled:
            return ""

        fields = (
            idempotency.get("fields")
            or effective.get("idempotency_key_fields")
            or []
        )

        if not isinstance(fields, list):
            fields = []

        material: Dict[str, Any] = {
            "tool": tool.get("name", ""),
            "operation": operation,
        }

        if fields:
            material["arguments"] = {
                str(path): self.get_argument_value(arguments, str(path), "")
                for path in fields
            }
        else:
            material["arguments"] = arguments

        raw = json.dumps(material, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()
