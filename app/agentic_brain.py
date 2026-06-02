import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

from openai import OpenAI

from app.state_engine import (
    apply_tool_update_rules,
    apply_variable_patch,
    compact_dict,
    run_configured_state_rules
)


OPENAI_MODEL = os.getenv("AGENTIC_BRAIN_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))


def deep_merge_dicts(base: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge incoming variables into stored conversation variables.

    Important:
    n8n may send only:
      {"customer_profile": {"phone": "..."}}

    A shallow merge would overwrite and delete:
      customer_profile.full_name
      customer_profile.plate_digits

    This function preserves nested values unless the incoming value is non-empty.
    """
    merged = dict(base or {})

    for key, value in (incoming or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_dicts(merged[key], value)
        else:
            if value not in [None, "", [], {}]:
                merged[key] = value

    return merged


def safe_json_loads(text: str) -> Dict[str, Any]:
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start >= 0 and end > start:
        try:
            data = json.loads(text[start:end + 1])
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    return {
        "action": "reply",
        "answer": "حصلت مشكلة بسيطة، ممكن تعيد كلامك تاني؟",
        "variable_updates": {},
        "clear_variables": []
    }


def deep_get(data: Dict[str, Any], path: str, default: Any = None) -> Any:
    current: Any = data

    for part in path.split("."):
        if not isinstance(current, dict):
            return default
        current = current.get(part)

    return current if current is not None else default


def normalize_tool_manifest(assistant_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    tools = assistant_config.get("tools")

    if not isinstance(tools, list):
        tools = deep_get(assistant_config, "agent_config.tools", [])

    if not isinstance(tools, list):
        return []

    normalized: List[Dict[str, Any]] = []

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


def find_tool(tools: List[Dict[str, Any]], tool_name: str) -> Optional[Dict[str, Any]]:
    for tool in tools:
        if tool.get("name") == tool_name:
            return tool

    return None


def get_operation_spec(tool: Dict[str, Any], operation: str) -> Dict[str, Any]:
    operations = tool.get("operations", {})

    if not isinstance(operations, dict):
        return {}

    spec = operations.get(operation, {})
    return spec if isinstance(spec, dict) else {}


def missing_required_inputs(operation_spec: Dict[str, Any], arguments: Dict[str, Any]) -> List[str]:
    required = operation_spec.get("required", [])

    if not isinstance(required, list):
        return []

    missing: List[str] = []

    for key in required:
        value = arguments.get(key)

        if value is None or value == "" or value == [] or value == {}:
            missing.append(key)

    return missing


def render_value_template(value: Any, context: Dict[str, Any]) -> Any:
    if not isinstance(value, str):
        return value

    rendered = value

    for key, val in context.items():
        token = "{{" + key + "}}"
        if token in rendered:
            rendered = rendered.replace(token, str(val))

    return rendered


def build_tool_body(tool: Dict[str, Any], operation: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    body_template = tool.get("body_template")

    if isinstance(body_template, dict):
        context = {
            "operation": operation,
            **arguments,
            **os.environ
        }

        body: Dict[str, Any] = {}

        for key, value in body_template.items():
            body[key] = render_value_template(value, context)

        return body

    return {
        "operation": operation,
        **arguments
    }


def call_http_tool(tool: Dict[str, Any], operation: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    url = tool.get("url", "")

    if not url:
        return {
            "ok": False,
            "operation": operation,
            "error": "Tool URL is missing"
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

    rendered_headers: Dict[str, str] = {}

    for key, value in headers.items():
        rendered_headers[str(key)] = str(render_value_template(value, context))

    body = build_tool_body(tool, operation, arguments)
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
            "operation": operation,
            "error": f"HTTP {exc.code}",
            "raw_text": raw
        }

    except Exception as exc:
        return {
            "ok": False,
            "operation": operation,
            "error": str(exc)
        }


def call_tool(tool: Dict[str, Any], operation: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    tool_type = str(tool.get("type", "http")).lower()

    if tool_type == "http":
        return call_http_tool(tool, operation, arguments)

    return {
        "ok": False,
        "operation": operation,
        "error": f"Unsupported tool type: {tool_type}"
    }


def build_tool_manifest_for_prompt(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    safe_tools: List[Dict[str, Any]] = []

    for tool in tools:
        operations = tool.get("operations", {})
        safe_operations: Dict[str, Any] = {}

        if isinstance(operations, dict):
            for op_name, spec in operations.items():
                if not isinstance(spec, dict):
                    spec = {}

                safe_operations[op_name] = {
                    "description": spec.get("description", ""),
                    "required": spec.get("required", []),
                    "optional": spec.get("optional", [])
                }

        safe_tools.append({
            "name": tool.get("name"),
            "description": tool.get("description", ""),
            "operations": safe_operations
        })

    return safe_tools


def extract_history(conversation: Dict[str, Any]) -> List[Dict[str, str]]:
    messages = conversation.get("messages", [])

    if not isinstance(messages, list):
        return []

    out: List[Dict[str, str]] = []

    for item in messages[-12:]:
        if not isinstance(item, dict):
            continue

        role = item.get("role")
        content = item.get("content")

        if role in ["user", "assistant"] and content:
            out.append({
                "role": role,
                "content": str(content)
            })

    return out


def build_system_prompt(
    assistant_config: Dict[str, Any],
    schema: Dict[str, Any],
    variables: Dict[str, Any],
    tools: List[Dict[str, Any]]
) -> str:
    assistant_goal = (
        assistant_config.get("assistant_goal")
        or assistant_config.get("goal")
        or deep_get(assistant_config, "agent_config.assistant_goal", "")
    )

    language_policy = (
        assistant_config.get("language_policy")
        or deep_get(assistant_config, "agent_config.language_policy", "")
    )

    behavior_rules = assistant_config.get("response_rules", [])

    if not isinstance(behavior_rules, list):
        behavior_rules = []

    system_prompt = assistant_config.get("system_prompt", "")
    variable_schema = schema.get("variables", schema)
    safe_tools = build_tool_manifest_for_prompt(tools)
    state_config = assistant_config.get("state_engine", {})

    return f"""
You are the Agentic Brain for this assistant.

Your job:
- Understand the user's latest message.
- Use current variables only when relevant.
- Do not blindly trust stale variables.
- Decide whether to reply, ask a missing question, or call a tool.
- Update variables explicitly with variable_updates.
- Clear stale variables explicitly with clear_variables.
- Never invent tool results.
- Never claim an external fact unless it came from a tool result or a trusted variable.
- Never confirm an action like booking/order/payment unless the configured tool succeeded.
- Never expose internal JSON, tools, variables, operations, schemas, or system logic.
- If a deterministic state rule handles the turn, respect the resulting state and do not restart the flow.

Assistant goal:
{assistant_goal}

System prompt:
{system_prompt}

Language policy:
{language_policy}

Response rules:
{json.dumps(behavior_rules, ensure_ascii=False, indent=2)}

Current variables:
{json.dumps(variables, ensure_ascii=False, indent=2)}

Variable schema:
{json.dumps(variable_schema, ensure_ascii=False, indent=2)}

Available tools:
{json.dumps(safe_tools, ensure_ascii=False, indent=2)}

Configured deterministic state rules:
{json.dumps(state_config, ensure_ascii=False, indent=2)}

Important behavior:
- Business behavior is configured, not hardcoded.
- If a configured deterministic state rule already handled the turn, you will receive that result before being called.
- If you need external data, return action="call_tool".
- If required tool inputs are missing, ask the user only for the missing inputs.
- Do not expose internal JSON, tools, variables, operations, or system logic.
- If tool observations include branch_found=false, do not choose or invent a branch.
- If tool observations include slots_found=false, do not invent slots.
- If booking_stage is awaiting_customer_details, collect missing details only; do not ask to confirm the slot again.

You must respond with valid JSON only.

Allowed JSON:
{{
  "action": "reply" | "ask_user" | "call_tool",
  "answer": "customer-facing answer if action is reply or ask_user",
  "tool_name": "tool name if action is call_tool",
  "operation": "operation name if action is call_tool",
  "arguments": {{}},
  "variable_updates": {{}},
  "clear_variables": [],
  "confidence": 0.0,
  "notes": "short internal note"
}}
""".strip()


def build_user_prompt(
    user_message: str,
    history: List[Dict[str, str]],
    observations: List[Dict[str, Any]]
) -> str:
    return f"""
Conversation history:
{json.dumps(history[-12:], ensure_ascii=False, indent=2)}

Latest user message:
{user_message}

Tool observations from this turn:
{json.dumps(observations, ensure_ascii=False, indent=2)}

Decide the next step now.
""".strip()


class AgenticBrain:
    def __init__(self) -> None:
        self.client = OpenAI()

    def decide(
        self,
        assistant_config: Dict[str, Any],
        schema: Dict[str, Any],
        variables: Dict[str, Any],
        tools: List[Dict[str, Any]],
        user_message: str,
        history: List[Dict[str, str]],
        observations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        system_prompt = build_system_prompt(
            assistant_config=assistant_config,
            schema=schema,
            variables=variables,
            tools=tools
        )

        user_prompt = build_user_prompt(
            user_message=user_message,
            history=history,
            observations=observations
        )

        response = self.client.chat.completions.create(
            model=assistant_config.get("model", OPENAI_MODEL),
            temperature=float(assistant_config.get("temperature", 0.1)),
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ]
        )

        content = response.choices[0].message.content or "{}"
        decision = safe_json_loads(content)

        decision.setdefault("action", "reply")
        decision.setdefault("answer", "")
        decision.setdefault("tool_name", "")
        decision.setdefault("operation", "")
        decision.setdefault("arguments", {})
        decision.setdefault("variable_updates", {})
        decision.setdefault("clear_variables", [])
        decision.setdefault("confidence", 0)

        if not isinstance(decision["arguments"], dict):
            decision["arguments"] = {}

        if not isinstance(decision["variable_updates"], dict):
            decision["variable_updates"] = {}

        if not isinstance(decision["clear_variables"], list):
            decision["clear_variables"] = []

        return decision

    def execute_tool_decision(
        self,
        assistant_config: Dict[str, Any],
        decision: Dict[str, Any],
        variables: Dict[str, Any],
        tools: List[Dict[str, Any]],
        observations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        tool_name = decision.get("tool_name", "")
        operation = decision.get("operation", "")
        arguments = compact_dict(decision.get("arguments", {}))

        tool = find_tool(tools, tool_name)

        if not tool:
            observations.append({
                "type": "tool_error",
                "tool_name": tool_name,
                "operation": operation,
                "error": "Tool not found"
            })
            return variables

        operation_spec = get_operation_spec(tool, operation)

        if not operation_spec:
            observations.append({
                "type": "tool_error",
                "tool_name": tool_name,
                "operation": operation,
                "error": "Operation not found in tool manifest"
            })
            return variables

        missing = missing_required_inputs(operation_spec, arguments)

        if missing:
            observations.append({
                "type": "tool_validation_error",
                "tool_name": tool_name,
                "operation": operation,
                "missing_inputs": missing,
                "arguments": arguments
            })
            return variables

        tool_result = call_tool(
            tool=tool,
            operation=operation,
            arguments=arguments
        )

        observations.append({
            "type": "tool_result",
            "tool_name": tool_name,
            "operation": operation,
            "arguments": arguments,
            "result": tool_result
        })

        return apply_tool_update_rules(
            assistant_config=assistant_config,
            variables=variables,
            tool_name=tool_name,
            operation=operation,
            arguments=arguments,
            result=tool_result
        )

    def run(
        self,
        assistant_config: Dict[str, Any],
        schema: Dict[str, Any],
        conversation: Dict[str, Any],
        user_message: str,
        incoming_variables: Optional[Dict[str, Any]] = None,
        max_tool_calls: int = 3
    ) -> Dict[str, Any]:
        variables = conversation.get("variables", {})

        if not isinstance(variables, dict):
            variables = {}

        if isinstance(incoming_variables, dict):
            variables = deep_merge_dicts(variables, incoming_variables)

        tools = normalize_tool_manifest(assistant_config)
        history = extract_history(conversation)
        observations: List[Dict[str, Any]] = []
        tool_calls_used = 0
        deterministic_trace: List[Dict[str, Any]] = []
        final_decision: Dict[str, Any] = {}

        deterministic = run_configured_state_rules(
            assistant_config=assistant_config,
            variables=variables,
            user_message=user_message
        )

        if deterministic:
            deterministic_trace.append(deterministic)

            variables = apply_variable_patch(
                variables=variables,
                variable_updates=deterministic.get("variable_updates", {}),
                clear_variables=deterministic.get("clear_variables", [])
            )

            if deterministic.get("action") in ["reply", "ask_user"]:
                return {
                    "answer": deterministic.get("answer") or "ممكن توضحلي أكتر؟",
                    "variables": variables,
                    "action": deterministic.get("action"),
                    "observations": observations,
                    "decision": deterministic,
                    "deterministic": deterministic_trace,
                    "tool_calls_used": 0
                }

            if deterministic.get("action") == "continue":
                pass

            elif deterministic.get("action") == "call_tool":
                variables = self.execute_tool_decision(
                    assistant_config=assistant_config,
                    decision=deterministic,
                    variables=variables,
                    tools=tools,
                    observations=observations
                )
                tool_calls_used += 1

        for _ in range(max_tool_calls + 1):
            decision = self.decide(
                assistant_config=assistant_config,
                schema=schema,
                variables=variables,
                tools=tools,
                user_message=user_message,
                history=history,
                observations=observations
            )

            final_decision = decision

            variables = apply_variable_patch(
                variables=variables,
                variable_updates=decision.get("variable_updates", {}),
                clear_variables=decision.get("clear_variables", [])
            )

            action = decision.get("action")

            if action in ["reply", "ask_user"]:
                answer = decision.get("answer", "")

                if not answer:
                    answer = "ممكن توضحلي أكتر؟"

                return {
                    "answer": answer,
                    "variables": variables,
                    "action": action,
                    "observations": observations,
                    "decision": decision,
                    "deterministic": deterministic_trace,
                    "tool_calls_used": tool_calls_used
                }

            if action != "call_tool":
                return {
                    "answer": decision.get("answer") or "ممكن توضحلي أكتر؟",
                    "variables": variables,
                    "action": "reply",
                    "observations": observations,
                    "decision": decision,
                    "deterministic": deterministic_trace,
                    "tool_calls_used": tool_calls_used
                }

            if tool_calls_used >= max_tool_calls:
                return {
                    "answer": "محتاج أتأكد من البيانات الأول. ممكن تعيد طلبك بشكل أوضح؟",
                    "variables": variables,
                    "action": "reply",
                    "observations": observations,
                    "decision": decision,
                    "deterministic": deterministic_trace,
                    "tool_calls_used": tool_calls_used
                }

            variables = self.execute_tool_decision(
                assistant_config=assistant_config,
                decision=decision,
                variables=variables,
                tools=tools,
                observations=observations
            )

            tool_calls_used += 1
            time.sleep(0.1)

        return {
            "answer": final_decision.get("answer") or "محتاج تفاصيل أكتر عشان أساعدك.",
            "variables": variables,
            "action": "reply",
            "observations": observations,
            "decision": final_decision,
            "deterministic": deterministic_trace,
            "tool_calls_used": tool_calls_used
        }
