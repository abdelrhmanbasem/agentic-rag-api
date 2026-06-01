import json
import os
import time
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI


OPENAI_MODEL = os.getenv("AGENTIC_BRAIN_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))


class AgenticBrainError(Exception):
    pass


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


def deep_get(data: Dict[str, Any], path: str, default: Any = None) -> Any:
    current: Any = data

    for part in path.split("."):
        if not isinstance(current, dict):
            return default
        current = current.get(part)

    return current if current is not None else default


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


def normalize_tool_manifest(assistant_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    tools = assistant_config.get("tools")

    if not isinstance(tools, list):
        tools = deep_get(assistant_config, "agent_config.tools", [])

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

    missing = []
    for key in required:
        value = arguments.get(key)
        if value is None or value == "" or value == [] or value == {}:
            missing.append(key)

    return missing


def render_template(value: Any, context: Dict[str, Any]) -> Any:
    if not isinstance(value, str):
        return value

    rendered = value

    for key, val in context.items():
        token = "{{" + key + "}}"
        if token in rendered:
            rendered = rendered.replace(token, str(val))

    return rendered


def call_http_tool(tool: Dict[str, Any], operation: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    url = tool.get("url", "")
    if not url:
        return {
            "ok": False,
            "error": "Tool URL is missing",
            "operation": operation
        }

    method = str(tool.get("method", "POST")).upper()
    timeout = int(tool.get("timeout_seconds", 30))

    headers = tool.get("headers", {})
    if not isinstance(headers, dict):
        headers = {}

    rendered_headers = {}
    context = {
        "operation": operation,
        **arguments,
        **os.environ
    }

    for key, value in headers.items():
        rendered_headers[key] = render_template(value, context)

    body = {
        "operation": operation,
        **arguments
    }

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
    safe_tools = []

    for tool in tools:
        operations = tool.get("operations", {})
        safe_operations = {}

        if isinstance(operations, dict):
            for op_name, spec in operations.items():
                if not isinstance(spec, dict):
                    spec = {}

                safe_operations[op_name] = {
                    "description": spec.get("description", ""),
                    "required": spec.get("required", []),
                    "optional": spec.get("optional", []),
                    "updates": spec.get("updates", []),
                    "does_not_update": spec.get("does_not_update", [])
                }

        safe_tools.append({
            "name": tool.get("name"),
            "description": tool.get("description", ""),
            "operations": safe_operations
        })

    return safe_tools


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

    return f"""
You are the Agentic Brain for this assistant.

Your job:
- Understand the user's latest message.
- Use current variables only when they are relevant and not stale.
- Decide the next step.
- Call tools when external data is needed.
- Update variables explicitly.
- Clear stale variables explicitly.
- Reply naturally to the user.

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

Critical rules:
1. Never invent tool results.
2. If you need information from a tool, return action="call_tool".
3. If required tool inputs are missing, return action="ask_user" with a clear question.
4. Do not copy stale variables into new answers.
5. Only update variables using variable_updates.
6. If a new user selection conflicts with old variables, clear the old conflicting variables.
7. If a tool operation returns a list, such as list_branches, do not set selected_branch unless the user selected one.
8. If a tool result says branch_found=false, do not invent a branch. Ask the user to choose or clarify.
9. If the user asks for slots/availability, you need a confirmed branch and a date/day.
10. If user gives a relative date/day like بكرة, الخميس الجاي, 4 يونيو, pass it to the tool as date_text. The tool should normalize it.
11. If user confirms a booking, do not create booking unless all required fields are present and the user clearly confirmed the exact slot.
12. Your final answer must be customer-facing only. Do not mention JSON, tools, variables, operations, or internal logic.

You must respond with valid JSON only.

Allowed JSON shape:
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


def apply_variable_patch(
    variables: Dict[str, Any],
    variable_updates: Dict[str, Any],
    clear_variables: List[str]
) -> Dict[str, Any]:
    patched = dict(variables or {})

    if isinstance(clear_variables, list):
        for key in clear_variables:
            if isinstance(key, str):
                patched.pop(key, None)

    if isinstance(variable_updates, dict):
        for key, value in variable_updates.items():
            if value is None or value == "":
                continue
            patched[key] = value

    return patched


def extract_history(conversation: Dict[str, Any]) -> List[Dict[str, str]]:
    messages = conversation.get("messages", [])
    if not isinstance(messages, list):
        return []

    out = []
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

        if "action" not in decision:
            decision["action"] = "reply"

        decision.setdefault("answer", "")
        decision.setdefault("tool_name", "")
        decision.setdefault("operation", "")
        decision.setdefault("arguments", {})
        decision.setdefault("variable_updates", {})
        decision.setdefault("clear_variables", [])
        decision.setdefault("confidence", 0)

        if not isinstance(decision.get("arguments"), dict):
            decision["arguments"] = {}

        if not isinstance(decision.get("variable_updates"), dict):
            decision["variable_updates"] = {}

        if not isinstance(decision.get("clear_variables"), list):
            decision["clear_variables"] = []

        return decision

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
            variables = {
                **variables,
                **incoming_variables
            }

        tools = normalize_tool_manifest(assistant_config)
        history = extract_history(conversation)
        observations: List[Dict[str, Any]] = []

        tool_calls_used = 0
        final_decision: Dict[str, Any] = {}

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
                    "tool_calls_used": tool_calls_used
                }

            if action != "call_tool":
                return {
                    "answer": decision.get("answer") or "ممكن توضحلي أكتر؟",
                    "variables": variables,
                    "action": "reply",
                    "observations": observations,
                    "decision": decision,
                    "tool_calls_used": tool_calls_used
                }

            if tool_calls_used >= max_tool_calls:
                return {
                    "answer": "محتاج أتأكد من البيانات الأول. ممكن تعيد طلبك بشكل أوضح؟",
                    "variables": variables,
                    "action": "reply",
                    "observations": observations,
                    "decision": decision,
                    "tool_calls_used": tool_calls_used
                }

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
                continue

            operation_spec = get_operation_spec(tool, operation)

            if not operation_spec:
                observations.append({
                    "type": "tool_error",
                    "tool_name": tool_name,
                    "operation": operation,
                    "error": "Operation not found in tool manifest"
                })
                continue

            missing = missing_required_inputs(operation_spec, arguments)

            if missing:
                observations.append({
                    "type": "tool_validation_error",
                    "tool_name": tool_name,
                    "operation": operation,
                    "missing_inputs": missing,
                    "arguments": arguments
                })
                continue

            tool_calls_used += 1

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

            time.sleep(0.1)

        return {
            "answer": final_decision.get("answer") or "محتاج تفاصيل أكتر عشان أساعدك.",
            "variables": variables,
            "action": "reply",
            "observations": observations,
            "decision": final_decision,
            "tool_calls_used": tool_calls_used
        }
