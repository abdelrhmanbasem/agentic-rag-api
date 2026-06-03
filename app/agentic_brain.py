import copy
import json
import os
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from openai import OpenAI

from app.response_composer import ResponseComposer
from app.tool_runner import ToolRunner
from app.subagents.base import (
    SubagentContext,
    apply_variable_patch,
    apply_tool_update_rules,
    deep_merge,
    get_subagent_variable_scope
)
from app.subagents.handoff_subagent import HandoffSubagent
from app.subagents.location_subagent import LocationSubagent
from app.subagents.booking_subagent import BookingSubagent
from app.subagents.lookup_subagent import LookupSubagent
from app.subagents.troubleshooting_subagent import TroubleshootingSubagent


OPENAI_MODEL = os.getenv("AGENTIC_BRAIN_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))


class AgenticBrain:
    def __init__(self) -> None:
        self.client = OpenAI()
        self.response_composer = ResponseComposer()

        self.subagents = [
            HandoffSubagent(),
            LocationSubagent(),
            BookingSubagent(),
            LookupSubagent(),
            TroubleshootingSubagent()
        ]

    def run(
        self,
        assistant_config: Dict[str, Any],
        schema: Dict[str, Any],
        conversation: Dict[str, Any],
        user_message: str,
        incoming_variables: Optional[Dict[str, Any]] = None,
        max_tool_calls: int = 4
    ) -> Dict[str, Any]:
        variables = conversation.get("variables", {})

        if not isinstance(variables, dict):
            variables = {}

        variables = copy.deepcopy(variables)

        if isinstance(incoming_variables, dict):
            variables = deep_merge(variables, incoming_variables)

        variables = self.attach_conversation_metadata(
            variables=variables,
            conversation=conversation
        )

        state_before = copy.deepcopy(variables)

        history = self.extract_history(conversation)
        tool_runner = ToolRunner(assistant_config)
        observations: List[Dict[str, Any]] = []

        trace: Dict[str, Any] = {
            "message": user_message,
            "state_before": copy.deepcopy(state_before),
            "subagent_attempts": [],
            "selected_subagent": "",
            "observations": [],
            "llm_decision": {},
            "response_composer": {},
            "state_after": {},
            "final_answer": ""
        }

        for subagent in self.ordered_subagents(assistant_config):
            subagent_name = getattr(subagent, "name", "unknown")

            scoped_variables = get_subagent_variable_scope(
                assistant_config=assistant_config,
                subagent_name=subagent_name,
                variables=variables
            )

            context = SubagentContext(
                assistant_config=assistant_config,
                schema=schema,
                variables=scoped_variables,
                user_message=user_message,
                history=history,
                tool_runner=tool_runner,
                observations=observations,
                max_tool_calls=max_tool_calls
            )

            result = subagent.run(context)

            trace["subagent_attempts"].append({
                "subagent": subagent_name,
                "handled": result.handled,
                "notes": result.notes
            })

            if not result.handled:
                if result.variable_updates:
                    variables = apply_variable_patch(
                        variables,
                        result.variable_updates,
                        result.clear_variables
                    )

                continue

            variables = self.merge_subagent_result(
                variables=variables,
                result=result
            )

            if result.observations:
                observations.extend(result.observations)

            selected_subagent = result.selected_subagent or subagent_name
            state_after = copy.deepcopy(variables)

            composer_output = self.compose_customer_answer(
                assistant_config=assistant_config,
                user_message=user_message,
                variables_before=state_before,
                variables_after=state_after,
                subagent_result=result,
                selected_subagent=selected_subagent,
                observations=observations,
                debug=bool(conversation.get("debug", False))
            )

            final_answer = composer_output.get("answer", "") or result.answer or ""

            trace["selected_subagent"] = selected_subagent
            trace["observations"] = observations
            trace["response_composer"] = self.public_composer_trace(composer_output)
            trace["state_after"] = state_after
            trace["final_answer"] = final_answer

            return {
                "answer": final_answer,
                "variables": variables,
                "action": result.action,
                "tool_calls_used": result.tool_calls_used,
                "trace": trace
            }

        fallback = self.run_llm_fallback(
            assistant_config=assistant_config,
            schema=schema,
            variables=variables,
            user_message=user_message,
            history=history,
            tool_runner=tool_runner,
            max_tool_calls=max_tool_calls
        )

        variables = fallback.get("variables", variables)
        state_after = copy.deepcopy(variables)

        fallback_result = SimpleNamespace(
            handled=True,
            action=fallback.get("action", "reply"),
            answer=fallback.get("answer", ""),
            variable_updates={},
            clear_variables=[],
            observations=fallback.get("observations", []),
            selected_subagent="llm_fallback",
            tool_calls_used=fallback.get("tool_calls_used", 0),
            notes="llm fallback generated decision"
        )

        composer_output = self.compose_customer_answer(
            assistant_config=assistant_config,
            user_message=user_message,
            variables_before=state_before,
            variables_after=state_after,
            subagent_result=fallback_result,
            selected_subagent="llm_fallback",
            observations=fallback.get("observations", []),
            debug=bool(conversation.get("debug", False))
        )

        final_answer = composer_output.get("answer", "") or fallback.get("answer", "")

        trace["selected_subagent"] = "llm_fallback"
        trace["observations"] = fallback.get("observations", [])
        trace["llm_decision"] = fallback.get("decision", {})
        trace["response_composer"] = self.public_composer_trace(composer_output)
        trace["state_after"] = state_after
        trace["final_answer"] = final_answer

        return {
            "answer": final_answer,
            "variables": variables,
            "action": fallback.get("action", "reply"),
            "tool_calls_used": fallback.get("tool_calls_used", 0),
            "trace": trace
        }

    def compose_customer_answer(
        self,
        *,
        assistant_config: Dict[str, Any],
        user_message: str,
        variables_before: Dict[str, Any],
        variables_after: Dict[str, Any],
        subagent_result: Any,
        selected_subagent: str,
        observations: List[Dict[str, Any]],
        debug: bool
    ) -> Dict[str, Any]:
        return self.response_composer.compose(
            assistant_config=assistant_config,
            user_message=user_message,
            variables_before=copy.deepcopy(variables_before),
            variables_after=copy.deepcopy(variables_after),
            subagent_result=subagent_result,
            selected_subagent=selected_subagent,
            observations=copy.deepcopy(observations),
            llm_client=None,
            debug=debug
        )

    @staticmethod
    def public_composer_trace(composer_output: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(composer_output, dict):
            return {}

        output = {
            "used": composer_output.get("used_composer"),
            "reason": composer_output.get("reason")
        }

        if composer_output.get("composer_packet") is not None:
            output["packet"] = composer_output.get("composer_packet")

        return output

    @staticmethod
    def attach_conversation_metadata(
        variables: Dict[str, Any],
        conversation: Dict[str, Any]
    ) -> Dict[str, Any]:
        variables = copy.deepcopy(variables)

        for key in ["conversation_id", "user_id", "channel"]:
            value = conversation.get(key)

            if value and not variables.get(key):
                variables[key] = value

        return variables

    @staticmethod
    def merge_subagent_result(variables: Dict[str, Any], result: Any) -> Dict[str, Any]:
        updates = result.variable_updates if isinstance(result.variable_updates, dict) else {}
        clear = result.clear_variables if isinstance(result.clear_variables, list) else []

        return apply_variable_patch(
            variables=variables,
            updates=updates,
            clear=clear
        )

    def ordered_subagents(self, assistant_config: Dict[str, Any]) -> List[Any]:
        configured_order = (
            assistant_config.get("subagent_order")
            or ["handoff", "location", "lookup", "booking", "troubleshooting"]
        )

        by_name = {
            getattr(agent, "name", ""): agent
            for agent in self.subagents
        }

        ordered = []

        for name in configured_order:
            agent = by_name.get(name)

            if agent:
                ordered.append(agent)

        for agent in self.subagents:
            if agent not in ordered:
                ordered.append(agent)

        return ordered

    def run_llm_fallback(
        self,
        assistant_config: Dict[str, Any],
        schema: Dict[str, Any],
        variables: Dict[str, Any],
        user_message: str,
        history: List[Dict[str, str]],
        tool_runner: ToolRunner,
        max_tool_calls: int
    ) -> Dict[str, Any]:
        observations: List[Dict[str, Any]] = []
        tool_calls_used = 0
        decision: Dict[str, Any] = {}

        fallback_messages = assistant_config.get("fallback_messages", {})
        empty_answer = fallback_messages.get("empty_answer", "")
        max_tool_answer = fallback_messages.get("max_tool_calls", "")
        default_final = fallback_messages.get("default_final", "")

        for _ in range(max_tool_calls + 1):
            decision = self.llm_decide(
                assistant_config=assistant_config,
                schema=schema,
                variables=variables,
                user_message=user_message,
                history=history,
                observations=observations
            )

            variables = apply_variable_patch(
                variables=variables,
                updates=decision.get("variable_updates", {}),
                clear=decision.get("clear_variables", [])
            )

            action = decision.get("action", "reply")

            if action in ["reply", "ask_user"]:
                answer = decision.get("answer", "") or empty_answer or default_final

                return {
                    "answer": answer,
                    "variables": variables,
                    "action": action,
                    "observations": observations,
                    "decision": decision,
                    "tool_calls_used": tool_calls_used
                }

            if action != "call_tool":
                answer = decision.get("answer", "") or default_final

                return {
                    "answer": answer,
                    "variables": variables,
                    "action": "reply",
                    "observations": observations,
                    "decision": decision,
                    "tool_calls_used": tool_calls_used
                }

            if tool_calls_used >= max_tool_calls:
                return {
                    "answer": max_tool_answer or default_final,
                    "variables": variables,
                    "action": "reply",
                    "observations": observations,
                    "decision": decision,
                    "tool_calls_used": tool_calls_used
                }

            tool_name = decision.get("tool_name", "")
            operation = decision.get("operation", "")
            arguments = decision.get("arguments", {})

            if not isinstance(arguments, dict):
                arguments = {}

            tool_result = tool_runner.call(
                tool_name=tool_name,
                operation=operation,
                arguments=arguments
            )

            observations.append({
                "source": "llm_fallback",
                "subagent": "llm_fallback",
                "tool_name": tool_name,
                "operation": operation,
                "arguments": arguments,
                "result": tool_result
            })

            variables = apply_tool_update_rules(
                assistant_config=assistant_config,
                variables=variables,
                operation=operation,
                arguments=arguments,
                result=tool_result
            )

            tool_calls_used += 1

        return {
            "answer": default_final,
            "variables": variables,
            "action": "reply",
            "observations": observations,
            "decision": decision,
            "tool_calls_used": tool_calls_used
        }

    def llm_decide(
        self,
        assistant_config: Dict[str, Any],
        schema: Dict[str, Any],
        variables: Dict[str, Any],
        user_message: str,
        history: List[Dict[str, str]],
        observations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        prompt = self.build_system_prompt(
            assistant_config=assistant_config,
            schema=schema,
            variables=variables
        )

        user_prompt = {
            "history": history[-12:],
            "latest_user_message": user_message,
            "tool_observations_this_turn": observations
        }

        response = self.client.chat.completions.create(
            model=assistant_config.get("model", OPENAI_MODEL),
            temperature=float(assistant_config.get("temperature", 0.05)),
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": prompt
                },
                {
                    "role": "user",
                    "content": json.dumps(user_prompt, ensure_ascii=False)
                }
            ]
        )

        content = response.choices[0].message.content or "{}"
        return self.safe_json_loads(content)

    def build_system_prompt(
        self,
        assistant_config: Dict[str, Any],
        schema: Dict[str, Any],
        variables: Dict[str, Any]
    ) -> str:
        exposed_tools = []

        for tool in assistant_config.get("tools", []):
            if not isinstance(tool, dict):
                continue

            operations = tool.get("operations", {})
            exposed_tools.append({
                "name": tool.get("name"),
                "description": tool.get("description", ""),
                "operations": operations
            })

        return f"""
You are the fallback reasoning brain for this assistant.

Use configured subagents first when applicable. You are only called when no subagent handled the turn.

Rules:
- Do not expose tools, JSON, variables, schemas, or internal logic.
- Do not invent tool results.
- If external data is needed, return action="call_tool".
- If required information is missing, return action="ask_user".
- Only update variables explicitly with variable_updates.
- Clear stale variables with clear_variables.
- Never confirm bookings, payments, orders, or irreversible actions unless a tool result confirms success.
- If a tool observation says ok=false, handle it safely and do not hallucinate.
- If a tool observation says branch_found=false, do not invent or choose a branch.
- If a tool observation says slots_found=false, do not invent slots.
- Do not suggest checking appointment slots too early unless the user directly asks to book.
- The final customer-facing wording may be rewritten later by the response composer.
- Focus on correct reasoning, state, action, and tool usage.

Assistant goal:
{assistant_config.get("assistant_goal", "")}

System prompt:
{assistant_config.get("system_prompt", "")}

Language policy:
{assistant_config.get("language_policy", "")}

Response rules:
{json.dumps(assistant_config.get("response_rules", []), ensure_ascii=False, indent=2)}

Current variables:
{json.dumps(variables, ensure_ascii=False, indent=2)}

Variable schema:
{json.dumps(schema.get("variables", schema), ensure_ascii=False, indent=2)}

Available tools:
{json.dumps(exposed_tools, ensure_ascii=False, indent=2)}

Return valid JSON only:
{{
  "action": "reply" | "ask_user" | "call_tool",
  "answer": "short draft answer only; final wording will be handled by response composer",
  "tool_name": "tool name when calling tool",
  "operation": "operation name when calling tool",
  "arguments": {{}},
  "variable_updates": {{}},
  "clear_variables": [],
  "confidence": 0.0,
  "notes": "internal note"
}}
""".strip()

    @staticmethod
    def safe_json_loads(text: str) -> Dict[str, Any]:
        try:
            data = json.loads(text)

            if isinstance(data, dict):
                return AgenticBrain.normalize_decision(data)
        except Exception:
            pass

        start = text.find("{")
        end = text.rfind("}")

        if start >= 0 and end > start:
            try:
                data = json.loads(text[start:end + 1])

                if isinstance(data, dict):
                    return AgenticBrain.normalize_decision(data)
            except Exception:
                pass

        return AgenticBrain.normalize_decision({
            "action": "reply",
            "answer": "",
            "variable_updates": {},
            "clear_variables": [],
            "arguments": {}
        })

    @staticmethod
    def normalize_decision(decision: Dict[str, Any]) -> Dict[str, Any]:
        decision.setdefault("action", "reply")
        decision.setdefault("answer", "")
        decision.setdefault("tool_name", "")
        decision.setdefault("operation", "")
        decision.setdefault("arguments", {})
        decision.setdefault("variable_updates", {})
        decision.setdefault("clear_variables", [])
        decision.setdefault("confidence", 0)
        decision.setdefault("notes", "")

        if not isinstance(decision["arguments"], dict):
            decision["arguments"] = {}

        if not isinstance(decision["variable_updates"], dict):
            decision["variable_updates"] = {}

        if not isinstance(decision["clear_variables"], list):
            decision["clear_variables"] = []

        return decision

    @staticmethod
    def extract_history(conversation: Dict[str, Any]) -> List[Dict[str, str]]:
        messages = conversation.get("messages", [])

        if not isinstance(messages, list):
            return []

        output = []

        for item in messages[-12:]:
            if not isinstance(item, dict):
                continue

            role = item.get("role")
            content = item.get("content")

            if role in ["user", "assistant"] and content:
                output.append({
                    "role": role,
                    "content": str(content)
                })

        return output
