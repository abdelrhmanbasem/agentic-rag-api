from typing import Any, Dict, List

from app.subagents.base import (
    SubagentContext,
    SubagentResult,
    apply_tool_update_rules,
    compact_dict,
    matches_any,
    render_template
)


class LookupSubagent:
    name = "lookup"

    def get_config(self, assistant_config: Dict[str, Any]) -> Dict[str, Any]:
        return assistant_config.get("subagents", {}).get(self.name, {})

    def run(self, context: SubagentContext) -> SubagentResult:
        config = self.get_config(context.assistant_config)

        if not config.get("enabled", False):
            return SubagentResult(handled=False)

        normalization = context.assistant_config.get("normalization", {})
        variables = dict(context.variables or {})
        observations: List[Dict[str, Any]] = []

        rules = config.get("rules", [])

        if not isinstance(rules, list):
            return SubagentResult(handled=False)

        for rule in rules:
            if not isinstance(rule, dict):
                continue

            phrases = rule.get("phrases", [])

            if not isinstance(phrases, list):
                continue

            if not matches_any(context.user_message, phrases, normalization):
                continue

            tool_name = rule.get("tool_name") or config.get("default_tool_name", "")
            operation = rule.get("operation", "")
            arguments = rule.get("arguments", {})

            if not isinstance(arguments, dict):
                arguments = {}

            tool_result = context.tool_runner.call(
                tool_name=tool_name,
                operation=operation,
                arguments=compact_dict(arguments)
            )

            observations.append({
                "subagent": self.name,
                "operation": operation,
                "arguments": arguments,
                "result": tool_result
            })

            updated_variables = apply_tool_update_rules(
                assistant_config=context.assistant_config,
                variables=variables,
                operation=operation,
                arguments=arguments,
                result=tool_result
            )

            template = (
                rule.get("success_template", "")
                if tool_result.get("ok") is not False
                else rule.get("error_template", "")
            )

            answer = render_template(template, {
                "variables": updated_variables,
                "result": tool_result,
                "arguments": arguments,
                "message": context.user_message
            })

            return SubagentResult(
                handled=True,
                action="reply",
                answer=answer,
                variable_updates=updated_variables,
                observations=observations,
                selected_subagent=self.name,
                tool_calls_used=1,
                notes=f"matched lookup rule {rule.get('id', '')}"
            )

        return SubagentResult(handled=False)
