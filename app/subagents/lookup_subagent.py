from typing import Any, Dict, List

from app.subagents.base import (
    SubagentContext,
    SubagentResult,
    apply_tool_update_rules,
    build_object_from_mapping,
    compact_dict,
    matches_any,
    render_template
)


class LookupSubagent:
    name = "lookup"

    def get_config(self, assistant_config: Dict[str, Any]) -> Dict[str, Any]:
        return (
            assistant_config.get("subagents", {})
            .get(self.name, {})
        )

    def run(self, context: SubagentContext) -> SubagentResult:
        config = self.get_config(context.assistant_config)

        if not config.get("enabled", False):
            return SubagentResult(handled=False)

        normalization = context.assistant_config.get("normalization", {})
        rules = config.get("rules", [])

        if not isinstance(rules, list):
            return SubagentResult(handled=False)

        for rule in rules:
            phrases = rule.get("phrases", [])

            if not matches_any(context.user_message, phrases, normalization):
                continue

            tool_name = rule.get("tool_name") or config.get("default_tool_name", "")
            operation = rule.get("operation", "")
            arguments = build_object_from_mapping(rule.get("arguments", {}), {
                "variables": context.variables,
                "message": context.user_message
            })

            result = context.tool_runner.call(
                tool_name=tool_name,
                operation=operation,
                arguments=compact_dict(arguments)
            )

            observations: List[Dict[str, Any]] = [{
                "subagent": self.name,
                "operation": operation,
                "arguments": arguments,
                "result": result
            }]

            updated_variables = apply_tool_update_rules(
                assistant_config=context.assistant_config,
                variables=context.variables,
                operation=operation,
                arguments=arguments,
                result=result
            )

            template_key = "success_template" if result.get("ok") is True else "error_template"
            answer = render_template(rule.get(template_key, ""), {
                "variables": updated_variables,
                "result": result,
                "arguments": arguments
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
