from typing import Any, Dict

from app.subagents.base import (
    SubagentContext,
    SubagentResult,
    matches_any,
    render_template
)


class TroubleshootingSubagent:
    name = "troubleshooting"

    def get_config(self, assistant_config: Dict[str, Any]) -> Dict[str, Any]:
        return (
            assistant_config.get("subagents", {})
            .get(self.name, {})
        )

    def run(self, context: SubagentContext) -> SubagentResult:
        config = self.get_config(context.assistant_config)

        if not config.get("enabled", False):
            return SubagentResult(handled=False)

        variables = context.variables or {}
        normalization = (
            context.assistant_config
            .get("normalization", {})
        )

        skip_if_paths_exist = config.get("skip_if_paths_exist", [])
        for path in skip_if_paths_exist:
            current = self._get_variable(variables, path)
            if current not in [None, "", [], {}]:
                return SubagentResult(handled=False)

        rules = config.get("rules", [])
        if not isinstance(rules, list):
            return SubagentResult(handled=False)

        for rule in rules:
            phrases = rule.get("phrases", [])
            if not isinstance(phrases, list):
                continue

            if not matches_any(context.user_message, phrases, normalization):
                continue

            variable_updates = rule.get("variable_updates", {})
            clear_variables = rule.get("clear_variables", [])
            template = rule.get("answer_template", "")

            answer = render_template(template, {
                "variables": variables,
                "message": context.user_message
            })

            return SubagentResult(
                handled=True,
                action="ask_user",
                answer=answer,
                variable_updates=variable_updates if isinstance(variable_updates, dict) else {},
                clear_variables=clear_variables if isinstance(clear_variables, list) else [],
                selected_subagent=self.name,
                notes=f"matched rule {rule.get('id', '')}"
            )

        return SubagentResult(handled=False)

    @staticmethod
    def _get_variable(variables: Dict[str, Any], path: str) -> Any:
        current: Any = variables

        for part in str(path).split("."):
            if not isinstance(current, dict):
                return None
            current = current.get(part)

        return current
