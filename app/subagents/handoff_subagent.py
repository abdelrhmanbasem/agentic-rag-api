from typing import Any, Dict

from app.subagents.base import (
    SubagentContext,
    SubagentResult,
    matches_any,
    render_template
)


class HandoffSubagent:
    name = "handoff"

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
        phrases = config.get("trigger_phrases", [])

        if not matches_any(context.user_message, phrases, normalization):
            return SubagentResult(handled=False)

        updates = config.get("variable_updates", {})
        clear = config.get("clear_variables", [])
        answer = render_template(config.get("answer_template", ""), {
            "variables": context.variables,
            "message": context.user_message
        })

        return SubagentResult(
            handled=True,
            action="handoff",
            answer=answer,
            variable_updates=updates if isinstance(updates, dict) else {},
            clear_variables=clear if isinstance(clear, list) else [],
            selected_subagent=self.name,
            notes="handoff triggered"
        )
