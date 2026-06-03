from typing import Any, Dict

from app.subagents.base import (
    SubagentContext,
    SubagentResult,
    apply_variable_patch,
    deep_get,
    matches_any,
    render_template
)


class TroubleshootingSubagent:
    name = "troubleshooting"

    def get_config(self, assistant_config: Dict[str, Any]) -> Dict[str, Any]:
        return (
            assistant_config
            .get("subagents", {})
            .get(self.name, {})
        )

    def run(self, context: SubagentContext) -> SubagentResult:
        config = self.get_config(context.assistant_config)

        if not config.get("enabled", False):
            return SubagentResult(handled=False)

        variables = dict(context.variables or {})
        normalization = context.assistant_config.get("normalization", {})

        direct_booking_phrases = config.get("direct_booking_phrases", [])

        if matches_any(context.user_message, direct_booking_phrases, normalization):
            return SubagentResult(handled=False)

        skip_if_paths_exist = config.get("skip_if_paths_exist", [])

        for path in skip_if_paths_exist:
            current = deep_get(variables, path)
            if current not in [None, "", [], {}]:
                return SubagentResult(handled=False)

        active_state = config.get("active_state", "active")
        state_path = config.get("state_path", "troubleshooting.stage")
        count_path = config.get("count_path", "troubleshooting.diagnostic_count")
        selected_rule_path = config.get("selected_rule_path", "troubleshooting.selected_rule_id")

        current_state = deep_get(variables, state_path, "")
        current_count = deep_get(variables, count_path, 0)
        selected_rule_id = deep_get(variables, selected_rule_path, "")

        try:
            current_count_int = int(current_count or 0)
        except Exception:
            current_count_int = 0

        if current_state == active_state and selected_rule_id:
            rule = self.find_rule(config, selected_rule_id)

            if rule:
                return self.continue_rule(
                    context=context,
                    config=config,
                    variables=variables,
                    rule=rule,
                    current_count=current_count_int
                )

        rules = config.get("rules", [])

        if not isinstance(rules, list):
            return SubagentResult(handled=False)

        for rule in rules:
            phrases = rule.get("phrases", [])

            if not isinstance(phrases, list):
                continue

            if not matches_any(context.user_message, phrases, normalization):
                continue

            return self.start_rule(
                context=context,
                config=config,
                variables=variables,
                rule=rule
            )

        return SubagentResult(handled=False)

    def start_rule(
        self,
        context: SubagentContext,
        config: Dict[str, Any],
        variables: Dict[str, Any],
        rule: Dict[str, Any]
    ) -> SubagentResult:
        state_path = config.get("state_path", "troubleshooting.stage")
        count_path = config.get("count_path", "troubleshooting.diagnostic_count")
        selected_rule_path = config.get("selected_rule_path", "troubleshooting.selected_rule_id")
        active_state = config.get("active_state", "active")

        steps = rule.get("diagnostic_steps", [])

        if not isinstance(steps, list) or not steps:
            updates = dict(rule.get("variable_updates", {}))

            answer = render_template(rule.get("answer_template", ""), {
                "variables": apply_variable_patch(variables, updates, []),
                "message": context.user_message
            })

            return SubagentResult(
                handled=True,
                action="ask_user",
                answer=answer,
                variable_updates=updates,
                clear_variables=rule.get("clear_variables", []),
                selected_subagent=self.name,
                notes=f"matched rule {rule.get('id', '')}"
            )

        step = steps[0]
        updates = {}

        updates.update(rule.get("variable_updates", {}))
        updates.update(step.get("variable_updates", {}))
        updates[state_path] = active_state
        updates[count_path] = 1
        updates[selected_rule_path] = rule.get("id", "")

        answer = render_template(step.get("answer_template", ""), {
            "variables": apply_variable_patch(variables, updates, []),
            "message": context.user_message
        })

        return SubagentResult(
            handled=True,
            action="ask_user",
            answer=answer,
            variable_updates=updates,
            clear_variables=rule.get("clear_variables", []),
            selected_subagent=self.name,
            notes=f"started diagnostic rule {rule.get('id', '')}"
        )

    def continue_rule(
        self,
        context: SubagentContext,
        config: Dict[str, Any],
        variables: Dict[str, Any],
        rule: Dict[str, Any],
        current_count: int
    ) -> SubagentResult:
        state_path = config.get("state_path", "troubleshooting.stage")
        count_path = config.get("count_path", "troubleshooting.diagnostic_count")

        steps = rule.get("diagnostic_steps", [])

        if not isinstance(steps, list) or current_count >= len(steps):
            updates = {
                state_path: config.get("complete_state", "complete")
            }

            return SubagentResult(
                handled=False,
                variable_updates=updates,
                selected_subagent=self.name,
                notes="diagnostic flow already complete"
            )

        step = steps[current_count]
        next_count = current_count + 1

        updates = {}
        updates.update(step.get("variable_updates", {}))
        updates[count_path] = next_count

        if step.get("complete", False) is True or next_count >= len(steps):
            updates[state_path] = config.get("complete_state", "complete")
        else:
            updates[state_path] = config.get("active_state", "active")

        answer = render_template(step.get("answer_template", ""), {
            "variables": apply_variable_patch(variables, updates, []),
            "message": context.user_message
        })

        return SubagentResult(
            handled=True,
            action="ask_user",
            answer=answer,
            variable_updates=updates,
            clear_variables=step.get("clear_variables", []),
            selected_subagent=self.name,
            notes=f"continued diagnostic rule {rule.get('id', '')} step {next_count}"
        )

    @staticmethod
    def find_rule(config: Dict[str, Any], rule_id: str) -> Dict[str, Any]:
        rules = config.get("rules", [])

        if not isinstance(rules, list):
            return {}

        for rule in rules:
            if isinstance(rule, dict) and rule.get("id") == rule_id:
                return rule

        return {}
