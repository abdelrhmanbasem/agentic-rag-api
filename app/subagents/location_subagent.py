import re
from typing import Any, Dict, List

from app.subagents.base import (
    SubagentContext,
    SubagentResult,
    apply_tool_update_rules,
    apply_variable_patch,
    compact_dict,
    deep_get,
    matches_any,
    normalize_text,
    render_template
)


class LocationSubagent:
    name = "location"

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
        observations: List[Dict[str, Any]] = []

        location = self.extract_location(
            message=context.user_message,
            config=config,
            normalization=normalization
        )

        if not location:
            return SubagentResult(handled=False)

        tool_name = config.get("tool_name", "")
        operation = config.get("operation", "find_nearest_branch")

        arguments = {
            config.get("location_argument", "location"): location
        }

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

        manual_updates: Dict[str, Any] = {}

        if tool_result.get("ok") is True:
            manual_updates[config.get("user_area_path", "user_area")] = (
                tool_result.get("user_area") or location
            )

        if tool_result.get("branch_found") is True:
            branch = (
                tool_result.get("branch")
                or tool_result.get("nearest_branch")
                or tool_result.get("location_branch")
            )

            if branch:
                manual_updates[config.get("nearest_branch_path", "nearest_branch")] = branch
                manual_updates[config.get("location_branch_path", "location_branch")] = branch

                if config.get("set_selected_branch_on_match", True):
                    manual_updates[config.get("selected_branch_path", "selected_branch")] = branch

        updated_variables = apply_variable_patch(
            variables=updated_variables,
            updates=manual_updates,
            clear=[]
        )

        answer_template = self.choose_answer_template(
            context=context,
            config=config,
            variables=updated_variables,
            tool_result=tool_result
        )

        answer = render_template(answer_template, {
            "variables": updated_variables,
            "result": tool_result,
            "arguments": arguments,
            "location": location
        })

        if not answer:
            answer = config.get("templates", {}).get(
                "fallback",
                "تمام، محتاج تفاصيل أكتر عشان أساعدك."
            )

        return SubagentResult(
            handled=True,
            action="reply",
            answer=answer,
            variable_updates=updated_variables,
            observations=observations,
            selected_subagent=self.name,
            tool_calls_used=1,
            notes="location resolved"
        )

    def extract_location(
        self,
        message: str,
        config: Dict[str, Any],
        normalization: Dict[str, Any]
    ) -> str:
        raw_message = str(message or "").strip()

        patterns = config.get("location_patterns", [])

        if isinstance(patterns, list):
            for item in patterns:
                if not isinstance(item, dict):
                    continue

                regex = item.get("regex", "")
                group = int(item.get("group", 1))

                if not regex:
                    continue

                try:
                    match = re.search(regex, raw_message, flags=re.IGNORECASE)
                except re.error:
                    continue

                if match:
                    value = match.group(group).strip()
                    value = self.cleanup_location(value, config)
                    if value:
                        return value

        trigger_phrases = config.get("location_trigger_phrases", [])

        if matches_any(raw_message, trigger_phrases, normalization):
            cleaned = raw_message

            for phrase in trigger_phrases:
                normalized_phrase = normalize_text(str(phrase), normalization)
                normalized_cleaned = normalize_text(cleaned, normalization)

                if normalized_phrase and normalized_phrase in normalized_cleaned:
                    cleaned = re.sub(str(phrase), " ", cleaned, flags=re.IGNORECASE)

            cleaned = self.cleanup_location(cleaned, config)

            if cleaned:
                return cleaned

        return ""

    @staticmethod
    def cleanup_location(location: str, config: Dict[str, Any]) -> str:
        value = str(location or "").strip()

        stop_phrases = config.get("location_cleanup_phrases", [])

        if isinstance(stop_phrases, list):
            for phrase in stop_phrases:
                value = value.replace(str(phrase), " ")

        value = re.sub(r"[،,.!?؟]+", " ", value)
        value = re.sub(r"\s+", " ", value).strip()

        return value

    def choose_answer_template(
        self,
        context: SubagentContext,
        config: Dict[str, Any],
        variables: Dict[str, Any],
        tool_result: Dict[str, Any]
    ) -> str:
        templates = config.get("templates", {})
        normalization = context.assistant_config.get("normalization", {})

        if tool_result.get("ok") is False:
            return templates.get("tool_error", "")

        if tool_result.get("branch_found") is not True:
            return templates.get("branch_not_found", "")

        direct_booking = matches_any(
            context.user_message,
            config.get("direct_booking_phrases", []),
            normalization
        )

        if direct_booking:
            return templates.get("branch_found_direct_booking", "") or templates.get("branch_found", "")

        min_diagnostic_turns = int(config.get("min_diagnostic_turns_before_visit_offer", 2))
        diagnostic_count_path = config.get("diagnostic_count_path", "troubleshooting.diagnostic_count")
        diagnostic_count = deep_get(variables, diagnostic_count_path, 0)

        try:
            diagnostic_count_int = int(diagnostic_count or 0)
        except Exception:
            diagnostic_count_int = 0

        if diagnostic_count_int >= min_diagnostic_turns:
            return templates.get("branch_found_after_diagnostics", "") or templates.get("branch_found", "")

        return templates.get("branch_found_no_visit_offer", "") or templates.get("branch_found", "")
