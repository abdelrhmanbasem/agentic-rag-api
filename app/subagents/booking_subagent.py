import re
from typing import Any, Dict, List, Optional

from app.subagents.base import (
    SubagentContext,
    SubagentResult,
    apply_tool_update_rules,
    apply_variable_patch,
    build_object_from_mapping,
    compact_dict,
    deep_get,
    extract_by_patterns,
    format_missing_fields,
    get_missing_paths,
    matches_any,
    normalize_text,
    normalize_digits,
    render_template
)


class BookingSubagent:
    name = "booking"

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
        tool_calls_used = 0

        stage_path = config.get("stage_path", "booking.stage")
        stage = deep_get(variables, stage_path, "")

        # Extract date phrases only when booking is active or user is asking for slots/appointments.
        date_text = self.extract_date_text(context.user_message, config, normalization)
        if date_text and self.should_extract_date(context, config, stage):
            variables = apply_variable_patch(
                variables,
                {
                    config.get("date_text_path", "date_text"): date_text
                },
                []
            )

        # Customer details extraction only during customer-detail collection.
        extraction_active_stages = config.get("extraction_active_stages", [
            config.get("stages", {}).get("awaiting_customer_details", "awaiting_customer_details")
        ])

        should_extract_customer_details = stage in extraction_active_stages
        extracted: Dict[str, Any] = {}

        if should_extract_customer_details:
            extracted = extract_by_patterns(
                message=context.user_message,
                patterns=config.get("extraction_patterns", []),
                variables=variables,
                normalization_config=normalization
            )

            if extracted:
                variables = apply_variable_patch(variables, extracted, [])

        stage = deep_get(variables, stage_path, stage)

        if stage == config.get("stages", {}).get("awaiting_confirmation", "awaiting_confirmation"):
            return self.handle_awaiting_confirmation(
                context=context,
                config=config,
                variables=variables,
                observations=observations,
                tool_calls_used=tool_calls_used
            )

        if stage == config.get("stages", {}).get("awaiting_customer_details", "awaiting_customer_details"):
            return self.handle_awaiting_customer_details(
                context=context,
                config=config,
                variables=variables,
                observations=observations,
                tool_calls_used=tool_calls_used
            )

        selected_slot = self.resolve_slot_selection(
            context=context,
            config=config,
            variables=variables
        )

        if selected_slot:
            return self.handle_slot_selected(
                context=context,
                config=config,
                variables=variables,
                selected_slot=selected_slot
            )

        if self.is_booking_or_availability_request(context, config, variables):
            return self.handle_booking_request(
                context=context,
                config=config,
                variables=variables,
                observations=observations,
                tool_calls_used=tool_calls_used
            )

        if extracted:
            return SubagentResult(
                handled=False,
                variable_updates=variables,
                selected_subagent=self.name,
                notes="customer detail extraction only"
            )

        return SubagentResult(handled=False)

    def handle_awaiting_confirmation(
        self,
        context: SubagentContext,
        config: Dict[str, Any],
        variables: Dict[str, Any],
        observations: List[Dict[str, Any]],
        tool_calls_used: int
    ) -> SubagentResult:
        normalization = context.assistant_config.get("normalization", {})
        confirmation_phrases = config.get("confirmation_phrases", [])
        rejection_phrases = config.get("rejection_phrases", [])

        if matches_any(context.user_message, rejection_phrases, normalization):
            updates = config.get("on_reject_updates", {})
            clear = config.get("on_reject_clear", [])

            patched = apply_variable_patch(
                variables,
                updates if isinstance(updates, dict) else {},
                clear if isinstance(clear, list) else []
            )

            answer = render_template(config.get("templates", {}).get("slot_rejected", ""), {
                "variables": patched
            })

            return SubagentResult(
                handled=True,
                action="ask_user",
                answer=answer,
                variable_updates=patched,
                clear_variables=[],
                selected_subagent=self.name,
                observations=observations,
                tool_calls_used=tool_calls_used,
                notes="pending booking rejected"
            )

        if not matches_any(context.user_message, confirmation_phrases, normalization):
            answer = render_template(config.get("templates", {}).get("repeat_confirmation", ""), {
                "variables": variables
            })

            return SubagentResult(
                handled=True,
                action="ask_user",
                answer=answer,
                variable_updates=variables,
                selected_subagent=self.name,
                observations=observations,
                tool_calls_used=tool_calls_used,
                notes="awaiting explicit confirmation"
            )

        # Important: persist customer_confirmed_booking=true.
        variables = apply_variable_patch(
            variables,
            config.get("on_confirm_updates", {}),
            []
        )

        missing = get_missing_paths(
            config.get("required_before_create", []),
            variables
        )

        if missing:
            answer = self.render_missing_question(config, variables, missing)

            variables = apply_variable_patch(
                variables,
                config.get("on_missing_details_updates", {}),
                []
            )

            return SubagentResult(
                handled=True,
                action="ask_user",
                answer=answer,
                variable_updates=variables,
                selected_subagent=self.name,
                observations=observations,
                tool_calls_used=tool_calls_used,
                notes="confirmed slot but missing customer details"
            )

        return self.call_create_booking(
            context=context,
            config=config,
            variables=variables,
            observations=observations,
            tool_calls_used=tool_calls_used
        )

    def handle_awaiting_customer_details(
        self,
        context: SubagentContext,
        config: Dict[str, Any],
        variables: Dict[str, Any],
        observations: List[Dict[str, Any]],
        tool_calls_used: int
    ) -> SubagentResult:
        missing = get_missing_paths(
            config.get("required_before_create", []),
            variables
        )

        if missing:
            answer = self.render_missing_question(config, variables, missing)

            return SubagentResult(
                handled=True,
                action="ask_user",
                answer=answer,
                variable_updates=variables,
                selected_subagent=self.name,
                observations=observations,
                tool_calls_used=tool_calls_used,
                notes="still missing customer details"
            )

        return self.call_create_booking(
            context=context,
            config=config,
            variables=variables,
            observations=observations,
            tool_calls_used=tool_calls_used
        )

    def handle_slot_selected(
        self,
        context: SubagentContext,
        config: Dict[str, Any],
        variables: Dict[str, Any],
        selected_slot: Dict[str, Any]
    ) -> SubagentResult:
        slot_mapping = config.get("slot_to_pending_booking_mapping", {})
        pending_booking_path = config.get("pending_booking_path", "booking.pending")

        pending = build_object_from_mapping(slot_mapping, {
            "variables": variables,
            "slot": selected_slot,
            "message": context.user_message
        })

        updates = dict(config.get("on_slot_selected_updates", {}))
        updates[pending_booking_path] = pending

        patched = apply_variable_patch(variables, updates, [])

        answer = render_template(config.get("templates", {}).get("confirm_slot", ""), {
            "variables": patched,
            "slot": selected_slot,
            "pending": pending
        })

        return SubagentResult(
            handled=True,
            action="ask_user",
            answer=answer,
            variable_updates=patched,
            clear_variables=config.get("on_slot_selected_clear", []),
            selected_subagent=self.name,
            notes="slot selected"
        )

    def handle_booking_request(
        self,
        context: SubagentContext,
        config: Dict[str, Any],
        variables: Dict[str, Any],
        observations: List[Dict[str, Any]],
        tool_calls_used: int
    ) -> SubagentResult:
        required_for_slots = config.get("required_before_list_slots", [])
        missing = get_missing_paths(required_for_slots, variables)

        if missing:
            variables = apply_variable_patch(
                variables,
                config.get("on_missing_list_slots_updates", {}),
                []
            )

            answer = self.render_missing_question(config, variables, missing)

            return SubagentResult(
                handled=True,
                action="ask_user",
                answer=answer,
                variable_updates=variables,
                selected_subagent=self.name,
                observations=observations,
                tool_calls_used=tool_calls_used,
                notes="booking request missing required inputs"
            )

        operations = config.get("operations", {})
        tool_name = config.get("tool_name", "")
        operation = operations.get("list_slots", "")
        arguments_mapping = config.get("list_slots_arguments", {})

        arguments = build_object_from_mapping(arguments_mapping, {
            "variables": variables,
            "message": context.user_message
        })

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

        tool_calls_used += 1

        updated_variables = apply_tool_update_rules(
            assistant_config=context.assistant_config,
            variables=variables,
            operation=operation,
            arguments=arguments,
            result=tool_result
        )

        result_context = {
            "variables": updated_variables,
            "result": tool_result,
            "arguments": arguments
        }

        slots_found_path = config.get("slots_found_result_path", "slots_found")
        slots_found = deep_get(tool_result, slots_found_path, False)

        if tool_result.get("ok") is False:
            template = config.get("templates", {}).get("tool_error", "")
        elif slots_found is True:
            template = config.get("templates", {}).get("slots_found", "")
        else:
            template = config.get("templates", {}).get("no_slots", "")

        answer = render_template(template, result_context)

        return SubagentResult(
            handled=True,
            action="reply",
            answer=answer,
            variable_updates=updated_variables,
            observations=observations,
            selected_subagent=self.name,
            tool_calls_used=tool_calls_used,
            notes="listed slots"
        )

    def call_create_booking(
        self,
        context: SubagentContext,
        config: Dict[str, Any],
        variables: Dict[str, Any],
        observations: List[Dict[str, Any]],
        tool_calls_used: int
    ) -> SubagentResult:
        operations = config.get("operations", {})
        tool_name = config.get("tool_name", "")
        operation = operations.get("create_booking", "")
        arguments_mapping = config.get("create_booking_arguments", {})

        arguments = build_object_from_mapping(arguments_mapping, {
            "variables": variables,
            "message": context.user_message
        })

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

        tool_calls_used += 1

        updated_variables = apply_tool_update_rules(
            assistant_config=context.assistant_config,
            variables=variables,
            operation=operation,
            arguments=arguments,
            result=tool_result
        )

        template_key = "booking_confirmed" if tool_result.get("ok") is True else "booking_failed"

        answer = render_template(config.get("templates", {}).get(template_key, ""), {
            "variables": updated_variables,
            "result": tool_result,
            "arguments": arguments
        })

        return SubagentResult(
            handled=True,
            action="reply",
            answer=answer,
            variable_updates=updated_variables,
            observations=observations,
            selected_subagent=self.name,
            tool_calls_used=tool_calls_used,
            notes="create booking attempted"
        )

    def resolve_slot_selection(
        self,
        context: SubagentContext,
        config: Dict[str, Any],
        variables: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        slots_path = config.get("available_slots_path", "available_slots")
        slots = deep_get(variables, slots_path, [])

        if not isinstance(slots, list) or not slots:
            return None

        resolver = config.get("slot_resolver", {})
        normalization = context.assistant_config.get("normalization", {})
        normalized_message = normalize_text(context.user_message, normalization)

        ordinal_map = resolver.get("ordinal_map", {})

        if isinstance(ordinal_map, dict):
            for phrase, index in ordinal_map.items():
                normalized_phrase = normalize_text(str(phrase), normalization)

                if normalized_phrase and normalized_phrase in normalized_message:
                    try:
                        i = int(index)
                    except Exception:
                        continue

                    if 0 <= i < len(slots) and isinstance(slots[i], dict):
                        return slots[i]

        time_field = resolver.get("time_field")
        time_normalizer = resolver.get("time_normalizer")

        if time_field and time_normalizer == "hour_exact_or_same_hour":
            requested_time = self.extract_time(context.user_message, config, normalization)

            if requested_time:
                for slot in slots:
                    slot_time = self.extract_time(str(deep_get(slot, time_field, "")), config, normalization)

                    if slot_time == requested_time:
                        return slot

                requested_hour = requested_time.split(":")[0]

                for slot in slots:
                    slot_time = self.extract_time(str(deep_get(slot, time_field, "")), config, normalization)

                    if slot_time and slot_time.split(":")[0] == requested_hour:
                        return slot

        return None

    def extract_time(self, text: str, config: Dict[str, Any], normalization: Dict[str, Any]) -> str:
        time_config = config.get("time_normalization", {})
        digit_map = normalization.get("digit_map", {})

        for src, dst in time_config.get("replacements", {}).items():
            text = text.replace(src, dst)

        for src, dst in digit_map.items():
            text = text.replace(src, dst)

        regex = time_config.get("regex", r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?")

        try:
            match = re.search(regex, text, flags=re.IGNORECASE)
        except re.error:
            return ""

        if not match:
            return ""

        hour = int(match.group(1))
        minute = int(match.group(2) or "0")
        suffix = (match.group(3) or "").lower()

        if suffix == "pm" and hour < 12:
            hour += 12

        if suffix == "am" and hour == 12:
            hour = 0

        if hour < 0 or hour > 23 or minute < 0 or minute > 59:
            return ""

        return f"{hour:02d}:{minute:02d}"

    def should_extract_date(self, context: SubagentContext, config: Dict[str, Any], stage: str) -> bool:
        active_stages = config.get("active_request_stages", [])

        if stage in active_stages:
            return True

        return matches_any(
            context.user_message,
            config.get("trigger_phrases", []),
            context.assistant_config.get("normalization", {})
        )

    def extract_date_text(self, message: str, config: Dict[str, Any], normalization: Dict[str, Any]) -> str:
        raw_message = normalize_digits(str(message or ""), normalization.get("digit_map", {})).strip()
        normalized = normalize_text(raw_message, normalization)

        for item in config.get("date_extraction_patterns", []):
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
                if value:
                    return value

        direct_terms = config.get("date_direct_terms", [])

        for term in direct_terms:
            normalized_term = normalize_text(str(term), normalization)
            if normalized_term and normalized_term in normalized:
                return str(term)

        return ""

    def is_booking_or_availability_request(
        self,
        context: SubagentContext,
        config: Dict[str, Any],
        variables: Dict[str, Any]
    ) -> bool:
        normalization = context.assistant_config.get("normalization", {})
        phrases = config.get("trigger_phrases", [])

        if matches_any(context.user_message, phrases, normalization):
            return True

        stage_path = config.get("stage_path", "booking.stage")
        stage = deep_get(variables, stage_path, "")

        return stage in config.get("active_request_stages", [])

    def render_missing_question(self, config: Dict[str, Any], variables: Dict[str, Any], missing: List[str]) -> str:
        labels = config.get("field_labels", {})
        missing_text = format_missing_fields(missing, labels)

        return render_template(config.get("templates", {}).get("missing_fields", ""), {
            "variables": variables,
            "missing_fields": missing_text
        })
