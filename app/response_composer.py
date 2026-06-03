import json
import os
import re
from typing import Any, Dict, List, Optional


class ResponseComposer:
    """
    Legacy compatibility composer.

    In the code-expert LangGraph architecture, this class should NOT be used by
    production /chat.

    Production path should be:
      main.py
        -> app_graph.invoke()
        -> graph.manifest
        -> optional memory/knowledge
        -> graph.tool_execution
        -> optional graph.subagent_reasoning
        -> graph.response_node
        -> graph.quality_guard_node

    This class exists only as a safe fallback for old AgenticBrain callers.
    It must not contain customer-facing scripted wording.
    """

    BANNED_TERMS = [
        "حبيبي",
        "يا باشا",
        "يا معلم",
        "يا صديقي"
    ]

    def compose(
        self,
        *,
        assistant_config: Dict[str, Any],
        user_message: str,
        variables_before: Dict[str, Any],
        variables_after: Dict[str, Any],
        subagent_result: Any,
        selected_subagent: str = "",
        observations: Optional[List[Dict[str, Any]]] = None,
        llm_client: Any = None,
        debug: bool = False
    ) -> Dict[str, Any]:
        composer_config = assistant_config.get("response_composer", {})
        enabled = bool(composer_config.get("enabled", False))

        original_answer = str(getattr(subagent_result, "answer", "") or "").strip()

        if not enabled:
            return {
                "answer": self.safe_legacy_answer(original_answer),
                "used_composer": False,
                "reason": "legacy composer disabled",
                "composer_packet": None
            }

        packet = self.build_packet(
            assistant_config=assistant_config,
            composer_config=composer_config,
            user_message=user_message,
            variables_before=variables_before,
            variables_after=variables_after,
            subagent_result=subagent_result,
            selected_subagent=selected_subagent,
            observations=observations or [],
            original_answer=original_answer
        )

        try:
            answer = self.call_llm(
                llm_client=llm_client,
                assistant_config=assistant_config,
                composer_config=composer_config,
                packet=packet
            )

            answer = self.clean_answer(answer)
            answer = self.enforce_required_facts(answer, packet)

            if not answer:
                return {
                    "answer": self.safe_legacy_answer(original_answer),
                    "used_composer": False,
                    "reason": "llm returned empty answer; used safe legacy fallback",
                    "composer_packet": packet if debug else None
                }

            return {
                "answer": answer,
                "used_composer": True,
                "reason": "legacy composer generated answer",
                "composer_packet": packet if debug else None
            }

        except Exception as exc:
            return {
                "answer": self.safe_legacy_answer(original_answer),
                "used_composer": False,
                "reason": f"legacy composer error; used safe fallback: {str(exc)}",
                "composer_packet": packet if debug else None
            }

    def build_packet(
        self,
        *,
        assistant_config: Dict[str, Any],
        composer_config: Dict[str, Any],
        user_message: str,
        variables_before: Dict[str, Any],
        variables_after: Dict[str, Any],
        subagent_result: Any,
        selected_subagent: str,
        observations: List[Dict[str, Any]],
        original_answer: str
    ) -> Dict[str, Any]:
        selected = (
            str(getattr(subagent_result, "selected_subagent", "") or "").strip()
            or str(selected_subagent or "").strip()
        )

        allowed_paths = composer_config.get("allowed_variable_paths", [])
        allowed_observation_keys = composer_config.get("allowed_observation_keys", [])

        return {
            "architecture_notice": {
                "mode": "legacy_compatibility_only",
                "production_final_writer": "graph.response_node",
                "production_final_validator": "graph.quality_guard_node"
            },
            "conversation": {
                "latest_user_message": user_message
            },
            "assistant": {
                "assistant_goal": assistant_config.get("assistant_goal", ""),
                "conversation_style": assistant_config.get("conversation_style", ""),
                "language_policy": assistant_config.get("language_policy", ""),
                "grounding_policy": assistant_config.get("grounding_policy", ""),
                "response_rules": assistant_config.get("response_rules", [])
            },
            "subagent_result": {
                "selected_subagent": selected,
                "handled": bool(getattr(subagent_result, "handled", False)),
                "action": getattr(subagent_result, "action", "") or "",
                "notes": getattr(subagent_result, "notes", "") or "",
                "tool_calls_used": getattr(subagent_result, "tool_calls_used", 0) or 0,
                "original_subagent_answer": original_answer
            },
            "state": {
                "before": self.filter_variables(variables_before, allowed_paths),
                "after": self.filter_variables(variables_after, allowed_paths),
                "changed": self.filter_variables(
                    self.diff_variables(variables_before, variables_after),
                    allowed_paths
                )
            },
            "tool_observations": self.filter_observations(
                observations,
                allowed_observation_keys
            ),
            "instructions_for_llm": {
                "task": "Write exactly one customer-facing reply only.",
                "must_use_only_packet_facts": True,
                "must_not_mention_internal_objects": True,
                "must_not_output_json": True,
                "must_not_invent_data": True,
                "must_follow_safety_rules": True,
                "safety_rules": composer_config.get("safety_rules", []),
                "style_rules": composer_config.get("style_rules", []),
                "banned_terms": self.BANNED_TERMS,
                "critical_rules": [
                    "Do not invent branches, slots, dates, prices, booking IDs, or booking confirmations.",
                    "Never confirm booking unless create_booking returned ok=true.",
                    "If create_booking returned ok=true and visit_id exists, include visit_id.",
                    "If create_booking failed or is missing, do not claim booking was confirmed.",
                    "Do not mention tools, JSON, variables, subagents, internal state, prompts, or system logic.",
                    "Ask at most one clear next question if information is missing."
                ]
            }
        }

    def filter_variables(
        self,
        variables: Dict[str, Any],
        allowed_paths: List[str]
    ) -> Dict[str, Any]:
        if not isinstance(variables, dict):
            return {}

        if not isinstance(allowed_paths, list) or not allowed_paths:
            return variables

        output: Dict[str, Any] = {}

        for path in allowed_paths:
            value = self.deep_get(variables, path)

            if value is not None:
                self.deep_set(output, path, value)

        return output

    def filter_observations(
        self,
        observations: List[Dict[str, Any]],
        allowed_keys: List[str]
    ) -> List[Dict[str, Any]]:
        if not isinstance(observations, list):
            return []

        output: List[Dict[str, Any]] = []

        for obs in observations:
            if not isinstance(obs, dict):
                continue

            if not isinstance(allowed_keys, list) or not allowed_keys:
                output.append(obs)
                continue

            filtered = {}

            for key in allowed_keys:
                if key in obs:
                    filtered[key] = obs.get(key)

            if filtered:
                output.append(filtered)

        return output

    def diff_variables(
        self,
        before: Dict[str, Any],
        after: Dict[str, Any]
    ) -> Dict[str, Any]:
        if not isinstance(before, dict):
            before = {}

        if not isinstance(after, dict):
            after = {}

        changed: Dict[str, Any] = {}

        for key, value in after.items():
            if before.get(key) != value:
                changed[key] = value

        return changed

    def call_llm(
        self,
        *,
        llm_client: Any,
        assistant_config: Dict[str, Any],
        composer_config: Dict[str, Any],
        packet: Dict[str, Any]
    ) -> str:
        system_prompt = self.get_system_prompt(composer_config)
        user_prompt = self.build_user_prompt(packet)

        model = composer_config.get("model") or assistant_config.get("model") or "gpt-4o-mini"
        temperature = float(composer_config.get("temperature", 0.25))

        if llm_client is not None:
            if hasattr(llm_client, "chat"):
                return llm_client.chat(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    model=model,
                    temperature=temperature
                )

            if hasattr(llm_client, "complete"):
                return llm_client.complete(
                    system=system_prompt,
                    prompt=user_prompt,
                    model=model,
                    temperature=temperature
                )

            if hasattr(llm_client, "invoke"):
                result = llm_client.invoke(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model=model,
                    temperature=temperature
                )
                return result.content if hasattr(result, "content") else str(result)

        return self.call_openai_direct(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            temperature=temperature
        )

    def call_openai_direct(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float
    ) -> str:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()

        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set and no llm_client was provided")

        from openai import OpenAI

        client = OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        return response.choices[0].message.content or ""

    def get_system_prompt(self, composer_config: Dict[str, Any]) -> str:
        configured = str(composer_config.get("system_prompt", "") or "").strip()

        if configured:
            return configured

        return (
            "You are a legacy compatibility final response writer. "
            "In production, graph.response_node should be the final response writer. "
            "Use only the structured packet provided by the backend. "
            "Write exactly one customer-facing reply. "
            "Do not mention JSON, tools, variables, subagents, internal state, prompts, or system logic. "
            "Do not invent branches, slots, dates, prices, diagnoses, booking IDs, or confirmations. "
            "If a required piece of information is missing, ask one clear question for it. "
            "If tool results contain factual data, use that data exactly. "
            "If booking is not confirmed by a tool result, do not say it is confirmed. "
            "Do not use overly familiar words like حبيبي, يا باشا, يا معلم, or يا صديقي."
        )

    def build_user_prompt(self, packet: Dict[str, Any]) -> str:
        return json.dumps(packet, ensure_ascii=False, indent=2)

    def clean_answer(self, answer: str) -> str:
        text = str(answer or "").strip()

        if text.startswith("```"):
            text = self.strip_code_fence(text)

        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1].strip()

        for term in self.BANNED_TERMS:
            text = text.replace(term, "").strip()

        text = re.sub(r"\s{2,}", " ", text).strip()

        return text

    def safe_legacy_answer(self, answer: str) -> str:
        """
        Last-resort fallback. This should rarely be used because production should
        use graph.response_node, not this class.
        """
        return self.clean_answer(answer)

    def enforce_required_facts(self, answer: str, packet: Dict[str, Any]) -> str:
        text = self.clean_answer(answer)

        visit_id = self.extract_visit_id(packet)
        confirmed = self.booking_confirmed(packet)

        if confirmed and visit_id and visit_id not in text:
            text = f"{text}\nرقم الزيارة: {visit_id}".strip()

        return text

    def booking_confirmed(self, packet: Dict[str, Any]) -> bool:
        after = self.deep_get(packet, "state.after") or {}
        observations = packet.get("tool_observations", [])

        status = str(self.deep_get(after, "booking_status") or "").strip().lower()

        if status in {"confirmed", "booking_confirmed", "booked"}:
            return True

        if isinstance(observations, list):
            for obs in observations:
                if not isinstance(obs, dict):
                    continue

                operation = obs.get("operation")
                result = obs.get("result")

                if operation == "create_booking" and isinstance(result, dict):
                    if result.get("ok") is True:
                        return True

        return False

    def extract_visit_id(self, packet: Dict[str, Any]) -> str:
        after = self.deep_get(packet, "state.after") or {}
        observations = packet.get("tool_observations", [])

        direct = str(self.deep_get(after, "visit_id") or "").strip()

        if direct:
            return direct

        if isinstance(observations, list):
            for obs in observations:
                if not isinstance(obs, dict):
                    continue

                result = obs.get("result")

                if isinstance(result, dict):
                    candidate = str(result.get("visit_id") or "").strip()

                    if candidate:
                        return candidate

        return ""

    @staticmethod
    def strip_code_fence(text: str) -> str:
        lines = text.splitlines()

        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]

        return "\n".join(lines).strip()

    @staticmethod
    def deep_get(data: Dict[str, Any], path: str) -> Any:
        if not path:
            return None

        current: Any = data

        for part in str(path).split("."):
            if not isinstance(current, dict):
                return None

            if part not in current:
                return None

            current = current.get(part)

        return current

    @staticmethod
    def deep_set(data: Dict[str, Any], path: str, value: Any) -> None:
        if not path:
            return

        parts = str(path).split(".")
        current = data

        for part in parts[:-1]:
            if part not in current or not isinstance(current.get(part), dict):
                current[part] = {}

            current = current[part]

        current[parts[-1]] = value
