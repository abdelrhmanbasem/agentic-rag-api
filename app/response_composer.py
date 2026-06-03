import json
import os
from typing import Any, Dict, List, Optional


class ResponseComposer:
    """
    LLM-only customer response composer.

    This class does NOT write customer-facing scripted replies.
    It only:
      1. Builds a structured context packet from the brain/subagents/tools.
      2. Calls the LLM.
      3. Returns the LLM-generated answer.
      4. Falls back to the original subagent answer only if the LLM call fails.

    Subagents remain responsible for:
      - routing
      - state updates
      - missing-field checks
      - tool calls
      - booking safety
      - deterministic fallback answer

    The LLM remains responsible for:
      - natural wording
      - flexible phrasing
      - deciding the best next conversational wording based on the provided state
      - never inventing data outside the packet
    """

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
        enabled = bool(composer_config.get("enabled", True))

        original_answer = str(getattr(subagent_result, "answer", "") or "").strip()

        if not enabled:
            return {
                "answer": original_answer,
                "used_composer": False,
                "reason": "composer disabled",
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

            if not answer:
                return {
                    "answer": original_answer,
                    "used_composer": False,
                    "reason": "llm returned empty answer; used original subagent answer",
                    "composer_packet": packet if debug else None
                }

            return {
                "answer": answer,
                "used_composer": True,
                "reason": "llm composed answer",
                "composer_packet": packet if debug else None
            }

        except Exception as exc:
            return {
                "answer": original_answer,
                "used_composer": False,
                "reason": f"composer error; used original subagent answer: {str(exc)}",
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

        packet = {
            "conversation": {
                "latest_user_message": user_message
            },
            "assistant": {
                "assistant_goal": assistant_config.get("assistant_goal", ""),
                "language_policy": assistant_config.get("language_policy", ""),
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
                "before": self.filter_variables(
                    variables_before,
                    composer_config.get("allowed_variable_paths", [])
                ),
                "after": self.filter_variables(
                    variables_after,
                    composer_config.get("allowed_variable_paths", [])
                ),
                "changed": self.filter_variables(
                    self.diff_variables(variables_before, variables_after),
                    composer_config.get("allowed_variable_paths", [])
                )
            },
            "tool_observations": self.filter_observations(
                observations,
                composer_config.get("allowed_observation_keys", [])
            ),
            "instructions_for_llm": {
                "task": "Write the next customer-facing reply only.",
                "must_use_only_packet_facts": True,
                "must_not_mention_internal_objects": True,
                "must_not_output_json": True,
                "must_follow_safety_rules": True,
                "safety_rules": composer_config.get("safety_rules", []),
                "style_rules": composer_config.get("style_rules", [])
            }
        }

        extra_context_keys = composer_config.get("include_assistant_config_keys", [])

        if isinstance(extra_context_keys, list) and extra_context_keys:
            packet["assistant_config_context"] = {}

            for key in extra_context_keys:
                if key in assistant_config:
                    packet["assistant_config_context"][key] = assistant_config.get(key)

        return packet

    def filter_variables(
        self,
        variables: Dict[str, Any],
        allowed_paths: List[str]
    ) -> Dict[str, Any]:
        if not isinstance(variables, dict):
            return {}

        if not allowed_paths:
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

            if not allowed_keys:
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
        temperature = float(composer_config.get("temperature", 0.35))

        if llm_client is not None:
            if hasattr(llm_client, "chat"):
                return llm_client.chat(
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": user_prompt
                        }
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
                return llm_client.invoke(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model=model,
                    temperature=temperature
                )

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
        api_key = os.getenv("OPENAI_API_KEY", "")

        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set and no llm_client was provided")

        from openai import OpenAI

        client = OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
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

        return response.choices[0].message.content or ""

    def get_system_prompt(self, composer_config: Dict[str, Any]) -> str:
        configured = str(composer_config.get("system_prompt", "") or "").strip()

        if configured:
            return configured

        return (
            "You are the final response writer for a customer-service assistant. "
            "Use only the structured packet provided by the backend. "
            "Write exactly one customer-facing reply. "
            "Do not mention JSON, tools, variables, subagents, internal state, or system logic. "
            "Do not invent branches, slots, dates, prices, diagnoses, booking IDs, or confirmations. "
            "If a required piece of information is missing, ask one clear question for it. "
            "If tool results contain factual data, use that data exactly. "
            "If booking is not confirmed by a tool result, do not say it is confirmed."
        )

    def build_user_prompt(self, packet: Dict[str, Any]) -> str:
        return json.dumps(packet, ensure_ascii=False, indent=2)

    def clean_answer(self, answer: str) -> str:
        text = str(answer or "").strip()

        if text.startswith("```"):
            text = self.strip_code_fence(text)

        if text.startswith('"') and text.endswith('"'):
            text = text[1:-1].strip()

        return text

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
