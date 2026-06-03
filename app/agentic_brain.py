import os
from typing import Any, Dict, Optional


class AgenticBrain:
    """
    Legacy compatibility guard.

    This class is intentionally disabled by default.

    The code-expert architecture requires production /chat to use:

        app.main
          -> app.graph.app_graph.invoke()
          -> manifest
          -> optional memory / knowledge
          -> tool_execution
          -> optional subagent_reasoning
          -> response_node
          -> quality_guard_node

    AgenticBrain represented the old architecture:
      - sequential deterministic subagent loop
      - ResponseComposer as final writer
      - llm_fallback after subagents fail

    That old path must not be used for production traffic.
    """

    def __init__(self) -> None:
        self.enabled = (
            os.getenv("ENABLE_LEGACY_AGENTIC_BRAIN", "false")
            .strip()
            .lower()
            in {"1", "true", "yes", "y", "on"}
        )

        if not self.enabled:
            return

        # Delayed imports only when explicitly enabled, so production does not
        # accidentally load or depend on the legacy pipeline.
        from openai import OpenAI
        from app.response_composer import ResponseComposer
        from app.tool_runner import ToolRunner
        from app.subagents.handoff_subagent import HandoffSubagent
        from app.subagents.location_subagent import LocationSubagent
        from app.subagents.booking_subagent import BookingSubagent
        from app.subagents.lookup_subagent import LookupSubagent
        from app.subagents.troubleshooting_subagent import TroubleshootingSubagent

        self.client = OpenAI()
        self.response_composer = ResponseComposer()
        self.ToolRunner = ToolRunner

        self.subagents = [
            HandoffSubagent(),
            LocationSubagent(),
            LookupSubagent(),
            BookingSubagent(),
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
        if not self.enabled:
            raise RuntimeError(
                "AgenticBrain is disabled. Production must use app.graph.app_graph.invoke(). "
                "If this error appears from /chat, app/main.py is still using the old architecture."
            )

        return self.run_legacy(
            assistant_config=assistant_config,
            schema=schema,
            conversation=conversation,
            user_message=user_message,
            incoming_variables=incoming_variables,
            max_tool_calls=max_tool_calls
        )

    def run_legacy(
        self,
        assistant_config: Dict[str, Any],
        schema: Dict[str, Any],
        conversation: Dict[str, Any],
        user_message: str,
        incoming_variables: Optional[Dict[str, Any]] = None,
        max_tool_calls: int = 4
    ) -> Dict[str, Any]:
        """
        Legacy path intentionally not implemented.

        If you ever need rollback, recover the old implementation from Git history
        and set ENABLE_LEGACY_AGENTIC_BRAIN=true only for emergency rollback.

        Keeping this file as a guard is safer than keeping the full old pipeline,
        because it prevents accidental production imports from silently reactivating
        the old deterministic architecture.
        """
        raise RuntimeError(
            "Legacy AgenticBrain execution is not available in this build. "
            "Use app.graph.app_graph.invoke() instead."
        )
