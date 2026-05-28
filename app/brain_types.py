# app/brain_types.py
# Shared lightweight types/constants for the assistant brain.
#
# Keep this file dependency-free so all future assistants can reuse it.

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


USER_STATES = {
    "new": "new",
    "curious": "curious",
    "interested": "interested",
    "qualified": "qualified",
    "ready": "ready",
    "hesitant": "hesitant",
    "price_sensitive": "price_sensitive",
    "comparing": "comparing",
    "skeptical": "skeptical",
    "frustrated": "frustrated",
    "confused": "confused",
    "closing": "closing",
}


SALES_MOVES = {
    "answer_directly": "answer_directly",
    "answer_then_soft_close": "answer_then_soft_close",
    "reassure_then_soft_close": "reassure_then_soft_close",
    "compare_options": "compare_options",
    "advise": "advise",
    "ask_one_detail": "ask_one_detail",
    "book_or_confirm": "book_or_confirm",
    "offer_human_followup": "offer_human_followup",
    "soft_close": "soft_close",
    "repair": "repair",
    "handoff": "handoff",
}


ANSWER_STYLES = {
    "short": "short",
    "short_persuasive": "short_persuasive",
    "reassuring": "reassuring",
    "advisor": "advisor",
    "human_operator": "human_operator",
    "calm_support": "calm_support",
}


GPT_POLICIES = {
    "never": "never",
    "avoid": "avoid",
    "allowed": "allowed",
    "recommended": "recommended",
    "required": "required",
}


@dataclass
class BrainDecision:
    user_state: str = "curious"
    workflow: str = "general"
    conversation_stage: str = "general"
    sales_move: str = "answer_directly"
    answer_style: str = "short"
    should_use_gpt: bool = False
    gpt_policy: str = "avoid"
    reason: str = ""
    confidence: float = 0.75
    cta: str = "continue"
    risk_flags: List[str] = field(default_factory=list)
    detected_signals: List[str] = field(default_factory=list)
    recommended_action: str = "continue_conversation"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_state": self.user_state,
            "workflow": self.workflow,
            "conversation_stage": self.conversation_stage,
            "sales_move": self.sales_move,
            "answer_style": self.answer_style,
            "should_use_gpt": self.should_use_gpt,
            "gpt_policy": self.gpt_policy,
            "reason": self.reason,
            "confidence": self.confidence,
            "cta": self.cta,
            "risk_flags": self.risk_flags,
            "detected_signals": self.detected_signals,
            "recommended_action": self.recommended_action,
            "metadata": self.metadata,
        }


@dataclass
class ComposeResult:
    answer: str
    action: str = "continue"
    updates: Dict[str, Any] = field(default_factory=dict)
    skip_summary: bool = True
    skip_memory: bool = True
    model_tier: str = "brain"
    answer_mode: str = "assistant_brain"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "action": self.action,
            "updates": self.updates,
            "skip_summary": self.skip_summary,
            "skip_memory": self.skip_memory,
            "model_tier": self.model_tier,
            "answer_mode": self.answer_mode,
        }
