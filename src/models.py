"""
models.py
─────────
Pydantic models for type-safe, validated data throughout the pipeline.
Replaces raw dicts — catches bad data at the boundary, not deep in logic.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator


# ── Conversation data model ───────────────────────────────────────────────────

class Message(BaseModel):
    role: str                            # "user" | "bot" | "agent"
    text: str
    text_normalized: Optional[str] = None
    timestamp: Optional[datetime] = None
    node_id: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("role")
    @classmethod
    def role_must_be_known(cls, v: str) -> str:
        allowed = {"user", "bot", "agent", "system"}
        if v.lower() not in allowed:
            raise ValueError(f"Unknown role '{v}'. Allowed: {allowed}")
        return v.lower()

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Message text cannot be empty")
        return v


class Conversation(BaseModel):
    id: str
    messages: list[Message]
    intent: Optional[str] = None
    issue: Optional[str] = None
    issue_type: Optional[str] = None
    issue_confidence: Optional[float] = None
    is_escalated: bool = False
    started_at: Optional[datetime] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    # Computed helpers
    @property
    def message_count(self) -> int:
        return len(self.messages)

    @property
    def user_messages(self) -> list[Message]:
        return [m for m in self.messages if m.role == "user"]

    @property
    def full_text(self) -> str:
        return " ".join(m.text for m in self.messages)

    @property
    def full_text_normalized(self) -> str:
        return " ".join(
            (m.text_normalized or m.text) for m in self.messages
        )

    def to_csv_row(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "intent": self.intent or "",
            "issue": self.issue or "",
            "issue_type": self.issue_type or "",
            "issue_confidence": self.issue_confidence,
            "is_escalated": self.is_escalated,
            "message_count": self.message_count,
            "started_at": self.started_at.isoformat() if self.started_at else "",
            "full_text": self.full_text[:2000],
        }


# ── LLM output model ──────────────────────────────────────────────────────────

class LLMClassification(BaseModel):
    issue: str
    issue_type: str
    confidence: float = 0.0
    reasoning: str

    @field_validator("confidence", mode="before")
    @classmethod
    def confidence_in_range(cls, v) -> float:
        try:
            v = float(v)
        except (TypeError, ValueError):
            v = 0.0
        return round(max(0.0, min(1.0, v)), 4)

    @field_validator("issue", "issue_type")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip().lower()


# ── Pipeline run stats ────────────────────────────────────────────────────────

class PipelineStats(BaseModel):
    total_loaded: int = 0
    after_dedup: int = 0
    after_filter: int = 0
    after_sample: int = 0
    llm_success: int = 0
    llm_failed: int = 0
    llm_low_confidence: int = 0
    fallback_to_keyword: int = 0
    export_rows: int = 0
    duration_seconds: float = 0.0
    errors: list[str] = Field(default_factory=list)
