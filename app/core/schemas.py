"""Versioned, validated routing result schema (v1.0).

Every response the router produces is an instance of :class:`RouteResult`.
The router only *predicts* what the user wants; it never executes anything.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator

SCHEMA_VERSION = "1.0"

SideEffects = Literal["none", "read", "write", "destructive", "financial"]


class Status(str, Enum):
    success = "success"  # confident single intent, required fields present
    needs_clarification = "needs_clarification"  # missing required info or low-confidence guess
    ambiguous = "ambiguous"  # top candidates too close to call
    unknown = "unknown"  # nothing in the registry fits
    multi_intent = "multi_intent"  # several actions, see `tasks`
    invalid_input = "invalid_input"  # empty / non-linguistic input


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Alternative(_Strict):
    intent: str
    action: str
    target: str | None = None
    domain: str
    # Raw aggregated cosine similarity. NOT a probability.
    score: float = Field(ge=-1.0, le=1.0)


class Task(_Strict):
    """One action inside a multi-intent request."""

    index: int = Field(ge=0)
    text: str
    status: Status
    intent: str | None
    action: str | None
    target: str | None
    domain: str | None
    entities: dict[str, JsonValue] = Field(default_factory=dict)
    filters: dict[str, JsonValue] = Field(default_factory=dict)
    missing_fields: list[str] = Field(default_factory=list)
    optional_missing_fields: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    depends_on: list[int] = Field(default_factory=list)
    side_effects: SideEffects = "none"
    requires_confirmation: bool = False


class ConversationContext(_Strict):
    """Explicit conversation state. The caller stores it and sends it back.

    The router keeps no hidden state between calls.
    """

    previous_message: str | None = Field(default=None, max_length=2000)
    intent: str | None = Field(default=None, max_length=100)
    action: str | None = Field(default=None, max_length=50)
    target: str | None = Field(default=None, max_length=100)
    domain: str | None = Field(default=None, max_length=50)
    entities: dict[str, JsonValue] = Field(default_factory=dict)
    filters: dict[str, JsonValue] = Field(default_factory=dict)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("entities", "filters")
    @classmethod
    def _limit_size(cls, value: dict[str, Any]) -> dict[str, Any]:
        if len(value) > 50:
            raise ValueError("at most 50 keys allowed")
        if len(str(value)) > 10_000:
            raise ValueError("value too large")
        return value


class DebugInfo(_Strict):
    normalized_text: str
    routing_text: str  # instruction part after typo correction, as embedded
    top_scores: list[Alternative] = Field(default_factory=list)
    action_scores: dict[str, float] = Field(default_factory=dict)
    typo_corrections: list[list[str]] = Field(default_factory=list)
    thresholds: dict[str, float] = Field(default_factory=dict)
    timings_ms: dict[str, float] = Field(default_factory=dict)
    fallback_used: bool = False
    model: str


class RouteResult(_Strict):
    version: Literal["1.0"] = SCHEMA_VERSION
    status: Status
    intent: str | None = None  # canonical intent id, "unknown", or None when ambiguous/multi
    action: str | None = None  # broad action, e.g. "search"
    target: str | None = None  # resource type, e.g. "job_openings"
    domain: str | None = None
    entities: dict[str, JsonValue] = Field(default_factory=dict)
    filters: dict[str, JsonValue] = Field(default_factory=dict)
    missing_fields: list[str] = Field(default_factory=list)  # required and absent
    optional_missing_fields: list[str] = Field(default_factory=list)  # useful but optional
    needs_clarification: bool = False
    clarification_question: str | None = None
    # Heuristic score in [0, 1] derived from similarity and margin. Not a calibrated
    # probability and never an authorization signal.
    confidence: float = Field(ge=0.0, le=1.0)
    response_time: float | None = Field(default=None, ge=0.0, description="Total response time in milliseconds")
    alternatives: list[Alternative] = Field(default_factory=list)
    tasks: list[Task] = Field(default_factory=list)
    is_follow_up: bool = False
    context_used: bool = False
    # How the intent was chosen: embedding similarity, supplied context, or the optional LLM fallback.
    resolved_by: Literal["embedding", "context", "fallback", "none"] = "embedding"
    side_effects: SideEffects = "none"
    requires_confirmation: bool = False
    reasoning_summary: str = Field(default="", max_length=400)
    context: ConversationContext | None = None  # updated context to send back next turn
    debug: DebugInfo | None = None

    @field_validator("confidence")
    @classmethod
    def _round(cls, value: float) -> float:
        return round(value, 4)

    @field_validator("response_time")
    @classmethod
    def _round_response_time(cls, value: float | None) -> float | None:
        return round(value, 2) if value is not None else None
