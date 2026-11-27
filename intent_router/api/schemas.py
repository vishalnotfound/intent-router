"""HTTP request/response models. The routing result itself is intent_router.core.schemas.RouteResult."""

from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from intent_router.core.schemas import ConversationContext, RouteResult  # noqa: F401  (re-exported)


class RouteOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    include_debug: bool = Field(False, description="Include scores, thresholds and timings.")
    reference_date: date | None = Field(None, description="'Today' for relative dates. Default: server date.")


class RouteRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Hard upper bound for parsing; the configurable limit (ROUTER_MAX_MESSAGE_CHARS) returns 413.
    message: str = Field(..., max_length=100_000, description="The user's message (untrusted).")
    context: ConversationContext | None = Field(
        None, description="Optional explicit context, normally the `context` from the previous response."
    )
    options: RouteOptions = Field(default_factory=RouteOptions)


class HealthResponse(BaseModel):
    status: str
    model: str
    intents: int
    domains: list[str]
    thresholds: dict[str, float]
    scoring: dict[str, Any]
    fallback: str
    verifier: str = "none"
    startup_ms: dict[str, float]


class ErrorBody(BaseModel):
    code: str
    message: str
    details: list[dict[str, Any]] | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody
