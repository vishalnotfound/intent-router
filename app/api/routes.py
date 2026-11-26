"""HTTP endpoints. Routing only: nothing here executes the predicted intent."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.api.schemas import ErrorResponse, HealthResponse, RouteRequest
from app.core.schemas import RouteResult
from app.services.intent_service import IntentService, MessageTooLongError

router = APIRouter()


def _service(request: Request) -> IntentService:
    service = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Router is still starting")
    return service


@router.post(
    "/route",
    response_model=RouteResult,
    responses={413: {"model": ErrorResponse}, 422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
def route(body: RouteRequest, request: Request) -> RouteResult:
    # A sync endpoint runs in FastAPI's thread pool, so CPU-bound inference never
    # blocks the event loop; the embedder serializes model access internally.
    service = _service(request)
    try:
        return service.route(
            body.message,
            body.context,
            include_debug=body.options.include_debug,
            reference_date=body.options.reference_date,
        )
    except MessageTooLongError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    return HealthResponse(**_service(request).health())
