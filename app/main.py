"""FastAPI application.

Run:  uvicorn app.main:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.routes import router
from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.services.intent_service import IntentService

log = get_logger(__name__)

# Bodies larger than this are rejected before parsing (the message limit is separate).
MAX_BODY_BYTES = 256 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    setup_logging(settings.log_level)
    # Load the model and precompute intent embeddings once, before serving traffic.
    app.state.service = IntentService(settings)
    yield


app = FastAPI(
    title="Intent Router",
    version="1.0.0",
    description="Local, lightweight natural-language intent router. Returns predictions only; "
                "it never executes anything.",
    lifespan=lifespan,
)
app.include_router(router)

_INDEX = Path(__file__).resolve().parent / "static" / "index.html"


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    return FileResponse(_INDEX)


def _error(status: int, code: str, message: str, details=None) -> JSONResponse:
    body = {"error": {"code": code, "message": message}}
    if details is not None:
        body["error"]["details"] = details
    return JSONResponse(status_code=status, content=body)


class _BodyTooLargeError(Exception):
    pass


class _LimitBodySizeMiddleware:
    def __init__(self, asgi_app, max_bytes: int):
        self.asgi_app = asgi_app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.asgi_app(scope, receive, send)
            return
        for name, value in scope.get("headers", []):
            if name == b"content-length":
                if value.isdigit() and int(value) > self.max_bytes:
                    resp = _error(413, "payload_too_large", f"Request body exceeds {self.max_bytes} bytes")
                    await resp(scope, receive, send)
                    return
                break

        received = 0
        response_started = False

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message.get("type") == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise _BodyTooLargeError()
            return message

        async def tracked_send(message):
            nonlocal response_started
            if message.get("type") == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.asgi_app(scope, limited_receive, tracked_send)
        except _BodyTooLargeError:
            if not response_started:
                resp = _error(413, "payload_too_large", f"Request body exceeds {self.max_bytes} bytes")
                await resp(scope, receive, send)


app.add_middleware(_LimitBodySizeMiddleware, max_bytes=MAX_BODY_BYTES)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    # Do not echo the submitted input back (it may contain sensitive text).
    details = [{"loc": list(e.get("loc", [])), "msg": e.get("msg", ""), "type": e.get("type", "")}
               for e in exc.errors()]
    return _error(422, "validation_error", "Request body is invalid", details)


@app.exception_handler(StarletteHTTPException)
async def http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    code = {413: "message_too_long", 404: "not_found", 405: "method_not_allowed", 503: "unavailable"}.get(
        exc.status_code, "http_error")
    return _error(exc.status_code, code, str(exc.detail))


@app.exception_handler(Exception)
async def unhandled_error(request: Request, exc: Exception) -> JSONResponse:
    log.exception("Unhandled error while processing %s", request.url.path)
    return _error(500, "internal_error", "Internal error")
