"""HTTP API: validation, limits and error format."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(service):
    from intent_router.main import app

    with TestClient(app) as c:  # runs the lifespan (model is cached in-process)
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["intents"] >= 25 and "reject" in body["thresholds"]


def test_route_success(client):
    r = client.post("/route", json={"message": "Find Python jobs in Noida",
                                    "options": {"reference_date": "2026-10-22"}})
    assert r.status_code == 200
    body = r.json()
    assert body["version"] == "1.0" and body["intent"] == "jobs.search"
    assert body["entities"] == {"skills": ["Python"], "location": "Noida"}
    assert body["debug"] is None


def test_route_with_context_round_trip(client):
    first = client.post("/route", json={"message": "Find Python jobs in Noida"}).json()
    second = client.post("/route", json={"message": "Only remote ones", "context": first["context"]}).json()
    assert second["context_used"] and second["filters"]["work_mode"] == "remote"


def test_debug_option(client):
    body = client.post("/route", json={"message": "hello", "options": {"include_debug": True}}).json()
    assert body["debug"]["model"] and "embed_ms" in body["debug"]["timings_ms"]


@pytest.mark.parametrize("payload", [
    {},
    {"message": 123},
    {"message": None},
    {"message": "hi", "unexpected": True},
    {"message": "hi", "context": {"intent": 5}},
    {"message": "hi", "options": {"reference_date": "not-a-date"}},
])
def test_invalid_bodies_return_422_without_echoing_input(client, payload):
    r = client.post("/route", json=payload)
    assert r.status_code == 422
    body = r.json()
    assert body["error"]["code"] == "validation_error"
    assert all("input" not in detail for detail in body["error"]["details"])  # no echoed input key


def test_malformed_json(client):
    r = client.post("/route", content=b"{not json", headers={"content-type": "application/json"})
    assert r.status_code == 422


def test_message_over_configured_limit_returns_413(client, service):
    r = client.post("/route", json={"message": "a" * (service.settings.max_message_chars + 1)})
    assert r.status_code == 413 and r.json()["error"]["code"] == "message_too_long"


def test_oversized_body_rejected_before_parsing(client):
    r = client.post("/route", content=b"x" * (300 * 1024), headers={"content-type": "application/json"})
    assert r.status_code == 413 and r.json()["error"]["code"] == "payload_too_large"


def test_oversized_chunked_body_rejected_without_content_length(client):
    def chunks():
        for _ in range(10):
            yield b"x" * (32 * 1024)

    r = client.post("/route", content=chunks(), headers={"content-type": "application/json"})
    assert r.status_code == 413 and r.json()["error"]["code"] == "payload_too_large"


def test_small_chunked_body_without_content_length_is_routed(client):
    payload = b'{"message": "Find Python jobs in Noida"}'

    def chunks():
        for k in range(0, len(payload), 8):
            yield payload[k:k + 8]

    r = client.post("/route", content=chunks(), headers={"content-type": "application/json"})
    assert r.status_code == 200 and r.json()["entities"] == {"skills": ["Python"], "location": "Noida"}


def test_empty_message_is_a_valid_routing_result(client):
    body = client.post("/route", json={"message": "   "}).json()
    assert body["status"] == "invalid_input"


def test_unknown_path(client):
    assert client.get("/nope").status_code == 404
