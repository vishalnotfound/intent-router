"""Optional NLI verifier for destructive/financial requests, plus the follow-up and
gerund negation rules that cover what the verifier cannot judge."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import numpy as np
import pytest

from intent_router.core.schemas import Status
from intent_router.router.engine import RouterEngine
from intent_router.router.verifier import (
    NliVerifier,
    NoVerifier,
    VerifierNotAvailableError,
    build_verifier,
    hypotheses_for,
)

REF = date(2026, 10, 22)
LABELS = {0: "contradiction", 1: "entailment", 2: "neutral"}


class FakeCrossEncoder:
    """Returns fixed [contradiction, entailment, neutral] rows, or raises."""

    def __init__(self, rows=None, error: Exception | None = None):
        self.rows = rows
        self.error = error
        self.calls: list[list[tuple[str, str]]] = []
        self.model = SimpleNamespace(config=SimpleNamespace(id2label=LABELS))

    def predict(self, pairs, apply_softmax=True, show_progress_bar=False):
        self.calls.append(list(pairs))
        if self.error:
            raise self.error
        return np.array([self.rows[i % len(self.rows)] for i in range(len(pairs))], dtype=float)


def engine_with(service, verifier) -> RouterEngine:
    e = service.engine
    return RouterEngine(registry=e.registry, index=e.index, embedder=e.embedder, pipeline=e.pipeline,
                        thresholds=e.t, settings=e.settings, corrector=e.corrector, fallback=e.fallback,
                        verifier=verifier)


def cancel_intent(service):
    return service.registry.get("ecommerce.cancel_or_return")


# --------------------------------------------------------------------------- decision rule
def test_blocks_when_contradicted_and_nothing_entailed(service):
    v = NliVerifier(FakeCrossEncoder([[0.9, 0.05, 0.05]]))
    verdict = v.check("I no longer want to cancel my order", cancel_intent(service))
    assert verdict.blocked and verdict.contradiction == 0.9 and not verdict.error


def test_one_entailed_statement_keeps_the_route(service):
    # "I want to return these shoes" contradicts "cancel an order" but entails "return an item".
    v = NliVerifier(FakeCrossEncoder([[0.95, 0.02, 0.03], [0.02, 0.97, 0.01]]))
    assert not v.check("I want to return these shoes", cancel_intent(service)).blocked


def test_checks_one_statement_per_meaning(service):
    model = FakeCrossEncoder([[0.0, 1.0, 0.0]])
    NliVerifier(model).check("Cancel my order 4421", cancel_intent(service))
    assert [h for _, h in model.calls[0]] == hypotheses_for(cancel_intent(service))
    assert len(model.calls[0]) == 4


def test_read_only_intents_are_never_checked(service):
    model = FakeCrossEncoder([[1.0, 0.0, 0.0]])
    assert NliVerifier(model).check("Find Python jobs", service.registry.get("jobs.search")) is None
    assert model.calls == []


def test_inference_failure_fails_closed(service):
    verdict = NliVerifier(FakeCrossEncoder(error=RuntimeError("boom"))).check("Cancel order 1", cancel_intent(service))
    assert verdict.blocked and verdict.error


def test_model_without_nli_labels_is_rejected():
    bad = FakeCrossEncoder([[1, 0, 0]])
    bad.model.config.id2label = {0: "LABEL_0", 1: "LABEL_1"}
    with pytest.raises(ValueError):
        NliVerifier(bad)


def test_disabled_by_default(settings):
    assert settings.verifier == "none" and isinstance(build_verifier(settings), NoVerifier)


def test_enabled_but_missing_model_fails_at_startup(settings):
    missing = settings.model_copy(update={"verifier": "nli", "verifier_model": "nobody/no-such-nli-model",
                                          "verifier_revision": None, "allow_model_download": False})
    with pytest.raises(VerifierNotAvailableError):
        build_verifier(missing)


# --------------------------------------------------------------------------- engine integration (fake model)
def test_blocked_request_is_not_executable(service):
    engine = engine_with(service, NliVerifier(FakeCrossEncoder([[0.97, 0.01, 0.02]])))
    r = engine.route("Cancel my order 4421", reference_date=REF)
    assert r.status is Status.needs_clarification and r.intent is None and r.action is None
    assert not r.requires_confirmation and r.context is None
    assert "verifier" in r.reasoning_summary
    assert r.entities["other"] == [{"type": "identifier", "value": {"kind": "order", "value": "4421"}}]


def test_verifier_does_not_run_for_read_requests(service):
    model = FakeCrossEncoder([[1.0, 0.0, 0.0]])
    r = engine_with(service, NliVerifier(model)).route("Find Python jobs in Noida", reference_date=REF)
    assert r.status is Status.success and model.calls == []


def test_verifier_checks_each_task_of_a_multi_intent_request(service):
    engine = engine_with(service, NliVerifier(FakeCrossEncoder([[0.97, 0.01, 0.02]])))
    r = engine.route("Check where my order is. Also cancel my other order 4421", reference_date=REF)
    assert r.status is Status.multi_intent
    assert r.tasks[0].intent == "ecommerce.order_status" and r.tasks[0].status is Status.success
    assert r.tasks[1].intent is None and not r.requires_confirmation


# --------------------------------------------------------------------------- real model (optional)
@pytest.fixture(scope="module")
def nli_engine(service, settings):
    try:
        verifier = build_verifier(settings.model_copy(update={"verifier": "nli"}))
    except VerifierNotAvailableError as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"NLI verifier model not downloaded: {exc}")
    return engine_with(service, verifier)


@pytest.mark.parametrize("text", [
    "I no longer want to cancel my order",
    "Keep my order 4421, no cancellation",
    "I've changed my mind, I'm keeping the shoes",
    "My order 4421 should stay active",
])
def test_real_verifier_blocks_withdrawn_requests_the_rules_miss(nli_engine, text):
    r = nli_engine.route(text, reference_date=REF)
    # Never an executable cancellation (some of these route to order status instead).
    assert r.intent != "ecommerce.cancel_or_return" and not r.requires_confirmation
    assert r.status is not Status.success or r.intent == "ecommerce.order_status"


@pytest.mark.parametrize("text", ["I no longer want to cancel my order", "Keep my order 4421, no cancellation"])
def test_real_verifier_turns_a_cancel_route_into_a_question(service, nli_engine, text):
    assert service.route(text, reference_date=REF).intent == "ecommerce.cancel_or_return"  # rules alone
    r = nli_engine.route(text, reference_date=REF)
    assert r.status is Status.needs_clarification and r.intent is None and "verifier" in r.reasoning_summary


@pytest.mark.parametrize("text, intent", [
    ("Cancel my order 4421", "ecommerce.cancel_or_return"),
    ("I want to return these shoes", "ecommerce.cancel_or_return"),
    ("I'd like to exchange this shirt for a bigger size", "ecommerce.cancel_or_return"),
    ("please cancel the laptop order I placed yesterday", "ecommerce.cancel_or_return"),
    ("Book a flight from Delhi to Mumbai tomorrow", "travel.book"),
])
def test_real_verifier_keeps_genuine_requests(nli_engine, text, intent):
    r = nli_engine.route(text, reference_date=REF)
    assert r.status is Status.success and r.intent == intent and r.requires_confirmation


# --------------------------------------------------------------------------- rules the verifier relies on
def test_negated_gerund_is_caught_by_the_rules(route):
    r = route("Please hold off on cancelling order 4421")
    assert r.status is Status.needs_clarification and r.intent is None


@pytest.mark.parametrize("text", ["keep it", "actually I'll keep it", "I changed my mind", "leave it as is",
                                  "no, I want to keep the order"])
def test_withdrawal_follow_ups_after_a_cancel(route, text):
    ctx = route("Cancel my order 4421").context
    r = route(text, ctx)
    assert r.status is Status.needs_clarification and r.intent is None and not r.requires_confirmation


@pytest.mark.parametrize("text", ["yes do it", "go ahead"])
def test_confirmation_follow_ups_still_continue(route, text):
    ctx = route("Cancel my order 4421").context
    r = route(text, ctx)
    assert r.status is Status.success and r.intent == "ecommerce.cancel_or_return" and r.requires_confirmation
