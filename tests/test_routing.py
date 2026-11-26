"""Semantic routing with the real embedding model."""

from __future__ import annotations

import pytest

from app.core.schemas import RouteResult, Status


@pytest.mark.parametrize("text", [
    "Find Python jobs in Noida.",
    "Are there any Python developer openings around Noida?",
    "Show me vacancies for Python developers near Noida.",
])
def test_paraphrases_share_action_and_target(route, text):
    r = route(text)
    assert r.status is Status.success
    assert (r.intent, r.action, r.target, r.domain) == ("jobs.search", "search", "job_openings", "jobs")
    assert r.entities["skills"] == ["Python"] and r.entities["location"] == "Noida"


def test_parameters_differ_when_wording_differs(route):
    assert "role" not in route("Find Python jobs in Noida.").entities
    assert route("Show me vacancies for Python developers near Noida.").entities["role"] == "developer"


@pytest.mark.parametrize("text, intent", [
    ("What is photosynthesis?", "general.question"),
    ("Search the web for the latest news on electric scooters", "web.search"),
    ("Summarize this article in three sentences", "document.summarize"),
    ("Translate 'good morning' into German", "text.translate"),
    ("What is 15% of 3200?", "math.calculate"),
    ("Write a Python function that checks for palindromes", "programming.write_code"),
    ("Why does my Java program throw a NullPointerException?", "programming.debug"),
    ("Compare the iPhone 15 and the Pixel 8", "ecommerce.compare_products"),
    ("Where is my order #A1234?", "ecommerce.order_status"),
    ("Cancel my order 12345", "ecommerce.cancel_or_return"),
    ("Plan a 3-day trip to Goa", "planning.create_plan"),
    ("Remind me to call mom at 6 pm tomorrow", "planning.schedule_event"),
    ("Book a flight from Delhi to Mumbai next Friday", "travel.book"),
    ("My wifi keeps disconnecting", "support.troubleshoot"),
    ("hello there!", "conversation.chitchat"),
])
def test_core_intents(route, text, intent):
    r = route(text)
    assert r.intent == intent, (r.status, r.intent, r.alternatives)
    assert r.status in (Status.success, Status.needs_clarification)


def test_action_verb_matters_not_only_topic(route):
    """"Find" and "compare" requests about the same products route to different actions."""
    assert route("Find a laptop under 60000").action == "search"
    assert route("Compare the Dell XPS 13 and the MacBook Air").action == "compare"


@pytest.mark.parametrize("text, intent", [
    ("pyhton jbos in noida", "jobs.search"),
    ("summarise this artcle", "document.summarize"),
    ("my laptp wont turn on", "support.troubleshoot"),
])
def test_typos(route, text, intent):
    assert route(text).intent == intent


def test_laptop_request_extracts_without_inventing_budget(route):
    r = route("Find a cheap laptop for programming that I can upgrade later")
    assert (r.intent, r.action) == ("ecommerce.product_search", "search")
    assert r.entities["product"] == "laptop" and r.entities["use_case"] == "programming"
    assert r.entities["requirements"] == ["I can upgrade later"]
    assert "budget" not in r.filters and "budget" in r.optional_missing_fields


def test_missing_required_field_asks_for_it(route):
    r = route("Translate this paragraph please")
    assert r.intent == "text.translate"
    assert r.status is Status.needs_clarification and r.missing_fields == ["target_language"]
    assert r.needs_clarification and r.clarification_question


def test_result_is_valid_and_bounded(route):
    r = route("Find Python jobs in Noida", include_debug=True)
    RouteResult.model_validate(r.model_dump())  # round-trips through validation
    assert 0.0 <= r.confidence <= 1.0
    assert all(-1 <= a.score <= 1 for a in r.alternatives)
    assert r.debug.timings_ms["total_ms"] > 0 and r.debug.routing_text == "Find programming jobs in the city"
    assert r.context is not None and r.context.intent == "jobs.search"


def test_registry_examples_route_to_their_own_intent(service):
    """Every example phrase in the registry is labelled data: with the shipped weights and
    margin, each must score highest for its own intent, clear of the runner-up."""
    import numpy as np

    index = service.index
    sb = index.score(index._emb["pos"], index._emb["head"] if index.head_weight > 0 else None)
    owners = np.concatenate([np.full(sl.stop - sl.start, i) for i, sl in enumerate(index._pos_slices)])
    ordered = np.sort(sb.final, axis=1)
    bad = [index.example_texts[k] for k in range(len(owners))
           if sb.final[k].argmax() != owners[k] or ordered[k, -1] - ordered[k, -2] < service.thresholds.margin]
    assert not bad, bad


def test_side_effects_are_flagged_for_downstream_authorization(route):
    r = route("Cancel my order 12345")
    assert r.side_effects == "destructive" and r.requires_confirmation is True
    r = route("Book a flight from Delhi to Mumbai next Friday")
    assert r.side_effects == "financial" and r.requires_confirmation is True
    assert route("What is photosynthesis?").requires_confirmation is False
