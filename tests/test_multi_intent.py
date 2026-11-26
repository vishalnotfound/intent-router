"""Several actions in one message."""

from __future__ import annotations

from app.core.schemas import Status


def test_search_then_compare_with_dependency(route):
    r = route("Find Python jobs in Noida and compare their salaries")
    assert r.status is Status.multi_intent and r.intent is None
    assert [t.intent for t in r.tasks] == ["jobs.search", "general.compare"]
    assert [t.action for t in r.tasks] == ["search", "compare"]
    assert r.tasks[1].depends_on == [0]  # "their" refers to the jobs found first
    assert r.tasks[0].entities == {"skills": ["Python"], "location": "Noida"}
    assert r.tasks[1].entities.get("criteria") == ["salaries"]


def test_sequenced_tasks(route):
    r = route("Summarize this article and then translate it to Hindi")
    assert [t.intent for t in r.tasks] == ["document.summarize", "text.translate"]
    assert r.tasks[1].depends_on == [0]
    assert r.tasks[1].entities["target_language"] == "Hindi"


def test_independent_tasks_have_no_dependency(route):
    r = route("Find a cheap laptop for college. Also, what's the capital of Peru?")
    assert [t.intent for t in r.tasks] == ["ecommerce.product_search", "general.question"]
    assert r.tasks[1].depends_on == []


def test_and_between_nouns_does_not_split(route):
    r = route("Compare Python and Java for backend development")
    assert r.status is not Status.multi_intent
    assert r.intent == "general.compare" and r.entities["items"] == ["Python", "Java"]


def test_greeting_is_not_a_separate_task(route):
    r = route("Hi! Can you summarize this report for me?")
    assert r.status is not Status.multi_intent and r.intent == "document.summarize"


def test_multi_intent_tasks_carry_their_own_safety_flags(route):
    r = route("Check where my order is. Also cancel my other order 4421")
    assert [t.intent for t in r.tasks] == ["ecommerce.order_status", "ecommerce.cancel_or_return"]
    assert r.tasks[1].requires_confirmation and not r.tasks[0].requires_confirmation
    assert r.requires_confirmation  # any task needing confirmation flags the whole result
