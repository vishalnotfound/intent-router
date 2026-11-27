"""Regressions: negation and exclusions, follow-up vs. new request, repeated intents,
unsupported directions. Each check covers status, task count and the actual payload."""

from __future__ import annotations

from datetime import date

import pytest

from intent_router.core.schemas import ConversationContext, Status
from intent_router.extractors.base import ExtractionContext
from intent_router.extractors.negation import is_bare_refusal, negated_actions, unnegated_actions
from intent_router.extractors.pipeline import map_parameters
from intent_router.router.registry import IntentDefinition, UnsupportedVariant
from intent_router.services.context_service import ACTION_VERBS


def jobs_context(route) -> ConversationContext:
    first = route("Find Python jobs in Noida")
    assert first.status is Status.success and first.context is not None
    return first.context


def assert_not_executable(r):
    assert r.status is not Status.success
    assert r.intent is None and r.action is None and r.tasks == []
    assert not r.requires_confirmation and r.side_effects == "none"
    assert r.context is None  # nothing to continue into a cancellation


# --------------------------------------------------------------------------- 1. exclusions
@pytest.mark.parametrize("text, skill, excluded", [
    ("Find Python jobs anywhere except Delhi", "Python", ["Delhi"]),
    ("Show me React jobs in any city other than Mumbai", "React", ["Mumbai"]),
    ("Python developer openings, but not in Bangalore", "Python", ["Bangalore"]),
    ("Find Python jobs excluding Delhi and Mumbai", "Python", ["Delhi", "Mumbai"]),
])
def test_excluded_location_is_never_an_included_location(route, text, skill, excluded):
    r = route(text)
    # jobs.search has no exclusion filter, so the router asks instead of searching everywhere.
    assert r.status is Status.needs_clarification and r.needs_clarification
    assert r.intent == "jobs.search" and r.tasks == []
    assert r.entities["skills"] == [skill]
    assert "location" not in r.entities and r.filters == {}
    assert [o["value"] for o in r.entities["other"] if o.get("excluded")] == excluded
    for place in excluded:
        assert place in r.clarification_question
    assert "exclude" in r.clarification_question.lower()


def test_exclusion_implied_by_an_included_value_still_succeeds(route):
    r = route("Find Python jobs in Noida, not Delhi")
    assert r.status is Status.success
    assert r.entities["location"] == "Noida" and r.entities["skills"] == ["Python"]
    assert {"type": "location", "value": "Delhi", "excluded": True} in r.entities["other"]


def test_excluded_brand_is_not_an_included_brand(route):
    r = route("Find a phone except Samsung")
    assert r.status is Status.needs_clarification and r.intent == "ecommerce.product_search"
    assert r.entities["product"] == "phone" and "brand" not in r.entities


def test_follow_up_excluding_the_current_value_drops_it_and_asks(route):
    r = route("but not in Noida", jobs_context(route))
    assert r.context_used and r.status is Status.needs_clarification and r.intent == "jobs.search"
    assert r.entities["skills"] == ["Python"] and "location" not in r.entities


def test_exclusion_filter_is_used_when_the_intent_declares_one(pipeline):
    """Where the filter contract supports exclusions, the excluded value is preserved there."""
    pipe, _ = pipeline
    intent = IntentDefinition.model_validate({
        "id": "jobs.search_with_exclusions", "label": "search for job openings", "action": "search",
        "domain": "jobs", "description": "Job search whose API accepts excluded locations.",
        "examples": ["a", "b", "c"],
        "parameters": [
            {"name": "skills", "types": ["technology"], "multiple": True},
            {"name": "location", "types": ["location"]},
            {"name": "exclude_locations", "types": ["location"], "kind": "filter", "excludes": "location"},
        ],
    })
    spans = pipe.extract("Find Python jobs anywhere except Delhi and Mumbai", ExtractionContext(date(2026, 10, 22)))
    mapped = map_parameters(spans, intent)
    assert mapped.unsupported_exclusions == []
    assert mapped.entities == {"skills": ["Python"]}
    assert mapped.filters == {"exclude_locations": ["Delhi", "Mumbai"]}


def test_excludes_must_reference_a_declared_parameter():
    with pytest.raises(ValueError):
        IntentDefinition.model_validate({
            "id": "x.y", "label": "do something", "action": "search", "domain": "x",
            "description": "Invalid exclusion reference.", "examples": ["a", "b", "c"],
            "parameters": [{"name": "exclude_city", "types": ["location"], "excludes": "city"}],
        })


# --------------------------------------------------------------------------- 1. negated actions
@pytest.mark.parametrize("text", [
    "Do not cancel order 4421",
    "Don't cancel my order 4421",
    "I don't want to cancel order 4421",
    "never cancel order 4421",
    "Please do not return the shoes I bought",
])
def test_negated_destructive_action_is_never_executable(route, text):
    r = route(text)
    assert r.status is Status.needs_clarification
    assert_not_executable(r)
    assert "not to" in r.clarification_question


def test_negated_cancel_keeps_the_order_id_for_the_caller(route):
    r = route("Do not cancel order 4421")
    assert r.entities["other"] == [{"type": "identifier", "value": {"kind": "order", "value": "4421"}}]


def test_unnegated_cancel_still_routes_with_confirmation(route):
    r = route("Cancel my order 4421")
    assert r.status is Status.success and r.intent == "ecommerce.cancel_or_return"
    assert r.entities["order_id"] == {"kind": "order", "value": "4421"} and r.requires_confirmation


@pytest.mark.parametrize("text", ["Do not cancel it", "actually don't", "No, don't do it", "never mind"])
def test_negation_after_a_cancel_request_withdraws_it(route, text):
    ctx = route("Cancel my order 4421").context
    r = route(text, ctx)
    assert r.status is Status.needs_clarification
    assert_not_executable(r)


def test_negated_action_inside_a_multi_intent_request(route):
    r = route("Check where my order is. Also don't cancel my other order 4421")
    assert r.status is Status.multi_intent and len(r.tasks) == 2
    assert r.tasks[0].intent == "ecommerce.order_status"
    assert r.tasks[1].intent is None and r.tasks[1].status is Status.needs_clarification
    assert not r.tasks[1].requires_confirmation and not r.requires_confirmation


@pytest.mark.parametrize("text, intent", [
    ("don't let me forget to submit the report by Thursday", "planning.schedule_event"),
    ("I can't find my order 4421", "ecommerce.order_status"),
    ("headphones that don't cost a fortune", "ecommerce.product_search"),
])
def test_statements_with_not_are_not_negated_requests(route, text, intent):
    r = route(text)
    assert r.status is Status.success and r.intent == intent


def test_negation_helpers():
    assert negated_actions("Do not cancel order 4421", ACTION_VERBS) == ["cancel"]
    assert negated_actions("don't go ahead and cancel it", ACTION_VERBS) == ["cancel"]
    assert negated_actions("I can't find my order", ACTION_VERBS) == []
    assert negated_actions("don't let me forget to book a cab", ACTION_VERBS) == []
    assert unnegated_actions("Don't book yet, just show me flights", ACTION_VERBS) == ["show"]
    assert unnegated_actions("Don't show my orders", ACTION_VERBS) == []  # "orders" is a noun
    assert is_bare_refusal("Actually, don't.") and is_bare_refusal("no, stop")
    assert not is_bare_refusal("don't cancel the shoes, cancel the shirt")


# --------------------------------------------------------------------------- 2. follow-ups
@pytest.mark.parametrize("text, entities", [
    ("Find Java jobs", {"skills": ["Java"]}),
    ("Search for Java developer jobs", {"skills": ["Java"], "role": "developer"}),
    ("I'm looking for Java jobs", {"skills": ["Java"]}),
])
def test_explicit_new_request_starts_a_fresh_search(route, text, entities):
    r = route(text, jobs_context(route))
    assert r.status is Status.success and r.intent == "jobs.search"
    assert not r.is_follow_up and not r.context_used and r.resolved_by == "embedding"
    assert r.entities == entities and r.filters == {}  # Noida is not inherited
    assert r.context.entities == entities and r.context.previous_message == text


def test_genuine_follow_up_keeps_previous_values(route):
    r = route("Only remote ones", jobs_context(route))
    assert r.status is Status.success and r.is_follow_up and r.context_used
    assert r.entities == {"skills": ["Python"], "location": "Noida"}
    assert r.filters == {"work_mode": "remote"}


@pytest.mark.parametrize("text, entities", [
    ("Find Java jobs there", {"skills": ["Java"], "location": "Noida"}),
    ("Find Java jobs instead", {"skills": ["Java"], "location": "Noida"}),
    ("Find Java jobs too", {"skills": ["Python", "Java"], "location": "Noida"}),
])
def test_request_verb_with_a_reference_back_is_still_a_follow_up(route, text, entities):
    r = route(text, jobs_context(route))
    assert r.status is Status.success and r.context_used and r.entities == entities


# --------------------------------------------------------------------------- 3. repeated intent
@pytest.mark.parametrize("text, depends", [
    ("Find Java developer jobs in Delhi and then search for Python developer jobs in Mumbai", [[], [0]]),
    ("Search for Java jobs in Delhi, then look for Python jobs in Mumbai", [[], [0]]),
    ("Find Java developer jobs in Delhi. After that, find Python developer jobs in Mumbai", [[], [0]]),
    ("Find Java jobs in Delhi and Python jobs in Mumbai", [[], []]),
    ("Find Java jobs in Delhi and also Python jobs in Mumbai", [[], []]),
])
def test_two_searches_with_their_own_parameters(route, text, depends):
    r = route(text)
    assert r.status is Status.multi_intent and r.intent is None
    assert len(r.tasks) == 2 and [t.index for t in r.tasks] == [0, 1]
    assert [t.intent for t in r.tasks] == ["jobs.search", "jobs.search"]
    assert all(t.status is Status.success for t in r.tasks)
    assert r.tasks[0].entities["skills"] == ["Java"] and r.tasks[0].entities["location"] == "Delhi"
    assert r.tasks[1].entities["skills"] == ["Python"] and r.tasks[1].entities["location"] == "Mumbai"
    assert all("other" not in t.entities for t in r.tasks)
    assert [t.depends_on for t in r.tasks] == depends


def test_explicit_search_pair_keeps_role(route):
    r = route("Find Java developer jobs in Delhi and then search for Python developer jobs in Mumbai")
    assert [t.entities.get("role") for t in r.tasks] == ["developer", "developer"]


@pytest.mark.parametrize("text, entities", [
    ("Find jobs requiring Python and Java", {"skills": ["Python", "Java"]}),
    ("Find Python and Java jobs in Noida", {"skills": ["Python", "Java"], "location": "Noida"}),
])
def test_one_search_with_a_list_of_skills(route, text, entities):
    r = route(text)
    assert r.status is Status.success and r.intent == "jobs.search" and r.tasks == []
    assert r.entities == entities


def test_comparison_is_not_split_into_searches(route):
    r = route("Compare Python jobs in Noida and Java jobs in Delhi")
    assert r.status is not Status.multi_intent and r.tasks == []


# --------------------------------------------------------------------------- 4. unsupported direction
@pytest.mark.parametrize("text", [
    "Find candidates with Python experience in Noida",
    "I want to hire a Python developer in Noida",
    "Show me resumes of Java developers in Pune",
    "Search for applicants with React skills",
    "Find a candidate with 5 years of Java experience",
])
def test_candidate_search_is_not_a_job_search(route, text):
    r = route(text)
    assert r.status is Status.unknown and r.intent == "unknown" and r.tasks == []
    assert r.action == "unknown" and r.context is None
    assert "candidates" in r.clarification_question


def test_candidate_search_follow_up_is_not_continued(route):
    r = route("Find candidates instead", jobs_context(route))
    assert r.status is Status.unknown and not r.context_used


@pytest.mark.parametrize("text", [
    "who is hiring React engineers in Chennai",
    "I'm a candidate looking for Python jobs in Noida",
    "Find jobs for candidates with Python experience",
])
def test_job_seekers_still_get_job_search(route, text):
    r = route(text)
    assert r.status is Status.success and r.intent == "jobs.search"


def test_unsupported_variant_patterns_are_validated():
    with pytest.raises(ValueError):
        UnsupportedVariant(name="bad", label="do bad things", patterns=["(unclosed"])
