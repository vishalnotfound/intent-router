"""Follow-up messages with an explicit, caller-supplied context."""

from __future__ import annotations

from intent_router.core.schemas import ConversationContext, Status


def jobs_context(route) -> ConversationContext:
    first = route("Find Python jobs in Noida")
    assert first.context is not None
    return first.context


def test_follow_up_adds_a_filter(route):
    r = route("Only remote ones.", jobs_context(route))
    assert r.is_follow_up and r.context_used and r.resolved_by == "context"
    assert r.intent == "jobs.search"
    assert r.entities["skills"] == ["Python"] and r.entities["location"] == "Noida"
    assert r.filters["work_mode"] == "remote"


def test_follow_up_replaces_a_value(route):
    r = route("What about Delhi?", jobs_context(route))
    assert r.intent == "jobs.search" and r.entities["location"] == "Delhi"
    assert r.entities["skills"] == ["Python"]


def test_context_chains_across_turns(route):
    ctx = jobs_context(route)
    second = route("Only remote ones.", ctx)
    third = route("with 3+ years experience", second.context)
    assert third.filters["work_mode"] == "remote"
    assert third.filters["experience"]["min_years"] == 3
    assert third.entities["location"] == "Noida"


def test_additive_follow_up_unions_lists(route):
    r = route("also Java", jobs_context(route))
    assert r.entities["skills"] == ["Python", "Java"]


def test_new_request_ignores_context(route):
    r = route("Write a Python function to merge two dictionaries", jobs_context(route))
    # The point here is the context logic: a request with its own action is not a follow-up.
    # (Whether MiniLM separates write_code from refactor for this sentence is a routing
    # question; with the tuned margin it may report the two as ambiguous. See README.)
    assert not r.context_used and r.intent != "jobs.search"
    assert r.domain == "programming"


def test_follow_up_without_context_asks_instead_of_guessing(route):
    for text in ("Only remote ones.", "What about the cheaper ones?", "same but in Mumbai"):
        r = route(text)
        assert r.status is Status.needs_clarification and r.intent is None and not r.context_used, text
        assert "follow-up" in r.reasoning_summary


def test_context_naming_an_unknown_intent_is_ignored(route):
    ctx = ConversationContext(intent="does.not_exist", entities={"x": 1})
    r = route("Only remote ones.", ctx)
    assert not r.context_used


def test_router_keeps_no_hidden_state(route):
    route("Find Python jobs in Noida")
    r = route("Only remote ones.")  # no context passed -> must not remember the previous call
    assert not r.context_used and r.intent is None


def test_out_of_scope_context_is_not_continued(route):
    ctx = ConversationContext(intent="unsupported.device_or_service_control")
    r = route("please", ctx)
    assert not r.context_used and r.status is not Status.success


def test_non_anaphoric_short_message_without_context_is_routed(route):
    r = route("Not working")
    assert r.clarification_question is None or "earlier request" not in r.clarification_question


def test_and_in_value_list_replaces_instead_of_appending(route):
    r = route("what about Java and React?", jobs_context(route))
    assert r.intent == "jobs.search" and r.entities["skills"] == ["Java", "React"]
