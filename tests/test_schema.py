"""Schema and registry validation (no model needed)."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from intent_router.core.schemas import SCHEMA_VERSION, ConversationContext, RouteResult, Status
from intent_router.router.registry import IntentRegistry, RegistryError, RegistryFile


def minimal_result(**overrides):
    data = {"status": "success", "intent": "jobs.search", "action": "search", "confidence": 0.5}
    data.update(overrides)
    return RouteResult(**data)


def test_defaults_are_consistent():
    r = minimal_result()
    assert r.version == SCHEMA_VERSION == "1.0"
    assert r.entities == {} and r.filters == {} and r.missing_fields == [] and r.tasks == []
    assert r.alternatives == [] and r.debug is None and r.context is None


@pytest.mark.parametrize("value", [-0.01, 1.01, float("nan")])
def test_confidence_must_be_in_unit_interval(value):
    with pytest.raises(ValidationError):
        minimal_result(confidence=value)


def test_unknown_fields_are_rejected():
    with pytest.raises(ValidationError):
        minimal_result(executed=True)


def test_status_values():
    assert {s.value for s in Status} == {
        "success", "needs_clarification", "ambiguous", "unknown", "multi_intent", "invalid_input"}
    with pytest.raises(ValidationError):
        minimal_result(status="done")


def test_entities_must_be_json_values():
    with pytest.raises(ValidationError):
        minimal_result(entities={"x": object()})
    r = minimal_result(entities={"skills": ["Python"], "salary": {"min": 1, "max": None}})
    assert json.loads(r.model_dump_json())["entities"]["salary"]["max"] is None


def test_json_round_trip():
    r = minimal_result(entities={"location": "Noida"}, reasoning_summary="ok")
    again = RouteResult.model_validate_json(r.model_dump_json())
    assert again == r


def test_context_limits():
    with pytest.raises(ValidationError):
        ConversationContext(entities={f"k{i}": i for i in range(51)})
    with pytest.raises(ValidationError):
        ConversationContext(previous_message="x" * 2001)
    with pytest.raises(ValidationError):
        ConversationContext(unexpected="field")
    assert ConversationContext(intent="jobs.search").entities == {}


def _file(intents, actions=None):
    actions = actions or {"search": {"examples": ["find one", "look for some", "search for it"]}}
    return RegistryFile.model_validate({"actions": actions, "intents": intents})


def _intent(**overrides):
    data = {"id": "x.search", "label": "search things", "action": "search", "domain": "x",
            "description": "Search for things in a test domain.", "examples": ["a b", "c d", "e f"]}
    data.update(overrides)
    return data


def test_registry_loads_real_files(settings):
    reg = IntentRegistry.load(settings.intents_file, settings.domain_dirs)
    assert len(reg) >= 25
    assert {"jobs", "ecommerce", "travel", "programming", "documents", "general", "planning"} <= set(reg.domains)
    for intent in reg.intents:
        assert intent.action in reg.actions


def test_registry_rejects_duplicate_ids():
    with pytest.raises(RegistryError):
        IntentRegistry([_file([_intent(), _intent()])])


def test_registry_rejects_undeclared_action():
    with pytest.raises(RegistryError):
        IntentRegistry([_file([_intent(action="teleport")])])


def test_registry_rejects_reserved_action():
    with pytest.raises(RegistryError):
        IntentRegistry([_file([_intent()], actions={"unknown": {"examples": ["a", "b", "c"]}})])


def test_registry_rejects_bad_definitions():
    with pytest.raises(ValidationError):
        _file([_intent(id="NoDot")])
    with pytest.raises(ValidationError):
        _file([_intent(examples=["only one"])])
    with pytest.raises(ValidationError):
        _file([_intent(required_any=[["missing_param"]])])


def test_destructive_intents_require_confirmation_by_default():
    f = _file([_intent(side_effects="destructive")])
    assert f.intents[0].requires_confirmation is True
    f = _file([_intent(side_effects="read")])
    assert f.intents[0].requires_confirmation is False
