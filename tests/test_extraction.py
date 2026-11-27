"""Deterministic extraction (no model needed)."""

from __future__ import annotations

from datetime import date

import pytest

from intent_router.extractors.base import ExtractionContext
from intent_router.extractors.pipeline import map_parameters
from intent_router.extractors.rules import extract_dates, extract_money, extract_times
from intent_router.router.normalize import mask_entities, normalize_text, prepare

CTX = ExtractionContext(reference_date=date(2026, 10, 22))  # Thursday


def extract(pipeline_fixture, text, intent_id):
    pipe, registry = pipeline_fixture
    spans = pipe.extract(text, CTX)
    return map_parameters(spans, registry.get(intent_id))


def money(text):
    return [s.value for s in extract_money(text) if s.type in ("money", "money_constraint")]


# ----------------------------------------------------------------------------- money
def test_price_ceiling_with_currency():
    vals = money("shoes under 3000 rupees")
    assert {"min": None, "max": 3000, "currency": "INR"}.items() <= next(v for v in vals if "max" in v).items()


def test_shorthand_amounts_do_not_invent_currency():
    v = next(v for v in money("laptop under 50k") if "max" in v)
    assert v["max"] == 50000 and v["currency"] is None


def test_lpa_is_yearly_inr():
    v = next(v for v in money("jobs paying above 12 LPA") if "min" in v)
    assert v == {**v, "min": 1200000, "currency": "INR", "period": "year"}


def test_money_range():
    v = next(v for v in money("between 30k and 50k") if v.get("min") and v.get("max"))
    assert (v["min"], v["max"]) == (30000, 50000)


def test_plain_numbers_with_units_are_not_money():
    assert money("under 2 kg") == []
    assert money("above 100 employees") == []


def test_4k_tv_is_not_four_thousand():
    assert not any(v.get("max") == 4000 or v.get("amount") == 4000 for v in money("55 inch 4K TV"))


# ----------------------------------------------------------------------------- dates and times
def dates(text):
    return {s.value.get("date") or (s.value["start"], s.value["end"]) for s in extract_dates(text, CTX)}


@pytest.mark.parametrize("text, expected", [
    ("tomorrow", "2026-10-23"),
    ("yesterday", "2026-10-21"),
    ("on Friday", "2026-10-23"),
    ("next Tuesday", "2026-10-27"),
    ("2026-12-01", "2026-12-01"),
    ("15th March 2027", "2027-03-15"),
    ("31 March", "2027-03-31"),  # already past this year -> next occurrence
    ("3 days ago", "2026-10-19"),
])
def test_dates(text, expected):
    assert expected in dates(text)


def test_date_ranges():
    assert ("2026-10-26", "2026-11-01") in dates("next week")
    assert ("2026-12-01", "2026-12-31") in dates("in December")
    assert ("2026-10-24", "2026-10-25") in dates("this weekend")


def test_numeric_date_uses_configured_order():
    assert "2026-11-12" in dates("12/11/2026")  # DMY default
    mdy = ExtractionContext(reference_date=date(2026, 10, 22), date_order="MDY")
    assert "2026-12-11" in {s.value["date"] for s in extract_dates("12/11/2026", mdy)}


def test_may_as_verb_is_not_a_date():
    assert dates("you may go") == set()


@pytest.mark.parametrize("text, expected", [("6 pm", "18:00"), ("10:30 am", "10:30"), ("noon", "12:00"),
                                            ("at 7 tomorrow morning", "07:00"), ("16:45", "16:45")])
def test_times(text, expected):
    assert expected in {s.value.get("time") for s in extract_times(text) if s.type == "time"}


# ----------------------------------------------------------------------------- parameter mapping
def test_job_search_extraction(pipeline):
    m = extract(pipeline, "Find Python jobs in Noida", "jobs.search")
    assert m.entities == {"skills": ["Python"], "location": "Noida"}
    assert "role" in m.optional_missing  # not invented


def test_job_filters(pipeline):
    m = extract(pipeline, "senior backend developer jobs in Bangalore, remote, 5+ years experience, above 25 LPA",
                "jobs.search")
    assert m.entities["role"] == "backend developer"
    assert m.filters["work_mode"] == "remote"
    assert m.filters["experience"]["min_years"] == 5 and m.filters["experience"]["level"] == "senior"
    assert m.filters["salary"]["min"] == 2500000


def test_laptop_example_preserves_wording_and_does_not_invent_budget(pipeline):
    m = extract(pipeline, "Find a cheap laptop for programming that I can upgrade later", "ecommerce.product_search")
    assert m.entities["product"] == "laptop"
    assert m.entities["use_case"] == "programming"
    assert m.entities["requirements"] == ["I can upgrade later"]
    assert m.filters["price_preference"] == "low"
    assert "budget" not in m.filters and "budget" in m.optional_missing


def test_travel_roles(pipeline):
    m = extract(pipeline, "get me two bus tickets from Pune to Bangalore next Friday", "travel.book")
    assert m.entities["origin"] == "Pune" and m.entities["destination"] == "Bangalore"
    assert m.entities["mode"] == "bus" and m.entities["travellers"] == {"count": 2}


def test_translation_languages_and_quoted_text(pipeline):
    m = extract(pipeline, "translate 'where is the station' from English to Japanese", "text.translate")
    assert m.entities["source_language"] == "English"
    assert m.entities["target_language"] == "Japanese"
    assert m.entities["text"] == "where is the station"


def test_required_field_missing(pipeline):
    m = extract(pipeline, "translate this sentence please", "text.translate")
    assert m.missing == ["target_language"] and m.clarification


def test_required_any_group(pipeline):
    m = extract(pipeline, "remind me to call mom", "planning.schedule_event")
    assert m.missing == ["date|time"]
    assert m.entities["title"] == "call mom"


def test_comparison_items(pipeline):
    m = extract(pipeline, "Samsung S24 vs OnePlus 12 camera comparison", "ecommerce.compare_products")
    assert m.entities["items"] == ["Samsung S24", "OnePlus 12"] and m.entities["criteria"] == ["camera"]


def test_identifiers_and_errors(pipeline):
    assert extract(pipeline, "track order #A12345", "ecommerce.order_status").entities["order_id"]["value"] == "A12345"
    m = extract(pipeline, "Why does my script throw a KeyError: 'name'?", "programming.debug")
    assert m.entities["error"].startswith("KeyError")


def test_fuzzy_gazetteer_match(pipeline):
    m = extract(pipeline, "pyhton jobs in hydrabad", "jobs.search")
    assert m.entities["skills"] == ["Python"] and m.entities["location"] == "Hyderabad"


def test_unmapped_values_are_reported_not_dropped(pipeline):
    m = extract(pipeline, "Draft an email to my manager asking for leave next week", "document.create")
    assert any(o["type"] == "date_range" for o in m.entities["other"])


# ----------------------------------------------------------------------------- normalization
def test_normalization_is_safe():
    assert normalize_text("  Hello  “world”​  ") == 'Hello "world"'


def test_prepare_separates_pasted_content():
    p = prepare("Summarize this: The quarterly report shows revenue grew by ten percent while costs fell.")
    assert p.instruction == "Summarize this"
    assert p.content.startswith("The quarterly report")


def test_prepare_keeps_filter_lists_in_instruction():
    p = prepare("Find jobs: Python, Noida, remote, full-time, five years experience, above 10 LPA")
    assert p.content is None and "Noida" in p.instruction


def test_masking_replaces_places_and_technologies(pipeline):
    pipe, _ = pipeline
    text = "Find Python jobs in Noida"
    assert mask_entities(text, pipe.extract(text, CTX)) == "Find programming jobs in the city"


def test_extreme_dates_do_not_crash():
    for text in ("10000 years ago", "9999999 days ago", "last 9999999 days", "in January 0000"):
        assert extract_dates(text, CTX) == []


def test_exact_date_not_clobbered_by_month_only(pipeline):
    m = extract(pipeline, "Remind me on 31st of March", "planning.schedule_event")
    assert m.entities["date"]["date"] == "2027-03-31"
    m2 = extract(pipeline, "Remind me by March 10", "planning.schedule_event")
    assert m2.entities["date"]["date"] == "2027-03-10"
    since = extract_dates("Orders since March", CTX)
    assert since[0].value["start"] == "2026-03-01"


def test_dotted_technology_not_hijacked_as_file(pipeline):
    m = extract(pipeline, "Write a Node.js API", "programming.write_code")
    assert m.entities["language"] == ["Node.js"]
    assert "other" not in m.entities


def test_language_overlay_requires_language_after_to(pipeline):
    m = extract(pipeline, "How do I say 'hello' in Japanese to my friend?", "text.translate")
    assert m.entities["target_language"] == "Japanese"
    assert "source_language" not in m.entities


def test_brand_and_product_both_preserved(pipeline):
    m = extract(pipeline, "Find boAt headphones under 2000", "ecommerce.product_search")
    assert m.entities["brand"] == ["boAt"]
    assert m.entities["product"] == "headphones"


def test_unknown_city_not_fuzzy_corrupted(pipeline):
    m = extract(pipeline, "Find Python jobs in Raipur", "jobs.search")
    assert m.entities["location"] == "Raipur"


def test_money_unit_words_and_ranges():
    assert money("under 2k words") == []
    assert money("10k steps") == []
    r = next(v for v in money("budget 5000 to 10000") if v.get("min") == 5000)
    assert r["max"] == 10000
    r2 = next(v for v in money("between $500 and 1000") if v.get("min") == 500)
    assert r2["max"] == 1000 and r2["currency"] == "USD"


def test_conflicting_dicts_not_merged_or_dropped(pipeline):
    m = extract(pipeline, "Find laptops under $500 and phones above 20000 INR", "ecommerce.product_search")
    assert m.filters["budget"]["max"] == 500 and m.filters["budget"]["currency"] == "USD"
    assert m.filters["budget"]["min"] is None
    assert any(o["type"] == "money_constraint" and o["value"]["min"] == 20000 for o in m.entities["other"])

    d = extract(pipeline, "Book a flight on Friday, return next Monday", "travel.book")
    assert d.entities["date"]["date"] == "2026-10-23"
    assert any(o["type"] == "date" and o["value"]["date"] == "2026-10-26" for o in d.entities["other"])


def test_multiword_comparison_criteria(pipeline):
    m = extract(pipeline, "Samsung S24 vs OnePlus 12 battery life", "ecommerce.compare_products")
    assert m.entities["items"] == ["Samsung S24", "OnePlus 12"]
    assert m.entities["criteria"] == ["battery life"]
