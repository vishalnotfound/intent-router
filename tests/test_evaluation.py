"""Regression guard on the hand-written dev set.

The floors are deliberately below the measured dev numbers (see README); they
catch large regressions, not small drifts. Final accuracy is reported on the
held-out test split with scripts/evaluate.py, not asserted here.
"""

from __future__ import annotations

from intent_router.core.config import DATA_DIR
from intent_router.evaluation import evaluate, load_cases, macro_f1, value_matches


def test_metric_helpers():
    assert macro_f1([("a", "a"), ("b", "b")]) == 1.0
    assert macro_f1([("a", "b"), ("b", "a")]) == 0.0
    assert value_matches({"max": 5}, {"max": 5, "min": None})
    assert not value_matches({"max": 5}, {"max": 6})
    assert not value_matches(True, 1)
    assert value_matches(["Python", "java"], ["Java", "python"])


def test_dev_set_regression_floor(service):
    report = evaluate(service, load_cases(DATA_DIR / "eval" / "dev.jsonl"))
    assert report["macro_f1_with_reject_class"] >= 0.80, report["errors"]
    assert report["per_category"]["malformed"]["accuracy"] == 1.0
    assert report["extraction_field_accuracy"] >= 0.9
    assert report["unknown_handling"].get("rejected", 0) >= 0.7
