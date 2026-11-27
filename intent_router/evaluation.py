"""Offline evaluation of the router on hand-written JSONL cases.

Label conventions in the eval files:
- an intent id: the request should be routed to that intent
- "unknown" / "ambiguous" / "clarify": the router should NOT commit to an intent
  (any of unknown, ambiguous, or a clarification question counts as correct)
- "invalid": empty or non-linguistic input
- "multi": several tasks, listed in expected.tasks (and optionally expected.depends_on)

Optional stricter expectations (counted only in ``strict_accuracy``, so the other
metrics stay comparable with earlier reports):
- expected.status: the exact response status, e.g. "needs_clarification"
- expected.absent: fields that must NOT be returned, e.g. ["entities.location"]
- expected.task_payloads: for "multi", one {"entities": ..., "filters": ...} per task
"""

from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np

from intent_router.core.schemas import ConversationContext, RouteResult, Status

REJECT_LABELS = {"unknown", "ambiguous", "clarify", "invalid"}
DEFAULT_REFERENCE_DATE = date(2026, 10, 22)  # eval expectations for relative dates assume this "today"


def load_cases(path: str | Path) -> list[dict[str, Any]]:
    cases = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            cases.append(json.loads(line))
    return cases


def predicted_label(result: RouteResult, registry) -> str:
    if result.status is Status.invalid_input:
        return "invalid"
    if result.status is Status.multi_intent:
        return "multi"
    if result.status is Status.unknown:
        return "unknown"
    if result.status is Status.ambiguous:
        return "ambiguous"
    if not result.intent:
        return "clarify"
    intent = registry.get(result.intent)
    if intent is not None and intent.always_clarify:
        return "clarify"
    return result.intent


def is_low_confidence_guess(result: RouteResult, registry) -> bool:
    if result.status is not Status.needs_clarification or not result.intent or result.missing_fields:
        return False
    intent = registry.get(result.intent)
    return intent is not None and not intent.always_clarify and result.resolved_by == "embedding"


def _norm(v: Any) -> Any:
    return v.lower().strip() if isinstance(v, str) else v


def value_matches(gold: Any, pred: Any) -> bool:
    """Dicts match on the gold keys only; lists as unordered sets; strings case-insensitively."""
    if isinstance(gold, dict):
        return isinstance(pred, dict) and all(k in pred and value_matches(v, pred[k]) for k, v in gold.items())
    if isinstance(gold, list):
        if not isinstance(pred, list):
            return False
        return sorted(map(lambda x: json.dumps(_norm(x), sort_keys=True), gold)) == sorted(
            map(lambda x: json.dumps(_norm(x), sort_keys=True), pred))
    if isinstance(gold, bool) or isinstance(pred, bool):
        return isinstance(gold, bool) and isinstance(pred, bool) and gold is pred
    if isinstance(gold, (int, float)) and isinstance(pred, (int, float)):
        return abs(gold - pred) < 1e-6
    return _norm(gold) == _norm(pred)


def payload_matches(expected: dict[str, Any], obj: Any) -> bool:
    """The status (if given) and every gold entity/filter match, and no field listed in
    expected["absent"] is present. ``obj`` is a RouteResult or one of its tasks."""
    if "status" in expected and obj.status.value != expected["status"]:
        return False
    for section in ("entities", "filters"):
        values = getattr(obj, section)
        if not all(key in values and value_matches(gold, values[key])
                   for key, gold in expected.get(section, {}).items()):
            return False
    for path in expected.get("absent", []):
        section, _, key = path.partition(".")
        if key in getattr(obj, section, {}):
            return False
    return True


def strict_case_ok(case: dict[str, Any], result: RouteResult, label_ok: bool) -> bool:
    """Label correct, plus the explicit status and the full payload (entities, filters, tasks)."""
    exp = case["expected"]
    if not label_ok:
        return False
    if exp["label"] == "multi" or exp["label"] in REJECT_LABELS:
        # Entities of rejected or multi-intent results are not part of the gold label.
        top = {k: exp[k] for k in ("status", "absent") if k in exp}
        if not payload_matches(top, result):
            return False
        payloads = exp.get("task_payloads")
        return payloads is None or (len(payloads) == len(result.tasks)
                                    and all(payload_matches(p, t) for p, t in zip(payloads, result.tasks)))
    return payload_matches(exp, result)


def macro_f1(pairs: list[tuple[str, str]]) -> float:
    labels = sorted({g for g, _ in pairs})
    f1s = []
    for label in labels:
        tp = sum(1 for g, p in pairs if g == label and p == label)
        fp = sum(1 for g, p in pairs if g != label and p == label)
        fn = sum(1 for g, p in pairs if g == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def evaluate(service, cases: list[dict[str, Any]], reference_date: date = DEFAULT_REFERENCE_DATE) -> dict[str, Any]:
    registry = service.registry
    per_category: dict[str, list[bool]] = defaultdict(list)
    latencies: list[float] = []
    errors: list[dict[str, Any]] = []
    f1_pairs: list[tuple[str, str]] = []
    intent_hits: list[bool] = []
    false_rejections = 0
    unknown_outcomes: Counter = Counter()
    field_total = field_hits = 0
    field_errors: list[dict[str, Any]] = []
    strict_hits: list[bool] = []
    strict_errors: list[dict[str, Any]] = []
    status_checked = 0

    for case in cases:
        exp = case["expected"]
        gold = exp["label"]
        ctx = ConversationContext(**case["context"]) if case.get("context") else None
        t0 = time.perf_counter()
        try:
            result = service.route(case["text"], ctx, reference_date=reference_date)
        except Exception as exc:  # noqa: BLE001 - record and continue
            per_category[case["category"]].append(False)
            strict_hits.append(False)
            errors.append({"id": case["id"], "text": case["text"], "gold": gold, "pred": f"error: {exc}"})
            continue
        latencies.append((time.perf_counter() - t0) * 1000)
        pred = predicted_label(result, registry)

        if gold == "multi":
            task_ids = [t.intent for t in result.tasks]
            ok = pred == "multi" and task_ids == exp["tasks"]
            if ok and "depends_on" in exp:
                ok = [t.depends_on for t in result.tasks] == exp["depends_on"]
        elif gold == "invalid":
            ok = pred == "invalid"
        elif gold in REJECT_LABELS:
            ok = pred in REJECT_LABELS
        else:
            ok = pred == gold
            if "context_used" in exp:
                ok = ok and result.context_used == exp["context_used"]
            intent_hits.append(pred == gold)
            if pred in REJECT_LABELS:
                false_rejections += 1

        if gold == "unknown":
            if pred in REJECT_LABELS:
                unknown_outcomes["rejected"] += 1
            elif is_low_confidence_guess(result, registry):
                unknown_outcomes["asked_to_confirm_a_guess"] += 1
            else:
                unknown_outcomes["wrongly_accepted"] += 1

        if gold not in ("multi", "invalid"):
            f1_pairs.append(("reject" if gold in REJECT_LABELS else gold,
                             "reject" if pred in REJECT_LABELS else pred))

        if pred == gold and gold not in REJECT_LABELS and gold != "multi":
            for section in ("entities", "filters"):
                for key, gold_value in exp.get(section, {}).items():
                    field_total += 1
                    pred_value = getattr(result, section).get(key)
                    if value_matches(gold_value, pred_value):
                        field_hits += 1
                    else:
                        field_errors.append({"id": case["id"], "field": f"{section}.{key}", "gold": gold_value,
                                             "pred": pred_value})

        strict = strict_case_ok(case, result, ok)
        status_checked += "status" in exp
        strict_hits.append(strict)
        if ok and not strict:
            strict_errors.append({"id": case["id"], "text": case["text"], "gold": gold,
                                  "expected_status": exp.get("status"), "status": result.status.value,
                                  "entities": result.entities, "filters": result.filters,
                                  "tasks": [{"intent": t.intent, "entities": t.entities, "filters": t.filters}
                                            for t in result.tasks] or None})

        per_category[case["category"]].append(ok)
        if not ok:
            errors.append({"id": case["id"], "category": case["category"], "text": case["text"], "gold": gold,
                           "pred": pred, "status": result.status.value, "confidence": result.confidence,
                           "tasks": [t.intent for t in result.tasks] or None})

    in_scope = len(intent_hits)
    n_unknown = sum(unknown_outcomes.values())
    all_ok = [x for v in per_category.values() for x in v]
    return {
        "cases": len(cases),
        "overall_accuracy": round(float(np.mean(all_ok)), 4) if all_ok else 0.0,
        "intent_accuracy": round(float(np.mean(intent_hits)), 4) if intent_hits else 0.0,
        "macro_f1_with_reject_class": round(macro_f1(f1_pairs), 4),
        "false_rejection_rate": round(false_rejections / in_scope, 4) if in_scope else 0.0,
        "unknown_handling": {k: round(v / n_unknown, 4) for k, v in unknown_outcomes.items()} if n_unknown else {},
        "extraction_field_accuracy": round(field_hits / field_total, 4) if field_total else None,
        "extraction_fields_checked": field_total,
        # Label AND explicit status (when labelled) AND every gold field / task payload.
        "strict_accuracy": round(float(np.mean(strict_hits)), 4) if strict_hits else 0.0,
        "strict_status_checked": status_checked,
        "per_category": {k: {"n": len(v), "accuracy": round(float(np.mean(v)), 4)} for k, v in sorted(per_category.items())},
        "latency_ms": {
            "p50": round(float(np.percentile(latencies, 50)), 2),
            "p95": round(float(np.percentile(latencies, 95)), 2),
            "mean": round(float(np.mean(latencies)), 2),
            "total_latency": round(float(sum(latencies)), 2)    
        } if latencies else {},
        "errors": errors,
        "field_errors": field_errors,
        "strict_errors": strict_errors,
    }


def format_report(report: dict[str, Any], title: str = "") -> str:
    lines = [f"== {title} ==" if title else "== evaluation =="]
    lines.append(f"cases: {report['cases']}   overall accuracy: {report['overall_accuracy']:.3f}")
    lines.append(f"intent accuracy (in-scope): {report['intent_accuracy']:.3f}   "
                 f"macro-F1 (intents + reject class): {report['macro_f1_with_reject_class']:.3f}")
    lines.append(f"false rejection rate (in-scope sent to unknown/ambiguous/clarify): {report['false_rejection_rate']:.3f}")
    lines.append(f"unknown requests: {report['unknown_handling']}")
    if report["extraction_field_accuracy"] is not None:
        lines.append(f"extraction field accuracy: {report['extraction_field_accuracy']:.3f} "
                     f"({report['extraction_fields_checked']} fields, on correctly routed cases)")
    lines.append(f"strict accuracy (label + status + full payload): {report['strict_accuracy']:.3f} "
                 f"({report['strict_status_checked']} cases with an explicit status)")
    lines.append(f"latency per request (ms): {report['latency_ms']}")
    lines.append("per category:")
    for cat, v in report["per_category"].items():
        lines.append(f"  {cat:18s} n={v['n']:3d}  acc={v['accuracy']:.3f}")
    return "\n".join(lines)
