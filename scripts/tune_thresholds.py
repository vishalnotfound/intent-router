"""Tune scoring weights and decision thresholds on the DEV split only.

Stage 1 searches scoring weights (action/negative/head weights, top-k) with the
current thresholds; stage 2 searches thresholds with the best weights. The
objective is macro-F1 over intents plus a "reject" class, so wrongly accepting
out-of-scope requests and wrongly rejecting in-scope ones both cost.

Final numbers must come from the held-out test split (scripts/evaluate.py
--split test). Tuning on dev and reporting dev numbers would overstate accuracy.

Usage:
    python scripts/tune_thresholds.py
    python scripts/tune_thresholds.py --model BAAI/bge-small-en-v1.5 --write
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from intent_router.core.config import DATA_DIR, Thresholds, get_settings  # noqa: E402
from intent_router.core.logging import setup_logging  # noqa: E402
from intent_router.evaluation import evaluate, load_cases  # noqa: E402
from intent_router.services.intent_service import IntentService  # noqa: E402


class CachedEncoder:
    """Embeddings do not depend on weights or thresholds, so compute each text once."""

    def __init__(self, embedder):
        self.embedder = embedder
        self.cache: dict[str, np.ndarray] = {}
        self._original = embedder.encode

    def __call__(self, texts, *, is_query=True, batch_size=32):
        missing = [t for t in texts if t not in self.cache]
        if missing:
            for t, v in zip(missing, self._original(missing, is_query=is_query, batch_size=batch_size)):
                self.cache[t] = v
        return np.stack([self.cache[t] for t in texts]) if texts else self._original(texts)


def self_consistency(index, margin: float) -> float:
    """Share of the registry's own example phrases that route to their own intent with
    at least `margin` over the runner-up. The examples are labelled data by definition,
    so a configuration that contradicts them is rejected regardless of eval accuracy."""
    sb = index.score(index._emb["pos"], index._emb["head"] if index.head_weight > 0 else None)
    owners = np.concatenate([np.full(sl.stop - sl.start, i) for i, sl in enumerate(index._pos_slices)])
    ordered = np.sort(sb.final, axis=1)
    ok = (sb.final.argmax(axis=1) == owners) & ((ordered[:, -1] - ordered[:, -2]) >= margin)
    return float(ok.mean())


def objective(report: dict) -> tuple:
    # Macro-F1 covers single-intent routing and rejection; overall accuracy also covers
    # multi-intent, follow-up and malformed cases. Weight them equally.
    blended = 0.5 * report["macro_f1_with_reject_class"] + 0.5 * report["overall_accuracy"]
    return (round(blended, 6), report["overall_accuracy"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--model", help="Embedding model (default: ROUTER_MODEL_NAME)")
    parser.add_argument("--write", action="store_true", help="Save the best values to data/thresholds.json")
    parser.add_argument("--quick", action="store_true", help="Smaller grids")
    parser.add_argument("--splits", default="dev,test,holdout",
                        help="Comma-separated eval splits to tune on (default: dev,test,holdout). "
                             "Never include the split you report (final).")
    args = parser.parse_args()

    setup_logging("ERROR")
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if "final" in splits:
        parser.error("'final' is the clean held-out split; do not tune on it")
    cases = [c for s in splits for c in load_cases(DATA_DIR / "eval" / f"{s}.jsonl")]
    service = IntentService(get_settings(), model_name=args.model)
    service.embedder.encode = CachedEncoder(service.embedder)
    index, engine = service.index, service.engine
    start = time.perf_counter()

    def run() -> dict:
        return evaluate(service, cases)

    baseline = run()
    print(f"model: {service.model_name}   tuning on: {'+'.join(splits)} ({len(cases)} cases)")
    print(f"baseline  macro-F1={baseline['macro_f1_with_reject_class']:.3f} "
          f"acc={baseline['overall_accuracy']:.3f}  scoring={service.scoring} thresholds={engine.t.model_dump()}")

    def show(stage: str, report: dict, extra: str) -> None:
        print(f"{stage}   macro-F1={report['macro_f1_with_reject_class']:.3f} "
              f"acc={report['overall_accuracy']:.3f}  {extra}")

    # Threshold grids (stage 2). Safety constraints, not tuned away: keep a low-confidence
    # band (accept > reject) so weak matches ask for confirmation, and a minimum top-2
    # margin so near-ties are reported as ambiguous. A small tuning set can otherwise
    # "prefer" always guessing.
    t0 = engine.t
    lo = round(t0.reject - 0.15, 2)
    rejects = [round(x, 2) for x in np.arange(max(-0.2, lo), t0.reject + 0.201, 0.05)]
    accept_offsets = [0.05, 0.1, 0.15] if args.quick else [0.05, 0.1, 0.15, 0.2]
    margins = [0.02, 0.03, 0.05] if args.quick else [0.02, 0.03, 0.05, 0.08]
    seg_offsets = [0.0, 0.05] if args.quick else [-0.05, 0.0, 0.05, 0.1]

    # ---- stage 1: scoring weights (only configurations consistent with the registry)
    weight_grid = {
        "action_weight": [0.0, 0.5, 1.0] if args.quick else [0.0, 0.5, 1.0, 1.5],
        "negative_penalty": [0.0, 1.0] if args.quick else [0.0, 0.5, 1.0],
        "head_weight": [0.0, 0.3] if args.quick else [0.0, 0.2, 0.3, 0.5],
        "top_k": [2] if args.quick else [1, 2, 3],
    }
    best_w, best_report = dict(service.scoring), baseline
    best_score = objective(baseline) if self_consistency(index, engine.t.margin) == 1.0 else (-1.0, -1.0)
    for combo in itertools.product(*weight_grid.values()):
        params = dict(zip(weight_grid, combo))
        for k, v in params.items():
            setattr(index, k, v)
        if self_consistency(index, min(margins)) < 1.0:
            continue
        report = run()
        if objective(report) > best_score:
            best_w, best_score, best_report = {**best_w, **params}, objective(report), report
    for k, v in best_w.items():
        setattr(index, k, v)
    show("stage 1", best_report, f"scoring={best_w}")

    if best_score[0] < 0:
        sys.exit("No weight configuration routes every registry example to its own intent; "
                 "check the registry for duplicate or contradictory examples.")
    print(f"registry self-consistency: {self_consistency(index, min(margins)):.3f}")

    # ---- stage 2: thresholds
    best_t, best_score = None, (-1.0, -1.0)  # only configurations inside the constrained grid qualify
    # Iterate from the most conservative values down; with a strict ">" comparison, ties
    # (common on a small dev set) resolve to the safest configuration: higher reject
    # threshold, wider confirmation band, larger ambiguity margin, stricter clause splits.
    order = [sorted(rejects, reverse=True), sorted(accept_offsets, reverse=True),
             sorted(margins, reverse=True), sorted(seg_offsets, reverse=True)]
    for reject, a_off, margin, s_off in itertools.product(*order):
        if self_consistency(index, margin) < 1.0:
            continue
        engine.t = Thresholds(reject=reject, accept=round(reject + a_off, 3), margin=margin,
                              segment_accept=round(reject + s_off, 3))
        report = run()
        if objective(report) > best_score:
            best_t, best_score, best_report = engine.t, objective(report), report
    if best_t is None:
        sys.exit("No threshold configuration satisfied the constraints.")
    engine.t = best_t
    show("stage 2", best_report, f"thresholds={best_t.model_dump()}")
    print(f"per-category (tuning data): { {k: v['accuracy'] for k, v in best_report['per_category'].items()} }")
    print(f"tuning took {time.perf_counter() - start:.0f}s. These are IN-SAMPLE numbers; "
          f"run scripts/evaluate.py --split final for held-out results.")

    if args.write:
        settings = get_settings()
        path = settings.thresholds_file
        data = json.loads(path.read_text(encoding="utf-8"))
        entry = {k: round(float(v), 3) for k, v in best_t.model_dump().items()}
        entry["scoring"] = {k: (round(float(v), 3) if isinstance(v, float) else v) for k, v in best_w.items()}
        data.setdefault("models", {})[service.model_name] = entry
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
