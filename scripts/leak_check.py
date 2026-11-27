"""Check the router for memory growth over many requests.

Phases (each runs ROUNDS x requests and records process RSS after every round):
  steady    the evaluation messages, repeated (normal traffic: memory should plateau)
  unique    every message contains new random words (worst case for any per-word cache)
  context   follow-ups that pass the previous response's context back in (context chains)
  api       the HTTP endpoint through FastAPI's TestClient, from 4 threads

tracemalloc compares Python allocations between the end of warm-up and the end of each
phase and prints the source lines whose memory grew the most. Native memory (PyTorch,
tokenizers) is not visible to tracemalloc; RSS covers it.

Usage:
    python scripts/leak_check.py                 # default sizes (~5 minutes)
    python scripts/leak_check.py --rounds 3 --per-round 300   # quicker
"""

from __future__ import annotations

import argparse
import gc
import random
import string
import sys
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import psutil  # noqa: E402

from intent_router.core.config import DATA_DIR, get_settings  # noqa: E402
from intent_router.core.logging import setup_logging  # noqa: E402
from intent_router.evaluation import load_cases  # noqa: E402

REF = date(2026, 10, 22)
PROC = psutil.Process()


def rss_mb() -> float:
    gc.collect()
    return PROC.memory_info().rss / 2**20


def random_message(rng: random.Random) -> str:
    words = ["".join(rng.choices(string.ascii_lowercase, k=rng.randint(5, 9))) for _ in range(rng.randint(2, 5))]
    templates = ["find {} jobs in {}", "summarize the {} report", "translate {} to French", "{} {}",
                 "compare {} and {}", "my {} keeps crashing"]
    t = rng.choice(templates)
    return t.format(*(words + words)[: t.count("{}")]) + " " + " ".join(words[2:])


def top_growth(before, after, limit: int = 8) -> list[str]:
    stats = after.compare_to(before, "lineno")
    lines = []
    for s in stats[:limit]:
        if s.size_diff <= 0:
            continue
        frame = s.traceback[0]
        lines.append(f"    {s.size_diff / 1024:+9.1f} KiB  {s.count_diff:+7d} blocks  {frame.filename}:{frame.lineno}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--per-round", type=int, default=600)
    parser.add_argument("--phases", default="steady,unique,context,api")
    args = parser.parse_args()

    setup_logging("ERROR")
    from intent_router.services.intent_service import IntentService

    service = IntentService(get_settings())
    texts = [c["text"] for s in ("dev", "test", "holdout", "final")
             for c in load_cases(DATA_DIR / "eval" / f"{s}.jsonl")]
    rng = random.Random(7)

    # Warm-up: first-call allocations (lazy imports, regex caches, torch kernels) are not leaks.
    for t in texts:
        service.route(t, reference_date=REF, include_debug=True)
    tracemalloc.start(10)
    baseline_snapshot = tracemalloc.take_snapshot()
    baseline_rss = rss_mb()
    print(f"after warm-up: RSS {baseline_rss:.1f} MB")

    def steady(i):
        service.route(texts[i % len(texts)], reference_date=REF, include_debug=i % 2 == 0)

    def unique(i):
        service.route(random_message(rng), reference_date=REF)

    chain = {"ctx": None}
    followups = ["Find Python jobs in Noida", "Only remote ones", "What about Delhi?", "also Java",
                 "with 3+ years experience", "under 20 LPA"]

    def context(i):
        text = followups[i % len(followups)]
        ctx = None if i % len(followups) == 0 else chain["ctx"]
        r = service.route(text, ctx, reference_date=REF)
        chain["ctx"] = r.context

    client = None
    if "api" in args.phases:
        from fastapi.testclient import TestClient

        from intent_router.main import app

        # Reuse the same router instead of running the app's lifespan, which would build a
        # second one; this phase measures the HTTP layer only.
        app.state.service = service
        client = TestClient(app)
        pool = ThreadPoolExecutor(max_workers=4)

    def api_round(n):
        def call(i):
            r = client.post("/route", json={"message": texts[i % len(texts)]})
            assert r.status_code == 200
        list(pool.map(call, range(n)))

    phases = {"steady": steady, "unique": unique, "context": context}
    results = {}
    for phase in [p.strip() for p in args.phases.split(",") if p.strip()]:
        series = []
        start_snapshot = tracemalloc.take_snapshot()
        for _ in range(args.rounds):
            if phase == "api":
                api_round(args.per_round)
            else:
                for i in range(args.per_round):
                    phases[phase](i)
            series.append(round(rss_mb(), 1))
        end_snapshot = tracemalloc.take_snapshot()
        traced_now, traced_peak = tracemalloc.get_traced_memory()
        first, last = series[0], series[-1]
        # Growth after the first round is what matters; round 1 may still settle allocator pools.
        per_1k = (last - first) / max(1, (args.rounds - 1) * args.per_round) * 1000
        results[phase] = (series, per_1k)
        print(f"\n[{phase}] {args.rounds} x {args.per_round} requests")
        print(f"  RSS after each round (MB): {series}")
        print(f"  growth after round 1: {last - first:+.1f} MB  ({per_1k:+.2f} MB per 1000 requests)")
        print(f"  Python heap traced: {traced_now / 2**20:.1f} MB (peak {traced_peak / 2**20:.1f} MB)")
        growth = top_growth(start_snapshot, end_snapshot)
        print("  largest Python allocation growth during this phase:")
        print("\n".join(growth) if growth else "    (none)")

    if client is not None:
        pool.shutdown()
        client.close()

    print("\noverall Python allocation growth since warm-up:")
    print("\n".join(top_growth(baseline_snapshot, tracemalloc.take_snapshot(), 10)) or "    (none)")
    print(f"\nfinal RSS {rss_mb():.1f} MB (after warm-up {baseline_rss:.1f} MB)")
    corrector = service.engine.corrector
    if corrector is not None:
        print(f"typo-correction index entries (static, built at startup): {len(corrector._index)}")
    print(f"per-request caches in the router: {[n for n in vars(service.engine) if 'cache' in n.lower()] or 'none'}")


if __name__ == "__main__":
    main()
