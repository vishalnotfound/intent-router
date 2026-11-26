"""Measure startup, memory and per-request latency on THIS machine.

Reports separately:
- library import time, model load time, intent-embedding build (cold vs. disk cache)
- process memory (RSS) before/after the model and the index
- per-request latency p50/p95 for each stage: extraction, embedding, scoring,
  decision/parameter mapping, schema validation, JSON serialization, total
- throughput and latency with concurrent requests (threads)

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --model BAAI/bge-small-en-v1.5 --threads 4 --json reports/bench.json
"""

from __future__ import annotations

import time

_T_START = time.perf_counter()

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def rss_mb() -> float | None:
    try:
        import psutil

        return round(psutil.Process().memory_info().rss / 2**20, 1)
    except ImportError:
        return None


def pct(values, q):
    import numpy as np

    return round(float(np.percentile(values, q)), 2) if values else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--model", help="Embedding model (default: ROUTER_MODEL_NAME)")
    parser.add_argument("--threads", type=int, help="torch intra-op threads (default: router default)")
    parser.add_argument("--repeat", type=int, default=3, help="Passes over the request set")
    parser.add_argument("--concurrency", type=int, default=4, help="Threads for the concurrent test")
    parser.add_argument("--json", help="Write results to this JSON file")
    args = parser.parse_args()

    if args.threads:
        os.environ["ROUTER_TORCH_THREADS"] = str(args.threads)
    rss_start = rss_mb()

    t = time.perf_counter()
    import torch  # noqa: F401
    import sentence_transformers  # noqa: F401
    import_ms = (time.perf_counter() - t) * 1000

    from datetime import date

    from app.core.config import DATA_DIR, Settings
    from app.core.logging import setup_logging
    from app.evaluation import load_cases
    from app.models.loader import load_sentence_model
    from app.services.intent_service import IntentService

    setup_logging("ERROR")
    rss_after_import = rss_mb()

    # Cold start: empty embedding cache directory.
    with tempfile.TemporaryDirectory() as tmp:
        settings = Settings(embedding_cache_dir=Path(tmp))
        model_name = args.model or settings.model_name
        t = time.perf_counter()
        load_sentence_model(settings, model_name)
        model_load_ms = (time.perf_counter() - t) * 1000
        rss_after_model = rss_mb()
        cold = IntentService(settings, model_name=model_name)
        warm = IntentService(settings, model_name=model_name)  # second build reads the disk cache
    rss_after_index = rss_mb()
    service = warm
    import torch as _torch

    texts = [c["text"] for split in ("dev", "test") for c in load_cases(DATA_DIR / "eval" / f"{split}.jsonl")]
    texts = [t for t in texts if t.strip()]
    ref = date(2026, 10, 22)
    for t in texts[:10]:  # warm-up
        service.route(t, reference_date=ref)

    stages: dict[str, list[float]] = {k: [] for k in
                                      ("extract_ms", "embed_ms", "score_ms", "resolve_ms", "validate_ms", "total_ms")}
    stages["serialize_ms"] = []
    stages["wall_ms"] = []
    for _ in range(args.repeat):
        for t in texts:
            t0 = time.perf_counter()
            r = service.route(t, include_debug=True, reference_date=ref)
            t1 = time.perf_counter()
            r.model_dump_json()
            t2 = time.perf_counter()
            for k in stages:
                if k in r.debug.timings_ms:
                    stages[k].append(r.debug.timings_ms[k])
            stages["serialize_ms"].append((t2 - t1) * 1000)
            stages["wall_ms"].append((t1 - t0) * 1000)

    # Single-text embedding only, for reference.
    embed_only = []
    for t in texts[:60]:
        t0 = time.perf_counter()
        service.embedder.encode([t])
        embed_only.append((time.perf_counter() - t0) * 1000)

    # Concurrent requests.
    conc_lat: list[float] = []

    def one(text):
        t0 = time.perf_counter()
        service.route(text, reference_date=ref)
        conc_lat.append((time.perf_counter() - t0) * 1000)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        list(pool.map(one, texts * 2))
    conc_elapsed = time.perf_counter() - t0

    results = {
        "machine": {"platform": platform.platform(), "processor": platform.processor(),
                    "cpu_count": os.cpu_count(), "python": platform.python_version()},
        "model": service.model_name,
        "torch_threads": _torch.get_num_threads(),
        "startup_ms": {
            "python_to_main": round((time.perf_counter() - _T_START) * 1000, 1),
            "import_torch_and_sentence_transformers": round(import_ms, 1),
            "model_load": round(model_load_ms, 1),
            "intent_embeddings_cold": round(cold.startup_ms["intent_embeddings"], 1),
            "intent_embeddings_from_cache": round(warm.startup_ms["intent_embeddings"], 1),
            "registry_and_extractors": round(warm.startup_ms["registry"] + warm.startup_ms["extractors"], 1),
            "registry_texts_embedded": service.index.num_texts,
        },
        "memory_rss_mb": {"process_start": rss_start, "after_imports": rss_after_import,
                          "after_model_load": rss_after_model, "after_two_indexes": rss_after_index},
        "latency_ms": {k: {"p50": pct(v, 50), "p95": pct(v, 95), "mean": round(sum(v) / len(v), 3)}
                       for k, v in stages.items() if v},
        "embedding_only_single_text_ms": {"p50": pct(embed_only, 50), "p95": pct(embed_only, 95)},
        "requests_measured": len(stages["wall_ms"]),
        "concurrent": {"threads": args.concurrency, "requests": len(conc_lat),
                       "throughput_rps": round(len(conc_lat) / conc_elapsed, 1),
                       "p50_ms": pct(conc_lat, 50), "p95_ms": pct(conc_lat, 95)},
    }

    print(f"model: {results['model']}   torch threads: {results['torch_threads']}   "
          f"machine: {results['machine']['processor'] or results['machine']['platform']}")
    print("startup (ms):")
    for k, v in results["startup_ms"].items():
        print(f"  {k:42s} {v}")
    print("memory RSS (MB):", results["memory_rss_mb"])
    print(f"latency per request over {results['requests_measured']} requests (ms):")
    for k, v in results["latency_ms"].items():
        print(f"  {k:14s} p50={v['p50']:>7}  p95={v['p95']:>7}  mean={v['mean']:>7}")
    print("embedding only, single text (ms):", results["embedding_only_single_text_ms"])
    print("concurrent:", results["concurrent"])
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
