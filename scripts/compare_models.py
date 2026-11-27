"""Compare MiniLM and BGE on the same evaluation examples.

Each model runs in its own process (so memory numbers are not mixed up):
  1. optional: tune weights/thresholds for that model on the DEV split (--tune)
  2. evaluate on DEV and on the held-out TEST split
  3. benchmark latency and memory

Thresholds are model-specific: comparing models with one model's thresholds is
not a fair comparison, so use --tune unless both already have tuned entries in
data/thresholds.json.

Usage:
    python scripts/compare_models.py --tune
    python scripts/compare_models.py --models sentence-transformers/all-MiniLM-L6-v2 BAAI/bge-small-en-v1.5
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from intent_router.core.config import ALTERNATIVE_MODEL, DEFAULT_MODEL  # noqa: E402

# dev, test, holdout: the tuning pool (test and holdout were written blind but their errors
# were later inspected while improving the registry). final: written blind, never inspected
# case by case and never tuned on; the only clean estimate.
SPLITS = ("dev", "test", "holdout", "final")


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--models", nargs="+", default=[DEFAULT_MODEL, ALTERNATIVE_MODEL])
    parser.add_argument("--tune", action="store_true", help="Tune each model on dev first (writes thresholds.json)")
    parser.add_argument("--out", default="reports", help="Directory for JSON reports")
    args = parser.parse_args()

    out = ROOT / args.out
    out.mkdir(exist_ok=True)
    py = sys.executable
    rows = []
    for model in args.models:
        slug = re.sub(r"[^A-Za-z0-9]+", "_", model)
        if args.tune:
            run([py, "scripts/tune_thresholds.py", "--model", model, "--write"])
        reports = {}
        for split in SPLITS:
            path = out / f"{slug}_{split}.json"
            run([py, "scripts/evaluate.py", "--model", model, "--split", split, "--json", str(path)])
            reports[split] = json.loads(path.read_text(encoding="utf-8"))
        run([py, "scripts/benchmark.py", "--model", model, "--repeat", "2", "--json", str(out / f"{slug}_bench.json")])
        bench = json.loads((out / f"{slug}_bench.json").read_text(encoding="utf-8"))
        row = {"model": model}
        for split, rep in reports.items():
            row[f"{split}_macro_f1"] = rep["macro_f1_with_reject_class"]
            row[f"{split}_intent_acc"] = rep["intent_accuracy"]
            row[f"{split}_overall_acc"] = rep["overall_accuracy"]
            row[f"{split}_unknown_rejected"] = rep["unknown_handling"].get("rejected", 0.0)
            row[f"{split}_false_rejection"] = rep["false_rejection_rate"]
            row[f"{split}_extraction_acc"] = rep["extraction_field_accuracy"]
        row.update({
            "p50_ms": bench["latency_ms"]["wall_ms"]["p50"],
            "p95_ms": bench["latency_ms"]["wall_ms"]["p95"],
            "embed_p50_ms": bench["latency_ms"]["embed_ms"]["p50"],
            "rss_after_model_mb": bench["memory_rss_mb"]["after_model_load"],
            "model_load_ms": bench["startup_ms"]["model_load"],
        })
        rows.append(row)

    print()
    for key in rows[0]:
        print(f"{key:28s} " + "  ".join(f"{str(r[key]):>40s}" for r in rows))
    (out / "model_comparison.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwrote {out / 'model_comparison.json'}")


if __name__ == "__main__":
    main()
