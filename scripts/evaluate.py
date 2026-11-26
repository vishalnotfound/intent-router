"""Evaluate the router on the hand-written dataset.

Usage:
    python scripts/evaluate.py                    # dev split, default model
    python scripts/evaluate.py --split test       # held-out split: use for final numbers only
    python scripts/evaluate.py --model BAAI/bge-small-en-v1.5 --errors
    python scripts/evaluate.py --json reports/eval_dev.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import DATA_DIR, get_settings  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402
from app.evaluation import evaluate, format_report, load_cases  # noqa: E402
from app.services.intent_service import IntentService  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--split", choices=["dev", "test", "holdout", "final", "tuning", "all"], default="dev",
                        help="tuning = dev+test+holdout (what thresholds are tuned on); final = clean held-out set")
    parser.add_argument("--model", help="Embedding model (default: ROUTER_MODEL_NAME)")
    parser.add_argument("--errors", action="store_true", help="Print misrouted cases and extraction mismatches")
    parser.add_argument("--json", help="Write the full report to this JSON file")
    args = parser.parse_args()

    setup_logging("WARNING")
    splits = {"all": ["dev", "test", "holdout", "final"], "tuning": ["dev", "test", "holdout"]}.get(
        args.split, [args.split])
    cases = [c for s in splits for c in load_cases(DATA_DIR / "eval" / f"{s}.jsonl")]
    service = IntentService(get_settings(), model_name=args.model)
    report = evaluate(service, cases)
    report["model"] = service.model_name
    report["split"] = args.split
    report["thresholds"] = service.thresholds.model_dump()
    print(format_report(report, f"{service.model_name} on {args.split}"))
    if args.errors:
        print("\nmisrouted:")
        for e in report["errors"]:
            print(f"  [{e['id']}] {e['text']!r}: gold={e['gold']} pred={e['pred']} "
                  f"{('tasks=' + str(e['tasks'])) if e.get('tasks') else ''}")
        print("\nextraction mismatches:")
        for e in report["field_errors"]:
            print(f"  [{e['id']}] {e['field']}: gold={e['gold']} pred={e['pred']}")
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
