"""Command-line interface for routing requests locally.

Examples:
    python -m intent_router.cli "Find Python jobs in Noida"
    python -m intent_router.cli "Only remote ones" --context context.json
    python -m intent_router.cli --interactive          # carries context from turn to turn
    python -m intent_router.cli "book a cab" --debug --model BAAI/bge-small-en-v1.5
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from intent_router.core.config import get_settings
from intent_router.core.logging import setup_logging
from intent_router.core.schemas import ConversationContext


def _print(result, compact: bool) -> None:
    data = result.model_dump(mode="json")
    print(json.dumps(data, ensure_ascii=False, indent=None if compact else 2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m intent_router.cli", description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("message", nargs="?", help="Message to route (omit with --interactive)")
    parser.add_argument("--context", help="JSON file with a ConversationContext")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive session with context carry-over")
    parser.add_argument("--debug", action="store_true", help="Include scores and timings")
    parser.add_argument("--compact", action="store_true", help="One-line JSON output")
    parser.add_argument("--model", help="Embedding model name (overrides ROUTER_MODEL_NAME)")
    parser.add_argument("--reference-date", type=date.fromisoformat, help="'Today' for relative dates (YYYY-MM-DD)")
    args = parser.parse_args(argv)

    if not args.message and not args.interactive:
        parser.error("give a message or use --interactive")

    setup_logging("WARNING")
    from intent_router.services.intent_service import IntentService, MessageTooLongError

    service = IntentService(get_settings(), model_name=args.model)
    context = None
    if args.context:
        context = ConversationContext.model_validate_json(Path(args.context).read_text(encoding="utf-8"))

    def route_once(text: str, ctx):
        try:
            return service.route(text, ctx, include_debug=args.debug, reference_date=args.reference_date)
        except MessageTooLongError as exc:
            print(json.dumps({"error": {"code": "message_too_long", "message": str(exc)}}))
            return None

    if not args.interactive:
        result = route_once(args.message, context)
        if result is None:
            return 2
        _print(result, args.compact)
        return 0

    print("Interactive mode. Context from each successful turn is passed to the next. "
          "Commands: :reset, :context, :quit", file=sys.stderr)
    while True:
        try:
            text = input("> ")
        except (EOFError, KeyboardInterrupt):
            print(file=sys.stderr)
            return 0
        if text.strip() == ":quit":
            return 0
        if text.strip() == ":reset":
            context = None
            print("(context cleared)", file=sys.stderr)
            continue
        if text.strip() == ":context":
            print(context.model_dump_json(indent=2) if context else "(no context)")
            continue
        result = route_once(text, context)
        if result is None:
            continue
        _print(result, args.compact)
        if result.context is not None:
            context = result.context


if __name__ == "__main__":
    raise SystemExit(main())
