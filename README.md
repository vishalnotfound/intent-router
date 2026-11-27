# Intent Router

A lightweight, local, CPU-only natural-language intent router in Python. It sits **in front of stronger (and costlier) LLMs** to classify intent, extract structured parameters, split multi-intent messages, and reject unknown or malformed input—saving token costs and avoiding unnecessary calls to large models.

## Why Use This (Cost & Token Optimization)

Sending every user message straight to a frontier or large LLM burns input/output tokens on routing, tool schemas, clarification loops, and out-of-scope chatter. Intent Router runs locally on CPU (~90 MB `all-MiniLM-L6-v2`) with **zero per-request token cost**:

- **Avoids hitting stronger models on clear or deterministic routes:** Resolved requests return structured `intent`, `action`, `target`, `domain`, `entities`, and `filters` so your app can dispatch directly to code/APIs—or pass only minimal structured context downstream instead of raw conversation history and full tool catalogs.
- **Blocks wasted LLM calls early:** Empty/non-linguistic input (`invalid_input`), gibberish or out-of-scope requests (`unknown`), missing required parameters (`needs_clarification`), and near-ties (`ambiguous`) are handled locally without spending a single token.
- **Zero cloud dependency at runtime:** After a one-time model download, inference runs completely offline (`local_files_only=True`). Even the optional local LLM fallback (`ROUTER_FALLBACK=local_llm`) is off by default and only runs on inconclusive cases.
- **Routes only—never executes:** Returns validated JSON (`RouteResult`, schema `1.0`) and flags `side_effects` / `requires_confirmation` so downstream handlers enforce authorization.

## How It Works

1. **Normalize & separate payload:** NFKC normalization, control-char stripping, and separation of pasted text/code blocks (`"Summarize this: ..."`) so payload text doesn't skew routing.
2. **Extract spans:** Deterministic regex parsers (money, price limits, dates, times, durations, quantities, IDs, emails, URLs, errors) + gazetteers (`intent_router/data/gazetteers.json` & domain files) + verbatim phrases (`use_case`, `requirements`, `topic`).
3. **Delexicalize & fix typos:** Masks entities (cities, technologies, amounts, quotes) with generic placeholders and applies conservative typo correction before embedding.
4. **Embed & score:** Single batched forward pass (`text` + leading `head` words) scored against intent examples, optional negative examples, and topic-neutral action phrases.
5. **Decide & validate:** Applies per-model thresholds (`intent_router/data/thresholds.json`), maps spans to declared parameters, handles follow-up `ConversationContext` and multi-intent splitting, and validates output with Pydantic v2.

## Quickstart (Windows)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt      # or requirements.txt for runtime only
python scripts/download_models.py        # one-time ~90 MB download (all-MiniLM-L6-v2)
```

### Run the API & Web UI

```powershell
.\run.bat
# or: uvicorn intent_router.main:app --host 127.0.0.1 --port 8000
```

- Web UI: `http://127.0.0.1:8000/`
- Docs: `http://127.0.0.1:8000/docs`
- Health: `GET /health`

```powershell
curl.exe -s -X POST http://127.0.0.1:8000/route `
  -H "Content-Type: application/json" `
  -d "{\"message\": \"Find Python jobs in Noida and compare their salaries\"}"
```

### Run the CLI

```powershell
python -m intent_router.cli "Find Python jobs in Noida"
python -m intent_router.cli --interactive        # carries ConversationContext across turns
```

## Response & Context

Every `POST /route` returns a strict `RouteResult` (`extra="forbid"`):

- **Core routing:** `status` (`success`, `needs_clarification`, `ambiguous`, `unknown`, `multi_intent`, `invalid_input`), `intent`, `action`, `target`, `domain`
- **Extracted slots:** `entities`, `filters`, `missing_fields`, `optional_missing_fields`
- **Multi-intent & follow-ups:** `tasks` (ordered with `depends_on`), `context` (stateless `ConversationContext` to echo back on the next turn), `is_follow_up`, `context_used`
- **Safety & metadata:** `side_effects`, `requires_confirmation`, `confidence` (heuristic in `[0, 1]`, **not** a calibrated probability or auth signal), `alternatives`, `resolved_by` (`embedding` / `context` / `fallback` / `none`)

## Adding Intents & Configuration

- **Intents & domains:** Edit [intent_router/data/intents.json](intent_router/data/intents.json) or drop new `*.json` files into [intent_router/data/domains](intent_router/data/domains). No engine code changes required; the embedding cache (`.cache/embeddings/`) invalidates automatically when texts change.
- **Exclusion filters:** a parameter with `"excludes": "<other parameter>"` (e.g. an `exclude_locations` filter with `"excludes": "location"`) collects excluded values such as "anywhere except Delhi". Without one, the router returns `needs_clarification` explaining that the value can't be excluded, and never returns it as an included value.
- **Unsupported directions:** `"unsupported_variants"` on an intent lists regex-recognised requests it must not serve (see `candidate_search` on `jobs.search`); a match returns `unknown`. Patterns can only reject, never route.
- **Settings:** Copy [.env.example](.env.example) to `.env`. Key `ROUTER_*` options include `ROUTER_MODEL_NAME` (`all-MiniLM-L6-v2` or `BAAI/bge-small-en-v1.5`), `ROUTER_MAX_MESSAGE_CHARS` (`2000`), `ROUTER_ENABLE_MULTI_INTENT` (`true`), `ROUTER_DATE_ORDER` (`DMY`), and `ROUTER_FALLBACK` (`none` | `local_llm`).
- **Threshold tuning:** `python scripts/tune_thresholds.py` (tunes on `dev+test+holdout`, never `final`).

## Evaluation & Tests

Measured with `sentence-transformers/all-MiniLM-L6-v2` on the `final` split (`71` unseen cases) and `all` (`437` cases):

| Split | Overall Acc | Intent Acc | Macro-F1 (intents + reject) | Extraction Field Acc |
|---|---|---|---|---|
| `final` (71) | `0.930` | `0.922` | `0.877` | `1.000` (52 fields) |
| `all` (437) | `0.924` | `0.930` | `0.928` | `1.000` (339 fields) |

```powershell
python -m pytest                     # full suite
python scripts/evaluate.py --split final --errors
python scripts/evaluate.py --split regression --errors   # negation, follow-up, repeated-intent, scope checks
python scripts/benchmark.py          # measure latency & RSS on your machine
```

## Notes & Limits

- **No execution:** The router only classifies and extracts; downstream callers must authorize destructive/financial intents (`requires_confirmation: true`).
- **Negation:** A negated request ("Do not cancel order 4421", or "actually don't" after a cancellation) returns `needs_clarification` with no intent, so it is never an executable route. Negation and exclusion are detected with explicit cue lists, so unusual phrasings may still be missed.
- **Optional NLI verifier (`ROUTER_VERIFIER=nli`, off by default):** a local cross-encoder (`cross-encoder/nli-deberta-v3-xsmall`, pinned revision, ~295 MB; `python scripts/download_models.py --verifier`) checks destructive/financial requests against each intent's `verify_hypotheses` and blocks ones the message contradicts, e.g. "I no longer want to cancel my order", which the rules miss. It only ever blocks (→ `needs_clarification`), fails closed, and skips short follow-ups, where it cannot tell "yes do it" from "keep it". Measured on a 4-thread Intel CPU: cancel/return/booking requests p50 16 → 113 ms, other requests unchanged, +~210 MB RSS, +1.2 s startup, concurrent throughput 56 → 43 req/s. On `final` it adds one false block (an exchange request: 0.930 → 0.915); `tuning` and `regression` are unchanged.
- **Follow-ups:** With a `context`, a complete request of its own ("Find Java jobs") starts fresh; refinements ("Only remote ones", "Find Java jobs there") keep the previous values.
- **Strict metric:** `strict_accuracy` additionally checks an optional `expected.status`, `expected.absent` fields and per-task `expected.task_payloads`. The `regression` split uses them; it is not part of `all` or the tuning splits.
- **Not a general NLU model:** Works best on requests matching registered intent phrasing; unsupported requests sharing keywords with supported intents may still score above `reject`.
- **Input safety:** Requests are capped at `ROUTER_MAX_MESSAGE_CHARS` (default `2000`, HTTP `413`) and `256 KB` body size (HTTP `413`), and raw messages are not logged unless `ROUTER_LOG_MESSAGES=true`.
