# Intent Router

A lightweight, local, general-purpose natural-language intent router in Python.
It reads a user message and returns validated JSON describing **what the user wants**
(action, target, domain, intent), **with which parameters** (entities, filters), and
**how sure it is**. When it is not sure, it says so (`unknown`, `ambiguous`,
`needs_clarification`) instead of guessing.

- Runs on CPU with a small pretrained sentence-embedding model (`all-MiniLM-L6-v2`, ~90 MB).
- No training or fine-tuning. No API keys, no cloud calls, no network after the one-time download.
- Intents live in JSON files; new domains plug in without code changes.
- It only **routes**. It never executes anything; the caller decides what to do.

```text
POST /route  {"message": "Find Python jobs in Noida and compare their salaries"}

{
  "version": "1.0",
  "status": "multi_intent",
  "tasks": [
    {"index": 0, "text": "Find Python jobs in Noida", "status": "success",
     "intent": "jobs.search", "action": "search", "target": "job_openings", "domain": "jobs",
     "entities": {"skills": ["Python"], "location": "Noida"}, "depends_on": [], ...},
    {"index": 1, "text": "compare their salaries", "status": "success",
     "intent": "general.compare", "action": "compare", "target": "items", "domain": "general",
     "entities": {"criteria": ["salaries"]}, "depends_on": [0], ...}
  ],
  "reasoning_summary": "Found 2 separate requests: jobs.search, general.compare.",
  ...
}
```

---

## Contents

1. [How it works](#how-it-works)
2. [Windows setup](#windows-setup)
3. [Running it](#running-it)
4. [Response format](#response-format)
5. [Conversation context (follow-ups)](#conversation-context-follow-ups)
6. [Adding intents and domains](#adding-intents-and-domains)
7. [Configuration](#configuration)
8. [Thresholds and tuning](#thresholds-and-tuning)
9. [Evaluation results](#evaluation-results)
10. [Performance](#performance)
11. [Optional local LLM fallback](#optional-local-llm-fallback)
12. [Security and reliability](#security-and-reliability)
13. [Limitations](#limitations)
14. [Tests](#tests)
15. [Project layout](#project-layout)

---

## How it works

Routing and extraction are separate stages. The embedding model is only used to
compare meanings; it is never asked to produce JSON fields.

```text
message
  │
  ├─ 1. normalize        Unicode NFKC, control chars removed, quotes/whitespace unified.
  │                      Pasted content ("Summarize this: <article>") and code blocks are
  │                      separated from the instruction so they don't dominate the embedding.
  │
  ├─ 2. extract spans    Deterministic parsers: money & price limits, dates, times, durations,
  │                      quantities, ids, emails/urls, errors, plus gazetteers (cities, skills,
  │                      languages, products...) and verbatim phrases (use case, requirements).
  │
  ├─ 3. delexicalize     Places/technologies/names/prices/quoted text -> generic words:
  │                      "Find Python jobs in Noida" -> "Find programming jobs in the city".
  │                      + conservative typo correction ("jbos" -> "jobs") for routing only.
  │
  ├─ 4. embed            One batched forward pass: whole text + "head" view (first words)
  │                      for the message, its clauses, and (if given) context.
  │
  ├─ 5. score            per intent: similarity to its examples (top-k mean, blended with the
  │                      head view) − negative-example penalty − action-stage penalty
  │
  ├─ 6. decide           reject / low-confidence / ambiguous / accept, plus guards for lone
  │                      nouns, gibberish, out-of-scope requests, follow-ups, multi-intent
  │
  ├─ 7. map parameters   spans -> the parameters the chosen intent declares;
  │                      required-but-missing and recommended-but-missing are reported
  │
  └─ 8. validate         Pydantic v2 RouteResult (schema version 1.0)
```

### Why the extra machinery around the embeddings

Small sentence encoders measure *topic* similarity far more than *intent*. Measured with
MiniLM while building this:

| Pair | Cosine |
|---|---|
| "Find Python jobs in Noida" vs "**Compare** Python jobs in Noida" | 0.85 |
| "Find Python jobs in Noida" vs "Find **Java** jobs in **Pune**" | 0.31 |
| "Write a function to **merge two dictionaries**" vs "Write a function to **reverse a string**" | 0.33 |

So plain nearest-example matching gets the action wrong and is thrown off by names.
Each component below addresses one measured failure:

- **Delexicalization** (step 3) removes proper nouns and values from both the message and the
  registry examples, so "Java/Pune" and "Python/Noida" no longer decide the result.
- **Head view**: the first few words of a request usually hold the verb and object type
  ("Write a programming function"); blending a head-only similarity raised raw top-1
  accuracy on the dev set's in-scope cases from 0.91 to 0.97.
- **Action stage** (hierarchical): every action (search, compare, summarize, …) has
  *topic-neutral* example phrases ("compare these", "what's the difference between them").
  The topic shifts all action scores roughly equally, so the verb decides. Intents whose
  action trails the best action are penalized.
- **Negative examples**: an intent can be penalized when the message is closer to its
  "confusing" examples than to its real ones. Tuning found no measurable benefit, so the shipped
  weight is 0 for both models; the mechanism stays configurable (`ROUTER_NEGATIVE_PENALTY`).

### Decision policy

| Condition | Result |
|---|---|
| empty or no letters/digits | `invalid_input` |
| fewer than half the words are recognizable ("zxcv bnm lkjh") | `unknown` |
| best score < `reject` | `unknown` (closest candidates still listed in `alternatives`) |
| best match is a known *unsupported* kind of request (device control, payments, …) | `unknown` |
| runner-up within `margin` of the best | `ambiguous` (`action` is filled if all close candidates share it) |
| message is a lone noun ("Python", "Delhi") | `ambiguous` + "What would you like me to do with …?" |
| `reject` ≤ best < `accept` | `needs_clarification` ("Just to check: do you want me to …?") |
| accepted but a required parameter is missing | `needs_clarification` + `missing_fields` |
| vague request ("can you help me with something?") | `needs_clarification` |
| several clauses that each route confidently | `multi_intent` with ordered `tasks` |
| otherwise | `success` |

`confidence` is a monotonic heuristic in [0, 1] built from how far the best score is above
`reject` and how far it is ahead of the runner-up. **It is not a calibrated probability**
("0.8" does not mean "right 80% of the time") and it is **never** an authorization signal.

---

## Windows setup

Tested on Windows 11 with Python 3.12. Python 3.10+ should work.

```powershell
cd C:\codes\ir
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# CPU-only PyTorch (smaller than the default wheel; no GPU needed)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Router + test/benchmark tools
pip install -r requirements-dev.txt      # or: pip install -r requirements.txt  (runtime only)
```

If PowerShell refuses to run `Activate.ps1`, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use `.\.venv\Scripts\python.exe` directly.

### First-time model download (the only step that needs internet)

```powershell
python scripts/download_models.py          # all-MiniLM-L6-v2 (~90 MB)
python scripts/download_models.py --all    # also BAAI/bge-small-en-v1.5 (~130 MB), for comparison
```

Models go to the normal Hugging Face cache (`%USERPROFILE%\.cache\huggingface\hub`), or to
`ROUTER_MODEL_CACHE_DIR` if set.

### Offline use after the download

Nothing to do: by default the router loads models with `local_files_only=True` and sets
`HF_HUB_OFFLINE=1`, so it never contacts the network at startup or while routing. If the model
is missing it fails fast with a message telling you to run the download script.
(Set `ROUTER_ALLOW_MODEL_DOWNLOAD=true` only if you want automatic downloads.)

On Windows without Developer Mode, Hugging Face prints a warning about symlinks; it is harmless
(the cache just uses a little more disk).

---

## Running it

### HTTP API

```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

On Windows, run `run.bat` from the project folder instead. It uses `.venv` if available,
otherwise the Python installation on `PATH`.

The model loads and intent embeddings are computed (or read from `.cache/embeddings`) once at
startup, before the first request is served. Interactive docs: http://127.0.0.1:8000/docs

```powershell
# PowerShell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/route -ContentType 'application/json' `
  -Body '{"message": "Find a cheap laptop for programming that I can upgrade later"}' | ConvertTo-Json -Depth 10

# curl (Windows 10+ ships curl.exe)
curl.exe -s -X POST http://127.0.0.1:8000/route -H "Content-Type: application/json" -d "{\"message\": \"Book a train from Delhi to Agra on Saturday\", \"options\": {\"include_debug\": true}}"

curl.exe -s http://127.0.0.1:8000/health
```

Request body:

```json
{
  "message": "Only remote ones",
  "context": { "...": "the `context` object from the previous response (optional)" },
  "options": { "include_debug": false, "reference_date": "2026-10-22" }
}
```

`reference_date` is "today" for relative dates ("tomorrow", "next Friday"); default is the server date.

| HTTP status | When | Body |
|---|---|---|
| 200 | any routing outcome, including `unknown` / `invalid_input` | `RouteResult` |
| 413 | message longer than `ROUTER_MAX_MESSAGE_CHARS` (2000), or body > 256 KB | `{"error": {"code": ...}}` |
| 422 | malformed JSON, wrong types, unknown fields | `{"error": {"code": "validation_error", "details": [...]}}` (input is not echoed) |
| 500 | unexpected error | `{"error": {"code": "internal_error"}}` (no internals leaked) |

### Command line

```powershell
python -m app.cli "Find Python jobs in Noida"
python -m app.cli "Remind me to call mom at 6 pm tomorrow" --debug
python -m app.cli "Only remote ones" --context context.json
python -m app.cli --interactive            # carries context from turn to turn; :reset, :context, :quit
python -m app.cli "book a cab" --model BAAI/bge-small-en-v1.5
```

---

## Response format

Every response is a `RouteResult` (Pydantic v2, `extra="forbid"`, schema version `"1.0"`).
Missing values are always `null` (scalars) or empty (`{}` / `[]`); keys are never omitted.

| Field | Meaning |
|---|---|
| `version` | `"1.0"` |
| `status` | `success`, `needs_clarification`, `ambiguous`, `unknown`, `multi_intent`, `invalid_input` |
| `intent` | canonical id, e.g. `jobs.search`; `"unknown"` for unknown; `null` when ambiguous / multi / no intent |
| `action` | broad action: `search`, `retrieve`, `compare`, `summarize`, `extract`, `create`, `modify`, `explain`, `troubleshoot`, `plan`, `classify`, `translate`, `calculate`, `converse`, `other`, or `unknown` |
| `target` | resource type, e.g. `job_openings`, `products`, `document` |
| `domain` | `jobs`, `ecommerce`, `documents`, `programming`, `planning`, `travel`, `general`, … |
| `entities` | parameters the intent declares (`kind: entity`); values the intent has no slot for go to `entities.other` instead of being dropped |
| `filters` | constraints (`kind: filter`): salary/budget ranges, work mode, sort order, rating, … |
| `missing_fields` | required parameters that are absent (`"date\|time"` = at least one of them) |
| `optional_missing_fields` | recommended parameters that are absent (e.g. `budget`) — never invented |
| `needs_clarification`, `clarification_question` | whether to ask the user, and a suggested question |
| `confidence` | heuristic in [0, 1]; see above |
| `alternatives` | runner-up intents with their raw similarity `score` (not a probability) |
| `tasks` | for `multi_intent`: ordered tasks, each with its own fields and `depends_on` |
| `is_follow_up`, `context_used`, `resolved_by` | how context and the fallback were used (`embedding`, `context`, `fallback`, `none`) |
| `side_effects`, `requires_confirmation` | from the intent definition (`none`/`read`/`write`/`destructive`/`financial`); destructive and financial intents require confirmation by default |
| `reasoning_summary` | one short sentence about the classification (no chain-of-thought) |
| `context` | the updated context to send with the next message |
| `debug` | only with `include_debug`: routed text, top scores, action scores, thresholds, timings |

The laptop example from the brief:

```text
"Find a cheap laptop for programming that I can upgrade later"

intent   ecommerce.product_search   action search   target products   domain ecommerce
entities {"product": "laptop", "use_case": "programming", "requirements": ["I can upgrade later"]}
filters  {"price_preference": "low"}
optional_missing_fields ["budget"]          <- not invented
```

"Requirements" keep the user's wording verbatim when the exact meaning is uncertain.
Values are never inferred beyond their surface form: "under 50k" gives `currency: null`,
not INR; "12 LPA" gives INR per year because LPA means that.

---

## Conversation context (follow-ups)

The router holds **no hidden state**. Each response includes a `context` object; send it back
with the next message to make follow-ups work:

```text
1. {"message": "Find Python jobs in Noida"}
   -> jobs.search, entities {skills: [Python], location: Noida}, context {...}

2. {"message": "Only remote ones.", "context": <context from 1>}
   -> jobs.search, filters {work_mode: remote}, entities unchanged, context_used: true

3. {"message": "What about Delhi?", "context": <context from 2>}
   -> jobs.search, location: Delhi (replaced), work_mode still remote
```

- A message with a follow-up phrase ("only …", "what about …", "… ones", "same but …") and no
  action verb of its own refines the previous request.
- A message with its own action ("Write a Python function …") is a new request; the context is ignored.
- "also Java" / "Java too" add to list values instead of replacing them.
- The same follow-up phrase **without** context returns `needs_clarification`
  ("This sounds like a follow-up, but I don't have the earlier request").
- A context naming an intent that does not exist in the registry is ignored.

---

## Adding intents and domains

Intents live in `app/data/intents.json` (core) and `app/data/domains/*.json` (extensions, loaded
automatically). Adding a domain needs no code change. A minimal domain file:

```json
{
  "version": "1.0",
  "intents": [
    {
      "id": "food.find_recipe",
      "label": "find a recipe",
      "action": "search",
      "target": "recipes",
      "domain": "food",
      "description": "Find recipes by dish, ingredient, cuisine or diet.",
      "examples": ["find a recipe for paneer butter masala", "what can I cook with eggs and spinach",
                   "vegan dessert recipes", "quick dinner ideas under 20 minutes"],
      "negative_examples": ["order dinner from a restaurant"],
      "parameters": [
        {"name": "dish", "types": ["topic"]},
        {"name": "cuisine", "types": ["cuisine"], "kind": "filter"},
        {"name": "time", "types": ["duration"], "kind": "filter"}
      ],
      "side_effects": "none"
    }
  ],
  "gazetteers": {
    "cuisine": {"Indian": ["indian"], "Italian": ["italian"], "Thai": ["thai"]}
  }
}
```

- `id` must look like `domain.name`; ids must be unique; `action` must be declared (core
  actions, or an `actions` section in your file with at least 3 **topic-neutral** example phrases).
- `parameters[].types` lists extracted span types in priority order. Built-in types include
  `money`, `money_constraint`, `date`, `date_range`, `time`, `duration`, `quantity`, `percentage`,
  `number`, `rating`, `experience`, `people_count`, `location`, `origin`, `destination`,
  `technology`, `natural_language`, `target_language`, `source_language`, `identifier`, `email`,
  `url`, `phone`, `file`, `error_message`, `quoted_text`, `code`, `content`, `document_ref`,
  `comparison_items`, `comparison_criteria`, `topic`, `use_case`, `requirements`, `recipient`,
  `reminder_task`, `person`, plus every gazetteer name (here: `cuisine`).
- `required: true`, `required_any: [["date","time"]]`, `recommended: true` control
  `missing_fields` / `optional_missing_fields`.
- `always_clarify` marks intents that carry no concrete task; `out_of_scope: true` marks kinds
  of requests to refuse (see `domains/unsupported.json` — delete entries you do support).
- Write 8–15 varied examples per intent (imperative, question, indirect, casual). Coverage of
  phrasing matters much more than quantity of near-duplicates.

After editing, re-run `python scripts/evaluate.py` (and ideally add eval cases for the new intent).
The intent-embedding cache is keyed by the text content, so it refreshes automatically.

User messages can never modify the registry: it is loaded once at startup and there is no API for changing it.

---

## Configuration

All settings are environment variables prefixed `ROUTER_` (or a `.env` file; see `.env.example`).

| Variable | Default | Notes |
|---|---|---|
| `ROUTER_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | or `BAAI/bge-small-en-v1.5` |
| `ROUTER_ALLOW_MODEL_DOWNLOAD` | `false` | `false` = local files only, no network |
| `ROUTER_MODEL_CACHE_DIR` | HF default | |
| `ROUTER_TORCH_THREADS` | `min(4, cores/2)` | more threads made single requests slower |
| `ROUTER_REJECT_THRESHOLD`, `ROUTER_ACCEPT_THRESHOLD`, `ROUTER_MARGIN_THRESHOLD` | per model, from `app/data/thresholds.json` | explicit values override the file |
| `ROUTER_TOP_K`, `ROUTER_ACTION_WEIGHT`, `ROUTER_NEGATIVE_PENALTY`, `ROUTER_HEAD_WEIGHT` | per model, from `thresholds.json` (`scoring`) | |
| `ROUTER_DELEXICALIZE`, `ROUTER_TYPO_CORRECTION` | `true` | |
| `ROUTER_MAX_MESSAGE_CHARS` | `2000` | longer → HTTP 413 |
| `ROUTER_MAX_TASKS` | `4` | clauses considered for multi-intent |
| `ROUTER_ENABLE_MULTI_INTENT` | `true` | |
| `ROUTER_DATE_ORDER` | `DMY` | how to read `12/11/2026`; ambiguous ones are flagged in the value |
| `ROUTER_FALLBACK` | `none` | `local_llm` enables the optional fallback |
| `ROUTER_LOG_LEVEL` | `INFO` | |
| `ROUTER_LOG_MESSAGES` | `false` | user text is never logged unless `true` |

---

## Thresholds and tuning

The values in `app/data/thresholds.json` are **provisional**. Similarity scores depend on the
model, the example phrases and your users' language, so thresholds must be re-tuned whenever
any of those change, and they do not guarantee any accuracy.

- `reject`: below this best score → `unknown`.
- `accept`: between `reject` and `accept` → ask the user to confirm the best guess.
- `margin`: if the runner-up is within this distance → `ambiguous`.
- `segment_accept`: each clause of a multi-intent split must reach this score.

Tune on labelled examples, never on the set you report:

```powershell
python scripts/tune_thresholds.py                                          # MiniLM, prints the best values
python scripts/tune_thresholds.py --model BAAI/bge-small-en-v1.5 --write   # save to thresholds.json
python scripts/evaluate.py --split final --errors                          # then measure on held-out data
```

The tuner grid-searches scoring weights, then thresholds, on the `dev + test + holdout` splits
(366 cases), maximizing the mean of macro-F1 (intents + a "reject" class) and overall accuracy.
It refuses to tune on `final`. Constraints that are enforced rather than tuned:

- `accept` stays above `reject`, so weak matches always ask for confirmation;
- `margin` is at least 0.02, so near-ties are reported as ambiguous;
- **every example phrase in the registry must route to its own intent** with at least that
  margin (they are labelled data by definition; one tuning run "improved" eval accuracy with
  weights that made "Book a flight from Delhi to Mumbai next Friday", a `travel.book` example,
  come out ambiguous);
- ties resolve to the most conservative setting (higher `reject`, wider confirmation band).

Why the pool and not just `dev`: tuning on the 120-case dev split alone was mostly noise. Two
configurations that scored the same on dev (0.944 vs 0.949 macro-F1) differed by 6 points on
the holdout split. More tuning data reduces that, but expect a few points of variation anyway.

To adapt it to your traffic: write 200+ real (anonymized) messages with the intent you want,
including out-of-scope ones and vague ones, tune on most of them and keep ~20% aside, unseen,
for the final number.

---

## Evaluation results

### Data

All evaluation data is hand-written (`app/data/eval/*.jsonl`, 437 cases). None of it is used as
intent examples. Categories: direct requests, paraphrases, misspellings, indirect requests,
requests with several constraints, overlapping intents, vague/ambiguous input, out-of-scope
requests, multi-action messages, "A and B" messages that must *not* split, empty/malformed input,
and follow-ups with and without context. Relative dates assume "today" = 2026-10-22.

| Split | Cases | Role | How "clean" it is |
|---|---|---|---|
| `dev` | 120 | tuning | used for development and tuning from the start |
| `test` | 123 | tuning | written blind; first run gave macro-F1 **0.800**; its errors were then inspected while improving the registry |
| `holdout` | 123 | tuning | written blind after that; first run gave macro-F1 **0.884**; its errors were then inspected |
| `final` | 71 | **reporting** | written blind before the last changes; never tuned on, individual errors not looked at until the numbers below were fixed |

Only `final` is a clean estimate, and with 71 cases it is a rough one: one case is about
1.4 points, and a 95% interval on accuracy is roughly ±6 points. The other splits are shown for
transparency; they are in-sample for the thresholds.

### Results (current registry and thresholds)

Metrics: **intent accuracy** = in-scope single requests routed to the right intent;
**macro-F1** = over all intents plus a "reject" class (unknown/ambiguous/clarify), so wrong
guesses and wrong refusals both cost; **unknown rejected** = out-of-scope requests that were
not routed to any intent; **false rejection** = in-scope requests sent to unknown/ambiguous/clarify;
**extraction** = exact match of the expected entity/filter values, on correctly routed cases.

| | MiniLM `final` | BGE `final` | MiniLM `holdout`* | BGE `holdout`* | MiniLM `test`* | BGE `test`* |
|---|---|---|---|---|---|---|
| macro-F1 (intents + reject) | **0.877** | **0.934** | 0.875 | 0.865 | 0.925 | 0.940 |
| intent accuracy | 0.922 | 0.941 | 0.904 | 0.872 | 0.946 | 0.936 |
| overall accuracy (all categories) | 0.930 | 0.944 | 0.886 | 0.862 | 0.935 | 0.935 |
| unknown requests rejected | 0.875 (7/8) | 0.875 (7/8) | 0.727 | 0.636 | 0.818 | 0.818 |
| false rejection of in-scope | 0.078 | 0.039 | 0.064 | 0.117 | 0.043 | 0.065 |
| extraction field accuracy | 1.000 (50 fields) | 1.000 | 1.000 (73) | 1.000 | 1.000 (100) | 1.000 |

\* in-sample for threshold tuning (see above).

Per category on `final` (MiniLM): ambiguous, follow-ups with/without context, indirect,
malformed, multi-intent, multi-constraint and "A and B" non-splits were all 100%;
direct 89%, typos 80%, unknown 88%. The five MiniLM misses on `final`:

| Message | Expected | Got |
|---|---|---|
| Write an apology email to a customer | `document.create` | ambiguous |
| Write a C++ program that prints a pyramid | `programming.write_code` | ambiguous |
| What do data analysts earn in Mumbai? | `jobs.salary_insights` | ambiguous |
| wat is the boiling point of milk | `general.question` | unknown |
| Get me a pizza delivered | unknown | `ecommerce.order_status` |

Four of the five are cautious failures (asking instead of guessing); one is a wrong guess.

### MiniLM vs BGE

```powershell
python scripts/compare_models.py          # add --tune to re-tune both first (~15 min)
```

Each model was tuned separately on the same 366 cases (thresholds are model-specific: BGE
similarities sit in a much higher band — gibberish scores ~0.54 — so MiniLM's thresholds are
meaningless for it) and evaluated on the same splits.

| | all-MiniLM-L6-v2 | bge-small-en-v1.5 |
|---|---|---|
| macro-F1 on `final` / `holdout` | 0.877 / 0.875 | 0.934 / 0.865 |
| latency p50 / p95, interleaved on the same inputs | **16.5 / 26.1 ms** | 27.3 / 45.0 ms |
| throughput, 4 concurrent clients | **56 req/s** | 34 req/s |
| disk | ~90 MB | ~130 MB |
| tuned thresholds (reject / accept / margin) | 0.25 / 0.45 / 0.02 | 0.50 / 0.70 / 0.02 |

BGE did better on `final` and worse on `holdout`; given the set sizes, the accuracy difference is
not conclusive. BGE is about 1.65× slower. **MiniLM stays the default**; switch with
`ROUTER_MODEL_NAME=BAAI/bge-small-en-v1.5` if you can afford the latency, and re-check on your own data.

### What changed the numbers (measured on dev during development)

| Stage of development | dev macro-F1 |
|---|---|
| first full evaluation (examples + action stage + delexicalization) | 0.78 |
| + topic-neutral question/search templates, lone-noun and follow-up rules | 0.85 |
| + head view | 0.89 |
| + tuned weights and thresholds | 0.95 (in-sample) |

Delexicalization was added before the first full evaluation, so it has no row of its own. Before
it, the three "Python jobs in Noida" paraphrases did not even reach `jobs.search`'s top-3
(best example similarity 0.31); after it they scored 0.77–0.91.

Reproduce: `python scripts/evaluate.py --split final --errors` (or `--split all`, `--model ...`, `--json out.json`).

---

## Performance

**Measured on the development PC, not on the target machine**: Intel Core i5-14400 (10 cores,
16 threads), 15.7 GB RAM, Windows 11, Python 3.12, torch 2.13 CPU, 4 torch threads. The target
(Ryzen 5 5600H, 8 GB) was not available; it has fewer and slower cores, so expect higher numbers.
Run the benchmark there before relying on any figure:

```powershell
python scripts/benchmark.py                      # add --model ..., --threads N, --json out.json
```

Timing on this PC was also noisy: the same single-sentence encode measured 5 ms in one session and
24 ms in another (power state / background load). Treat the numbers as orders of magnitude.

### Per-request latency, MiniLM (376 requests: dev + test + holdout + final messages, ×2)

| Stage | p50 | p95 |
|---|---|---|
| extraction (all regex/gazetteer/phrase extractors) | 0.5 ms | 2.9 ms |
| embedding (one batched forward pass: text + head view, plus clauses/context if any) | 14.1 ms | 22.6 ms |
| scoring (similarities, action stage) | 2.0 ms | 2.3 ms |
| decision + parameter mapping | 0.15 ms | 0.19 ms |
| Pydantic validation | 0.05 ms | 0.06 ms |
| JSON serialization | 0.05 ms | 0.07 ms |
| **total (`route()` wall time)** | **17.2 ms** | **26.0 ms** |

Embedding a single text alone: p50 11.2 ms. Inference dominates; everything else together is
about 3 ms. HTTP overhead (uvicorn) is not included.

**Concurrency**: inference is serialized by a lock, so throughput per process is about
1 / latency. With 4 concurrent client threads: 56 requests/s, p50 71 ms, p95 96 ms (queueing).
Micro-batching was not added: a single request is already one small forward pass, and at this
load batching would add waiting time without much gain. For more throughput, run more worker
processes (each needs its own ~450 MB).

**Threads**: 10 torch threads (the default on this PC) made a single encode ~2× slower than
2–4 threads (25 ms vs 12 ms). The router therefore uses `min(4, cores/2)` threads unless
`ROUTER_TORCH_THREADS` says otherwise.

### Startup and memory

| | MiniLM | BGE |
|---|---|---|
| `import torch, sentence_transformers` | 8.5–14 s | (same) |
| model load from local cache | 0.4–1.1 s | 0.3–0.5 s |
| embed 927 registry phrases (first start) | 2.3 s | 4.4 s |
| … from the on-disk cache (later starts) | 0.14 s | 0.14 s |
| process RSS after imports | 382 MB | 382 MB |
| process RSS after model load | 405 MB | 404 MB |

Most of the memory and startup time is PyTorch + Transformers themselves, not the model.
A running API process is roughly 450–500 MB, which leaves plenty of room on an 8 GB machine.
The registry-embedding cache lives in `.cache/embeddings` and is keyed by the model name and
the exact text of every phrase, so it is invalidated automatically when intents change.

---

## Optional local LLM fallback

Disabled by default. When enabled, a small local instruction-tuned model is asked to choose
**one** intent id among the top-5 embedding candidates (or `"unknown"`) — only for
`ambiguous` or low-confidence results.

```powershell
python scripts/download_models.py --fallback      # Qwen/Qwen2.5-0.5B-Instruct, about 1 GB
$env:ROUTER_FALLBACK = "local_llm"
uvicorn app.main:app
```

- It can only pick an id the embedding stage already proposed; anything else is discarded.
- Its output must parse as JSON; the user message is passed as quoted data with an instruction
  to ignore instructions inside it.
- It never extracts entities, so it cannot invent parameter values.
- If the model cannot be loaded, the fallback disables itself after one logged warning and
  routing continues normally.
- `resolved_by: "fallback"` marks results it decided.

What was verified here: the validation rules and the engine integration (with a stub model)
are covered by `tests/test_fallback.py`, and the "model not available" path was exercised for
real. **Generation quality, latency and memory were not measured**, because no complete
instruction model was available offline on this machine. Text generation on CPU is much slower
than one embedding pass, and the model needs its own memory on top of the router's ~450 MB, so
measure both on the target machine (`--debug` shows timings) before enabling it on an 8 GB PC.

---

## Security and reliability

- User input is untrusted data: it is normalized, length-limited (2000 chars, 256 KB body), and
  never interpreted as instructions to the router. There is no endpoint that changes the registry.
- The router **never executes** anything and never reports execution results. `operation` in an
  intent definition is only a hint for the caller's own handler table.
- `confidence` is not authorization. Intents with `side_effects: destructive` or `financial`
  (cancel order, bookings) set `requires_confirmation: true`; the caller must still authorize them.
- All model-derived and extracted values pass through the Pydantic schema; unknown fields are rejected.
- User messages are not logged by default (only length, status, intent, confidence, latency).
  Validation errors do not echo the submitted input.
- Concurrency: the FastAPI endpoint is synchronous and runs in the thread pool; model inference is
  serialized by a lock (concurrent forward passes on a small CPU only slow each other down).
  The engine itself is stateless.
- Only safe data is cached: embeddings of the registry's own example phrases (`.cache/embeddings`).
  User messages and their embeddings are never cached.

---

## Limitations

This is an embedding-similarity router, not a language-understanding model. It works well for
requests that resemble its examples and degrades for everything else. Specifically:

**Routing**

- **Similarity is not understanding.** Small sentence encoders weight topic words far more than
  the requested action (see the table in [How it works](#how-it-works)). Delexicalization, the
  head view and the action stage reduce this but do not remove it: "Write an apology email" and
  "Write a C++ program" still came back ambiguous on the final set, and `write_code` vs
  `refactor` is a recurring confusion.
- **Open-ended questions on unseen topics score low.** `general.question` can only be as broad
  as its examples; "wat is the boiling point of milk" was rejected as unknown.
- **Out-of-scope detection is incomplete.** 64–88% of out-of-scope requests were rejected,
  depending on the split. It relies on the reject threshold plus the listed "unsupported"
  categories; requests that share words with a supported intent leak through
  ("Get me a pizza delivered" → order status).
- **It is cautious.** 4–8% of in-scope requests come back as `ambiguous` / `needs_clarification`.
  That is a design choice (don't pretend), but your UI must handle the clarification question.
- **`confidence` is not calibrated** and must not be used for authorization.
- **Small evaluation sets.** The clean final set has 71 cases (±6 points). Two threshold settings
  that scored equal on the 120-case dev set differed by 6 points on another split.
- **Negative examples currently have zero weight.** The mechanism is implemented and the
  registry has negative examples, but tuning found no measurable benefit for either model.
- **English only.** Code-mixed Hindi/English ("Noida mein Python jobs") was not evaluated.

**Extraction** (deterministic, so precise but limited)

- Gazetteers are finite: unknown cities are only found when capitalized after "in/near/around";
  unknown products, brands and roles are not extracted (the text is still preserved in `topic`
  or `requirements` where the intent declares them).
- Dates: "next Friday" means the next occurrence (English is ambiguous here; the original text is
  kept in the value). `12/11/2026` is read day-first unless `ROUTER_DATE_ORDER=MDY`.
  "in 3 days" is a duration, not a date. Recurrence ("every weekday") is not parsed. "at 6"
  without am/pm is left out rather than guessed.
- No negation: "not remote" or "except internships" is not turned into a negative filter.
- Amounts without a currency keep `currency: null` (the router does not assume INR).

**Multi-intent and follow-ups**

- Clause splitting is syntactic (sentence breaks, "then", "and" + a verb), then validated
  semantically. Long or unusual sentences may not split, and a split clause is routed without
  the rest of the sentence.
- Dependencies are detected from pronouns and sequence words ("their", "it", "then") only.
- Follow-up detection is phrase-based ("only…", "what about…", "…ones"). A follow-up that looks
  like a full new request is treated as one.

**Other**

- The optional LLM fallback was not quality-tested (see above).
- Typo correction affects only the routing text and only words close to the registry vocabulary;
  it can occasionally "correct" a rare real word to a domain word.
- Performance figures come from a faster machine than the target.

---

## Tests

134 tests, all passing on this machine (also re-checked in a fresh virtual environment with
newer library versions: torch 2.14.1+cpu, sentence-transformers 5.7.0, FastAPI 0.159, Pydantic 2.14).

```powershell
python -m pytest                 # all tests (loads the model once; about a minute)
python -m pytest tests/test_extraction.py tests/test_schema.py   # no model needed, < 1 s
```

| File | Covers |
|---|---|
| `test_schema.py` | schema validity, bounds, `extra="forbid"`, JSON round trip, registry validation |
| `test_extraction.py` | money, dates, times, ranges, roles, comparisons, masking, no invented values |
| `test_routing.py` | paraphrases, typos, core intents, action vs. topic, side-effect flags, every registry example routes to its own intent |
| `test_ambiguity.py` | unknown, out-of-scope, lone nouns, vague requests, empty input, registry immutability |
| `test_multi_intent.py` | ordered tasks, dependencies, no false splits on "A and B" |
| `test_context.py` | follow-ups with and without context, chaining, no hidden state |
| `test_api.py` | `/health`, `/route`, 413/422 handling, error format |
| `test_fallback.py` | fallback validation rules, failure path, engine integration (stub) |
| `test_evaluation.py` | metric helpers, regression floor on the dev set |

---

## Project layout

```text
app/
  main.py                 FastAPI app (lifespan loads the model once; error handlers; body limit)
  cli.py                  python -m app.cli
  evaluation.py           metrics used by scripts and tests
  api/routes.py           POST /route, GET /health
  api/schemas.py          request models
  core/config.py          settings (env ROUTER_*), per-model thresholds
  core/logging.py         logging without user text
  core/schemas.py         RouteResult, Task, ConversationContext (schema v1.0)
  models/loader.py        loads the sentence-transformer once, offline by default
  models/embeddings.py    thread-safe, normalized embeddings
  router/normalize.py     normalization, payload/code separation, delexicalization, typo correction
  router/registry.py      intent registry + validation
  router/similarity.py    scoring (examples, head view, negatives, action stage) + embedding cache
  router/confidence.py    thresholds, margin, heuristic confidence
  router/segmenter.py     clause splitting for multi-intent
  router/fallback.py      optional local-LLM fallback (off by default)
  router/engine.py        orchestration
  extractors/base.py      Span / Extractor interface
  extractors/rules.py     regex extractors (money, dates, ...)
  extractors/entities.py  gazetteers, place names, phrases
  extractors/pipeline.py  overlap resolution + parameter mapping
  services/intent_service.py   builds everything at startup; route(); health()
  services/context_service.py  follow-up detection and context merging
  data/intents.json       core actions + intents
  data/domains/*.json     jobs, ecommerce, travel, unsupported (extensions)
  data/gazetteers.json    known values (cities, skills, languages, ...)
  data/thresholds.json    per-model thresholds and scoring weights (provisional)
  data/eval/*.jsonl       hand-written evaluation sets (dev / test / holdout / final)
scripts/
  download_models.py  benchmark.py  evaluate.py  tune_thresholds.py  compare_models.py
tests/
```
#   i n t e n t - r o u t e r  
 