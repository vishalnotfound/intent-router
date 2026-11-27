"""Exclusions ("anywhere except Delhi") and negated actions ("do not cancel ...").

Extraction is deterministic, so negation is handled here with explicit cue
lists instead of being left to the embedding model, which barely separates
"cancel my order" from "do not cancel my order".

- :func:`mark_exclusions` flags categorical spans (places, skills, brands, ...)
  that the user wants *excluded*. Parameter mapping never uses a flagged span as
  a value to include.
- :func:`negated_actions` finds action verbs the user told us *not* to perform.
"""

from __future__ import annotations

import re
from dataclasses import replace

from intent_router.extractors.base import Span

I = re.IGNORECASE

# Only categorical values can be excluded. Numbers, dates, ids and free-text
# phrases are left alone ("order no 4421" must not exclude the id).
_EXCLUDABLE_SOURCES = frozenset({"gazetteer", "fuzzy", "pattern", "overlay"})

_EXCLUSION_CUE = re.compile(
    r"(?:\bexcept(?:\s+for)?|\bexcluding|\bexclude|\bother\s+than|\bapart\s+from|\baside\s+from|"
    r"\b(?:anywhere|anything|any\s+\w+)\s+but|\bbut\s+not|\bnot|\boutside(?:\s+of)?|\bavoid(?:ing)?|\bwithout|"
    r"\bminus|\bskip(?:ping)?|\bleave\s+out|\bleaving\s+out)\s*$",
    I,
)
# Words that may sit between the cue and the value: "except in Delhi", "not the remote ones".
_FILLER = re.compile(r"\s*\b(?:in|at|from|the|a|an|any|for|on|near|around|of|to|into)\s*$", I)
# A list continuing an exclusion: "except Delhi and Mumbai", "not Delhi, Pune or Noida".
_LIST_JOIN = re.compile(r"^\s*(?:,\s*)?(?:(?:and|or|nor)\s+)?(?:(?:in|at|from|the)\s+)?$", I)


def _cue_before(prefix: str) -> bool:
    prefix = prefix.rstrip()
    for _ in range(3):
        m = _FILLER.search(prefix)
        if not m:
            break
        prefix = prefix[:m.start()].rstrip()
    return bool(_EXCLUSION_CUE.search(prefix))


def mark_exclusions(text: str, spans: list[Span]) -> list[Span]:
    """Return ``spans`` with excluded categorical values flagged ``excluded=True``."""
    excluded_positions: set[tuple[int, int]] = set()
    last_excluded_end: int | None = None
    candidates = sorted((s for s in spans if s.source in _EXCLUDABLE_SOURCES and s.length > 0),
                        key=lambda s: (s.start, -s.length))
    for s in candidates:
        if (s.start, s.end) in excluded_positions:
            continue
        chained = last_excluded_end is not None and _LIST_JOIN.match(text[last_excluded_end:s.start]) is not None
        if chained or _cue_before(text[:s.start]):
            excluded_positions.add((s.start, s.end))
            last_excluded_end = s.end
        elif last_excluded_end is not None and s.start >= last_excluded_end:
            last_excluded_end = None
    if not excluded_positions:
        return spans
    return [replace(s, excluded=True) if (s.start, s.end) in excluded_positions and s.source in _EXCLUDABLE_SOURCES
            else s for s in spans]


# --------------------------------------------------------------------------- negated actions
# Instructions not to do something. Statements about ability or state ("I can't
# find my order", "the app won't open") are not instructions and are not listed.
_NEGATION = re.compile(
    r"\b(?:do\s+not|don[’']?t|dont|never|no\s+need\s+to|not\s+to|"
    r"(?:i|we)\s+(?:do\s+not|don[’']?t|dont)\s+(?:want|need|wish)\s+(?:you\s+)?to|"
    r"(?:i|we)\s+(?:would|'d)\s+(?:rather|prefer)\s+(?:you\s+)?not|"
    r"(?:please\s+)?refrain\s+from|hold\s+off\s+on)\s+",
    I,
)
# Words allowed between the negation and the verb: "don't go ahead and cancel".
_NEG_FILLER = frozenset({"please", "go", "ahead", "and", "actually", "yet", "just", "even", "accidentally", "ever",
                         "really", "you", "to", "be", "the", "my"})
_WORD = re.compile(r"[A-Za-z]+")


_BARE_REFUSAL = re.compile(
    r"^(?:(?:actually|wait|no|nope|hmm|oh|sorry|ok(?:ay)?|on\s+second\s+thought)[\s,.!]+)*"
    r"(?:no|nope|don[’']?t|dont|do\s+not|never\s*mind|nevermind|forget\s+it|stop|hold\s+on|wait|"
    r"not\s+now|leave\s+it(?:\s+as\s+(?:it\s+)?is)?|(?:i(?:'ve|\s+have)?\s+)?changed\s+my\s+mind|"
    r"(?:i(?:'ll|\s+will|\s+want\s+to|'d\s+like\s+to)?\s+)?keep\s+(?:it|them|the\s+\w+|my\s+\w+))"
    r"(?:[\s,]+(?:do\s+(?:it|that|this)|it|that|this|then|please|yet|after\s+all|anymore))*[\s.!]*$",
    I,
)


def is_bare_refusal(text: str) -> bool:
    """A message that only withdraws the previous request: "actually, don't", "no, stop", "never mind"."""
    return bool(_BARE_REFUSAL.match(text.strip()))


def _verb(word: str, action_verbs: frozenset[str]) -> str | None:
    # Exact forms only: stemming would read nouns such as "orders" or "plans" as verbs.
    word = word.lower()
    return word if word in action_verbs else None


def _gerund_verb(word: str, action_verbs: frozenset[str]) -> str | None:
    """"cancelling" -> "cancel", "booking" -> "book", "reserving" -> "reserve". Only
    used right after a negation cue ("hold off on cancelling"), where a noun is unlikely."""
    word = word.lower()
    if not word.endswith("ing") or len(word) < 6:
        return None
    stem = word[:-3]
    for candidate in (stem, stem + "e", stem[:-1] if len(stem) > 2 and stem[-1] == stem[-2] else None):
        if candidate and candidate in action_verbs:
            return candidate
    return None


def _negated_verbs(text: str, action_verbs: frozenset[str]) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    for m in _NEGATION.finditer(text):
        for k, wm in enumerate(_WORD.finditer(text, m.end())):
            verb = _verb(wm.group(0), action_verbs) or _gerund_verb(wm.group(0), action_verbs)
            if verb is not None:
                found.append((wm.start(), verb))
                break
            if wm.group(0).lower() not in _NEG_FILLER or k >= 3:
                break
    return found


def negated_actions(text: str, action_verbs: frozenset[str]) -> list[str]:
    """Action verbs the message tells us not to perform, e.g. ["cancel"] for "Do not cancel order 4421"."""
    return [verb for _, verb in _negated_verbs(text, action_verbs)]


def unnegated_actions(text: str, action_verbs: frozenset[str]) -> list[str]:
    """Action verbs in the message that are not under a negation."""
    negated_at = {pos for pos, _ in _negated_verbs(text, action_verbs)}
    return [verb for wm in _WORD.finditer(text)
            if wm.start() not in negated_at and (verb := _verb(wm.group(0), action_verbs)) is not None]
