"""Explicit conversation context handling.

The router is stateless: the caller passes a ConversationContext (usually the
`context` object from the previous response) and receives an updated one.
"""

from __future__ import annotations

import re
from typing import Any

from intent_router.core.schemas import ConversationContext
from intent_router.router.segmenter import VERB_CUES

I = re.IGNORECASE

_FOLLOWUP_START = re.compile(
    r"^(?:only|just|but|and|also|what about|how about|same|instead|now|make it|ok(?:ay)?|then|any|show only|filter|"
    r"sort|in|near|under|below|above|over|cheaper|remote|with|without|for|from|what if|try|exclude|include|more|"
    r"less|fewer|not|no|same but|actually)\b",
    I,
)
_FOLLOWUP_ANY = re.compile(r"\b(?:ones?|those|them|these|that one|instead|as well|too|the same|same thing)\b", I)
_EXPLICIT_FOLLOWUP_START = re.compile(
    r"^(?:only|same(?:\s+but|\s+thing)?|instead|also|what about|how about|"
    r"and\s+(?:in|for|with|under|below|above|over|on|at))\b",
    I,
)
_ADDITIVE = re.compile(r"(?:\b(?:also|too|as well|plus|add)\b|^and\b)", I)
MAX_PREVIOUS_CHARS = 500


def looks_like_follow_up(text: str) -> bool:
    words = text.split()
    if not words:
        return False
    if _FOLLOWUP_START.match(text):
        return True
    return len(words) <= 6 and bool(_FOLLOWUP_ANY.search(text))


def refers_to_previous(text: str) -> bool:
    """Phrases that only make sense relative to an earlier result ("the cheaper ones")."""
    return bool(re.search(r"\b(?:ones|those ones|the same|same but|same thing|that one|instead|what about|how about)\b",
                          text, I))


def is_follow_up_phrase(text: str) -> bool:
    return looks_like_follow_up(text) or refers_to_previous(text)


def needs_prior_context(text: str) -> bool:
    """Strong follow-up cues that still sound like a follow-up when no context is given."""
    return refers_to_previous(text) or bool(_EXPLICIT_FOLLOWUP_START.match(text.strip()))


# Verbs that make a message a request of its own ("now summarize it"), as opposed to
# a refinement ("only remote ones", "what about Delhi?"). Question words and fillers
# are excluded because they also start refinements.
_NOT_ACTIONS = {"what", "how", "which", "who", "where", "when", "can", "could", "would", "please", "also", "then",
                "look", "help"}
ACTION_VERBS = frozenset(VERB_CUES - _NOT_ACTIONS)


def has_own_action(text: str) -> bool:
    return any(w.lower() in ACTION_VERBS for w in re.findall(r"[A-Za-z]+", text))


def is_additive(text: str) -> bool:
    return bool(_ADDITIVE.search(text))


def merge_values(old: dict[str, Any], new: dict[str, Any], additive: bool) -> dict[str, Any]:
    """New values override old ones; lists are unioned when the user adds ("also Java")."""
    merged = dict(old)
    for key, value in new.items():
        if key == "other":
            continue
        if additive and isinstance(merged.get(key), list) and isinstance(value, list):
            merged[key] = merged[key] + [v for v in value if v not in merged[key]]
        else:
            merged[key] = value
    return merged


def combined_text(context: ConversationContext, message: str) -> str | None:
    if not context.previous_message:
        return None
    return f"{context.previous_message} {message}"[-MAX_PREVIOUS_CHARS:]


def next_context(message: str, *, intent: str, action: str, target: str | None, domain: str,
                 entities: dict[str, Any], filters: dict[str, Any], confidence: float,
                 previous: ConversationContext | None = None, follow_up: bool = False) -> ConversationContext:
    prev_text = message
    if follow_up and previous and previous.previous_message:
        prev_text = f"{previous.previous_message} {message}"
    return ConversationContext(
        previous_message=prev_text[-MAX_PREVIOUS_CHARS:],
        intent=intent,
        action=action,
        target=target,
        domain=domain,
        entities={k: v for k, v in entities.items() if k != "other"},
        filters=dict(filters),
        confidence=confidence,
    )
