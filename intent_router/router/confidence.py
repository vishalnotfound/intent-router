"""Decision policy on top of similarity scores.

Cosine similarity is not a probability. The ``confidence`` value produced here is
a monotonic heuristic that combines how far the best score is above the reject
threshold with how clearly it beats the runner-up. It is useful for ranking and
for spotting weak results, but it is not calibrated: "0.8" does not mean "right
80% of the time". Calibrate it on your own labelled data if you need that.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from intent_router.core.config import Thresholds
from intent_router.router.registry import IntentDefinition, IntentRegistry


class Outcome(str, Enum):
    accept = "accept"
    low_confidence = "low_confidence"  # between reject and accept thresholds
    ambiguous = "ambiguous"  # top-2 margin too small
    unknown = "unknown"  # below reject threshold


@dataclass
class Candidate:
    index: int
    intent: IntentDefinition
    score: float


@dataclass
class Decision:
    outcome: Outcome
    best: Candidate
    runner_up: Candidate | None
    margin: float
    confidence: float
    close: list[Candidate] = field(default_factory=list)  # candidates within the margin of best
    shared_action: str | None = None  # set when all close candidates share one action
    ranked: list[Candidate] = field(default_factory=list)  # top candidates, best first

    @property
    def accepted(self) -> bool:
        return self.outcome is Outcome.accept


def rank(scores: np.ndarray, registry: IntentRegistry, limit: int | None = None) -> list[Candidate]:
    order = np.argsort(-scores)
    if limit:
        order = order[:limit]
    return [Candidate(int(i), registry.intents[int(i)], float(scores[int(i)])) for i in order]


def heuristic_confidence(best: float, second: float | None, t: Thresholds) -> float:
    span = max(t.accept - t.reject, 1e-6)
    abs_part = float(np.clip((best - t.reject) / span, 0.0, 1.5)) / 1.5
    if second is None:
        margin_part = 1.0
    else:
        margin_part = float(np.clip((best - second) / max(t.margin, 1e-6), 0.0, 3.0)) / 3.0
    return round(min(0.99, 0.6 * abs_part + 0.4 * margin_part), 4)


def decide(scores: np.ndarray, registry: IntentRegistry, t: Thresholds) -> Decision:
    ranked = rank(scores, registry)
    best = ranked[0]
    runner = ranked[1] if len(ranked) > 1 else None
    margin = best.score - runner.score if runner else 1.0
    confidence = heuristic_confidence(best.score, runner.score if runner else None, t)

    close = [c for c in ranked[1:] if best.score - c.score < t.margin]
    actions = {best.intent.action, *(c.intent.action for c in close)}
    shared_action = best.intent.action if len(actions) == 1 else None

    if best.score < t.reject:
        outcome = Outcome.unknown
    elif close:
        outcome = Outcome.ambiguous
    elif best.score < t.accept:
        outcome = Outcome.low_confidence
    else:
        outcome = Outcome.accept
    return Decision(outcome, best, runner, margin, confidence, close, shared_action, ranked[:8])
