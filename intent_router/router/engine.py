"""Routing engine: normalize -> embed -> score -> decide -> extract -> validate.

The engine is stateless. Conversation state arrives only through the explicit
ConversationContext argument, and the updated context is returned in the result.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from dataclasses import replace as dataclass_replace
from datetime import date
from typing import Any

import numpy as np

from intent_router.core.config import Settings, Thresholds
from intent_router.core.schemas import (
    SCHEMA_VERSION,
    Alternative,
    ConversationContext,
    DebugInfo,
    RouteResult,
    Status,
    Task,
)
from intent_router.extractors.base import SOFT_TYPES, ExtractionContext
from intent_router.extractors.negation import is_bare_refusal, negated_actions, unnegated_actions
from intent_router.extractors.pipeline import (
    ExtractionPipeline,
    MappedParameters,
    map_parameters,
    missing_for,
    unmapped_entities,
)
from intent_router.models.embeddings import Embedder
from intent_router.router.confidence import Candidate, Decision, Outcome, decide, rank
from intent_router.router.fallback import FallbackResolver, NoFallback
from intent_router.router.normalize import TypoCorrector, has_linguistic_content, mask_entities, prepare
from intent_router.router.registry import IntentDefinition, IntentRegistry
from intent_router.router.segmenter import (
    VERB_CUES,
    Segment,
    content_words,
    has_anaphora,
    split_clauses,
    split_parallel,
)
from intent_router.router.similarity import IntentIndex
from intent_router.router.verifier import ActionVerifier, NoVerifier
from intent_router.services.context_service import (
    ACTION_VERBS,
    combined_text,
    has_own_action,
    is_additive,
    is_follow_up_phrase,
    merge_values,
    needs_prior_context,
    next_context,
)

# Verbs whose negation ("do not cancel", "don't pay") must stop a request from being
# routed for execution: the request verbs plus account and money operations.
NEGATABLE_VERBS = ACTION_VERBS | frozenset(
    "stop unsubscribe refund pay transfer submit purchase renew terminate deactivate replace post publish run "
    "execute proceed place charge withdraw deposit close confirm approve".split()
)
# "Look for Java jobs" is a request of its own, although "look" alone is not.
_LOOK_FOR = re.compile(r"\blook(?:ing)?\s+(?:for|up)\b", re.IGNORECASE)
# Words that tie a message to the previous one even when it has its own verb.
_REFINES_PREVIOUS = re.compile(r"\b(?:again|instead|too|as well|also|still|more|other|another|similar)\b",
                               re.IGNORECASE)


@dataclass
class Resolution:
    """Outcome for one request or one clause of a multi-intent request."""

    status: Status
    intent: IntentDefinition | None
    action: str | None = None
    target: str | None = None
    domain: str | None = None
    entities: dict[str, Any] = field(default_factory=dict)
    filters: dict[str, Any] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    optional_missing: list[str] = field(default_factory=list)
    question: str | None = None
    confidence: float = 0.0
    alternatives: list[Alternative] = field(default_factory=list)
    resolved_by: str = "embedding"
    reasoning: str = ""
    fallback_used: bool = False
    tasks: list[Task] = field(default_factory=list)

    @property
    def needs_clarification(self) -> bool:
        return self.status is not Status.success


def _alt(c: Candidate) -> Alternative:
    i = c.intent
    return Alternative(intent=i.id, action=i.action, target=i.target, domain=i.domain,
                       score=round(float(np.clip(c.score, -1, 1)), 4))


class RouterEngine:
    def __init__(self, *, registry: IntentRegistry, index: IntentIndex, embedder: Embedder,
                 pipeline: ExtractionPipeline, thresholds: Thresholds, settings: Settings,
                 corrector: TypoCorrector | None = None, fallback: FallbackResolver | None = None,
                 verifier: ActionVerifier | None = None):
        self.registry = registry
        self.index = index
        self.embedder = embedder
        self.pipeline = pipeline
        self.t = thresholds
        self.settings = settings
        self.corrector = corrector
        self.fallback = fallback or NoFallback()
        self.verifier = verifier or NoVerifier()

    # ------------------------------------------------------------------ public
    def route(self, message: str, context: ConversationContext | None = None, *,
              reference_date: date | None = None, include_debug: bool = False) -> RouteResult:
        start = time.perf_counter()
        timings: dict[str, float] = {}
        prepared = prepare(message)
        instruction = prepared.instruction
        if not has_linguistic_content(instruction):
            return self._invalid(prepared.normalized, include_debug, start)

        ectx = ExtractionContext(reference_date=reference_date or date.today(), date_order=self.settings.date_order,
                                 content=prepared.content, code_blocks=prepared.code_blocks)

        # Extraction does not depend on the intent, so it runs first; its spans are
        # also used to delexicalize the text that gets embedded.
        t0 = time.perf_counter()
        spans = self.pipeline.extract(instruction, ectx)
        routing_text, corrections = self._routing_view(instruction, spans)
        segments = (split_clauses(instruction, self.settings.max_tasks)
                    if self.settings.enable_multi_intent else [Segment(instruction, 0, len(instruction), "start")])
        if (len(segments) == 1 and self.settings.enable_multi_intent
                and not any(s.type == "comparison_items" for s in spans)):  # "compare X in A and Y in B"
            segments = split_parallel(instruction, spans, SOFT_TYPES) or segments
        seg_texts = []
        if len(segments) > 1:
            seg_texts = [self._routing_view(s.text, self.pipeline.extract(s.text, ectx))[0] for s in segments]
        ctx_intent = self.registry.get(context.intent) if context and context.intent else None
        combined = None
        if ctx_intent is not None:
            raw_combined = combined_text(context, instruction)
            if raw_combined:
                combined = self._routing_view(raw_combined, self.pipeline.extract(raw_combined, ectx))[0]
        timings["extract_ms"] = (time.perf_counter() - t0) * 1000

        texts = [routing_text, *seg_texts] + ([combined] if combined else [])
        use_heads = self.index.head_weight > 0
        batch = texts + (self.index.head_texts(texts) if use_heads else [])
        t0 = time.perf_counter()
        vectors = self.embedder.encode(batch)  # one forward pass for every view
        timings["embed_ms"] = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        scores = self.index.score(vectors[:len(texts)], vectors[len(texts):] if use_heads else None)
        full = decide(scores.final[0], self.registry, self.t)
        seg_decisions = [decide(scores.final[1 + k], self.registry, self.t) for k in range(len(seg_texts))]
        combined_decision = decide(scores.final[-1], self.registry, self.t) if combined else None
        timings["score_ms"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        result = self._dispatch(instruction, spans, full, segments, seg_decisions, combined_decision, context,
                                ctx_intent, ectx, routing_text)
        timings["resolve_ms"] = (time.perf_counter() - t0) * 1000
        return self._build(result, context, instruction, prepared, routing_text, corrections, scores,
                           timings, start, include_debug)

    # ------------------------------------------------------------------ dispatch
    def _dispatch(self, instruction, spans, full, segments, seg_decisions, combined_decision, context, ctx_intent,
                  ectx, routing_text):
        if self._mostly_unrecognized(routing_text):
            return self._unrecognized(full, spans)
        if ctx_intent is not None:
            follow = self._follow_up(instruction, spans, full, combined_decision, context, ctx_intent)
            if follow is not None:
                return follow
        elif (len(instruction.split()) <= 6 and needs_prior_context(instruction) and not has_own_action(instruction)
              and not (full.accepted and full.best.intent.action == "converse")):
            return self._context_missing(spans, full)

        words = content_words(instruction)
        if (len(words) == 1 and words[0].lower() not in VERB_CUES and full.outcome is not Outcome.unknown
                and full.best.intent.action != "converse" and not full.best.intent.out_of_scope):
            return self._subject_only(words[0], full, spans)

        if len(segments) > 1:
            units = self._multi_units(segments, seg_decisions, full, ectx)
            if units is not None and len(units) >= 2:
                return self._multi(units, ectx)
            if units is not None and len(units) == 1 and units[0][1].accepted:
                return self._resolve(units[0][1], instruction, spans)
        return self._resolve(full, instruction, spans)

    def _mostly_unrecognized(self, routing_text: str) -> bool:
        """True for keyboard mashing such as "zxcv bnm lkjh": fewer than half of the words
        (after typo correction) exist in the model vocabulary or the registry vocabulary."""
        if self.corrector is None:
            return False
        words = re.findall(r"[A-Za-z]+", routing_text)
        if len(words) < 2:
            return False
        known = sum(1 for w in words if w.lower() in self.corrector.known)
        return known / len(words) < 0.5

    def _unrecognized(self, full: Decision, spans) -> Resolution:
        return Resolution(
            status=Status.unknown, intent=None, action="unknown", entities=self._other(spans),
            question="I couldn't understand that. Could you rephrase?", confidence=0.0,
            alternatives=[_alt(c) for c in self._top(full, include_best=True)], resolved_by="none",
            reasoning="Most of the words are not recognizable, so the message is not routed.",
        )

    def _routing_view(self, text: str, spans) -> tuple[str, list[tuple[str, str]]]:
        """The text that is embedded: delexicalized, then typo-corrected."""
        if self.settings.delexicalize:
            text = mask_entities(text, spans)
        if self.corrector is None or not self.settings.typo_correction:
            return text, []
        return self.corrector.correct(text)

    # ------------------------------------------------------------------ single request
    def _resolve(self, d: Decision, text: str, spans) -> Resolution:
        outcome, chosen, fallback_used, resolved_by = d.outcome, d.best, False, "embedding"

        if outcome in (Outcome.ambiguous, Outcome.low_confidence) and self.fallback.name != "none":
            candidates = d.ranked[:5]
            choice = self.fallback.resolve(text, [c.intent for c in candidates])
            if choice is not None:
                if choice.intent_id == "unknown":
                    fallback_used, resolved_by = True, "fallback"
                    outcome = Outcome.unknown
                else:
                    matched = next((c for c in candidates if c.intent.id == choice.intent_id), None)
                    if matched is not None:
                        fallback_used, resolved_by = True, "fallback"
                        outcome = Outcome.accept
                        chosen = matched

        if outcome in (Outcome.accept, Outcome.low_confidence, Outcome.ambiguous):
            # A supported intent's unsupported variant ("find candidates" vs "find jobs").
            considered = [chosen, *(d.close if outcome is Outcome.ambiguous else [])]
            for c in considered:
                variant = self.registry.unsupported_variant(c.intent.id, text)
                if variant is not None:
                    return Resolution(
                        status=Status.unknown, intent=None, action="unknown", entities=self._other(spans),
                        question=f"Sorry, I can't {variant.label}. I can only {c.intent.label}.",
                        confidence=0.0, alternatives=[_alt(x) for x in self._top(d, include_best=True)],
                        resolved_by="none", fallback_used=fallback_used,
                        reasoning=(f"The request looks like '{c.intent.id}' but asks to {variant.label} "
                                   f"('{variant.name}'), which this router does not support."),
                    )

        if outcome in (Outcome.accept, Outcome.low_confidence) and chosen.intent.out_of_scope:
            return Resolution(
                status=Status.unknown, intent=None, action="unknown", entities=self._other(spans),
                question="Sorry, I can't help with that kind of request.",
                confidence=0.0, alternatives=[_alt(c) for c in self._top(d, include_best=True)],
                resolved_by=resolved_by if fallback_used else "none", fallback_used=fallback_used,
                reasoning=(f"Closest match '{chosen.intent.id}' ({chosen.intent.label}) is a kind of request "
                           f"this router does not support (score {chosen.score:.2f})."),
            )

        if outcome is Outcome.unknown:
            return Resolution(
                status=Status.unknown, intent=None, action="unknown", entities=self._other(spans),
                question="I'm not sure what you'd like me to do. Could you rephrase or add a little detail?",
                confidence=0.0, alternatives=[_alt(c) for c in self._top(d, include_best=True)],
                resolved_by=resolved_by if fallback_used else "none", fallback_used=fallback_used,
                reasoning=(f"No intent matched well enough (best '{d.best.intent.id}' scored {d.best.score:.2f}, "
                           f"rejection threshold {self.t.reject:.2f})."
                           if not fallback_used else "The fallback model judged that no known intent fits."),
            )

        if outcome is Outcome.ambiguous:
            close = [d.best, *d.close]
            labels = [c.intent.label for c in close[:3]]
            domains = {c.intent.domain for c in close}
            return Resolution(
                status=Status.ambiguous, intent=None, action=d.shared_action,
                domain=domains.pop() if len(domains) == 1 else None, entities=self._other(spans),
                question="Did you want me to " + ", or ".join(labels) + "?",
                confidence=d.confidence, alternatives=[_alt(c) for c in self._top(d, include_best=True)],
                resolved_by="none",
                reasoning=(f"'{d.best.intent.id}' ({d.best.score:.2f}) and '{d.close[0].intent.id}' "
                           f"({d.close[0].score:.2f}) are within the {self.t.margin:.2f} margin."
                           + (f" Both are '{d.shared_action}' requests." if d.shared_action else "")),
            )

        intent = chosen.intent
        negated = self._negated(text, intent)
        if negated:
            return self._negated_resolution(negated, intent, d, spans)
        if outcome in (Outcome.accept, Outcome.low_confidence) and not intent.always_clarify:
            # Optional second gate for destructive/financial intents ("I no longer want to
            # cancel my order" has no negated verb for the rules to find).
            verdict = self.verifier.check(text, intent)
            if verdict is not None and verdict.blocked:
                return self._withdrawn(
                    f"It sounds like you don't want me to {intent.label}. What would you like me to do instead?",
                    ("The verifier could not check this request, so it is not routed for execution."
                     if verdict.error else
                     f"The NLI verifier judged that the message contradicts '{intent.id}' (contradiction "
                     f"{verdict.contradiction:.2f}, best entailment {verdict.entailment:.2f}), so it is not "
                     "routed as a request to carry out."),
                    d, spans)
        mapped = map_parameters(spans, intent)
        if mapped.unsupported_exclusions:
            return self._exclusion_resolution(intent, mapped, d.confidence,
                                              [_alt(c) for c in self._top(d, include_best=False, exclude=intent.id)])
        if outcome is Outcome.low_confidence:
            status = Status.needs_clarification
            question = f"Just to check: do you want me to {intent.label}?"
            reasoning = (f"Weak match for '{intent.id}' (score {chosen.score:.2f}, below the acceptance "
                         f"threshold {self.t.accept:.2f}).")
        elif intent.always_clarify:
            status = Status.needs_clarification
            question = intent.always_clarify
            reasoning = f"Matched '{intent.id}' (score {chosen.score:.2f}): the request does not say what to do yet."
        else:
            status = Status.needs_clarification if mapped.missing else Status.success
            question = mapped.clarification
            if fallback_used:
                reasoning = f"Embedding scores were inconclusive; the local fallback model chose '{intent.id}'."
            else:
                reasoning = (f"Best match '{intent.id}' ({intent.label}), score {chosen.score:.2f}, "
                             f"ahead of the next intent by {d.margin:.2f}.")
            if mapped.missing:
                reasoning += " Missing required: " + ", ".join(mapped.missing) + "."
        return Resolution(
            status=status, intent=intent, action=intent.action, target=intent.target, domain=intent.domain,
            entities=mapped.entities, filters=mapped.filters, missing=mapped.missing,
            optional_missing=mapped.optional_missing, question=question, confidence=d.confidence,
            alternatives=[_alt(c) for c in self._top(d, include_best=False, exclude=intent.id)],
            resolved_by=resolved_by, reasoning=reasoning, fallback_used=fallback_used,
        )

    # ------------------------------------------------------------------ negation & exclusions
    @staticmethod
    def _negated(text: str, intent: IntentDefinition) -> list[str]:
        """Negated action verbs that stop ``intent`` from being routed for execution.

        For an intent with side effects any negated request verb is enough ("do not
        cancel order 4421"). A read-only intent is still routed when another,
        non-negated request verb carries it ("don't book yet, just show me flights").
        """
        negated = negated_actions(text, NEGATABLE_VERBS)
        if not negated:
            return []
        if intent.side_effects not in ("none", "read"):
            return negated
        return [] if unnegated_actions(text, NEGATABLE_VERBS) else negated

    def _negated_resolution(self, verbs: list[str], intent: IntentDefinition, d: Decision | None,
                            spans) -> Resolution:
        verb = verbs[0]
        return self._withdrawn(
            f"You asked me not to {verb}, so I haven't treated this as a request to {intent.label}. "
            "What would you like me to do instead?",
            f"The action '{verb}' is negated, so '{intent.id}' is not routed as a request to carry out.",
            d, spans)

    def _withdrawn(self, question: str, reasoning: str, d: Decision | None, spans) -> Resolution:
        """A request the user does not want carried out: no intent, nothing executable."""
        return Resolution(
            status=Status.needs_clarification, intent=None, action=None, entities=self._other(spans),
            question=question, confidence=0.0,
            alternatives=[_alt(c) for c in self._top(d, include_best=True)] if d else [],
            resolved_by="none", reasoning=reasoning,
        )

    @staticmethod
    def _exclusion_resolution(intent: IntentDefinition, mapped: MappedParameters, confidence: float,
                              alternatives: list[Alternative], *, resolved_by: str = "embedding",
                              entities: dict[str, Any] | None = None,
                              filters: dict[str, Any] | None = None) -> Resolution:
        """The user excluded a value the intent can only include ("anywhere except Delhi")."""
        params = list(dict.fromkeys(name for name, _ in mapped.unsupported_exclusions))
        values = [str(v) for _, v in mapped.unsupported_exclusions]
        shown = " and ".join(values)
        param = params[0].replace("_", " ")
        question = (f"I can't exclude {shown} when I {intent.label}: the {param} filter can only include values, "
                    f"not exclude them. Should I search without a {param} filter, or is there a specific "
                    f"{param} you want?")
        return Resolution(
            status=Status.needs_clarification, intent=intent, action=intent.action, target=intent.target,
            domain=intent.domain, entities=entities if entities is not None else mapped.entities,
            filters=filters if filters is not None else mapped.filters, missing=mapped.missing,
            optional_missing=mapped.optional_missing, question=question, confidence=confidence,
            alternatives=alternatives, resolved_by=resolved_by,
            reasoning=(f"Matched '{intent.id}', but the request excludes {shown} ({', '.join(params)}) and "
                       f"'{intent.id}' has no exclusion filter for it, so it cannot run as asked."),
        )

    def _top(self, d: Decision, *, include_best: bool, exclude: str | None = None) -> list[Candidate]:
        out, seen = [], set()
        for c in d.ranked:
            if c.intent.id in seen or c.intent.id == exclude or (not include_best and c is d.best):
                continue
            seen.add(c.intent.id)
            out.append(c)
        limit = self.settings.max_alternatives + (1 if include_best else 0)
        return out[:limit]

    def _other(self, spans) -> dict[str, Any]:
        other = unmapped_entities(spans)
        return {"other": other} if other else {}

    # ------------------------------------------------------------------ follow-ups
    def _follow_up(self, text: str, spans, full: Decision, combined: Decision | None, context: ConversationContext,
                   ctx_intent: IntentDefinition) -> Resolution | None:
        if ctx_intent.out_of_scope or ctx_intent.always_clarify:
            return None
        if is_bare_refusal(text) and ctx_intent.side_effects not in ("none", "read"):
            # "actually, don't" after "Cancel my order 4421" withdraws the request.
            return self._negated_resolution(["do that"], ctx_intent, None, spans)
        cue = is_follow_up_phrase(text)
        words = len(text.split())
        same = full.accepted and full.best.intent.id == ctx_intent.id
        combined_agrees = (combined is not None and combined.best.intent.id == ctx_intent.id
                           and combined.outcome in (Outcome.accept, Outcome.low_confidence))
        # A follow-up phrase without an action of its own ("what about Sunday?", "only
        # remote ones") refines the previous request even if it weakly matches something
        # else on its own. A message with its own action verb is a new request.
        # The combined "previous + current" text is dominated by the previous message, so
        # it only informs confidence; it never decides on its own that this is a follow-up.
        # Matching the previous intent is not enough either: "Find Java jobs" after
        # "Find Python jobs in Noida" is a complete request of its own and starts a
        # fresh search, while "Find remote ones" or "search again in Delhi" refine it.
        if same:
            follow = not self._standalone_request(text, cue)
        elif cue and not has_own_action(text):
            follow = True
        elif full.accepted:
            follow = False
        elif has_own_action(text):
            follow = False
        else:
            follow = cue or words <= 4
        if not follow:
            return None
        # Negated actions and unsupported variants are judged on the message itself
        # by the regular path, which never turns them into an executable route.
        if self._negated(text, ctx_intent) or self.registry.unsupported_variant(ctx_intent.id, text):
            return None

        mapped = map_parameters(spans, ctx_intent)
        additive = is_additive(text)
        entities = merge_values(context.entities, mapped.entities, additive)
        filters = merge_values(context.filters, mapped.filters, additive)
        if "other" in mapped.entities:
            entities["other"] = mapped.entities["other"]
        filled = {k for k in (*entities, *filters) if k != "other"}
        missing, optional, question = missing_for(ctx_intent, filled)
        if combined_agrees:
            confidence = combined.confidence
        else:
            base = context.confidence if context.confidence is not None else 0.7
            confidence = round(base * self.settings.followup_confidence_factor, 4)
        if mapped.unsupported_exclusions:
            # "not in Noida" after "Find Python jobs in Noida" drops the excluded value and
            # cannot run as asked; "but not Delhi" is already implied by location=Noida.
            remaining = []
            for name, value in mapped.unsupported_exclusions:
                spec = ctx_intent.param(name)
                bucket = filters if spec is not None and spec.kind == "filter" else entities
                if bucket.get(name) == value:
                    bucket.pop(name)
                if spec is not None and not spec.multiple and bucket.get(name) is not None:
                    continue
                remaining.append((name, value))
            if remaining:
                mapped.unsupported_exclusions = remaining
                missing, optional, _ = missing_for(ctx_intent, {k for k in (*entities, *filters) if k != "other"})
                mapped.missing, mapped.optional_missing = missing, optional
                return self._exclusion_resolution(ctx_intent, mapped, confidence, [], resolved_by="context",
                                                  entities=entities, filters=filters)
        changed = sorted(k for k in (*mapped.entities, *mapped.filters) if k != "other")
        reasoning = (f"Interpreted as a follow-up to '{ctx_intent.id}' using the supplied context"
                     + (f"; updated: {', '.join(changed)}." if changed else "; no new parameters found."))
        return Resolution(
            status=Status.needs_clarification if missing else Status.success, intent=ctx_intent,
            action=ctx_intent.action, target=ctx_intent.target, domain=ctx_intent.domain, entities=entities,
            filters=filters, missing=missing, optional_missing=optional, question=question,
            confidence=confidence, resolved_by="context", reasoning=reasoning,
        )

    @staticmethod
    def _standalone_request(text: str, cue: bool) -> bool:
        """A complete new request: its own request verb and nothing tying it to the last turn."""
        if cue or has_anaphora(text) or _REFINES_PREVIOUS.search(text):
            return False
        return has_own_action(text) or bool(_LOOK_FOR.search(text))

    def _subject_only(self, word: str, full: Decision, spans) -> Resolution:
        """A lone noun ("Python", "Delhi") names a subject but no action."""
        return Resolution(
            status=Status.ambiguous, intent=None, entities=self._other(spans),
            question=f"What would you like me to do with \"{word}\"?",
            confidence=min(full.confidence, 0.3), alternatives=[_alt(c) for c in self._top(full, include_best=True)],
            resolved_by="none",
            reasoning=f"The message names a subject ('{word}') but no action, so it is not routed.",
        )

    def _context_missing(self, spans, full: Decision) -> Resolution:
        return Resolution(
            status=Status.needs_clarification, intent=None, entities=self._other(spans),
            question="This sounds like a follow-up, but I don't have the earlier request. What would you like me to do?",
            confidence=0.0, alternatives=[_alt(c) for c in self._top(full, include_best=True)],
            resolved_by="none",
            reasoning="Looks like a follow-up to an earlier message, but no conversation context was supplied.",
        )

    # ------------------------------------------------------------------ multi-intent
    def _solid(self, d: Decision) -> bool:
        return d.best.score >= self.t.segment_accept and d.outcome is not Outcome.unknown

    def _distinct_requests(self, first: Segment, second: Segment, intent: IntentDefinition,
                           ectx: ExtractionContext) -> bool:
        """Two clauses of the same intent that are separate requests, not one request
        told in two parts: the second does not refer back and gives a different value
        for a parameter the first one also sets.

        "Find Java jobs in Delhi and then search for Python jobs in Mumbai" -> two searches.
        "Find Python jobs in Noida. Show me remote ones" -> one search.
        """
        # Both clauses already classify as the intent on their own (weak splits only
        # happen before a request verb), so referring back is what makes the second
        # clause a refinement of the first.
        if second.has_anaphora:
            return False

        def values(seg: Segment) -> dict[str, Any]:
            m = map_parameters(self.pipeline.extract(seg.text, ectx), intent)
            return {k: v for k, v in (*m.entities.items(), *m.filters.items()) if k != "other"}

        a, b = values(first), values(second)
        return any(k in a and a[k] != v for k, v in b.items())

    def _multi_units(self, segments: list[Segment], decisions: list[Decision], full: Decision,
                     ectx: ExtractionContext) -> list[tuple[Segment, Decision]] | None:
        """Validate a clause split. Returns None when the split should be ignored."""
        for k, seg in enumerate(segments):
            if seg.boundary == "weak" and not (self._solid(decisions[k]) and self._solid(decisions[k - 1])):
                return None
        parallel = any(seg.boundary == "parallel" for seg in segments)
        if parallel and not (full.accepted and all(
                self._solid(d) and d.best.intent.id == full.best.intent.id for d in decisions)):
            # A parallel split only separates repeated requests for the message's own intent.
            return None

        units: list[list] = []  # [segment, decision, pending_context_text]
        pending: Segment | None = None
        for seg, d in zip(segments, decisions):
            if not self._solid(d) and not seg.starts_with_request:
                pending = seg if pending is None else Segment(f"{pending.text} {seg.text}", pending.start, seg.end,
                                                              pending.boundary)
                continue
            if pending is not None:
                seg = Segment(f"{pending.text} {seg.text}", pending.start, seg.end, pending.boundary, seg.sequential)
                pending = None
            units.append([seg, d])
        if pending is not None and units:
            last_seg, last_d = units[-1]
            units[-1] = [Segment(f"{last_seg.text} {pending.text}", last_seg.start, pending.end, last_seg.boundary,
                                 last_seg.sequential), last_d]

        merged: list[list] = []
        for seg, d in units:
            prev_d = merged[-1][1] if merged else None
            same_intent = (prev_d is not None and prev_d.accepted and d.accepted
                           and prev_d.best.intent.id == d.best.intent.id
                           and not self._distinct_requests(merged[-1][0], seg, d.best.intent, ectx))
            # "my app returns a 500 error, how do I fix it?" -> one troubleshooting request:
            # same action, refers back, and adds almost nothing of its own.
            continues = (prev_d is not None and seg.has_anaphora and not seg.sequential
                         and prev_d.best.intent.action == d.best.intent.action
                         and len(content_words(seg.text)) <= 2)
            if same_intent or continues:
                prev_seg = merged[-1][0]
                merged[-1][0] = Segment(f"{prev_seg.text} {seg.text}", prev_seg.start, seg.end, prev_seg.boundary,
                                        prev_seg.sequential)
            else:
                merged.append([seg, d])

        if parallel and len(merged) < len(segments):
            return None  # not separate requests after all: route the whole message as one
        non_chat = [u for u in merged if not (u[1].accepted and u[1].best.intent.action == "converse")]
        if non_chat:
            merged = non_chat
        return [(u[0], u[1]) for u in merged]

    def _multi(self, units: list[tuple[Segment, Decision]], ectx: ExtractionContext) -> Resolution:
        tasks: list[Task] = []
        for k, (seg, d) in enumerate(units):
            # Short clauses score lower than whole requests; a clause is judged against the
            # clause threshold (segment_accept), which the split was validated with.
            if d.outcome is Outcome.low_confidence and d.best.score >= self.t.segment_accept:
                d = dataclass_replace(d, outcome=Outcome.accept)
            r = self._resolve(d, seg.text, self.pipeline.extract(seg.text, ectx))
            depends = [k - 1] if k > 0 and (seg.has_anaphora or seg.sequential) else []
            tasks.append(Task(
                index=k, text=seg.text, status=r.status, intent=r.intent.id if r.intent else None,
                action=r.action, target=r.target, domain=r.domain, entities=r.entities, filters=r.filters,
                missing_fields=r.missing, optional_missing_fields=r.optional_missing, confidence=r.confidence,
                depends_on=depends, side_effects=r.intent.side_effects if r.intent else "none",
                requires_confirmation=bool(r.intent.requires_confirmation) if r.intent else False,
            ))
        res = Resolution(status=Status.multi_intent, intent=None, tasks=tasks,
                         confidence=min(t.confidence for t in tasks), resolved_by="embedding",
                         reasoning=f"Found {len(tasks)} separate requests: "
                                   + ", ".join(t.intent or t.status.value for t in tasks) + ".")
        unclear = [t for t in tasks if t.status is not Status.success]
        if unclear:
            res.question = f"Part {unclear[0].index + 1} (\"{unclear[0].text}\") needs more detail."
        return res

    # ------------------------------------------------------------------ result assembly
    def _build(self, r: Resolution, context, instruction, prepared, routing_text, corrections, scores, timings,
               start, include_debug) -> RouteResult:
        tasks = r.tasks
        intent = r.intent
        new_context = None
        if intent is not None and r.status in (Status.success, Status.needs_clarification):
            new_context = next_context(instruction, intent=intent.id, action=intent.action, target=intent.target,
                                       domain=intent.domain, entities=r.entities, filters=r.filters,
                                       confidence=r.confidence, previous=context,
                                       follow_up=r.resolved_by == "context")
        elif tasks:
            last = next((t for t in reversed(tasks) if t.intent), None)
            if last is not None:
                new_context = next_context(last.text, intent=last.intent, action=last.action, target=last.target,
                                           domain=last.domain, entities=last.entities, filters=last.filters,
                                           confidence=last.confidence)

        t0 = time.perf_counter()
        side_effects = intent.side_effects if intent else "none"
        requires_confirmation = bool(intent.requires_confirmation) if intent else False
        if tasks:
            requires_confirmation = any(t.requires_confirmation for t in tasks)
        result = RouteResult(
            version=SCHEMA_VERSION,
            status=r.status,
            intent=intent.id if intent else ("unknown" if r.status is Status.unknown else None),
            action=r.action,
            target=r.target,
            domain=r.domain,
            entities=r.entities,
            filters=r.filters,
            missing_fields=r.missing,
            optional_missing_fields=r.optional_missing,
            needs_clarification=r.needs_clarification and r.status is not Status.multi_intent
            or any(t.status is not Status.success for t in tasks),
            clarification_question=r.question,
            confidence=float(np.clip(r.confidence, 0.0, 1.0)),
            response_time=round((time.perf_counter() - start) * 1000, 2),
            alternatives=r.alternatives,
            tasks=tasks,
            is_follow_up=r.resolved_by == "context",
            context_used=r.resolved_by == "context",
            resolved_by=r.resolved_by,
            side_effects=side_effects,
            requires_confirmation=requires_confirmation,
            reasoning_summary=r.reasoning[:400],
            context=new_context,
        )
        timings["validate_ms"] = (time.perf_counter() - t0) * 1000
        timings["total_ms"] = (time.perf_counter() - start) * 1000
        result.response_time = round(timings["total_ms"], 2)
        if include_debug:
            top = rank(scores.final[0], self.registry, limit=5)
            result.debug = DebugInfo(
                normalized_text=prepared.normalized,
                routing_text=routing_text,
                top_scores=[_alt(c) for c in top],
                action_scores={a: round(float(s), 4) for a, s in zip(scores.action_names, scores.actions[0])},
                typo_corrections=[list(c) for c in corrections],
                thresholds=self.t.model_dump(),
                timings_ms={k: round(v, 3) for k, v in timings.items()},
                fallback_used=r.fallback_used,
                model=self.embedder.model_name,
            )
        return result

    def _invalid(self, normalized: str, include_debug: bool, start: float | None = None) -> RouteResult:
        total_ms = round((time.perf_counter() - start) * 1000, 2) if start is not None else 0.0
        result = RouteResult(
            status=Status.invalid_input, intent=None, confidence=0.0, needs_clarification=True,
            clarification_question="The message is empty. What would you like me to do?",
            resolved_by="none", reasoning_summary="The message is empty or contains no letters or digits.",
            response_time=total_ms,
        )
        if include_debug:
            result.debug = DebugInfo(normalized_text=normalized, routing_text="", model=self.embedder.model_name,
                                     thresholds=self.t.model_dump())
        return result
