"""Runs extractors, resolves overlaps, and maps spans onto intent parameters."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from intent_router.extractors.base import (
    GAZETTEER_PRIORITY,
    PRIORITY,
    SOFT_TYPES,
    ExtractionContext,
    Extractor,
    Span,
)
from intent_router.extractors.entities import (
    GazetteerExtractor,
    LocationPatternExtractor,
    PhraseExtractor,
    add_role_overlays,
)
from intent_router.extractors.negation import mark_exclusions
from intent_router.extractors.rules import RuleExtractor
from intent_router.router.registry import IntentDefinition

# Atomic types reported under entities["other"] when the chosen intent has no
# parameter for them. Domain-specific gazetteer types are not reported this way.
REPORTED_WHEN_UNMAPPED = frozenset({
    "email", "url", "phone", "file", "identifier", "error_message", "money", "money_constraint", "percentage",
    "quantity", "date", "date_range", "time", "duration", "location", "people_count", "code", "quoted_text",
})


def resolve_overlaps(spans: list[Span]) -> list[Span]:
    """Among overlapping atomic spans keep the highest-priority, then longest one."""
    tech_positions = {(s.start, s.end) for s in spans if s.type == "technology" and s.length > 0}
    atomic = [
        s for s in spans
        if s.type not in SOFT_TYPES and not (s.type == "file" and (s.start, s.end) in tech_positions)
    ]
    soft = [s for s in spans if s.type in SOFT_TYPES]
    atomic.sort(key=lambda s: (-PRIORITY.get(s.type, GAZETTEER_PRIORITY), -s.length, s.start))
    kept: list[Span] = []
    for s in atomic:
        if s.length and any(s.overlaps(k) for k in kept if k.length):
            continue
        if any(k.type == s.type and k.start == s.start and k.end == s.end for k in kept):
            continue
        kept.append(s)
    seen: set[tuple[str, str]] = set()
    unique_soft: list[Span] = []
    for s in soft:
        # A quote inside an error message, URL or identifier is part of that value.
        if s.type == "quoted_text" and any(
            k.type in ("error_message", "url", "identifier", "file") and k.start <= s.start and s.end <= k.end
            for k in kept
        ):
            continue
        key = (s.type, json.dumps(s.value, sort_keys=True, default=str))
        if key not in seen:
            seen.add(key)
            unique_soft.append(s)
    return sorted(kept + unique_soft, key=lambda s: (s.start, -s.length))


class ExtractionPipeline:
    def __init__(self, extractors: list[Extractor]):
        self.extractors = extractors

    def extract(self, text: str, ctx: ExtractionContext) -> list[Span]:
        spans: list[Span] = []
        for extractor in self.extractors:
            spans.extend(extractor.extract(text, ctx))
        spans = resolve_overlaps(spans)
        return mark_exclusions(text, add_role_overlays(text, spans))


@dataclass
class MappedParameters:
    entities: dict[str, Any] = field(default_factory=dict)
    filters: dict[str, Any] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    optional_missing: list[str] = field(default_factory=list)
    clarification: str | None = None
    # Values the user excluded for a parameter that has no exclusion filter:
    # [(parameter name, value), ...]. The request cannot be run as asked.
    unsupported_exclusions: list[tuple[str, Any]] = field(default_factory=list)


def _merge_dicts(values: list[dict]) -> tuple[dict, list[int]]:
    merged = dict(values[0])
    texts = [merged.get("text")] if merged.get("text") else []
    consumed = [0]
    for idx, v in enumerate(values[1:], start=1):
        conflict = any(merged.get(k) is not None and v.get(k) is not None and merged[k] != v[k]
                       for k in v if k not in ("text", "approximate"))
        if not conflict:
            new_min = v.get("min") if merged.get("min") is None else merged.get("min")
            new_max = v.get("max") if merged.get("max") is None else merged.get("max")
            if new_min is not None and new_max is not None and new_min > new_max:
                conflict = True
        if conflict:
            continue
        consumed.append(idx)
        for k, val in v.items():
            if k == "text":
                if val and val not in texts:
                    texts.append(val)
            elif merged.get(k) is None and val is not None:
                merged[k] = val
    if texts:
        merged["text"] = " / ".join(texts)
    return merged, consumed


def _dedupe(values: list[Any]) -> list[Any]:
    out, seen = [], set()
    for v in values:
        for item in (v if isinstance(v, list) else [v]):
            key = json.dumps(item, sort_keys=True, default=str)
            if key not in seen:
                seen.add(key)
                out.append(item)
    return out


def map_parameters(spans: list[Span], intent: IntentDefinition) -> MappedParameters:
    """Fill the intent's declared parameters from extracted spans. Never invents values."""
    result = MappedParameters()
    used_positions: set[tuple[int, int]] = set()
    used_ids: set[int] = set()

    def is_used(s: Span) -> bool:
        return id(s) in used_ids or (s.length > 0 and (s.start, s.end) in used_positions)

    def mark(s: Span) -> None:
        used_ids.add(id(s))
        if s.length > 0:
            used_positions.add((s.start, s.end))

    filled: set[str] = set()
    # An exclusion parameter ("excludes": "location") collects the excluded values of
    # the parameter it negates; included values never go there and vice versa.
    exclusion_params = {p.excludes: p for p in intent.parameters if p.excludes}
    for p in intent.parameters:
        chosen: list[Span] = []
        if p.excludes:
            target = intent.param(p.excludes)
            chosen = [s for s in spans if s.excluded and target is not None and s.type in target.types
                      and not is_used(s)]
            chosen.sort(key=lambda s: s.start)
        elif p.multiple:
            for t in p.types:
                chosen += [s for s in spans if s.type == t and not s.excluded and not is_used(s)]
            chosen.sort(key=lambda s: s.start)
        else:
            for t in p.types:
                candidates = [s for s in spans if s.type == t and not s.excluded and not is_used(s)]
                if candidates:
                    chosen = candidates
                    break
        if not chosen:
            continue
        if p.excludes:
            value = _dedupe([s.value for s in chosen])
            for s in chosen:
                mark(s)
            (result.filters if p.kind == "filter" else result.entities)[p.name] = value
            filled.add(p.name)
            continue
        if p.multiple:
            value: Any = _dedupe([s.value for s in chosen])
        elif all(isinstance(s.value, dict) for s in chosen) and len(chosen) > 1:
            value, used_indices = _merge_dicts([s.value for s in chosen])
            chosen = [chosen[i] for i in used_indices]
        else:
            value = chosen[0].value
            chosen = [chosen[0]]
        for s in chosen:
            mark(s)
        (result.filters if p.kind == "filter" else result.entities)[p.name] = value
        filled.add(p.name)

    # Excluded values for a parameter the intent can only include: dropping the
    # exclusion would silently run a broader request, so report it instead.
    excluded_records: list[dict[str, Any]] = []
    for s in spans:
        if not s.excluded or is_used(s):
            continue
        p = next((p for p in intent.parameters if not p.excludes and s.type in p.types), None)
        if p is not None and p.name not in exclusion_params:
            mark(s)
            if not p.multiple and p.name in filled:
                # "jobs in Noida, not Delhi": one included value already rules the other out.
                excluded_records.append({"type": s.type, "value": s.value, "excluded": True})
            elif all(v != s.value for name, v in result.unsupported_exclusions if name == p.name):
                result.unsupported_exclusions.append((p.name, s.value))
                excluded_records.append({"type": s.type, "value": s.value, "excluded": True})

    result.missing, result.optional_missing, result.clarification = missing_for(intent, filled)
    other = excluded_records + unmapped_entities([s for s in spans if not is_used(s)])
    if other:
        result.entities["other"] = other
    return result


def missing_for(intent: IntentDefinition, filled: set[str]) -> tuple[list[str], list[str], str | None]:
    """Required-but-absent parameters, recommended-but-absent ones, and a question.

    A required_any group is reported as "a|b" meaning "at least one of a, b".
    """
    missing: list[str] = []
    optional: list[str] = []
    question: str | None = None
    for p in intent.parameters:
        if p.required and p.name not in filled:
            missing.append(p.name)
            question = question or p.question or f"Could you tell me the {p.name.replace('_', ' ')}?"
    for group in intent.required_any:
        if not any(name in filled for name in group):
            missing.append("|".join(group))
            question = question or intent.required_any_question or (
                "Could you tell me the " + " or ".join(g.replace("_", " ") for g in group) + "?")
    for p in intent.parameters:
        if p.recommended and not p.required and p.name not in filled:
            optional.append(p.name)
    return missing, optional, question


def unmapped_entities(spans: list[Span]) -> list[dict[str, Any]]:
    return [{"type": s.type, "value": s.value, **({"excluded": True} if s.excluded else {})}
            for s in spans if s.type in REPORTED_WHEN_UNMAPPED]


def load_gazetteers(path: Path, extra: dict[str, dict[str, list[str]]] | None = None) -> dict[str, dict[str, list[str]]]:
    merged: dict[str, dict[str, list[str]]] = {}
    if Path(path).exists():
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        for k, v in raw.items():
            if not k.startswith("_"):
                merged[k] = {c: list(a) for c, a in v.items()}
    for gname, entries in (extra or {}).items():
        bucket = merged.setdefault(gname, {})
        for canonical, aliases in entries.items():
            bucket.setdefault(canonical, [])
            bucket[canonical].extend(a for a in aliases if a not in bucket[canonical])
    return merged


def build_pipeline(gazetteers: dict[str, dict[str, list[str]]], known_words: set[str] | None = None) -> ExtractionPipeline:
    # Words that must never be read as place or person names.
    non_place_words = set()
    for gname in ("technology", "natural_language", "brand", "product", "document_type"):
        for canonical, aliases in gazetteers.get(gname, {}).items():
            non_place_words.update(a.lower() for a in aliases)
            non_place_words.add(canonical.lower())
    place_and_tech = set(non_place_words)
    for canonical, aliases in gazetteers.get("location", {}).items():
        place_and_tech.update(a.lower() for a in aliases)
    return ExtractionPipeline([
        RuleExtractor(),
        GazetteerExtractor(gazetteers, known_words=known_words),
        LocationPatternExtractor(non_place_words),
        PhraseExtractor(excluded_person_words=place_and_tech),
    ])
