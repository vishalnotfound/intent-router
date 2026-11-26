"""Splits a message into candidate clauses for multi-intent detection.

Splitting is a syntactic step: it only proposes clause boundaries. Whether the
clauses really are separate requests is decided by the engine, which classifies
each clause semantically and keeps the split only when the clauses hold up on
their own.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

I = re.IGNORECASE

# Words that typically start a new request ("... and compare their salaries").
VERB_CUES = frozenset("""
compare find search show list get give tell summarize summarise translate write draft create make book schedule
remind send explain calculate convert check track cancel plan fix debug sort filter rank email save add remove
delete look help suggest recommend what how why which who where when can could would please set order buy rewrite
extract generate build open pull fetch analyze analyse estimate review proofread reserve notify share download
upload clean refactor optimize optimise describe also then return exchange
""".split())

QUESTION_WORDS = frozenset({"what", "how", "which", "who", "where", "when", "why"})

_STRONG = re.compile(
    r"(?:(?<=[.!?])\s+(?=[A-Za-z])|\s*;\s*|,?\s+(?:and\s+then|after\s+that|afterwards|and\s+also)\s+|,\s+then\s+|\s+then\s+(?=\w+))",
    I,
)
_WEAK = re.compile(r"(?:,\s*and\s+|\s+and\s+|,\s+|\s+plus\s+)", I)
_SEQUENCE = re.compile(r"(?:then|after\s+that|afterwards)", I)
_ANAPHORA = re.compile(
    r"\b(?:their|them|those|these|they|it|its|there|the results?|the ones|that list|the above|each of them|both|"
    r"the same)\b", I
)
_LEADING_FILLER = re.compile(r"^(?:also|then|and|please|plus)\s+", I)


@dataclass
class Segment:
    text: str
    start: int
    end: int
    boundary: str  # "start" | "strong" | "weak"
    sequential: bool = False  # introduced by "then" / "after that"

    @property
    def starts_with_request(self) -> bool:
        words = _LEADING_FILLER.sub("", self.text).split()
        return bool(words) and words[0].lower().strip(",.!?") in VERB_CUES

    @property
    def has_anaphora(self) -> bool:
        return bool(_ANAPHORA.search(self.text))


def _first_word(text: str) -> str:
    words = _LEADING_FILLER.sub("", text.strip()).split()
    return words[0].lower().strip(",.!?") if words else ""


def split_clauses(text: str, max_segments: int = 4) -> list[Segment]:
    """Return clauses; a single-element list means no plausible split."""
    cuts: list[tuple[int, int, str, bool]] = []
    for m in _STRONG.finditer(text):
        cuts.append((m.start(), m.end(), "strong", bool(_SEQUENCE.search(m.group(0)))))
    for m in _WEAK.finditer(text):
        if any(a <= m.start() < b or a < m.end() <= b for a, b, _, _ in cuts):
            continue
        first = _first_word(text[m.end():])
        # ", which should I get?" is a tag on the same request; only "and what/how ..." splits.
        if first in QUESTION_WORDS and "and" not in m.group(0).lower():
            continue
        if first in VERB_CUES:
            cuts.append((m.start(), m.end(), "weak", False))
    cuts.sort()

    segments: list[Segment] = []
    pos, boundary, sequential = 0, "start", False
    for start, end, kind, seq in cuts:
        piece = text[pos:start].strip(" ,;")
        if piece:
            segments.append(Segment(piece, pos, start, boundary, sequential))
        pos, boundary, sequential = end, kind, seq
    tail = text[pos:].strip(" ,;")
    if tail:
        segments.append(Segment(tail, pos, len(text), boundary, sequential))

    # Fold fragments shorter than two words into their neighbour.
    merged: list[Segment] = []
    for seg in segments:
        if merged and len(seg.text.split()) < 2:
            prev = merged[-1]
            merged[-1] = Segment(text[prev.start:seg.end].strip(" ,;"), prev.start, seg.end, prev.boundary, prev.sequential)
        elif merged and len(merged[-1].text.split()) < 2:
            prev = merged[-1]
            merged[-1] = Segment(text[prev.start:seg.end].strip(" ,;"), prev.start, seg.end, prev.boundary, prev.sequential)
        else:
            merged.append(seg)
    if len(merged) > max_segments:
        return [Segment(text, 0, len(text), "start")]
    for seg in merged:  # "Also, what's the capital of Peru?" -> "what's the capital of Peru?"
        cleaned = _CLAUSE_FILLER.sub("", seg.text).strip()
        if cleaned:
            seg.text = cleaned
    return merged or [Segment(text, 0, len(text), "start")]


_CLAUSE_FILLER = re.compile(r"^(?:(?:also|then|and|plus|please|so|additionally)\b[\s,]*)+", I)
_STOPWORDS = frozenset("""
a an the my me i you your it this that these those for to of in on at with and or please pls some any is are be
do can could would will just now ok okay about some them they their its there
""".split())
_WORDS = re.compile(r"[A-Za-z][A-Za-z'+#.-]*")


def content_words(text: str) -> list[str]:
    return [w for w in _WORDS.findall(text) if w.lower() not in _STOPWORDS]
