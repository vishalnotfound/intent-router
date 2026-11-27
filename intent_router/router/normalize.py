"""Safe text normalization and request preprocessing.

The original user message is always preserved separately; these functions only
produce derived views of it.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass, field

_QUOTES = str.maketrans({"“": '"', "”": '"', "„": '"', "‘": "'", "’": "'",
                         "«": '"', "»": '"', "–": "-", "—": "-", " ": " "})
_CONTROL = re.compile(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f-\u009f​-‏﻿]")
_SPACES = re.compile(r"[ \t\f\v]+")
_CODE_FENCE = re.compile(r"```[\w+-]*\n?(.*?)```", re.DOTALL)
_INLINE_CODE = re.compile(r"`([^`\n]{1,300})`")
_WORD = re.compile(r"[A-Za-z][A-Za-z']*")
_REFERS_TO_CONTENT = re.compile(
    r"\b(this|these|following|below|text|paragraph|article|passage|email|message|document|notes?|"
    r"review|content|code|snippet|it|that|sentence|thread|transcript|essay|letter|post)\b",
    re.IGNORECASE,
)
# Routing only needs the instruction; very long text is truncated for the embedding.
MAX_ROUTING_CHARS = 600


def normalize_text(text: str) -> str:
    """NFKC-normalize, drop control characters, unify quotes/dashes, collapse spaces."""
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_QUOTES)
    text = _CONTROL.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _SPACES.sub(" ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def has_linguistic_content(text: str) -> bool:
    """True when the text contains at least one letter or digit."""
    return any(ch.isalnum() for ch in text)


@dataclass
class PreparedText:
    normalized: str  # full normalized message
    instruction: str  # the part that expresses the request (used for routing)
    content: str | None = None  # pasted payload after "summarize this: ..."
    code_blocks: list[str] = field(default_factory=list)


def prepare(text: str) -> PreparedText:
    """Separate the instruction from pasted content and code.

    Long pasted content would otherwise dominate the sentence embedding: in
    "Summarize this: <article about climate>" the embedding would be about climate.
    """
    normalized = normalize_text(text)
    code_blocks = [m.group(1).strip() for m in _CODE_FENCE.finditer(normalized)]
    instruction = _CODE_FENCE.sub(" ", normalized)

    content = None
    # "instruction: payload" or "instruction\npayload" where the instruction refers
    # to the payload ("summarize this:", "translate the following:").
    m = re.match(r"^(.{3,200}?)(?::\s+|\n+)(.+)$", instruction, re.DOTALL)
    if m:
        head, tail = m.group(1).strip(), m.group(2).strip()
        if len(head.split()) <= 25 and len(tail.split()) >= 8 and _REFERS_TO_CONTENT.search(head):
            instruction, content = head, tail

    instruction = _SPACES.sub(" ", instruction.replace("\n", " ")).strip()
    if len(instruction) > MAX_ROUTING_CHARS:
        instruction = instruction[:MAX_ROUTING_CHARS].rsplit(" ", 1)[0]
    return PreparedText(normalized=normalized, instruction=instruction or normalized[:MAX_ROUTING_CHARS],
                        content=content, code_blocks=code_blocks)


# Delexicalization: replace specific values with generic words before embedding.
# Small sentence encoders weight proper nouns heavily ("Find Python jobs in Noida"
# vs "Find Java jobs in Pune" scores only ~0.3 with MiniLM), so both the request and
# the registry examples are embedded with these placeholders instead.
MASK_PLACEHOLDERS = {
    "location": "the city",
    "technology": "programming",
    "person": "someone",
    "email": "an email address",
    "url": "a link",
    "phone": "a phone number",
    "file": "a file",
    # Quoted text is payload ("translate 'see you tomorrow'"), not part of the request.
    "quoted_text": "this text",
    # Prices: "between 5000 and 10000" should not look like a comparison.
    "money": "money",
    "money_constraint": "within budget",
    "rating": "well rated",
}


def mask_entities(text: str, spans) -> str:
    """Replace maskable spans (non-overlapping, right to left) with generic words."""
    candidates = sorted((s for s in spans if s.length > 0 and (s.type in MASK_PLACEHOLDERS or s.type == "identifier")
                         and s.end <= len(text)), key=lambda s: (s.start, -s.length))
    chosen = []
    last_end = -1
    for s in candidates:
        if s.start >= last_end:
            chosen.append(s)
            last_end = s.end
    out = text
    for s in reversed(chosen):
        if s.type == "identifier":
            kind = s.value.get("kind") if isinstance(s.value, dict) else None
            placeholder = f"{kind} number" if kind else "the number"
        else:
            placeholder = MASK_PLACEHOLDERS[s.type]
        out = out[:s.start] + placeholder + out[s.end:]
    return out


_POLITE_LEAD = re.compile(
    r"^(?:(?:please|pls|kindly|hey|hi|hello|ok|okay|so|um|uh|well)\b[\s,!.]*|"
    r"(?:can|could|would|will)\s+(?:you|u)\s+(?:please\s+|pls\s+|kindly\s+)?|"
    r"i\s+(?:want|need|would like|'d like|wanna)(?:\s+you)?\s+to\s+|i'?d\s+like\s+to\s+|"
    r"help\s+me\s+(?:to\s+)?(?=\w+\s)|let'?s\s+)+",
    re.IGNORECASE,
)


def request_head(text: str, n_words: int) -> str:
    """First words of the request after politeness ("could you please ..."), where the
    verb and object type usually are. Used as a second embedding view."""
    stripped = _POLITE_LEAD.sub("", text.strip()) or text.strip()
    return " ".join(stripped.split()[:n_words])


class TypoCorrector:
    """Corrects likely misspellings in the routing text only (never in extraction).

    A token is treated as a misspelling when it is not a whole word in the
    embedding model's own tokenizer vocabulary *and* it is close to a word from
    the registry/gazetteer vocabulary. Unknown proper nouns are left alone unless
    they closely resemble a known domain word.
    """

    MAX_EDITS = 2  # candidate generation radius (deletions on both sides, SymSpell-style)

    def __init__(self, domain_vocab: set[str], known_words: set[str] | None = None,
                 cutoff: float = 0.75, min_len: int = 4):
        self.domain_vocab = sorted(w for w in domain_vocab if len(w) >= 3)
        self.known = set(known_words or ()) | set(self.domain_vocab)
        self.cutoff = cutoff
        self.min_len = min_len
        # Static deletion index over the registry vocabulary, built once. Lookups touch only
        # words within MAX_EDITS of the token instead of scanning the whole vocabulary
        # (~3.5 ms per unknown word), so no per-request cache is needed and nothing from
        # user messages is kept in memory.
        buckets: dict[str, list[str]] = {}
        for word in self.domain_vocab:
            for variant in _deletions(word, self.MAX_EDITS):
                buckets.setdefault(variant, []).append(word)
        self._index = {k: tuple(v) for k, v in buckets.items()}

    def _fix(self, token: str) -> str | None:
        low = token.lower()
        if len(low) < self.min_len or low in self.known or not low.isalpha():
            return None
        candidates = {w for v in _deletions(low, self.MAX_EDITS) for w in self._index.get(v, ())}
        if not candidates:
            return None
        # Same acceptance rule as before: best difflib ratio >= cutoff.
        match = difflib.get_close_matches(low, sorted(candidates), n=1, cutoff=self.cutoff)
        return match[0] if match else None

    def correct(self, text: str) -> tuple[str, list[tuple[str, str]]]:
        changes: list[tuple[str, str]] = []

        def repl(m: re.Match) -> str:
            fixed = self._fix(m.group(0))
            if fixed and fixed != m.group(0).lower():
                changes.append((m.group(0), fixed))
                return fixed
            return m.group(0)

        return _WORD.sub(repl, text), changes


def _deletions(word: str, max_edits: int) -> set[str]:
    """The word plus every string obtained by deleting up to `max_edits` characters."""
    out = {word}
    frontier = {word}
    for _ in range(max_edits):
        frontier = {w[:i] + w[i + 1:] for w in frontier for i in range(len(w))}
        out |= frontier
    return out


def build_domain_vocab(texts: list[str]) -> set[str]:
    vocab: set[str] = set()
    for t in texts:
        vocab.update(w.lower() for w in _WORD.findall(t))
    return vocab
