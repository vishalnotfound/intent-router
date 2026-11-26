"""Gazetteer- and phrase-based extractors.

Gazetteers come from data/gazetteers.json plus domain files, so new entity
types (e.g. "travel_mode") need no code. Phrase extractors keep the user's
wording verbatim when the exact meaning is uncertain ("that I can upgrade later").
"""

from __future__ import annotations

import calendar
import difflib
import re

from app.extractors.base import ExtractionContext, Span

I = re.IGNORECASE
FUZZY_TYPES = ("technology", "location", "job_role", "product")
_TOKEN = re.compile(r"[A-Za-z][A-Za-z+#.]*[A-Za-z+#]|[A-Za-z]")


class GazetteerExtractor:
    """Case-insensitive whole-word matching of known values, plus conservative
    fuzzy matching for likely misspellings ("pyhton" -> Python)."""

    name = "gazetteer"

    def __init__(self, gazetteers: dict[str, dict[str, list[str]]], known_words: set[str] | None = None,
                 fuzzy_types: tuple[str, ...] = FUZZY_TYPES, fuzzy_cutoff: float = 0.8):
        self.known_words = {w.lower() for w in (known_words or set())}
        self.fuzzy_cutoff = fuzzy_cutoff
        self._patterns: list[tuple[str, re.Pattern, dict[str, str]]] = []
        self._fuzzy_index: dict[str, tuple[str, str]] = {}
        for etype, entries in gazetteers.items():
            if etype.startswith("_"):
                continue
            alias_map: dict[str, str] = {}
            for canonical, aliases in entries.items():
                for alias in aliases:
                    alias_map.setdefault(alias.lower(), canonical)
            if not alias_map:
                continue
            parts = []
            for alias in sorted(alias_map, key=len, reverse=True):
                plural = "(?:s|es)?" if alias[-1].isalpha() and len(alias) >= 3 else ""
                parts.append(re.escape(alias).replace(r"\ ", r"[\s-]+") + plural)
            rx = re.compile(r"(?<![A-Za-z0-9_])(?:" + "|".join(parts) + r")(?![A-Za-z0-9_+#])", I)
            self._patterns.append((etype, rx, alias_map))
            if etype in fuzzy_types:
                for alias, canonical in alias_map.items():
                    if alias.isalpha() and len(alias) >= 5:
                        self._fuzzy_index.setdefault(alias, (etype, canonical))
            self.known_words.update(a for a in alias_map if a.isalpha())
        self._fuzzy_vocab = sorted(self._fuzzy_index)

    @staticmethod
    def _lookup(alias_map: dict[str, str], matched: str) -> str | None:
        key = re.sub(r"[\s-]+", " ", matched.lower())
        for candidate in (key, key[:-2] if key.endswith("es") else None, key[:-1] if key.endswith("s") else None):
            if candidate and candidate in alias_map:
                return alias_map[candidate]
            if candidate:
                hyphen = candidate.replace(" ", "-")
                if hyphen in alias_map:
                    return alias_map[hyphen]
        return None

    def extract(self, text: str, ctx: ExtractionContext) -> list[Span]:
        spans: list[Span] = []
        for etype, rx, alias_map in self._patterns:
            for m in rx.finditer(text):
                canonical = self._lookup(alias_map, m.group(0))
                if canonical is not None:
                    spans.append(Span(etype, canonical, m.group(0), m.start(), m.end(), "gazetteer"))
        if self._fuzzy_vocab:
            covered = [(s.start, s.end) for s in spans]
            for m in _TOKEN.finditer(text):
                token = m.group(0).lower()
                if (len(token) < 5 or not token.isalpha() or token in self.known_words
                        or any(a < m.end() and m.start() < b for a, b in covered)):
                    continue
                match = difflib.get_close_matches(token, self._fuzzy_vocab, n=1, cutoff=self.fuzzy_cutoff)
                if match:
                    etype, canonical = self._fuzzy_index[match[0]]
                    if etype == "location" and sorted(token) != sorted(match[0]):
                        ratio = difflib.SequenceMatcher(None, token, match[0]).ratio()
                        if ratio < 0.85:
                            continue
                    spans.append(Span(etype, canonical, m.group(0), m.start(), m.end(), "fuzzy"))
        return spans


# --------------------------------------------------------------------------- locations
_NOT_PLACES = {
    "i", "the", "my", "our", "your", "a", "an", "this", "that", "google", "microsoft", "amazon", "excel", "word",
    "sheets", "docs", "slack", "teams", "zoom", "netflix", "youtube", "instagram", "linkedin", "facebook",
    "chrome", "windows", "mac", "iphone", "android", "pdf", "english", "total", "summary", "detail", "details",
}
_NOT_PLACES.update(m.lower() for m in calendar.month_name if m)
_NOT_PLACES.update(d.lower() for d in calendar.day_name)
_LOC_PATTERN = re.compile(
    r"\b(?:in|near|around|based in|located in|close to|nearby)\s+(?P<loc>[A-Z][a-z]{2,}(?:[ -][A-Z][a-z]{2,}){0,2})"
)


class LocationPatternExtractor:
    """Capitalized place names after location prepositions ("jobs in Zirakpur")."""

    name = "location_pattern"

    def __init__(self, excluded_words: set[str]):
        self.excluded = {w.lower() for w in excluded_words} | _NOT_PLACES

    def extract(self, text: str, ctx: ExtractionContext) -> list[Span]:
        spans = []
        for m in _LOC_PATTERN.finditer(text):
            loc = m.group("loc")
            if any(w.lower() in self.excluded for w in re.split(r"[ -]", loc)):
                continue
            spans.append(Span("location", loc, loc, m.start("loc"), m.end("loc"), "pattern"))
        return spans


def add_role_overlays(text: str, spans: list[Span]) -> list[Span]:
    """Origin/destination for locations, source/target for languages."""
    extra: list[Span] = []
    lang_starts = {sp.start for sp in spans if sp.type == "natural_language"}
    for s in spans:
        before = text[max(0, s.start - 25):s.start].lower()
        if s.type == "location":
            if re.search(r"\bfrom\s+$", before):
                extra.append(Span("origin", s.value, s.text, s.start, s.end, "overlay"))
            elif re.search(r"\b(?:to|towards|visit|visiting)\s+$", before):
                extra.append(Span("destination", s.value, s.text, s.start, s.end, "overlay"))
        elif s.type == "natural_language":
            to_lang = re.match(r"\s+(?:to|into)\s+", text[s.end:], I)
            followed_by_lang = bool(to_lang and (s.end + to_lang.end()) in lang_starts)
            if re.search(r"\bfrom\s+$", before) or followed_by_lang:
                extra.append(Span("source_language", s.value, s.text, s.start, s.end, "overlay"))
            elif re.search(r"\b(?:to|into|in)\s+$", before):
                extra.append(Span("target_language", s.value, s.text, s.start, s.end, "overlay"))
    return spans + extra


# --------------------------------------------------------------------------- comparisons
_CRITERIA = (r"prices?|costs?|salary|salaries|pay|packages?|features|specs|specifications|performance|battery life|"
             r"battery|cameras?|quality|reviews|ratings|benefits|speed|size|weight|display|design|durability|mileage|"
             r"fees|value|safety|reliability|security|support|warranty|storage|ram|processor|growth|workload|"
             r"syntax|popularity|difficulty|career prospects")
_END = r"(?=\s+(?:for|in terms of|based on|on the basis of|regarding|which|when|if)\b|[?.!,;]|$)"
_COMPARE_PATTERNS = [
    re.compile(r"\bcompar(?:e|ing)\s+(?:the\s+)?(?P<a>.+?)\s+(?:and|with|to|vs\.?|versus|against)\s+(?:the\s+)?(?P<b>.+?)" + _END, I),
    re.compile(r"\bdifferences?\s+between\s+(?:the\s+|a\s+|an\s+)?(?P<a>.+?)\s+and\s+(?:the\s+|a\s+|an\s+)?(?P<b>.+?)" + _END, I),
    re.compile(r"\b(?:which|what)(?:\s+one)?\s+(?:is|would be)\s+(?:better|best|cheaper|faster|easier|healthier|preferable)"
               r"(?:\s+for\s+[^,?]+?)?[,:]?\s+(?:the\s+)?(?P<a>[^,?.!]{1,40}?)\s+or\s+(?:the\s+)?(?P<b>[^,?.!]{1,40}?)(?=[?.!,;]|$)", I),
    re.compile(r"\bshould\s+i\s+(?:buy|get|choose|pick|go\s+for|learn|use|take)\s+(?:the\s+)?(?P<a>[^,?.!]{1,40}?)\s+or\s+"
               r"(?:the\s+)?(?P<b>[^,?.!]{1,40}?)(?=[?.!,;]|$)", I),
    re.compile(r"\bhow\s+(?:does|do|is|are)\s+(?:the\s+)?(?P<a>.+?)\s+compare\s+(?:to|with|against)\s+(?:the\s+)?(?P<b>.+?)"
               r"(?=[?.!,;]|$)", I),
    re.compile(r"\bpros\s+and\s+cons\s+of\s+(?P<a>.+?)\s+(?:vs\.?|versus|and|or)\s+(?P<b>.+?)(?=[?.!,;]|$)", I),
    re.compile(r"^(?:(?:is|are)\s+)?(?:the\s+)?(?P<a>[^,?.!]{1,40}?)\s+or\s+(?:the\s+)?(?P<b>[^,?.!]{1,40}?)[,?]?\s+"
               r"which\s+(?:one\s+)?(?:is|should|would)", I),
    re.compile(r"(?:^|[,:;]\s*|\b(?:is|are|buy|get|choose|pick|use|learn|prefer)\s+)(?:the\s+)?"
               r"(?P<a>[A-Za-z0-9][^,?.!]{0,40}?)\s+(?:vs\.?|versus)\s+(?:the\s+)?(?P<b>[^,?.!]{1,40}?)"
               r"(?=\s+(?:for|in terms of|which|comparison|review)\b|[?.!,;]|$)", I),
]
_CRITERIA_PATTERNS = [
    re.compile(r"\bcompar(?:e|ing)\s+(?:the\s+|their\s+|its\s+|his\s+|her\s+)(?P<c>" + _CRITERIA + r")\b", I),
    re.compile(r"\b(?:in terms of|based on|on the basis of|with respect to)\s+(?P<c>[^,?.!]{2,40}?)(?=[?.!,;]|$)", I),
    re.compile(r"\b(?P<c>" + _CRITERIA + r")\s+comparison\b", I),
]
_LEAD_FILLER = re.compile(r"^(?:should i (?:buy|get|choose)|is|are|the|a|an|compare|comparing)\s+", I)
_TRAIL_FILLER = re.compile(r"\s+(?:comparison|review|reviews|specs|for me|please|now)$", I)


def _clean_item(item: str) -> str:
    item = item.strip(" '\"")
    for _ in range(2):
        item = _LEAD_FILLER.sub("", item)
        item = _TRAIL_FILLER.sub("", item)
    return item.strip(" '\"")


def extract_comparisons(text: str) -> list[Span]:
    spans: list[Span] = []
    for rx in _COMPARE_PATTERNS:
        m = rx.search(text)
        if not m:
            continue
        a, b = _clean_item(m.group("a")), _clean_item(m.group("b"))
        cm = re.search(rf"\s+(?P<c>{_CRITERIA})$", b, I)
        if cm and b[:cm.start()].strip():
            crit = cm.group("c")
            spans.append(Span("comparison_criteria", crit.lower(), crit, m.end("b") - len(crit), m.end("b"), "phrase"))
            b = b[:cm.start()].strip()
        items = [x.strip() for x in re.split(r",\s*(?:and\s+)?", a) if x.strip()] + [b]
        items = [x for x in items if 0 < len(x.split()) <= 6]
        if len(items) >= 2:
            spans.append(Span("comparison_items", items, m.group(0), m.start(), m.end(), "phrase"))
            break
    for rx in _CRITERIA_PATTERNS:
        for m in rx.finditer(text):
            spans.append(Span("comparison_criteria", m.group("c").strip().lower(), m.group("c"),
                              m.start("c"), m.end("c"), "phrase"))
    return spans


# --------------------------------------------------------------------------- free-text phrases
_LEAD = re.compile(r"^(?:(?:please|pls|kindly|hey|hi|hello|ok|okay|so|um|uh|well|yo)\b[\s,!.]*)+", I)
_ASK = re.compile(
    r"^(?:(?:can|could|would|will)\s+(?:you|u)\s+(?:please\s+|pls\s+|kindly\s+)?|i\s+(?:want|need|would like|'d like|wanna)"
    r"(?:\s+you)?\s+to\s+|i'?m\s+looking\s+to\s+|i\s+am\s+looking\s+to\s+|help\s+me\s+(?:to\s+)?|let'?s\s+|let\s+me\s+)",
    I,
)
_VERB = re.compile(
    r"^(?:quickly\s+|just\s+)?(?:search|look\s+up|lookup|google|find\s+out|find|show|tell|give|get|explain|describe|"
    r"define|summari[sz]e|translate|write|draft|create|generate|make|build|compose|plan|recommend|suggest|research|"
    r"check|browse|look|list|fetch|pull\s+up|open|display|calculate|compute|code|implement)"
    r"(?:\s+me)?(?:\s+(?:about|for|up|on|out|into))?(?:\s+(?:the\s+web|online|the\s+internet|google|on\s+google|the\s+net))?"
    r"(?:\s+(?:for|about|on))?\s+",
    I,
)


def extract_topic(text: str) -> list[Span]:
    """The request with politeness and the leading imperative verb removed.

    Kept verbatim so a downstream handler gets the user's own wording.
    """
    start = 0
    rest = text
    for _ in range(4):
        before = rest
        for rx in (_LEAD, _ASK, _VERB):
            m = rx.match(rest)
            if m and m.end() < len(rest):
                start += m.end()
                rest = rest[m.end():]
        if rest == before:
            break
    stripped = re.sub(r"(?:[\s,]+(?:please|pls|thanks|thank you))?[\s?.!]*$", "", rest, flags=I)
    if len(stripped) < 2 or not any(ch.isalpha() for ch in stripped):
        return []
    return [Span("topic", stripped, stripped, start, start + len(stripped), "phrase")]


_USE_CASE = re.compile(
    r"\bfor\s+(?P<u>(?:(?:[a-z]+\s+)?[a-z]{3,}ing|programming|coding|gaming|office work|work|school|college|students?|kids|"
    r"beginners|home use|daily use|travel|photography|gym|streaming|music production|video editing|"
    r"content creation|data science|machine learning)(?:\s+(?!(?:that|which|with|under|below|in|near|and|or|but|from|at)\b)"
    r"[a-z]+){0,2})\b",
    I,
)
_USE_CASE_STOP = {"something", "anything", "nothing", "everything", "morning", "evening", "building", "clothing",
                  "nothing else", "meeting", "wedding", "upcoming", "string", "spring"}
_REQUIREMENT_PATTERNS = [
    re.compile(r"\b(?:that|which)\s+(?P<r>(?:i|you|we|it|they|can|will|has|have|is|are|should|must|supports?|comes?|"
               r"works?|lasts?|does|doesn't|don't|fits?|runs?)\b[^,.;?!]{2,80})", I),
    re.compile(r"\bwho\s+(?P<r>(?:likes|loves|enjoys|is into|plays|needs|wants|prefers)\b[^,.;?!]{2,60})", I),
    re.compile(r"\bwith\s+(?P<r>(?:(?:a|an|good|great|excellent|long|big|large|high|fast|decent|nice|at least|more than|"
               r"strong|lots of|plenty of)\s+)*(?:[\w-]+\s+){0,3}?(?:camera|battery(?:\s+life)?|ratings?|reviews?|display|"
               r"screen|storage|ram|memory|processor|warranty|sound|bass|noise\s+cancell?ing|anc|features?|design|"
               r"keyboard|performance|graphics|gpu|ssd|backlight|pockets?|delivery|discount|offers?|cushioning|grip)\b)", I),
]
_RECIPIENT = re.compile(
    r"\b(?:email|e-mail|mail|letter|message|note|reply|text|memo|invitation|invite|card)\s+(?:to|for)\s+"
    r"(?P<r>(?:my|the|our|a|an|his|her|their)?\s*[A-Za-z]+(?:\s+[A-Za-z]+){0,2}?)"
    r"(?=\s+(?:asking|about|regarding|requesting|for|to|that|explaining|saying|with|on|telling|informing|"
    r"announcing|inviting|and)\b|[,.!?]|$)",
    I,
)
_REMINDER = [
    re.compile(r"\bremind\s+(?:me|us)\s+(?:to|about|that)\s+(?P<t>.+?)(?=\s+(?:at|on|by|tomorrow|today|tonight|next|this|"
               r"every|in|before|after)\b|[,.!?]|$)", I),
    re.compile(r"\b(?:set|create|add|make)\s+(?:a\s+)?reminder\s+(?:to|for|about)\s+(?P<t>.+?)(?=\s+(?:at|on|by|tomorrow|"
               r"today|tonight|next|this|every|in)\b|[,.!?]|$)", I),
    re.compile(r"\b(?:don'?t|do not)\s+let\s+me\s+forget\s+(?:to\s+)?(?P<t>.+?)(?=\s+(?:at|on|by|tomorrow|today|tonight|"
               r"next|this|every|before)\b|[,.!?]|$)", I),
    re.compile(r"\b(?:schedule|book|set up|arrange|add)\s+(?:a|an|the|my)?\s*(?P<t>(?:[\w-]+\s+){0,3}?(?:meeting|call|"
               r"appointment|standup|stand-up|interview|session|review|sync|catch-up|alarm|demo))\b", I),
]
_PERSON = re.compile(r"\bwith\s+(?P<p>(?:the|my|our)\s+[a-z]+\s+team|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b")
_DOC_REF = re.compile(
    r"\b(?P<d>(?:this|these|that|those|the|my|attached|following)\s+(?:attached\s+)?(?:pdf|document|doc|file|report|"
    r"article|paper|contract|agreement|policy|email thread|email|text|paragraph|page|essay|notes|meeting notes|"
    r"transcript|spreadsheet|sheet|slides|presentation|chapter|book|invoices?|receipts?|review|message|thread|"
    r"letter|cover letter|resume|cv|code|snippet|function|query|design doc|sentence|passage)s?)\b",
    I,
)


class PhraseExtractor:
    """Verbatim phrase extraction: topic, use case, requirements, comparisons, etc."""

    name = "phrases"

    def __init__(self, excluded_person_words: set[str] | None = None):
        self.excluded_person_words = {w.lower() for w in (excluded_person_words or set())} | _NOT_PLACES

    def extract(self, text: str, ctx: ExtractionContext) -> list[Span]:
        spans = extract_topic(text) + extract_comparisons(text)
        for m in _USE_CASE.finditer(text):
            value = m.group("u").strip()
            if value.lower() not in _USE_CASE_STOP and value.split()[-1].lower() not in _USE_CASE_STOP:
                spans.append(Span("use_case", value, value, m.start("u"), m.end("u"), "phrase"))
        for rx in _REQUIREMENT_PATTERNS:
            for m in rx.finditer(text):
                value = m.group("r").strip()
                spans.append(Span("requirements", value, value, m.start("r"), m.end("r"), "phrase"))
        for m in _RECIPIENT.finditer(text):
            value = m.group("r").strip()
            spans.append(Span("recipient", value, value, m.start("r"), m.end("r"), "phrase"))
        for rx in _REMINDER:
            m = rx.search(text)
            if m:
                value = m.group("t").strip()
                spans.append(Span("reminder_task", value, value, m.start("t"), m.end("t"), "phrase"))
                break
        for m in _PERSON.finditer(text):
            value = m.group("p")
            is_team = value.lower().startswith(("the ", "my ", "our "))
            if is_team or not any(w.lower() in self.excluded_person_words for w in value.split()):
                spans.append(Span("person", value, value, m.start("p"), m.end("p"), "phrase"))
        for m in _DOC_REF.finditer(text):
            spans.append(Span("document_ref", m.group("d"), m.group("d"), m.start("d"), m.end("d"), "phrase"))
        return spans
