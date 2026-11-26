"""Deterministic regex extractors for values with reliable surface forms.

Covered: emails, URLs, phone numbers, file names, identifiers, error messages,
quoted text, code, money and price constraints, percentages, quantities,
durations, experience, ratings, people counts, length constraints, dates, date
ranges, times, math expressions and plain numbers.

Everything returned keeps the exact source text. Nothing is inferred beyond the
surface form, e.g. "50k" has currency null, not INR.
"""

from __future__ import annotations

import calendar
import re
from datetime import date, timedelta

from app.extractors.base import ExtractionContext, Span

I = re.IGNORECASE

NUMBER_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
                "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "a couple of": 2, "couple of": 2}
_WORDNUMS = "one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve"
_NUMW = rf"(?:\d+|{_WORDNUMS})"


def to_number(raw: str) -> int | float:
    value = float(raw.replace(",", ""))
    return int(value) if value.is_integer() else round(value, 4)


def word_number(raw: str) -> int | float:
    low = raw.lower().strip()
    return NUMBER_WORDS[low] if low in NUMBER_WORDS else to_number(low)


def _span(type_: str, value, m: re.Match, group: int | str = 0, source: str = "rules") -> Span:
    return Span(type_, value, m.group(group), m.start(group), m.end(group), source)


# --------------------------------------------------------------------------- contacts / ids
_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_URL = re.compile(r"\b(?:https?://|www\.)[^\s<>\"']+[^\s<>\"'.,;:!?)]", I)
_PHONE = re.compile(r"(?<![\w#])(?:\+\d{1,3}[\s-]?)?(?:\d[\s-]?){9,11}\d(?![\w])")
_FILE = re.compile(
    r"(?<![\w/\\])(?:[A-Za-z]:\\|\.{0,2}/)?(?:[\w\-. ]+[\\/])*[\w\-]+\."
    r"(?:pdf|docx?|xlsx?|csv|txt|pptx?|json|ya?ml|py|js|ts|tsx|jsx|java|cpp|cs|go|rb|php|md|png|jpe?g|gif|zip|html?|xml|log|sql|ipynb)\b",
    I,
)
_ID_KEYWORD = re.compile(
    r"\b(?P<kind>order|ticket|invoice|booking|pnr|reference|ref|case|tracking|transaction|account)\s*"
    r"(?:no\.?|number|num|id)?\s*[:#]?\s*(?P<id>#?[A-Z0-9][A-Z0-9-]{2,})\b",
    I,
)
_ID_HASH = re.compile(r"(?<![\w&])#(?P<id>[A-Za-z0-9][A-Za-z0-9-]{2,})\b")
_ID_CODE = re.compile(r"\b(?P<id>[A-Z]{2,6}-\d{2,}(?:-\d+)?)\b")


def extract_contacts_and_ids(text: str) -> list[Span]:
    spans = [_span("email", m.group(0), m) for m in _EMAIL.finditer(text)]
    spans += [_span("url", m.group(0), m) for m in _URL.finditer(text)]
    spans += [_span("phone", re.sub(r"[\s-]", "", m.group(0)), m) for m in _PHONE.finditer(text)]
    spans += [_span("file", m.group(0).strip(), m) for m in _FILE.finditer(text)]
    for m in _ID_KEYWORD.finditer(text):
        ident = m.group("id").lstrip("#")
        if any(ch.isdigit() for ch in ident):
            spans.append(Span("identifier", {"kind": m.group("kind").lower(), "value": ident},
                              m.group(0), m.start(), m.end()))
    for rx in (_ID_HASH, _ID_CODE):
        for m in rx.finditer(text):
            ident = m.group("id")
            if any(ch.isdigit() for ch in ident):
                spans.append(Span("identifier", {"kind": None, "value": ident}, m.group(0), m.start(), m.end()))
    return spans


# --------------------------------------------------------------------------- errors / quotes / code
_EXC = re.compile(r"\b[A-Z][A-Za-z]*(?:Error|Exception)\b(?:\s*:\s*[^\n?!]{1,120}?(?=$|[?!\n]|,\s*(?:how|what|why|help|please|any)\b))?")
_ERR_PHRASE = re.compile(
    r"\b(?:segmentation fault|stack overflow error|permission denied|command not found|module not found|"
    r"no module named \S+|syntax error|out of memory|connection refused|timed out|"
    r"[45]\d\d (?:error|not found|internal server error|bad request|forbidden|unauthorized)|error code \w+)\b",
    I,
)
_DQUOTE = re.compile(r'"([^"\n]{1,300})"')
_SQUOTE = re.compile(r"(?<![\w])'([^'\n]{1,200}?)'(?![\w])")
_INLINE_CODE = re.compile(r"`([^`\n]{1,300})`")


def extract_errors_quotes_code(text: str, ctx: ExtractionContext) -> list[Span]:
    spans = [_span("error_message", m.group(0).strip(), m) for m in _EXC.finditer(text)]
    spans += [_span("error_message", m.group(0), m) for m in _ERR_PHRASE.finditer(text)]
    for rx in (_DQUOTE, _SQUOTE):
        spans += [_span("quoted_text", m.group(1), m, 1) for m in rx.finditer(text)]
    spans += [_span("code", m.group(1), m, 1) for m in _INLINE_CODE.finditer(text)]
    for block in ctx.code_blocks:
        spans.append(Span("code", block, "```code```", len(text), len(text)))
    if ctx.content:
        spans.append(Span("content", ctx.content, "<content>", len(text), len(text)))
    return spans


# --------------------------------------------------------------------------- money
_CURRENCIES = {
    "₹": "INR", "rs": "INR", "rs.": "INR", "inr": "INR", "rupee": "INR", "rupees": "INR",
    "$": "USD", "us$": "USD", "usd": "USD", "dollar": "USD", "dollars": "USD", "bucks": "USD",
    "€": "EUR", "eur": "EUR", "euro": "EUR", "euros": "EUR", "£": "GBP", "gbp": "GBP",
}
_MULTIPLIERS = {"k": 1e3, "thousand": 1e3, "lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "lacs": 1e5, "lpa": 1e5,
                "crore": 1e7, "crores": 1e7, "cr": 1e7, "million": 1e6, "mn": 1e6, "billion": 1e9, "bn": 1e9}
_PERIODS = {"month": "month", "mo": "month", "year": "year", "yr": "year", "annum": "year", "a.": "year",
            "hour": "hour", "hr": "hour", "day": "day", "week": "week"}


def _money_pattern(sfx: str = "") -> str:
    return (
        rf"(?:(?P<pc{sfx}>₹|rs\.?|inr|us\$|\$|usd|€|eur|£|gbp)\s*)?"
        rf"(?P<num{sfx}>\d{{1,3}}(?:,\d{{2,3}})+(?:\.\d+)?|\d+(?:\.\d+)?)"
        rf"(?:\s*(?P<mult{sfx}>k|thousand|lakhs?|lacs?|lpa|crores?|cr|million|mn|billion|bn)\b)?"
        rf"(?:\s*(?P<sc{sfx}>rupees?|rs\.?|inr|dollars?|usd|euros?|eur|gbp|bucks)\b)?"
    )


_PERIOD_RX = r"(?:\s*(?:/|per|a|an|p\.)\s*(?P<period>month|mo|year|yr|annum|a\.|hour|hr|day|week)\b)?"
_MONEY = re.compile(r"(?<![\w.])" + _money_pattern() + _PERIOD_RX, I)
_MAX_OPS = (r"under|below|less than|lesser than|cheaper than|upto|up to|within|max(?:imum)?|at most|"
            r"not more than|no more than|not exceeding|under a budget of|my budget is|budget of|budget is|budget:|budget|<=?")
_MIN_OPS = r"over|above|more than|greater than|at least|min(?:imum)?|starting from|starting at|upwards of|>=?|from"
_APPROX_OPS = r"around|about|approximately|approx\.?|roughly|~|close to"
_OP_MONEY = re.compile(
    rf"(?<![\w])(?P<op>{_MAX_OPS}|{_MIN_OPS}|{_APPROX_OPS})\s*(?:of\s+)?" + _money_pattern() + _PERIOD_RX
    + r"(?P<post>\s*\+|\s+or\s+(?:less|below|under|more|above)\b|\s+and\s+(?:below|above|under|over)\b)?",
    I,
)
_POST_MONEY = re.compile(
    r"(?<![\w.])" + _money_pattern() + _PERIOD_RX
    + r"(?P<post>\s*\+|\s+or\s+(?:less|below|under|more|above)\b|\s+and\s+(?:below|above|under|over)\b|\s+plus\b)",
    I,
)
_RANGE_MONEY = re.compile(
    r"(?<![\w.])(?P<between>between\s+)?" + _money_pattern("a") + r"\s*(?P<sep>-|to|and)\s*"
    + _money_pattern("b") + _PERIOD_RX,
    I,
)
_UNIT_WORDS = {
    "kg", "kgs", "g", "gb", "tb", "mb", "km", "kms", "cm", "mm", "inch", "inches", "ft", "feet", "hours", "hour",
    "hrs", "days", "day", "weeks", "week", "months", "month", "years", "year", "yrs", "minutes", "mins", "people",
    "persons", "adults", "stars", "star", "percent", "times", "items", "pieces", "pcs", "litres", "liters", "ml",
    "mah", "mp", "hz", "ghz", "watts", "rpm", "nights", "words", "sentences", "pages", "employees", "users",
    "followers", "reviews", "ratings", "km/h", "kmph", "calories", "steps",
}
_FOLLOW_OK = re.compile(r"^\s*(?:$|[,.;:!?)]|(?:and|or|for|in|with|that|which|please|only|budget|range|rs|inr|"
                        r"near|around|on|at|to|but|if|as)\b)", I)
_RESOLUTION_NEXT = re.compile(r"^\s*(?:uhd\s+|ultra hd\s+|oled\s+|led\s+|smart\s+)?"
                              r"(?:tv|television|monitor|display|resolution|video|screen|camera)\b", I)


def _money_value(m: re.Match, sfx: str = "", inherit: dict | None = None) -> dict | None:
    num = m.group(f"num{sfx}")
    mult = (m.group(f"mult{sfx}") or "").lower()
    cur = (m.group(f"pc{sfx}") or m.group(f"sc{sfx}") or "").lower()
    explicit = bool(mult or cur)
    if not explicit and inherit:
        mult, cur = inherit.get("mult", ""), inherit.get("cur", "")
    amount = to_number(num) * _MULTIPLIERS.get(mult, 1)
    currency = _CURRENCIES.get(cur)
    period = None
    if "period" in m.groupdict() and m.group("period"):
        period = _PERIODS.get(m.group("period").lower())
    if mult == "lpa":
        currency, period = "INR", "year"
    amount = int(amount) if float(amount).is_integer() else round(amount, 2)
    return {"amount": amount, "currency": currency, "period": period, "explicit": explicit,
            "mult": mult, "cur": cur}


def _followed_by_unit(text: str, end: int) -> bool:
    nxt = re.match(r"\s*([A-Za-z/]+)", text[end:])
    return bool(nxt and nxt.group(1).lower() in _UNIT_WORDS)


def _acceptable_bare(value: dict, text: str, end: int) -> bool:
    """A number with no currency/multiplier counts as money only in price-like position."""
    if value["explicit"]:
        return bool(value["cur"]) or not _followed_by_unit(text, end)
    if value["amount"] < 100:
        return False
    if _followed_by_unit(text, end):
        return False
    return bool(_FOLLOW_OK.match(text[end:]))


def _clean(v: dict) -> dict:
    return {k: v[k] for k in ("amount", "currency", "period")}


_RANGE_PRICE_LEAD = re.compile(r"\b(?:budget|salary|price)(?:\s+(?:of|is)|:)?\s+$", I)


def extract_money(text: str) -> list[Span]:
    spans: list[Span] = []
    for m in _MONEY.finditer(text):
        v = _money_value(m)
        if not v["explicit"]:
            continue
        if v["mult"] == "k" and _RESOLUTION_NEXT.match(text[m.end():]):
            spans.append(Span("quantity", {"amount": to_number(m.group("num")), "unit": "K resolution"},
                              m.group(0), m.start(), m.end()))
            continue
        if not v["cur"] and _followed_by_unit(text, m.end()):
            continue
        spans.append(Span("money", {**_clean(v), "text": m.group(0).strip()}, m.group(0).strip(),
                          m.start(), m.start() + len(m.group(0).rstrip())))

    for m in _OP_MONEY.finditer(text):
        v = _money_value(m)
        op = m.group("op").lower()
        if op == "from" and not v["explicit"]:
            continue
        if v["mult"] == "k" and _RESOLUTION_NEXT.match(text[m.end():]):
            continue
        if not _acceptable_bare(v, text, m.end()):
            continue
        post = (m.group("post") or "").strip().lower()
        kind = "max" if re.fullmatch(_MAX_OPS, op, I) else "min" if re.fullmatch(_MIN_OPS, op, I) else "approx"
        if post in ("+", "plus") or post.endswith(("more", "above", "over")):
            kind = "min"
        elif post.endswith(("less", "below", "under")):
            kind = "max"
        spans.append(_constraint_span(kind, v, m))

    for m in _POST_MONEY.finditer(text):
        v = _money_value(m)
        if not v["explicit"]:
            continue
        if not v["cur"] and _followed_by_unit(text, m.end()):
            continue
        post = m.group("post").strip().lower()
        kind = "max" if post.endswith(("less", "below", "under")) else "min"
        spans.append(_constraint_span(kind, v, m))

    for m in _RANGE_MONEY.finditer(text):
        if m.group("sep").lower() == "and" and not m.group("between"):
            continue
        b = _money_value(m, "b")
        a = _money_value(m, "a", inherit=b if b["explicit"] else None)
        if a["explicit"] and not b["explicit"] and a["cur"]:
            b = _money_value(m, "b", inherit={"cur": a["cur"], "mult": ""})
        lead = _RANGE_PRICE_LEAD.search(text[:m.start()])
        bare_price_range = bool(lead and a["amount"] >= 100 and not _followed_by_unit(text, m.end()))
        if not (a["explicit"] or b["explicit"] or bare_price_range):
            continue
        if not (b["cur"] or a["cur"]) and _followed_by_unit(text, m.end()):
            continue
        if a["amount"] > b["amount"]:
            continue
        start = lead.start() if bare_price_range and lead else m.start()
        end = m.start() + len(m.group(0).rstrip())
        raw_text = text[start:end].strip()
        spans.append(Span("money_constraint",
                          {"min": a["amount"], "max": b["amount"], "currency": b["currency"] or a["currency"],
                           "period": b["period"] or a["period"], "approximate": False, "text": raw_text},
                          raw_text, start, end))
    return spans


def _constraint_span(kind: str, v: dict, m: re.Match) -> Span:
    value = {"min": None, "max": None, "currency": v["currency"], "period": v["period"],
             "approximate": kind == "approx", "text": m.group(0).strip()}
    if kind in ("min", "max"):
        value[kind] = v["amount"]
    else:
        value["min"] = value["max"] = v["amount"]
    return Span("money_constraint", value, m.group(0).strip(), m.start(), m.start() + len(m.group(0).rstrip()))


# --------------------------------------------------------------------------- measures
_PERCENT = re.compile(r"(?<![\w.])(?P<n>\d+(?:\.\d+)?)\s*(?:%|percent\b|per cent\b)", I)
_QUANTITY_UNITS = {
    "kg": "kg", "kgs": "kg", "g": "g", "gm": "g", "gram": "g", "grams": "g", "gb": "GB", "tb": "TB", "mb": "MB",
    "km": "km", "kms": "km", "kilometer": "km", "kilometers": "km", "kilometre": "km", "kilometres": "km",
    "mile": "mile", "miles": "mile", "meter": "m", "meters": "m", "metre": "m", "metres": "m", "cm": "cm", "mm": "mm",
    "inch": "inch", "inches": "inch", '"': "inch", "ft": "ft", "feet": "ft", "litre": "L", "litres": "L",
    "liter": "L", "liters": "L", "ml": "mL", "mah": "mAh", "mp": "MP", "hz": "Hz", "ghz": "GHz", "watt": "W",
    "watts": "W", "rpm": "rpm", "lb": "lb", "lbs": "lb", "pounds": "lb", "celsius": "C", "fahrenheit": "F",
}
_QUANTITY = re.compile(
    r"(?<![\w.])(?P<n>\d+(?:\.\d+)?)\s*-?\s*(?P<u>" + "|".join(sorted(map(re.escape, _QUANTITY_UNITS), key=len, reverse=True))
    + r")(?![\w])",
    I,
)
_DURATION_UNITS = {"minute": "minute", "min": "minute", "hour": "hour", "hr": "hour", "day": "day", "night": "night",
                   "week": "week", "month": "month", "year": "year", "yr": "year", "weekend": "weekend"}
_DURATION = re.compile(
    rf"(?<![\w])(?P<n>\d+(?:\.\d+)?|a couple of|couple of|{_WORDNUMS}|a|an)\s*[- ]?\s*"
    r"(?P<u>minutes?|mins?|hours?|hrs?|days?|nights?|weeks?|months?|years?|yrs?)\b(?!\s+ago)",
    I,
)
_EXPERIENCE = re.compile(
    r"(?<![\w.])(?P<min>\d+(?:\.\d+)?)\s*(?:(?:-|to)\s*(?P<max>\d+(?:\.\d+)?))?\s*(?P<plus>\+|plus)?\s*"
    r"(?:years?|yrs?)(?:\s+(?:of\s+)?(?:work\s+|relevant\s+|industry\s+)?(?P<word>experience|exp|experienced)\b)?",
    I,
)
_EXPERIENCE_PREFIX = re.compile(
    r"\bexperience\s*(?:of|:)?\s*(?P<min>\d+(?:\.\d+)?)\s*(?:(?:-|to)\s*(?P<max>\d+))?\s*\+?\s*(?:years?|yrs?)", I
)
_LEVELS = {"fresher": "fresher", "freshers": "fresher", "entry level": "entry", "entry-level": "entry",
           "junior": "junior", "mid level": "mid", "mid-level": "mid", "mid senior": "mid-senior",
           "mid-senior": "mid-senior", "senior": "senior", "lead": "lead", "principal": "principal",
           "experienced": "experienced"}
_LEVEL = re.compile(r"\b(?P<l>" + "|".join(sorted(map(re.escape, _LEVELS), key=len, reverse=True)) + r")\b", I)
_RATING_1 = re.compile(
    r"(?<![\w.])(?P<n>[1-5](?:\.\d)?)\s*(?P<plus>\+)?\s*-?\s*stars?\b"
    r"(?P<post>\s*(?:and|&|or)\s*(?:above|up|more|higher)\b)?",
    I,
)
_RATING_2 = re.compile(
    r"\b(?:rated|rating|ratings)\s*(?:of\s+)?(?P<op>above|over|at least|more than|>=?)?\s*(?P<n>[1-5](?:\.\d)?)"
    r"(?P<plus>\s*\+|\s*(?:stars?\s*)?(?:and|or)\s+(?:above|more|higher)\b)?",
    I,
)
_PEOPLE = re.compile(
    rf"\b(?:for|party of|group of|with)\s+(?P<g>(?P<n>{_NUMW})\s+(?:people|persons?|adults?|guests?|pax|"
    r"travell?ers?|passengers?|members?|kids|children|friends))\b",
    I,
)
_PEOPLE_TABLE = re.compile(rf"\btable for\s+(?P<g>(?P<n>{_NUMW}))\b", I)
_TICKETS = re.compile(rf"\b(?P<g>(?P<n>{_NUMW})\s+(?:\w+\s+)?(?:tickets?|seats?))\b", I)
_LENGTH = re.compile(
    rf"(?<![\w.])(?P<n>{_NUMW})\s*-?\s*(?P<u>words?|sentences?|lines?|bullet points?|bullets|paragraphs?|pages?|points|tweets?)\b",
    I,
)


def extract_measures(text: str) -> list[Span]:
    spans: list[Span] = []
    for m in _PERCENT.finditer(text):
        spans.append(_span("percentage", {"value": to_number(m.group("n"))}, m))
    for m in _QUANTITY.finditer(text):
        unit = _QUANTITY_UNITS[m.group("u").lower()]
        spans.append(_span("quantity", {"amount": to_number(m.group("n")), "unit": unit}, m))
    for m in _DURATION.finditer(text):
        unit = m.group("u").lower().rstrip("s")
        unit = _DURATION_UNITS.get(unit, unit)
        spans.append(_span("duration", {"amount": word_number(m.group("n")), "unit": unit}, m))
    for rx in (_EXPERIENCE, _EXPERIENCE_PREFIX):
        for m in rx.finditer(text):
            gd = m.groupdict()
            if rx is _EXPERIENCE and not (gd.get("word") or gd.get("plus")):
                continue
            value = {"min_years": to_number(gd["min"]),
                     "max_years": to_number(gd["max"]) if gd.get("max") else None, "level": None}
            spans.append(_span("experience", value, m))
    for m in _LEVEL.finditer(text):
        spans.append(_span("experience", {"min_years": None, "max_years": None,
                                          "level": _LEVELS[m.group("l").lower()]}, m))
    for m in _RATING_1.finditer(text):
        n = to_number(m.group("n"))
        gte = bool(m.group("plus") or m.group("post"))
        spans.append(_span("rating", {"min": n} if gte else {"equals": n}, m))
    for m in _RATING_2.finditer(text):
        n = to_number(m.group("n"))
        gte = bool(m.group("op") or m.group("plus"))
        spans.append(_span("rating", {"min": n} if gte else {"equals": n}, m))
    for rx in (_PEOPLE, _PEOPLE_TABLE, _TICKETS):
        for m in rx.finditer(text):
            # For "two bus tickets" only the number is the span, so "bus" stays free for other extractors.
            group = "n" if rx is _TICKETS else "g"
            spans.append(_span("people_count", {"count": word_number(m.group("n"))}, m, group))
    for m in _LENGTH.finditer(text):
        unit = m.group("u").lower()
        unit = "bullet points" if unit.startswith("bullet") else unit if unit.endswith("s") else unit + "s"
        spans.append(_span("length_constraint", {"amount": word_number(m.group("n")), "unit": unit}, m))
    return spans


# --------------------------------------------------------------------------- dates and times
_MONTHS = {name.lower(): i for i, name in enumerate(calendar.month_name) if name}
_MONTHS.update({name.lower(): i for i, name in enumerate(calendar.month_abbr) if name})
_MONTHS["sept"] = 9
_MON = r"(?P<mon>" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")"
_WEEKDAYS = {name.lower(): i for i, name in enumerate(calendar.day_name)}
_WEEKDAY_ABBR = {"mon": 0, "tue": 1, "tues": 1, "wed": 2, "thu": 3, "thur": 3, "thurs": 3, "fri": 4, "sat": 5, "sun": 6}

_ISO = re.compile(r"\b(?P<y>\d{4})-(?P<m>\d{1,2})-(?P<d>\d{1,2})\b")
_NUMERIC = re.compile(r"\b(?P<a>\d{1,2})(?P<sep>[/.])(?P<b>\d{1,2})(?P=sep)(?P<y>\d{4}|\d{2})\b|"
                      r"\b(?P<a2>\d{1,2})-(?P<b2>\d{1,2})-(?P<y2>\d{4})\b")
_DAY_MON = re.compile(r"\b(?P<d>\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?" + _MON + r"\b\.?(?:,?\s+(?P<y>\d{4}))?", I)
_MON_DAY = re.compile(r"\b" + _MON + r"\.?\s+(?P<d>\d{1,2})(?:st|nd|rd|th)?\b(?:,?\s+(?P<y>\d{4}))?", I)
_MONTH_ONLY = re.compile(
    r"\b(?P<prep>in|during|for|this|next|last|by|until|till|from|of|early|mid|late|since)\s+" + _MON
    + r"\b(?:\s+(?P<y>\d{4}))?", I)
_ORDINAL_DAY = re.compile(r"\b(?:on\s+)?the\s+(?P<d>\d{1,2})(?:st|nd|rd|th)\b(?!\s+(?:of\s+)?[A-Za-z]{3,9}\b)", I)
_RELATIVE = re.compile(r"\b(?P<w>day after tomorrow|day before yesterday|today|tonight|tomorrow|tmrw|tmr|"
                       r"tomorow|yesterday)\b", I)
_WEEKDAY = re.compile(
    r"\b(?:(?P<mod>next|this|coming|last|on|by|until|before)\s+)?(?P<wd>monday|tuesday|wednesday|thursday|friday|"
    r"saturday|sunday)s?\b|\b(?P<mod2>next|this|coming|last|on|by)\s+(?P<wd2>mon|tues|tue|wed|thurs|thur|thu|fri|sat|sun)\b",
    I,
)
_RANGE_REL = re.compile(r"\b(?P<mod>this|next|last|coming|past)\s+(?P<unit>week|weekend|month|year)\b", I)
_RANGE_N = re.compile(rf"\b(?P<mod>last|past|next|previous|coming)\s+(?P<n>{_NUMW})\s+(?P<unit>days?|weeks?|months?)\b", I)
_AGO = re.compile(rf"\b(?P<n>{_NUMW})\s+(?P<unit>days?|weeks?|months?|years?)\s+ago\b", I)

_TIME_12 = re.compile(r"\b(?P<h>1[0-2]|0?[1-9])(?:[:.](?P<m>[0-5]\d))?\s*(?P<ap>a\.?m\.?|p\.?m\.?)(?![a-z])", I)
_TIME_24 = re.compile(r"(?<![\d:/.-])(?P<h>[01]?\d|2[0-3]):(?P<m>[0-5]\d)(?![\d:])")
_TIME_WORD = re.compile(r"\b(?P<w>noon|midday|midnight)\b", I)
_TIME_POD = re.compile(r"\b(?:at|for)\s+(?P<h>1[0-2]|[1-9])\b(?=[^.?!]{0,25}\b(?P<pod>morning|afternoon|evening|night)\b)", I)
_PART_OF_DAY = re.compile(r"\b(?P<w>morning|afternoon|evening|night)\b", I)


def _add_months(d: date, n: int) -> date | None:
    month = d.month - 1 + n
    year = d.year + month // 12
    month = month % 12 + 1
    if not (1 <= year <= 9999):
        return None
    return date(year, month, min(d.day, calendar.monthrange(year, month)[1]))


def _safe_date(y: int, m: int, d: int) -> date | None:
    try:
        return date(y, m, d)
    except ValueError:
        return None


def _safe_shift_days(d: date, days: int) -> date | None:
    try:
        return d + timedelta(days=days)
    except OverflowError:
        return None


def _date_value(d: date, text: str, **extra) -> dict:
    return {"date": d.isoformat(), "text": text, **extra}


def _range_value(start: date, end: date, text: str) -> dict:
    return {"start": start.isoformat(), "end": end.isoformat(), "text": text}


def extract_dates(text: str, ctx: ExtractionContext) -> list[Span]:
    today = ctx.reference_date
    spans: list[Span] = []

    def upcoming(month: int, day: int) -> tuple[date | None, bool]:
        d = _safe_date(today.year, month, day)
        if d and d < today:
            d = _safe_date(today.year + 1, month, day)
        return d, True

    for m in _ISO.finditer(text):
        d = _safe_date(int(m["y"]), int(m["m"]), int(m["d"]))
        if d:
            spans.append(_span("date", _date_value(d, m.group(0)), m))
    for m in _NUMERIC.finditer(text):
        a, b, y = (m["a"], m["b"], m["y"]) if m["a"] else (m["a2"], m["b2"], m["y2"])
        a, b, y = int(a), int(b), int(y)
        y = y + 2000 if y < 100 else y
        day, month = (a, b) if ctx.date_order == "DMY" else (b, a)
        if month > 12 and day <= 12:  # unambiguous the other way round
            day, month = month, day
        d = _safe_date(y, month, day)
        if d:
            ambiguous = a <= 12 and b <= 12 and a != b
            spans.append(_span("date", _date_value(d, m.group(0), order_assumed=ctx.date_order if ambiguous else None), m))
    for rx in (_DAY_MON, _MON_DAY):
        for m in rx.finditer(text):
            month = _MONTHS[m["mon"].lower()]
            if m["y"]:
                d, inferred = _safe_date(int(m["y"]), month, int(m["d"])), False
            else:
                d, inferred = upcoming(month, int(m["d"]))
            if d:
                extra = {"year_inferred": True} if inferred else {}
                spans.append(_span("date", _date_value(d, m.group(0), **extra), m))
    exact_date_spans = [(s.start, s.end) for s in spans]
    for m in _MONTH_ONLY.finditer(text):
        span_start = m.start("mon") if m["prep"].lower() not in ("this", "next", "last") else m.start()
        if any(a < m.end() and span_start < b for a, b in exact_date_spans):
            continue
        month = _MONTHS[m["mon"].lower()]
        prep = m["prep"].lower()
        if m["y"]:
            year = int(m["y"])
        elif prep == "since":
            year = today.year if month <= today.month else today.year - 1
        elif prep == "last":
            year = today.year if month < today.month else today.year - 1
        else:
            year = today.year if month >= today.month else today.year + 1
        start = _safe_date(year, month, 1)
        if start is None:
            continue
        end = date(year, month, calendar.monthrange(year, month)[1])
        spans.append(Span("date_range", _range_value(start, end, m.group(0)), m.group(0), span_start, m.end()))
    for m in _ORDINAL_DAY.finditer(text):
        day = int(m["d"])
        d = _safe_date(today.year, today.month, day)
        if d is None or d < today:
            nxt = _add_months(today.replace(day=1), 1)
            d = _safe_date(nxt.year, nxt.month, day) if nxt else None
        if d:
            spans.append(_span("date", _date_value(d, m.group(0)), m))
    for m in _RELATIVE.finditer(text):
        w = m["w"].lower()
        offset = {"today": 0, "tonight": 0, "tomorrow": 1, "tmrw": 1, "tmr": 1, "tomorow": 1, "yesterday": -1,
                  "day after tomorrow": 2, "day before yesterday": -2}[w]
        extra = {"part_of_day": "night"} if w == "tonight" else {}
        d = _safe_shift_days(today, offset)
        if d:
            spans.append(_span("date", _date_value(d, m.group(0), **extra), m))
    for m in _WEEKDAY.finditer(text):
        if m["wd"]:
            wd, mod = _WEEKDAYS[m["wd"].lower()], (m["mod"] or "").lower()
        else:
            wd, mod = _WEEKDAY_ABBR[m["wd2"].lower()], m["mod2"].lower()
        delta = (wd - today.weekday()) % 7
        if mod == "last":
            delta = delta - 7 if delta else -7
        elif mod == "this" and delta == 0:
            delta = 0
        elif delta == 0:
            delta = 7
        # "next Friday" is ambiguous in English; we use the next occurrence and keep the text.
        d = _safe_shift_days(today, delta)
        if d:
            spans.append(_span("date", _date_value(d, m.group(0)), m))
    for m in _RANGE_REL.finditer(text):
        mod, unit = m["mod"].lower(), m["unit"].lower()
        step = {"this": 0, "coming": 1, "next": 1, "last": -1, "past": -1}[mod]
        start = end = None
        if unit == "week":
            monday = _safe_shift_days(today, -today.weekday())
            if monday is not None:
                start = _safe_shift_days(monday, 7 * step)
                end = _safe_shift_days(start, 6) if start is not None else None
        elif unit == "weekend":
            wd = today.weekday()
            this_sat = _safe_shift_days(today, -1 if wd == 6 else (5 - wd) % 7)
            offset = {"this": 0, "coming": 0, "next": 1, "last": -1, "past": -1}[mod]
            if this_sat is not None:
                start = _safe_shift_days(this_sat, 7 * offset)
                end = _safe_shift_days(start, 1) if start is not None else None
        elif unit == "month":
            first = _add_months(today.replace(day=1), step)
            if first is not None:
                start, end = first, date(first.year, first.month, calendar.monthrange(first.year, first.month)[1])
        else:
            year = today.year + step
            if 1 <= year <= 9999:
                start, end = date(year, 1, 1), date(year, 12, 31)
        if start and end:
            spans.append(_span("date_range", _range_value(start, end, m.group(0)), m))
    for m in _RANGE_N.finditer(text):
        n = int(word_number(m["n"]))
        unit = m["unit"].lower().rstrip("s")
        days = n if unit == "day" else 7 * n if unit == "week" else None
        if m["mod"].lower() in ("last", "past", "previous"):
            start = _safe_shift_days(today, -days) if days is not None else _add_months(today, -n)
            if start:
                spans.append(_span("date_range", _range_value(start, today, m.group(0)), m))
        else:
            end = _safe_shift_days(today, days) if days is not None else _add_months(today, n)
            if end:
                spans.append(_span("date_range", _range_value(today, end, m.group(0)), m))
    for m in _AGO.finditer(text):
        n = int(word_number(m["n"]))
        unit = m["unit"].lower().rstrip("s")
        if unit in ("day", "week"):
            d = _safe_shift_days(today, -(n if unit == "day" else 7 * n))
        elif unit == "month":
            d = _add_months(today, -n)
        else:
            d = _safe_date(today.year - n, today.month, min(today.day, 28))
        if d:
            spans.append(_span("date", _date_value(d, m.group(0)), m))
    return spans


def extract_times(text: str) -> list[Span]:
    spans: list[Span] = []
    for m in _TIME_12.finditer(text):
        h, mins = int(m["h"]), int(m["m"] or 0)
        pm = m["ap"].lower().startswith("p")
        h = (h % 12) + (12 if pm else 0)
        spans.append(_span("time", {"time": f"{h:02d}:{mins:02d}", "text": m.group(0)}, m))
    for m in _TIME_24.finditer(text):
        spans.append(_span("time", {"time": f"{int(m['h']):02d}:{m['m']}", "text": m.group(0)}, m))
    for m in _TIME_WORD.finditer(text):
        t = "00:00" if m["w"].lower() == "midnight" else "12:00"
        spans.append(_span("time", {"time": t, "text": m.group(0)}, m))
    for m in _TIME_POD.finditer(text):
        h = int(m["h"])
        if m["pod"].lower() in ("afternoon", "evening", "night") and h < 12:
            h += 12
        spans.append(Span("time", {"time": f"{h:02d}:00", "text": m.group(0)}, m.group(0), m.start(), m.end()))
    for m in _PART_OF_DAY.finditer(text):
        spans.append(_span("part_of_day", m["w"].lower(), m))
    return spans


# --------------------------------------------------------------------------- numbers / math
_MATH = re.compile(
    r"(?<![\w.,])(?:\d+(?:\.\d+)?\s*(?:[+*/×÷^x]|\s-\s)\s*)+\d+(?:\.\d+)?(?![\w.])"
)
_NUMBER = re.compile(r"(?<![\w.,#])(?:\d{1,3}(?:,\d{2,3})+|\d+)(?:\.\d+)?(?![\w])")


def extract_numbers(text: str) -> list[Span]:
    spans = [_span("math_expression", m.group(0), m) for m in _MATH.finditer(text)]
    spans += [_span("number", to_number(m.group(0)), m) for m in _NUMBER.finditer(text)]
    return spans


class RuleExtractor:
    """All regex-based extractors bundled behind the Extractor interface."""

    name = "rules"

    def extract(self, text: str, ctx: ExtractionContext) -> list[Span]:
        spans: list[Span] = []
        spans += extract_contacts_and_ids(text)
        spans += extract_errors_quotes_code(text, ctx)
        spans += extract_money(text)
        spans += extract_measures(text)
        spans += extract_dates(text, ctx)
        spans += extract_times(text)
        spans += extract_numbers(text)
        return spans
