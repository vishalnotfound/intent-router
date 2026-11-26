"""Extractor interface.

An extractor turns text into typed :class:`Span` objects. Extractors never
decide the intent; the pipeline maps spans to the parameters the chosen intent
declares. New extractors (a pretrained NER model, a local LLM) only need to
implement :class:`Extractor` and be added to the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Protocol, runtime_checkable


@dataclass
class Span:
    type: str
    value: Any  # must be JSON-serializable
    text: str  # the exact text it came from
    start: int
    end: int
    source: str = "rules"

    def overlaps(self, other: "Span") -> bool:
        return self.start < other.end and other.start < self.end

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass
class ExtractionContext:
    reference_date: date
    date_order: str = "DMY"
    content: str | None = None  # pasted payload separated from the instruction
    code_blocks: list[str] = field(default_factory=list)


@runtime_checkable
class Extractor(Protocol):
    name: str

    def extract(self, text: str, ctx: ExtractionContext) -> list[Span]: ...


# Types whose spans may overlap others (phrases, overlays and free text).
# All other types are "atomic": among overlapping atomic spans only one survives.
SOFT_TYPES = frozenset({
    "origin", "destination", "target_language", "source_language", "quoted_text", "code", "content",
    "document_ref", "comparison_items", "comparison_criteria", "topic", "use_case", "requirements",
    "recipient", "reminder_task", "person", "brand",
})

# Soft types never reported under entities["other"] when no parameter consumed them.
NOT_REPORTED_WHEN_UNMAPPED = SOFT_TYPES - {"quoted_text", "code"}

PRIORITY = {
    "email": 100, "url": 100, "file": 95, "phone": 90, "identifier": 85, "error_message": 85,
    "experience": 80, "rating": 80, "money_constraint": 75, "length_constraint": 72, "people_count": 72,
    "money": 70, "percentage": 65, "quantity": 60, "date_range": 58, "date": 55, "time": 55,
    "duration": 50, "part_of_day": 45, "math_expression": 40, "number": 10,
}
GAZETTEER_PRIORITY = 30
