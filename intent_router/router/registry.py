"""Intent registry: loads and validates intent definitions from JSON.

The core file (``data/intents.json``) defines the base actions and intents.
Every ``*.json`` file in the configured domain directories is a domain extension
that may add actions, intents and gazetteer entries without touching the engine.
The registry is read-only after startup; nothing at request time can modify it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intent_router.core.schemas import SideEffects

_ID_RE = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+$")
_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")

# Ids the engine reserves for its own outcomes.
RESERVED_ACTIONS = {"unknown"}


class ParameterSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    # Extracted entity types that can fill this parameter, in priority order.
    types: list[str] = Field(min_length=1)
    kind: Literal["entity", "filter"] = "entity"
    multiple: bool = False
    required: bool = False
    # Listed in optional_missing_fields when absent (useful but not required).
    recommended: bool = False
    question: str | None = None  # clarification prompt when required and missing
    # Name of the parameter whose *excluded* values this one collects, e.g. an
    # "exclude_locations" filter with "excludes": "location". Without such a
    # parameter, an exclusion ("anywhere except Delhi") yields a clarification.
    excludes: str | None = None

    @field_validator("name")
    @classmethod
    def _check_name(cls, value: str) -> str:
        if not _NAME_RE.match(value):
            raise ValueError(f"invalid parameter name {value!r}")
        return value


class UnsupportedVariant(BaseModel):
    """An unsupported direction of an intent, recognised by case-insensitive regexes.

    A pattern can only ever reject a request that would otherwise be routed to the
    intent declaring it; it never routes anything.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    label: str = Field(min_length=3, max_length=80)  # "search for candidates to hire"
    patterns: list[str] = Field(min_length=1)

    @field_validator("patterns")
    @classmethod
    def _check_patterns(cls, value: list[str]) -> list[str]:
        for pattern in value:
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"invalid pattern {pattern!r}: {exc}") from exc
        return value

    def matches(self, text: str) -> bool:
        return any(re.search(p, text, re.IGNORECASE) for p in self.patterns)


class IntentDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    label: str = Field(min_length=3, max_length=80)  # short phrase used in clarification questions
    action: str
    target: str | None = None
    domain: str
    description: str = Field(min_length=10)
    examples: list[str] = Field(min_length=3)
    negative_examples: list[str] = Field(default_factory=list)
    parameters: list[ParameterSpec] = Field(default_factory=list)
    # Each inner list: at least one of these parameters must be present.
    required_any: list[list[str]] = Field(default_factory=list)
    required_any_question: str | None = None
    side_effects: SideEffects = "none"
    requires_confirmation: bool | None = None  # default: True for destructive/financial
    operation: str | None = None  # hint for the caller's own handler table; never executed here
    # For intents that carry no concrete task (e.g. "can you help me?"): always ask this.
    always_clarify: str | None = None
    # Known kinds of requests this deployment does NOT support (device control, payments...).
    # Matching one yields status "unknown" instead of a weak guess at a supported intent.
    out_of_scope: bool = False
    # Requests that look like this intent but go in a direction it does not support
    # (searching for candidates instead of job openings). Matching one yields status
    # "unknown". Deterministic on purpose: sentence embeddings place "find Python
    # candidates" and "find Python jobs" almost on top of each other.
    unsupported_variants: list[UnsupportedVariant] = Field(default_factory=list)
    # Statements the optional NLI verifier checks the message against, one per thing
    # the intent can mean ("The user wants to cancel an order."). Default: one built
    # from the label. Used only for intents that require confirmation.
    verify_hypotheses: list[str] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def _check_id(cls, value: str) -> str:
        if not _ID_RE.match(value):
            raise ValueError(f"intent id {value!r} must look like 'domain.name'")
        return value

    @model_validator(mode="after")
    def _check_consistency(self) -> "IntentDefinition":
        names = [p.name for p in self.parameters]
        if len(names) != len(set(names)):
            raise ValueError(f"{self.id}: duplicate parameter names")
        for p in self.parameters:
            if p.excludes is not None and (p.excludes not in names or p.excludes == p.name):
                raise ValueError(f"{self.id}: parameter {p.name!r} excludes unknown parameter {p.excludes!r}")
        for group in self.required_any:
            for name in group:
                if name not in names:
                    raise ValueError(f"{self.id}: required_any references unknown parameter {name!r}")
        if self.requires_confirmation is None:
            self.requires_confirmation = self.side_effects in ("destructive", "financial")
        if len({e.strip().lower() for e in self.examples}) != len(self.examples):
            raise ValueError(f"{self.id}: duplicate examples")
        return self

    def param(self, name: str) -> ParameterSpec | None:
        return next((p for p in self.parameters if p.name == name), None)


class ActionDefinition(BaseModel):
    """A broad action. Examples must be topic-neutral ("compare these two")
    so that the action stage reacts to the verb rather than the subject."""

    model_config = ConfigDict(extra="forbid")

    description: str = ""
    examples: list[str] = Field(default_factory=list)


class RegistryFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str = "1.0"
    description: str | None = None
    actions: dict[str, ActionDefinition] = Field(default_factory=dict)
    intents: list[IntentDefinition] = Field(default_factory=list)
    gazetteers: dict[str, dict[str, list[str]]] = Field(default_factory=dict)


class RegistryError(ValueError):
    pass


class IntentRegistry:
    def __init__(self, files: list[RegistryFile], sources: list[str] | None = None):
        self.actions: dict[str, ActionDefinition] = {}
        self.gazetteers: dict[str, dict[str, list[str]]] = {}
        intents: list[IntentDefinition] = []
        for f in files:
            for action, definition in f.actions.items():
                if action in RESERVED_ACTIONS:
                    raise RegistryError(f"action {action!r} is reserved")
                if action in self.actions:  # extensions may add examples to an action
                    merged = self.actions[action]
                    merged.examples.extend(e for e in definition.examples if e not in merged.examples)
                else:
                    self.actions[action] = definition.model_copy(deep=True)
            intents.extend(f.intents)
            for gname, entries in f.gazetteers.items():
                bucket = self.gazetteers.setdefault(gname, {})
                for canonical, aliases in entries.items():
                    bucket.setdefault(canonical, [])
                    bucket[canonical].extend(a for a in aliases if a not in bucket[canonical])

        seen: set[str] = set()
        for intent in intents:
            if intent.id in seen:
                raise RegistryError(f"duplicate intent id {intent.id!r}")
            seen.add(intent.id)
            if intent.action not in self.actions:
                raise RegistryError(
                    f"intent {intent.id!r} uses undeclared action {intent.action!r}; "
                    "declare it in an 'actions' section"
                )
        if not intents:
            raise RegistryError("registry contains no intents")
        for name, action in self.actions.items():
            if len(action.examples) < 3:
                raise RegistryError(f"action {name!r} needs at least 3 topic-neutral examples")
        self.intents: tuple[IntentDefinition, ...] = tuple(intents)
        self._by_id = {i.id: i for i in self.intents}
        self.sources = sources or []

    def unsupported_variant(self, intent_id: str, text: str) -> UnsupportedVariant | None:
        """The unsupported direction of ``intent_id`` that ``text`` asks for, if any."""
        intent = self._by_id.get(intent_id)
        if intent is None:
            return None
        return next((v for v in intent.unsupported_variants if v.matches(text)), None)

    def __len__(self) -> int:
        return len(self.intents)

    def get(self, intent_id: str) -> IntentDefinition | None:
        return self._by_id.get(intent_id)

    def index_of(self, intent_id: str) -> int:
        return next(i for i, it in enumerate(self.intents) if it.id == intent_id)

    @property
    def domains(self) -> list[str]:
        return sorted({i.domain for i in self.intents})

    @classmethod
    def load(cls, core_file: Path, domain_dirs: list[Path] | None = None) -> "IntentRegistry":
        paths = [Path(core_file)]
        for directory in domain_dirs or []:
            directory = Path(directory)
            if directory.is_dir():
                paths.extend(sorted(directory.glob("*.json")))
        files = []
        for path in paths:
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                files.append(RegistryFile.model_validate(raw))
            except Exception as exc:  # noqa: BLE001
                raise RegistryError(f"invalid registry file {path}: {exc}") from exc
        return cls(files, sources=[str(p) for p in paths])
