"""Optional NLI check that a request with side effects is really being asked for.

Disabled by default (ROUTER_VERIFIER=none). With ROUTER_VERIFIER=nli, a small
local cross-encoder (default cross-encoder/nli-deberta-v3-xsmall, pinned to one
revision, loaded from the local Hugging Face cache only) reads the message
together with statements such as "The user wants to cancel an order." and
blocks the route when the message contradicts every statement it could mean.

Why a cross-encoder: a bi-encoder (the router's embedding model) compresses each
sentence into one vector, so "cancel order 4421" and "do not cancel order 4421"
come out ~0.98 similar. A cross-encoder attends across both texts and reads the
negation.

Scope and safety properties:
- It runs only for intents that require confirmation (destructive or financial),
  after the deterministic negation rules, and only on standalone requests. On
  short follow-ups ("yes do it", "keep it") it cannot tell confirmation from
  withdrawal, so follow-ups rely on the rules alone.
- It can only stop a route (-> needs_clarification), never create or change one.
- If inference fails, the request is not routed for execution (fail closed).
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Protocol

from intent_router.core.logging import get_logger
from intent_router.router.registry import IntentDefinition

log = get_logger(__name__)


@dataclass
class Verdict:
    blocked: bool
    contradiction: float  # highest contradiction probability over the statements
    entailment: float  # highest entailment probability over the statements
    error: bool = False


class ActionVerifier(Protocol):
    name: str

    def check(self, text: str, intent: IntentDefinition) -> Verdict | None: ...


class NoVerifier:
    name = "none"

    def check(self, text: str, intent: IntentDefinition) -> Verdict | None:
        return None


def hypotheses_for(intent: IntentDefinition) -> list[str]:
    return list(intent.verify_hypotheses) or [f"The user wants to {intent.label}."]


class NliVerifier:
    """Blocks when some statement is contradicted (>= ``contradiction``) and none is
    entailed (>= ``entailment``): "Do not cancel order 4421" contradicts "The user
    wants to cancel an order", while "I want to return these shoes" entails the
    return statement even though it contradicts the cancel one."""

    name = "nli"

    def __init__(self, model, *, contradiction: float = 0.5, entailment: float = 0.5, max_chars: int = 500):
        self.model = model
        self.contradiction = contradiction
        self.entailment = entailment
        self.max_chars = max_chars
        inner = getattr(model, "model", None)  # CrossEncoder wraps a transformers model
        config = getattr(inner, "config", None) or getattr(model, "config", None)
        labels = {str(v).lower(): int(k) for k, v in getattr(config, "id2label", {}).items()}
        if "contradiction" not in labels or "entailment" not in labels:
            raise ValueError(f"verifier model labels {labels} lack 'contradiction'/'entailment'")
        self._c, self._e = labels["contradiction"], labels["entailment"]
        self._lock = threading.Lock()  # one inference at a time, like the embedder

    def check(self, text: str, intent: IntentDefinition) -> Verdict | None:
        if not intent.requires_confirmation:
            return None
        pairs = [(text[:self.max_chars], h) for h in hypotheses_for(intent)]
        try:
            with self._lock:
                probs = self.model.predict(pairs, apply_softmax=True, show_progress_bar=False)
        except Exception as exc:  # noqa: BLE001 - fail closed for actions with side effects
            log.warning("Verifier failed; not routing %s for execution: %s", intent.id, exc)
            return Verdict(blocked=True, contradiction=0.0, entailment=0.0, error=True)
        contradiction = float(probs[:, self._c].max())
        entailment = float(probs[:, self._e].max())
        blocked = contradiction >= self.contradiction and entailment < self.entailment
        return Verdict(blocked=blocked, contradiction=round(contradiction, 4), entailment=round(entailment, 4))


class VerifierNotAvailableError(RuntimeError):
    pass


def build_verifier(settings) -> ActionVerifier:
    """Loads the verifier at startup when enabled, so the first request is not slow
    and a missing model fails loudly instead of silently disabling the check."""
    if settings.verifier != "nli":
        return NoVerifier()
    if not settings.allow_model_download:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from sentence_transformers import CrossEncoder

    start = time.perf_counter()
    try:
        model = CrossEncoder(
            settings.verifier_model,
            revision=settings.verifier_revision,
            device=settings.device,
            cache_folder=settings.model_cache_dir,
            local_files_only=not settings.allow_model_download,
        )
    except Exception as exc:  # noqa: BLE001 - re-raised with guidance
        raise VerifierNotAvailableError(
            f"ROUTER_VERIFIER=nli but model '{settings.verifier_model}' could not be loaded. Download it with: "
            f"python scripts/download_models.py --verifier  (or set ROUTER_VERIFIER=none). Original error: {exc}"
        ) from exc
    log.info("Loaded verifier %s in %.2fs", settings.verifier_model, time.perf_counter() - start)
    return NliVerifier(model, contradiction=settings.verifier_contradiction,
                       entailment=settings.verifier_entailment)
