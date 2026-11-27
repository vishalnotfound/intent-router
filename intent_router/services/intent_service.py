"""Builds the router once at startup and exposes a single `route` call."""

from __future__ import annotations

import time
from datetime import date

from intent_router.core.config import Settings, get_settings
from intent_router.core.logging import get_logger
from intent_router.core.schemas import ConversationContext, RouteResult
from intent_router.extractors.base import ExtractionContext
from intent_router.extractors.pipeline import build_pipeline, load_gazetteers
from intent_router.models.embeddings import Embedder
from intent_router.models.loader import load_sentence_model
from intent_router.router.engine import RouterEngine
from intent_router.router.fallback import build_fallback
from intent_router.router.normalize import MASK_PLACEHOLDERS, TypoCorrector, build_domain_vocab, mask_entities
from intent_router.router.registry import IntentRegistry
from intent_router.router.similarity import IntentIndex
from intent_router.router.verifier import build_verifier

log = get_logger(__name__)


class MessageTooLongError(ValueError):
    pass


class IntentService:
    def __init__(self, settings: Settings | None = None, model_name: str | None = None):
        self.settings = settings or get_settings()
        s = self.settings
        self.model_name = model_name or s.model_name
        self.startup_ms: dict[str, float] = {}

        t0 = time.perf_counter()
        self.registry = IntentRegistry.load(s.intents_file, s.domain_dirs)
        self.startup_ms["registry"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        model = load_sentence_model(s, self.model_name)
        self.embedder = Embedder(model, self.model_name, s.query_prefix)
        self.startup_ms["model_load"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        known_words = self.embedder.tokenizer_words()
        gazetteers = load_gazetteers(s.gazetteer_file, self.registry.gazetteers)
        self.pipeline = build_pipeline(gazetteers, known_words=known_words)
        vocab_texts = [e for i in self.registry.intents for e in (*i.examples, i.description)]
        vocab_texts += [e for a in self.registry.actions.values() for e in a.examples]
        vocab_texts += [a for entries in gazetteers.values() for aliases in entries.values() for a in aliases]
        vocab_texts += list(MASK_PLACEHOLDERS.values())
        corrector = TypoCorrector(build_domain_vocab(vocab_texts), known_words=known_words)
        self.startup_ms["extractors"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        fixed_ctx = ExtractionContext(reference_date=date(2000, 1, 1))
        preprocess = (lambda text: mask_entities(text, self.pipeline.extract(text, fixed_ctx))) if s.delexicalize else None
        self.scoring = s.resolve_scoring(self.model_name)
        self.index = IntentIndex(
            self.registry, self.embedder, head_words=s.head_words, preprocess=preprocess,
            cache_dir=s.embedding_cache_dir if s.enable_embedding_cache else None, **self.scoring,
        )
        self.startup_ms["intent_embeddings"] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        self.verifier = build_verifier(s)  # NoVerifier unless ROUTER_VERIFIER=nli
        if self.verifier.name != "none":
            self.startup_ms["verifier_load"] = (time.perf_counter() - t0) * 1000

        self.thresholds = s.resolve_thresholds(self.model_name)
        self.engine = RouterEngine(
            registry=self.registry, index=self.index, embedder=self.embedder, pipeline=self.pipeline,
            thresholds=self.thresholds, settings=s, corrector=corrector, fallback=build_fallback(s),
            verifier=self.verifier,
        )
        self.startup_ms["total"] = sum(self.startup_ms.values())
        log.info("Router ready: %d intents, %d domains, model=%s, startup %.0f ms",
                 len(self.registry), len(self.registry.domains), self.model_name, self.startup_ms["total"])

    def route(self, message: str, context: ConversationContext | None = None, *,
              include_debug: bool = False, reference_date: date | None = None) -> RouteResult:
        if len(message) > self.settings.max_message_chars:
            raise MessageTooLongError(f"message exceeds {self.settings.max_message_chars} characters")
        t0 = time.perf_counter()
        result = self.engine.route(message, context, reference_date=reference_date, include_debug=include_debug)
        elapsed = (time.perf_counter() - t0) * 1000
        if self.settings.log_messages:
            log.info("route text=%r status=%s intent=%s %.1fms", message, result.status.value, result.intent, elapsed)
        else:
            log.info("route chars=%d status=%s intent=%s conf=%.2f %.1fms", len(message), result.status.value,
                     result.intent, result.confidence, elapsed)
        return result

    def health(self) -> dict:
        return {
            "status": "ok",
            "model": self.model_name,
            "intents": len(self.registry),
            "domains": self.registry.domains,
            "thresholds": self.thresholds.model_dump(),
            "scoring": self.scoring,
            "fallback": self.settings.fallback,
            "verifier": self.verifier.name,
            "startup_ms": {k: round(v, 1) for k, v in self.startup_ms.items()},
        }
