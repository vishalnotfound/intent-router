"""Semantic scoring of a request against the intent registry.

Three signals, all from the same embedding model:

1. Intent similarity: mean of the top-k cosine similarities between the request
   and each intent's example phrases (optionally blended with its description).
2. Negative penalty: if the request is closer to an intent's negative ("confusing")
   examples than to its positives, that intent is pushed down.
3. Action stage: similarity to *topic-neutral* action phrases ("compare these",
   "summarize this"). Sentence embeddings are dominated by the topic, so "find
   Python jobs" and "compare Python jobs" look almost identical (cosine ~0.85).
   Because action phrases carry no topic, the topic shifts all action scores
   roughly equally and the verb decides the ranking. Intents whose action is
   behind the best action are penalized by ``action_weight * gap``.

None of these numbers is a probability.
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from app.core.logging import get_logger
from app.models.embeddings import Embedder
from app.router.normalize import request_head
from app.router.registry import IntentRegistry

log = get_logger(__name__)

CACHE_FORMAT_VERSION = "3"


@dataclass
class ScoreBreakdown:
    final: np.ndarray  # (Q, I) combined score used for decisions
    positive: np.ndarray  # (Q, I) top-k mean similarity to examples
    negative: np.ndarray  # (Q, I) max similarity to negative examples (-1 if none)
    actions: np.ndarray  # (Q, A) action-stage scores
    action_names: list[str]


def _topk_mean(block: np.ndarray, k: int) -> np.ndarray:
    k = min(k, block.shape[1])
    if k == 1:
        return block.max(axis=1)
    part = np.partition(block, block.shape[1] - k, axis=1)[:, -k:]
    return part.mean(axis=1)


class IntentIndex:
    def __init__(self, registry: IntentRegistry, embedder: Embedder, *, top_k: int = 2,
                 description_weight: float = 0.0, negative_penalty: float = 1.0,
                 action_weight: float = 1.0, head_words: int = 5, head_weight: float = 0.3,
                 cache_dir: Path | None = None, preprocess: Callable[[str], str] | None = None):
        self.registry = registry
        self.embedder = embedder
        self.top_k = top_k
        self.description_weight = description_weight
        self.negative_penalty = negative_penalty
        self.action_weight = action_weight
        self.head_words = head_words
        self.head_weight = head_weight

        self.action_names = list(registry.actions)
        self._action_of_intent = np.array(
            [self.action_names.index(i.action) for i in registry.intents], dtype=np.int64
        )

        pos_texts, self._pos_slices = self._flatten([i.examples for i in registry.intents])
        neg_texts, self._neg_slices = self._flatten([i.negative_examples for i in registry.intents])
        if preprocess is not None:  # same delexicalization as applied to requests
            pos_texts = [preprocess(t) for t in pos_texts]
            neg_texts = [preprocess(t) for t in neg_texts]
        self.example_texts = pos_texts
        act_texts, self._act_slices = self._flatten(
            [registry.actions[a].examples for a in self.action_names]
        )
        desc_texts = [i.description for i in registry.intents]
        head_texts = [request_head(t, head_words) for t in pos_texts]

        groups = {"pos": pos_texts, "neg": neg_texts, "act": act_texts, "desc": desc_texts, "head": head_texts}
        self._emb = self._load_or_compute(groups, cache_dir)

    @staticmethod
    def _flatten(lists: list[list[str]]) -> tuple[list[str], list[slice]]:
        texts: list[str] = []
        slices: list[slice] = []
        for items in lists:
            start = len(texts)
            texts.extend(items)
            slices.append(slice(start, len(texts)))
        return texts, slices

    def _cache_path(self, groups: dict[str, list[str]], cache_dir: Path) -> Path:
        h = hashlib.sha256()
        h.update(CACHE_FORMAT_VERSION.encode())
        h.update(self.embedder.model_name.encode())
        for name in sorted(groups):
            h.update(name.encode())
            for t in groups[name]:
                h.update(b"\x00" + t.encode("utf-8"))
        slug = re.sub(r"[^A-Za-z0-9]+", "_", self.embedder.model_name).strip("_")
        return Path(cache_dir) / f"{slug}-{h.hexdigest()[:16]}.npz"

    def _load_or_compute(self, groups: dict[str, list[str]], cache_dir: Path | None) -> dict[str, np.ndarray]:
        path = self._cache_path(groups, cache_dir) if cache_dir else None
        if path and path.exists():
            try:
                with np.load(path) as data:
                    loaded = {k: data[k] for k in groups}
                if all(loaded[k].shape[0] == len(groups[k]) for k in groups):
                    log.info("Loaded cached intent embeddings from %s", path.name)
                    return loaded
            except Exception as exc:  # noqa: BLE001 - corrupt cache is recomputed
                log.warning("Ignoring unreadable embedding cache %s: %s", path, exc)

        # Registry phrases are documents, not queries: no query prefix.
        computed = {k: self.embedder.encode(v, is_query=False, batch_size=64) for k, v in groups.items()}
        if path:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".npz")
                os.close(fd)
                np.savez(tmp, **computed)
                os.replace(tmp, path)
            except OSError as exc:
                log.warning("Could not write embedding cache: %s", exc)
        return computed

    @property
    def num_texts(self) -> int:
        return sum(v.shape[0] for v in self._emb.values())

    def head_texts(self, texts: list[str]) -> list[str]:
        return [request_head(t, self.head_words) for t in texts]

    def score(self, queries: np.ndarray, heads: np.ndarray | None = None) -> ScoreBreakdown:
        """Score query vectors (Q, d); `heads` are the matching head-view vectors."""
        n_intents = len(self.registry.intents)
        q = queries.astype(np.float32, copy=False)

        pos_sims = q @ self._emb["pos"].T
        positive = np.empty((q.shape[0], n_intents), dtype=np.float32)
        positive_max = np.empty((q.shape[0], n_intents), dtype=np.float32)
        for i, sl in enumerate(self._pos_slices):
            positive[:, i] = _topk_mean(pos_sims[:, sl], self.top_k)
            positive_max[:, i] = pos_sims[:, sl].max(axis=1)
        if heads is not None and self.head_weight > 0:
            # Blend in the head view: the start of a request carries the verb and object
            # type, while trailing specifics dominate whole-sentence similarity.
            head_sims = heads.astype(np.float32, copy=False) @ self._emb["head"].T
            w = self.head_weight
            for i, sl in enumerate(self._pos_slices):
                positive[:, i] = (1 - w) * positive[:, i] + w * _topk_mean(head_sims[:, sl], self.top_k)

        base = positive
        if self.description_weight > 0:
            desc = q @ self._emb["desc"].T
            base = (1 - self.description_weight) * positive + self.description_weight * desc

        negative = np.full((q.shape[0], n_intents), -1.0, dtype=np.float32)
        if self._emb["neg"].shape[0]:
            neg_sims = q @ self._emb["neg"].T
            for i, sl in enumerate(self._neg_slices):
                if sl.stop > sl.start:
                    negative[:, i] = neg_sims[:, sl].max(axis=1)

        act_sims = q @ self._emb["act"].T
        actions = np.empty((q.shape[0], len(self.action_names)), dtype=np.float32)
        for a, sl in enumerate(self._act_slices):
            actions[:, a] = _topk_mean(act_sims[:, sl], self.top_k)
        action_gap = actions.max(axis=1, keepdims=True) - actions[:, self._action_of_intent]

        # Penalize only when the closest negative example beats the closest positive one.
        final = (
            base
            - self.negative_penalty * np.maximum(0.0, negative - positive_max)
            - self.action_weight * action_gap
        )
        return ScoreBreakdown(final=np.clip(final, -1.0, 1.0), positive=positive, negative=negative,
                              actions=actions, action_names=self.action_names)
