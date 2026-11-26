"""Thin, thread-safe wrapper around a SentenceTransformer model."""

from __future__ import annotations

import threading

import numpy as np


class Embedder:
    """Encodes text into L2-normalized float32 vectors (dot product == cosine).

    Inference is serialized with a lock: on a small CPU, concurrent forward passes
    oversubscribe cores and make every request slower. Requests that arrive
    together are still served in order with predictable latency.
    """

    def __init__(self, model, model_name: str, query_prefix: str = ""):
        self.model = model
        self.model_name = model_name
        self.query_prefix = query_prefix
        self._lock = threading.Lock()
        get_dim = getattr(model, "get_embedding_dimension", None) or model.get_sentence_embedding_dimension
        self.dim = int(get_dim())

    def encode(self, texts: list[str], *, is_query: bool = True, batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        if is_query and self.query_prefix:
            texts = [self.query_prefix + t for t in texts]
        with self._lock:
            vectors = self.model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        return np.asarray(vectors, dtype=np.float32)

    def tokenizer_words(self) -> set[str]:
        """Whole-word entries of the model's tokenizer vocabulary (for typo checks)."""
        tokenizer = getattr(self.model, "tokenizer", None)
        if tokenizer is None:
            return set()
        try:
            vocab = tokenizer.get_vocab()
        except Exception:  # noqa: BLE001
            return set()
        return {w for w in vocab if w.isalpha() and not w.startswith("##")}
