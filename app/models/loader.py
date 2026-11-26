"""Loads the sentence-embedding model exactly once per process."""

from __future__ import annotations

import os
import threading
import time

from app.core.config import Settings
from app.core.logging import get_logger

log = get_logger(__name__)

_lock = threading.Lock()
_models: dict[tuple[str, str], object] = {}


class ModelNotAvailableError(RuntimeError):
    pass


def load_sentence_model(settings: Settings, model_name: str | None = None):
    """Return a cached SentenceTransformer, loading it on first use.

    With ``allow_model_download=False`` (the default) the model is loaded with
    ``local_files_only=True`` so startup never touches the network.
    """
    name = model_name or settings.model_name
    key = (name, settings.device)
    with _lock:
        if key in _models:
            return _models[key]

        if not settings.allow_model_download:
            # Guarantee no network access from the Hugging Face libraries.
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        import torch
        from sentence_transformers import SentenceTransformer

        # A 22M-parameter model gains nothing from many threads; oversubscription made
        # single requests ~2x slower in measurements. Default: up to 4 threads.
        threads = settings.torch_threads or max(1, min(4, (os.cpu_count() or 2) // 2))
        torch.set_num_threads(threads)

        start = time.perf_counter()
        try:
            model = SentenceTransformer(
                name,
                device=settings.device,
                cache_folder=settings.model_cache_dir,
                local_files_only=not settings.allow_model_download,
            )
        except Exception as exc:  # noqa: BLE001 - re-raised with guidance
            raise ModelNotAvailableError(
                f"Could not load embedding model '{name}'. If this is the first run, download it "
                f"with: python scripts/download_models.py --model {name}  "
                f"(or set ROUTER_ALLOW_MODEL_DOWNLOAD=true). Original error: {exc}"
            ) from exc
        model.eval()
        log.info("Loaded embedding model %s on %s in %.2fs", name, settings.device,
                 time.perf_counter() - start)
        _models[key] = model
        return model
