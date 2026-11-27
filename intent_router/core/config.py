"""Application settings.

All values can be overridden with environment variables prefixed ``ROUTER_``
(for example ``ROUTER_MODEL_NAME``) or through a ``.env`` file in the project root.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "intent_router" / "data"

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
ALTERNATIVE_MODEL = "BAAI/bge-small-en-v1.5"


class Thresholds(BaseModel):
    """Decision thresholds on the aggregated similarity score.

    These are PROVISIONAL. Similarity distributions differ per model, per intent
    description set and per input distribution, so re-tune them with
    ``scripts/tune_thresholds.py`` whenever any of those change.
    """

    reject: float = Field(0.35, ge=-1.0, le=1.0, description="Below this best score -> unknown.")
    accept: float = Field(0.50, ge=-1.0, le=1.0, description="At/above this (with margin) -> accepted.")
    margin: float = Field(0.04, ge=0.0, le=1.0, description="Min gap between top-2 intents.")
    segment_accept: float = Field(
        0.50, ge=-1.0, le=1.0, description="Min score for each clause of a multi-intent split."
    )


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ROUTER_", env_file=str(PROJECT_ROOT / ".env"), extra="ignore"
    )

    # --- embedding model -------------------------------------------------
    model_name: str = DEFAULT_MODEL
    model_cache_dir: str | None = None  # None -> default Hugging Face cache
    allow_model_download: bool = False  # False -> local_files_only (no network at startup)
    device: str = "cpu"
    torch_threads: int | None = None  # None -> torch default
    query_prefix: str = ""  # optional instruction prefix for asymmetric models

    # --- registry and data files ----------------------------------------
    intents_file: Path = DATA_DIR / "intents.json"
    domain_dirs: list[Path] = [DATA_DIR / "domains"]
    gazetteer_file: Path = DATA_DIR / "gazetteers.json"
    thresholds_file: Path = DATA_DIR / "thresholds.json"

    # --- embedding cache (intent texts only; user messages are never cached)
    enable_embedding_cache: bool = True
    embedding_cache_dir: Path = PROJECT_ROOT / ".cache" / "embeddings"

    # --- scoring --------------------------------------------------------
    top_k: int = Field(2, ge=1, le=10)  # mean of the top-k example similarities per intent
    description_weight: float = Field(0.0, ge=0.0, le=1.0)
    negative_penalty: float = Field(1.0, ge=0.0, le=5.0)
    action_weight: float = Field(1.0, ge=0.0, le=5.0)  # weight of the action-stage penalty
    head_words: int = Field(5, ge=2, le=12)  # length of the "head" view of a request
    head_weight: float = Field(0.3, ge=0.0, le=1.0)  # weight of the head view (0 disables it)
    typo_correction: bool = True  # correct likely misspellings in the routing text only
    delexicalize: bool = True  # replace places/technologies/names with generic words before embedding
    # Optional overrides; when None the per-model values from thresholds_file are used.
    reject_threshold: float | None = None
    accept_threshold: float | None = None
    margin_threshold: float | None = None

    # --- behaviour ------------------------------------------------------
    max_message_chars: int = Field(2000, ge=1, le=100_000)
    max_tasks: int = Field(4, ge=1, le=10)
    max_alternatives: int = Field(3, ge=0, le=10)
    enable_multi_intent: bool = True
    followup_confidence_factor: float = Field(0.8, ge=0.0, le=1.0)
    date_order: Literal["DMY", "MDY"] = "DMY"

    # --- optional generative fallback (disabled by default) -------------
    fallback: Literal["none", "local_llm"] = "none"
    fallback_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    fallback_max_new_tokens: int = Field(200, ge=16, le=1024)

    # --- optional NLI check for destructive/financial requests (off by default)
    verifier: Literal["none", "nli"] = "none"
    verifier_model: str = "cross-encoder/nli-deberta-v3-xsmall"
    verifier_revision: str | None = "a150876415327c80daeff35ca6f68f5ed8cf5c24"  # pinned; None = latest
    verifier_contradiction: float = Field(0.5, ge=0.0, le=1.0)  # block at/above this contradiction...
    verifier_entailment: float = Field(0.5, ge=0.0, le=1.0)  # ...unless some statement is entailed this much

    # --- logging --------------------------------------------------------
    log_level: str = "INFO"
    log_messages: bool = False  # never log raw user text unless explicitly enabled

    def resolve_scoring(self, model_name: str | None = None) -> dict:
        """Scoring weights: per-model values from thresholds_file ("scoring"), unless the
        setting was given explicitly (environment / .env / constructor)."""
        keys = ("top_k", "description_weight", "negative_penalty", "action_weight", "head_weight")
        values = {k: getattr(self, k) for k in keys}
        if self.thresholds_file.exists():
            raw = json.loads(self.thresholds_file.read_text(encoding="utf-8"))
            per_model = raw.get("models", {}).get(model_name or self.model_name, {}).get("scoring", {})
            for k, v in per_model.items():
                if k in values and k not in self.model_fields_set:
                    values[k] = v
        return values

    def resolve_thresholds(self, model_name: str | None = None) -> Thresholds:
        """Per-model thresholds from the JSON file, then explicit env overrides."""
        name = model_name or self.model_name
        values: dict = {}
        if self.thresholds_file.exists():
            raw = json.loads(self.thresholds_file.read_text(encoding="utf-8"))
            values.update(raw.get("default", {}))
            values.update(raw.get("models", {}).get(name, {}))
        values = {k: v for k, v in values.items() if k in Thresholds.model_fields}
        if self.reject_threshold is not None:
            values["reject"] = self.reject_threshold
        if self.accept_threshold is not None:
            values["accept"] = self.accept_threshold
        if self.margin_threshold is not None:
            values["margin"] = self.margin_threshold
        return Thresholds(**values)


@lru_cache
def get_settings() -> Settings:
    return Settings()
