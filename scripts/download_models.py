"""One-time model download. After this, the router runs fully offline.

Usage:
    python scripts/download_models.py                  # default MiniLM model
    python scripts/download_models.py --all            # MiniLM + BGE (for comparison)
    python scripts/download_models.py --model BAAI/bge-small-en-v1.5
    python scripts/download_models.py --fallback       # optional local LLM (~1 GB)
    python scripts/download_models.py --verifier       # optional NLI verifier (~295 MB)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from intent_router.core.config import ALTERNATIVE_MODEL, DEFAULT_MODEL, get_settings  # noqa: E402


def download_embedding_model(name: str, cache_dir: str | None) -> None:
    from sentence_transformers import SentenceTransformer

    print(f"Downloading / verifying {name} ...")
    model = SentenceTransformer(name, device="cpu", cache_folder=cache_dir)
    vec = model.encode(["hello world"], normalize_embeddings=True)
    print(f"  ok: embedding dimension {vec.shape[1]}")


def download_fallback_model(name: str, cache_dir: str | None) -> None:
    from huggingface_hub import snapshot_download

    print(f"Downloading optional fallback LLM {name} ...")
    path = snapshot_download(name, cache_dir=cache_dir)
    print(f"  ok: {path}")


VERIFIER_FILES = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
                  "special_tokens_map.json", "added_tokens.json", "spm.model", "README.md"]


def download_verifier_model(name: str, revision: str | None, cache_dir: str | None) -> None:
    from huggingface_hub import snapshot_download

    # Only the PyTorch weights and tokenizer (~295 MB for the default model), not the ONNX copies.
    print(f"Downloading optional NLI verifier {name} (revision {revision or 'latest'}) ...")
    path = snapshot_download(name, revision=revision, cache_dir=cache_dir, allow_patterns=VERIFIER_FILES)
    print(f"  ok: {path}")


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--model", action="append", help="Embedding model name (repeatable).")
    parser.add_argument("--all", action="store_true", help="Download MiniLM and BGE.")
    parser.add_argument("--fallback", action="store_true", help="Also download the fallback LLM.")
    parser.add_argument("--verifier", action="store_true",
                        help="Also download the NLI verifier for destructive/financial requests (~295 MB).")
    args = parser.parse_args()

    models = args.model or [settings.model_name]
    if args.all:
        models = [DEFAULT_MODEL, ALTERNATIVE_MODEL]
    for name in dict.fromkeys(models):
        download_embedding_model(name, settings.model_cache_dir)
    if args.fallback:
        download_fallback_model(settings.fallback_model, settings.model_cache_dir)
    if args.verifier:
        download_verifier_model(settings.verifier_model, settings.verifier_revision, settings.model_cache_dir)
    print("Done. The router loads models with local_files_only=True by default (no network).")


if __name__ == "__main__":
    main()
