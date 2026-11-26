from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import Settings  # noqa: E402

REFERENCE_DATE = date(2026, 10, 22)  # a Thursday; date expectations in tests assume this "today"


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings()


@pytest.fixture(scope="session")
def service(settings):
    """The real router with the real embedding model, built once per test session."""
    from app.models.loader import ModelNotAvailableError
    from app.services.intent_service import IntentService

    try:
        return IntentService(settings)
    except ModelNotAvailableError as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"embedding model not downloaded: {exc}")


@pytest.fixture(scope="session")
def route(service):
    def _route(text, context=None, **kwargs):
        kwargs.setdefault("reference_date", REFERENCE_DATE)
        return service.route(text, context, **kwargs)

    return _route


@pytest.fixture(scope="session")
def pipeline(settings):
    """Extraction pipeline only (no model needed)."""
    from app.extractors.pipeline import build_pipeline, load_gazetteers
    from app.router.registry import IntentRegistry

    registry = IntentRegistry.load(settings.intents_file, settings.domain_dirs)
    return build_pipeline(load_gazetteers(settings.gazetteer_file, registry.gazetteers)), registry
