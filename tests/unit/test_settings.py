from __future__ import annotations

import pytest

from potency_kg.settings import (
    load_embedding_settings,
    load_generation_settings,
    load_jev_settings,
)


def test_generation_normalizes_openai_base_and_never_serializes_secret() -> None:
    """Changing the normalized endpoint or leaking the key must break this test."""
    settings = load_generation_settings(
        {"LLM_URI": "https://models.example.test", "LLM_MODEL": "small", "LLM_API_KEY": "secret"}
    )
    assert str(settings.base_url) == "https://models.example.test/v1"
    assert settings.instructor_model == "openai/small"
    assert "secret" not in repr(settings)
    assert "secret" not in settings.model_dump_json()


@pytest.mark.parametrize("uri", ["https://models.example.test/v1", "https://models.example.test"])
def test_generation_accepts_both_openai_compatible_forms(uri: str) -> None:
    """Dropping either supported URI form must break this test."""
    settings = load_generation_settings({"LLM_URI": uri, "LLM_MODEL": "m", "LLM_API_KEY": "x"})
    assert str(settings.base_url).endswith("/v1")


@pytest.mark.parametrize("uri", ["https://user:secret@example.test", "https://@example.test", "https://example.test/v1?key=x", "https://example.test/v1#key"])
def test_generation_rejects_credentials_embedded_in_uri(uri: str) -> None:
    """Accepting an endpoint credential must break this test."""
    with pytest.raises(ValueError):
        load_generation_settings({"LLM_URI": uri, "LLM_MODEL": "m", "LLM_API_KEY": "x"})


@pytest.mark.parametrize(
    ("loader", "environment", "missing"),
    [
        (load_generation_settings, {}, "LLM_URI, LLM_MODEL, LLM_API_KEY"),
        (load_embedding_settings, {}, "EMBEDDING_LLM_URI, EMBEDDING_LLM_MODEL, EMBEDDING_LLM_API_KEY"),
        (load_jev_settings, {}, "TYPESAFE_API_KEY"),
    ],
)
def test_each_loader_names_its_missing_variables(loader: object, environment: dict[str, str], missing: str) -> None:
    """Replacing actionable missing-variable errors with generic failures must break this test."""
    with pytest.raises(ValueError, match=missing):
        loader(environment)  # type: ignore[operator]


def test_embedding_requires_official_settings_and_immutable_provider_defaults() -> None:
    """Allowing another model or dimensions must break this test."""
    environment = {
        "EMBEDDING_LLM_URI": "https://api.openai.com/v1",
        "EMBEDDING_LLM_MODEL": "text-embedding-3-large",
        "EMBEDDING_LLM_API_KEY": "embedding-secret",
    }
    settings = load_embedding_settings(environment)
    assert str(settings.base_url) == "https://api.openai.com/v1"
    assert (settings.model, settings.dimensions, settings.encoding_format) == ("text-embedding-3-large", 3072, "float")
    assert settings.normalization == "provider_unit_length_v1"
    assert "embedding-secret" not in settings.model_dump_json()
    with pytest.raises(ValueError):
        load_embedding_settings({**environment, "EMBEDDING_LLM_MODEL": "other"})


@pytest.mark.parametrize(
    ("loader", "environment", "secret"),
    [
        (load_generation_settings, {"LLM_URI": "https://models.example.test", "LLM_MODEL": "m", "LLM_API_KEY": "generation-secret"}, "generation-secret"),
        (load_embedding_settings, {"EMBEDDING_LLM_URI": "https://api.openai.com/v1", "EMBEDDING_LLM_MODEL": "text-embedding-3-large", "EMBEDDING_LLM_API_KEY": "embedding-secret"}, "embedding-secret"),
        (load_jev_settings, {"TYPESAFE_API_KEY": "jev-secret"}, "jev-secret"),
    ],
)
def test_all_settings_representations_omit_credential_values(loader: object, environment: dict[str, str], secret: str) -> None:
    """Persisting any of the three credential values must break this test."""
    settings = loader(environment)  # type: ignore[operator]
    assert secret not in repr(settings)
    assert secret not in settings.model_dump_json()
