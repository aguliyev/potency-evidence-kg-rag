"""Lazy, sanitized configuration for model-backed stages."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal
from urllib.parse import urlsplit

from pydantic import AnyHttpUrl, BaseModel, ConfigDict

GENERATION_ENV = ("LLM_URI", "LLM_MODEL", "LLM_API_KEY")
EMBEDDING_ENV = ("EMBEDDING_LLM_URI", "EMBEDDING_LLM_MODEL", "EMBEDDING_LLM_API_KEY")
JEV_ENV = ("TYPESAFE_API_KEY",)


class _Settings(BaseModel):
    model_config = ConfigDict(frozen=True)


class GenerationSettings(_Settings):
    base_url: AnyHttpUrl
    model: str
    instructor_model: str
    credential_env: Literal["LLM_API_KEY"]
    sanitized_origin: str
    timeout_seconds: float = 120.0
    max_attempts: int = 3
    max_concurrency: int = 4


class EmbeddingSettings(_Settings):
    base_url: AnyHttpUrl
    model: Literal["text-embedding-3-large"]
    credential_env: Literal["EMBEDDING_LLM_API_KEY"]
    sanitized_origin: str
    dimensions: Literal[3072] = 3072
    encoding_format: Literal["float"] = "float"
    normalization: Literal["provider_unit_length_v1"] = "provider_unit_length_v1"
    timeout_seconds: float = 60.0
    max_attempts: int = 3
    batch_size: int = 64
    max_concurrency: int = 4


class JevSettings(_Settings):
    credential_env: Literal["TYPESAFE_API_KEY"]
    model: str = "jev-latest"
    timeout_seconds: float = 30.0
    max_attempts: int = 3
    max_concurrency: int = 8


def _require(env: Mapping[str, str], names: tuple[str, ...]) -> tuple[str, ...]:
    missing = tuple(name for name in names if not env.get(name))
    if missing:
        raise ValueError(f"missing required environment variables: {', '.join(missing)}")
    return tuple(env[name] for name in names)


def _normalized_endpoint(value: str) -> tuple[AnyHttpUrl, str]:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("endpoint must be an absolute HTTP(S) URL")
    if "@" in parsed.netloc or parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment:
        raise ValueError("endpoint may not contain userinfo, query, or fragment credentials")
    path = parsed.path.rstrip("/")
    if path not in {"", "/v1"}:
        raise ValueError("endpoint path must be empty or /v1")
    origin = f"{parsed.scheme}://{parsed.netloc}/v1"
    return AnyHttpUrl(origin), origin


def load_generation_settings(env: Mapping[str, str]) -> GenerationSettings:
    uri, model, _credential_value = _require(env, GENERATION_ENV)
    base_url, origin = _normalized_endpoint(uri)
    result = GenerationSettings(
        base_url=base_url,
        model=model,
        instructor_model=f"openai/{model}",
        credential_env="LLM_API_KEY",
        sanitized_origin=origin,
    )
    return result


def load_embedding_settings(env: Mapping[str, str]) -> EmbeddingSettings:
    uri, model, _credential_value = _require(env, EMBEDDING_ENV)
    base_url, origin = _normalized_endpoint(uri)
    if origin != "https://api.openai.com/v1":
        raise ValueError("official embedding endpoint must be https://api.openai.com/v1")
    result = EmbeddingSettings(
        base_url=base_url,
        model=model,
        credential_env="EMBEDDING_LLM_API_KEY",
        sanitized_origin=origin,
    )
    return result


def load_jev_settings(env: Mapping[str, str]) -> JevSettings:
    _credential_value, = _require(env, JEV_ENV)
    return JevSettings(credential_env="TYPESAFE_API_KEY")
