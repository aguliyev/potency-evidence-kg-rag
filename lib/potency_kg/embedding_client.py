"""OpenAI-compatible embedding boundary with strict, secret-safe validation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from openai import OpenAI
from pydantic import SecretStr

from .settings import EmbeddingSettings


@dataclass(frozen=True)
class EmbeddingClientIdentity:
    provider: str
    model: str
    dimensions: int
    encoding_format: Literal["float"]
    normalization: str
    sanitized_origin: str
    credential_env: str


@dataclass(frozen=True)
class EmbeddingBatch:
    model: str
    vectors: tuple[tuple[float, ...], ...]
    dimensions: int
    encoding_format: Literal["float"]
    normalization: str
    prompt_tokens: int
    total_tokens: int


class EmbeddingInputError(ValueError):
    """The caller supplied an invalid embedding request."""


class EmbeddingProtocolError(RuntimeError):
    """The embedding provider returned an invalid response."""


class EmbeddingClient(Protocol):
    identity: EmbeddingClientIdentity

    def embed(self, texts: Sequence[str]) -> EmbeddingBatch: ...


class OpenAIEmbeddingClient:
    """Adapter for the official float, full-dimension embeddings API."""

    def __init__(
        self,
        settings: EmbeddingSettings,
        api_key: SecretStr,
        client: OpenAI | None = None,
    ) -> None:
        self._settings = settings
        try:
            self._client = client if client is not None else OpenAI(
                base_url=str(settings.base_url),
                api_key=api_key.get_secret_value(),
                timeout=settings.timeout_seconds,
            )
        except Exception:
            raise EmbeddingProtocolError("embedding client construction failed") from None
        self.identity = EmbeddingClientIdentity(
            provider="openai-compatible",
            model=settings.model,
            dimensions=settings.dimensions,
            encoding_format=settings.encoding_format,
            normalization=settings.normalization,
            sanitized_origin=settings.sanitized_origin,
            credential_env=settings.credential_env,
        )

    def __repr__(self) -> str:
        return f"OpenAIEmbeddingClient(identity={self.identity!r})"

    def embed(self, texts: Sequence[str]) -> EmbeddingBatch:
        inputs = list(texts)
        if not inputs or any(not isinstance(value, str) or not value.strip() for value in inputs):
            raise EmbeddingInputError("embedding inputs must be non-empty text")
        if len(inputs) > 64 or len(inputs) > self._settings.batch_size:
            raise EmbeddingInputError("embedding batch exceeds configured size 64")

        try:
            response = self._client.embeddings.create(
                model=self._settings.model,
                input=inputs,
                encoding_format=self._settings.encoding_format,
                dimensions=self._settings.dimensions,
            )
        except Exception:
            raise EmbeddingProtocolError("embedding request failed") from None
        try:
            ordered = sorted(response.data, key=lambda item: item.index)
            indexes = [item.index for item in ordered]
            if indexes != list(range(len(inputs))):
                raise EmbeddingProtocolError("embedding response indexes do not match inputs")

            vectors = tuple(tuple(float(value) for value in item.embedding) for item in ordered)
            if any(len(vector) != self._settings.dimensions for vector in vectors):
                raise EmbeddingProtocolError("embedding response dimension is not 3072")
            if any(not math.isfinite(value) for vector in vectors for value in vector):
                raise EmbeddingProtocolError("embedding response contains non-finite values")

            return EmbeddingBatch(
                model=response.model,
                vectors=vectors,
                dimensions=self._settings.dimensions,
                encoding_format=self._settings.encoding_format,
                normalization=self._settings.normalization,
                prompt_tokens=response.usage.prompt_tokens,
                total_tokens=response.usage.total_tokens,
            )
        except EmbeddingProtocolError:
            raise
        except Exception:
            raise EmbeddingProtocolError("embedding response is malformed") from None
