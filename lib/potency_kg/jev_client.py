"""Secret-safe TypeSafe/Jev boundary for fixed bundle classification."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from pydantic import SecretStr
from typesafe_sdk import (
    Noul,
    RetryPolicy,
    TypeSafeAPIConnectionError,
    TypeSafeAPIResponseValidationError,
    TypeSafeAPITimeoutError,
    TypeSafeInternalServerError,
    TypeSafeRateLimitError,
    TypeSafeClient,
)

from .processing_policy import BUNDLE_ORDER
from .prompt_resources.jev_bundles import JEV_QUESTIONS
from .settings import JevSettings, load_jev_settings


@dataclass(frozen=True)
class NoulQuestion:
    name: str
    question: str
    criteria: str


@dataclass(frozen=True)
class JevClientIdentity:
    provider: str
    model: str
    credential_env: str
    max_attempts: int
    max_concurrency: int
    normalization: str = "redaction-v1"


@dataclass(frozen=True)
class JevResponse:
    model: str
    probabilities: Mapping[str, float]
    request_id: str | None = None
    usage: Mapping[str, int | float | None] | None = None
    attempts: int = 1


class JevClient(Protocol):
    identity: JevClientIdentity

    def classify(self, text: str, questions: Sequence[NoulQuestion]) -> JevResponse: ...


class JevInputError(ValueError):
    """A Jev classification request violates the fixed adapter contract."""


class JevTransientError(RuntimeError):
    """A declared transient TypeSafe failure exhausted its bounded retries."""

    def __init__(self, message: str, *, attempts: int) -> None:
        super().__init__(message)
        self.attempts = attempts


class JevProtocolError(RuntimeError):
    """The TypeSafe response did not satisfy the complete seven-score contract."""


_TRANSIENT = (
    TypeSafeAPIConnectionError,
    TypeSafeAPITimeoutError,
    TypeSafeRateLimitError,
    TypeSafeInternalServerError,
    TimeoutError,
    ConnectionError,
)


def fixed_noul_questions() -> tuple[NoulQuestion, ...]:
    return tuple(
        NoulQuestion(name=name, question=JEV_QUESTIONS[name][0], criteria=JEV_QUESTIONS[name][1])
        for name in BUNDLE_ORDER
    )


def validate_jev_response(response: JevResponse, questions: Sequence[NoulQuestion]) -> JevResponse:
    names = tuple(question.name for question in questions)
    if names != BUNDLE_ORDER or tuple(response.probabilities) != BUNDLE_ORDER:
        raise JevProtocolError("Jev response probabilities do not match the fixed questions")
    checked: dict[str, float] = {}
    for name in BUNDLE_ORDER:
        value = response.probabilities[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise JevProtocolError("Jev response probabilities must be numeric and in [0, 1]")
        number = float(value)
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise JevProtocolError("Jev response probabilities must be numeric and in [0, 1]")
        checked[name] = number
    if not isinstance(response.model, str) or not response.model.strip():
        raise JevProtocolError("Jev response model is missing")
    return JevResponse(
        model=response.model,
        probabilities=checked,
        request_id=response.request_id,
        usage=dict(response.usage or {}),
        attempts=response.attempts,
    )


class TypeSafeJevClient:
    """One TypeSafe System One request containing exactly seven Noul questions."""

    def __init__(self, settings: JevSettings, api_key: SecretStr, client: object | None = None) -> None:
        self._settings = settings
        try:
            self._client = client if client is not None else TypeSafeClient(
                api_key=api_key.get_secret_value(),
                model=settings.model,
                timeout=settings.timeout_seconds,
                retry=RetryPolicy(max_retries=0),
            )
        except Exception:
            raise JevProtocolError("Jev client construction failed") from None
        self.identity = JevClientIdentity(
            provider="typesafe-jev",
            model=settings.model,
            credential_env=settings.credential_env,
            max_attempts=settings.max_attempts,
            max_concurrency=settings.max_concurrency,
        )

    @classmethod
    def from_environment(cls, env: Mapping[str, str]) -> "TypeSafeJevClient":
        settings = load_jev_settings(env)
        return cls(settings, SecretStr(env[settings.credential_env]))

    def __repr__(self) -> str:
        return f"TypeSafeJevClient(identity={self.identity!r})"

    def classify(self, text: str, questions: Sequence[NoulQuestion]) -> JevResponse:
        question_tuple = tuple(questions)
        if not isinstance(text, str) or not text.strip():
            raise JevInputError("Jev input must be nonempty text")
        if question_tuple != fixed_noul_questions():
            raise JevInputError("Jev requires the seven fixed questions in bundle order")
        provider_questions = {
            question.name: Noul(
                instructions=question.question,
                criteria={"true": question.criteria, "false": "The criterion is not explicitly supported."},
            )
            for question in question_tuple
        }
        attempts = 0
        while attempts < self._settings.max_attempts:
            attempts += 1
            try:
                raw = self._client.system_one(
                    state=text,
                    questions=provider_questions,
                    model=self._settings.model,
                    timeout=self._settings.timeout_seconds,
                )
                answers = raw.answers
                probabilities = {name: answers[name].noul for name in BUNDLE_ORDER}
                usage = {
                    "input_tokens": getattr(raw.usage, "input_tokens", None),
                    "output_tokens": getattr(raw.usage, "output_tokens", None),
                }
                result = JevResponse(
                    model=raw.model,
                    probabilities=probabilities,
                    request_id=getattr(raw, "request_id", None) or getattr(raw, "id", None),
                    usage=usage,
                    attempts=attempts,
                )
                return validate_jev_response(result, question_tuple)
            except _TRANSIENT:
                if attempts >= self._settings.max_attempts:
                    raise JevTransientError(
                        "Jev request failed after transient retries",
                        attempts=attempts,
                    ) from None
            except (JevProtocolError, JevInputError):
                raise
            except TypeSafeAPIResponseValidationError:
                raise JevProtocolError("Jev response is malformed") from None
            except Exception:
                raise JevProtocolError("Jev request failed") from None
        raise JevTransientError(
            "Jev request failed after transient retries",
            attempts=attempts,
        )
