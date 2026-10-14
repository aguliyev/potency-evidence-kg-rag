"""Instructor-backed structured generation with explicit, secret-safe configuration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from threading import local
from time import monotonic
from urllib.parse import urlsplit
from typing import Protocol
import logging

import instructor
from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError
from pydantic import BaseModel, SecretStr, ValidationError

from .settings import GenerationSettings, load_generation_settings


@dataclass(frozen=True)
class GenerationClientIdentity:
    provider: str
    model: str
    sanitized_origin: str
    credential_env: str
    instructor_mode: str
    max_attempts: int


class StructuredExtractionClient(Protocol):
    identity: GenerationClientIdentity

    def extract(self, model_text: str, response_model: type[BaseModel], prompt: str) -> BaseModel: ...


class GenerationInputError(ValueError):
    """The structured generation input is incomplete."""


class GenerationTransientError(RuntimeError):
    """A transient provider failure exhausted bounded retries."""


class GenerationProtocolError(RuntimeError):
    """The provider or Instructor response was unusable."""


class GenerationSchemaError(GenerationProtocolError):
    """A response failed the requested structured schema."""


class GenerationUnselectedTypeError(GenerationProtocolError):
    """A parseable response used an entity type outside the selected bundles."""


_TRANSIENT = (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError, TimeoutError, ConnectionError)


logger = logging.getLogger(__name__)


# Anthropic's OpenAI-compatible endpoint rejects forced tool_choice (Instructor's openai default) on
# adaptive-thinking models, so use markdown-JSON mode there. Other endpoints keep the default.
def _mode_for(settings: GenerationSettings) -> instructor.Mode | None:
    return instructor.Mode.MD_JSON if urlsplit(settings.sanitized_origin).hostname == "api.anthropic.com" else None


def _contains_unselected_type(error: ValidationError, response_model: type[BaseModel]) -> bool:
    allowed = set(getattr(response_model, "__allowed_entity_types__", ()))
    if not allowed:
        return False
    for item in error.errors(include_url=False):
        candidate = item.get("input")
        if isinstance(candidate, dict) and isinstance(candidate.get("type"), str):
            if candidate["type"] not in allowed:
                return True
        context = item.get("ctx")
        if isinstance(context, dict) and isinstance(context.get("tag"), str):
            if context["tag"] not in allowed:
                return True
    return False


class InstructorExtractionClient:
    """Thin Instructor adapter for an explicitly mapped OpenAI-compatible provider."""

    def __init__(self, settings: GenerationSettings, api_key: SecretStr, client: object | None = None) -> None:
        self._settings = settings
        mode = _mode_for(settings)
        try:
            self._client = client if client is not None else instructor.from_provider(
                settings.instructor_model,
                base_url=str(settings.base_url),
                api_key=api_key.get_secret_value(),
                **({"mode": mode} if mode else {}),
            )
        except Exception:
            raise GenerationProtocolError("generation client construction failed") from None
        self.identity = GenerationClientIdentity(
            provider="openai-compatible",
            model=settings.model,
            sanitized_origin=settings.sanitized_origin,
            credential_env=settings.credential_env,
            instructor_mode=mode.value if mode else "openai",
            max_attempts=settings.max_attempts,
        )
        self._request_state = local()

    @classmethod
    def from_environment(cls, env: Mapping[str, str]) -> "InstructorExtractionClient":
        settings = load_generation_settings(env)
        return cls(settings, SecretStr(env[settings.credential_env]))

    def __repr__(self) -> str:
        return f"InstructorExtractionClient(identity={self.identity!r})"

    def usage_for_current_thread(self) -> dict[str, int | float | None]:
        return dict(getattr(self._request_state, "usage", {}))

    def attempts_for_current_thread(self) -> int:
        return int(getattr(self._request_state, "attempts", 0))

    @staticmethod
    def _completion_usage(completion: object) -> dict[str, int | float | None]:
        usage = getattr(completion, "usage", None)
        if usage is None:
            return {}
        if isinstance(usage, BaseModel):
            values = usage.model_dump(mode="json")
        elif isinstance(usage, Mapping):
            values = dict(usage)
        else:
            values = vars(usage) if hasattr(usage, "__dict__") else {}
        return {
            str(name): value
            for name, value in values.items()
            if (isinstance(value, (int, float)) and not isinstance(value, bool)) or value is None
        }

    def extract(self, model_text: str, response_model: type[BaseModel], prompt: str) -> BaseModel:
        if not isinstance(model_text, str) or not model_text.strip() or not isinstance(prompt, str) or not prompt.strip():
            raise GenerationInputError("generation text and prompt must be nonempty")
        kwargs = {
            "response_model": response_model,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": model_text},
            ],
            "max_retries": 0,
        }
        self._request_state.usage = {}
        self._request_state.attempts = 0
        # Log only sizes and outcome types: text, prompts, and error details stay out of logs.
        logger.info("generation request: model=%s origin=%s mode=%s input_chars=%d prompt_chars=%d max_attempts=%d",
                    self._settings.model, self._settings.sanitized_origin, self.identity.instructor_mode,
                    len(model_text), len(prompt), self._settings.max_attempts)
        started = monotonic()
        attempts = 0
        while attempts < self._settings.max_attempts:
            attempts += 1
            self._request_state.attempts = attempts
            attempt_started = monotonic()
            try:
                create_with_completion = getattr(self._client, "create_with_completion", None)
                if create_with_completion is not None:
                    result, completion = create_with_completion(**kwargs)
                    self._request_state.usage = self._completion_usage(completion)
                else:
                    create = getattr(self._client, "create", None)
                    if create is None:
                        create = self._client.chat.completions.create
                        kwargs["model"] = self._settings.model
                    result = create(**kwargs)
                logger.info("generation request succeeded: attempt=%d elapsed=%.1fs usage=%s",
                            attempts, monotonic() - started, self._request_state.usage)
                return result if isinstance(result, response_model) else response_model.model_validate(result)
            except _TRANSIENT as error:
                logger.warning("generation request transient failure: attempt=%d/%d error=%s elapsed=%.1fs",
                               attempts, self._settings.max_attempts, type(error).__name__,
                               monotonic() - attempt_started)
                if attempts >= self._settings.max_attempts:
                    raise GenerationTransientError("generation request failed after transient retries") from None
            except ValidationError as error:
                logger.warning("generation response failed validation: attempt=%d errors=%d",
                               attempts, error.error_count())
                if _contains_unselected_type(error, response_model):
                    raise GenerationUnselectedTypeError(
                        "generation response used a type outside selected bundles"
                    ) from None
                raise GenerationSchemaError("generation response failed schema validation") from None
            except Exception as error:
                logger.warning("generation request failed: attempt=%d error=%s elapsed=%.1fs",
                               attempts, type(error).__name__, monotonic() - attempt_started)
                raise GenerationProtocolError("generation request failed") from None
        raise GenerationTransientError("generation request failed after transient retries")
