from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import SecretStr

import potency_kg.jev_client as jev_client_module
from potency_kg.jev_client import (
    JevInputError,
    JevProtocolError,
    JevTransientError,
    NoulQuestion,
    TypeSafeJevClient,
    fixed_noul_questions,
)
from potency_kg.processing_policy import BUNDLE_ORDER
from potency_kg.settings import JevSettings


class _RecordingTypeSafe:
    def __init__(self, response) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def system_one(self, state, questions, **kwargs):
        self.calls.append({"state": state, "questions": questions, **kwargs})
        return self.response


def _provider_response(probabilities: dict[str, float]):
    return SimpleNamespace(
        model="jev-test",
        request_id="request-7",
        answers={name: SimpleNamespace(type="noul", noul=value) for name, value in probabilities.items()},
        usage=SimpleNamespace(input_tokens=17, output_tokens=7),
    )


def test_adapter_sends_one_request_with_all_seven_fixed_order_noul_questions() -> None:
    """Splitting a passage into calls or changing question order/type would break the Jev cost contract."""
    sdk = _RecordingTypeSafe(_provider_response({name: index / 10 for index, name in enumerate(BUNDLE_ORDER)}))
    client = TypeSafeJevClient(JevSettings(credential_env="TYPESAFE_API_KEY"), SecretStr("secret"), client=sdk)

    response = client.classify("one passage", fixed_noul_questions())

    assert len(sdk.calls) == 1
    assert sdk.calls[0]["state"] == "one passage"
    assert tuple(sdk.calls[0]["questions"]) == BUNDLE_ORDER
    assert all(question.type == "noul" for question in sdk.calls[0]["questions"].values())
    assert tuple(response.probabilities) == BUNDLE_ORDER
    assert response.request_id == "request-7"
    assert response.usage == {"input_tokens": 17, "output_tokens": 7}
    assert "secret" not in repr(client)


@pytest.mark.parametrize("invalid", [True, -0.01, 1.01, float("nan"), "0.5"])
def test_adapter_rejects_every_non_numeric_or_out_of_range_probability(invalid) -> None:
    """Persisting a malformed Jev score would make deterministic threshold selection impossible."""
    values = {name: 0.5 for name in BUNDLE_ORDER}
    values["product_context"] = invalid
    client = TypeSafeJevClient(
        JevSettings(credential_env="TYPESAFE_API_KEY"),
        SecretStr("secret"),
        client=_RecordingTypeSafe(_provider_response(values)),
    )
    with pytest.raises(JevProtocolError, match="probabilities"):
        client.classify("one passage", fixed_noul_questions())


def test_real_adapter_construction_maps_the_typesafe_key_explicitly(monkeypatch: pytest.MonkeyPatch) -> None:
    """Relying on ambient SDK environment lookup or leaking the key would violate lazy credential ownership."""
    created: list[dict[str, object]] = []
    sdk = _RecordingTypeSafe(_provider_response({name: 0.5 for name in BUNDLE_ORDER}))

    def build(**kwargs):
        created.append(kwargs)
        return sdk

    monkeypatch.setattr(jev_client_module, "TypeSafeClient", build)
    client = TypeSafeJevClient(JevSettings(credential_env="TYPESAFE_API_KEY"), SecretStr("construction-secret"))
    client.classify("text", fixed_noul_questions())

    assert created[0]["api_key"] == "construction-secret"
    assert created[0]["model"] == "jev-latest"
    assert "construction-secret" not in repr(client)


def test_adapter_rejects_modified_question_text_even_when_names_and_order_match() -> None:
    """Changing fixed criteria while retaining names must not silently change the Jev response version contract."""
    sdk = _RecordingTypeSafe(_provider_response({name: 0.5 for name in BUNDLE_ORDER}))
    client = TypeSafeJevClient(
        JevSettings(credential_env="TYPESAFE_API_KEY"),
        SecretStr("secret"),
        client=sdk,
    )
    modified = list(fixed_noul_questions())
    first = modified[0]
    modified[0] = NoulQuestion(first.name, first.question, "different criterion")

    with pytest.raises(JevInputError, match="fixed questions"):
        client.classify("one passage", modified)

    assert sdk.calls == []


def test_adapter_reports_exhausted_transient_attempts_without_provider_details() -> None:
    """The owner needs the bounded attempt count without receiving secret-bearing provider text."""
    class _FailingTypeSafe:
        def __init__(self) -> None:
            self.calls = 0

        def system_one(self, **kwargs):
            self.calls += 1
            raise TimeoutError("secret timeout detail")

    sdk = _FailingTypeSafe()
    client = TypeSafeJevClient(
        JevSettings(credential_env="TYPESAFE_API_KEY", max_attempts=3),
        SecretStr("secret"),
        client=sdk,
    )

    with pytest.raises(JevTransientError) as captured:
        client.classify("one passage", fixed_noul_questions())

    assert sdk.calls == 3
    assert captured.value.attempts == 3
    assert "secret timeout detail" not in str(captured.value)
