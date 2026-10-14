from __future__ import annotations

from types import SimpleNamespace

import potency_kg.llm_client as llm_client_module
import pytest
from potency_kg.contracts import ScopedPassageIdentity
from potency_kg.extraction import (
    build_response_model,
    selected_entity_types,
    validate_complete_response,
)
from potency_kg.llm_client import (
    GenerationUnselectedTypeError,
    InstructorExtractionClient,
)
from potency_kg.redactions import substitute_redactions
from potency_kg.settings import load_generation_settings
from pydantic import SecretStr, ValidationError


def _settings():
    return load_generation_settings(
        {"LLM_URI": "https://models.example.test", "LLM_MODEL": "generation-test", "LLM_API_KEY": "secret"}
    )


def test_dynamic_response_model_deduplicates_overlaps_and_rejects_unselected_types() -> None:
    """Allowing a type outside selected bundles, or duplicate overlap types, would bypass the Jev gate."""
    selected = ("product_context", "alias_identity")
    assert selected_entity_types(selected) == ("Product", "Application", "Indication", "AliasCandidate")
    model = build_response_model(selected)
    valid = {
        "entities": [
            {"type": "Product", "local_id": "e1", "name": "CASGEVY", "supporting_quote": "CASGEVY"},
            {"type": "AliasCandidate", "local_id": "e2", "candidate": "exa-cel", "target_local_id": "e1", "supporting_quote": "exa-cel"},
        ],
        "assertions": [{"local_id": "a1", "claim": "CASGEVY is also called exa-cel.", "supporting_quote": "CASGEVY is also called exa-cel"}],
        "relationships": [{"local_id": "r1", "relation_type": "applies_to", "source_local_id": "a1", "target_local_id": "e1", "assertion_local_id": "a1"}],
        "source_refs": [],
    }
    assert model.model_validate(valid).entities[0].type == "Product"
    with pytest.raises(ValidationError):
        model.model_validate({**valid, "entities": [{"type": "Assay", "local_id": "bad", "name": "assay", "supporting_quote": "assay"}]})


def test_complete_response_resolves_local_references_and_restores_only_grounded_original_text() -> None:
    """Dangling local IDs or a quote crossing a redaction must reject the complete response before persistence."""
    passage = ScopedPassageIdentity("t", "p", "s", "v", "p1")
    substitution = substitute_redactions(passage, "CASGEVY uses (b)(4); CASGEVY is reviewed.")
    model = build_response_model(("product_context",))
    response = model.model_validate(
        {
            "entities": [{"type": "Product", "local_id": "e1", "name": "CASGEVY", "supporting_quote": "CASGEVY", "occurrence_index": 1}],
            "assertions": [{"local_id": "a1", "claim": "CASGEVY is reviewed.", "supporting_quote": "CASGEVY is reviewed"}],
            "relationships": [{"local_id": "r1", "relation_type": "applies_to", "source_local_id": "a1", "target_local_id": "e1", "assertion_local_id": "a1"}],
            "source_refs": [{"local_id": "sr1", "record_local_id": "a1", "supporting_quote": "CASGEVY is reviewed"}],
        }
    )
    grounded = validate_complete_response(response, substitution, ("product_context",))
    entity = grounded["entities"][0]
    assert entity["original_quote"] == "CASGEVY"
    assert substitution.original_text[entity["original_start"]:entity["original_end"]] == "CASGEVY"
    bad = response.model_copy(deep=True)
    bad.relationships[0].target_local_id = "missing"
    with pytest.raises(ValueError, match="passage-local"):
        validate_complete_response(bad, substitution, ("product_context",))


def test_complete_response_accepts_a_grounded_source_reference_to_a_relationship() -> None:
    """Excluding relationship IDs from the local reference graph would drop edge-level provenance."""
    passage = ScopedPassageIdentity("t", "p", "s", "v", "p1")
    substitution = substitute_redactions(passage, "CASGEVY is reviewed.")
    model = build_response_model(("product_context",))
    response = model.model_validate(
        {
            "entities": [
                {
                    "type": "Product",
                    "local_id": "e1",
                    "name": "CASGEVY",
                    "supporting_quote": "CASGEVY",
                }
            ],
            "assertions": [
                {
                    "local_id": "a1",
                    "claim": "CASGEVY is reviewed.",
                    "supporting_quote": "CASGEVY is reviewed",
                }
            ],
            "relationships": [
                {
                    "local_id": "r1",
                    "relation_type": "applies_to",
                    "source_local_id": "a1",
                    "target_local_id": "e1",
                    "assertion_local_id": "a1",
                }
            ],
            "source_refs": [
                {
                    "local_id": "sr1",
                    "record_local_id": "r1",
                    "supporting_quote": "CASGEVY is reviewed",
                }
            ],
        }
    )

    grounded = validate_complete_response(response, substitution, ("product_context",))

    assert grounded["source_refs"][0]["record_local_id"] == "r1"
    assert grounded["source_refs"][0]["original_quote"] == "CASGEVY is reviewed"


def test_complete_response_rejects_a_blank_assertion_claim() -> None:
    """A whitespace-only claim must fail complete validation rather than fail later in materialization."""
    passage = ScopedPassageIdentity("t", "p", "s", "v", "p1")
    substitution = substitute_redactions(passage, "CASGEVY")
    model = build_response_model(("product_context",))
    response = model.model_validate(
        {
            "entities": [],
            "assertions": [
                {
                    "local_id": "a1",
                    "claim": "   ",
                    "supporting_quote": "CASGEVY",
                }
            ],
            "relationships": [],
            "source_refs": [],
        }
    )

    with pytest.raises(ValueError, match="atomic"):
        validate_complete_response(response, substitution, ("product_context",))


def test_instructor_adapter_uses_explicit_normalized_provider_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Letting Instructor infer custom credentials or a non-/v1 endpoint would misroute the real request."""
    created: list[tuple[str, dict[str, object]]] = []

    class _Instructor:
        def create(self, **kwargs):
            return kwargs["response_model"].model_validate({"entities": [], "assertions": [], "relationships": [], "source_refs": []})

    def build(selector: str, **kwargs):
        created.append((selector, kwargs))
        return _Instructor()

    monkeypatch.setattr(llm_client_module.instructor, "from_provider", build)
    client = InstructorExtractionClient(_settings(), SecretStr("construction-secret"))
    model = build_response_model(("potency_strategy",))
    client.extract("text", model, "prompt")

    assert created == [("openai/generation-test", {"base_url": "https://models.example.test/v1", "api_key": "construction-secret"})]
    assert "construction-secret" not in repr(client)


def test_meta_generation_uses_json_mode_without_forced_tool_choice(monkeypatch: pytest.MonkeyPatch) -> None:
    """The configured Meta endpoint accepts only automatic tool selection."""
    created: list[dict[str, object]] = []

    def build(_selector: str, **kwargs):
        created.append(kwargs)
        return object()

    monkeypatch.setattr(llm_client_module.instructor, "from_provider", build)
    settings = _settings().model_copy(update={
        "base_url": "https://api.meta.ai/v1",
        "sanitized_origin": "https://api.meta.ai/v1",
    })
    client = InstructorExtractionClient(settings, SecretStr("secret"))

    assert created[0]["mode"] is llm_client_module.instructor.Mode.MD_JSON
    assert client.identity.instructor_mode == llm_client_module.instructor.Mode.MD_JSON.value


def test_instructor_adapter_exposes_sanitized_usage_for_the_current_request() -> None:
    """Dropping the raw completion usage would make persisted extraction cost metadata false."""
    class _Instructor:
        def create_with_completion(self, **kwargs):
            parsed = kwargs["response_model"].model_validate(
                {"entities": [], "assertions": [], "relationships": [], "source_refs": []}
            )
            completion = SimpleNamespace(
                usage=SimpleNamespace(
                    prompt_tokens=11,
                    completion_tokens=3,
                    total_tokens=14,
                )
            )
            return parsed, completion

    client = InstructorExtractionClient(_settings(), SecretStr("secret"), client=_Instructor())
    model = build_response_model(("potency_strategy",))

    client.extract("text", model, "prompt")

    assert client.usage_for_current_thread() == {
        "prompt_tokens": 11,
        "completion_tokens": 3,
        "total_tokens": 14,
    }


def test_instructor_adapter_exposes_transient_attempt_count_for_persistence() -> None:
    """Hiding adapter-level transient retries would understate the immutable run's request attempts."""
    class _Instructor:
        def __init__(self) -> None:
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise TimeoutError("temporary")
            return kwargs["response_model"].model_validate(
                {"entities": [], "assertions": [], "relationships": [], "source_refs": []}
            )

    provider = _Instructor()
    client = InstructorExtractionClient(_settings(), SecretStr("secret"), client=provider)
    model = build_response_model(("potency_strategy",))

    client.extract("text", model, "prompt")

    assert provider.calls == 2
    assert client.attempts_for_current_thread() == 2


def test_dynamic_response_model_canonicalizes_bundle_order() -> None:
    """Equivalent selected bundle sets must not produce order-dependent schema or extraction identity."""
    canonical = build_response_model(("product_context", "alias_identity"))
    reversed_input = build_response_model(("alias_identity", "product_context"))

    assert reversed_input.__selected_bundles__ == ("product_context", "alias_identity")
    assert reversed_input.model_json_schema() == canonical.model_json_schema()


def test_alias_candidate_must_reference_a_local_product_or_assay() -> None:
    """Accepting an Application target would let extraction propose an untyped identity merge."""
    passage = ScopedPassageIdentity("t", "p", "s", "v", "p1")
    substitution = substitute_redactions(passage, "BLA 125785 is called exa-cel.")
    model = build_response_model(("product_context", "alias_identity"))
    response = model.model_validate(
        {
            "entities": [
                {
                    "type": "Application",
                    "local_id": "app1",
                    "application_id": "125785",
                    "supporting_quote": "125785",
                },
                {
                    "type": "AliasCandidate",
                    "local_id": "alias1",
                    "candidate": "exa-cel",
                    "target_local_id": "app1",
                    "supporting_quote": "exa-cel",
                },
            ],
            "assertions": [],
            "relationships": [],
            "source_refs": [],
        }
    )

    with pytest.raises(ValueError, match="local Product or Assay"):
        validate_complete_response(response, substitution, ("product_context", "alias_identity"))


@pytest.mark.parametrize("entity_type", ("Product", "Assay"))
def test_alias_candidate_accepts_a_typed_local_product_or_assay_target(entity_type: str) -> None:
    """Rejecting either allowed target type would discard an extraction-origin proposal."""
    passage = ScopedPassageIdentity("t", "p", "s", "v", "p1")
    substitution = substitute_redactions(passage, "CASGEVY is called exa-cel.")
    model = build_response_model(("product_context", "alias_identity", "assay_evidence"))
    target = {"type": entity_type, "local_id": "target", "name": "CASGEVY", "supporting_quote": "CASGEVY"}
    if entity_type == "Assay":
        target["role"] = "release"
    response = model.model_validate({"entities": [target, {"type": "AliasCandidate", "local_id": "alias", "candidate": "exa-cel", "target_local_id": "target", "supporting_quote": "exa-cel"}], "assertions": [], "relationships": [], "source_refs": []})

    grounded = validate_complete_response(response, substitution, ("product_context", "alias_identity", "assay_evidence"))

    assert grounded["entities"][1]["target_local_id"] == "target"


def test_instructor_adapter_preserves_only_the_unselected_type_retry_signal() -> None:
    """Wrapping schema validation as a generic provider error would prevent the one allowed corrective retry."""
    class _Instructor:
        def create(self, **kwargs):
            return {
                "entities": [
                    {
                        "type": "Assay",
                        "local_id": "bad",
                        "name": "assay",
                        "supporting_quote": "assay",
                    }
                ],
                "assertions": [],
                "relationships": [],
                "source_refs": [],
            }

    client = InstructorExtractionClient(_settings(), SecretStr("secret"), client=_Instructor())
    model = build_response_model(("product_context",))

    with pytest.raises(GenerationUnselectedTypeError, match="outside selected bundles"):
        client.extract("assay", model, "prompt")


def test_redaction_mentions_are_derived_from_spans_not_the_model() -> None:
    from potency_kg.contracts import ScopedPassageIdentity
    from potency_kg.extraction import build_response_model, validate_complete_response
    from potency_kg.redactions import substitute_redactions

    sub = substitute_redactions(ScopedPassageIdentity("t", "p", "s", "v", "x"), "Dose (b)(4) and (b)(4) kept.")
    model = build_response_model(("redaction_uncertainty",))
    out = validate_complete_response(model.model_validate({"entities": [{"type": "RedactionMention", "local_id": "m"}]}), sub, ("redaction_uncertainty",))
    got = [(e["model_start"], e["model_end"], e["original_quote"]) for e in out["entities"]]
    assert got == [(s.model_start, s.model_end, "(b)(4)") for s in sub.spans] and len(got) == 2


def test_ungroundable_quote_drops_record_and_dependents_not_the_passage() -> None:
    from potency_kg.contracts import ScopedPassageIdentity
    from potency_kg.extraction import build_response_model, validate_complete_response
    from potency_kg.redactions import substitute_redactions

    sub = substitute_redactions(ScopedPassageIdentity("t", "p", "s", "v", "x"), "CASGEVY treats SCD. It is approved.")
    model = build_response_model(("product_context", "alias_identity"))
    out = validate_complete_response(model.model_validate({
        "entities": [
            {"type": "Product", "local_id": "e1", "name": "CASGEVY", "supporting_quote": "CASGEVY"},
            {"type": "Indication", "local_id": "e2", "name": "x", "supporting_quote": "not in the passage"},
            {"type": "AliasCandidate", "local_id": "e3", "candidate": "c", "target_local_id": "e2", "supporting_quote": "SCD"},
        ],
        "assertions": [{"local_id": "a1", "claim": "It is approved.", "supporting_quote": "It is approved."}],
        "relationships": [
            {"local_id": "r1", "relation_type": "applies_to", "source_local_id": "a1", "target_local_id": "e1", "assertion_local_id": "a1"},
            {"local_id": "r2", "relation_type": "applies_to", "source_local_id": "a1", "target_local_id": "e2", "assertion_local_id": "a1"},
        ],
    }), sub, ("product_context", "alias_identity"))
    assert [e["local_id"] for e in out["entities"]] == ["e1"]
    assert [r["local_id"] for r in out["relationships"]] == ["r1"]
    assert {d["local_id"] for d in out["dropped"]} == {"e2", "e3", "r2"}
