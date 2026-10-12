"""Passage-scoped Jev classification, immutable response reuse, and current selection state."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from uuid import UUID, uuid4

from sqlalchemy import text
from sqlalchemy.engine import Engine

from .aliases import approved_alias_catalog
from .contracts import Scope, ScopedPassageIdentity
from .ids import (
    canonical_json,
    derive_jev_response_version,
    derive_jev_selection_version,
)
from .jev_client import (
    JevClient,
    JevResponse,
    fixed_noul_questions,
    validate_jev_response,
)
from .processing_policy import BUNDLE_ORDER, JevSelectionPolicy, select_bundles
from .project_profiles import ProjectProfile, load_project_profile
from .prompt_resources.jev_bundles import BUNDLE_TO_TYPES, JEV_QUESTIONS
from .redactions import substitute_redactions


@dataclass(frozen=True)
class JevProcessingReport:
    active_passages: int
    classified: int
    reused_responses: int
    selected: int
    structural_only: int
    failed: int
    jev_response_version: str | None
    jev_selection_version: str | None


@dataclass(frozen=True)
class _Prepared:
    passage: ScopedPassageIdentity
    original_text: str
    model_text: str
    input_sha256: str
    response_version: str


_PAGE_ONLY = re.compile(r"\s*(?:page\s*)?\d+(?:\s*(?:of|/)\s*\d+)?\s*", re.IGNORECASE)


def _profile_for_scope(engine: Engine, scope: Scope) -> ProjectProfile:
    root = Path(__file__).resolve().parents[2]
    descriptor = root / "configs" / "casgevy-potency.json"
    with engine.connect() as connection:
        catalog = approved_alias_catalog(connection, scope)
    return load_project_profile(scope, descriptor, catalog)


def _profile_payload(profile: ProjectProfile) -> dict[str, object]:
    return {
        "name": profile.name,
        "aliases": [
            {"alias_id": entry.alias_id, "alias_value": entry.alias_value,
             "normalized_alias": entry.normalized_alias, "entity_type": entry.entity_type,
             "canonical_key": entry.canonical_key, "canonical_name": entry.canonical_name,
             "normalization_version": entry.normalization_version}
            for entry in profile.alias_catalog.entries
        ],
        "alias_catalog_version": profile.alias_catalog.version,
        "applications": dict(profile.applications),
        "descriptor_sha256": profile.descriptor_sha256,
        "profile_version": profile.profile_version,
    }


def _question_payload() -> dict[str, dict[str, str]]:
    return {
        name: {"question": JEV_QUESTIONS[name][0], "criteria": JEV_QUESTIONS[name][1]}
        for name in BUNDLE_ORDER
    }


def _active_rows(engine: Engine, scope: Scope) -> list[dict[str, object]]:
    with engine.connect() as connection:
        return [
            dict(row)
            for row in connection.execute(
                text(
                    """
                    SELECT p.source_id, p.source_version, p.passage_id, p.text
                    FROM source_passage AS p
                    JOIN source_head AS h
                      ON (h.tenant_id,h.project_id,h.source_id,h.source_version) =
                         (p.tenant_id,p.project_id,p.source_id,p.source_version)
                    JOIN source_document AS d
                      ON (d.tenant_id,d.project_id,d.source_id,d.source_version) =
                         (p.tenant_id,p.project_id,p.source_id,p.source_version)
                    WHERE p.tenant_id=:tenant AND p.project_id=:project
                      AND d.expected_passage_count = (
                        SELECT count(*) FROM source_passage AS complete
                        WHERE (complete.tenant_id,complete.project_id,complete.source_id,complete.source_version) =
                              (d.tenant_id,d.project_id,d.source_id,d.source_version)
                      )
                      AND EXISTS (
                        SELECT 1 FROM passage_embedding AS binding
                        JOIN embedding_run AS run
                          ON (run.embedding_run_id,run.tenant_id,run.project_id,run.source_id,run.source_version,run.passage_id) =
                             (binding.embedding_run_id,binding.tenant_id,binding.project_id,binding.source_id,binding.source_version,binding.passage_id)
                        WHERE (binding.tenant_id,binding.project_id,binding.source_id,binding.source_version,binding.passage_id) =
                              (p.tenant_id,p.project_id,p.source_id,p.source_version,p.passage_id)
                          AND run.status='succeeded'
                      )
                    ORDER BY p.source_id,p.source_version,p.passage_id
                    """
                ),
                {"tenant": scope.tenant_id, "project": scope.project_id},
            ).mappings()
        ]


def _structural_reason(value: str) -> str | None:
    if not value.strip():
        return "empty_passage"
    if _PAGE_ONLY.fullmatch(value) is not None:
        return "page_number_only"
    return None


def _prepare(row: Mapping[str, object], scope: Scope, client: JevClient, profile: ProjectProfile) -> _Prepared:
    passage = ScopedPassageIdentity(scope.tenant_id, scope.project_id, str(row["source_id"]), str(row["source_version"]), str(row["passage_id"]))
    original = str(row["text"])
    model_text = substitute_redactions(passage, original).model_text
    question_payload = _question_payload()
    prompt_hash = sha256(canonical_json(question_payload).encode()).hexdigest()
    bundle_hash = sha256(canonical_json(BUNDLE_TO_TYPES).encode()).hexdigest()
    response_version = derive_jev_response_version(
        model=client.identity.model,
        questions=question_payload,
        redacted_input=model_text,
        normalization=client.identity.normalization,
        prompt_hash=prompt_hash,
        bundle_hash=bundle_hash,
        profile=_profile_payload(profile),
    )
    return _Prepared(passage, original, model_text, sha256(model_text.encode()).hexdigest(), response_version)


def _cached_response(
    engine: Engine,
    item: _Prepared,
    profile: ProjectProfile,
) -> tuple[UUID, JevResponse] | None:
    """Return a response only when its complete persisted profile snapshot still matches."""
    payload = _profile_payload(profile)
    with engine.connect() as connection:
        row = connection.execute(
            text(
                """
                SELECT response_id,model,request_id,probabilities,usage,attempts
                FROM jev_response
                WHERE (tenant_id,project_id,source_id,source_version,passage_id) =
                      (:tenant,:project,:source_id,:source_version,:passage_id)
                  AND jev_response_version=:version AND input_sha256=:input_sha256
                  AND profile_name IS NOT DISTINCT FROM :profile_name
                  AND profile_aliases=CAST(:profile_aliases AS jsonb)
                  AND profile_source_sha256 IS NOT DISTINCT FROM :profile_source_sha256
                  AND profile_version=:profile_version
                  AND alias_catalog_version=:catalog_version AND status='succeeded'
                ORDER BY created_at DESC LIMIT 1
                """
            ),
            {
                "tenant": item.passage.tenant_id,
                "project": item.passage.project_id,
                "source_id": item.passage.source_id,
                "source_version": item.passage.source_version,
                "passage_id": item.passage.passage_id,
                "version": item.response_version,
                "input_sha256": item.input_sha256,
                "profile_name": profile.name,
                "profile_aliases": json.dumps(payload["aliases"]),
                "profile_source_sha256": profile.descriptor_sha256,
                "profile_version": profile.profile_version,
                "catalog_version": profile.alias_catalog.version,
            },
        ).mappings().one_or_none()
    if row is None:
        return None
    probabilities = {name: row["probabilities"][name] for name in JEV_QUESTIONS}
    return row["response_id"], validate_jev_response(
        JevResponse(row["model"], probabilities, row["request_id"], row["usage"], row["attempts"]),
        fixed_noul_questions(),
    )


def _mark_structural(engine: Engine, passage: ScopedPassageIdentity, reason: str) -> None:
    with engine.begin() as connection:
        connection.execute(
            text("""UPDATE passage_processing SET state='structural_only',reason=:reason,
                    active_jev_selection_id=NULL,active_extraction_run_id=NULL,
                    validated_extraction_run_id=NULL,updated_at=now()
                    WHERE (tenant_id,project_id,source_id,source_version,passage_id)=(:tenant,:project,:source_id,:source_version,:passage_id)"""),
            {
                **dict(zip(("tenant", "project", "source_id", "source_version", "passage_id"), passage.as_tuple(), strict=True)),
                "reason": reason,
            },
        )


def _persist(
    engine: Engine,
    item: _Prepared,
    client: JevClient,
    profile: ProjectProfile,
    policy: JevSelectionPolicy,
    result: JevResponse | Exception,
    cached_id: UUID | None,
) -> tuple[bool, bool, str | None]:
    identity = item.passage
    params = {
        "tenant": identity.tenant_id,
        "project": identity.project_id,
        "source_id": identity.source_id,
        "source_version": identity.source_version,
        "passage_id": identity.passage_id,
    }
    questions = {name: JEV_QUESTIONS[name][0] for name in JEV_QUESTIONS}
    criteria = {name: JEV_QUESTIONS[name][1] for name in JEV_QUESTIONS}
    with engine.begin() as connection:
        if isinstance(result, Exception):
            connection.execute(
                text(
                    """
                    INSERT INTO jev_response
                      (response_id,tenant_id,project_id,source_id,source_version,passage_id,jev_response_version,
                       model,credential_env,questions,criteria,probabilities,normalization,prompt_hash,bundle_hash,input_sha256,
                       profile_name,profile_aliases,profile_source_sha256,profile_version,alias_catalog_version,usage,attempts,status,error_detail)
                    VALUES (:response_id,:tenant,:project,:source_id,:source_version,:passage_id,:version,
                       :model,:credential,CAST(:questions AS jsonb),CAST(:criteria AS jsonb),'{}',:normalization,:prompt_hash,:bundle_hash,:input_sha256,
                       :profile_name,CAST(:aliases AS jsonb),:profile_sha,:profile_version,:catalog_version,'{}',:attempts,'failed','Jev classification failed')
                    """
                ),
                {
                    **params,
                    "response_id": uuid4(),
                    "version": item.response_version,
                    "model": client.identity.model,
                    "credential": client.identity.credential_env,
                    "questions": json.dumps(questions),
                    "criteria": json.dumps(criteria),
                    "normalization": client.identity.normalization,
                    "prompt_hash": sha256(canonical_json(_question_payload()).encode()).hexdigest(),
                    "bundle_hash": sha256(canonical_json(BUNDLE_TO_TYPES).encode()).hexdigest(),
                    "input_sha256": item.input_sha256,
                    "profile_name": profile.name,
                    "aliases": json.dumps(_profile_payload(profile)["aliases"]),
                    "profile_sha": profile.descriptor_sha256,
                    "profile_version": profile.profile_version,
                    "catalog_version": profile.alias_catalog.version,
                    "attempts": getattr(result, "attempts", 1),
                },
            )
            connection.execute(
                text("""UPDATE passage_processing SET state='failed',reason='jev_failed',
                        active_jev_selection_id=NULL,active_extraction_run_id=NULL,
                        validated_extraction_run_id=NULL,updated_at=now()
                        WHERE (tenant_id,project_id,source_id,source_version,passage_id)=(:tenant,:project,:source_id,:source_version,:passage_id)"""),
                params,
            )
            return False, True, None

        response = validate_jev_response(result, fixed_noul_questions())
        response_id = cached_id
        if response_id is None:
            response_id = uuid4()
            connection.execute(
                text(
                    """
                    INSERT INTO jev_response
                      (response_id,tenant_id,project_id,source_id,source_version,passage_id,jev_response_version,
                       model,request_id,credential_env,questions,criteria,probabilities,normalization,prompt_hash,bundle_hash,input_sha256,
                       profile_name,profile_aliases,profile_source_sha256,profile_version,alias_catalog_version,usage,attempts,status)
                    VALUES (:response_id,:tenant,:project,:source_id,:source_version,:passage_id,:version,
                       :model,:request_id,:credential,CAST(:questions AS jsonb),CAST(:criteria AS jsonb),CAST(:probabilities AS jsonb),:normalization,:prompt_hash,:bundle_hash,:input_sha256,
                       :profile_name,CAST(:aliases AS jsonb),:profile_sha,:profile_version,:catalog_version,CAST(:usage AS jsonb),:attempts,'succeeded')
                    """
                ),
                {
                    **params,
                    "response_id": response_id,
                    "version": item.response_version,
                    "model": response.model,
                    "request_id": response.request_id,
                    "credential": client.identity.credential_env,
                    "questions": json.dumps(questions),
                    "criteria": json.dumps(criteria),
                    "probabilities": json.dumps(dict(response.probabilities)),
                    "normalization": client.identity.normalization,
                    "prompt_hash": sha256(canonical_json(_question_payload()).encode()).hexdigest(),
                    "bundle_hash": sha256(canonical_json(BUNDLE_TO_TYPES).encode()).hexdigest(),
                    "input_sha256": item.input_sha256,
                    "profile_name": profile.name,
                    "aliases": json.dumps(_profile_payload(profile)["aliases"]),
                    "profile_sha": profile.descriptor_sha256,
                    "profile_version": profile.profile_version,
                    "catalog_version": profile.alias_catalog.version,
                    "usage": json.dumps(dict(response.usage or {})),
                    "attempts": response.attempts,
                },
            )
        thresholds = policy.resolve()
        selected = select_bundles(response.probabilities, thresholds)
        selection_version = derive_jev_selection_version(item.response_version, thresholds.values)
        selection_id = connection.execute(
            text(
                """SELECT selection_id FROM jev_selection
                    WHERE (tenant_id,project_id,source_id,source_version,passage_id)=(:tenant,:project,:source_id,:source_version,:passage_id)
                      AND response_id=:response_id AND jev_selection_version=:selection_version
                    ORDER BY created_at DESC LIMIT 1"""
            ),
            {**params, "response_id": response_id, "selection_version": selection_version},
        ).scalar_one_or_none()
        if selection_id is None:
            selection_id = uuid4()
            connection.execute(
                text(
                    """INSERT INTO jev_selection
                         (selection_id,response_id,tenant_id,project_id,source_id,source_version,passage_id,thresholds,selected_bundles,jev_selection_version)
                       VALUES (:selection_id,:response_id,:tenant,:project,:source_id,:source_version,:passage_id,
                               CAST(:thresholds AS jsonb),CAST(:selected AS jsonb),:selection_version)"""
                ),
                {
                    **params,
                    "selection_id": selection_id,
                    "response_id": response_id,
                    "thresholds": json.dumps(dict(thresholds.values)),
                    "selected": json.dumps(selected),
                    "selection_version": selection_version,
                },
            )
        current = connection.execute(
            text(
                """SELECT pp.state,pp.active_jev_selection_id,pp.active_extraction_run_id
                    FROM passage_processing AS pp
                    WHERE (pp.tenant_id,pp.project_id,pp.source_id,pp.source_version,pp.passage_id)=
                          (:tenant,:project,:source_id,:source_version,:passage_id)"""
            ),
            params,
        ).mappings().one()
        if selected:
            keep_extraction = (
                current["active_jev_selection_id"] == selection_id
                and current["state"] == "extracted"
                and current["active_extraction_run_id"] is not None
            )
            connection.execute(
                text(
                    """UPDATE passage_processing
                       SET state=CAST(:state AS processing_state),reason=:reason,active_jev_selection_id=:selection_id,
                           active_extraction_run_id=CASE WHEN :keep THEN active_extraction_run_id ELSE NULL END,
                           validated_extraction_run_id=CASE WHEN :keep THEN validated_extraction_run_id ELSE NULL END,
                           updated_at=now()
                       WHERE (tenant_id,project_id,source_id,source_version,passage_id)=(:tenant,:project,:source_id,:source_version,:passage_id)"""
                ),
                {**params, "state": "extracted" if keep_extraction else "selected", "reason": "extracted_successfully" if keep_extraction else "jev_bundle_selected", "selection_id": selection_id, "keep": keep_extraction},
            )
        else:
            connection.execute(
                text(
                    """UPDATE passage_processing SET state='structural_only',reason='jev_no_bundle_above_threshold',
                           active_jev_selection_id=:selection_id,active_extraction_run_id=NULL,
                           validated_extraction_run_id=NULL,updated_at=now()
                       WHERE (tenant_id,project_id,source_id,source_version,passage_id)=(:tenant,:project,:source_id,:source_version,:passage_id)"""
                ),
                {**params, "selection_id": selection_id},
            )
        return bool(selected), False, selection_version


def classify_active_passages(
    engine: Engine,
    scope: Scope,
    client: JevClient,
    policy: JevSelectionPolicy,
    *,
    profile: ProjectProfile | None = None,
) -> JevProcessingReport:
    """Classify eligible passages with bounded one-passage requests and per-passage commits."""
    rows = _active_rows(engine, scope)
    profile = profile or _profile_for_scope(engine, scope)
    prepared: list[_Prepared] = []
    structural = 0
    for row in rows:
        item = _prepare(row, scope, client, profile)
        structural_reason = _structural_reason(item.original_text)
        if structural_reason is not None:
            _mark_structural(engine, item.passage, structural_reason)
            structural += 1
        else:
            prepared.append(item)

    outcomes: dict[_Prepared, tuple[JevResponse | Exception, UUID | None, bool]] = {}
    to_call: list[_Prepared] = []
    for item in prepared:
        cached = _cached_response(engine, item, profile)
        if cached is None:
            to_call.append(item)
        else:
            outcomes[item] = (cached[1], cached[0], True)

    def classify_one(item: _Prepared) -> JevResponse:
        return validate_jev_response(client.classify(item.model_text, fixed_noul_questions()), fixed_noul_questions())

    max_workers = max(1, int(client.identity.max_concurrency))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(classify_one, item): item for item in to_call}
        for future in as_completed(futures):
            item = futures[future]
            try:
                outcomes[item] = (future.result(), None, False)
            except Exception as error:
                outcomes[item] = (error, None, False)

    classified = reused = selected_count = failed = 0
    response_versions: list[str] = []
    selection_versions: list[str] = []
    for item in prepared:
        result, cached_id, was_reused = outcomes[item]
        selected, did_fail, selection_version = _persist(engine, item, client, profile, policy, result, cached_id)
        response_versions.append(item.response_version)
        if selection_version:
            selection_versions.append(selection_version)
        classified += 0 if did_fail else 1
        reused += int(was_reused)
        selected_count += int(selected)
        failed += int(did_fail)
        structural += int(not selected and not did_fail)
    return JevProcessingReport(
        active_passages=len(rows),
        classified=classified,
        reused_responses=reused,
        selected=selected_count,
        structural_only=structural,
        failed=failed,
        jev_response_version=response_versions[0] if len(set(response_versions)) == 1 else None,
        jev_selection_version=selection_versions[0] if len(set(selection_versions)) == 1 else None,
    )
